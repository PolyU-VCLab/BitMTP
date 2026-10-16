"""Fast Bonsai-MTP generation backed by GraphAtlas.

The three decoders have different fidelity contracts:

  * `fast_confidence_generate` — commits the first draft token and extends
    through consecutive confident drafts, without AR verification.

  * `fast_speculative_generate` — every iteration runs a draft graph plus a
    verifier.  The default width-(D+1) verifier is AR-causal, and gamma=0
    requires exact token matches against its own logits.  It is not guaranteed
    to be bitwise identical to width-1 AR because the two CUDA graphs can take
    different BF16 numerical paths.  `exact_m1_verify=True` instead reuses the
    same width-1 graph as accelerated AR and is the strict sequence oracle.
    The compatibility margin argument accepts zero only.

  * `fast_hybrid_generate` — picks confidence vs verification per iteration
    via the confidence-streak heuristic.  Confidence
    commits are approximate relative to AR; native strict verifier
    fallback inherits the selected verifier's fidelity contract above.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from .model import (
    ModelArgs, Transformer, make_sdlm_cache,
)
from .fast_runtime import GraphAtlas
from .contracts import validate_verification_options, validate_native_verifier
from .runtime_config import compile_helper


_BONSAI_EOS_IDS = frozenset([151643, 151645])  # <|endoftext|>, <|im_end|>


def _first_stop_index(text: str, stop_strings: Optional[Sequence[str]]) -> int:
    """Earliest char index of any stop substring, or len(text) if none."""
    if not stop_strings:
        return len(text)
    best = len(text)
    for s in stop_strings:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1 and idx < best:
            best = idx
    return best


def _decode_hits_stop(tokenizer, token_ids: Sequence[int], stop_strings: Optional[Sequence[str]]) -> bool:
    if not stop_strings or not token_ids:
        return False
    text = tokenizer.decode(list(token_ids), skip_special_tokens=True)
    return _first_stop_index(text, stop_strings) < len(text)


# ===========================================================================
# Pre-allocated GPU buffers shared across a single generate() call
# Reuse token and position buffers to avoid per-iteration allocations and
# repeated scalar transfers to the host.
# ===========================================================================

@dataclass
class _DecodeBufs:
    """GPU buffers preallocated for the lifetime of a single generate call."""
    gen_buf: torch.Tensor       # int64 [max_gen_len + M_max + 1]
    pending_buf: torch.Tensor   # int64 [M_max + 1]
    commit_buf: torch.Tensor    # int64 [M_max + 1]
    pos_arange: torch.Tensor    # int64 [max(M_max+1, prefill_arange_len)]
    eos_tensor: torch.Tensor    # int64 [num_eos] (>=1; placeholder if no EOS)
    eos_detect: torch.Tensor    # bool  [M_max + 1]
    has_eos: bool
    M_max: int


def _alloc_decode_bufs(
    atlas: "GraphAtlas",
    max_gen_len: int,
    eos_ids,
    prompt_len: int = 0,
) -> _DecodeBufs:
    device = atlas.device
    M_max = max(atlas.M_max, 1)
    # pos_arange is reused for both per-iter (length M_max+1) and the prefill
    # position vector (length prompt_len). Size it for the larger of the two.
    arange_len = max(M_max + 1, prompt_len)
    has_eos = bool(eos_ids)
    eos_list = sorted(eos_ids) if eos_ids else [0]
    return _DecodeBufs(
        gen_buf=torch.zeros(max_gen_len + M_max + 1, dtype=torch.long, device=device),
        pending_buf=torch.zeros(M_max + 1, dtype=torch.long, device=device),
        commit_buf=torch.zeros(M_max + 1, dtype=torch.long, device=device),
        pos_arange=torch.arange(arange_len, dtype=torch.long, device=device),
        eos_tensor=torch.tensor(eos_list, dtype=torch.long, device=device),
        eos_detect=torch.zeros(M_max + 1, dtype=torch.bool, device=device),
        has_eos=has_eos,
        M_max=M_max,
    )


# ---------------------------------------------------------------------------
# Compiled argmax/confidence helpers; the explicit Orin profile runs eagerly.
# ---------------------------------------------------------------------------

@compile_helper(mode="max-autotune-no-cudagraphs", dynamic=False)
def _fused_draft_chain(
    logits: torch.Tensor,        # [D, vocab] bf16
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-pass argmax + confidence chain (BitMTP accepted-prefix semantics).

    Standard execution compiles these operations; Orin uses the same expressions
    without compilation.
    Row 0 is the first prediction from the materialized anchor and is
    always committed. Its draft projection can differ from native AR; this
    confidence decision is not strict verification. The threshold gates rows 1..D-1:
    ``iter_accept = 1 + longest_true_prefix(conf[1:] > threshold)``.
    """
    max_logits, draft_idx = logits.max(dim=-1)
    lse = torch.logsumexp(logits.float(), dim=-1)
    conf = torch.exp(max_logits.float() - lse)
    chain_after_first = (conf[1:] > threshold).int().cumprod(0).sum()
    iter_accept = 1 + chain_after_first
    return draft_idx, iter_accept


@compile_helper(mode="max-autotune-no-cudagraphs", dynamic=False)
def _fused_verify_accept(
    draft_tokens: torch.Tensor,
    verify_tokens: torch.Tensor,
    D: int,
) -> torch.Tensor:
    """Accept only the longest prefix of identical token IDs.

    Logits are deliberately not an input: a different token tied for the
    largest logit still rejects if it is not the verifier's selected argmax.
    Width-(D+1) and width-1 execution can differ numerically; this predicate
    does not claim universal equivalence across those numerical paths.
    """
    return (draft_tokens == verify_tokens[:D]).int().cumprod(0).sum()


def _commit_and_eos(
    bufs: _DecodeBufs,
    n_commit: int,
    eos_set,
) -> Tuple[List[int], bool, int]:
    """Bulk-copy commit_buf[:n_commit] to CPU (1 sync) and do Python-set EOS scan.

    Returns (committed_tokens_cpu_list, eos_hit, n_kept).
    The EOS-token itself is dropped from the kept list (matches baseline).
    """
    if n_commit <= 0:
        return [], False, 0
    toks = bufs.commit_buf[:n_commit].tolist()       # 1 D2H bulk transfer
    if not eos_set:
        return toks, False, n_commit
    n_kept = 0
    eos_hit = False
    for t in toks:
        if t in eos_set:
            eos_hit = True
            break
        n_kept += 1
    return toks, eos_hit, n_kept


# ===========================================================================
# AR (single-token) via CUDA Graph -- fair baseline vs MTP modes
# ===========================================================================

@torch.inference_mode()
def fast_ar_generate(
    model: Transformer,
    atlas: GraphAtlas,
    input_ids: torch.Tensor,
    max_gen_len: int = 512,
    eos_ids=_BONSAI_EOS_IDS,
    stop_strings: Optional[Sequence[str]] = None,
    tokenizer=None,
) -> Tuple[torch.Tensor, dict]:
    """AR greedy decode using captured M=1 CUDA Graphs.

    Token and position buffers are allocated once per call. The host reads
    each selected token to update the decode state and check for EOS.
    """
    args = model.args
    device = atlas.device
    prompt_ids = input_ids[0]
    seq_len = prompt_ids.shape[0]

    if 1 not in atlas.m_set:
        raise ValueError("atlas does not include M=1 for AR path")

    bufs = _alloc_decode_bufs(atlas, max_gen_len, eos_ids, prompt_len=seq_len)

    t0 = time.time()
    cache_tmp = make_sdlm_cache(args, device=device)
    pos = bufs.pos_arange[:seq_len]
    logits_pf = model.forward_sdlm_prefill(
        token_values=prompt_ids, cache=cache_tmp, positions=pos,
        block_size=1, fast=False, causal_attn=True,
    )
    # Greedy first token straight into gen_buf (no host roundtrip).
    bufs.gen_buf[0] = logits_pf[-1].argmax()
    cache_len = seq_len
    sigma = atlas.select_segment(cache_len + max_gen_len + atlas.M_max)
    shared = atlas.kv_buffers[sigma]
    shared.zero_()
    for i, (ck, cv) in enumerate(cache_tmp):
        shared[0, i, :, :, :cache_len, :].copy_(ck[0])
        shared[1, i, :, :, :cache_len, :].copy_(cv[0])
    del cache_tmp
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    prefill_time = time.time() - t0

    # First-token EOS check (baseline-style .item() + Python set lookup —
    # cheapest per-iter EOS option since AR already needs a .item() per step).
    next_tok = int(bufs.gen_buf[0].item())
    gen_count = 1
    if bufs.has_eos and next_tok in eos_ids:
        out = bufs.gen_buf[:0].clone()
        return out, {
            "prefill_time": prefill_time, "decode_time": 1e-9,
            "total_time": prefill_time,
            "prefill_tokens": seq_len, "decode_tokens": 0,
            "timed_decode_tokens": 0, "untimed_output_tokens": 0,
            "n_iters": 0, "avg_accept": None, "ttft": prefill_time,
        }

    t1 = time.time()
    while gen_count < max_gen_len:
        # Segment migration if cache_len approaches σ.
        if cache_len + atlas.M_max > sigma:
            new_sigma = atlas.select_segment(cache_len + atlas.M_max + 64)
            if new_sigma != sigma:
                atlas.migrate_segment(sigma, new_sigma, cache_len)
                sigma = new_sigma

        g = atlas.get_graph(1, sigma, "ar")
        g.tokens_buf[0] = next_tok
        g.positions_buf[0] = cache_len

        atlas.prepare_step(g, cache_len, n_pend=0)
        g.cuda_graph.replay()

        # Commit the new KV (slot σ) -> history[cache_len]
        atlas.commit_kv(g, cache_len, n_accept=1)
        cache_len += 1

        # 1 sync per iter on the same width-1 numerical path as baseline AR.
        next_tok = int(g.logits_out[-1].argmax().item())
        if bufs.has_eos and next_tok in eos_ids:
            break  # EOS excluded from output

        bufs.gen_buf[gen_count] = next_tok
        gen_count += 1

        if (
            stop_strings
            and tokenizer is not None
            and _decode_hits_stop(
                tokenizer, bufs.gen_buf[:gen_count].tolist(), stop_strings)
        ):
            break

        if cache_len + 1 > atlas.segment_ladder[-1]:
            break

    torch.cuda.synchronize()
    decode_time = max(time.time() - t1, 1e-9)
    total_time = time.time() - t0
    out = bufs.gen_buf[:gen_count].clone()
    return out, {
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "total_time": total_time,
        "prefill_tokens": seq_len,
        "decode_tokens": int(out.numel()),
        "timed_decode_tokens": max(0, int(out.numel()) - 1),
        "untimed_output_tokens": min(1, int(out.numel())),
        "n_iters": int(out.numel()),
        "avg_accept": 1.0,
        "ttft": prefill_time,
    }


# ===========================================================================
# Shared helpers
# ===========================================================================

def _prefill_into_atlas(
    model: Transformer,
    atlas: GraphAtlas,
    prompt_ids: torch.Tensor,
    sigma: int,
) -> int:
    """Variable-shape prefill, then copy KV into atlas.kv_buffers[sigma]."""
    args = model.args
    device = atlas.device
    seq_len = prompt_ids.shape[0]
    cache = make_sdlm_cache(args, device=device)
    pos = torch.arange(seq_len, device=device)
    _ = model.forward_sdlm_prefill(
        token_values=prompt_ids, cache=cache, positions=pos,
        block_size=1, fast=False, causal_attn=True,
    )
    shared = atlas.kv_buffers[sigma]
    for i, (ck, cv) in enumerate(cache):
        shared[0, i, :, :, :seq_len, :].copy_(ck[0])
        shared[1, i, :, :, :seq_len, :].copy_(cv[0])
    return seq_len


def _first_decode_step_variable(
    model: Transformer,
    cache_tmp,
    prompt_ids: torch.Tensor,
    D: int,
    MASK: int,
    threshold: float,
    eos_ids,
) -> Tuple[List[int], int, bool]:
    """Iter-1 special: variable-shape forward_sdlm_decode of [anchor + MASK*(D-1)].

    Replicates Bonsai's `gpu_sdlm_confidence_generate` first iter (with
    tail=empty for graph-based control; anchor.position = past_len - 1).
    Returns (accepted_tokens, accept_count, eos_hit).
    """
    device = prompt_ids.device
    seq_len = prompt_ids.shape[0]
    anchor = prompt_ids[-1:]
    masks = torch.full((D - 1,), MASK, dtype=prompt_ids.dtype, device=device)
    block_ids = torch.cat([anchor, masks])
    # Anchor reuses last prompt position (seq_len - 1); masks at seq_len..seq_len+D-2.
    block_pos = torch.cat([
        torch.tensor([seq_len - 1], device=device, dtype=torch.long),
        torch.arange(seq_len, seq_len + D - 1, device=device, dtype=torch.long),
    ])
    logits, _new_ks, _new_vs = model.forward_sdlm_decode(
        token_values=block_ids, positions=block_pos, cache=cache_tmp,
        block_size=D, causal_attn=True,
    )
    return _confidence_accept(logits, threshold, eos_ids, D)


def _confidence_accept(
    logits: torch.Tensor,           # [D, vocab]
    threshold: float,
    eos_ids,
    D: int,
) -> Tuple[List[int], int, bool]:
    """Greedy confidence-streak acceptance over a [D, vocab] logit block.

    Matches Bonsai's HF/`gpu_sdlm_confidence_generate` semantics exactly:
      * accept_count = 1 + longest streak `conf[1:] > tau`
      * truncate at the first EOS within accept_count
      * always commit ≥ 1 token (the anchor's prediction)
    """
    probs = F.softmax(logits.float(), dim=-1)
    confidence, tokens = probs.max(dim=-1)  # [D]
    gate_after_first = confidence[1:] > threshold
    if bool(gate_after_first.all()):
        accept_count = D
    else:
        first_false = int(torch.argmax((~gate_after_first).int()).item())
        accept_count = min(1 + first_false, D)

    accepted = []
    eos_hit = False
    for j in range(accept_count):
        tok = int(tokens[j].item())
        if tok in eos_ids:
            eos_hit = True
            break
        accepted.append(tok)
    if eos_hit:
        accept_count = len(accepted)
    return accepted, accept_count, eos_hit


# ===========================================================================
# Confidence-chain (main fast decode)
# ===========================================================================

@torch.inference_mode()
def fast_confidence_generate(
    model: Transformer,
    atlas: GraphAtlas,
    input_ids: torch.Tensor,
    threshold: float = 0.9,
    max_gen_len: int = 512,
    eos_ids=_BONSAI_EOS_IDS,
    use_fp8_draft: bool = False,
    use_bf16_draft: bool = False,
) -> Tuple[torch.Tensor, dict]:
    """SDLM confidence-chain via CUDA Graph atlas.

    Targets token-sequence parity with ``generate.py``'s confidence decoder;
    this is not an AR-equivalence claim.
    """
    args = model.args
    device = atlas.device
    D = args.block_size
    MASK = args.mask_token_id
    prompt_ids = input_ids[0]
    seq_len = prompt_ids.shape[0]

    # --- Prefill (variable shape; iter-1 also uses var-shape decode) ---
    t0 = time.time()
    cache_tmp = make_sdlm_cache(args, device=device)
    pos = torch.arange(seq_len, device=device)
    _ = model.forward_sdlm_prefill(
        token_values=prompt_ids, cache=cache_tmp, positions=pos,
        block_size=1, fast=False, causal_attn=True,
    )

    # --- Iter 1: variable-shape decode of the first diffusion block. ---
    accepted_1, accept_1, eos_1 = _first_decode_step_variable(
        model, cache_tmp, prompt_ids, D, MASK, threshold, eos_ids,
    )
    torch.cuda.synchronize()
    prefill_time = time.time() - t0

    generated: List[int] = list(accepted_1[:max_gen_len])
    accept_1 = len(generated)
    n_iters = 1
    accept_total = accept_1
    untimed_output_tokens = accept_1
    if eos_1 or accept_1 == 0 or len(generated) >= max_gen_len:
        # EOS at iter 1: short-circuit.
        out = torch.tensor(generated, dtype=torch.long, device=device)
        return out, {
            "prefill_time": prefill_time,
            "decode_time": 1e-9,
            "total_time": prefill_time,
            "prefill_tokens": seq_len,
            "decode_tokens": len(generated),
            "timed_decode_tokens": 0,
            "untimed_output_tokens": untimed_output_tokens,
            "n_iters": n_iters,
            "avg_accept": accept_total / max(1, n_iters),
            "ttft": prefill_time,
            "draft_iters": n_iters, "drafted_tokens": D * n_iters,
            "accepted_drafts": len(generated),
            "accept_rate": len(generated) / max(1, D * n_iters),
            "verifier_bonus_tokens": 0,
            "committed_tokens": len(generated),
            "committed_per_iter": len(generated) / max(1, n_iters),
            "conf_path_frac": 1.0, "n_conf_path": n_iters,
            "n_verify_path": 0,
        }

    # --- Pick sigma and copy iter-1 cache (= prompt KV only, NOT the
    #     mask/anchor KV which are discarded). cache_tmp was modified by
    #     forward_sdlm_decode to include the iter-1 trailing KV; crop. ---
    cache_len = seq_len
    sigma = atlas.select_segment(seq_len + max_gen_len + atlas.M_max)
    shared = atlas.kv_buffers[sigma]
    shared.zero_()
    for i, (ck, cv) in enumerate(cache_tmp):
        shared[0, i, :, :, :cache_len, :].copy_(ck[0, :, :cache_len, :])
        shared[1, i, :, :, :cache_len, :].copy_(cv[0, :, :cache_len, :])
    del cache_tmp
    torch.cuda.empty_cache()

    pending: List[int] = list(accepted_1)        # tail to recompute next iter
    eos_found = False
    t1 = time.time()

    while len(generated) < max_gen_len and not eos_found:
        n_iters += 1
        n_pend = len(pending)
        # Pick the smallest captured M that fits n_pend + D.
        M = _select_M_for_pending(atlas, n_pend)
        P = M - D

        # Segment migration if needed (cache_len + M_max might overflow sigma).
        if cache_len + atlas.M_max > sigma:
            new_sigma = atlas.select_segment(cache_len + atlas.M_max + 64)
            if new_sigma != sigma:
                atlas.migrate_segment(sigma, new_sigma, cache_len)
                sigma = new_sigma

        graph = atlas.get_graph(M, sigma, _draft_role(atlas, use_fp8_draft, use_bf16_draft))

        # Fill tokens
        pending_t = torch.tensor(pending, dtype=torch.long, device=device)
        graph.tokens_buf[:n_pend].copy_(pending_t)
        if n_pend < P:
            graph.tokens_buf[n_pend:P].zero_()        # PAD
        graph.tokens_buf[P] = pending_t[-1]            # anchor duplicate
        if P + 1 < M:
            graph.tokens_buf[P + 1:].fill_(MASK)

        # Fill positions
        graph.positions_buf[:n_pend].copy_(
            torch.arange(cache_len, cache_len + n_pend, device=device, dtype=torch.long)
        )
        if n_pend < P:
            graph.positions_buf[n_pend:P].zero_()
        graph.positions_buf[P:].copy_(
            torch.arange(cache_len + n_pend - 1,
                         cache_len + n_pend + D - 1,
                         device=device, dtype=torch.long)
        )

        # Mask
        atlas.prepare_step(graph, cache_len, n_pend=n_pend)

        # Replay
        graph.cuda_graph.replay()

        # Commit pending KV (slots [σ, σ+n_pend) -> history)
        atlas.commit_kv(graph, cache_len, n_pend)
        cache_len += n_pend

        # Sample + accept
        logits = graph.logits_out  # [D, vocab]
        accepted, accept_count, eos_hit = _confidence_accept(
            logits, threshold, eos_ids, D,
        )
        remaining = max(0, max_gen_len - len(generated))
        if accept_count > remaining:
            accepted = accepted[:remaining]
            accept_count = len(accepted)
        accept_total += accept_count

        pending = accepted
        if accept_count == 0:
            break
        generated.extend(accepted)
        if eos_hit:
            eos_found = True

        # Stop if exhausted
        if cache_len + n_pend + D > atlas.segment_ladder[-1]:
            break

    torch.cuda.synchronize()
    decode_time = max(time.time() - t1, 1e-9)
    total_time = time.time() - t0
    out = torch.tensor(generated[:max_gen_len], dtype=torch.long, device=device)
    return out, {
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "total_time": total_time,
        "prefill_tokens": seq_len,
        "decode_tokens": int(out.numel()),
        "timed_decode_tokens": max(
            0, int(out.numel()) - untimed_output_tokens),
        "untimed_output_tokens": untimed_output_tokens,
        "n_iters": n_iters,
        "avg_accept": accept_total / max(1, n_iters),
        "ttft": prefill_time,
        "draft_iters": n_iters, "drafted_tokens": D * n_iters,
        "accepted_drafts": int(out.numel()),
        "accept_rate": int(out.numel()) / max(1, D * n_iters),
        "verifier_bonus_tokens": 0,
        "committed_tokens": int(out.numel()),
        "committed_per_iter": int(out.numel()) / max(1, n_iters),
        "conf_path_frac": 1.0, "n_conf_path": n_iters,
        "n_verify_path": 0,
    }


def _select_M_for_pending(atlas: GraphAtlas, n_pend: int) -> int:
    """Smallest captured M with P = M - D >= n_pend.

    For D=8 and m_set={9,10,12,16,17}: n_pend=1→9, 2→10, 3-4→12, 5-8→16, ≥9→17.
    """
    target = n_pend + atlas.D
    for M in atlas.m_set:
        if M >= target and M - atlas.D >= n_pend:
            return M
    return atlas.m_set[-1]


def _draft_role(atlas: GraphAtlas, use_fp8: bool,
                use_bf16: bool = False) -> str:
    """Pick the acceptance-tolerant draft output-projection role.

    RTX 3090 atlases capture ``draft_bf16`` as the paper's explicit fallback
    for hardware without native FP8.  The ordinary ``draft`` role is the exact
    packed low-bit projection for ternary checkpoints, not a BF16 fallback.
    """
    if use_fp8 and use_bf16:
        raise RuntimeError("FP8 and BF16 draft roles are mutually exclusive")
    role = "draft_fp8" if use_fp8 else ("draft_bf16" if use_bf16 else "draft")
    if role not in atlas.roles:
        raise RuntimeError(f"requested draft role was not captured: {role}")
    return role


def _verify_role(atlas: GraphAtlas, use_fp8: bool = False, use_q1: bool = False,
                 use_w1a16: bool = False) -> str:
    """Select native T2 or native Q1-backbone BF16 verification only."""
    validate_native_verifier(use_fp8, use_q1, use_w1a16)
    if "verify" not in atlas.roles:
        raise RuntimeError("requested native verify role was not captured")
    return "verify"


def _exact_m1_graph_verify(
    atlas: GraphAtlas,
    sigma: int,
    cache_len: int,
    pending: torch.Tensor,
    n_pend: int,
    draft_tokens: torch.Tensor,
) -> Tuple[int, int, int]:
    """Verify a draft with the captured M=1 graph used by accelerated AR.

    Stop at the first mismatch.  This introduces one scalar synchronization
    per accepted proposal, but avoids replaying the remaining M=1 graphs; at
    Bonsai's typical acceptance length this is substantially faster than
    always replaying all D proposals.  Returns
    ``(accept_count, logical_cache_len, correction_or_bonus)``.
    """
    D = int(draft_tokens.numel())
    g = atlas.get_graph(1, sigma, "ar")
    base_len = cache_len - n_pend

    # The draft graph materialized pending at width M>1. Recompute that tail
    # with the exact same M=1 graph and accumulation order as fast AR.
    for j in range(n_pend):
        pos = base_len + j
        g.tokens_buf[0].copy_(pending[j])
        g.positions_buf[0] = pos
        atlas.prepare_step(g, pos, n_pend=0)
        g.cuda_graph.replay()
        atlas.commit_kv(g, pos, 1)
    pred = int(g.logits_out[-1].argmax().item())
    proposed = draft_tokens.tolist()

    # Sequential AR replay over only the accepted prefix.
    accept_count = 0
    for j, token in enumerate(proposed):
        if int(token) != pred:
            break
        accept_count += 1
        pos = cache_len + j
        g.tokens_buf[0] = int(token)
        g.positions_buf[0] = pos
        atlas.prepare_step(g, pos, n_pend=0)
        g.cuda_graph.replay()
        atlas.commit_kv(g, pos, 1)
        pred = int(g.logits_out[-1].argmax().item())

    return accept_count, cache_len + accept_count, pred


# ===========================================================================
# Speculative
# ===========================================================================

@torch.inference_mode()
def fast_speculative_generate(
    model: Transformer,
    atlas: GraphAtlas,
    input_ids: torch.Tensor,
    max_gen_len: int = 512,
    eos_ids=_BONSAI_EOS_IDS,
    use_fp8_draft: bool = False,
    use_fp8_verify: bool = False,
    use_q1_verify: bool = False,
    use_w1a16_verify: bool = False,
    exact_m1_verify: bool = False,
    rabv_margin: float = 0.0,
    use_bf16_draft: bool = False,
) -> Tuple[torch.Tensor, dict]:
    """Self-speculative SDLM via CUDA Graphs.

    With ``rabv_margin=0``, proposals must exactly match the selected
    verifier's argmax.  The default width-(D+1) verifier is AR-causal but can
    differ numerically from width-1 AR; ``exact_m1_verify=True`` selects the
    same width-1 numerical path as accelerated AR. Nonzero margins
    are rejected:
      1. Draft graph predicts D tokens for the diffusion block.
      2. Verify graph (M_verify=D+1, role='verify') gives AR-causal logits.
      3. accept_count = longest matching prefix; commit accept_count + 1 bonus.

    Committed token IDs are copied to the host together for EOS checking.
    """
    validate_verification_options(rabv_margin, exact_m1_verify)
    validate_native_verifier(use_fp8_verify, use_q1_verify, use_w1a16_verify)
    args = model.args
    device = atlas.device
    D = args.block_size
    MASK = args.mask_token_id
    prompt_ids = input_ids[0]
    seq_len = prompt_ids.shape[0]

    bufs = _alloc_decode_bufs(atlas, max_gen_len, eos_ids, prompt_len=seq_len)
    M_verify = D + 1
    draft_role = _draft_role(atlas, use_fp8_draft, use_bf16_draft)
    verify_role = _verify_role(atlas, use_fp8_verify, use_q1_verify, use_w1a16_verify)

    t0 = time.time()
    cache_tmp = make_sdlm_cache(args, device=device)
    pos = bufs.pos_arange[:seq_len]
    logits_pf = model.forward_sdlm_prefill(
        token_values=prompt_ids, cache=cache_tmp, positions=pos,
        block_size=1, fast=False, causal_attn=True,
    )
    bufs.gen_buf[0] = logits_pf[-1].argmax()
    cache_len = seq_len
    sigma = atlas.select_segment(cache_len + max_gen_len + atlas.M_max)
    shared = atlas.kv_buffers[sigma]
    shared.zero_()
    for i, (ck, cv) in enumerate(cache_tmp):
        shared[0, i, :, :, :cache_len, :].copy_(ck[0])
        shared[1, i, :, :, :cache_len, :].copy_(cv[0])
    del cache_tmp
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    prefill_time = time.time() - t0

    # First token EOS check: baseline-style .item() + Python set lookup.
    next_tok = int(bufs.gen_buf[0].item())
    gen_count = 1
    if bufs.has_eos and next_tok in eos_ids:
        out = bufs.gen_buf[:0].clone()
        return out, {
            "prefill_time": prefill_time, "decode_time": 1e-9,
            "total_time": prefill_time,
            "prefill_tokens": seq_len, "decode_tokens": 0,
            "timed_decode_tokens": 0, "untimed_output_tokens": 0,
            "n_iters": 0, "avg_accept": None, "ttft": prefill_time,
            "draft_iters": 0, "drafted_tokens": 0,
            "accepted_drafts": 0, "accept_rate": None,
            "verifier_bonus_tokens": 0, "committed_tokens": 0,
            "committed_per_iter": None,
            "conf_path_frac": None, "n_conf_path": 0, "n_verify_path": 0,
        }

    # Pending starts as [next_tok] for iter 2.
    bufs.pending_buf[0] = next_tok
    n_pend = 1
    n_iters = 1
    accept_total = 1
    eos_found = False
    draft_iters = 0
    drafted_total = 0
    accepted_draft_total = 0
    verifier_bonus_total = 0

    t1 = time.time()
    while gen_count < max_gen_len and not eos_found:
        n_iters += 1
        draft_iters += 1
        drafted_total += D
        M_draft = _select_M_for_pending(atlas, n_pend)
        P_draft = M_draft - D

        if cache_len + atlas.M_max > sigma:
            new_sigma = atlas.select_segment(cache_len + atlas.M_max + 64)
            if new_sigma != sigma:
                atlas.migrate_segment(sigma, new_sigma, cache_len)
                sigma = new_sigma

        # ---- Draft graph ----
        gd = atlas.get_graph(M_draft, sigma, draft_role)
        gd.tokens_buf[:n_pend].copy_(bufs.pending_buf[:n_pend])
        if n_pend < P_draft:
            gd.tokens_buf[n_pend:P_draft].zero_()
        gd.tokens_buf[P_draft] = bufs.pending_buf[n_pend - 1]
        if P_draft + 1 < M_draft:
            gd.tokens_buf[P_draft + 1:].fill_(MASK)
        # positions: [cache_len..cache_len+n_pend), zeros [n_pend..P_draft),
        # then [cache_len+n_pend-1 .. cache_len+n_pend+D-1)
        torch.add(bufs.pos_arange[:n_pend], cache_len, out=gd.positions_buf[:n_pend])
        if n_pend < P_draft:
            gd.positions_buf[n_pend:P_draft].zero_()
        torch.add(bufs.pos_arange[:D], cache_len + n_pend - 1,
                  out=gd.positions_buf[P_draft:P_draft + D])
        atlas.prepare_step(gd, cache_len, n_pend=n_pend)
        gd.cuda_graph.replay()
        draft_tokens = gd.logits_out.argmax(dim=-1)    # [D] int64

        # Commit pending KV
        atlas.commit_kv(gd, cache_len, n_pend)
        cache_len += n_pend

        if exact_m1_verify:
            accept_count, cache_len, bonus = _exact_m1_graph_verify(
                atlas, sigma, cache_len, bufs.pending_buf, n_pend,
                draft_tokens)
            if accept_count > 0:
                bufs.commit_buf[:accept_count].copy_(draft_tokens[:accept_count])
            bufs.commit_buf[accept_count] = bonus
            n_commit = accept_count + 1
        else:
            # ---- Width-(D+1) batched verifier ----
            gv = atlas.get_graph(M_verify, sigma, verify_role)
            gv.tokens_buf[0] = bufs.pending_buf[n_pend - 1]   # anchor
            gv.tokens_buf[1:1 + D].copy_(draft_tokens)
            torch.add(bufs.pos_arange[:M_verify], cache_len - 1,
                      out=gv.positions_buf[:M_verify])
            atlas.prepare_step(gv, cache_len, n_pend=0)
            gv.cuda_graph.replay()
            verify_tokens = gv.logits_out.argmax(dim=-1)   # [D+1]

            # Fused (draft == verify).cumprod.sum (1 sync).
            accept_count_t = _fused_verify_accept(
                draft_tokens, verify_tokens, D)
            accept_count = int(accept_count_t.item())

            # Build commit_buf on GPU: draft[:acc] || verify[acc]
            if accept_count > 0:
                bufs.commit_buf[:accept_count].copy_(draft_tokens[:accept_count])
            bufs.commit_buf[accept_count] = verify_tokens[accept_count]
            n_commit = accept_count + 1

            # Reuse verifier KV for accepted draft tokens.
            if accept_count > 0:
                atlas.commit_kv_from_verify(gv, cache_len, accept_count)
                cache_len += accept_count

        # Bulk D2H once + Python set EOS scan (replaces N .item() calls).
        toks, eos_hit, n_kept = _commit_and_eos(bufs, n_commit, eos_ids)
        if eos_hit:
            eos_found = True
            n_commit = n_kept

        if gen_count + n_commit > max_gen_len:
            n_commit = max(0, max_gen_len - gen_count)

        accepted_emitted = min(accept_count, n_commit)
        bonus_emitted = int(n_commit > accept_count)
        if accepted_emitted + bonus_emitted != n_commit:
            raise RuntimeError("speculative emitted-token accounting mismatch")
        accepted_draft_total += accepted_emitted
        verifier_bonus_total += bonus_emitted

        if n_commit > 0:
            bufs.gen_buf[gen_count:gen_count + n_commit].copy_(bufs.commit_buf[:n_commit])
            gen_count += n_commit
            accept_total += n_commit

        if eos_found or gen_count >= max_gen_len:
            break

        # Pending = bonus only (1 token).
        if n_commit > 0:
            bufs.pending_buf[0] = bufs.commit_buf[n_commit - 1]
            n_pend = 1
        else:
            break

        if cache_len + n_pend + D > atlas.segment_ladder[-1]:
            break

    torch.cuda.synchronize()
    decode_time = max(time.time() - t1, 1e-9)
    total_time = time.time() - t0
    out = bufs.gen_buf[:gen_count].clone()
    if drafted_total != D * draft_iters:
        raise RuntimeError("speculative drafted-token accounting mismatch")
    if n_iters != draft_iters + 1:
        raise RuntimeError("speculative iteration accounting mismatch")
    if int(out.numel()) != 1 + accepted_draft_total + verifier_bonus_total:
        raise RuntimeError("speculative output-origin accounting mismatch")
    if verifier_bonus_total > draft_iters:
        raise RuntimeError("speculative bonus count exceeds verify iterations")
    committed_per_iter = ((int(out.numel()) - 1) / draft_iters
                          if draft_iters > 0 else 0.0)
    return out, {
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "total_time": total_time,
        "prefill_tokens": seq_len,
        "decode_tokens": int(out.numel()),
        "timed_decode_tokens": max(0, int(out.numel()) - 1),
        "untimed_output_tokens": min(1, int(out.numel())),
        "n_iters": n_iters,
        "avg_accept": committed_per_iter,
        "ttft": prefill_time,
        "draft_iters": draft_iters, "drafted_tokens": drafted_total,
        "accepted_drafts": accepted_draft_total,
        "accept_rate": (accepted_draft_total / drafted_total
                        if drafted_total > 0 else 0.0),
        "verifier_bonus_tokens": verifier_bonus_total,
        "committed_tokens": int(out.numel()) - 1,
        "committed_per_iter": committed_per_iter,
        "conf_path_frac": 0.0, "n_conf_path": 0,
        "n_verify_path": draft_iters,
    }


# ===========================================================================
# Hybrid: confidence streak + speculative fallback
# ===========================================================================

@torch.inference_mode()
def fast_hybrid_generate(
    model: Transformer,
    atlas: GraphAtlas,
    input_ids: torch.Tensor,
    threshold: float = 0.9,
    hybrid_m: Optional[int] = None,    # confidence-streak gate; None => default
    max_gen_len: int = 512,
    eos_ids=_BONSAI_EOS_IDS,
    use_fp8_draft: bool = False,
    use_bf16_draft: bool = False,
    use_fp8_verify: bool = False,
    use_q1_verify: bool = False,
    use_w1a16_verify: bool = False,
    exact_m1_verify: bool = False,
    rabv_margin: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """Hybrid decode: confidence-streak switches between draft-only and verify.

    The confidence branch is approximate relative to AR.  The verifier branch
    follows ``fast_speculative_generate``'s fidelity contract: width-(D+1)
    gamma=0 is exact against that verifier but not guaranteed bitwise-identical
    to width-1 AR, and ``exact_m1_verify=True`` uses the width-1 oracle
    path. Nonzero margins are rejected.

    Token and position buffers are allocated once per call. The host reads
    the confidence streak to select the branch, then checks committed tokens
    for EOS.

    Defaults: threshold=0.9, gamma=0, m=6 for D=8, m=3 for D=4.
    Confidence commits are approximate; this whole hybrid decoder is not lossless AR.
    """
    validate_verification_options(rabv_margin, exact_m1_verify)
    validate_native_verifier(use_fp8_verify, use_q1_verify, use_w1a16_verify)
    args = model.args
    device = atlas.device
    D = args.block_size
    MASK = args.mask_token_id
    if hybrid_m is None:
        hybrid_m = 6 if D == 8 else (3 if D == 4 else max(1, (3 * D) // 4))

    prompt_ids = input_ids[0]
    seq_len = prompt_ids.shape[0]

    bufs = _alloc_decode_bufs(atlas, max_gen_len, eos_ids, prompt_len=seq_len)
    M_verify = D + 1
    draft_role = _draft_role(atlas, use_fp8_draft, use_bf16_draft)
    verify_role = _verify_role(atlas, use_fp8_verify, use_q1_verify, use_w1a16_verify)

    t0 = time.time()
    cache_tmp = make_sdlm_cache(args, device=device)
    pos = bufs.pos_arange[:seq_len]
    logits_pf = model.forward_sdlm_prefill(
        token_values=prompt_ids, cache=cache_tmp, positions=pos,
        block_size=1, fast=False, causal_attn=True,
    )
    bufs.gen_buf[0] = logits_pf[-1].argmax()
    cache_len = seq_len
    sigma = atlas.select_segment(cache_len + max_gen_len + atlas.M_max)
    shared = atlas.kv_buffers[sigma]
    shared.zero_()
    for i, (ck, cv) in enumerate(cache_tmp):
        shared[0, i, :, :, :cache_len, :].copy_(ck[0])
        shared[1, i, :, :, :cache_len, :].copy_(cv[0])
    del cache_tmp
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    prefill_time = time.time() - t0

    # First token EOS check: baseline-style .item() + Python set lookup.
    next_tok = int(bufs.gen_buf[0].item())
    gen_count = 1
    if bufs.has_eos and next_tok in eos_ids:
        out = bufs.gen_buf[:0].clone()
        return out, {
            "prefill_time": prefill_time, "decode_time": 1e-9,
            "total_time": prefill_time,
            "prefill_tokens": seq_len, "decode_tokens": 0,
            "timed_decode_tokens": 0, "untimed_output_tokens": 0,
            "n_iters": 0, "avg_accept": None, "ttft": prefill_time,
            "draft_iters": 0, "drafted_tokens": 0,
            "accepted_drafts": 0, "accept_rate": None,
            "verifier_bonus_tokens": 0, "committed_tokens": 0,
            "committed_per_iter": None,
            "conf_path_frac": None, "n_conf_path": 0, "n_verify_path": 0,
        }

    bufs.pending_buf[0] = next_tok
    n_pend = 1
    n_iters = 1
    accept_total = 1
    eos_found = False
    n_conf_path = 0      # diagnostics: confidence-path iter count
    n_verify_path = 0
    draft_iters = 0
    drafted_total = 0
    accepted_draft_total = 0
    verifier_bonus_total = 0

    t1 = time.time()
    while gen_count < max_gen_len and not eos_found:
        n_iters += 1
        draft_iters += 1
        drafted_total += D
        M_draft = _select_M_for_pending(atlas, n_pend)
        P_draft = M_draft - D

        if cache_len + atlas.M_max > sigma:
            new_sigma = atlas.select_segment(cache_len + atlas.M_max + 64)
            if new_sigma != sigma:
                atlas.migrate_segment(sigma, new_sigma, cache_len)
                sigma = new_sigma

        # ---- Draft graph ----
        gd = atlas.get_graph(M_draft, sigma, draft_role)
        gd.tokens_buf[:n_pend].copy_(bufs.pending_buf[:n_pend])
        if n_pend < P_draft:
            gd.tokens_buf[n_pend:P_draft].zero_()
        gd.tokens_buf[P_draft] = bufs.pending_buf[n_pend - 1]
        if P_draft + 1 < M_draft:
            gd.tokens_buf[P_draft + 1:].fill_(MASK)
        torch.add(bufs.pos_arange[:n_pend], cache_len, out=gd.positions_buf[:n_pend])
        if n_pend < P_draft:
            gd.positions_buf[n_pend:P_draft].zero_()
        torch.add(bufs.pos_arange[:D], cache_len + n_pend - 1,
                  out=gd.positions_buf[P_draft:P_draft + D])
        atlas.prepare_step(gd, cache_len, n_pend=n_pend)
        gd.cuda_graph.replay()

        atlas.commit_kv(gd, cache_len, n_pend)
        cache_len += n_pend

        # Confidence-chain commit remains approximate; verifier acceptance
        # itself always requires identical token IDs.
        draft_tokens, iter_accept_t = _fused_draft_chain(
            gd.logits_out, threshold,
        )

        # For the native batched verifier, enqueue its input/mask
        # before the confidence sync.  Exact M=1 verification instead replays
        # the AR graph only if the confidence gate actually falls back.
        gv = None
        if not exact_m1_verify:
            gv = atlas.get_graph(M_verify, sigma, verify_role)
            gv.tokens_buf[0] = bufs.pending_buf[n_pend - 1]
            gv.tokens_buf[1:1 + D].copy_(draft_tokens)
            torch.add(bufs.pos_arange[:M_verify], cache_len - 1,
                      out=gv.positions_buf[:M_verify])
            atlas.prepare_step(gv, cache_len, n_pend=0)

        # Read iter_accept on the host after verifier-input preparation has
        # been enqueued; this value selects verification or confidence commit.
        # BitMTP semantics: per-iter chain >= hybrid_m => skip verify.
        iter_accept = int(iter_accept_t.item())

        use_verify = iter_accept < hybrid_m

        if not use_verify:
            # Pure confidence commit (chain in this iter alone >= hybrid_m).
            bufs.commit_buf[:iter_accept].copy_(draft_tokens[:iter_accept])
            n_commit = iter_accept
            accepted_this_iter = iter_accept
            bonus_this_iter = 0
            used_verify = False
            n_conf_path += 1
        else:
            if exact_m1_verify:
                accept_count, cache_len, bonus = _exact_m1_graph_verify(
                    atlas, sigma, cache_len, bufs.pending_buf, n_pend,
                    draft_tokens)
                if accept_count > 0:
                    bufs.commit_buf[:accept_count].copy_(draft_tokens[:accept_count])
                bufs.commit_buf[accept_count] = bonus
                n_commit = accept_count + 1
            else:
                # Width-(D+1) verifier replay (prep already queued).
                gv.cuda_graph.replay()
                verify_tokens = gv.logits_out.argmax(dim=-1)     # [D+1]

                accept_count_t = _fused_verify_accept(
                    draft_tokens, verify_tokens, D)
                accept_count = int(accept_count_t.item())        # SYNC 2
                if accept_count > 0:
                    bufs.commit_buf[:accept_count].copy_(draft_tokens[:accept_count])
                bufs.commit_buf[accept_count] = verify_tokens[accept_count]
                n_commit = accept_count + 1
                # Reuse verify draft KV.
                if accept_count > 0:
                    atlas.commit_kv_from_verify(gv, cache_len, accept_count)
                    cache_len += accept_count
            accepted_this_iter = accept_count
            bonus_this_iter = 1
            used_verify = True
            n_verify_path += 1

        # Bulk D2H once + Python set EOS scan.
        toks, eos_hit, n_kept = _commit_and_eos(bufs, n_commit, eos_ids)
        if eos_hit:
            eos_found = True
            n_commit = n_kept

        if gen_count + n_commit > max_gen_len:
            n_commit = max(0, max_gen_len - gen_count)

        # Attribute only tokens that are actually emitted after EOS and
        # max-length clipping.  Draft acceptances come first; a verifier
        # correction/bonus, when present, is the final candidate.
        accepted_emitted = min(accepted_this_iter, n_commit)
        bonus_emitted = int(
            bool(bonus_this_iter) and n_commit > accepted_this_iter)
        if accepted_emitted + bonus_emitted != n_commit:
            raise RuntimeError(
                "hybrid emitted-token accounting mismatch: "
                f"accepted={accepted_emitted} bonus={bonus_emitted} "
                f"committed={n_commit}")
        accepted_draft_total += accepted_emitted
        verifier_bonus_total += bonus_emitted

        if n_commit > 0:
            bufs.gen_buf[gen_count:gen_count + n_commit].copy_(bufs.commit_buf[:n_commit])
            gen_count += n_commit
            accept_total += n_commit
        else:
            eos_found = True

        if eos_found or gen_count >= max_gen_len:
            break

        # Pending semantics differ by branch.
        # - Confidence path (no KV reuse): pending = full committed block,
        #   draft KVs need to be re-computed next iter (n_pend = n_commit).
        # - Verify path (KV reuse applied): bonus only (n_pend = 1).
        if n_commit > 0:
            if used_verify:
                bufs.pending_buf[0] = bufs.commit_buf[n_commit - 1]
                n_pend = 1
            else:
                bufs.pending_buf[:n_commit].copy_(bufs.commit_buf[:n_commit])
                n_pend = n_commit
        else:
            break

        if cache_len + n_pend + D > atlas.segment_ladder[-1]:
            break

    torch.cuda.synchronize()
    decode_time = max(time.time() - t1, 1e-9)
    total_time = time.time() - t0
    out = bufs.gen_buf[:gen_count].clone()
    if drafted_total != D * draft_iters:
        raise RuntimeError("hybrid drafted-token accounting mismatch")
    if draft_iters != n_conf_path + n_verify_path:
        raise RuntimeError("hybrid path/iteration accounting mismatch")
    if n_iters != draft_iters + 1:
        raise RuntimeError("hybrid n_iters accounting mismatch")
    if int(out.numel()) != 1 + accepted_draft_total + verifier_bonus_total:
        raise RuntimeError("hybrid output-origin accounting mismatch")
    if verifier_bonus_total > n_verify_path:
        raise RuntimeError("hybrid verifier bonus count exceeds verify paths")
    if accepted_draft_total > drafted_total:
        raise RuntimeError("hybrid accepted draft count exceeds drafted count")
    committed_per_iter = ((int(out.numel()) - 1) / draft_iters
                          if draft_iters > 0 else 0.0)
    return out, {
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "total_time": total_time,
        "prefill_tokens": seq_len,
        "decode_tokens": int(out.numel()),
        "timed_decode_tokens": max(0, int(out.numel()) - 1),
        "untimed_output_tokens": min(1, int(out.numel())),
        "n_iters": n_iters,
        "avg_accept": committed_per_iter,
        "draft_iters": draft_iters,
        "drafted_tokens": drafted_total,
        "accepted_drafts": accepted_draft_total,
        "accept_rate": (accepted_draft_total / drafted_total
                        if drafted_total > 0 else 0.0),
        "verifier_bonus_tokens": verifier_bonus_total,
        "committed_tokens": int(out.numel()) - 1,
        "committed_per_iter": committed_per_iter,
        "ttft": prefill_time,
        "conf_path_frac": n_conf_path / max(1, n_conf_path + n_verify_path),
        "n_conf_path": n_conf_path,
        "n_verify_path": n_verify_path,
    }
