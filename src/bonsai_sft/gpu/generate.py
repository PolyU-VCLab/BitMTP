"""Bonsai GPU model loading and uncaptured AR/confidence generation.

The packed checkpoint configuration selects Q1-g128 or T2-g128 kernels.
`build_model` is also used by the captured runtime in `fast_generate`.
Use `run_generation` for the released AR, confidence, speculative, and hybrid CLI.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from bonsai_sft.gpu.model import (
    ModelArgs, Transformer, make_sdlm_cache,
)


_BONSAI_EOS_IDS = frozenset([151643, 151645])  # <|endoftext|>, <|im_end|>


# ============================================================================
# Helpers
# ============================================================================

def load_args_from_ckpt(ckpt_dir: str, use_kernel: bool) -> ModelArgs:
    with open(os.path.join(ckpt_dir, "bonsai_gpu_config.json"), "r") as f:
        cfg = json.load(f)
    return ModelArgs(
        dim=cfg["dim"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], n_kv_heads=cfg["n_kv_heads"],
        head_dim=cfg["head_dim"], ffn_dim=cfg["ffn_dim"],
        vocab_size=cfg["vocab_size"], norm_eps=cfg["norm_eps"],
        rope_theta=cfg["rope_theta"], rope_original_max=cfg["rope_original_max"],
        rope_factor=cfg["rope_factor"],
        tie_word_embeddings=cfg["tie_word_embeddings"],
        block_size=cfg["block_size"], causal_attn=cfg["causal_attn"],
        mask_token_id=cfg["mask_token_id"],
        use_kernel=use_kernel,
        kernel_kind=cfg.get("kernel_kind", "q1"),
    )


def build_model(ckpt_dir: str, use_kernel: bool, device) -> Transformer:
    args = load_args_from_ckpt(ckpt_dir, use_kernel=use_kernel)
    m = Transformer(args).to(device=device, dtype=torch.bfloat16)
    if use_kernel and args.kernel_kind == "t2g128":
        state_file = "model_state_t2g128.pt"
    else:
        state_file = "model_state_q1g128.pt" if use_kernel else "model_state_fp16.pt"
    sd = torch.load(os.path.join(ckpt_dir, state_file), map_location=device)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if "lm_head" not in k]
    if real_missing:
        raise RuntimeError(f"Missing weights: {real_missing[:5]}")
    if unexpected:
        raise RuntimeError(f"Unexpected weights: {unexpected[:5]}")
    m.tie_weights()
    if args.kernel_kind == "t2g128":
        # Prefill uses the BF16 tied matrix, but training/deployment quantizes
        # its activation per token.  Decode graphs use the exact packed T2 head.
        m._force_a8_lm_head = True
    m.eval()
    return m


def _crop_cache(cache, length: int):
    """Truncate each layer's KV cache along seq dimension to `length`."""
    return [(k[:, :, :length, :].contiguous(), v[:, :, :length, :].contiguous()) for k, v in cache]


# ============================================================================
# AR baseline
# ============================================================================

@torch.no_grad()
def gpu_ar_generate(
    model: Transformer,
    input_ids: torch.Tensor,
    max_gen_len: int = 512,
    eos_ids=_BONSAI_EOS_IDS,
    temperature: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """Greedy AR generate using the Q1 kernel (M=1 path)."""
    device = input_ids.device
    seq_len = input_ids.shape[-1]
    ids = input_ids[0].clone()  # [seq_len]
    args = model.args

    # 1. Prefill (BF16 fast path; length-0 cache to start).
    t0 = time.time()
    cache = make_sdlm_cache(args, device=device)
    positions = torch.arange(seq_len, device=device)
    logits = model.forward_sdlm_prefill(
        token_values=ids, cache=cache, positions=positions,
        block_size=1, fast=False, causal_attn=True,
    )
    torch.cuda.synchronize()
    prefill_time = time.time() - t0

    # 2. AR decode (M=1).
    t1 = time.time()
    gen_tokens: List[int] = []
    next_logit = logits[-1]
    if temperature > 0:
        probs = F.softmax(next_logit / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
    else:
        nxt = int(next_logit.argmax().item())
    gen_tokens.append(nxt)

    for _ in range(max_gen_len - 1):
        if nxt in eos_ids:
            break
        cur = torch.tensor([nxt], device=device, dtype=ids.dtype)
        cur_pos = torch.tensor([seq_len + len(gen_tokens) - 1], device=device)
        logits_step, new_ks, new_vs = model.forward_sdlm_decode(
            token_values=cur, positions=cur_pos, cache=cache,
            block_size=1, causal_attn=True,
        )
        cache = list(zip(new_ks, new_vs))
        nl = logits_step[-1]
        if temperature > 0:
            probs = F.softmax(nl / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
        else:
            nxt = int(nl.argmax().item())
        gen_tokens.append(nxt)

    torch.cuda.synchronize()
    decode_time = time.time() - t1
    # Strip trailing EOS for accounting parity with HF.
    out = gen_tokens[:-1] if gen_tokens and gen_tokens[-1] in eos_ids else gen_tokens
    return torch.tensor(out, dtype=torch.long, device=device), {
        "prefill_time": prefill_time,
        "decode_time": max(decode_time, 1e-9),
        "total_time": prefill_time + decode_time,
        "prefill_tokens": seq_len,
        "decode_tokens": len(out),
        "n_iters": len(gen_tokens),
        "avg_accept": 1.0,
        "ttft": prefill_time,
    }


# ============================================================================
# SDLM confidence-chain decode
# ============================================================================

@torch.no_grad()
def gpu_sdlm_confidence_generate(
    model: Transformer,
    input_ids: torch.Tensor,
    max_gen_len: int = 512,
    threshold: float = 0.9,
    eos_ids=_BONSAI_EOS_IDS,
    temperature: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """SDLM confidence-chain generate.

    Algorithm:
        anchor = last committed token
        x = [anchor, MASK, MASK, ..., MASK]  (length D)
        pos: anchor reuses prev position (-1) so it overlaps with the
             committed-anchor key in the cache; the D-1 mask tokens take
             positions cache_len .. cache_len+D-2.
        forward -> logits[D, vocab]
        prob, tok = softmax.max
        accept = 1 + longest_prefix(prob[1:] > tau)   (>=1 always)
        commit tok[:accept], crop cache to committed length
    """
    device = input_ids.device
    args = model.args
    D = args.block_size
    MASK = args.mask_token_id

    seq_len = input_ids.shape[-1]
    generated = input_ids[0].clone()  # 1-D
    cache: List[Tuple[torch.Tensor, torch.Tensor]] = make_sdlm_cache(args, device=device)
    prefill_time = 0.0
    n_iters = 0
    accept_total = 0

    t0 = time.time()
    while generated.numel() < seq_len + max_gen_len:
        n_iters += 1

        past_len = cache[0][0].shape[2]
        # Inputs: [generated[past_len:], anchor=generated[-1], MASK*(D-1)]
        tail = generated[past_len:]
        anchor = generated[-1:]
        masks = torch.full((D - 1,), MASK, dtype=generated.dtype, device=device)
        x = torch.cat([tail, anchor, masks])

        # Positions:
        #   tail[i] -> past_len + i
        #   anchor  -> generated.numel() - 1  (reuses last committed position)
        #   mask[i] -> generated.numel() + i
        commit_end = generated.numel()        # past_len + tail.numel()
        pos_tail = torch.arange(past_len, commit_end, device=device)
        pos_anchor = torch.tensor([commit_end - 1], device=device)
        pos_masks = torch.arange(commit_end, commit_end + D - 1, device=device)
        positions = torch.cat([pos_tail, pos_anchor, pos_masks])

        # Use forward_sdlm_prefill with appended block, then crop cache.
        # (More robust than forward_sdlm_decode for the iter 1 case where the
        # cache starts at length 0.)
        # However, repeatedly running prefill on full tail re-computes KV for
        # already-committed tokens. We instead use _decode_ when past_len >= 1
        # and the tail length matches the "committed-since-last-call" delta.
        if past_len == 0:
            # First iter: full prefill.
            logits = model.forward_sdlm_prefill(
                token_values=x, cache=cache, positions=positions,
                block_size=D, fast=False, causal_attn=True,
            )
        else:
            # Subsequent iters: x = [committed-since-last + anchor + MASK*(D-1)].
            # Use forward_sdlm_prefill with non-empty `cache` — it will
            # concatenate the new KV into the layer caches and return logits
            # for the whole new chunk; we then take the last D rows.
            logits = model.forward_sdlm_prefill(
                token_values=x, cache=cache, positions=positions,
                block_size=D, fast=False, causal_attn=True,
            )
        block_logits = logits[-D:]

        # Confidence and token IDs from logits.
        probs = F.softmax(block_logits.float(), dim=-1)
        confidence, tokens = probs.max(dim=-1)  # [D]

        # Decide accept length (HF semantics):
        #   gate = confidence > tau                    (length D)
        #   first_false = argmax(~gate)                (0..D-1)
        #   accept = max(1, min(D-if-all-true else first_false, D))
        # NB: this slices `tokens[:accept]` starting at index 0 (anchor's
        # re-prediction).
        gate = confidence > threshold
        if gate.all():
            accept_count = D
        else:
            first_false = int(torch.argmax((~gate).int()).item())
            accept_count = max(1, min(first_false, D))

        # Crop trailing EOS.
        new_tokens = tokens[:accept_count]
        eos_hit_idx = -1
        for j in range(accept_count):
            if int(new_tokens[j].item()) in eos_ids:
                eos_hit_idx = j
                break
        if eos_hit_idx >= 0:
            new_tokens = new_tokens[:eos_hit_idx]
            accept_count = eos_hit_idx
        accept_total += accept_count

        # Crop the cache BEFORE committing new tokens (matches HF semantics).
        # HF does `past_kv.crop(generated.size(1))` while `generated` is still
        # the previous-iter length. The newly-accepted tokens' KV are then
        # *recomputed* in the next iter using the real accepted-token IDs as
        # the input tail — NOT the MASK-token KV from this iter. Cropping after
        # commit would keep the MASK-as-input KV, which is numerically wrong.
        cache = _crop_cache(cache, generated.numel())
        if accept_count > 0:
            generated = torch.cat([generated, new_tokens])

        if n_iters == 1:
            torch.cuda.synchronize()
            prefill_time = time.time() - t0

        if eos_hit_idx >= 0:
            break

    torch.cuda.synchronize()
    total_time = time.time() - t0
    decode_time = max(total_time - prefill_time, 1e-9)
    out = generated[seq_len:]
    return out, {
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "total_time": total_time,
        "prefill_tokens": seq_len,
        "decode_tokens": int(out.numel()),
        "n_iters": n_iters,
        "avg_accept": accept_total / max(1, n_iters),
        "ttft": prefill_time,
    }


# ============================================================================
# Eager ternary speculative / hybrid decode
# ============================================================================

def _eager_draft_block(
    model: Transformer,
    cache,
    pending: torch.Tensor,
    cache_len: int,
    D: int,
    mask_token_id: int,
):
    """Materialize pending tokens and produce one SDLM width-D draft block."""
    device = pending.device
    n_pend = int(pending.numel())
    anchor = pending[-1:]
    masks = torch.full(
        (D - 1,), mask_token_id, dtype=pending.dtype, device=device)
    token_values = torch.cat([pending, anchor, masks])
    positions = torch.cat([
        torch.arange(cache_len, cache_len + n_pend, device=device),
        torch.tensor([cache_len + n_pend - 1], device=device),
        torch.arange(
            cache_len + n_pend,
            cache_len + n_pend + D - 1,
            device=device,
        ),
    ])
    logits = model.forward_sdlm_prefill(
        token_values=token_values,
        cache=cache,
        positions=positions,
        block_size=D,
        fast=False,
        causal_attn=True,
    )[-D:]
    # Keep only the real pending-token KV; the duplicate anchor and MASK KV
    # are draft workspace and must never enter committed history.
    cache_len += n_pend
    cache = _crop_cache(cache, cache_len)
    max_logits, draft_tokens = logits.max(dim=-1)
    confidence = torch.exp(max_logits.float() - torch.logsumexp(logits.float(), dim=-1))
    return cache, cache_len, draft_tokens, confidence


def _exact_m1_verify(
    model: Transformer,
    cache,
    cache_len: int,
    anchor: torch.Tensor,
    draft_tokens: torch.Tensor,
):
    """Verify every draft with the same M=1 path as ``gpu_ar_generate``.

    A width-(D+1) causal verify is mathematically equivalent, but CUDA GEMM
    kernels selected for M=1 and M>1 can differ at near-tie argmaxes.  This
    sequential verifier is deliberately slower and is the token-exact gate.
    """
    device = anchor.device
    anchor_pos = cache_len - 1
    verify_cache = _crop_cache(cache, anchor_pos)
    cur = anchor
    accepted: List[int] = []

    for j in range(int(draft_tokens.numel())):
        pos = torch.tensor([anchor_pos + j], device=device)
        logits, new_ks, new_vs = model.forward_sdlm_decode(
            token_values=cur,
            positions=pos,
            cache=verify_cache,
            block_size=1,
            causal_attn=True,
        )
        verify_cache = list(zip(new_ks, new_vs))
        pred = int(logits[-1].argmax().item())
        proposed = int(draft_tokens[j].item())
        if proposed != pred:
            return accepted, pred, verify_cache, anchor_pos + 1 + len(accepted)
        accepted.append(proposed)
        cur = draft_tokens[j:j + 1]

    # All D drafts matched. Materialize the final accepted token and obtain
    # the fresh AR bonus token.
    pos = torch.tensor([anchor_pos + len(accepted)], device=device)
    logits, new_ks, new_vs = model.forward_sdlm_decode(
        token_values=cur,
        positions=pos,
        cache=verify_cache,
        block_size=1,
        causal_attn=True,
    )
    verify_cache = list(zip(new_ks, new_vs))
    bonus = int(logits[-1].argmax().item())
    return accepted, bonus, verify_cache, anchor_pos + 1 + len(accepted)


def _append_until_eos(
    output: List[int], values: Sequence[int], eos_ids, max_gen_len: int,
):
    kept = []
    eos_hit = False
    for value in values:
        value = int(value)
        if value in eos_ids:
            eos_hit = True
            break
        if len(output) >= max_gen_len:
            break
        output.append(value)
        kept.append(value)
    return kept, eos_hit


@torch.no_grad()
def _gpu_sdlm_spec_or_hybrid_generate(
    model: Transformer,
    input_ids: torch.Tensor,
    max_gen_len: int,
    eos_ids,
    threshold: float,
    hybrid_m: Optional[int],
):
    device = input_ids.device
    args = model.args
    D = args.block_size
    prompt = input_ids[0]
    seq_len = int(prompt.numel())

    t0 = time.time()
    cache = make_sdlm_cache(args, device=device)
    positions = torch.arange(seq_len, device=device)
    logits = model.forward_sdlm_prefill(
        token_values=prompt,
        cache=cache,
        positions=positions,
        block_size=1,
        fast=False,
        causal_attn=True,
    )
    first = int(logits[-1].argmax().item())
    torch.cuda.synchronize()
    prefill_time = time.time() - t0

    output: List[int] = []
    n_iters = 1
    accept_total = 0
    n_conf_path = 0
    n_verify_path = 0
    if first in eos_ids or max_gen_len <= 0:
        return torch.empty(0, dtype=torch.long, device=device), {
            "prefill_time": prefill_time,
            "decode_time": 1e-9,
            "total_time": prefill_time,
            "prefill_tokens": seq_len,
            "decode_tokens": 0,
            "n_iters": n_iters,
            "avg_accept": 0.0,
            "ttft": prefill_time,
        }

    output.append(first)
    accept_total = 1
    pending = torch.tensor([first], dtype=prompt.dtype, device=device)
    cache_len = seq_len
    eos_found = False

    t1 = time.time()
    while len(output) < max_gen_len and not eos_found:
        n_iters += 1
        cache, cache_len, draft, confidence = _eager_draft_block(
            model, cache, pending, cache_len, D, args.mask_token_id)

        use_conf_path = False
        streak = 0
        if hybrid_m is not None:
            gate = confidence > threshold
            if bool(gate.all()):
                streak = D
            else:
                streak = max(1, int(torch.argmax((~gate).int()).item()))
            use_conf_path = streak >= hybrid_m

        if use_conf_path:
            n_conf_path += 1
            proposed = [int(x) for x in draft[:streak].tolist()]
            kept, eos_found = _append_until_eos(
                output, proposed, eos_ids, max_gen_len)
            accept_total += len(kept)
            if kept and not eos_found and len(output) < max_gen_len:
                pending = torch.tensor(kept, dtype=prompt.dtype, device=device)
            else:
                break
        else:
            n_verify_path += 1
            accepted, bonus, cache, cache_len = _exact_m1_verify(
                model, cache, cache_len, pending[-1:], draft)
            committed = accepted + [bonus]
            kept, eos_found = _append_until_eos(
                output, committed, eos_ids, max_gen_len)
            accept_total += len(kept)
            if not eos_found and kept and len(output) < max_gen_len:
                # The bonus is the only committed token not yet in KV.
                pending = torch.tensor([kept[-1]], dtype=prompt.dtype, device=device)
            else:
                break

    torch.cuda.synchronize()
    decode_time = max(time.time() - t1, 1e-9)
    total_time = prefill_time + decode_time
    result = torch.tensor(output, dtype=torch.long, device=device)
    stats = {
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "total_time": total_time,
        "prefill_tokens": seq_len,
        "decode_tokens": len(output),
        "n_iters": n_iters,
        "avg_accept": accept_total / max(1, n_iters),
        "ttft": prefill_time,
    }
    if hybrid_m is not None:
        stats.update({
            "n_conf_path": n_conf_path,
            "n_verify_path": n_verify_path,
            "conf_path_frac": n_conf_path / max(1, n_conf_path + n_verify_path),
        })
    return result, stats


@torch.no_grad()
def gpu_sdlm_speculative_generate(
    model: Transformer,
    input_ids: torch.Tensor,
    max_gen_len: int = 512,
    eos_ids=_BONSAI_EOS_IDS,
) -> Tuple[torch.Tensor, dict]:
    """Token-exact eager self-speculation for frozen ternary weights."""
    return _gpu_sdlm_spec_or_hybrid_generate(
        model, input_ids, max_gen_len, eos_ids, threshold=1.0, hybrid_m=None)


@torch.no_grad()
def gpu_sdlm_hybrid_generate(
    model: Transformer,
    input_ids: torch.Tensor,
    threshold: float = 0.9,
    hybrid_m: int = 6,
    max_gen_len: int = 512,
    eos_ids=_BONSAI_EOS_IDS,
) -> Tuple[torch.Tensor, dict]:
    """Eager confidence-gated hybrid with exact M=1 verify fallback."""
    return _gpu_sdlm_spec_or_hybrid_generate(
        model, input_ids, max_gen_len, eos_ids,
        threshold=threshold, hybrid_m=hybrid_m)


if __name__ == "__main__":
    raise SystemExit("Use python -m bonsai_sft.gpu.run_generation for generation.")
