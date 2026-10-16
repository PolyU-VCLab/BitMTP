"""CUDA Graph atlas with shared per-segment KV history and workspace.

Graphs are keyed by replay width, capacity segment, and numerical role.
Each segment shares a KV buffer with σ history slots and M_max workspace
slots. Captured forwards write at offset σ; GPU mask updates expose the
materialized prefix and the appropriate causal workspace rows.

Draft replays materialize pending rows. Verifier replays promote accepted
draft rows after the repeated anchor. Width and role switches reuse history;
segment growth copies the materialized prefix to a larger buffer.

Draft widths cover the pending suffix plus the D-row anchor/mask block.
Batched verification uses width D+1, and AR uses width 1. The evaluation
profile supplies the captured width set.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from .contracts import validate_graph_roles
from .runtime_config import compile_helper, is_orin_profile

from .model import (
    ModelArgs,
    Transformer,
    SDLMLayerCache,
)

_flex_attention = None
_create_block_mask = None
_FLEX_AVAILABLE = False
if not is_orin_profile():
    try:
        from torch.nn.attention.flex_attention import (
            flex_attention as _flex_attention,
            create_block_mask as _create_block_mask,
        )
        _FLEX_AVAILABLE = True
    except Exception:
        _FLEX_AVAILABLE = False


# ===========================================================================
# Default sizing
# ===========================================================================
DEFAULT_SEGMENT_LADDER = (256, 1024, 4096)
DEFAULT_M_SET = (1, 8, 9, 10, 12, 16, 17)
ROLES = ("draft", "verify", "ar")


# ===========================================================================
# Data structure
# ===========================================================================
@dataclass
class CapturedGraph:
    cuda_graph: "torch.cuda.CUDAGraph"
    tokens_buf: torch.Tensor              # int64 [M]
    positions_buf: torch.Tensor           # int64 [M]
    attn_mask: torch.Tensor               # [1, 1, M, σ+M] bf16
    logits_out: torch.Tensor              # [logits_end-logits_start, vocab]
    kv_full: List[SDLMLayerCache]         # views into shared per-σ buffer
    M: int
    sigma: int                            # max_kv_len
    role: str
    P: int                                # M - D
    logits_start: int
    logits_end: int
    lm_head_precision: str
    flex_attn_fn: Optional[object] = None  # per-graph compiled flex_attention


class GraphAtlas:
    """Pre-captured CUDA Graph atlas keyed by (M, σ, role)."""

    def __init__(
        self,
        model: Transformer,
        args: ModelArgs,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        segment_ladder: Tuple[int, ...] = DEFAULT_SEGMENT_LADDER,
        m_set: Tuple[int, ...] = DEFAULT_M_SET,
        roles: Tuple[str, ...] = ROLES,
        capture_verbose: bool = True,
        use_flex_attention: bool = True,
    ):
        self.model = model
        self.args = args
        self.device = device
        self.dtype = dtype
        self.segment_ladder = tuple(sorted(set(segment_ladder)))
        self.m_set = tuple(sorted(set(m_set)))
        self.roles = tuple(roles)
        validate_graph_roles(self.roles)
        self.D = args.block_size
        self.M_max = max(self.m_set)
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.n_kv = args.n_kv_heads
        self.head_dim = args.head_dim
        self.use_flex_attention = bool(use_flex_attention) and _FLEX_AVAILABLE

        if capture_verbose:
            print(f"[GraphAtlas] D={self.D}, segments={self.segment_ladder}, "
                  f"M_set={self.m_set}, roles={self.roles}, "
                  f"flex_attention={self.use_flex_attention}")

        if "draft_fp8" in self.roles:
            validate_graph_roles(self.roles, torch.cuda.get_device_capability(self.device))
            self.model.prepare_fp8_lm_head()
        if getattr(self.args, "kernel_kind", "q1") == "t2g128":
            # The ternary checkpoint trains the tied projection as W2A8, so
            # every graph role can use the exact packed Q2_0_g128 lm_head.
            self.model.prepare_t2_lm_head()

        self._allocate_shared_kv()
        self._allocate_input_bufs()
        self._capture_all()
        self._build_mask_templates()
        self._build_verify_templates()

    # -----------------------------------------------------------------
    # Shared KV buffer per σ. All (M, role) graphs at this σ alias into
    # this single buffer so that switching M is free (no KV copy).
    # -----------------------------------------------------------------
    def _allocate_shared_kv(self) -> None:
        self.kv_buffers: Dict[int, torch.Tensor] = {}
        for sigma in self.segment_ladder:
            buf = torch.zeros(
                2, self.n_layers, 1, self.n_kv,
                sigma + self.M_max, self.head_dim,
                device=self.device, dtype=self.dtype,
            )
            self.kv_buffers[sigma] = buf

    # -----------------------------------------------------------------
    # Per-M input buffers (tokens & positions). int64 because
    # nn.Embedding expects long indices.
    # -----------------------------------------------------------------
    def _allocate_input_bufs(self) -> None:
        self.tokens_m: Dict[int, torch.Tensor] = {}
        self.positions_m: Dict[int, torch.Tensor] = {}
        for M in self.m_set:
            self.tokens_m[M] = torch.zeros(M, device=self.device, dtype=torch.int64)
            self.positions_m[M] = torch.zeros(M, device=self.device, dtype=torch.int64)

    # -----------------------------------------------------------------
    # CUDA Graph capture for every (M, σ, role).
    # -----------------------------------------------------------------
    def _capture_all(self) -> None:
        """Capture (M, σ, role) graphs. M=1 only captures the 'ar' role
        (single-token AR step); M>=D captures draft + verify per request."""
        self.graphs: Dict[Tuple[int, int, str], CapturedGraph] = {}
        for sigma in self.segment_ladder:
            for M in self.m_set:
                if M == 1:
                    if "ar" in self.roles:
                        cg = self._capture_one(M, sigma, "ar")
                        self.graphs[(M, sigma, "ar")] = cg
                    continue
                for role in self.roles:
                    if role == "ar":
                        continue
                    cg = self._capture_one(M, sigma, role)
                    self.graphs[(M, sigma, role)] = cg

    def _capture_one(self, M: int, sigma: int, role: str) -> CapturedGraph:
        D = self.D
        P = M - D
        exact_lowbit_head = (
            "t2" if getattr(self.args, "kernel_kind", "q1") == "t2g128"
            else "bf16"
        )

        if role == "ar":
            # AR single-token: only the one row's logits.
            logits_start, logits_end = 0, M
            lm_head_precision = exact_lowbit_head
        elif role == "draft":
            # Draft: only the D block-token rows need logits.
            logits_start, logits_end = P, M
            lm_head_precision = exact_lowbit_head
        elif role == "draft_fp8":
            # Draft with FP8 output projection (acceptance-tolerant role,
            # BitMTP M1). Verify/AR retain the backbone-native projection:
            # packed T2 for ternary checkpoints, BF16 for the Q1 backbone path.
            logits_start, logits_end = P, M
            lm_head_precision = "fp8"
        elif role == "draft_bf16":
            # RTX 3090 / SM86 cannot execute torch._scaled_mm FP8.  The paper
            # specifies BF16 for the acceptance-tolerant draft projection on
            # this hardware; backbone layers remain packed T2/W2A8 and the
            # verifier/AR roles retain their exact packed T2 projection.
            logits_start, logits_end = P, M
            lm_head_precision = "bf16_raw"
        elif role == "verify":
            # Verify: rows [0, D+1) — pending P=1 + the D block-token rows.
            logits_start, logits_end = 0, min(D + 1, M)
            lm_head_precision = exact_lowbit_head
        else:
            raise ValueError(f"unsupported graph role: {role!r}")

        # Shared KV slice [2, n_layers, 1, n_kv, σ+M, head_dim].
        shared = self.kv_buffers[sigma]
        slot = shared[:, :, :, :, : sigma + M, :]
        k_per_layer = slot[0]                                  # [n_layers,1,n_kv,σ+M,hd]
        v_per_layer = slot[1]
        kv_full = [(k_per_layer[i], v_per_layer[i]) for i in range(self.n_layers)]

        # Per-graph mask (mutable, host fills before replay).
        attn_mask = torch.full(
            (1, 1, M, sigma + M), float('-inf'),
            device=self.device, dtype=self.dtype,
        )

        tokens_buf = self.tokens_m[M]
        positions_buf = self.positions_m[M]

        # ---- Build & inject per-graph flex_attention fn (closure on
        #      this graph's attn_mask). Mirrors BitMTP `generate.py`:
        #      score_mod reads mask_2d[q_idx, kv_idx], enable_gqa=True
        #      avoids any K/V expand+reshape memcopy. ----
        flex_attn_fn = None
        if self.use_flex_attention:
            mask_2d = attn_mask[0, 0]  # view, captured by closure
            kv_len = sigma + M

            def _noop_mask(b, h, q_idx, kv_idx):
                return q_idx >= 0
            block_mask = _create_block_mask(
                _noop_mask, B=1, H=None, Q_LEN=M, KV_LEN=kv_len,
                device=self.device,
            )

            @compile_helper(mode="max-autotune-no-cudagraphs")
            def flex_attn_fn(q, k, v, _mask_2d=mask_2d, _block_mask=block_mask):
                def score_mod(score, b, h, q_idx, kv_idx):
                    return score + _mask_2d[q_idx, kv_idx]
                return _flex_attention(
                    q, k, v, score_mod=score_mod,
                    block_mask=_block_mask, enable_gqa=True,
                )

            for layer in self.model.layers:
                layer.attention._flex_attn_fn = flex_attn_fn
        else:
            for layer in self.model.layers:
                layer.attention._flex_attn_fn = None

        # ---- Warmup outside graph: lets @torch.compile cache + populates
        #      cudnn/cublas handles so capture doesn't see lazy init. ----
        if not is_orin_profile():
            torch._dynamo.reset()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = self.model.forward_sdlm_decode_graph_kernel(
                    token_values=tokens_buf,
                    positions=positions_buf,
                    kv_full=kv_full,
                    block_size=M,
                    attn_mask=attn_mask,
                    logits_start=logits_start,
                    logits_end=logits_end,
                    lm_head_precision=lm_head_precision,
                )
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # ---- Capture ----
        cuda_graph = torch.cuda.CUDAGraph()
        recording_kwargs: Dict = {}
        try:
            # PyTorch 2.10+: thread_local capture mode lets parallel streams
            # capture without deadlock.
            ann = torch.cuda.graph.__init__.__annotations__
            if "capture_error_mode" in ann:
                recording_kwargs["capture_error_mode"] = "thread_local"
        except Exception:
            pass

        with torch.cuda.graph(cuda_graph, **recording_kwargs):
            logits = self.model.forward_sdlm_decode_graph_kernel(
                token_values=tokens_buf,
                positions=positions_buf,
                kv_full=kv_full,
                block_size=M,
                attn_mask=attn_mask,
                logits_start=logits_start,
                logits_end=logits_end,
                lm_head_precision=lm_head_precision,
            )

        return CapturedGraph(
            cuda_graph=cuda_graph,
            tokens_buf=tokens_buf,
            positions_buf=positions_buf,
            attn_mask=attn_mask,
            logits_out=logits,
            kv_full=kv_full,
            M=M, sigma=sigma, role=role, P=P,
            logits_start=logits_start, logits_end=logits_end,
            lm_head_precision=lm_head_precision,
            flex_attn_fn=flex_attn_fn,
        )

    # -----------------------------------------------------------------
    # Mask atlas: pre-compute templates per (n_pend, M, σ).
    # -----------------------------------------------------------------
    def _build_mask_templates(self) -> None:
        """Pre-build mask templates for confidence/spec/hybrid modes.

        Template shape: [1, 1, M, σ + M], -inf by default. Per-step GPU
        operations expose materialized history and hide a repeated anchor
        when needed. Workspace visibility is fixed in the template:

          rows [0, n_pend)     : pending committed tokens. Each row r sees
                                 past KV + pending slots [σ, σ + r].
          rows [n_pend, P)     : PAD (token=0, position=0). Attend to slot
                                 0 to avoid all-mask NaN; output discarded.
          row P                : anchor (= duplicate of pending[n_pend-1]).
          rows [P, M)          : diffusion block (anchor + D-1 masks).
                                 Block-causal triangle on slots [σ+P, σ+M);
                                 also see pending slots [σ, σ+n_pend-1)
                                 (i.e., earlier tail rows but NOT the
                                 last pending row, which is anchor_prev).
        """
        D = self.D
        self.mask_templates: Dict[Tuple[int, int, int], torch.Tensor] = {}
        for sigma in self.segment_ladder:
            for M in self.m_set:
                if M < D:
                    # M=1 (AR) uses an on-the-fly verify-style mask, no template.
                    continue
                P = M - D
                for np_val in range(0, P + 1):
                    tmpl = self._build_mask_template(M, sigma, P, D, np_val)
                    self.mask_templates[(np_val, M, sigma)] = tmpl

    def _build_mask_template(self, M: int, sigma: int, P: int, D: int,
                             n_pend: int) -> torch.Tensor:
        """One mask template for the (n_pend, M, σ) slot."""
        mask = torch.full(
            (1, 1, M, sigma + M), float('-inf'),
            device=self.device, dtype=self.dtype,
        )

        # Pending-causal: pending row r sees slots [σ, σ + r].
        for r in range(n_pend):
            for c in range(r + 1):
                mask[:, :, r, sigma + c] = 0.0

        # PAD rows (n_pend < r < P): attend to slot 0 to avoid all-mask
        # softmax → NaN. Output is unused.
        if n_pend < P:
            mask[:, :, n_pend:P, 0] = 0.0

        # Block-causal triangle for the diffusion block.
        for r in range(P, M):
            for c in range(r - P + 1):
                mask[:, :, r, sigma + P + c] = 0.0

        # Block rows see preceding pending rows EXCEPT the last (= anchor_prev).
        if n_pend > 1:
            mask[:, :, P:, sigma:sigma + n_pend - 1] = 0.0

        return mask

    # -----------------------------------------------------------------
    # Selection helpers
    # -----------------------------------------------------------------
    def select_segment(self, needed_kv_len: int) -> int:
        """Pick the smallest σ in the ladder ≥ needed_kv_len.

        `needed_kv_len` should already include the M decode width budget,
        i.e. the caller computes `cache_len + M`."""
        for sigma in self.segment_ladder:
            if needed_kv_len <= sigma:
                return sigma
        # Falls back to the largest segment; caller checks bounds.
        return self.segment_ladder[-1]

    # -----------------------------------------------------------------
    # Public per-step API used by fast_generate.
    # -----------------------------------------------------------------
    def prepare_step(
        self,
        graph: CapturedGraph,
        cache_len: int,
        n_pend: int,
        causal_attn: bool = True,
    ) -> None:
        """Fill the captured-graph mask buffer for this step.

        Branches by role:
          * 'draft'  : block-causal SDLM mask (template + history patch).
          * 'verify' : AR-causal mask with the repeated history anchor hidden;
                       row r sees trailing slots [σ, σ+r].
          * 'ar'     : AR-causal mask over history and the new input row.
        """
        # Native verification and AR use the AR-causal mask; draft roles
        # use the block-causal mask.
        if graph.role == 'ar' or graph.role.startswith('verify'):
            self._prepare_verify(graph, cache_len)
            return
        # ---- Draft path ----
        P = graph.P
        n_pend_clamp = max(0, min(n_pend, P))
        tmpl = self.mask_templates[(n_pend_clamp, graph.M, graph.sigma)]
        graph.attn_mask.copy_(tmpl)
        if cache_len > 0:
            graph.attn_mask[:, :, :, :cache_len] = 0.0
            if n_pend_clamp == 0:
                # Anchor_prev = last history slot, mask it for all rows
                # (P == 0, so [P:] == all rows).
                graph.attn_mask[:, :, :, cache_len - 1] = float('-inf')

    def _build_verify_templates(self) -> None:
        """Build masked-history, causal-workspace templates for verify/AR.

        Per-step preparation exposes materialized history and hides its
        final slot when the verifier replays that anchor.
        """
        self.verify_templates: Dict[Tuple[int, int], torch.Tensor] = {}
        for (M, sigma, role), g in self.graphs.items():
            if not (role == 'ar' or role.startswith('verify')):
                continue
            if (M, sigma) in self.verify_templates:
                continue
            tmpl = torch.full(
                (1, 1, M, sigma + M), float('-inf'),
                device=self.device, dtype=self.dtype,
            )
            for r in range(M):
                tmpl[:, :, r, sigma:sigma + r + 1] = 0.0
            self.verify_templates[(M, sigma)] = tmpl

    def _prepare_verify(self, graph: CapturedGraph, cache_len: int) -> None:
        """Prepare the AR-causal mask for a verify/AR graph (M ∈ {1, D+1}).

        Copy the template, expose materialized history, and hide the
        repeated history anchor for verifier roles.
        """
        M, sigma = graph.M, graph.sigma
        tmpl = self.verify_templates[(M, sigma)]
        graph.attn_mask.copy_(tmpl)
        if cache_len > 0:
            graph.attn_mask[:, :, :, :cache_len] = 0.0
        if graph.role != "ar":
            if not graph.role.startswith("verify"):
                raise RuntimeError(
                    f"verifier anchor de-dup called for role {graph.role!r}")
            if cache_len <= 0:
                raise RuntimeError("batched verifier requires a committed anchor")
            # The pending anchor was already materialized at history slot
            # cache_len-1 by the draft graph.  Verifier row 0 replays that same
            # token and position in the trailing region, so the history copy
            # must be hidden to avoid a duplicate anchor.  M=1 AR has no such
            # duplicate and is intentionally unchanged.
            graph.attn_mask[:, :, :, cache_len - 1] = float("-inf")

    def commit_kv(
        self,
        graph: CapturedGraph,
        cache_len: int,
        n_accept: int,
    ) -> None:
        """Copy consecutive replay KV rows into history within one segment.

        Source rows [σ:σ+n_accept] and destination rows
        [cache_len:cache_len+n_accept] share the same physical buffer.
        """
        if n_accept <= 0:
            return
        sigma = graph.sigma
        shared = self.kv_buffers[sigma]
        shared[:, :, :, :, cache_len:cache_len + n_accept, :].copy_(
            shared[:, :, :, :, sigma:sigma + n_accept, :]
        )

    def commit_kv_from_verify(
        self,
        verify_graph: CapturedGraph,
        cache_len: int,
        n_accept: int,
    ) -> None:
        """Reuse verify graph's draft-slot KV.

        The verify graph wrote D+1 rows into slots [σ, σ+D] at AR-causal
        positions (row 0 = anchor, rows 1..D = draft tokens). After verify,
        the first `n_accept` draft tokens are accepted; their KV at slots
        [σ+1, σ+1+n_accept) is AR-correct and can be copied directly into
        history[cache_len:cache_len+n_accept], saving an `n_accept`-token
        re-forward in the next iter's draft step (M_draft = D+1 instead of
        D+n_accept+1).
        """
        if n_accept <= 0:
            return
        sigma = verify_graph.sigma
        shared = self.kv_buffers[sigma]
        shared[:, :, :, :, cache_len:cache_len + n_accept, :].copy_(
            shared[:, :, :, :, sigma + 1:sigma + 1 + n_accept, :]
        )

    def migrate_segment(
        self,
        src_sigma: int,
        dst_sigma: int,
        cache_len: int,
    ) -> None:
        """Copy live KV [0, cache_len) from one σ buffer to another."""
        if src_sigma == dst_sigma:
            return
        src = self.kv_buffers[src_sigma]
        dst = self.kv_buffers[dst_sigma]
        dst[:, :, :, :, :cache_len, :].copy_(src[:, :, :, :, :cache_len, :])

    def get_graph(self, M: int, sigma: int, role: str) -> CapturedGraph:
        return self.graphs[(M, sigma, role)]


# ===========================================================================
# Prefill helper: populate a target σ KV buffer with the prompt's KV.
# ===========================================================================
def prefill_to_atlas(
    model: Transformer,
    atlas: GraphAtlas,
    prompt_ids: torch.Tensor,
    target_sigma: int,
    fast: bool = False,
    causal_attn: bool = True,
) -> int:
    """Run a single variable-shape prefill, write KV into atlas.kv_buffers[σ].

    Returns the new cache_len."""
    from .model import make_sdlm_cache
    args = model.args
    device = atlas.device
    seq_len = prompt_ids.shape[0]
    cache = make_sdlm_cache(args, device=device)
    positions = torch.arange(seq_len, device=device)
    _ = model.forward_sdlm_prefill(
        token_values=prompt_ids,
        cache=cache,
        positions=positions,
        block_size=1,
        fast=fast,
        causal_attn=causal_attn,
    )
    # Copy resulting cache into target σ buffer.
    shared = atlas.kv_buffers[target_sigma]
    for i, (ck, cv) in enumerate(cache):
        L = ck.shape[2]
        shared[0, i, :, :, :L, :].copy_(ck[0])
        shared[1, i, :, :, :L, :].copy_(cv[0])
    return seq_len
