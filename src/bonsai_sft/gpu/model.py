# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Includes xFormers-derived code under the BSD 3-Clause license.
# See licenses/xFormers-BSD.txt for the copyright notice and license terms.

"""Packed Q1/T2-g128 inference for Bonsai/Qwen3 models.

Adapted from BitMTP's W2A8 inference model. Architecture dimensions are
supplied through ModelArgs. The model provides prefill, captured block
drafting, and AR-causal verification, with per-head Q/K normalization,
SwiGLU, and YaRN-scaled rotary embeddings.

Q1 packs eight sign bits per byte; T2 packs four ternary codes per byte.
Both store per-row BF16 scales for groups of 128 input weights. Packed
linear layers use per-token INT8 activation quantization.
"""
from __future__ import annotations

import ctypes
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import require_kernel_shape, supports_layer_kernel_shape
from .runtime_config import compile_helper, is_orin_profile

# ---------------------------------------------------------------------------
# Optional acceleration. Orin uses SDPA prefill and explicit captured attention.
# ---------------------------------------------------------------------------
_HAS_FLEX = False
_HAS_XFORMERS = False
AttnBias = None
_XformersRMSNorm = None
if not is_orin_profile():
    try:
        import torch._dynamo  # noqa: F401
        torch._dynamo.config.capture_scalar_outputs = True
        from torch.nn.attention.flex_attention import (
            flex_attention as _flex_attention,  # noqa: F401
            create_block_mask as _create_block_mask,  # noqa: F401
        )
        _HAS_FLEX = True
    except Exception:
        _HAS_FLEX = False

    try:
        from xformers.ops import fmha, rope_padded  # noqa: F401
        from xformers.ops.fmha.attn_bias import (
            BlockDiagonalCausalWithOffsetPaddedKeysMask as AttnBias,
        )
        from xformers.ops import RMSNorm as _XformersRMSNorm
        _HAS_XFORMERS = True
    except Exception:
        _HAS_XFORMERS = False
        AttnBias = None
        _XformersRMSNorm = None


# ===========================================================================
# Q1_0_g128 CUDA kernel loader / dispatch
# ===========================================================================
_q1_lib = None
_q1_lib_path = None


def load_q1_kernel() -> None:
    """Load the Q1_0_g128 (W1A8) kernel `.so`.

    The Bonsai library is independent of the model draft block size and
    dispatches by complete GEMM shape (M,N,K); see contracts.py.
    """
    global _q1_lib, _q1_lib_path
    here = os.path.dirname(os.path.abspath(__file__))
    so = os.path.join(here, "q1g128_kernels", "libbonsai_q1.so")
    _q1_lib = ctypes.CDLL(so)
    _q1_lib_path = so
    _q1_lib.q1g128_int8xint1_linear.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
    ]
    _q1_lib.q1g128_int8xint1_linear.restype = None
    _q1_lib.q1g128_w1a16_linear.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    _q1_lib.q1g128_w1a16_linear.restype = None
    print(f"[Bonsai-GPU] loaded Q1 kernel: {so} (registered shapes in contracts.py)")


def _validate_packed_inputs(a, b, s, ws, *, kind, bits, activation_dtype):
    """Validate tensor metadata before a native call (no CUDA work)."""
    if a.dim() not in (2, 3) or b.dim() != 2:
        raise ValueError(f"{kind}: expected rank-2/3 activation and rank-2 weight")
    m = a.numel() // a.shape[-1] if a.shape[-1] else 0
    n, k = b.shape[0], b.shape[1] * (8 // bits)
    require_kernel_shape(kind, m, n, k)
    if a.shape[-1] != k:
        raise RuntimeError(f"{kind}: activation width mismatch: {a.shape[-1]} != {k}")
    if tuple(ws.shape) != (n, k // 128):
        raise RuntimeError(f"{kind}: invalid weight scale shape {tuple(ws.shape)}")
    if a.dtype != activation_dtype or b.dtype != torch.int8 or ws.dtype != torch.bfloat16:
        raise RuntimeError(f"{kind}: invalid activation, packed-weight, or scale dtype")
    tensors = (a, b, ws) if s is None else (a, b, s, ws)
    if s is not None and (s.dtype != torch.float32 or s.shape[-1:] != (1,) or s.numel() != m):
        raise RuntimeError(f"{kind}: activation scales must be float32 with one value per row")
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        raise RuntimeError(f"{kind}: inputs and scales must be contiguous CUDA tensors")
    if not all(t.device == a.device for t in tensors):
        raise RuntimeError(f"{kind}: inputs and scales must share one CUDA device")


def q1g128_int8xint1_linear(
    a_int8: torch.Tensor,   # [..., K] int8
    b_packed: torch.Tensor, # [N, K/8] int8 (sign bits packed)
    s: torch.Tensor,        # [..., 1] float32 (per-row act scale)
    ws: torch.Tensor,       # [N, K/128] bf16 (per-group weight scale)
) -> torch.Tensor:
    """Wrapper that flattens batch dims, calls the kernel, restores shape."""
    global _q1_lib
    _validate_packed_inputs(a_int8, b_packed, s, ws,
                            kind="q1", bits=1, activation_dtype=torch.int8)
    if _q1_lib is None:
        load_q1_kernel()
    if a_int8.dim() == 2:
        M = a_int8.shape[0]
        out_shape = (M, b_packed.shape[0])
    elif a_int8.dim() == 3:
        bsz, slen, _ = a_int8.shape
        M = bsz * slen
        out_shape = (bsz, slen, b_packed.shape[0])
    else:
        raise ValueError(f"q1g128 wrapper: bad input shape {a_int8.shape}")

    N = b_packed.shape[0]
    K = b_packed.shape[1] * 8  # 8 sign bits per byte
    stream = torch.cuda.current_stream()
    out = torch.empty(*out_shape, dtype=torch.bfloat16, device=a_int8.device)
    _q1_lib.q1g128_int8xint1_linear(
        ctypes.c_void_p(a_int8.data_ptr()),
        ctypes.c_void_p(b_packed.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(s.data_ptr()),
        ctypes.c_void_p(ws.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K),
        ctypes.c_void_p(stream.cuda_stream),
    )
    return out


_w1a16_out_cache: dict = {}


def q1g128_w1a16_linear(
    a_bf16: torch.Tensor,   # [M, K] bf16 activation (NOT quantized -> exact)
    b_packed: torch.Tensor, # [N, K/8] int8 (sign bits packed)
    ws: torch.Tensor,       # [N, K/128] bf16 (per-group weight scale)
) -> torch.Tensor:
    """1-bit weight x BF16 activation GEMM (exact float accumulation).

    Uses a PERSISTENT output buffer (keyed by M,N,device): allocating with
    torch.empty INSIDE a CUDA graph lets the caching allocator place this
    huge [M,151936] buffer over a region that aliases other graphs' transient
    tensors in the shared graph pool -> the ctypes kernel write corrupts it on
    replay (torch can't see the ctypes read/write deps). A persistent buffer
    created at warmup time is captured once and safely reused across replays.
    """
    global _q1_lib
    if a_bf16.dim() != 2:
        raise ValueError("w1a16 requires a rank-2 activation")
    _validate_packed_inputs(a_bf16, b_packed, None, ws,
                            kind="w1a16", bits=1, activation_dtype=torch.bfloat16)
    if _q1_lib is None:
        load_q1_kernel()
    M = a_bf16.shape[0]
    N = b_packed.shape[0]
    K = b_packed.shape[1] * 8
    stream = torch.cuda.current_stream()
    key = (M, N, a_bf16.device)
    out = _w1a16_out_cache.get(key)
    if out is None:
        out = torch.empty(M, N, dtype=torch.bfloat16, device=a_bf16.device)
        _w1a16_out_cache[key] = out
    _q1_lib.q1g128_w1a16_linear(
        ctypes.c_void_p(a_bf16.data_ptr()),
        ctypes.c_void_p(b_packed.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(ws.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K),
        ctypes.c_void_p(stream.cuda_stream),
    )
    return out


# ===========================================================================
# Q2_0_g128 (ternary W2A8) CUDA kernel
# ===========================================================================
_t2_lib = None


def load_t2_kernel() -> None:
    """Load the Bonsai ternary g128 INT8 x packed-INT2 kernel."""
    global _t2_lib
    here = os.path.dirname(os.path.abspath(__file__))
    so = os.path.join(here, "t2g128_kernels", "libbonsai_t2.so")
    _t2_lib = ctypes.CDLL(so)
    _t2_lib.t2g128_int8xint2_linear.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
    ]
    _t2_lib.t2g128_int8xint2_linear.restype = None
    print(f"[Bonsai-GPU] loaded T2 g128 kernel: {so}")


def t2g128_int8xint2_linear(
    a_int8: torch.Tensor,
    b_packed: torch.Tensor,
    s: torch.Tensor,
    ws: torch.Tensor,
) -> torch.Tensor:
    """Run faithful Q2_0_g128 W2A8 linear on natural-order packed weights."""
    global _t2_lib
    _validate_packed_inputs(a_int8, b_packed, s, ws,
                            kind="t2g128", bits=2, activation_dtype=torch.int8)
    if a_int8.dim() == 2:
        M = a_int8.shape[0]
        out_shape = (M, b_packed.shape[0])
    elif a_int8.dim() == 3:
        bsz, slen, _ = a_int8.shape
        M = bsz * slen
        out_shape = (bsz, slen, b_packed.shape[0])
    else:
        raise ValueError(f"t2g128 wrapper: bad input shape {a_int8.shape}")
    N = b_packed.shape[0]
    K = b_packed.shape[1] * 4
    if _t2_lib is None:
        load_t2_kernel()
    out = torch.empty(*out_shape, dtype=torch.bfloat16, device=a_int8.device)
    stream = torch.cuda.current_stream()
    _t2_lib.t2g128_int8xint2_linear(
        ctypes.c_void_p(a_int8.data_ptr()),
        ctypes.c_void_p(b_packed.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(s.data_ptr()),
        ctypes.c_void_p(ws.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K),
        ctypes.c_void_p(stream.cuda_stream),
    )
    return out


def packed_int8_linear(a_int8, b_packed, s, ws):
    """Dispatch by packed width: Q1 has K/8 bytes, ternary Q2 has K/4."""
    K = a_int8.shape[-1]
    if b_packed.shape[-1] * 8 == K:
        return q1g128_int8xint1_linear(a_int8, b_packed, s, ws)
    if b_packed.shape[-1] * 4 == K:
        return t2g128_int8xint2_linear(a_int8, b_packed, s, ws)
    raise ValueError(
        f"unknown packed weight layout A.K={K}, B.shape={tuple(b_packed.shape)}"
    )


# ===========================================================================
# RMSNorm
# ===========================================================================
class RMSNorm(nn.Module):
    """Standard RMSNorm with optional xformers compiled fast path."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self._xformers_norm = None

    def _ensure_xformers(self):
        if self._xformers_norm is None and _HAS_XFORMERS:
            self._xformers_norm = _XformersRMSNorm(
                self.weight.shape[0], eps=self.eps,
            ).to(device=self.weight.device, dtype=self.weight.dtype)
            self._xformers_norm.weight.data.copy_(self.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.eps)
        return self.weight * xf.to(dtype)

    def forward_compiled(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_xformers()
        if self._xformers_norm is None:
            return self.forward(x)
        return self._xformers_norm(x)

    def forward_and_quant(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _fused_norm_quant(x, self.weight, self.eps)


@compile_helper
def _fused_norm_quant(
    x: torch.Tensor, w: torch.Tensor, eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype = x.dtype
    h = x.float()
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + eps)
    normed_bf16 = w * h.to(dtype)
    nf = normed_bf16.float()
    s = 127.0 / nf.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    q = (nf * s).round().clamp(-128, 127).to(torch.int8)
    return q, s


@compile_helper
def _fused_add_norm_quant(
    x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """h = x + residual; quant(RMSNorm(h)) -> (h, q_int8, s_fp32)."""
    h = x + residual
    dtype = h.dtype
    hf = h.float()
    var = hf.pow(2).mean(-1, keepdim=True)
    hf = hf * torch.rsqrt(var + eps)
    normed_bf16 = w * hf.to(dtype)
    nf = normed_bf16.float()
    s = 127.0 / nf.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    q = (nf * s).round().clamp(-128, 127).to(torch.int8)
    return h, q, s


# ===========================================================================
# YaRN-scaled RoPE
# ===========================================================================
def _yarn_inv_freq(
    head_dim: int,
    base: float,
    factor: float,
    original_max: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    device=None,
) -> torch.Tensor:
    """YaRN inverse-frequency vector (matches HF transformers `_compute_yarn_parameters`)."""
    dim = head_dim  # partial_rotary_factor=1.0 for Qwen3
    pos_freqs = base ** (torch.arange(0, dim, 2, device=device).float() / dim)
    inv_freq_extrap = 1.0 / pos_freqs
    inv_freq_interp = 1.0 / (factor * pos_freqs)

    def _find_correction_dim(num_rotations: float) -> float:
        return (dim * math.log(original_max / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(_find_correction_dim(beta_fast)), 0)
    high = min(math.ceil(_find_correction_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001

    linear = (torch.arange(dim // 2, device=device, dtype=torch.float32) - low) / (high - low)
    ramp = torch.clamp(linear, 0.0, 1.0)
    inv_freq_extrap_factor = 1.0 - ramp
    return inv_freq_interp * (1.0 - inv_freq_extrap_factor) + inv_freq_extrap * inv_freq_extrap_factor


def _yarn_attention_factor(scale: float, mscale: float = 1.0, mscale_all_dim: float = 0.0) -> float:
    """HF `attention_factor` (a.k.a. cos/sin post-scaling)."""
    if scale <= 1.0:
        return 1.0
    def _g(s, m):
        return 0.1 * m * math.log(s) + 1.0
    if mscale and mscale_all_dim:
        return _g(scale, mscale) / _g(scale, mscale_all_dim)
    return _g(scale, mscale)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@compile_helper
def _fused_qkv_qknorm_rope_kv_write(
    xqkv: torch.Tensor,         # [seq_len, q_dim+k_dim+v_dim]
    q_norm_w: torch.Tensor,     # [head_dim]
    k_norm_w: torch.Tensor,     # [head_dim]
    norm_eps: float,
    cos_table: torch.Tensor,    # [max_pos, head_dim]
    sin_table: torch.Tensor,    # [max_pos, head_dim]
    positions: torch.Tensor,    # [seq_len]
    kv_full_k: torch.Tensor,    # [1, n_kv, max_kv+seq_len, head_dim]
    kv_full_v: torch.Tensor,    # [1, n_kv, max_kv+seq_len, head_dim]
    q_dim: int, k_dim: int,
    n_heads: int, n_kv_heads: int, head_dim: int,
    max_kv_len: int,
) -> torch.Tensor:
    """Fuse split + reshape + per-head qk-RMSNorm + RoPE + in-place KV write.

    Bonsai/Qwen3 has explicit q_norm / k_norm (RMSNorm over the head_dim axis)
    before RoPE. The fused kernel does all of split, reshape, per-head norm,
    RoPE table lookup, and the in-place write of the new KV slot at the
    *fixed* offset `max_kv_len`. The fixed offset is critical: it lets
    CUDA Graph capture the write without depending on the dynamic cache_len.
    """
    seq_len = xqkv.shape[0]
    dtype = xqkv.dtype
    xq = xqkv[:, :q_dim].view(seq_len, n_heads, head_dim)
    xk = xqkv[:, q_dim:q_dim+k_dim].view(seq_len, n_kv_heads, head_dim)
    xv = xqkv[:, q_dim+k_dim:].view(seq_len, n_kv_heads, head_dim)

    xq_f = xq.float()
    qvar = xq_f.pow(2).mean(-1, keepdim=True)
    xq = (q_norm_w * (xq_f * torch.rsqrt(qvar + norm_eps)).to(dtype))

    xk_f = xk.float()
    kvar = xk_f.pow(2).mean(-1, keepdim=True)
    xk = (k_norm_w * (xk_f * torch.rsqrt(kvar + norm_eps)).to(dtype))

    xq = xq.permute(1, 0, 2).unsqueeze(0)
    xk = xk.permute(1, 0, 2).unsqueeze(0)
    xv = xv.permute(1, 0, 2).unsqueeze(0)

    cos = cos_table[positions].unsqueeze(0).unsqueeze(0)
    sin = sin_table[positions].unsqueeze(0).unsqueeze(0)
    xq = (xq * cos) + (rotate_half(xq) * sin)
    xk = (xk * cos) + (rotate_half(xk) * sin)

    kv_full_k[:, :, max_kv_len:max_kv_len + seq_len, :] = xk
    kv_full_v[:, :, max_kv_len:max_kv_len + seq_len, :] = xv
    return xq


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    mscale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE using halves mode (HF Qwen3 convention)."""
    dtype = q.dtype
    if positions.dim() == 1:
        positions = positions.unsqueeze(0)
    freqs = positions.float().unsqueeze(-1) * inv_freq.float().unsqueeze(0)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * mscale).unsqueeze(1).to(dtype)
    sin = (emb.sin() * mscale).unsqueeze(1).to(dtype)
    q_emb = (q * cos) + (rotate_half(q) * sin)
    k_emb = (k * cos) + (rotate_half(k) * sin)
    return q_emb, k_emb


# ===========================================================================
# SDLM attention mask helpers (same geometry as `bonsai_sft.sdlm_attn_mask`)
# ===========================================================================
def create_sdlm_attention_mask(
    q_len: int,
    kv_len: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
    past_len: int = 0,
    causal_attn: bool = True,
) -> torch.Tensor:
    """4-D SDLM attention mask. 0 = visible, -inf = masked.

    Matches `BonsaiForSDLM._create_sdlm_inference_mask` geometry:
        * Outside the diffusion block: standard causal.
        * Inside the diffusion block: bidirectional (causal_attn=False) or
          causal (causal_attn=True). For Bonsai-MTP T2 we always use the
          causal variant.
        * The anchor-prev key position (immediately before the block) is
          masked so the duplicated anchor cannot leak twice.
    """
    if block_size <= 1:
        # AR step: standard causal mask.
        mask = torch.full((q_len, kv_len), float("-inf"), device=device, dtype=dtype)
        for i in range(q_len):
            mask[i, : past_len + i + 1] = 0.0
        return mask.unsqueeze(0).unsqueeze(0)

    # Build a full standard causal mask first (matches HF `_create_sdlm_inference_mask`).
    mask = torch.full((q_len, kv_len), float("-inf"), device=device, dtype=dtype)
    for i in range(q_len):
        mask[i, : past_len + i + 1] = 0.0

    # SDLM block tweaks (always, regardless of causal_attn): the
    # last `block_size` queries form the diffusion block.
    if q_len >= block_size:
        block_start = q_len - block_size
        # Inside the block: if NOT causal, queries can see all block keys
        # (bidirectional). Otherwise stays causal.
        if not causal_attn:
            mask[block_start:, past_len + block_start :] = 0.0
        # Mask out the anchor_prev key for ALL block queries (HF semantics).
        anchor_prev_kv_idx = past_len + block_start - 1
        if anchor_prev_kv_idx >= 0:
            mask[block_start:, anchor_prev_kv_idx] = float("-inf")
    return mask.unsqueeze(0).unsqueeze(0)


def create_sdlm_mask_fixed(
    block_size: int,
    max_kv_len: int,
    cache_len: int,
    device: torch.device,
    dtype: torch.dtype,
    causal_attn: bool = True,
) -> torch.Tensor:
    """Fixed-shape SDLM mask for CUDA Graph: [1, 1, block_size, max_kv_len]."""
    mask = torch.full((1, 1, block_size, max_kv_len), float("-inf"),
                      device=device, dtype=dtype)
    # Build the standard causal block for q in [0..block_size-1]:
    # each query i can see past keys [0..cache_len-1] and block keys [0..i].
    for i in range(block_size):
        mask[:, :, i, : cache_len + i + 1] = 0.0
    if not causal_attn:
        # Bidirectional within the block.
        mask[:, :, :, cache_len:cache_len + block_size] = 0.0
    # Anchor_prev masking (matches HF semantics, always applied).
    if cache_len > 0:
        mask[:, :, :, cache_len - 1] = float("-inf")
    return mask


# ===========================================================================
# Linear / quant primitives
# ===========================================================================
SDLMLayerCache = Tuple[torch.Tensor, torch.Tensor]


class Q1LinearKernel(nn.Module):
    """Q1_0_g128 W1A8 Linear backed by the custom CUDA kernel.

    Stored weight tensors (also what `convert_bonsai_mtp_to_gpu.py` writes):
        weight        int8 [out_features, in_features // 8]   sign bits
        weight_scale  bf16 [out_features, in_features // 128] per-group scale

    For shapes the kernel doesn't yet support (e.g. prefill M=prompt_len), we
    unpack the weight to BF16 on-the-fly and fall back to `F.linear`. The
    unpacked tensor is cached on first use.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        assert bias is False, "Q1_0_g128 layers do not carry bias"
        assert in_features % 128 == 0, (
            f"Q1LinearKernel: in_features={in_features} must be divisible by 128"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features // 8, dtype=torch.int8),
            requires_grad=False,
        )
        # Per-group scales kept BF16; matches the kernel's `ws` interface and
        # the on-disk Q1_0_g128 GGUF format byte-for-byte.
        self.register_buffer(
            "weight_scale",
            torch.zeros(out_features, in_features // 128, dtype=torch.bfloat16),
            persistent=True,
        )
        self._bf16_weight_cache: Optional[torch.Tensor] = None

    @compile_helper
    def quant_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_f32 = x.float()
        s = 127.0 / x_f32.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return (x_f32 * s).round().clamp(-128, 127).to(torch.int8), s

    def _unpack_to_bf16(self) -> torch.Tensor:
        """Reconstruct the BF16 weight on the Q1 grid from packed int8 + scale."""
        if self._bf16_weight_cache is not None:
            return self._bf16_weight_cache
        packed = self.weight.view(torch.uint8)        # [N, K/8]
        N, KdivS = packed.shape
        K = KdivS * 8
        # Unpack: bit b of byte j -> weight j*8+b, value +1 if bit=1 else -1.
        bits = (packed.unsqueeze(-1) >> torch.arange(8, device=packed.device, dtype=torch.uint8)) & 1
        signs = (bits.to(torch.int8) * 2 - 1)         # int8 in {-1, +1}, shape [N, K/8, 8]
        signs = signs.view(N, K)                       # [N, K]
        # Apply per-group BF16 scale.
        scales = self.weight_scale                     # [N, K/128] bf16
        scales_full = scales.repeat_interleave(128, dim=1).to(torch.bfloat16)  # [N, K]
        w_bf16 = signs.to(torch.bfloat16) * scales_full
        self._bf16_weight_cache = w_bf16.contiguous()
        return self._bf16_weight_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Determine effective M.
        M = x.shape[0] if x.dim() == 2 else x.shape[0] * x.shape[1]
        if supports_layer_kernel_shape("q1", M, self.out_features, self.in_features):
            x_q, s = self.quant_input(x)
            return q1g128_int8xint1_linear(x_q, self.weight, s, self.weight_scale)
        # Fallback: BF16 GEMM with INT8-fake-quantised activations.
        w = self._unpack_to_bf16()
        x_q = Q1Linear.act_quant(x)
        return F.linear(x_q, w)


class T2LinearKernel(nn.Module):
    """Bonsai Q2_0_g128 W2A8 linear.

    ``weight`` stores four natural-order ternary codes per byte.  Code values
    1/2/3 reconstruct -1/0/+1; ``weight_scale`` is the BF16 group-128 absmax
    table with shape [out_features, in_features/128].
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        assert bias is False
        assert in_features % 128 == 0
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features // 4, dtype=torch.int8),
            requires_grad=False,
        )
        self.register_buffer(
            "weight_scale",
            torch.zeros(out_features, in_features // 128, dtype=torch.bfloat16),
            persistent=True,
        )
        self._bf16_weight_cache: Optional[torch.Tensor] = None

    @compile_helper
    def quant_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_f32 = x.float()
        s = 127.0 / x_f32.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return (x_f32 * s).round().clamp(-128, 127).to(torch.int8), s

    def _unpack_to_bf16(self) -> torch.Tensor:
        if self._bf16_weight_cache is not None:
            return self._bf16_weight_cache
        p = self.weight.view(torch.uint8)
        N, Kdiv4 = p.shape
        shifts = torch.arange(4, device=p.device, dtype=torch.uint8) * 2
        codes = ((p.unsqueeze(-1) >> shifts) & 3).to(torch.int8)
        q = (codes - 2).view(N, Kdiv4 * 4)
        scales = self.weight_scale.repeat_interleave(128, dim=1)
        self._bf16_weight_cache = (q.to(torch.bfloat16) * scales).contiguous()
        return self._bf16_weight_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        M = x.shape[0] if x.dim() == 2 else x.shape[0] * x.shape[1]
        if supports_layer_kernel_shape("t2g128", M, self.out_features, self.in_features):
            x_q, s = self.quant_input(x)
            return t2g128_int8xint2_linear(x_q, self.weight, s, self.weight_scale)
        x_q = Q1Linear.act_quant(x)
        return F.linear(x_q, self._unpack_to_bf16())


class Q1Linear(nn.Module):
    """BF16 linear over frozen-grid weights with per-token INT8 activation quantization."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        assert bias is False
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.bfloat16),
            requires_grad=False,
        )

    @staticmethod
    @compile_helper(mode="reduce-overhead")
    def act_quant(x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        s = 127.0 / xf.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return ((xf * s).round().clamp(-128, 127) / s).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conversion has already frozen the weight grid.
        x = self.act_quant(x)
        return F.linear(x, self.weight)


# ===========================================================================
# Bonsai (Qwen3) Attention
# ===========================================================================
@dataclass
class ModelArgs:
    """Bonsai-1.7B (Qwen3) defaults; override via `Transformer(ModelArgs(...))`."""
    dim: int = 2048
    n_layers: int = 28
    n_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 128
    ffn_dim: int = 6144
    vocab_size: int = 151669
    norm_eps: float = 1.0e-6
    rope_theta: float = 1.0e6
    rope_original_max: int = 8192
    rope_factor: float = 4.0
    rope_mscale: float = 1.0          # YaRN `mscale` (paper default 1)
    rope_mscale_all_dim: float = 0.0  # set >0 to use ratio formula
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    tie_word_embeddings: bool = True
    use_kernel: bool = False
    kernel_kind: str = "q1"       # "q1" or faithful ternary "t2g128"
    # SDLM-specific
    block_size: int = 8
    causal_attn: bool = True
    mask_token_id: int = 151654


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.head_dim = args.head_dim
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = args.n_kv_heads
        self.heads_per_group = args.n_heads // args.n_kv_heads
        self.q_dim = self.n_local_heads * self.head_dim
        self.k_dim = self.n_local_kv_heads * self.head_dim
        self.v_dim = self.n_local_kv_heads * self.head_dim

        Linear = (T2LinearKernel if args.kernel_kind == "t2g128" else Q1LinearKernel) \
            if args.use_kernel else Q1Linear
        self.q_proj = Linear(args.dim, self.q_dim, bias=False)
        self.k_proj = Linear(args.dim, self.k_dim, bias=False)
        self.v_proj = Linear(args.dim, self.v_dim, bias=False)
        self.o_proj = Linear(self.q_dim, args.dim, bias=False)

        # Qwen3 per-head RMSNorm over the head_dim axis (NOT hidden_size).
        # These eat full-precision BF16 weights (not Q1 quantised).
        self.q_norm = RMSNorm(self.head_dim, eps=args.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.norm_eps)

        inv_freq = _yarn_inv_freq(
            head_dim=self.head_dim,
            base=args.rope_theta,
            factor=args.rope_factor,
            original_max=args.rope_original_max,
            beta_fast=args.rope_beta_fast,
            beta_slow=args.rope_beta_slow,
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.rope_mscale = _yarn_attention_factor(
            args.rope_factor, args.rope_mscale, args.rope_mscale_all_dim,
        )

        # Optional flex_attention fn (set per CUDA-Graph segment by FastGen).
        self._flex_attn_fn = None

    # --------------------------------------------------------------- helpers
    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply per-head RMSNorm to q and k.
        Input shape: [..., n_heads, head_dim]."""
        return self.q_norm(q), self.k_norm(k)

    # ------------------------------------------------------ SDLM (slow, correct)
    def forward_sdlm(
        self,
        x: torch.Tensor,                  # [seq_len, dim]
        positions: torch.Tensor,          # [seq_len]
        cache_k: Optional[torch.Tensor],  # [1, n_kv, past_len, head_dim] or None
        cache_v: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = x.shape[0]
        xq = self.q_proj(x).view(seq_len, self.n_local_heads, self.head_dim)
        xk = self.k_proj(x).view(seq_len, self.n_local_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = self._apply_qk_norm(xq, xk)

        # -> [1, n_heads, seq_len, head_dim]
        xq = xq.permute(1, 0, 2).unsqueeze(0)
        xk = xk.permute(1, 0, 2).unsqueeze(0)
        xv = xv.permute(1, 0, 2).unsqueeze(0)

        xq, xk = apply_rotary_pos_emb(xq, xk, positions, self.inv_freq, self.rope_mscale)

        if cache_k is not None and cache_k.shape[2] > 0:
            k = torch.cat([cache_k, xk], dim=2)
            v = torch.cat([cache_v, xv], dim=2)
        else:
            k, v = xk, xv

        kv_len = k.shape[2]
        k_exp = k.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1) \
                 .reshape(1, self.n_local_heads, kv_len, self.head_dim).contiguous()
        v_exp = v.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1) \
                 .reshape(1, self.n_local_heads, kv_len, self.head_dim).contiguous()
        if is_causal:
            out = F.scaled_dot_product_attention(xq, k_exp, v_exp, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(xq, k_exp, v_exp, attn_mask=attn_mask)
        out = out.squeeze(0).permute(1, 0, 2).contiguous().view(seq_len, self.q_dim)
        out = self.o_proj(out)
        # Return full concatenated KV so callers can replace the layer cache
        # after prefill or incremental decoding.
        return out, k, v

    # --------------------------------------------- SDLM batched (no cache)
    def forward_sdlm_batch(
        self,
        x: torch.Tensor,         # [batch, seq_len, dim]
        positions: torch.Tensor, # [batch, seq_len]
        attn_mask: torch.Tensor, # [batch, 1, seq_len, seq_len]
    ) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        xq = self.q_proj(x_flat).view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = self.k_proj(x_flat).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = self.v_proj(x_flat).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = self._apply_qk_norm(xq, xk)

        xq = xq.transpose(1, 2)  # [bsz, n_heads, seq_len, head_dim]
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        xq, xk = apply_rotary_pos_emb(xq, xk, positions, self.inv_freq, self.rope_mscale)

        n_rep = self.heads_per_group
        if n_rep > 1:
            xk = xk.unsqueeze(2).expand(-1, -1, n_rep, -1, -1) \
                   .reshape(bsz, self.n_local_heads, seq_len, self.head_dim)
            xv = xv.unsqueeze(2).expand(-1, -1, n_rep, -1, -1) \
                   .reshape(bsz, self.n_local_heads, seq_len, self.head_dim)

        out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.q_dim)
        out = self.o_proj(out.view(-1, self.q_dim)).view(bsz, seq_len, -1)
        return out

    # ----------------------------------------- SDLM fixed-shape (CUDA Graph)
    def forward_sdlm_graph(
        self,
        x: torch.Tensor,         # [block_size, dim]
        positions: torch.Tensor, # [block_size]
        cache_k: torch.Tensor,   # [1, n_kv, max_kv, head_dim]
        cache_v: torch.Tensor,
        attn_mask: torch.Tensor, # [1, 1, block_size, max_kv + block_size]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = x.shape[0]
        xq = self.q_proj(x).view(seq_len, self.n_local_heads, self.head_dim)
        xk = self.k_proj(x).view(seq_len, self.n_local_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = self._apply_qk_norm(xq, xk)

        xq = xq.permute(1, 0, 2).unsqueeze(0)
        xk = xk.permute(1, 0, 2).unsqueeze(0)
        xv = xv.permute(1, 0, 2).unsqueeze(0)

        xq, xk = apply_rotary_pos_emb(xq, xk, positions, self.inv_freq, self.rope_mscale)

        k = torch.cat([cache_k, xk], dim=2)
        v = torch.cat([cache_v, xv], dim=2)
        kv_len = k.shape[2]
        k_exp = k.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1) \
                 .reshape(1, self.n_local_heads, kv_len, self.head_dim).contiguous()
        v_exp = v.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1) \
                 .reshape(1, self.n_local_heads, kv_len, self.head_dim).contiguous()

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(xq, k_exp.transpose(2, 3)) * scale + attn_mask
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(xq.dtype)
        out = torch.matmul(attn, v_exp)
        out = out.squeeze(0).permute(1, 0, 2).contiguous().view(seq_len, self.q_dim)
        out = self.o_proj(out)
        return out, xk, xv

    # ----------------------------- BF16 fast prefill (merged QKV / merged o)
    def prepare_bf16_fast(self) -> None:
        """Pre-concat QKV weights for a single-matmul prefill projection.
        Assumes weights are already on the Q1 grid (frozen BF16)."""
        self._bf16_wqkv = torch.cat(
            [self.q_proj.weight.data, self.k_proj.weight.data, self.v_proj.weight.data],
            dim=0,
        )
        self._bf16_qkv_split = [self.q_dim, self.k_dim, self.v_dim]
        self._bf16_wo = self.o_proj.weight.data

    def forward_sdlm_fast(
        self, x: torch.Tensor, positions: torch.Tensor,
        cache_k: Optional[torch.Tensor], cache_v: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor], is_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = x.shape[0]
        x_q = Q1Linear.act_quant(x)
        xqkv = F.linear(x_q, self._bf16_wqkv)
        xq, xk, xv = xqkv.split(self._bf16_qkv_split, dim=-1)

        xq = xq.view(seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self._apply_qk_norm(xq, xk)

        xq = xq.permute(1, 0, 2).unsqueeze(0)
        xk = xk.permute(1, 0, 2).unsqueeze(0)
        xv = xv.permute(1, 0, 2).unsqueeze(0)

        xq, xk = apply_rotary_pos_emb(xq, xk, positions, self.inv_freq, self.rope_mscale)

        if cache_k is not None and cache_k.shape[2] > 0:
            k = torch.cat([cache_k, xk], dim=2)
            v = torch.cat([cache_v, xv], dim=2)
        else:
            k, v = xk, xv
        kv_len = k.shape[2]
        k_exp = k.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1) \
                 .reshape(1, self.n_local_heads, kv_len, self.head_dim).contiguous()
        v_exp = v.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1) \
                 .reshape(1, self.n_local_heads, kv_len, self.head_dim).contiguous()

        if is_causal:
            out = F.scaled_dot_product_attention(xq, k_exp, v_exp, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(xq, k_exp, v_exp, attn_mask=attn_mask)
        out = out.squeeze(0).permute(1, 0, 2).contiguous().view(seq_len, self.q_dim)
        out_q = Q1Linear.act_quant(out)
        out = F.linear(out_q, self._bf16_wo)
        return out, xk, xv

    # --------------------- Kernel-fast decode (merged QKV + per-group scale)
    def prepare_merged_weights(self, max_positions: int = 256) -> None:
        """Concatenate packed Q/K/V weights and their group-128 scales by output row."""
        device = self.q_proj.weight.device

        if not isinstance(self.q_proj, (Q1LinearKernel, T2LinearKernel)):
            return

        self.wqkv_weight = torch.cat(
            [self.q_proj.weight.data, self.k_proj.weight.data, self.v_proj.weight.data],
            dim=0,
        )
        self.wqkv_scale = torch.cat(
            [self.q_proj.weight_scale.data,
             self.k_proj.weight_scale.data,
             self.v_proj.weight_scale.data],
            dim=0,
        ).contiguous()
        self._qkv_split = [self.q_dim, self.k_dim, self.v_dim]

        # RoPE cos/sin lookup table.
        pos = torch.arange(max_positions, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.float())
        emb = torch.cat((freqs, freqs), dim=-1)
        self.rope_cos = (emb.cos() * self.rope_mscale).to(torch.bfloat16)
        self.rope_sin = (emb.sin() * self.rope_mscale).to(torch.bfloat16)
        self._attn_scale = 1.0 / math.sqrt(self.head_dim)

    def forward_sdlm_graph_kernel(
        self,
        input_q_s: Tuple[torch.Tensor, torch.Tensor],  # int8 + scale (post-norm)
        positions: torch.Tensor,        # [M]
        kv_full_k: torch.Tensor,        # [1, n_kv, max_kv+M, head_dim] (in-place write)
        kv_full_v: torch.Tensor,        # [1, n_kv, max_kv+M, head_dim]
        attn_mask: torch.Tensor,        # [1, 1, M, max_kv+M]
    ) -> torch.Tensor:
        """CUDA-Graph-safe Q1 kernel SDLM forward.

        Writes the new KV slot in-place at the fixed offset `max_kv_len`
        (= kv_full_k.shape[2] - M). The graph reads the full `[max_kv+M]`
        buffer for attention; the mask is responsible for masking out
        positions beyond `cache_len` (CPU-side fill before replay).
        """
        input_q, s = input_q_s
        seq_len = input_q.shape[0]
        max_kv_len = kv_full_k.shape[2] - seq_len

        xqkv = packed_int8_linear(input_q, self.wqkv_weight, s, self.wqkv_scale)

        xq = _fused_qkv_qknorm_rope_kv_write(
            xqkv,
            self.q_norm.weight, self.k_norm.weight, self.q_norm.eps,
            self.rope_cos, self.rope_sin, positions,
            kv_full_k, kv_full_v,
            self.q_dim, self.k_dim,
            self.n_local_heads, self.n_local_kv_heads, self.head_dim,
            max_kv_len,
        )

        # Attention: prefer flex_attention (native GQA, no K/V expand+reshape)
        # when an atlas-compiled fn is wired in; fallback to manual matmul.
        # The manual path expands KV heads to query-head count. The compiled
        # GQA path reads kv_full_k/v directly with n_kv_heads.
        if self._flex_attn_fn is not None:
            out = self._flex_attn_fn(xq, kv_full_k, kv_full_v)
        else:
            kv_len = kv_full_k.shape[2]
            k_exp = kv_full_k.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1) \
                     .reshape(1, self.n_local_heads, kv_len, self.head_dim)
            v_exp = kv_full_v.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1) \
                     .reshape(1, self.n_local_heads, kv_len, self.head_dim)
            attn = torch.matmul(xq, k_exp.transpose(2, 3)) * self._attn_scale + attn_mask
            attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(xq.dtype)
            out = torch.matmul(attn, v_exp)
        out = out.squeeze(0).permute(1, 0, 2).contiguous().view(seq_len, self.q_dim)
        out_q, out_s = self.o_proj.quant_input(out)
        out = packed_int8_linear(
            out_q, self.o_proj.weight, out_s, self.o_proj.weight_scale,
        )
        return out


# ===========================================================================
# Bonsai (Qwen3) FeedForward — SiLU(gate) * up -> down, no sub-norm
# ===========================================================================
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        Linear = (T2LinearKernel if args.kernel_kind == "t2g128" else Q1LinearKernel) \
            if args.use_kernel else Q1Linear
        self.gate_proj = Linear(args.dim, args.ffn_dim, bias=False)
        self.up_proj = Linear(args.dim, args.ffn_dim, bias=False)
        self.down_proj = Linear(args.ffn_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    # ----- BF16 fast: merge gate+up -----
    def prepare_bf16_fast(self) -> None:
        self._bf16_gu = torch.cat([self.gate_proj.weight.data, self.up_proj.weight.data], dim=0)
        self._bf16_gu_split = [self.gate_proj.out_features, self.up_proj.out_features]
        self._bf16_down = self.down_proj.weight.data

    def forward_fast(self, x: torch.Tensor) -> torch.Tensor:
        x_q = Q1Linear.act_quant(x)
        gu = F.linear(x_q, self._bf16_gu)
        g, u = gu.split(self._bf16_gu_split, dim=-1)
        inner = F.silu(g) * u
        inner_q = Q1Linear.act_quant(inner)
        return F.linear(inner_q, self._bf16_down)

    # ----- Q1 kernel: merge gate+up (rows concat -> rows concat scale) -----
    def prepare_merged_weights(self) -> None:
        if not isinstance(self.gate_proj, (Q1LinearKernel, T2LinearKernel)):
            return
        self.w_gu = torch.cat(
            [self.gate_proj.weight.data, self.up_proj.weight.data], dim=0,
        )
        self.w_gu_scale = torch.cat(
            [self.gate_proj.weight_scale.data, self.up_proj.weight_scale.data], dim=0,
        ).contiguous()
        self._gu_split = [self.gate_proj.out_features, self.up_proj.out_features]

    def forward_merged_kernel(
        self,
        input_q_s: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        input_q, s = input_q_s
        gu = packed_int8_linear(input_q, self.w_gu, s, self.w_gu_scale)
        g, u = gu.split(self._gu_split, dim=-1)
        inner = F.silu(g) * u
        inner_q, inner_s = self.down_proj.quant_input(inner)
        return packed_int8_linear(
            inner_q, self.down_proj.weight, inner_s, self.down_proj.weight_scale,
        )


# ===========================================================================
# Transformer block
# ===========================================================================
class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)

    # ---- variable-shape SDLM (BF16 prefill or correctness check) ----
    def forward_sdlm(self, x, positions, cache_k, cache_v, attn_mask,
                     is_causal=False):
        attn_out, k, v = self.attention.forward_sdlm(
            self.input_layernorm(x), positions, cache_k, cache_v, attn_mask,
            is_causal=is_causal,
        )
        h = x + attn_out
        h = h + self.feed_forward(self.post_attention_layernorm(h))
        return h, k, v

    def forward_sdlm_fast(self, x, positions, cache_k, cache_v, attn_mask,
                          is_causal=False):
        attn_out, k, v = self.attention.forward_sdlm_fast(
            self.input_layernorm(x), positions, cache_k, cache_v, attn_mask,
            is_causal=is_causal,
        )
        h = x + attn_out
        h = h + self.feed_forward.forward_fast(self.post_attention_layernorm(h))
        return h, k, v

    # ---- batched verify ----
    def forward_sdlm_batch(self, x, positions, attn_mask):
        attn_out = self.attention.forward_sdlm_batch(
            self.input_layernorm(x), positions, attn_mask,
        )
        h = x + attn_out
        bsz, slen, dim = h.shape
        ffn_in = self.post_attention_layernorm(h).view(-1, dim)
        ffn_out = self.feed_forward(ffn_in).view(bsz, slen, dim)
        return h + ffn_out

    # ---- fixed-shape CUDA Graph ----
    def forward_sdlm_graph(self, x, positions, cache_k, cache_v, attn_mask):
        attn_out, k, v = self.attention.forward_sdlm_graph(
            self.input_layernorm(x), positions, cache_k, cache_v, attn_mask,
        )
        h = x + attn_out
        h = h + self.feed_forward(self.post_attention_layernorm(h))
        return h, k, v

    def forward_sdlm_graph_kernel(self, attn_q_s, residual, positions,
                                  kv_full_k, kv_full_v, attn_mask):
        """Fully-kernelised graph step.

        Cross-layer fused: receives pre-normed+quantised input
        `attn_q_s = (q, s)`, returns `(h, ffn_out)` so the next layer can
        fuse its `residual_add + attention_norm + quant` with this layer's
        FFN output. The residual add is deferred to the caller.
        """
        attn_out = self.attention.forward_sdlm_graph_kernel(
            attn_q_s, positions, kv_full_k, kv_full_v, attn_mask,
        )
        h, ffn_q, ffn_s = _fused_add_norm_quant(
            residual, attn_out,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )
        ffn_out = self.feed_forward.forward_merged_kernel((ffn_q, ffn_s))
        return h, ffn_out


# ===========================================================================
# Transformer (full model)
# ===========================================================================
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        if args.tie_word_embeddings:
            # Logits projection shares the embedding weight (Qwen3-1.7B layout).
            self._lm_head_weight_ref = None  # populated in `tie_weights`
        else:
            self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)

    def tie_weights(self) -> None:
        if self.args.tie_word_embeddings:
            self._lm_head_weight_ref = self.embed_tokens.weight

    def _lm_head_weight(self) -> torch.Tensor:
        if self.args.tie_word_embeddings:
            w = self._lm_head_weight_ref
            if w is None:
                w = self.embed_tokens.weight
            return w
        return self.lm_head.weight

    @torch.no_grad()
    def prepare_fp8_lm_head(self) -> None:
        """Pre-quantise the (tied) lm_head weight to FP8 e4m3 for the
        acceptance-tolerant DRAFT output projection (BitMTP M1, role-width
        specialization). Bandwidth-halving on the 1-bit backbone's single
        largest weight (BF16 lm_head = 75.8% of decode weight traffic).

        Per-tensor weight scale (matches BitMTP `prepare_fp8_output`).
        Vocab is padded up to a multiple of 16 (cuBLAS FP8 N constraint); the
        padding logits are sliced off before returning, so they never affect
        argmax. Verify/AR paths keep BF16 -> verification stays exact."""
        FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
        W = self._lm_head_weight().data            # [vocab, dim] BF16
        vocab, dim = W.shape
        vocab_pad = (vocab + 15) // 16 * 16
        if vocab_pad != vocab:
            Wp = torch.zeros(vocab_pad, dim, dtype=W.dtype, device=W.device)
            Wp[:vocab].copy_(W)
        else:
            Wp = W
        w_scale = (Wp.abs().amax() / FP8_MAX).clamp_min(1e-12).to(torch.float32)
        self._lm_head_fp8_w = (Wp / w_scale).to(torch.float8_e4m3fn)  # [vocab_pad, dim]
        self._lm_head_w_scale = w_scale
        self._lm_head_fp8_max = FP8_MAX
        self._lm_head_vocab = vocab
        print(f"[FP8] lm_head prepared: shape={tuple(self._lm_head_fp8_w.shape)} "
              f"(vocab {vocab} -> pad {vocab_pad}), w_scale={w_scale.item():.6g}")

    @torch.no_grad()
    def prepare_q1_lm_head(self) -> None:
        """Pack the (tied) lm_head weight into Q1_0_g128 (1-bit) for a TRUE
        1-bit W1A8 output projection. The tied weight is already frozen onto
        the Q1 grid at convert time, and training ran lm_head as W1A8, so this
        REPRODUCES the training forward (in-distribution) -- suitable for the
        exact VERIFY path, unlike FP8 (post-training quant). Vocab is padded
        to N=151936 (the kernel's instantiated lm_head N); pad logits sliced."""
        W = self._lm_head_weight().data.to(torch.bfloat16)   # [vocab, dim], on grid
        vocab, dim = W.shape
        N_PAD = 151936
        Wp = torch.zeros(N_PAD, dim, dtype=torch.bfloat16, device=W.device)
        Wp[:vocab].copy_(W)
        # Pack: per-group scale d_g + sign bits (LSB=first weight), pos->1.
        n_groups = dim // 128
        wg = Wp.float().view(N_PAD, n_groups, 128)
        d_g = wg.abs().mean(dim=-1).to(torch.bfloat16)               # [N_PAD, dim/128]
        sign = ((Wp > 0) | (Wp == 0)).to(torch.uint8).view(N_PAD, dim // 8, 8)
        bit_pos = torch.arange(8, dtype=torch.uint8, device=W.device)
        packed = (sign << bit_pos.view(1, 1, 8)).sum(dim=-1).to(torch.uint8)
        self._lm_head_q1_w = packed.view(torch.int8).contiguous()    # [N_PAD, dim/8]
        self._lm_head_q1_scale = d_g.contiguous()                    # [N_PAD, dim/128]
        self._lm_head_q1_vocab = vocab
        print(f"[Q1] lm_head packed: w={tuple(self._lm_head_q1_w.shape)} "
              f"scale={tuple(self._lm_head_q1_scale.shape)} (vocab {vocab} -> pad {N_PAD})")

    @torch.no_grad()
    def prepare_t2_lm_head(self) -> None:
        """Pack the tied ternary lm_head on its exact Q2_0_g128 grid."""
        W = self._lm_head_weight().data.to(torch.bfloat16)
        vocab, dim = W.shape
        N_PAD = 151936
        Wp = torch.zeros(N_PAD, dim, dtype=torch.bfloat16, device=W.device)
        Wp[:vocab].copy_(W)
        groups = Wp.float().view(N_PAD, dim // 128, 128)
        scale = groups.abs().amax(dim=-1).to(torch.bfloat16)
        denom = scale.float().unsqueeze(-1).clamp_min(1e-30)
        q = torch.round(groups / denom).clamp(-1, 1).to(torch.int8)
        code = (q.view(N_PAD, dim) + 2).to(torch.uint8)
        code = code.view(N_PAD, dim // 4, 4)
        shifts = (torch.arange(4, device=W.device, dtype=torch.uint8) * 2)
        packed = (code << shifts.view(1, 1, 4)).sum(
            dim=-1, dtype=torch.uint8)
        self._lm_head_t2_w = packed.view(torch.int8).contiguous()
        self._lm_head_t2_scale = scale.contiguous()
        self._lm_head_t2_vocab = vocab
        print(f"[T2] lm_head packed: w={tuple(self._lm_head_t2_w.shape)} "
              f"scale={tuple(scale.shape)} (vocab {vocab} -> pad {N_PAD})")

    def _lm_head(self, h: torch.Tensor, precision: str = "bf16") -> torch.Tensor:
        if precision == "t2":
            x = h.float()
            s = 127.0 / x.abs().amax(-1, keepdim=True).clamp_min(1e-5)
            x_q = (x * s).round().clamp(-128, 127).to(torch.int8)
            logits = t2g128_int8xint2_linear(
                x_q, self._lm_head_t2_w, s.to(torch.float32),
                self._lm_head_t2_scale,
            )
            return logits[:, : self._lm_head_t2_vocab]
        if precision == "w1a16":
            logits = q1g128_w1a16_linear(
                h.to(torch.bfloat16), self._lm_head_q1_w, self._lm_head_q1_scale)
            return logits[:, : self._lm_head_q1_vocab]
        if precision == "q1":
            x = h.float()
            s = (127.0 / x.abs().amax(-1, keepdim=True).clamp_min(1e-5))
            x_q = (x * s).round().clamp(-128, 127).to(torch.int8)
            logits = q1g128_int8xint1_linear(
                x_q, self._lm_head_q1_w, s.to(torch.float32), self._lm_head_q1_scale)
            return logits[:, : self._lm_head_q1_vocab]
        if precision == "fp8":
            fp8_max = self._lm_head_fp8_max
            h_scale = (h.abs().amax() / fp8_max).clamp_min(1e-12).to(torch.float32)
            h_fp8 = (h / h_scale).to(torch.float8_e4m3fn)
            logits = torch._scaled_mm(
                h_fp8, self._lm_head_fp8_w.T,
                scale_a=h_scale, scale_b=self._lm_head_w_scale,
                out_dtype=torch.bfloat16,
            )
            return logits[:, : self._lm_head_vocab]
        if precision == "bf16_raw":
            # Explicit RTX 3090 fallback for the acceptance-tolerant draft
            # projection.  This must precede _force_a8_lm_head: the ordinary
            # "bf16" path for a ternary checkpoint intentionally replays the
            # training-time A8 fake quantization and therefore is not a raw
            # BF16 projection.
            return F.linear(h.to(torch.bfloat16), self._lm_head_weight())
        if precision != "bf16":
            raise RuntimeError(
                f"unsupported lm_head precision: {precision!r}")
        if getattr(self, "_force_a8_lm_head", False):
            # Ternary mode: the tied lm_head was trained as TernaryLinear
            # (W ternary-g128 x A8), so fake-quant the input per-token INT8
            # before the BF16 matmul. The weight is already frozen on the
            # ternary grid by the converter, making this numerically the
            # training-time forward.
            x = h.float()
            s = 127.0 / x.abs().amax(-1, keepdim=True).clamp_min(1e-5)
            x_q = ((x * s).round().clamp(-128, 127) / s).to(h.dtype)
            return F.linear(x_q, self._lm_head_weight())
        return F.linear(h, self._lm_head_weight())

    @torch.no_grad()
    def forward_sdlm_verify_batch(self, token_values, positions, attn_mask):
        h = self.embed_tokens(token_values)
        for layer in self.layers:
            h = layer.forward_sdlm_batch(h, positions, attn_mask)
        return self._lm_head(self.norm(h)).float()

    @torch.no_grad()
    def forward_sdlm_prefill(
        self,
        token_values: torch.Tensor,
        cache: List[SDLMLayerCache],
        positions: Optional[torch.Tensor] = None,
        block_size: int = 1,
        fast: bool = False,
        causal_attn: bool = True,
    ) -> torch.Tensor:
        seq_len = token_values.shape[0]
        if positions is None:
            positions = torch.arange(seq_len, device=token_values.device)
        h = self.embed_tokens(token_values)

        past_len = cache[0][0].shape[2] if cache and len(cache) > 0 else 0
        if block_size <= 1:
            # AR / standard causal: SDPA `is_causal=True` is exact.
            attn_mask = None
            is_causal = True
        else:
            # SDLM mask (need anchor_prev masking even when causal_attn=True).
            attn_mask = create_sdlm_attention_mask(
                q_len=seq_len, kv_len=past_len + seq_len, block_size=block_size,
                device=h.device, dtype=h.dtype, past_len=past_len,
                causal_attn=causal_attn,
            )
            is_causal = False
        fwd = "forward_sdlm_fast" if fast else "forward_sdlm"
        for i, layer in enumerate(self.layers):
            h, k, v = getattr(layer, fwd)(
                h, positions, cache[i][0], cache[i][1], attn_mask,
                is_causal=is_causal,
            )
            cache[i] = (k, v)
        return self._lm_head(self.norm(h)).float()

    @torch.no_grad()
    def forward_sdlm_decode_graph(
        self,
        token_values: torch.Tensor,    # [block_size]
        positions: torch.Tensor,       # [block_size]
        cache: List[SDLMLayerCache],
        block_size: int,
        attn_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        h = self.embed_tokens(token_values)
        new_ks, new_vs = [], []
        for i, layer in enumerate(self.layers):
            h, k, v = layer.forward_sdlm_graph(
                h, positions, cache[i][0], cache[i][1], attn_mask,
            )
            new_ks.append(k)
            new_vs.append(v)
        logits = self._lm_head(self.norm(h)).float()
        return logits, new_ks, new_vs

    @torch.no_grad()
    def forward_sdlm_decode_graph_kernel(
        self,
        token_values: torch.Tensor,      # [M]
        positions: torch.Tensor,         # [M]
        kv_full: List[SDLMLayerCache],   # per-layer (k_buf, v_buf), shape [1, n_kv, max_kv+M, head_dim]
        block_size: int,                 # M (decode width)
        attn_mask: torch.Tensor,         # [1, 1, M, max_kv+M]
        logits_start: int = 0,
        logits_end: int = -1,
        lm_head_precision: str = "bf16",
    ) -> torch.Tensor:
        """Fully kernelised CUDA-Graph forward (M1+M2).

        Returns logits[logits_start:logits_end]. KV is written in-place to
        `kv_full` at the trailing slot [max_kv:max_kv+M]; the caller
        (fast_runtime) is responsible for committing accepted rows back to
        the history region before the next replay.

        `lm_head_precision='fp8'` routes the output projection through the
        pre-quantised FP8 draft weight (acceptance-tolerant role).
        """
        if logits_end == -1:
            logits_end = block_size
        h = self.embed_tokens(token_values)
        attn_q_s = self.layers[0].input_layernorm.forward_and_quant(h)
        ffn_out = None
        for i, layer in enumerate(self.layers):
            if i > 0:
                h, q, s = _fused_add_norm_quant(
                    ffn_out, h,
                    layer.input_layernorm.weight,
                    layer.input_layernorm.eps,
                )
                attn_q_s = (q, s)
            h, ffn_out = layer.forward_sdlm_graph_kernel(
                attn_q_s, h, positions,
                kv_full[i][0], kv_full[i][1], attn_mask,
            )
        h = h + ffn_out
        h_normed = self.norm(h[logits_start:logits_end])
        logits = self._lm_head(h_normed, precision=lm_head_precision).float()
        return logits

    @torch.no_grad()
    def forward_sdlm_decode(
        self, token_values, positions, cache, block_size,
        cache_len: Optional[int] = None, causal_attn: bool = True,
    ):
        h = self.embed_tokens(token_values)
        past_len = cache_len if cache_len is not None else cache[0][0].shape[2]
        kv_len = past_len + block_size
        if block_size <= 1:
            # AR step with KV cache: q_len=1, kv_len=past_len+1. Standard
            # causal mask on this geometry would isolate q[0] to k[0] only
            # (SDPA "upper-left" semantics), so we use attn_mask=None +
            # is_causal=False (= full visibility over already-causal past KV).
            attn_mask = None
            is_causal_flag = False
        else:
            attn_mask = create_sdlm_attention_mask(
                q_len=block_size, kv_len=kv_len, block_size=block_size,
                device=h.device, dtype=h.dtype, past_len=past_len,
                causal_attn=causal_attn,
            )
            is_causal_flag = False
        new_ks, new_vs = [], []
        for i, layer in enumerate(self.layers):
            ck = cache[i][0][:, :, :past_len, :] if cache_len is not None else cache[i][0]
            cv = cache[i][1][:, :, :past_len, :] if cache_len is not None else cache[i][1]
            h, k, v = layer.forward_sdlm(h, positions, ck, cv, attn_mask,
                                         is_causal=is_causal_flag)
            new_ks.append(k)
            new_vs.append(v)
        logits = self._lm_head(self.norm(h)).float()
        return logits, new_ks, new_vs

    def prepare_bf16_fast(self) -> None:
        for layer in self.layers:
            layer.attention.prepare_bf16_fast()
            layer.feed_forward.prepare_bf16_fast()

    def prepare_merged_weights(self, max_positions: int = 256) -> None:
        for layer in self.layers:
            layer.attention.prepare_merged_weights(max_positions)
            layer.feed_forward.prepare_merged_weights()


# ===========================================================================
# Cache helpers
# ===========================================================================
def make_sdlm_cache(
    args: ModelArgs,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> List[SDLMLayerCache]:
    """Empty (length-0) SDLM KV cache for the prefill path. After prefill
    the cache is replaced by the (variable-length) freshly written K/V.
    """
    n_kv = args.n_kv_heads
    hd = args.head_dim
    return [
        (
            torch.zeros(1, n_kv, 0, hd, dtype=dtype, device=device),
            torch.zeros(1, n_kv, 0, hd, dtype=dtype, device=device),
        )
        for _ in range(args.n_layers)
    ]


def make_sdlm_graph_cache(
    args: ModelArgs,
    max_length: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> List[SDLMLayerCache]:
    """Pre-allocate fixed-shape KV cache buffers for the graph decode path.

    Length is `max_length` (zero-padded). The mask passed into
    `forward_sdlm_decode_graph` is responsible for blocking out positions
    beyond the actual cache_len.
    """
    n_kv = args.n_kv_heads
    hd = args.head_dim
    return [
        (
            torch.zeros(1, n_kv, max_length, hd, dtype=dtype, device=device),
            torch.zeros(1, n_kv, max_length, hd, dtype=dtype, device=device),
        )
        for _ in range(args.n_layers)
    ]


def update_sdlm_cache(
    cache: List[SDLMLayerCache],
    new_ks: List[torch.Tensor],
    new_vs: List[torch.Tensor],
    write_pos: int,
) -> None:
    """Write `new_ks/vs` (each [1, n_kv, seg_len, head_dim]) into `cache` at
    [:, :, write_pos:write_pos+seg_len, :]. Used by FastGen after a graph step."""
    seg_len = new_ks[0].shape[2]
    for i, (ck, cv) in enumerate(cache):
        ck[:, :, write_pos : write_pos + seg_len, :] = new_ks[i]
        cv[:, :, write_pos : write_pos + seg_len, :] = new_vs[i]
