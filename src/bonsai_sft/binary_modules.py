"""Group-128 quantization helpers and shared activation quantization.

Quantization formula (matches `ggml/src/ggml-quants.c::quantize_row_q1_0_g128_ref`
in the PrismML llama.cpp fork):

    For every group of GROUP_SIZE=128 consecutive weights along the input dim:
        d  = mean(|w_group|)                        # FP16 scale
        wq = sign(w) * d                            # ±d per weight, 1 sign bit
    Backward: Straight-Through Estimator (gradient flows through unchanged).

Activation quantization (per-token symmetric INT8, matches HF AutoBitLinear):
    scale = 127 / max(|x|, dim=-1)
    xq    = clamp(round(x * scale), -128, 127) / scale
    Backward: STE.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


GROUP_SIZE = 128  # Q1_0_g128 group size; matches block_q1_0_g128 in llama.cpp


# ---------------------------------------------------------------------------
# STE primitives
# ---------------------------------------------------------------------------
class BinaryWeightQuant(torch.autograd.Function):
    """Group-wise binary weight quant: w_q = sign(w) * mean(|w|, group=128).

    Input weight is laid out as [out_features, in_features]; groups are formed
    along the LAST axis (in_features), which is the contiguous dim that
    llama.cpp's Q1_0_g128 packs along. The input feature dimension must be
    divisible by the group size, 128.
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = weight.dtype
        w = weight.float()
        out_f, in_f = w.shape
        assert in_f % GROUP_SIZE == 0, (
            f"BinaryWeightQuant: in_features={in_f} not divisible by "
            f"GROUP_SIZE={GROUP_SIZE}"
        )
        wg = w.view(out_f, in_f // GROUP_SIZE, GROUP_SIZE)
        d = wg.abs().mean(dim=-1, keepdim=True).clamp_min(1e-5)
        wq = torch.sign(wg) * d
        # sign(0) == 0; replace zero with +d to match ggml's `x >= 0.0f` rule
        zero_mask = wq == 0
        if zero_mask.any():
            wq = torch.where(zero_mask, d.expand_as(wq), wq)
        return wq.view(out_f, in_f).to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Straight-through: pass gradient unchanged.
        return grad_output.clone()


class ActQuant(torch.autograd.Function):
    """Per-token symmetric INT8 activation quant (STE backward).

    Matches `transformers.integrations.bitnet.ActQuant`.
    """

    @staticmethod
    def forward(ctx, activation: torch.Tensor) -> torch.Tensor:
        orig_dtype = activation.dtype
        a = activation.float()
        scale = 127.0 / a.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
        aq = (a * scale).round().clamp(-128, 127) / scale
        return aq.to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.clone()


# ---------------------------------------------------------------------------
# Quantized modules
# ---------------------------------------------------------------------------
class BinaryLinear(nn.Linear):
    """nn.Linear with Q1_0_g128 weight QAT + per-token INT8 activation QAT."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        w_q = BinaryWeightQuant.apply(self.weight)
        x_q = ActQuant.apply(x)
        return F.linear(x_q, w_q, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, qat=Q1_0_g128+INT8act"
        )


class BinaryEmbedding(nn.Embedding):
    """nn.Embedding with Q1_0_g128 QAT on the weight table.

    Embedding lookup is a table read, so there is no activation to quantize:
    the output rows are exactly the (quantized) weight rows. STE flows back
    to the BF16 latent rows that were actually looked up (sparse update).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
            device=device,
            dtype=dtype,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        w_q = BinaryWeightQuant.apply(self.weight)
        return F.embedding(
            input_ids,
            w_q,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"qat=Q1_0_g128"
        )


@torch.no_grad()
def freeze_quantize_to_q1_0_g128(weight: torch.Tensor) -> torch.Tensor:
    """One-shot deterministic round: latent BF16 -> ±d_g BF16 (no STE).

    Used at export time so that downstream `llama-quantize ... Q1_0_g128`
    sees values already on the {+d_g, -d_g} grid -> lossless conversion to
    GGUF (matching `quantize_row_q1_0_g128_ref` byte-for-byte after scale).
    """
    orig_dtype = weight.dtype
    w = weight.detach().float()
    out_f, in_f = w.shape
    assert in_f % GROUP_SIZE == 0
    wg = w.view(out_f, in_f // GROUP_SIZE, GROUP_SIZE)
    d = wg.abs().mean(dim=-1, keepdim=True).clamp_min(1e-5)
    wq = torch.sign(wg) * d
    zero_mask = wq == 0
    if zero_mask.any():
        wq = torch.where(zero_mask, d.expand_as(wq), wq)
    return wq.view(out_f, in_f).to(orig_dtype)
