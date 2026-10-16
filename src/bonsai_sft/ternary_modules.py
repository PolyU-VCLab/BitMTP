"""Ternary (1.58-bit) QAT primitives for Ternary-Bonsai (Q2_0, group=128).

For every group of 128 consecutive weights along the input dimension:

    s   = max(max(|w_group|), 1e-8)      # per-group absmax scale
    q   = round(w / s)  in {-1, 0, +1}   # |w/s| <= 1 so round is naturally ternary
    w_q = q * s                          # values land on {-s, 0, +s}

Quantization is computed in FP32 and cast back to the weight dtype. Linear
layers also apply per-token symmetric INT8 activation quantization. The
ternary conversion covers embeddings, decoder projections, and the tied
output head; straight-through gradients update the latent weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .binary_modules import ActQuant, GROUP_SIZE

def _ternary_quantize(w: torch.Tensor, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Group-wise absmax ternary quant: w_q = round(w/amax)*amax, levels {-s,0,+s}."""
    orig_dtype = w.dtype
    wf = w.float()
    out_f, in_f = wf.shape
    assert in_f % group_size == 0, (
        f"ternary quant: in_features={in_f} not divisible by group={group_size}"
    )
    wg = wf.view(out_f, in_f // group_size, group_size)
    s = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    q = torch.round(wg / s).clamp_(-1, 1)
    wq = q * s
    return wq.view(out_f, in_f).to(orig_dtype)


class TernaryWeightQuant(torch.autograd.Function):
    """Group-128 absmax ternary weight quant with straight-through gradient."""

    @staticmethod
    def forward(ctx, weight: torch.Tensor) -> torch.Tensor:
        return _ternary_quantize(weight, GROUP_SIZE)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.clone()


class TernaryLinear(nn.Linear):
    """nn.Linear with group-128 ternary weight QAT + per-token INT8 activation QAT."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        w_q = TernaryWeightQuant.apply(self.weight)
        x_q = ActQuant.apply(x)
        return F.linear(x_q, w_q, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, qat=Ternary_g128+INT8act"
        )


class TernaryEmbedding(nn.Embedding):
    """nn.Embedding with group-128 ternary QAT on the weight table (STE)."""

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        w_q = TernaryWeightQuant.apply(self.weight)
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
            f"qat=Ternary_g128"
        )


@torch.no_grad()
def freeze_quantize_to_ternary_g128(weight: torch.Tensor) -> torch.Tensor:
    """One-shot deterministic round: latent BF16 -> {-s,0,+s} BF16 (no STE).

    Same op the QAT forward applies, so the export round is lossless relative to
    the training-time effective weights and matches the released ternary grid.
    """
    return _ternary_quantize(weight.detach(), GROUP_SIZE)
