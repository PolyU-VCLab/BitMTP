"""Replace Qwen3 projection and embedding modules with group-128 ternary QAT.

Ternary conversion quantizes embeddings, decoder projections, and the output
head. Norms and rotary-embedding buffers are not quantized.

When `tie_word_embeddings=True`, the embedding and output head are re-tied
after conversion so both paths use one latent parameter.
"""
from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


_LINEAR_TARGETS = {
    # attention
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    # MLP
    "gate_proj",
    "up_proj",
    "down_proj",
    # output head
    "lm_head",
}
_EMBEDDING_TARGETS = {"embed_tokens"}


# ---------------------------------------------------------------------------
# Ternary (1.58-bit) conversion — Ternary-Bonsai (Q2_0, group=128)
# ---------------------------------------------------------------------------
def _replace_ternary_linear(parent, name, module):
    from .ternary_modules import TernaryLinear
    new_mod = TernaryLinear(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=module.bias is not None,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    with torch.no_grad():
        new_mod.weight.copy_(module.weight)
        if module.bias is not None:
            new_mod.bias.copy_(module.bias)
    setattr(parent, name, new_mod)
    return new_mod


def _replace_ternary_embedding(parent, name, module):
    from .ternary_modules import TernaryEmbedding
    new_mod = TernaryEmbedding(
        num_embeddings=module.num_embeddings,
        embedding_dim=module.embedding_dim,
        padding_idx=module.padding_idx,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    with torch.no_grad():
        new_mod.weight.copy_(module.weight)
    setattr(parent, name, new_mod)
    return new_mod


def convert_qwen3_to_ternary_qat(model: nn.Module) -> Tuple[int, int]:
    """In-place replace nn.Linear/nn.Embedding with TERNARY (1.58-bit) QAT.

    Quantize token embeddings, decoder projections, and the output head on
    the ternary group-128 grid. Tied embedding/head parameters remain shared.

    Returns (n_linear_replaced, n_embedding_replaced).
    """
    from .ternary_modules import TernaryEmbedding, TernaryLinear

    cfg = getattr(model, "config", None)
    tie = bool(getattr(cfg, "tie_word_embeddings", False)) if cfg is not None else False

    n_lin = 0
    n_emb = 0
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if child_name in _LINEAR_TARGETS and isinstance(child, nn.Linear) and \
                    not isinstance(child, TernaryLinear):
                _replace_ternary_linear(parent, child_name, child)
                n_lin += 1
            elif child_name in _EMBEDDING_TARGETS and isinstance(child, nn.Embedding) and \
                    not isinstance(child, TernaryEmbedding):
                _replace_ternary_embedding(parent, child_name, child)
                n_emb += 1

    if tie:
        embed = getattr(model.get_input_embeddings(), "weight", None)
        head = model.get_output_embeddings()
        if embed is not None and head is not None:
            head.weight = embed  # share one latent Parameter (both quantize identically)
            try:
                model.tie_weights()
            except Exception as e:  # pragma: no cover
                logger.warning("model.tie_weights() raised: %s (ignored)", e)
            logger.info("Re-tied lm_head.weight -> embed_tokens.weight (ternary, id=%d)",
                        id(embed))

    logger.info(
        "Ternary QAT conversion: replaced %d nn.Linear -> TernaryLinear and "
        "%d nn.Embedding -> TernaryEmbedding (tie_word_embeddings=%s)",
        n_lin, n_emb, tie,
    )
    return n_lin, n_emb


def count_ternary_modules(model: nn.Module) -> Tuple[int, int]:
    from .ternary_modules import TernaryEmbedding, TernaryLinear
    n_lin = sum(1 for m in model.modules() if isinstance(m, TernaryLinear))
    n_emb = sum(1 for m in model.modules() if isinstance(m, TernaryEmbedding))
    return n_lin, n_emb
