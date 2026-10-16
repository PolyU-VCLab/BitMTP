"""Ternary quantization-aware SFT and block-prediction training for Bonsai/Qwen3.

Group-128 weight quantization and per-token INT8 activation quantization
support training and rounded export. The GPU package provides packed
inference and CUDA Graph decoding.

Modules:
    ternary_modules: group-128 ternary QAT primitives
    convert        : replace Qwen3 linear/embedding modules with QAT variants
    data           : ShareGPT/messages SFT dataset using Bonsai chat_template
    train          : HF Trainer entry point (DeepSpeed compatible)
    sdlm_train     : joint next-token and block-prediction training
    export_ternary : latent weights -> group-128 ternary-grid export
"""
from .binary_modules import ActQuant, GROUP_SIZE
from .ternary_modules import (
    TernaryWeightQuant,
    TernaryLinear,
    TernaryEmbedding,
    freeze_quantize_to_ternary_g128,
)
from .convert import convert_qwen3_to_ternary_qat, count_ternary_modules

__all__ = [
    "TernaryWeightQuant",
    "ActQuant",
    "TernaryLinear",
    "TernaryEmbedding",
    "GROUP_SIZE",
    "freeze_quantize_to_ternary_g128",
    "convert_qwen3_to_ternary_qat",
    "count_ternary_modules",
]
