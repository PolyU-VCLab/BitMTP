"""CPU-only dispatch and decoding contracts for the packaged CUDA sources.

The native functions print and return on an unsupported shape. Callers must
validate before crossing that C ABI, otherwise their output is uninitialized.
"""
from __future__ import annotations

import math


Q1_M = frozenset((1, 4, 5, 6, 7, 8, 9, 10, 12, 16, 17))
BONSAI_17B_NK = frozenset((
    (1024, 2048), (2048, 2048), (4096, 2048),
    (6144, 2048), (12288, 2048), (2048, 6144), (151936, 2048),
))
T2_4B_M = frozenset((1, 8, 9, 10, 12, 16, 17))
T2_4B_NK = frozenset((
    (1024, 2560), (6144, 2560), (19456, 2560),
    (2560, 4096), (2560, 9728), (151936, 2560),
))


def validate_public_model_scope(config: dict, *, hf: bool = False,
                                quant: str = "ternary",
                                quantize_embedding: bool = True) -> None:
    """Public CLI scope only; generic conversion/kernel helpers stay separate.

    Geometry and quantization-format checks do not authenticate checkpoint
    identity or establish that its weights were trained with a given recipe.
    An SFT HF configuration without block_size uses the converter's D8 default.
    """
    if not isinstance(config, dict):
        raise ValueError('public release requires a JSON object model configuration')
    names = (('hidden_size', 'num_hidden_layers', 'num_attention_heads',
              'num_key_value_heads', 'head_dim', 'intermediate_size', 'vocab_size')
             if hf else ('dim', 'n_layers', 'n_heads', 'n_kv_heads', 'head_dim',
                         'ffn_dim', 'vocab_size'))
    supported = {
        (2048, 28, 16, 8, 128, 6144, 151669): (4, 8),
        (2560, 36, 32, 8, 128, 9728, 151669): (8,),
    }
    geometry = tuple(config.get(name) for name in names)
    valid_geometry = all(type(value) is int for value in geometry) and geometry in supported
    mismatch = [] if valid_geometry else ['model_geometry']
    block_size = config.get('block_size', 8 if hf else None)
    supported_blocks = supported[geometry] if valid_geometry else ()
    if type(block_size) is not int or block_size not in supported_blocks:
        mismatch.append('block_size')
    if config.get('tie_word_embeddings') is not True:
        mismatch.append('tie_word_embeddings')
    if hf:
        if quant != 'ternary' or not quantize_embedding:
            mismatch.append('quantization')
    # Original T2 artifacts predate this optional metadata field. Absence
    # retains their historical frozen-embedding interpretation.
    elif config.get('kernel_kind') != 't2g128' or config.get('quantize_embedding', True) is not True:
        mismatch.append('quantization')
    if mismatch:
        raise ValueError(
            'public release supports only Bonsai 1.7B T2 D4/D8 or Bonsai 4B T2 D8 '
            '(including D8 SFT exports); unsupported configuration fields: '
            + ', '.join(mismatch))


def validate_public_device_scope(config: dict, capability) -> None:
    """4B generation requires SM90; NVIDIA H20 is the validated platform."""
    if config.get('dim') == 2560 and tuple(capability) != (9, 0):
        raise ValueError('public Bonsai 4B T2 D8 generation requires SM90; '
                         'NVIDIA H20 is the validated platform, not RTX 3090')


def supports_kernel_shape(kind: str, m: int, n: int, k: int) -> bool:
    """Match the complete (M,N,K) registration, not independent dimensions."""
    if kind == "q1":
        return m in Q1_M and (n, k) in BONSAI_17B_NK
    if kind == "w1a16":
        return m in Q1_M and (n, k) == (151936, 2048)
    if kind == "t2g128":
        return ((m in Q1_M and (n, k) in BONSAI_17B_NK)
                or (m in T2_4B_M and (n, k) in T2_4B_NK))
    raise ValueError(f"unknown kernel kind: {kind!r}")


def require_kernel_shape(kind: str, m: int, n: int, k: int) -> None:
    if not supports_kernel_shape(kind, m, n, k):
        raise RuntimeError(f"unsupported {kind} kernel shape: M={m}, N={n}, K={k}")


def supports_layer_kernel_shape(kind: str, m: int, n: int, k: int) -> bool:
    """Keep the 4B variable-width layer/prefill reconstruction fallback.

    Fixed-width graph forwards call the public packed dispatcher directly and
    can use every registered 4B D8 shape. The layer forward intentionally only
    uses the 4B native kernel for M=1, matching its source execution path.
    """
    if kind == "t2g128" and (n, k) in T2_4B_NK:
        return m == 1
    return supports_kernel_shape(kind, m, n, k)


def validate_verification_options(rabv_margin: float, exact_m1_verify: bool = False) -> None:
    """Compatibility parameter: strict verification accepts exactly zero only."""
    try:
        valid = math.isfinite(rabv_margin) and rabv_margin == 0.0
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("strict verifier requires rabv_margin=0; nonzero margins are unsupported")


def validate_native_verifier(use_fp8=False, use_q1=False, use_w1a16=False) -> None:
    """Verifier projection must match the backbone's native AR projection."""
    if use_fp8 or use_q1 or use_w1a16:
        raise ValueError("strict verification requires the native verifier projection")


def validate_graph_roles(roles, capability=None) -> None:
    """Keep device-specific draft approximation separate from native verification."""
    allowed = {"ar", "draft", "draft_fp8", "draft_bf16", "verify"}
    unknown = set(roles) - allowed
    if unknown:
        raise ValueError(f"unsupported strict graph roles: {sorted(unknown)}")
    if capability is not None and "draft_fp8" in roles and tuple(capability) < (8, 9):
        raise ValueError("FP8 draft projection is unsupported on this GPU; use draft_bf16 or native draft")
