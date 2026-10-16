"""Convert Bonsai Hugging Face weights to packed GPU inference files.

The conversion helper writes model_state_q1g128.pt for binary weights or
model_state_t2g128.pt for ternary weights, with group-128 BF16 scales. It
can also write model_state_fp16.pt containing the frozen-grid BF16 weights.
The public CLI supports T2-g128 model configurations.

Equivalent stored weight grids do not imply identical floating-point
execution across inference paths. GPU numerical checks are separate from
file conversion.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, Tuple

import torch
from safetensors import safe_open

from bonsai_sft.binary_modules import freeze_quantize_to_q1_0_g128, GROUP_SIZE
from bonsai_sft.ternary_modules import freeze_quantize_to_ternary_g128
from bonsai_sft.export_ternary import validate_sdlm_config
from .checkpoint_validation import validate_hf_checkpoint
from .contracts import validate_public_model_scope


# ---------------------------------------------------------------------------
# Layer-name mapping HF -> Bonsai GPU model
# ---------------------------------------------------------------------------
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def _hf_to_gpu_key(hf_key: str) -> str:
    """Rewrite a HF Qwen3 state-dict key to the Bonsai-GPU key convention.

    HF                                                   Bonsai-GPU
    --------------------------------------------------------------
    model.embed_tokens.weight                            embed_tokens.weight
    model.layers.{i}.input_layernorm.weight              layers.{i}.input_layernorm.weight
    model.layers.{i}.post_attention_layernorm.weight     layers.{i}.post_attention_layernorm.weight
    model.layers.{i}.self_attn.q_proj.weight             layers.{i}.attention.q_proj.weight
    model.layers.{i}.self_attn.k_proj.weight             layers.{i}.attention.k_proj.weight
    model.layers.{i}.self_attn.v_proj.weight             layers.{i}.attention.v_proj.weight
    model.layers.{i}.self_attn.o_proj.weight             layers.{i}.attention.o_proj.weight
    model.layers.{i}.self_attn.q_norm.weight             layers.{i}.attention.q_norm.weight
    model.layers.{i}.self_attn.k_norm.weight             layers.{i}.attention.k_norm.weight
    model.layers.{i}.mlp.gate_proj.weight                layers.{i}.feed_forward.gate_proj.weight
    model.layers.{i}.mlp.up_proj.weight                  layers.{i}.feed_forward.up_proj.weight
    model.layers.{i}.mlp.down_proj.weight                layers.{i}.feed_forward.down_proj.weight
    model.norm.weight                                    norm.weight
    lm_head.weight                                       (tied; dropped if tie_word_embeddings)
    """
    k = hf_key
    k = k.replace("model.embed_tokens.weight", "embed_tokens.weight")
    k = k.replace("model.norm.weight", "norm.weight")
    if k.startswith("model.layers."):
        k = k.replace("model.layers.", "layers.", 1)
        k = k.replace(".self_attn.", ".attention.")
        k = k.replace(".mlp.gate_proj.", ".feed_forward.gate_proj.")
        k = k.replace(".mlp.up_proj.", ".feed_forward.up_proj.")
        k = k.replace(".mlp.down_proj.", ".feed_forward.down_proj.")
    return k


# Which (gpu-side) keys correspond to BinaryLinear / BinaryEmbedding (i.e.
# weights that get Q1_0_g128 quantised). Norm weights are full-precision.
_LIN_SUFFIXES = (
    ".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".o_proj.weight",
    ".gate_proj.weight", ".up_proj.weight", ".down_proj.weight",
)
_EMBED_KEY = "embed_tokens.weight"


def _is_quantised(gpu_key: str, *, quantize_embedding: bool = True) -> bool:
    if gpu_key == _EMBED_KEY:
        return quantize_embedding
    return any(gpu_key.endswith(s) for s in _LIN_SUFFIXES)


# ---------------------------------------------------------------------------
# Q1_0_g128 packing
# ---------------------------------------------------------------------------
@torch.no_grad()
def _pack_q1_0_g128(w_frozen: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack a frozen-grid weight (BF16, ±d_g per group) into (packed, scales).

    Inputs
    ------
    w_frozen : bf16 [out, in],  every element is +d_g or -d_g for its group.

    Outputs
    -------
    packed   : int8 [out, in/8]   sign bits, LSB of byte j = sign of w[j*8+0],
                                  bit 1 of byte j = sign of w[j*8+1], ...
    scales   : bf16 [out, in/128] per-group scale d_g (group along input dim).

    Sign convention:  positive weight -> bit=1, negative weight -> bit=0.
    Matches `decode_i1s_to_i8s` in `q1g128_kernels.h`: bit -> +1 / 0 -> -1.
    """
    assert w_frozen.dtype == torch.bfloat16, w_frozen.dtype
    out_f, in_f = w_frozen.shape
    assert in_f % GROUP_SIZE == 0
    n_groups = in_f // GROUP_SIZE

    # 1. Recover per-group scale d_g = mean(|w|) over each group of 128.
    w_g = w_frozen.float().view(out_f, n_groups, GROUP_SIZE)
    d_g = w_g.abs().mean(dim=-1).to(torch.bfloat16)  # [out, n_groups]

    # 2. Sign bits.  +1 -> 1, -1 -> 0.  Treat exact 0 as +1 to match ggml.
    sign_pos = (w_frozen > 0)            # bool [out, in]
    zero_mask = (w_frozen == 0)
    sign_pos = sign_pos | zero_mask
    sign = sign_pos.to(torch.uint8)      # uint8 [out, in], values in {0,1}

    # 3. Pack 8 bits per byte.  LSB of byte j = sign[..., j*8 + 0].
    sign = sign.view(out_f, in_f // 8, 8)  # [out, in/8, 8]
    bit_pos = torch.arange(8, dtype=torch.uint8, device=sign.device)
    packed = (sign << bit_pos.view(1, 1, 8)).sum(dim=-1).to(torch.uint8)
    packed_i8 = packed.view(torch.int8).contiguous()
    return packed_i8, d_g.contiguous()


@torch.no_grad()
def _pack_t2_g128(w_frozen: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack exact {-s,0,+s} g128 weights as natural-order 2-bit codes.

    Code 1/2/3 represents -1/0/+1.  The CUDA kernel applies the BF16 absmax
    scale separately inside every 128-wide K group before cross-group sum.
    """
    assert w_frozen.dtype == torch.bfloat16
    out_f, in_f = w_frozen.shape
    assert in_f % GROUP_SIZE == 0
    wg = w_frozen.float().view(out_f, in_f // GROUP_SIZE, GROUP_SIZE)
    scale = wg.abs().amax(dim=-1).to(torch.bfloat16)
    q = torch.round(wg / scale.float().unsqueeze(-1).clamp_min(1e-30)) \
        .clamp(-1, 1).to(torch.int8).view(out_f, in_f)
    code = (q + 2).to(torch.uint8).view(out_f, in_f // 4, 4)
    shifts = torch.arange(4, dtype=torch.uint8, device=code.device) * 2
    packed = (code << shifts.view(1, 1, 4)).sum(dim=-1, dtype=torch.uint8)
    return packed.view(torch.int8).contiguous(), scale.contiguous()


# ---------------------------------------------------------------------------
# Main convert routine
# ---------------------------------------------------------------------------
@torch.no_grad()
def convert(
    src_ckpt_dir: str,
    out_dir: str,
    pack_kernel: bool = True,
    pack_fp16: bool = True,
    quant: str = "q1",
    quantize_embedding: bool = True,
) -> None:
    """Freeze HF weights onto the requested grid and optionally pack them.

    ``q1`` writes model_state_q1g128.pt (packed W1A8); ``ternary`` writes
    model_state_t2g128.pt (natural-order packed W2A8). Either format can also
    emit the frozen BF16 representation model_state_fp16.pt. Source config,
    shard containers, and all core tensor keys/shapes are checked before any
    output is created. Model quality and GPU parity require separate checks.
    """
    if quant not in ("q1", "ternary"):
        raise ValueError("quant must be 'q1' or 'ternary'")
    if os.path.lexists(out_dir):
        raise ValueError("conversion output already exists; use a new output directory")
    if not pack_kernel and not pack_fp16:
        raise ValueError("at least one output format must be enabled")
    if quant == "ternary" and not quantize_embedding:
        raise ValueError("ternary requires quantized embedding/head; --keep-embedding-fp is Q1-only")
    hf_config, shards = validate_hf_checkpoint(src_ckpt_dir)
    validate_sdlm_config(src_ckpt_dir, hf_config)
    freeze_fn = (freeze_quantize_to_ternary_g128 if quant == "ternary"
                 else freeze_quantize_to_q1_0_g128)
    print(f"[convert] source config: hidden={hf_config['hidden_size']} "
          f"layers={hf_config['num_hidden_layers']} vocab={hf_config['vocab_size']} "
          f"tie_embed={hf_config['tie_word_embeddings']}")

    # The read-only metadata preflight has checked every shard and core tensor.
    print(f"[convert] reading {len(shards)} safetensors shards")

    state_fp16: Dict[str, torch.Tensor] = {}
    state_q1g128: Dict[str, torch.Tensor] = {}

    for shard in shards:
        with safe_open(os.path.join(src_ckpt_dir, shard), framework="pt", device="cpu") as g:
            for hf_key in g.keys():
                t = g.get_tensor(hf_key)
                gpu_key = _hf_to_gpu_key(hf_key)
                if hf_key.startswith("lm_head.") and hf_config["tie_word_embeddings"]:
                    # Skip; we tie to embed_tokens at runtime.
                    continue

                if _is_quantised(
                    gpu_key, quantize_embedding=quantize_embedding
                ):
                    # Freeze latent BF16 weights onto the selected group-128 grid.
                    t_bf16 = t.to(torch.bfloat16)
                    if t_bf16.dim() != 2:
                        raise ValueError(
                            f"unexpected non-2D quantised tensor {hf_key}: {t.shape}"
                        )
                    in_f = t_bf16.shape[1]
                    if in_f % GROUP_SIZE != 0:
                        # Preserve tensors whose input dimension is not divisible
                        # by the quantization group size.
                        print(f"[convert] WARN: {hf_key} in_features={in_f} not divisible by "
                              f"{GROUP_SIZE}, keeping as BF16")
                        if pack_fp16:
                            state_fp16[gpu_key] = t_bf16
                        if pack_kernel:
                            state_q1g128[gpu_key] = t_bf16
                        continue

                    frozen = freeze_fn(t_bf16)

                    if pack_fp16:
                        state_fp16[gpu_key] = frozen

                    if pack_kernel:
                        if gpu_key == _EMBED_KEY:
                            # The kernel path keeps embedding as BF16 (lookup,
                            # not GEMM). Save the frozen BF16 version directly.
                            state_q1g128[gpu_key] = frozen
                        else:
                            packed, scales = (
                                _pack_t2_g128(frozen) if quant == "ternary"
                                else _pack_q1_0_g128(frozen)
                            )
                            state_q1g128[gpu_key.replace(".weight", ".weight")] = packed
                            state_q1g128[gpu_key.replace(".weight", ".weight_scale")] = scales
                else:
                    # Norm weights, q_norm/k_norm, etc. — keep BF16.
                    t_bf16 = t.to(torch.bfloat16)
                    if pack_fp16:
                        state_fp16[gpu_key] = t_bf16
                    if pack_kernel:
                        state_q1g128[gpu_key] = t_bf16

    # No conversion output is created until every source tensor was read and
    # converted successfully. Invalid/partial inputs cannot leave empty states.
    os.makedirs(out_dir, exist_ok=False)
    if pack_fp16:
        torch.save(state_fp16, os.path.join(out_dir, "model_state_fp16.pt"))
        print(f"[convert] wrote model_state_fp16.pt  ({len(state_fp16)} tensors)")
    if pack_kernel:
        kernel_file = ("model_state_t2g128.pt" if quant == "ternary"
                       else "model_state_q1g128.pt")
        torch.save(state_q1g128, os.path.join(out_dir, kernel_file))
        print(f"[convert] wrote {kernel_file} ({len(state_q1g128)} tensors)")

    # Write a minimal Bonsai-GPU config (for FastGen).
    bonsai_cfg = {
        "dim": hf_config["hidden_size"],
        "n_layers": hf_config["num_hidden_layers"],
        "n_heads": hf_config["num_attention_heads"],
        "n_kv_heads": hf_config["num_key_value_heads"],
        "head_dim": hf_config["head_dim"],
        "ffn_dim": hf_config["intermediate_size"],
        "vocab_size": hf_config["vocab_size"],
        "norm_eps": hf_config["rms_norm_eps"],
        "rope_theta": hf_config["rope_theta"],
        "rope_original_max": hf_config["rope_scaling"]["original_max_position_embeddings"],
        "rope_factor": hf_config["rope_scaling"]["factor"],
        "tie_word_embeddings": bool(hf_config["tie_word_embeddings"]),
        "block_size": int(hf_config.get("block_size", 8)),
        "causal_attn": bool(hf_config.get("causal_attn", True)),
        "mask_token_id": int(hf_config.get("text_mask_token_id", 151654)),
        "kernel_kind": "t2g128" if quant == "ternary" else "q1",
        "quantize_embedding": bool(quantize_embedding),
    }
    with open(os.path.join(out_dir, "bonsai_gpu_config.json"), "w") as f:
        json.dump(bonsai_cfg, f, indent=2)
    print(f"[convert] wrote bonsai_gpu_config.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description='Convert public Bonsai T2 checkpoints: 1.7B D4/D8 or 4B D8.')
    ap.add_argument("--src", required=True, help="HF safetensors directory of SFT or MTP checkpoint")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--no-kernel", action="store_true", help="skip packed Q1/T2 dump")
    ap.add_argument("--no-fp16", action="store_true", help="skip BF16 prefill dump")
    ap.add_argument("--quant", choices=("ternary",), default="ternary",
                    help="public release format: ternary (1.58-bit g128)")
    ap.add_argument(
        "--keep-embedding-fp", action="store_true",
        help="legacy compatibility flag; rejected by the public T2-only release",
    )
    args = ap.parse_args(argv)
    try:
        with open(os.path.join(args.src, 'config.json')) as source:
            config = json.load(source)
        validate_public_model_scope(config, hf=True, quant=args.quant,
                                    quantize_embedding=not args.keep_embedding_fp)
    except (OSError, ValueError, TypeError) as error:
        ap.error(str(error))
    convert(
        src_ckpt_dir=args.src,
        out_dir=args.out,
        pack_kernel=not args.no_kernel,
        pack_fp16=not args.no_fp16,
        quant=args.quant,
        quantize_embedding=not args.keep_embedding_fp,
    )


if __name__ == "__main__":
    main()
