"""Export a Ternary-Bonsai QAT checkpoint as rounded HF safetensors.

The exporter reapplies ternary QAT modules, rounds their latent weights with
the group-128 training quantizer, and saves model and tokenizer files.

Usage:
    python -m bonsai_sft.export_ternary \
        --ckpt ./ckpt_ternary \
        --out  ./ckpt_ternary/ternary_ready
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .convert import convert_qwen3_to_ternary_qat
from .ternary_modules import (
    TernaryEmbedding,
    TernaryLinear,
    freeze_quantize_to_ternary_g128,
)

logger = logging.getLogger(__name__)


def validate_sdlm_config(ckpt_dir: str, config) -> dict | None:
    """Check saved MTP metadata against the HF config before exporting files."""
    path = os.path.join(ckpt_dir, "sdlm_config.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as source:
        saved = json.load(source)
    if not isinstance(saved, dict):
        raise ValueError("sdlm_config.json must contain a JSON object")
    for key, value_type in (("block_size", int), ("causal_attn", bool),
                            ("text_mask_token_id", int)):
        value = saved.get(key)
        if type(value) is not value_type:
            raise ValueError(f"sdlm_config.json: invalid or missing {key}")
        hf_value = config.get(key) if isinstance(config, dict) else getattr(config, key, None)
        if type(hf_value) is not value_type or hf_value != value:
            raise ValueError(f"sdlm_config.json conflicts with config.json: {key}")
    if saved["block_size"] <= 0:
        raise ValueError("sdlm_config.json: block_size must be positive")
    vocab_size = config.get("vocab_size") if isinstance(config, dict) else config.vocab_size
    if not 0 <= saved["text_mask_token_id"] < vocab_size:
        raise ValueError("sdlm_config.json: text_mask_token_id is outside the vocabulary")
    return saved


@torch.no_grad()
def round_latent_to_ternary(model: torch.nn.Module) -> int:
    """In-place round every ternary-QAT weight to the {-s,0,+s} group-128 grid."""
    n = 0
    for _name, mod in model.named_modules():
        if isinstance(mod, (TernaryLinear, TernaryEmbedding)):
            new_w = freeze_quantize_to_ternary_g128(mod.weight.data)
            mod.weight.data.copy_(new_w)
            n += 1
    logger.info("Rounded %d Ternary* tensors to the group-128 {-s,0,+s} grid "
                "(s = absmax per group of 128)", n)
    return n


def export_round_to_safetensors(ckpt_dir: str, out_dir: str) -> None:
    if os.path.lexists(out_dir):
        raise ValueError("export output already exists; use a new output directory")
    logger.info("Loading latent BF16 checkpoint from %s ...", ckpt_dir)
    config = AutoConfig.from_pretrained(ckpt_dir)
    sdlm_config = validate_sdlm_config(ckpt_dir, config)
    os.makedirs(out_dir, exist_ok=False)
    config.use_cache = True
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    convert_qwen3_to_ternary_qat(model)
    round_latent_to_ternary(model)
    logger.info("Saving rounded BF16 checkpoint to %s ...", out_dir)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    tok.save_pretrained(out_dir)
    if sdlm_config is not None:
        with open(os.path.join(out_dir, "sdlm_config.json"), "w", encoding="utf-8") as output:
            json.dump(sdlm_config, output, indent=2)
            output.write("\n")

    for fname in ("chat_template.jinja", "generation_config.json",
                  "added_tokens.json", "merges.txt", "vocab.json",
                  "special_tokens_map.json"):
        src = os.path.join(ckpt_dir, fname)
        dst = os.path.join(out_dir, fname)
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy2(src, dst)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Latent BF16 HF checkpoint dir.")
    ap.add_argument("--out", required=True, help="Output dir for rounded HF safetensors and tokenizer.")
    args = ap.parse_args()
    export_round_to_safetensors(args.ckpt, args.out)


if __name__ == "__main__":
    main()
