"""Supervised fine-tuning for Bonsai with ternary group-128 QAT.

Model-specific recipes are provided in configs/training. Use
scripts/train_bonsai.py to launch a recipe or this module's --help to inspect
the native Trainer options.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .convert import (
    convert_qwen3_to_ternary_qat,
    count_ternary_modules,
)
from .data import BonsaiSFTCollator, BonsaiSFTDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument schemas
# ---------------------------------------------------------------------------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to Bonsai-*-unpacked (Qwen3 BF16 HF safetensors)."}
    )
    trust_remote_code: bool = field(default=False)
    quant_scheme: str = field(
        default="ternary",
        metadata={"choices": ("ternary",),
                  "help": "Group-128 absmax ternary QAT, including embeddings and the tied head."},
    )
    quantize_embedding: bool = field(
        default=True,
        metadata={"help": "Must be true: ternary QAT includes embeddings and the tied head."},
    )

    def __post_init__(self):
        if self.quant_scheme != "ternary" or self.quantize_embedding is not True:
            raise ValueError("Bonsai training requires quant_scheme='ternary' and quantize_embedding=True")


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to SFT JSONL (ShareGPT or messages)."})
    max_length: int = field(default=4096)
    enable_thinking: bool = field(default=False)
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Limit samples (smoke test)."},
    )


@dataclass
class BonsaiTrainingArguments(TrainingArguments):
    """SFT defaults: BF16 QAT, AdamW, warmup, and gradient clipping at 1.0.

    The saved recipes supply learning rate, batch size, and training duration.
    """
    optim: str = field(default="adamw_torch")
    bf16: bool = field(default=True)
    tf32: Optional[bool] = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    save_safetensors: bool = field(default=True)
    report_to: str = field(default="none")
    logging_steps: int = field(default=1)
    remove_unused_columns: bool = field(default=False)
    max_grad_norm: float = field(default=1.0)
    warmup_ratio: float = field(default=0.03)
    weight_decay: float = field(default=0.1)
    lr_scheduler_type: str = field(default="constant_with_warmup")
    dataloader_num_workers: int = field(default=4)
    dataloader_pin_memory: bool = field(default=True)


# ---------------------------------------------------------------------------
# Trainer note
# ---------------------------------------------------------------------------
# We do NOT subclass Trainer: the default `Trainer.save_model` already handles
# DeepSpeed ZeRO-3 weight-gather (when `stage3_gather_16bit_weights_on_model_save=True`
# in the DS config) and writes a regular HF checkpoint. The saved state_dict
# contains latent BF16 weights with the original tensor shapes. Reapply
# ternary QAT when loading for training, or round with export_ternary for export.


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _setup_logging(local_rank: int) -> None:
    level = logging.INFO if local_rank in (-1, 0) else logging.WARNING
    logging.basicConfig(
        level=level,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    transformers.utils.logging.set_verbosity(level)


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, BonsaiTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    _setup_logging(training_args.local_rank)

    logger.info("=== Bonsai SFT (QAT scheme=%s) ===", model_args.quant_scheme)
    logger.info("model      : %s", model_args.model_name_or_path)
    logger.info("data       : %s", data_args.data_path)
    logger.info("output_dir : %s", training_args.output_dir)
    logger.info("max_steps  : %s", training_args.max_steps)
    logger.info("per_device_bs : %s, grad_accum : %s",
                training_args.per_device_train_batch_size,
                training_args.gradient_accumulation_steps)
    logger.info("lr : %s, bf16 : %s, grad_ckpt : %s",
                training_args.learning_rate, training_args.bf16,
                training_args.gradient_checkpointing)

    # ----- Tokenizer -----
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----- Model -----
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    # Disable the generation cache when using gradient checkpointing.
    config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=model_args.trust_remote_code,
    )

    n_layers = int(getattr(config, "num_hidden_layers", 0))
    # Ternary-Bonsai: fully quantized (embed + 7*L projections + tied head).
    n_lin, n_emb = convert_qwen3_to_ternary_qat(model)
    n_lin_after, n_emb_after = count_ternary_modules(model)
    expected_lin = n_layers * 7 + 1  # + lm_head (separate module even when tied)
    expected_emb = 1
    qat_name = "Ternary"
    logger.info("%sLinear count after convert: %d (newly replaced: %d, expected: %d)",
                qat_name, n_lin_after, n_lin, expected_lin)
    logger.info("%sEmbedding count after convert: %d (newly replaced: %d, expected: %d)",
                qat_name, n_emb_after, n_emb, expected_emb)
    logger.info("Model: num_hidden_layers=%d, tie_word_embeddings=%s, quant_scheme=%s, "
                "quantize_embedding=%s",
                n_layers, getattr(config, "tie_word_embeddings", False),
                model_args.quant_scheme, model_args.quantize_embedding)
    if n_emb_after != expected_emb:
        raise RuntimeError(
            f"Expected {expected_emb} {qat_name}Embedding after conversion, got {n_emb_after}."
        )
    if n_lin_after < expected_lin:
        raise RuntimeError(
            f"Expected >= {expected_lin} {qat_name}Linear modules, got {n_lin_after}. "
            f"Check convert.py rules vs. Qwen3 module names."
        )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # ----- Data -----
    train_ds = BonsaiSFTDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        enable_thinking=data_args.enable_thinking,
        max_samples=data_args.max_samples,
    )
    collator = BonsaiSFTCollator(pad_token_id=tokenizer.pad_token_id)

    # ----- Train -----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        data_collator=collator,
    )

    # IMPORTANT: HF Trainer.train() does NOT auto-read args.resume_from_checkpoint;
    # it must be passed explicitly (matches bitmtp/train/train_sft.py). Without
    # this, --resume_from_checkpoint is silently ignored and training restarts
    # from scratch.
    resume_ckpt = training_args.resume_from_checkpoint
    if resume_ckpt:
        logger.info("Resuming training from checkpoint: %s", resume_ckpt)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(training_args.output_dir)
    if training_args.local_rank in (-1, 0):
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info("=== Bonsai SFT done (scheme=%s). Latent BF16 ckpt saved. ===",
                    model_args.quant_scheme)
        logger.info("Next: `python -m bonsai_sft.export_ternary --ckpt %s ...` "
                    "to export rounded group-128 ternary weights.", training_args.output_dir)


if __name__ == "__main__":
    main()
