"""Joint next-token and block-prediction trainer for Bonsai.

Initializes `BonsaiForSDLM` from SFT weights and normalizes the summed
AR and MTP losses by their combined valid-target count:

    loss = (sum(CE_AR) + sum(CE_MTP)) / (N_AR_valid + N_MTP_valid)

`compute_loss` combines the two supervised regions explicitly.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .data import IGNORE_INDEX
from .sdlm_data import BonsaiSDLMCollator, BonsaiSDLMDataset
from .sdlm_modeling import (
    BONSAI_MASK_TOKEN_ID,
    BONSAI_PAD_TOKEN_ID,
    BonsaiForSDLM,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument schemas
# ---------------------------------------------------------------------------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "SFT checkpoint (HF safetensors directory) used to initialize MTP."}
    )
    block_size: int = field(default=8, metadata={"help": "Diffusion block size D."})
    causal_attn: bool = field(
        default=True,
        metadata={"help": "If True, intra-block attention is causal (T2 variant)."},
    )
    mask_token_id: int = field(default=BONSAI_MASK_TOKEN_ID)
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
    data_path: str = field(metadata={"help": "Path to SFT JSONL (ShareGPT format)."})
    max_length: int = field(default=4096)
    enable_thinking: bool = field(default=False)
    expected_mask_repeat_times: int = field(default=2)
    max_samples: Optional[int] = field(default=None)


@dataclass
class BonsaiSDLMTrainingArguments(TrainingArguments):
    """Mirrors BitMTP block8_t2 launch defaults."""

    optim: str = field(default="adamw_torch")
    bf16: bool = field(default=True)
    tf32: Optional[bool] = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    save_safetensors: bool = field(default=True)
    report_to: str = field(default="none")
    logging_steps: int = field(default=10)
    remove_unused_columns: bool = field(default=False)
    max_grad_norm: float = field(default=1000.0)
    warmup_ratio: float = field(default=0.03)
    weight_decay: float = field(default=0.1)
    lr_scheduler_type: str = field(default="constant_with_warmup")
    dataloader_num_workers: int = field(default=4)
    dataloader_pin_memory: bool = field(default=True)
    grad_comm_bf16: bool = field(
        default=False,
        metadata={
            "help": "Plain-DDP only (e.g. gloo on MIG): register the bf16 "
                    "gradient-compression DDP comm hook. Halves the per-step "
                    "all-reduce payload (faster sync) at the cost of a tiny "
                    "numerical approximation. OFF by default => grads are "
                    "all-reduced in fp32 (bit-faithful, same training effect)."
        },
    )


# ---------------------------------------------------------------------------
# Trainer: AR + MTP sum-aggregation loss (matches BitMTP)
# ---------------------------------------------------------------------------
class BonsaiSDLMTrainer(Trainer):
    """Trainer with BitMTP-style two-region loss + AR/MTP metrics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # HF Trainer would otherwise pass num_items_in_batch as a kwarg to
        # the model forward — but our forward doesn't use HF's auto-loss
        # plumbing (we compute loss manually). Turn it off so the Trainer
        # path is consistent across micro-batches.
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")
        x0_len = inputs.pop("x0_len", None)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            position_ids=inputs.get("position_ids"),
            x0_len=x0_len,
        )
        logits = outputs.logits  # [B, L, V]
        bsz, seq_len, vocab = logits.shape

        # Standard next-token shift.
        shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab)
        shift_labels = labels[..., 1:].contiguous().view(-1)
        shift_seq_len = seq_len - 1

        # ---- AR / MTP region masks (after shift) ----
        # AR region: positions 0..x0_len-2 (the last x0 position lands on
        # BRIDGE which is IGNORE, so it naturally drops out).
        # MTP region: positions x0_len..end
        device = labels.device
        pos = torch.arange(shift_seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        x0e = x0_len.to(device).unsqueeze(1)
        ar_region = pos < (x0e - 1)
        mtp_region = pos >= x0e
        valid = (labels[..., 1:] != IGNORE_INDEX)

        ar_mask = (ar_region & valid).view(-1)
        mtp_mask = (mtp_region & valid).view(-1)

        per_tok = nn.functional.cross_entropy(
            shift_logits, shift_labels, ignore_index=IGNORE_INDEX, reduction="none"
        )

        n_ar = ar_mask.sum()
        n_mtp = mtp_mask.sum()
        ar_loss = per_tok[ar_mask].sum() if n_ar > 0 else torch.zeros((), device=device)
        mtp_loss = per_tok[mtp_mask].sum() if n_mtp > 0 else torch.zeros((), device=device)

        n_total = n_ar + n_mtp
        # Token-level mean across the full (AR + MTP) supervised population.
        loss = (ar_loss + mtp_loss) / n_total.clamp_min(1)

        # Light-touch metric logging (rank-0 only, every step is fine).
        if self.state.global_step % max(1, self.args.logging_steps) == 0 and self.is_world_process_zero():
            with torch.no_grad():
                ar_mean = (ar_loss / n_ar.clamp_min(1)).detach().float().item() if n_ar > 0 else 0.0
                mtp_mean = (mtp_loss / n_mtp.clamp_min(1)).detach().float().item() if n_mtp > 0 else 0.0
                # Stash into Trainer state for the next log step.
                self._last_ar_loss = ar_mean
                self._last_mtp_loss = mtp_mean
                self._last_n_ar = int(n_ar.item())
                self._last_n_mtp = int(n_mtp.item())

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        # Inject the most recent AR / MTP breakdown into Trainer logs.
        for k in ("_last_ar_loss", "_last_mtp_loss", "_last_n_ar", "_last_n_mtp"):
            if hasattr(self, k):
                key = k.lstrip("_")
                logs[key] = getattr(self, k)
        return super().log(logs, *args, **kwargs)

    def _wrap_model(self, model, training=True, dataloader=None):
        wrapped = super()._wrap_model(model, training=training, dataloader=dataloader)
        # Optional bf16 gradient compression for plain-DDP (gloo on MIG).
        # Only meaningful when the model got wrapped in DDP (not DeepSpeed/FSDP).
        if (
            training
            and getattr(self.args, "grad_comm_bf16", False)
            and isinstance(wrapped, torch.nn.parallel.DistributedDataParallel)
        ):
            from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import (
                bf16_compress_hook,
            )

            wrapped.register_comm_hook(state=None, hook=bf16_compress_hook)
            logger.info("Registered bf16_compress DDP comm hook (faster sync, "
                        "tiny grad approximation).")
        return wrapped

    def _save(self, output_dir=None, state_dict=None):
        """Drop our SDLM wrapper before delegating to HF save.

        We persist the inner Qwen3 model (which is what
        AutoModelForCausalLM.from_pretrained can re-load), then drop our
        small sdlm_config.json side-car for later resume.
        """
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        unwrapped = self.model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        unwrapped.save_pretrained(output_dir, safe_serialization=True)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Restore trained weights on resume — into the INNER hf model.

        Our `_save` persists the inner Qwen3 model via `save_pretrained`, so the
        checkpoint's safetensors keys are the inner model's own keys
        (`model.embed_tokens.*`, `model.layers.*`, `model.norm.weight`,
        `lm_head.weight`). HF's default loader would call `load_state_dict` on
        the BonsaiForSDLM *wrapper*, whose keys carry an extra `model.` prefix
        (`model.model.embed_tokens.*`, `model.lm_head.weight`). That one-level
        mismatch makes the default loader silently drop EVERY weight (all keys
        land in missing/unexpected) — the model keeps its base-pretrained values
        while the step-N optimizer momentum is still restored, i.e. a corrupt
        resume that loses all trained progress. So we load the shards straight
        into the inner model, where the keys match exactly.
        """
        import glob
        from safetensors.torch import load_file as _safe_load_file

        target = model if model is not None else self.model
        inner = target
        while hasattr(inner, "module"):
            inner = inner.module
        # BonsaiForSDLM -> inner AutoModelForCausalLM (whose keys match the file)
        inner = getattr(inner, "model", inner)

        shards = sorted(glob.glob(os.path.join(resume_from_checkpoint, "model*.safetensors")))
        single = os.path.join(resume_from_checkpoint, "model.safetensors")
        if not shards and os.path.isfile(single):
            shards = [single]
        if not shards:
            logger.warning("resume: no safetensors found in %s; falling back to "
                           "HF default loader.", resume_from_checkpoint)
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)

        state = {}
        for sh in shards:
            state.update(_safe_load_file(sh))
        missing, unexpected = inner.load_state_dict(state, strict=False)
        # Tied embeddings: save_pretrained omits lm_head.weight (it is tied to
        # embed_tokens.weight). Re-tie after load so lm_head points at the just-
        # restored embeddings, then treat the tied head key as legitimately
        # absent rather than a lost weight.
        tied = bool(getattr(getattr(inner, "config", None), "tie_word_embeddings", False))
        if tied and hasattr(inner, "tie_weights"):
            inner.tie_weights()
        # Non-persistent buffers (e.g. rotary inv_freq) legitimately absent.
        ignore = ("rotary", "inv_freq")
        missing = [
            k for k in missing
            if not any(t in k for t in ignore)
            and not (tied and k.endswith("lm_head.weight"))
        ]
        if unexpected:
            raise RuntimeError(
                f"resume: {len(unexpected)} unexpected keys when loading into "
                f"inner model (e.g. {unexpected[:3]}). Save/load key convention "
                f"diverged — refusing to resume with mismatched weights."
            )
        if missing:
            raise RuntimeError(
                f"resume: {len(missing)} weight keys NOT restored (e.g. "
                f"{missing[:3]}). Refusing to resume with partially-loaded "
                f"weights (would corrupt training)."
            )
        logger.info("resume: restored %d weight tensors into inner model from "
                    "%d shard(s) at %s", len(state), len(shards),
                    resume_from_checkpoint)


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
    parser = HfArgumentParser((ModelArguments, DataArguments, BonsaiSDLMTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    _setup_logging(training_args.local_rank)

    logger.info("=== Bonsai MTP training (block=%d, causal_attn=%s) ===",
                model_args.block_size, model_args.causal_attn)
    logger.info("model      : %s", model_args.model_name_or_path)
    logger.info("data       : %s", data_args.data_path)
    logger.info("output_dir : %s", training_args.output_dir)
    logger.info("per_dev_bs : %s  grad_accum : %s  lr : %s",
                training_args.per_device_train_batch_size,
                training_args.gradient_accumulation_steps,
                training_args.learning_rate)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = BONSAI_PAD_TOKEN_ID

    # DeepSpeed maintains FP32 master parameters for BF16 training. Plain
    # DDP loads FP32 parameters so BF16 autocast does not make optimizer
    # updates operate directly on BF16 parameters.
    use_deepspeed = bool(getattr(training_args, "deepspeed", None))
    load_dtype = torch.bfloat16 if use_deepspeed else torch.float32
    logger.info("master-weight path: %s (load_dtype=%s)",
                "DeepSpeed ZeRO-2 bf16 (fp32 master internal)" if use_deepspeed
                else "plain DDP + bf16 autocast (fp32 master)", load_dtype)

    model = BonsaiForSDLM(
        model_path=model_args.model_name_or_path,
        block_size=model_args.block_size,
        causal_attn=model_args.causal_attn,
        mask_token_id=model_args.mask_token_id,
        torch_dtype=load_dtype,
        apply_qat=True,
        quantize_embedding=model_args.quantize_embedding,
        quant_scheme=model_args.quant_scheme,
    )
    logger.info("quant_scheme=%s  quantize_embedding=%s",
                model_args.quant_scheme, model_args.quantize_embedding)
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    train_ds = BonsaiSDLMDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        block_size=model_args.block_size,
        mask_token_id=model_args.mask_token_id,
        max_length=data_args.max_length,
        enable_thinking=data_args.enable_thinking,
        expected_mask_repeat_times=data_args.expected_mask_repeat_times,
        max_samples=data_args.max_samples,
    )
    collator = BonsaiSDLMCollator(pad_token_id=tokenizer.pad_token_id or BONSAI_PAD_TOKEN_ID)

    trainer = BonsaiSDLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        data_collator=collator,
    )

    resume_ckpt = getattr(training_args, "resume_from_checkpoint", None)
    if resume_ckpt:
        logger.info("=== Resuming MTP training from checkpoint: %s ===", resume_ckpt)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(training_args.output_dir)
    if training_args.local_rank in (-1, 0):
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info("=== Bonsai MTP training complete. Saved to %s ===", training_args.output_dir)


if __name__ == "__main__":
    main()
