"""Group-128 ternary Bonsai wrapper for masked-block training.

Loads a Qwen3-family HF checkpoint and applies ternary QAT while preserving
tied embedding/output-head parameters. The mask token is 151654
(<|vision_pad|>); the chat EOS token is 151645 (<|im_end|>).

The training-time forward override builds the 4D SDLM mask from
position_ids + x0_len and feeds it to the underlying Qwen3 forward.
The mask logic itself is in `sdlm_attn_mask.py` and is model-agnostic.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

from .convert import convert_qwen3_to_ternary_qat
from .sdlm_attn_mask import (
    create_block_diff_mask_by_pe_4d,
    find_prefix_seq_length_by_pe,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bonsai (Qwen3) special-token constants for SDLM
# ---------------------------------------------------------------------------
BONSAI_MASK_TOKEN_ID: int = 151654   # <|vision_pad|> — safe to hijack
BONSAI_EOS_TOKEN_ID: int = 151645    # <|im_end|>
BONSAI_PAD_TOKEN_ID: int = 151643    # <|endoftext|>
BONSAI_IM_START_ID: int = 151644
BONSAI_ASSISTANT_WORD_ID: int = 77091
BONSAI_NEWLINE_ID: int = 198


class BonsaiForSDLM(nn.Module):
    """Bonsai/Qwen3 model with masked-block training and group-128 QAT.

    Loads Hugging Face weights, applies ternary QAT, and supplies
    a 4D block-attention mask during training. Evaluation uses causal attention.

    Args:
        model_path: Hugging Face model directory; MTP recipes start from SFT.
        block_size: diffusion block size D (default 8).
        causal_attn: if True, use causal mutual block (T2 variant); if
            False, use bidirectional mutual block (vanilla SDLM).
        mask_token_id: input-side mask token (default 151654 = vision_pad).
        torch_dtype: weight load dtype.
        apply_qat: if True, apply the selected QAT scheme to linear/embedding
            modules; False leaves the model's floating-point modules unchanged.
        quant_scheme: must be "ternary" for group-128 ternary quantization.
        quantize_embedding: must be True; ternary QAT includes embeddings and
            the tied output head.
    """

    _keys_to_ignore_on_save = None
    _keys_to_ignore_on_load_missing = None
    _keys_to_ignore_on_load_unexpected = None
    base_model_prefix = "model"

    def __init__(
        self,
        model_path: str,
        block_size: int = 8,
        causal_attn: bool = True,
        mask_token_id: int = BONSAI_MASK_TOKEN_ID,
        torch_dtype: torch.dtype = torch.bfloat16,
        apply_qat: bool = True,
        quantize_embedding: bool = True,
        quant_scheme: str = "ternary",
    ):
        super().__init__()
        if quant_scheme != "ternary" or quantize_embedding is not True:
            raise ValueError("BonsaiForSDLM requires quant_scheme='ternary' and quantize_embedding=True")

        # Honour any previously-saved sdlm_config.json (so we can resume).
        sdlm_cfg = os.path.join(model_path, "sdlm_config.json")
        if os.path.exists(sdlm_cfg):
            with open(sdlm_cfg, "r") as f:
                saved = json.load(f)
            block_size = int(saved.get("block_size", block_size))
            causal_attn = bool(saved.get("causal_attn", causal_attn))
            mask_token_id = int(saved.get("text_mask_token_id", mask_token_id))
            logger.info("Loaded SDLM config from %s: block=%d causal_attn=%s mask=%d",
                        sdlm_cfg, block_size, causal_attn, mask_token_id)

        config = AutoConfig.from_pretrained(model_path)
        config.use_cache = False  # play nice with grad-ckpt
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch_dtype,
        )

        if apply_qat:
            # Quantize embeddings, decoder projections, and the tied head.
            n_lin, n_emb = convert_qwen3_to_ternary_qat(self.model)
            logger.info("BonsaiForSDLM: applied ternary QAT (%d TernaryLinear, "
                        "%d TernaryEmbedding, quant_scheme=ternary)", n_lin, n_emb)

        self.block_size = int(block_size)
        self.causal_attn = bool(causal_attn)
        self.text_mask_token_id = int(mask_token_id)
        # Persist on inner config for later inspection/saving.
        self.model.config.block_size = self.block_size
        self.model.config.causal_attn = self.causal_attn
        self.model.config.text_mask_token_id = self.text_mask_token_id

    # ------------------------------------------------------------------ glue
    @property
    def config(self):
        return self.model.config

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def tie_weights(self):
        if hasattr(self.model, "tie_weights"):
            self.model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens):
        return self.model.resize_token_embeddings(new_num_tokens)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is not None:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        else:
            self.model.gradient_checkpointing_enable()

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        x0_len: Optional[torch.LongTensor] = None,
        sdlm_inference: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # Train path: build 4D SDLM mask from position_ids + x0_len.
        if self.training and position_ids is not None:
            if x0_len is None:
                x0_len = find_prefix_seq_length_by_pe(position_ids)
            # 4D bf16 mask, 0 where allowed, -inf where blocked.
            sdlm_mask, _ = create_block_diff_mask_by_pe_4d(
                block_size=self.block_size,
                x0_len_list=x0_len.to(input_ids.device),
                position_ids=position_ids,
                causal_attn=self.causal_attn,
            )
            # Fold the per-sample padding mask in.
            if attention_mask is not None and attention_mask.dim() == 2:
                pad = attention_mask[:, None, None, :].to(dtype=sdlm_mask.dtype)
                pad = (1.0 - pad) * torch.finfo(self.dtype).min
                sdlm_mask = sdlm_mask + pad
            return self.model(
                input_ids=input_ids,
                attention_mask=sdlm_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=None,  # we compute loss in the Trainer.
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                **kwargs,
            )
        # SDLM inference path: same 4D mask family as training, plus anchor_prev
        # masking so the diffusion block cannot peek at the duplicated anchor.
        if sdlm_inference:
            sdlm_mask = self._create_sdlm_inference_mask(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            return self.model(
                input_ids=input_ids,
                attention_mask=sdlm_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                **kwargs,
            )

        # Eval / inference path: standard causal attn.
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

    # ------------------------------------------------------------- inference
    def _create_sdlm_inference_mask(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache],
    ) -> torch.Tensor:
        """Build a 4D causal SDLM-inference mask.

        For ``causal_attn=False`` the last ``block_size`` query positions can
        bidirectionally attend within the diffusion block; for ``causal_attn=
        True`` (T2 variant, used by Bonsai-MTP) the block stays causal.

        Either way, the anchor_prev key index is masked so that the duplicated
        anchor token does not leak twice into the block — matching the train-
        time mask geometry exactly.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        dtype = self.dtype

        past_len = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_len = past_key_values.get_seq_length()
            else:
                past_len = past_key_values[0][0].shape[2]

        total_len = past_len + seq_len
        causal_mask = torch.full(
            (batch_size, 1, seq_len, total_len), float("-inf"),
            dtype=dtype, device=device,
        )
        for i in range(seq_len):
            causal_mask[:, :, i, : past_len + i + 1] = 0.0

        if seq_len >= self.block_size:
            block_start = seq_len - self.block_size
            if not self.causal_attn:
                causal_mask[:, :, block_start:, past_len + block_start:] = 0.0
            anchor_prev_kv_idx = past_len + block_start - 1
            if anchor_prev_kv_idx >= 0:
                causal_mask[:, :, block_start:, anchor_prev_kv_idx] = float("-inf")

        return causal_mask

    # ---------------------------------------------------------------- save
    def save_pretrained(self, save_directory: str, **kwargs):
        self.model.save_pretrained(save_directory, **kwargs)
        cfg = {
            "block_size": self.block_size,
            "causal_attn": self.causal_attn,
            "text_mask_token_id": self.text_mask_token_id,
            "mask_token_str": "<|vision_pad|>",
        }
        with open(os.path.join(save_directory, "sdlm_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
