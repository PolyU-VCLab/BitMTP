"""SFT datasets for Bonsai/Qwen3 chat models.

Data format (one JSON line per sample):
    {"id": int, "conversations": [{"from": "human"|"gpt", "value": str}, ...]}

Pipeline per sample:
    1. Convert ShareGPT -> OpenAI-style messages list.
    2. Render with Bonsai chat_template (enable_thinking=False).
    3. Tokenize with Bonsai tokenizer (Qwen3, vocab=151669).
    4. Build label mask: only assistant content tokens are supervised.
       Tokens belonging to system / user turns AND the role-header tokens
       around each assistant turn are set to -100 (IGNORE_INDEX).

Assistant-region detection works by scanning the token stream for the
pattern `<|im_start|> assistant \n ... <|im_end|>`. We supervise the
content portion (everything between `assistant \n` and `<|im_end|>`)
inclusive of the closing `<|im_end|>` so that the model learns to stop.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

IGNORE_INDEX: int = -100

# Bonsai tokenizer special-token IDs (Qwen3 family).
# Verified against the Bonsai tokenizer configuration
IM_START_ID: int = 151644            # <|im_start|>
IM_END_ID: int = 151645              # <|im_end|>
ASSISTANT_WORD_ID: int = 77091       # 'assistant' (single token after im_start)
USER_WORD_ID: int = 872              # 'user'
SYSTEM_WORD_ID: int = 8948           # 'system' (lazy; not strictly needed)
NEWLINE_ID: int = 198                # '\n' immediately after the role word


# ---------------------------------------------------------------------------
# ShareGPT -> messages
# ---------------------------------------------------------------------------
def sharegpt_to_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """Robust conversion to OpenAI-style messages.

    Accepts the following conversation dataset formats:
        * {"conversations": [{"from": "human"|"user"|"gpt"|"assistant"|"system",
                              "value": str}, ...]}                    (ShareGPT)
        * {"messages": [{"role": ..., "content": ...}, ...]}          (OpenAI)
        * {"input": str, "output": str, "system_prompt"?: str}        (legacy)
    """
    if "messages" in item:
        return [{"role": m["role"], "content": m["content"]} for m in item["messages"]]
    if "conversations" in item:
        msgs: List[Dict[str, str]] = []
        for c in item["conversations"]:
            src = c.get("from", "").lower()
            if src in ("human", "user"):
                role = "user"
            elif src in ("gpt", "assistant", "model"):
                role = "assistant"
            elif src == "system":
                role = "system"
            else:
                role = "user"
            msgs.append({"role": role, "content": c.get("value", "")})
        return msgs
    if "input" in item and "output" in item:
        msgs = []
        if item.get("system_prompt"):
            msgs.append({"role": "system", "content": item["system_prompt"]})
        msgs.append({"role": "user", "content": item["input"]})
        msgs.append({"role": "assistant", "content": item["output"]})
        return msgs
    raise ValueError(f"Unrecognized SFT sample format: keys={list(item.keys())}")


# ---------------------------------------------------------------------------
# Label-mask builder (Bonsai/Qwen3 chat-template aware)
# ---------------------------------------------------------------------------
def build_labels_for_qwen3(input_ids: Sequence[int]) -> List[int]:
    """Return labels: -100 everywhere except inside assistant content spans.

    A span starts right AFTER the sequence `<|im_start|> assistant \\n`
    (token IDs 151644, 77091, 198) and ENDS at the next `<|im_end|>` (151645).
    The closing `<|im_end|>` is INCLUDED as a supervised label so the model
    learns when to stop generating.

    This works whether or not the assistant turn includes the empty thinking
    head `<think>\\n\\n</think>\\n\\n`: those tokens fall inside the span and
    are simply included in the supervision (they appear verbatim in training
    data only on the last assistant turn under Bonsai's template).
    """
    n = len(input_ids)
    labels = [IGNORE_INDEX] * n
    i = 0
    while i < n - 2:
        if (
            input_ids[i] == IM_START_ID
            and input_ids[i + 1] == ASSISTANT_WORD_ID
            and input_ids[i + 2] == NEWLINE_ID
        ):
            j = i + 3
            while j < n and input_ids[j] != IM_END_ID:
                labels[j] = input_ids[j]
                j += 1
            if j < n:  # include closing <|im_end|>
                labels[j] = input_ids[j]
                i = j + 1
            else:
                i = n
            continue
        i += 1
    return labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class BonsaiSFTDataset(Dataset):
    """ShareGPT-format SFT dataset with on-the-fly templating + tokenization.

    Records are parsed once in __init__ and held in RAM (eager full load).
    Account for a separate in-memory dataset in each training process.

    Args:
        data_path: path to .jsonl (one JSON object per line).
        tokenizer: a Bonsai (Qwen3) tokenizer.
        max_length: maximum tokenized sequence length.
        enable_thinking: pass-through to chat_template; defaults to False.
        max_samples: optional cap, used by smoke tests.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 4096,
        enable_thinking: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        if not os.path.isfile(data_path):
            raise FileNotFoundError(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enable_thinking = enable_thinking
        self.data_path = data_path

        # Keep parsed records in memory; format and tokenize them on access.
        t0 = time.time()
        self.records: List[Dict[str, Any]] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for raw in f:
                if not raw.strip():
                    continue
                self.records.append(json.loads(raw))
                if max_samples is not None and len(self.records) >= max_samples:
                    break
        logger.info("Loaded %d samples from %s (eager in-memory) in %.1fs",
                    len(self.records), data_path, time.time() - t0)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        messages = sharegpt_to_messages(item)
        # Render to a single string (drop generation prompt: we want labels
        # to cover ALL assistant turns present in the sample).
        text: str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=self.enable_thinking,
        )
        ids: List[int] = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > self.max_length:
            ids = ids[: self.max_length]
        labels = build_labels_for_qwen3(ids)
        if not any(label != IGNORE_INDEX for label in labels[1:]):
            raise ValueError(
                f"Bonsai SFT sample {idx}: assistant-only supervision found no "
                "supervised assistant tokens after tokenization/truncation. "
                "Check the Qwen3 chat template and tokenizer marker IDs, retain "
                "an assistant response, or increase max_length."
            )
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Collator: right-pad to longest in batch, mask labels on pad with -100
# ---------------------------------------------------------------------------
@dataclass
class BonsaiSFTCollator:
    pad_token_id: int = 151643  # <|endoftext|>

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(int(f["input_ids"].size(0)) for f in features)
        bsz = len(features)
        input_ids = torch.full((bsz, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((bsz, max_len), IGNORE_INDEX, dtype=torch.long)
        attn = torch.zeros((bsz, max_len), dtype=torch.long)
        for i, f in enumerate(features):
            n = int(f["input_ids"].size(0))
            input_ids[i, :n] = f["input_ids"]
            labels[i, :n] = f["labels"]
            attn[i, :n] = 1
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
