"""SDLM data pipeline for Bonsai-Qwen3.

Data layout:

    1. Convert ShareGPT/messages -> OpenAI-style messages list.
    2. Render with Bonsai chat_template (enable_thinking=False),
       tokenize WITHOUT special tokens.
    3. Find every assistant-content region by scanning for the Qwen3
       pattern `<|im_start|> assistant \\n ... <|im_end|>`:
           span_start = idx_of(IM_START, ASSISTANT_WORD, NEWLINE) + 3
           span_end   = next IM_END after span_start  (exclusive)
       These (start, end) intervals are the *AR-supervised* regions and
       also the candidate area where SDLM diffusion blocks can be placed.
    4. Sample ~ expected_mask_repeat_times * total_resp_len mask positions
       from those intervals (min_distance=block_size, prevents overlap).
    5. Build:
           final_mask_ids       = [anchor, M, M, .., anchor, M, M, ...]
                                  per block: 1 real token + (D-1) mask
           final_mask_targets   = [y1, y2, ..., yD, y1, y2, ..., yD, ...]
                                  per block: D next-tokens (shifted by 1)
           final_mask_positions = continuous from each anchor's position_id
    6. Concatenate:
           input_ids = [x0_prefix, mask_blocks, PAD]
           labels    = [ar_labels, BRIDGE(IGNORE), mtp_targets]
           position  = [arange(|x0|), mask_positions, last_pos+1]
           x0_len    = |x0|
       After the Trainer's standard shift (logits[:-1] vs labels[1:]):
           h(x0_last) -> BRIDGE(IGNORE)
           h(anchor)  -> y1
           h(mask_j)  -> y_{j+1}
"""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import IGNORE_INDEX, sharegpt_to_messages
from .sdlm_modeling import (
    BONSAI_ASSISTANT_WORD_ID,
    BONSAI_EOS_TOKEN_ID,
    BONSAI_IM_START_ID,
    BONSAI_MASK_TOKEN_ID,
    BONSAI_NEWLINE_ID,
    BONSAI_PAD_TOKEN_ID,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Position-sampling helpers
# ---------------------------------------------------------------------------
def _sample_positions_from_intervals(
    start_positions: Sequence[int],
    end_positions: Sequence[int],
    num_sample: int,
    min_distance: int = 1,
) -> List[int]:
    """Uniformly-spaced sample with jitter; enforces min_distance.

    Identical to BitMTP `sample_positions_from_intervals`.
    """
    all_positions: List[int] = []
    for s, e in zip(start_positions, end_positions):
        all_positions.extend(range(s, e))
    if not all_positions:
        return []

    if num_sample >= len(all_positions):
        sampled = sorted(all_positions)
    else:
        step = max(len(all_positions) / num_sample, min_distance)
        actual_num = min(num_sample, max(1, int(len(all_positions) / step)))
        sampled_idx = [int(i * step) for i in range(actual_num)]
        sampled = [all_positions[i] for i in sampled_idx]
        jitter_range = int(step * 0.3)
        if jitter_range > 0:
            for i in range(len(sampled)):
                jitter = random.randint(-jitter_range, jitter_range)
                new_idx = max(0, min(sampled_idx[i] + jitter, len(all_positions) - 1))
                sampled[i] = all_positions[new_idx]

    sampled = sorted(set(sampled))
    if min_distance > 1 and len(sampled) > 1:
        filtered = [sampled[0]]
        for p in sampled[1:]:
            if p - filtered[-1] >= min_distance:
                filtered.append(p)
        sampled = filtered
    return sampled


def _build_mask_target(
    input_ids: np.ndarray,
    mask_start_positions: Sequence[int],
    block_size: int,
    eos_token_id: int,
) -> np.ndarray:
    """For each diffusion block, write [y1, y2, ..., yD] = next-D tokens.

    Identical to BitMTP `build_mask_target`.
    """
    n = len(input_ids)
    total = len(mask_start_positions) * block_size
    out = np.full(total, IGNORE_INDEX, dtype=np.int64)
    for i, pos in enumerate(mask_start_positions):
        start_idx = i * block_size
        for j in range(block_size):
            cur = pos + j + 1  # predict the NEXT token
            tgt = start_idx + j
            if tgt >= total or cur >= n:
                break
            v = int(input_ids[cur])
            out[tgt] = v
            if v == eos_token_id:
                break
    return out


def _build_mask_position_ids(
    mask_start_positions: Sequence[int], block_size: int
) -> np.ndarray:
    """For each block, position_ids = [anchor_pos, anchor_pos+1, ..., +D-1].

    Identical to BitMTP `build_mask_position_ids`.
    """
    total = len(mask_start_positions) * block_size
    out = np.ones(total, dtype=np.int64)
    for i, pos in enumerate(mask_start_positions):
        s = i * block_size
        out[s] = pos
        for j in range(1, block_size):
            out[s + j] = pos + j
    return out


# ---------------------------------------------------------------------------
# Qwen3 assistant-region detection (Bonsai-specific)
# ---------------------------------------------------------------------------
def _find_assistant_regions(
    input_ids: np.ndarray, eos_token_id: int = BONSAI_EOS_TOKEN_ID
) -> List[tuple]:
    """Return list of (start, end) intervals (token positions) for assistant
    content. start is the first content token (right after `\\nassistant\\n`);
    end is the position of the closing `<|im_end|>` (exclusive in our convention
    but the BitMTP code treats it as the last predictable index — see below).

    Detection pattern in Qwen3 chat_template:
        ... <|im_start|> 'assistant' '\\n'  CONTENT...  <|im_end|> '\\n' ...

    We supervise CONTENT through end (= index of <|im_end|>) so the model
    learns to stop. The interval used for diffusion-block sampling is
    [start, end) so we never put a block anchor on <|im_end|> itself
    (predicting "after eos" is meaningless).
    """
    n = len(input_ids)
    out: List[tuple] = []
    i = 0
    while i < n - 2:
        if (
            input_ids[i] == BONSAI_IM_START_ID
            and input_ids[i + 1] == BONSAI_ASSISTANT_WORD_ID
            and input_ids[i + 2] == BONSAI_NEWLINE_ID
        ):
            start = i + 3
            j = start
            while j < n and input_ids[j] != eos_token_id:
                j += 1
            # BitMTP convention: end = position of EOS, interval = [start, end)
            if j > start:
                out.append((int(start), int(j)))
            i = j + 1
        else:
            i += 1
    return out


# ---------------------------------------------------------------------------
# Core preprocessor: one sample -> SDLM tensors
# ---------------------------------------------------------------------------
def preprocess_bonsai_sdlm(
    messages: List[Dict[str, str]],
    tokenizer,
    block_size: int = 8,
    mask_token_id: int = BONSAI_MASK_TOKEN_ID,
    expected_mask_repeat_times: int = 2,
    max_length: int = 4096,
    enable_thinking: bool = False,
) -> Dict[str, torch.Tensor]:
    """Build one SDLM sample (x0_prefix + mask_blocks + bridge)."""
    eos_id = tokenizer.eos_token_id or BONSAI_EOS_TOKEN_ID
    pad_id = tokenizer.pad_token_id or BONSAI_PAD_TOKEN_ID

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    enc = tokenizer(
        text,
        return_tensors="np",
        padding=False,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"][0]
    n_x0 = len(input_ids)

    # 1) AR-supervision regions = assistant content (used for both AR loss
    #    targets and the candidate area for diffusion-block anchors).
    regions = _find_assistant_regions(input_ids, eos_token_id=eos_id)
    if not regions:
        raise ValueError(
            "Bonsai SDLM assistant-only supervision found no nonempty assistant "
            "content span after tokenization/truncation. Check the Qwen3 chat "
            "template and tokenizer marker IDs, retain an assistant response, "
            "or increase max_length; refusing whole-sequence supervision."
        )
    starts = [s for s, _ in regions]
    ends = [e for _, e in regions]
    total_resp_len = sum(e - s for s, e in regions)

    # 2) Decide how many diffusion blocks to spawn.
    headroom = max_length - n_x0 - 1  # -1 for the trailing PAD
    max_mask_len = min(total_resp_len * expected_mask_repeat_times, max(0, headroom))
    num_block = max(1, max_mask_len // block_size)

    # 3) Sample block anchor positions (one per block).
    mask_start_positions = _sample_positions_from_intervals(
        starts, ends, num_block, min_distance=block_size,
    )
    if not mask_start_positions:
        raise ValueError(
            "Bonsai SDLM found no assistant-region diffusion anchors; "
            "refusing to place a fallback anchor outside assistant content."
        )

    # 4) Build the mask-region tensors.
    sampled_start_tokens = input_ids[mask_start_positions]  # [num_block]
    n_mask = len(mask_start_positions) * block_size
    final_mask_ids = np.full(n_mask, mask_token_id, dtype=input_ids.dtype)
    final_mask_ids[::block_size] = sampled_start_tokens

    final_mask_targets = _build_mask_target(
        input_ids, mask_start_positions, block_size, eos_id
    )
    final_mask_position_ids = _build_mask_position_ids(mask_start_positions, block_size)

    # 5) AR labels: supervise assistant content, ignore the rest.
    ar_labels = np.full(n_x0, IGNORE_INDEX, dtype=np.int64)
    for s, e in regions:
        # Include the EOS at position e (so the AR loss also learns to stop).
        for p in range(s, min(e + 1, n_x0)):
            ar_labels[p] = int(input_ids[p])

    # 6) Concatenate.
    bridge = np.array([IGNORE_INDEX], dtype=np.int64)
    pad_in = np.array([pad_id], dtype=input_ids.dtype)
    pad_pos = np.array([int(final_mask_position_ids[-1]) + 1], dtype=np.int64)

    input_ids_final = np.concatenate([input_ids, final_mask_ids, pad_in])
    labels_final = np.concatenate([ar_labels, bridge, final_mask_targets])
    position_ids_final = np.concatenate(
        [np.arange(n_x0, dtype=np.int64), final_mask_position_ids, pad_pos]
    )

    return {
        "input_ids": torch.tensor(input_ids_final, dtype=torch.long),
        "labels": torch.tensor(labels_final, dtype=torch.long),
        "attention_mask": torch.ones(len(input_ids_final), dtype=torch.long),
        "position_ids": torch.tensor(position_ids_final, dtype=torch.long),
        "x0_len": torch.tensor(n_x0, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class BonsaiSDLMDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        block_size: int = 8,
        mask_token_id: int = BONSAI_MASK_TOKEN_ID,
        max_length: int = 4096,
        enable_thinking: bool = False,
        expected_mask_repeat_times: int = 2,
        max_samples: Optional[int] = None,
    ):
        if not os.path.isfile(data_path):
            raise FileNotFoundError(data_path)
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.mask_token_id = int(mask_token_id)
        self.max_length = int(max_length)
        self.enable_thinking = bool(enable_thinking)
        self.expected_mask_repeat_times = int(expected_mask_repeat_times)
        self.data_path = data_path

        # Index nonempty lines by byte offset. Parse records on demand so each
        # worker need not retain a full copy of the parsed dataset in memory.
        offsets: List[int] = []
        with open(data_path, "rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(pos)
                    if max_samples is not None and len(offsets) >= max_samples:
                        break
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self._fh = None  # opened lazily, per-process (safe across DataLoader fork)
        logger.info("Indexed %d SDLM samples (lazy) from %s",
                    len(self.offsets), data_path)

    def __len__(self) -> int:
        return int(len(self.offsets))

    def __getstate__(self):
        # Never pickle an open file handle to DataLoader workers.
        state = self.__dict__.copy()
        state["_fh"] = None
        return state

    def _get_fh(self):
        # Open a private file handle the first time this process touches data.
        # (After a DataLoader worker fork, self._fh is still None, so each
        # worker opens its own handle — no shared-offset races.)
        if self._fh is None:
            self._fh = open(self.data_path, "rb")
        return self._fh

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fh = self._get_fh()
        fh.seek(int(self.offsets[idx]))
        raw = fh.readline()
        record = json.loads(raw)
        messages = sharegpt_to_messages(record)
        return preprocess_bonsai_sdlm(
            messages=messages,
            tokenizer=self.tokenizer,
            block_size=self.block_size,
            mask_token_id=self.mask_token_id,
            expected_mask_repeat_times=self.expected_mask_repeat_times,
            max_length=self.max_length,
            enable_thinking=self.enable_thinking,
        )


# ---------------------------------------------------------------------------
# Collator (right-pad; position_ids in pad region = max_len+1 to avoid
# colliding with valid SDLM positions when the 4D mask is built)
# ---------------------------------------------------------------------------
@dataclass
class BonsaiSDLMCollator:
    pad_token_id: int = BONSAI_PAD_TOKEN_ID

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(int(f["input_ids"].size(0)) for f in features)
        bsz = len(features)
        input_ids = torch.full((bsz, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((bsz, max_len), IGNORE_INDEX, dtype=torch.long)
        attn = torch.zeros((bsz, max_len), dtype=torch.long)
        # pad position_ids to max_len+1 so they fall outside any real range
        pos = torch.full((bsz, max_len), max_len + 1, dtype=torch.long)
        x0 = torch.zeros((bsz,), dtype=torch.long)

        for i, f in enumerate(features):
            n = int(f["input_ids"].size(0))
            input_ids[i, :n] = f["input_ids"]
            labels[i, :n] = f["labels"]
            attn[i, :n] = 1
            pos[i, :n] = f["position_ids"]
            x0[i] = f["x0_len"]
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "position_ids": pos,
            "x0_len": x0,
        }
