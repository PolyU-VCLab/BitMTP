"""SDLM 4D attention-mask builder (BitMTP-compatible, model-agnostic).

The mask is determined by position_ids and x0_len, independently of the
underlying model family.

Three regions of the SDLM mask (see SDLM paper Figure 2(a)):
    * Causal block (top-left): standard causal attn within x0 prefix
    * Mutual block (bottom-right): all queries in the same diffusion block
      attend to all keys in the same block (bidirectional when
      causal_attn=False, causal when True — "T2" variant)
    * Prefix block (bottom-left): every diffusion-block query attends to
      its own causal prefix (length = block-anchor position in position_ids)
"""
from __future__ import annotations

import torch


def find_prefix_seq_length_by_pe(pe: torch.Tensor) -> torch.Tensor:
    """Detect x0 length by scanning for the first position-id drop.

    Args:
        pe: [batch_size, seq_len]
    Returns:
        [batch_size] tensor; -1 if no drop found (= all-x0 sample).
    """
    batch_size, seq_len = pe.shape
    prev = pe[:, :-1]
    curr = pe[:, 1:]
    drop_mask = curr < prev  # [B, seq-1]
    out = torch.full((batch_size,), -1, dtype=torch.long)
    for b in range(batch_size):
        drop_pos = torch.nonzero(drop_mask[b], as_tuple=False)
        if drop_pos.numel() > 0:
            out[b] = drop_pos[0].item() + 1
    return out


def create_block_diff_mask_by_pe_4d(
    block_size: int,
    x0_len_list: torch.Tensor,
    position_ids: torch.Tensor,
    causal_attn: bool = False,
):
    """Build the [B, 1, L, L] 4D SDLM attention mask.

    Args:
        block_size: diffusion block size D
        x0_len_list: [B] per-sample AR-prefix length
        position_ids: [B, L]
        causal_attn: if True, mutual block is causal (T2 variant);
                     if False, mutual block is bidirectional (original SDLM).
    Returns:
        (float_mask, bool_mask) both shaped [B, 1, L, L]; float_mask has
        0.0 where attention is allowed and -inf where blocked.
    """
    batch_size, seq_len = position_ids.shape
    device = position_ids.device

    q_idx = torch.arange(seq_len, device=device).view(1, seq_len, 1)
    kv_idx = torch.arange(seq_len, device=device).view(1, 1, seq_len)

    x0_len = x0_len_list.view(batch_size, 1, 1)
    x0_flag_q = q_idx < x0_len
    x0_flag_kv = kv_idx < x0_len

    q_block_idx = (q_idx - x0_len) // block_size
    kv_block_idx = (kv_idx - x0_len) // block_size

    # Causal block (top-left)
    block_causal = x0_flag_q & x0_flag_kv & (q_idx >= kv_idx)

    # Mutual block (bottom-right): same diffusion block, optional causal
    mutual_condition = (q_idx >= kv_idx) if causal_attn else torch.ones_like(q_idx, dtype=torch.bool)
    block_mutual = (
        ~x0_flag_q & ~x0_flag_kv
        & (q_block_idx == kv_block_idx)
        & mutual_condition
    )

    # Prefix block (bottom-left): query in diff-block sees its own prefix.
    # Each diffusion block's anchor position_id == the prefix length it can see.
    q_blk = torch.div(q_idx - x0_len, block_size, rounding_mode="floor")
    q_blk_start = (
        x0_len_list.view(batch_size, 1) + q_blk[:, :, 0] * block_size
    ).clamp(min=0, max=seq_len - 1)  # [B, L]
    prefix_len = position_ids.gather(1, q_blk_start).unsqueeze(2)  # [B, L, 1]
    block_prefix = (~x0_flag_q & x0_flag_kv) & (kv_idx < prefix_len)

    final_mask = block_causal | block_mutual | block_prefix
    customized = torch.full_like(final_mask, float("-inf"), dtype=torch.bfloat16)
    customized.masked_fill_(final_mask, 0.0)
    return customized.unsqueeze(1).to(device=device), final_mask.unsqueeze(1).to(device=device)
