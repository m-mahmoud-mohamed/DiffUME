"""Block-causal attention mask builder  (SDAR-VL §3.1).

The attention pattern that converts an autoregressive Qwen2.5-VL into a
*Block Discrete Denoising Diffusion* model (BD3) is:

    * **Within** an assistant block ⇢ bidirectional (full attention).
    * **Across** blocks ⇢ strictly **causal** w.r.t. block id.
    * The **prefix** (system + user, position ``i < assistant_start``) is
      treated as block ``-1`` and is fully visible to every later token but
      it itself attends only autoregressively (standard causal LM prefix
      behaviour).

Returns an additive float mask of shape ``(B, 1, L, L)`` with values in
``{0, -inf}`` ready to be added to the attention logits — exactly the
format Qwen2.5-VL's eager / sdpa kernels expect.
"""

from __future__ import annotations

from typing import Optional

import torch


def build_block_causal_mask(
    block_idx: torch.LongTensor,    # (B, L) — -1 outside assistant span, else block id
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return additive (B, 1, L, L) attention mask with 0 / -inf entries."""
    B, L = block_idx.shape
    device = device or block_idx.device

    # Treat prefix tokens (block_idx == -1) as a single virtual block "-1"
    # which is *earlier* than every assistant block. Use -1 directly in the
    # comparison: assistant blocks 0,1,... are all > -1.
    bi = block_idx                                         # (B, L)
    bi_q = bi.unsqueeze(2)                                 # (B, L, 1)
    bi_k = bi.unsqueeze(1)                                 # (B, 1, L)

    # Standard causal mask over absolute positions for the prefix tokens.
    pos = torch.arange(L, device=device)
    causal_pos = pos.unsqueeze(1) >= pos.unsqueeze(0)      # (L, L) — q can see k
    causal_pos = causal_pos.unsqueeze(0).expand(B, L, L)   # (B, L, L)

    # Block rule:
    #   query in assistant block bq, key in assistant block bk:
    #       allowed iff bk <= bq (block-causal) — same block ⇒ bidirectional.
    #   query in prefix (bq == -1): allowed iff key in prefix AND key_pos<=q_pos
    #   query in assistant: prefix keys (bk==-1) are ALWAYS visible.
    q_in_asst = bi_q >= 0                                  # (B, L, 1)
    k_in_asst = bi_k >= 0                                  # (B, 1, L)

    # Default: not allowed.
    allowed = torch.zeros((B, L, L), dtype=torch.bool, device=device)

    # Case A: both in prefix ⇒ standard causal.
    both_prefix = (~q_in_asst) & (~k_in_asst)
    allowed = allowed | (both_prefix & causal_pos)

    # Case B: query in assistant, key in prefix ⇒ always visible.
    asst_to_prefix = q_in_asst & (~k_in_asst)
    allowed = allowed | asst_to_prefix

    # Case C: both in assistant ⇒ block-causal (bk <= bq).
    both_asst = q_in_asst & k_in_asst
    block_causal = bi_k <= bi_q                             # (B, L, L) via broadcast
    allowed = allowed | (both_asst & block_causal)

    # Case D: query in prefix, key in assistant ⇒ never (already false).

    # Convert to additive float mask.
    neg_inf = torch.finfo(dtype).min
    mask = torch.where(
        allowed,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), neg_inf, dtype=dtype, device=device),
    )
    return mask.unsqueeze(1)                                # (B, 1, L, L)
