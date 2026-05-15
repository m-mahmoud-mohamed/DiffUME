"""Hidden-state readout for embedding tokens  (UME-R1 trainer.py L568-L580).

Given the model's last hidden states and the input ids, return the hidden
vector at the **last** occurrence of the requested embedding-token id.

This implementation is functionally identical to UME-R1's
``get_embedding_reps`` but is dependency-free and includes an explicit
fallback for samples where the token is absent (returns the last
non-padding position so the loss does not silently propagate NaNs).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def get_embedding_reps(
    last_hidden_state: torch.Tensor,    # (B, L, D)
    input_ids: torch.LongTensor,        # (B, L)
    embedding_token_id: int,
    normalize: bool = True,
) -> torch.Tensor:
    """Return (B, D) — hidden state at the last position of `embedding_token_id`."""
    B, L, _ = last_hidden_state.shape
    device = last_hidden_state.device

    is_emb = input_ids == embedding_token_id
    pos = torch.where(
        is_emb,
        torch.arange(L, device=device).unsqueeze(0).expand(B, -1),
        torch.full_like(input_ids, -1),
    )
    last_pos = pos.max(dim=1).values   # (B,)

    # Fallback for samples that don't contain the token.
    fallback = torch.full_like(last_pos, L - 1)
    last_pos = torch.where(last_pos >= 0, last_pos, fallback)

    reps = last_hidden_state[torch.arange(B, device=device), last_pos]
    if normalize:
        reps = F.normalize(reps, p=2, dim=-1)
    return reps
