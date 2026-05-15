"""Tests for ClipLoss + readout."""

import torch

from diffume.losses.infonce import ClipLoss
from diffume.models.readout import get_embedding_reps


def test_cliploss_minimum_at_identity():
    torch.manual_seed(0)
    B, D = 8, 16
    z = torch.randn(B, D)
    z = z / z.norm(dim=-1, keepdim=True)
    loss_fn = ClipLoss()
    # Same matrix → diagonal positives → low loss.
    same = loss_fn(z, z, logit_scale=50.0).item()
    # Random pair → higher loss.
    z2 = torch.randn(B, D)
    z2 = z2 / z2.norm(dim=-1, keepdim=True)
    rand = loss_fn(z, z2, logit_scale=50.0).item()
    assert same < rand


def test_get_embedding_reps_picks_last_occurrence():
    B, L, D = 2, 5, 4
    h = torch.randn(B, L, D)
    ids = torch.tensor([[0, 7, 1, 7, 2], [7, 1, 1, 1, 1]])
    reps = get_embedding_reps(h, ids, embedding_token_id=7, normalize=False)
    assert torch.allclose(reps[0], h[0, 3])     # last occurrence in row 0 is at 3
    assert torch.allclose(reps[1], h[1, 0])     # only occurrence in row 1 is at 0


def test_get_embedding_reps_fallback_when_absent():
    B, L, D = 1, 4, 3
    h = torch.randn(B, L, D)
    ids = torch.tensor([[0, 1, 2, 3]])
    reps = get_embedding_reps(h, ids, embedding_token_id=99, normalize=False)
    assert torch.allclose(reps[0], h[0, -1])    # falls back to last position
