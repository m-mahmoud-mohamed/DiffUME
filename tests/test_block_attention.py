"""Tests for the block-causal attention mask builder."""

import torch

from diffume.models.attention_block import build_block_causal_mask


def test_prefix_only_mask_is_causal():
    block_idx = torch.full((1, 5), -1, dtype=torch.long)
    m = build_block_causal_mask(block_idx).squeeze(0).squeeze(0)  # (5,5)
    # Causal: lower-triangular zeros, upper-triangular -inf.
    for i in range(5):
        for j in range(5):
            if j <= i:
                assert m[i, j].item() == 0.0
            else:
                assert m[i, j].item() < -1e30


def test_block_causal_within_block_bidirectional():
    # 2 prefix tokens (block -1) + 4 assistant tokens split into 2 blocks of 2.
    block_idx = torch.tensor([[-1, -1, 0, 0, 1, 1]], dtype=torch.long)
    m = build_block_causal_mask(block_idx).squeeze(0).squeeze(0)
    # Within block 0 (positions 2,3): bidirectional ⇒ both 0.
    assert m[2, 3].item() == 0.0
    assert m[3, 2].item() == 0.0
    # Within block 1 (positions 4,5): bidirectional.
    assert m[4, 5].item() == 0.0
    assert m[5, 4].item() == 0.0
    # Block 0 cannot see block 1.
    assert m[2, 4].item() < -1e30
    assert m[3, 5].item() < -1e30
    # Block 1 CAN see block 0.
    assert m[4, 2].item() == 0.0
    assert m[5, 3].item() == 0.0
    # Assistant always sees prefix.
    assert m[2, 0].item() == 0.0
    assert m[5, 1].item() == 0.0
    # Prefix never sees assistant.
    assert m[0, 2].item() < -1e30
    assert m[1, 4].item() < -1e30
    # Prefix-prefix is causal.
    assert m[0, 1].item() < -1e30
    assert m[1, 0].item() == 0.0
