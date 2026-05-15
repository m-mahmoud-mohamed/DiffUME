"""Block-wise discrete-diffusion forward noising  (SDAR-VL §3.1, §3.3).

Given an assistant span of length ``L_asst``, we partition it into
``B = ceil(L_asst / block_size)`` blocks (the **last** block may be short).
Per block we draw an independent Beta-distributed mask ratio ``t_b`` from
:class:`~diffume.diffusion.pbnc.PBNCSampler` (this is the ABNS step in §3.1).

For each token *i* in block *b* we draw ``u_i ~ U(0,1)`` and **mask** it
(replace its id with ``mask_id``) iff:

    u_i < t_b
    AND  position is `diffusable`     (i.e. it is in the assistant span and
                                       is not a readout token nor already
                                       force-masked)
    AND  position is **not** force-unmasked (readout tokens stay verbatim)

Force-masked positions (``<think>`` / ``</think>``) are clamped to ``mask_id``
unconditionally, regardless of ``t_b``.

Returns the noised ids together with the **realised** per-block mask ratio
``t_b' = ||m_b||_1 / L_b`` (EMRS, SDAR-VL §3.2 Eq. 6) which the loss uses
in place of the *intended* ``t_b``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .pbnc import PBNCSampler


@dataclass
class NoiseSchedule:
    block_size: int = 32


def _make_block_index(
    assistant_start: int, assistant_end: int, seq_len: int, block_size: int,
    device: torch.device,
) -> torch.LongTensor:
    """Return a (seq_len,) LongTensor, -1 outside assistant span, else block id."""
    block_idx = torch.full((seq_len,), -1, dtype=torch.long, device=device)
    n = assistant_end - assistant_start
    if n <= 0:
        return block_idx
    pos = torch.arange(n, device=device)
    block_idx[assistant_start:assistant_end] = pos // block_size
    return block_idx


def apply_blockwise_noise(
    input_ids: torch.LongTensor,            # (B, L)
    assistant_starts: torch.LongTensor,     # (B,)
    assistant_ends: torch.LongTensor,       # (B,)
    force_mask: torch.BoolTensor,           # (B, L) — must always be MASK
    readout_keep: torch.BoolTensor,         # (B, L) — must never be masked
    diffusable: torch.BoolTensor,           # (B, L)
    mask_id: int,
    sampler: PBNCSampler,
    step: int | None,
    schedule: NoiseSchedule,
    generator: Optional[torch.Generator] = None,
) -> dict:
    """Return dict with noised ids and per-token book-keeping for the loss.

    Returns
    -------
    {
        "noisy_ids":      LongTensor (B, L)   — input_ids with masks applied
        "is_masked":      BoolTensor (B, L)   — True where token was replaced
                                                by ``mask_id`` (excludes
                                                readout-keep positions)
        "block_idx":      LongTensor (B, L)   — block id ∈ [0, B_max), -1 outside
                                                assistant span
        "t_realised":     FloatTensor (B, B_max) — realised mask ratio per
                                                block (||m_b||_1 / L_b),
                                                NaN for blocks that have no
                                                diffusable tokens
        "block_lens":     LongTensor (B, B_max) — number of diffusable tokens
                                                per block (denominator L_b)
    }
    """
    B, L = input_ids.shape
    device = input_ids.device

    # Per-token block index.
    block_idx = torch.full((B, L), -1, dtype=torch.long, device=device)
    max_blocks = 0
    for i in range(B):
        bi = _make_block_index(
            int(assistant_starts[i]), int(assistant_ends[i]), L,
            schedule.block_size, device,
        )
        block_idx[i] = bi
        max_blocks = max(max_blocks, int(bi.max().item()) + 1 if (bi >= 0).any() else 0)
    max_blocks = max(max_blocks, 1)

    # Sample t_b per (batch, block).
    t_b = sampler.sample((B, max_blocks), step=step, device=device,
                         dtype=torch.float32)  # (B, max_blocks)

    # Random uniforms per token.
    u = torch.empty((B, L), device=device, dtype=torch.float32)
    if generator is not None:
        u.uniform_(generator=generator)
    else:
        u.uniform_()

    # Lookup t_b for each token; positions with block_idx==-1 use threshold 0.
    safe_idx = block_idx.clamp(min=0)
    t_per_tok = torch.gather(t_b, 1, safe_idx)             # (B, L)
    t_per_tok = torch.where(block_idx >= 0, t_per_tok,
                            torch.zeros_like(t_per_tok))

    # Decide which positions to mask.
    rand_mask = (u < t_per_tok) & diffusable
    final_mask = (rand_mask | force_mask) & ~readout_keep   # readout never masked
    noisy_ids = torch.where(final_mask, torch.full_like(input_ids, mask_id), input_ids)

    # Realised per-block mask ratio over diffusable positions.
    block_lens = torch.zeros((B, max_blocks), device=device, dtype=torch.long)
    masked_counts = torch.zeros((B, max_blocks), device=device, dtype=torch.long)
    for b in range(max_blocks):
        in_block = (block_idx == b)
        diff_in_block = in_block & diffusable
        block_lens[:, b] = diff_in_block.sum(dim=1)
        masked_counts[:, b] = (diff_in_block & rand_mask).sum(dim=1)

    denom = block_lens.clamp(min=1).float()
    t_realised = masked_counts.float() / denom
    t_realised = torch.where(block_lens > 0, t_realised,
                             torch.full_like(t_realised, float("nan")))

    return {
        "noisy_ids": noisy_ids,
        "is_masked": final_mask & ~force_mask,   # only stochastically-masked
        "block_idx": block_idx,
        "t_realised": t_realised,
        "block_lens": block_lens,
        "rand_mask": rand_mask,                  # diffusable positions actually masked
    }
