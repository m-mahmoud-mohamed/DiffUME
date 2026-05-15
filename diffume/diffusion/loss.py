"""EMRS-corrected block diffusion loss  (SDAR-VL §3.2, Eq. 6-7).

For each masked token ``i`` in block ``b`` the masked-prediction
cross-entropy ``CE_i = −log p_θ(x_i | x_noisy)`` is summed within the block
and **divided by the realised** mask-fraction ``t_b' = ||m_b||_1 / L_b``
(NOT by the intended ``t_b``).  Lemma 4 of the paper proves this estimator
is unbiased for the ideal NELBO ``L*(θ) = E[ −ℓ_b / t_b' ]`` while the
naive ``1 / t_b`` weighting is biased (Lemma 3, Eq. 28).

Final per-sample loss is the mean over blocks; final batch loss is the
mean over samples.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def diffusion_ce_loss_emrs(
    logits: torch.Tensor,                 # (B, L, V) on assistant tokens (full L OK; we mask via rand_mask)
    targets: torch.Tensor,                # (B, L) — original token ids
    rand_mask: torch.BoolTensor,          # (B, L) — positions stochastically masked
    block_idx: torch.LongTensor,          # (B, L) — -1 outside assistant span, else block id
    block_lens: torch.LongTensor,         # (B, B_max) — diffusable tokens per block
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    """Return scalar loss following SDAR-VL Eq. 7 with EMRS normalisation."""
    B, L, V = logits.shape
    B_max = block_lens.shape[1]
    device = logits.device

    # Per-token CE only at randomly-masked positions; zero elsewhere.
    ce = F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        reduction="none",
    ).reshape(B, L)
    ce = torch.where(rand_mask, ce, torch.zeros_like(ce))

    # Sum CE per block.
    block_ce = torch.zeros((B, B_max), device=device, dtype=ce.dtype)
    safe_idx = block_idx.clamp(min=0)
    in_span = (block_idx >= 0).to(ce.dtype)
    block_ce.scatter_add_(1, safe_idx, ce * in_span)

    # Number of stochastically masked tokens per block (the realised mask count).
    masked_counts = torch.zeros((B, B_max), device=device, dtype=ce.dtype)
    one_per_masked = rand_mask.to(ce.dtype) * in_span
    masked_counts.scatter_add_(1, safe_idx, one_per_masked)

    # EMRS realised ratio  t_b' = masked_counts / block_lens.
    block_lens_f = block_lens.to(ce.dtype)
    t_prime = masked_counts / block_lens_f.clamp(min=1.0)

    # ℓ_b / t_b'   (per-token-mean within block / t_b'  ==  sum / masked_counts)
    # We use the formulation ℓ_b / t_b' with ℓ_b = block_ce / block_lens
    # (matches Eq. 6 with per-block average masked-CE divided by realised ratio).
    per_block = (block_ce / block_lens_f.clamp(min=1.0)) / t_prime.clamp(min=eps)

    # Only count blocks that have at least one diffusable token AND
    # at least one stochastically masked token (else loss is undefined / 0).
    valid = (block_lens > 0) & (masked_counts > 0)
    per_block = torch.where(valid, per_block, torch.zeros_like(per_block))
    n_valid = valid.sum(dim=1).clamp(min=1).to(ce.dtype)
    per_sample = per_block.sum(dim=1) / n_valid

    if reduction == "mean":
        return per_sample.mean()
    elif reduction == "sum":
        return per_sample.sum()
    return per_sample
