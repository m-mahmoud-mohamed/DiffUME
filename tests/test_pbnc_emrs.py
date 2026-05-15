"""Tests for PBNC and EMRS-corrected diffusion loss."""

import torch

from diffume.diffusion.loss import diffusion_ce_loss_emrs
from diffume.diffusion.noising import NoiseSchedule, apply_blockwise_noise
from diffume.diffusion.pbnc import PBNCConfig, PBNCSampler


def test_pbnc_alpha_beta_lerp():
    s = PBNCSampler(PBNCConfig(
        scheduler_type="beta", target_mean=0.8,
        start_C=2.0, end_C=50.0, warmup_steps=1000,
    ))
    a0, b0 = s.get_alpha_beta(0)
    a1, b1 = s.get_alpha_beta(1000)
    # Mean held fixed at target_mean (0.8) at every step.
    assert abs(a0 / (a0 + b0) - 0.8) < 1e-6
    assert abs(a1 / (a1 + b1) - 0.8) < 1e-6
    # Concentration ramps start_C → end_C across warmup_steps.
    assert abs((a0 + b0) - 2.0) < 1e-6
    assert abs((a1 + b1) - 50.0) < 1e-6


def test_pbnc_sample_in_unit_interval():
    s = PBNCSampler()
    t = s.sample((4, 8), step=300)
    assert torch.all((t >= 0) & (t <= 1))


def test_pbnc_uniform_before_warmup():
    s = PBNCSampler(PBNCConfig(scheduler_type="beta", warmup_steps=100))
    # step <= 0 ⇒ uniform fallback (matches SDAR-VL P-rectify behaviour).
    t = s.sample((1024,), step=0)
    assert torch.all((t >= 0) & (t <= 1))
    # Uniform mean should be ~0.5, far from target_mean=0.8.
    assert abs(t.mean().item() - 0.5) < 0.1


def test_clamp_scheduler_bounds():
    s = PBNCSampler(PBNCConfig(scheduler_type="clamp", noise_min=0.2, noise_max=0.7))
    t = s.sample((10000,), step=10)
    assert t.min().item() >= 0.2 - 1e-6
    assert t.max().item() <= 0.7 + 1e-6


def test_apply_blockwise_noise_respects_constraints():
    torch.manual_seed(0)
    B, L = 2, 16
    input_ids = torch.arange(L).unsqueeze(0).repeat(B, 1) + 1000
    asst_s = torch.tensor([4, 4])
    asst_e = torch.tensor([16, 16])
    force_mask = torch.zeros((B, L), dtype=torch.bool)
    force_mask[:, 5] = True            # one force-mask position
    readout_keep = torch.zeros((B, L), dtype=torch.bool)
    readout_keep[:, 15] = True         # last token is gen_emb (never mask)
    diffusable = torch.zeros((B, L), dtype=torch.bool)
    diffusable[:, 4:15] = True         # span [4..14] is diffusable
    diffusable[:, 5] = False           # remove force-masked from diffusable
    sampler = PBNCSampler(PBNCConfig(
        scheduler_type="beta", target_mean=0.9,
        start_C=1000, end_C=1000, warmup_steps=1,
    ))
    out = apply_blockwise_noise(
        input_ids=input_ids, assistant_starts=asst_s, assistant_ends=asst_e,
        force_mask=force_mask, readout_keep=readout_keep, diffusable=diffusable,
        mask_id=999, sampler=sampler, step=10,
        schedule=NoiseSchedule(block_size=8),
    )
    # readout token never replaced.
    assert (out["noisy_ids"][:, 15] == input_ids[:, 15]).all()
    # force-masked token always replaced.
    assert (out["noisy_ids"][:, 5] == 999).all()
    # block_idx layout: prefix=-1, span starts at 4.
    assert (out["block_idx"][:, :4] == -1).all()
    assert out["block_idx"][0, 4].item() == 0


def test_emrs_loss_zero_when_no_masks():
    torch.manual_seed(0)
    B, L, V = 2, 8, 32
    logits = torch.randn(B, L, V, requires_grad=True)
    targets = torch.randint(0, V, (B, L))
    rand_mask = torch.zeros(B, L, dtype=torch.bool)
    block_idx = torch.zeros(B, L, dtype=torch.long)
    block_lens = torch.zeros(B, 1, dtype=torch.long)
    loss = diffusion_ce_loss_emrs(logits, targets, rand_mask, block_idx, block_lens)
    assert loss.item() == 0.0


def test_emrs_loss_finite_with_masks():
    torch.manual_seed(0)
    B, L, V = 2, 8, 32
    logits = torch.randn(B, L, V, requires_grad=True)
    targets = torch.randint(0, V, (B, L))
    rand_mask = torch.zeros(B, L, dtype=torch.bool)
    rand_mask[:, 2:4] = True
    block_idx = torch.full((B, L), -1, dtype=torch.long)
    block_idx[:, 0:8] = 0
    block_lens = torch.full((B, 1), 8, dtype=torch.long)
    loss = diffusion_ce_loss_emrs(logits, targets, rand_mask, block_idx, block_lens)
    assert torch.isfinite(loss).all()
    assert loss.requires_grad
