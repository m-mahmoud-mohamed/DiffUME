"""Noise schedulers for block-wise discrete diffusion (SDAR-VL §3.1, §3.3).

Three scheduler types are supported, mirroring the SDAR-VL reference
implementation (`finetuning_args.noise_scheduler_type`):

* ``"uniform"`` — ``t_b ~ U(0, 1)`` independently per block.

* ``"beta"`` — *Progressive Beta Noise Curriculum* (PBNC, also called
  the "P-rectify" scheduler).  The target mean ``μ`` is **held fixed**
  at ``scheduler_target_mean`` (default 0.8); the concentration ``C``
  ramps **linearly** from ``scheduler_start_C`` to ``scheduler_end_C``
  over ``scheduler_warmup_steps`` optimiser steps.  Before the first
  warmup step the schedule falls back to **uniform** sampling — the
  ramp is what makes the curriculum "progressive".

      C_τ = start_C + (end_C − start_C) · min(step / warmup_steps, 1)
      α   = μ · C_τ
      β   = (1 − μ) · C_τ
      t_b ~ Beta(α, β)

* ``"clamp"`` — uniform sample then clamped to
  ``[noise_min, noise_max]`` (used to restrict masking to a fixed
  band).

Reference defaults follow the SDAR-VL training YAMLs:
``target_mean=0.8``, ``start_C=2.0``, ``end_C=50.0``,
``warmup_steps=8090``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.distributions import Beta


SchedulerType = Literal["uniform", "beta", "clamp"]


@dataclass
class PBNCConfig:
    """Unified noise-scheduler config (covers uniform / beta / clamp).

    The name ``PBNCConfig`` is kept for backward compatibility with the
    earlier DiffUME code; the dataclass now also drives the uniform and
    clamp schedulers.
    """

    scheduler_type: SchedulerType = "beta"
    # Beta (P-rectify) — fixed mean, ramping concentration.
    target_mean: float = 0.8
    start_C: float = 2.0
    end_C: float = 50.0
    warmup_steps: int = 8090
    # Clamp scheduler bounds (only used when scheduler_type == "clamp").
    noise_min: float = 0.0
    noise_max: float = 1.0


class PBNCSampler:
    """Block-wise noise sampler.

    The class name is preserved for backward compatibility, but the
    sampler now dispatches to ``uniform`` / ``beta`` / ``clamp`` based on
    ``cfg.scheduler_type``.

    The ``step`` argument (current global optimiser step) drives the C
    ramp for the Beta scheduler.  Pass ``step=None`` (or any value >=
    ``warmup_steps``) to sample at the final, fully-peaked Beta.
    """

    def __init__(self, cfg: PBNCConfig | None = None):
        self.cfg = cfg or PBNCConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_alpha_beta(self, step: int | None = None) -> tuple[float, float]:
        """Return (α, β) of the Beta distribution at the given step."""
        cfg = self.cfg
        if step is None:
            c = cfg.end_C
        else:
            warm = max(1, int(cfg.warmup_steps))
            t = min(max(int(step), 0), warm) / warm
            c = cfg.start_C + (cfg.end_C - cfg.start_C) * t
        mu = float(cfg.target_mean)
        alpha = max(mu * c, 1e-6)
        beta = max((1.0 - mu) * c, 1e-6)
        return alpha, beta

    def sample(
        self,
        shape: tuple[int, ...],
        step: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return a tensor of independent ``t_b`` values of the given shape."""
        cfg = self.cfg
        if cfg.scheduler_type == "uniform":
            return torch.rand(shape, device=device, dtype=dtype)

        if cfg.scheduler_type == "clamp":
            t = torch.rand(shape, device=device, dtype=dtype)
            return t.clamp(min=cfg.noise_min, max=cfg.noise_max)

        # "beta" (PBNC / P-rectify):
        # before the first warmup step the SDAR-VL training procedure samples
        # *uniformly* — the curriculum only kicks in once the C ramp begins.
        if step is not None and int(step) <= 0:
            return torch.rand(shape, device=device, dtype=dtype)

        alpha, beta = self.get_alpha_beta(step)
        a = torch.full(shape, alpha, device=device, dtype=dtype)
        b = torch.full(shape, beta, device=device, dtype=dtype)
        return Beta(a, b).sample()
