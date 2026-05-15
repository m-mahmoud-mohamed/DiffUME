"""Single training-step routine combining diffusion + dual-contrastive losses.

Loss (per plan §2):

    L = λ_diff · (L_diff_q + L_diff_p)
      + λ_disc · ClipLoss(z_disc_q, z_disc_p, logit_scale)
      + λ_gen  · ClipLoss(z_gen_q,  z_gen_p,  logit_scale)

Each sample is processed twice (qry + pos), each going through one
**stochastic** forward of the DiffumeModel — so the contrastive
embeddings are read out from a slightly noisy hidden state, exactly
matching SDAR-VL's "denoise-then-readout" semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..diffusion.loss import diffusion_ce_loss_emrs
from ..losses.infonce import ClipLoss
from ..models.diffume_model import DiffumeModel


@dataclass
class LossWeights:
    diff: float = 1.0
    disc: float = 1.0
    gen: float = 1.0
    logit_scale: float = 50.0


class DiffumeStep:
    def __init__(self, model: DiffumeModel, weights: LossWeights | None = None):
        self.model = model
        self.weights = weights or LossWeights()
        self.disc_loss = ClipLoss(local_loss=True, gather_with_grad=True)
        self.gen_loss = ClipLoss(local_loss=True, gather_with_grad=True)

    def _side(self, side: dict, step: int):
        vk = {k: side[k] for k in ("pixel_values", "image_grid_thw") if k in side}
        out = self.model(
            input_ids=side["input_ids"],
            force_mask=side["force_mask"],
            readout_keep=side["readout_keep"],
            diffusable=side["diffusable"],
            assistant_starts=side["assistant_start"],
            assistant_ends=side["assistant_end"],
            step=step,
            vision_kwargs=vk,
            attention_mask_2d=side.get("attention_mask"),
        )
        l_diff = diffusion_ce_loss_emrs(
            logits=out["logits"],
            targets=side["input_ids"],
            rand_mask=out["rand_mask"],
            block_idx=out["block_idx"],
            block_lens=out["block_lens"],
        )
        z_disc, z_gen = self.model.get_embeddings(
            out["last_hidden_state"], side["input_ids"]
        )
        return l_diff, z_disc, z_gen

    def __call__(self, batch: dict, step: int) -> dict:
        l_diff_q, z_disc_q, z_gen_q = self._side(batch["qry"], step)
        l_diff_p, z_disc_p, z_gen_p = self._side(batch["pos"], step)

        ls = self.weights.logit_scale
        l_disc = self.disc_loss(z_disc_q, z_disc_p, logit_scale=ls)
        l_gen = self.gen_loss(z_gen_q, z_gen_p, logit_scale=ls)
        l_diff = l_diff_q + l_diff_p

        loss = (
            self.weights.diff * l_diff
            + self.weights.disc * l_disc
            + self.weights.gen * l_gen
        )
        return {
            "loss": loss,
            "loss_diff_q": l_diff_q.detach(),
            "loss_diff_p": l_diff_p.detach(),
            "loss_disc": l_disc.detach(),
            "loss_gen": l_gen.detach(),
        }
