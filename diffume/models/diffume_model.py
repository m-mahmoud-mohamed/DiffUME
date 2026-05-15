"""DiffUME model wrapper around Qwen2.5-VL.

Design philosophy (per project plan §6 Phase 2):

    * **No** vendored copy of `modeling_qwen2_5_vl.py` — we keep the model
      pristine and just *override the attention mask* by passing a 4D
      additive mask on every forward call.  Transformers ≥4.45 detects 4D
      masks and short-circuits its own causal-mask construction
      (`_update_causal_mask` returns the mask as-is).
    * **Block-causal attention** is realised by the
      :func:`~diffume.models.attention_block.build_block_causal_mask`
      helper, which takes a per-token ``block_idx`` tensor.
    * **Discrete diffusion forward**: at training time
      :meth:`forward_train` samples per-block noise via PBNC, masks the
      assistant span, runs Qwen2.5-VL with the block-causal mask, and
      returns logits + bookkeeping tensors expected by
      :func:`diffume.diffusion.loss.diffusion_ce_loss_emrs`.
    * **Embedding readout**: :meth:`encode` returns ``(z_disc, z_gen)`` —
      the L2-normalised hidden states at the last ``<disc_emb>`` and
      ``<gen_emb>`` positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..diffusion.noising import NoiseSchedule, apply_blockwise_noise
from ..diffusion.pbnc import PBNCConfig, PBNCSampler
from .attention_block import build_block_causal_mask
from .readout import get_embedding_reps


@dataclass
class DiffumeConfig:
    block_size: int = 4
    pbnc: PBNCConfig = None
    backbone_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    def __post_init__(self):
        if self.pbnc is None:
            self.pbnc = PBNCConfig()


class DiffumeModel(nn.Module):
    """Wraps a `Qwen2_5_VLForConditionalGeneration` instance."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        mask_id: int,
        gen_emb_id: int,
        disc_emb_id: int,
        cfg: DiffumeConfig | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg or DiffumeConfig()
        self.mask_id = int(mask_id)
        self.gen_emb_id = int(gen_emb_id)
        self.disc_emb_id = int(disc_emb_id)
        self.sampler = PBNCSampler(self.cfg.pbnc)
        self.schedule = NoiseSchedule(block_size=self.cfg.block_size)

    # ----------------------------- helpers -----------------------------

    def _block_idx_from_spans(
        self,
        seq_len: int,
        assistant_starts: torch.LongTensor,
        assistant_ends: torch.LongTensor,
        device: torch.device,
    ) -> torch.LongTensor:
        B = assistant_starts.numel()
        block_idx = torch.full((B, seq_len), -1, dtype=torch.long, device=device)
        for i in range(B):
            s = int(assistant_starts[i])
            e = int(assistant_ends[i])
            n = e - s
            if n > 0:
                block_idx[i, s:e] = (
                    torch.arange(n, device=device) // self.cfg.block_size
                )
        return block_idx

    def _vision_kwargs(self, batch: dict) -> dict:
        out = {}
        for k in ("pixel_values", "image_grid_thw", "pixel_values_videos",
                  "video_grid_thw", "second_per_grid_ts"):
            if batch.get(k) is not None:
                out[k] = batch[k]
        return out

    def _compute_position_ids(
        self,
        input_ids: torch.LongTensor,
        vision_kwargs: dict,
        attention_mask_2d: "torch.Tensor | None",
    ) -> "tuple[torch.Tensor | None, torch.Tensor | None]":
        """Return ``(position_ids, mm_token_type_ids)`` for multimodal inputs.

        For pure-text batches both are ``None`` (1D auto-positions are fine).
        For multimodal batches we compute the 3D RoPE positions ourselves via
        ``backbone.model.get_rope_index`` *before* the 4D block-causal mask is
        built (``get_rope_index`` only accepts a 2D attention mask).  We also
        return ``mm_token_type_ids`` so the caller can forward it to
        ``backbone.model`` — transformers >=5.x's ``Qwen2VLModel`` requires it
        whenever ``image_grid_thw``/``video_grid_thw`` is present.
        """
        has_images = vision_kwargs.get("pixel_values") is not None
        has_videos = vision_kwargs.get("pixel_values_videos") is not None
        if not (has_images or has_videos):
            return None, None  # pure-text batch — 1D auto-positions are fine

        cfg = self.backbone.config
        mm_token_type_ids = torch.zeros(
            input_ids.shape, dtype=torch.int, device=input_ids.device
        )
        if has_images:
            mm_token_type_ids[input_ids == cfg.image_token_id] = 1
        if has_videos:
            mm_token_type_ids[input_ids == cfg.video_token_id] = 2

        # ``get_rope_index`` lives on the inner ``*VLModel`` (Qwen2VLModel /
        # Qwen2_5_VLModel), NOT on the ``*ForConditionalGeneration`` wrapper.
        inner = getattr(self.backbone, "model", self.backbone)
        get_rope_index = getattr(inner, "get_rope_index", None)
        if get_rope_index is None:
            return None, mm_token_type_ids

        kwargs = dict(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=vision_kwargs.get("image_grid_thw"),
            video_grid_thw=vision_kwargs.get("video_grid_thw"),
            attention_mask=attention_mask_2d,  # 2D only — safe here
        )
        if vision_kwargs.get("second_per_grid_ts") is not None:
            kwargs["second_per_grid_ts"] = vision_kwargs["second_per_grid_ts"]
        position_ids, _ = get_rope_index(**kwargs)
        return position_ids, mm_token_type_ids  # (3, B, L), (B, L)

    # ----------------------------- training forward --------------------

    def forward(self, **kwargs) -> dict:
        """FSDP-compatible entry point — delegates to forward_train.

        FSDP hooks fire on ``forward()`` (called via ``model(...)``), which
        gathers sharded parameters back to their full shape before the actual
        computation.  Calling ``forward_train`` directly bypasses these hooks,
        leaving the embedding weight as a 1-D shard and crashing with
        ``RuntimeError: 'weight' must be 2-D``.  This thin wrapper ensures
        FSDP always sees the canonical entry point.
        """
        return self.forward_train(**kwargs)

    def forward_train(
        self,
        *,
        input_ids: torch.LongTensor,           # (B, L)
        force_mask: torch.BoolTensor,          # (B, L)
        readout_keep: torch.BoolTensor,        # (B, L)
        diffusable: torch.BoolTensor,          # (B, L)
        assistant_starts: torch.LongTensor,    # (B,)
        assistant_ends: torch.LongTensor,      # (B,)
        step: int | None = None,               # current global optimiser step
        vision_kwargs: dict | None = None,
        attention_mask_2d: torch.Tensor | None = None,   # for padding only
    ) -> dict:
        """Single noisy forward pass; returns logits + bookkeeping.

        The caller is responsible for combining the returned logits with
        :func:`diffume.diffusion.loss.diffusion_ce_loss_emrs` and the
        contrastive heads (see :class:`diffume.training.trainer_step`).
        """
        B, L = input_ids.shape
        device = input_ids.device
        vision_kwargs = vision_kwargs or {}

        # 1) Apply block-wise noise (and PBNC for t_b).
        noised = apply_blockwise_noise(
            input_ids=input_ids,
            assistant_starts=assistant_starts,
            assistant_ends=assistant_ends,
            force_mask=force_mask,
            readout_keep=readout_keep,
            diffusable=diffusable,
            mask_id=self.mask_id,
            sampler=self.sampler,
            step=step,
            schedule=self.schedule,
        )
        block_idx = noised["block_idx"]

        # 2) Pre-compute 3D multimodal RoPE position_ids using the *clean* input_ids
        # and the 2D padding mask.  This must happen before we build the 4D block-causal
        # mask because Qwen2.5-VL's get_rope_index only accepts 2D attention masks.
        # Passing the result explicitly to backbone.model() bypasses compute_3d_position_ids
        # (which would skip get_rope_index for 4D masks) and gives image tokens correct
        # temporal/height/width RoPE positions instead of the 1D fallback.
        position_ids, mm_token_type_ids = self._compute_position_ids(
            input_ids, vision_kwargs, attention_mask_2d,
        )

        # 3) Build block-causal additive mask.
        param_dtype = next(self.backbone.parameters()).dtype
        attn_mask_4d = build_block_causal_mask(
            block_idx, dtype=param_dtype, device=device,
        )  # (B, 1, L, L)

        # If a 2D padding mask was supplied, add -inf for padding keys.
        if attention_mask_2d is not None:
            pad_keys = (attention_mask_2d == 0).unsqueeze(1).unsqueeze(1)  # (B,1,1,L)
            neg_inf = torch.finfo(param_dtype).min
            attn_mask_4d = attn_mask_4d.masked_fill(pad_keys, neg_inf)

        # 4) Forward via backbone.model so we get last_hidden_state directly.
        # Calling backbone.model (Qwen2_5_VLModel) returns Qwen2_5_VLModelOutputWithPast
        # which always has last_hidden_state, whereas the CausalLM wrapper's hidden_states
        # field is None in transformers >=5.x (FlashAttentionKwargs doesn't include
        # output_hidden_states, so the language model never collects them).
        extra = {}
        if mm_token_type_ids is not None:
            extra["mm_token_type_ids"] = mm_token_type_ids
        model_out = self.backbone.model(
            input_ids=noised["noisy_ids"],
            attention_mask=attn_mask_4d,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
            **vision_kwargs,
            **extra,
        )
        last_hidden_state = model_out.last_hidden_state   # (B, L, D)
        logits = self.backbone.lm_head(last_hidden_state)  # (B, L, V)

        return {
            "logits": logits,                             # (B, L, V)
            "last_hidden_state": last_hidden_state,       # (B, L, D)
            "noisy_ids": noised["noisy_ids"],
            "rand_mask": noised["rand_mask"],
            "block_idx": block_idx,
            "block_lens": noised["block_lens"],
            "t_realised": noised["t_realised"],
        }

    def get_embeddings(
        self,
        last_hidden_state: torch.Tensor,
        input_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (z_disc, z_gen), each L2-normalised, shape (B, D)."""
        z_disc = get_embedding_reps(last_hidden_state, input_ids, self.disc_emb_id)
        z_gen = get_embedding_reps(last_hidden_state, input_ids, self.gen_emb_id)
        return z_disc, z_gen

    # ----------------------------- inference (clean encode) ------------

    @torch.no_grad()
    def encode(
        self,
        *,
        input_ids: torch.LongTensor,
        assistant_starts: torch.LongTensor,
        assistant_ends: torch.LongTensor,
        vision_kwargs: dict | None = None,
        attention_mask_2d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a *clean* (no-mask) block-causal forward and return (z_disc, z_gen).

        Used by Phase 4's ``embed_disc`` inference path: the assistant span
        already contains the full reasoning + readout tokens (as in the
        training corpus), so a single forward gives the right embeddings.
        """
        device = input_ids.device
        vision_kwargs = vision_kwargs or {}
        block_idx = self._block_idx_from_spans(
            input_ids.size(1), assistant_starts, assistant_ends, device,
        )
        position_ids, mm_token_type_ids = self._compute_position_ids(
            input_ids, vision_kwargs, attention_mask_2d,
        )
        param_dtype = next(self.backbone.parameters()).dtype
        attn_mask_4d = build_block_causal_mask(block_idx, dtype=param_dtype, device=device)
        if attention_mask_2d is not None:
            pad_keys = (attention_mask_2d == 0).unsqueeze(1).unsqueeze(1)
            neg_inf = torch.finfo(param_dtype).min
            attn_mask_4d = attn_mask_4d.masked_fill(pad_keys, neg_inf)

        extra = {}
        if mm_token_type_ids is not None:
            extra["mm_token_type_ids"] = mm_token_type_ids
        model_out = self.backbone.model(
            input_ids=input_ids,
            attention_mask=attn_mask_4d,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
            **vision_kwargs,
            **extra,
        )
        return self.get_embeddings(model_out.last_hidden_state, input_ids)
