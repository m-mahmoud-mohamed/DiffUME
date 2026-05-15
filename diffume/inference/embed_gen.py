"""Inference: generative embedding via block-wise diffusion decoding.

Unlike `embed_disc.py` which assumes the assistant turn is supplied
verbatim, this entrypoint *generates* the assistant span from scratch
using :func:`diffume.diffusion.sampling.block_diffusion_generate`,
then reads out the ``<gen_emb>`` hidden state.
"""

from __future__ import annotations

from typing import Optional

import torch
from PIL import Image

from ..data.prompting import replace_visual_placeholders
from ..diffusion.sampling import BlockDecodeConfig, block_diffusion_generate
from ..models.diffume_model import DiffumeModel
from ..models.readout import get_embedding_reps


@torch.no_grad()
def embed_generate(
    model: DiffumeModel,
    tokenizer,
    processor,
    ids: dict,
    *,
    user_text: str,
    image: Optional[Image.Image] = None,
    n_blocks: int = 4,
    n_steps: int = 8,
    device: str = "cuda",
):
    """Return (z_disc, z_gen) — generates the assistant span on the fly."""
    grid_thw_image = []
    image_pack = {}
    if image is not None:
        out = processor.image_processor(images=[image], return_tensors="pt")
        merge = getattr(processor.image_processor, "merge_size", 2)
        t, h, w = out["image_grid_thw"][0].tolist()
        n_pad = (t * h * w) // (merge * merge)
        grid_thw_image = [n_pad]
        image_pack = {"pixel_values": out["pixel_values"].to(device),
                      "image_grid_thw": out["image_grid_thw"].to(device)}

    user_text = replace_visual_placeholders(user_text, grid_thw_image=grid_thw_image)
    prefix = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False, return_tensors="pt").to(device)

    cfg = BlockDecodeConfig(
        block_size=model.cfg.block_size,
        n_blocks=n_blocks, n_steps=n_steps,
        eos_token_id=tokenizer.eos_token_id,
    )
    full_ids = block_diffusion_generate(
        model.backbone, prefix_input_ids=prefix_ids, mask_id=model.mask_id,
        cfg=cfg, extra_model_kwargs=image_pack,
    )

    model_out = model.backbone.model(
        input_ids=full_ids,
        position_ids=model._compute_position_ids(full_ids, image_pack, None),
        use_cache=False, return_dict=True, **image_pack,
    )
    lhs = model_out.last_hidden_state
    z_disc = get_embedding_reps(lhs, full_ids, model.disc_emb_id)
    z_gen = get_embedding_reps(lhs, full_ids, model.gen_emb_id)
    return z_disc[0].cpu(), z_gen[0].cpu(), full_ids[0].cpu()
