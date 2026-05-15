"""Inference: discriminative embedding (clean forward, no diffusion).

Given an input pair (text + image, with the standard UME-R1 prompt),
load a trained DiffumeModel and return the L2-normalised
``z_disc`` and ``z_gen`` embedding vectors.
"""

from __future__ import annotations

from typing import Iterable

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from ..data.collator import DiffumeCollator
from ..data.prompting import (
    SYSTEM_MESSAGE,
    build_force_masks,
    replace_visual_placeholders,
    tokenize_with_assistant_span,
)
from ..models.diffume_model import DiffumeConfig, DiffumeModel
from ..tokens import register_diffume_special_tokens


# Default UME-R1 instruction (UME-R1/src/r1-train/.../qwen_module.py:75).
EMBED_INSTRUCTION = (
    "Represent the above input text, images, videos, or any combination of "
    "the three as embeddings. First output the thinking process in <think> "
    "</think> tags and then summarize the entire input in a word or "
    "sentence. Finally, use the <gen_emb> tag to represent the entire input."
)


def load_diffume(checkpoint_dir: str, backbone_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                 dtype=torch.bfloat16):
    backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        backbone_name, torch_dtype=dtype, attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    processor = AutoProcessor.from_pretrained(backbone_name)
    ids = register_diffume_special_tokens(backbone, tokenizer)
    model = DiffumeModel(
        backbone, mask_id=ids["mask_id"],
        gen_emb_id=ids["gen_emb_id"], disc_emb_id=ids["disc_emb_id"],
        cfg=DiffumeConfig(backbone_name=backbone_name),
    )
    sd_path = f"{checkpoint_dir}/pytorch_model.bin"
    try:
        sd = torch.load(sd_path, map_location="cpu")
        model.load_state_dict(sd, strict=False)
    except FileNotFoundError:
        pass
    return model.eval(), tokenizer, processor, ids


@torch.no_grad()
def embed_pair(
    model: DiffumeModel,
    tokenizer,
    processor,
    ids: dict,
    *,
    user_text: str,
    assistant_text: str,
    image: Image.Image | None = None,
    device: str = "cuda",
):
    """Return (z_disc, z_gen) for a single example using the **clean** forward."""
    grid_thw_image = []
    image_pack = None
    if image is not None:
        out = processor.image_processor(images=[image], return_tensors="pt")
        merge = getattr(processor.image_processor, "merge_size", 2)
        t, h, w = out["image_grid_thw"][0].tolist()
        n_pad = (t * h * w) // (merge * merge)
        grid_thw_image = [n_pad]
        image_pack = {"pixel_values": out["pixel_values"].to(device),
                      "image_grid_thw": out["image_grid_thw"].to(device)}

    user_text = replace_visual_placeholders(
        user_text, grid_thw_image=grid_thw_image,
    )
    tok = tokenize_with_assistant_span(tokenizer, user_text, assistant_text)
    input_ids = tok["input_ids"].unsqueeze(0).to(device)
    asst_starts = torch.tensor([tok["assistant_start"]], dtype=torch.long, device=device)
    asst_ends = torch.tensor([tok["assistant_end"]], dtype=torch.long, device=device)

    z_disc, z_gen = model.encode(
        input_ids=input_ids,
        assistant_starts=asst_starts,
        assistant_ends=asst_ends,
        vision_kwargs=image_pack,
    )
    return z_disc[0].cpu(), z_gen[0].cpu()
