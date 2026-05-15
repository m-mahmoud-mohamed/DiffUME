"""SFT dataset adapter for `umer1_sft_processed`.

The on-disk schema (verified by `scripts/00_inspect_dataset.py`):

    Dataset({features: {'dataset_name': str, 'qry': Sub, 'pos': Sub}})

where ``Sub`` is a sub-dict::

    {
      "image":         PIL.Image,
      "video":         List[PIL.Image]   # all 28×28 placeholders in current dump
      "conversations": {"from": [...], "value": [...]}
    }

Each sample yields a (qry, pos) **paired** record: every `__getitem__`
returns a dict with ``qry_*`` and ``pos_*`` sub-fields ready for the
collator.  We rely on the Qwen2.5-VL `AutoProcessor` to tokenise text and
process images.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils.data import Dataset

from .prompting import (
    SYSTEM_MESSAGE,
    build_force_masks,
    extract_turns,
    replace_visual_placeholders,
    tokenize_with_assistant_span,
)

log = logging.getLogger(__name__)


def _is_placeholder_video(video) -> bool:
    """Heuristic: UME-R1 dump uses 28×28 PIL placeholders in `video` field."""
    if video is None:
        return True
    if isinstance(video, list):
        if len(video) == 0:
            return True
        first = video[0]
        return getattr(first, "size", None) == (28, 28)
    return False


def _process_image(image, processor, max_image_tokens: int = 1200):
    """Run image through Qwen2.5-VL image processor; return (pixel_values, grid_thw).

    ``max_image_tokens`` caps how many ``<|image_pad|>`` tokens one image may
    contribute.  Each patch is 14×14 pixels; merge_size=2 means 4 patches become
    one token, so longest_edge = max_image_tokens * 4 * 196 pixels total.
    """
    if image is None:
        return None, None
    # Cap pixel count so n_pad stays within budget even for very large images.
    # n_pad = T*H*W // 4  and  total_pixels = T*H*W * 196.
    longest_edge = max_image_tokens * 4 * 196  # ≈ 940 800 for max_image_tokens=1200
    out = processor.image_processor(
        images=[image],
        return_tensors="pt",
        size={"shortest_edge": 56 * 56, "longest_edge": longest_edge},
    )
    pv = out["pixel_values"]                       # (n_patches, C)
    grid_thw = out["image_grid_thw"]               # (1, 3) — T, H, W in patch units
    # Number of <|image_pad|> tokens to insert in the user turn.
    t, h, w = grid_thw[0].tolist()
    merge = getattr(processor.image_processor, "merge_size", 2)
    n_pad = (t * h * w) // (merge * merge)
    return {"pixel_values": pv, "image_grid_thw": grid_thw}, n_pad


class DiffumePairedDataset(Dataset):
    """Wraps the HF `umer1_sft_processed` dataset for paired SFT."""

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        processor,
        readout_ids: dict,
        *,
        max_length: int = 1024,
        system_message: str = SYSTEM_MESSAGE,
        skip_video: bool = True,         # placeholders only in current dump
    ) -> None:
        self.ds = hf_dataset
        self.tokenizer = tokenizer
        self.processor = processor
        self.mask_id = int(readout_ids["mask_id"])
        self.gen_emb_id = int(readout_ids["gen_emb_id"])
        self.disc_emb_id = int(readout_ids["disc_emb_id"])
        self.readout_ids = (self.gen_emb_id, self.disc_emb_id)
        self.max_length = max_length
        self.system_message = system_message
        self.skip_video = skip_video

    def __len__(self) -> int:
        return len(self.ds)

    def _process_side(self, sub: dict) -> dict:
        user_text, asst_text = extract_turns(sub)

        # Image processing.
        img = sub.get("image")
        # Derive a per-sample image token budget: reserve ≥25 % of max_length
        # for text; clamp to a minimum of 1 to avoid zero-pixel images.
        max_image_tokens = max(1, self.max_length * 3 // 4)
        img_pack, n_image_pads = _process_image(img, self.processor, max_image_tokens)
        grid_thw_image = [n_image_pads] if n_image_pads is not None else []

        # Skip video placeholders entirely.
        video = sub.get("video")
        grid_thw_video: list[int] = []
        video_pack = None
        if not (self.skip_video and _is_placeholder_video(video)):
            # (Real video handling not implemented yet — fall through.)
            pass
        # Strip <video> markers if we are skipping.
        if self.skip_video:
            user_text = user_text.replace("<video>", "")

        # Replace <image> markers with vision blocks.
        user_text = replace_visual_placeholders(
            user_text, grid_thw_image=grid_thw_image, grid_thw_video=grid_thw_video,
        )

        tok = tokenize_with_assistant_span(
            self.tokenizer, user_text, asst_text, self.system_message,
        )
        ids = tok["input_ids"]
        if ids.numel() > self.max_length:
            # The prefix (system + user turn, including <|image_pad|> tokens) MUST
            # NOT be truncated — any cut there would leave pixel_values with more
            # patch features than there are image-pad tokens in input_ids, causing
            # a ValueError inside Qwen2_5_VLModel.get_placeholder_mask.
            #
            # Strategy: truncate only the ASSISTANT span while always keeping the
            # last 16 tokens (readout + closing boilerplate).
            asst_start = tok["assistant_start"]
            if asst_start + 16 > self.max_length:
                # Even without any assistant text the prefix alone busts the limit.
                # This should be rare after the max_image_tokens cap above, but
                # guard against it by skipping and letting __getitem__ retry.
                raise ValueError(
                    f"Prefix too long ({asst_start} tokens) for "
                    f"max_length={self.max_length}; skipping sample."
                )
            # How many assistant tokens fit?
            asst_budget = self.max_length - asst_start  # ≥ 16
            tail = ids[-16:]
            asst_head_len = asst_budget - 16
            asst_head = ids[asst_start: asst_start + asst_head_len]
            ids = torch.cat([ids[:asst_start], asst_head, tail])

            lbl = tok["labels"]
            tok["labels"] = torch.cat(
                [lbl[:asst_start],
                 lbl[asst_start: asst_start + asst_head_len],
                 lbl[-16:]]
            )
            tok["assistant_start"] = asst_start          # unchanged
            tok["assistant_end"] = ids.numel()            # = max_length
            tok["input_ids"] = ids

        force_mask, readout_keep, diffusable = build_force_masks(
            tok["input_ids"], tok["assistant_start"], tok["assistant_end"],
            self.tokenizer, self.mask_id, self.readout_ids,
        )

        out = {
            "input_ids": tok["input_ids"],
            "labels": tok["labels"],
            "assistant_start": tok["assistant_start"],
            "assistant_end": tok["assistant_end"],
            "force_mask": force_mask,
            "readout_keep": readout_keep,
            "diffusable": diffusable,
        }
        if img_pack is not None:
            out.update(img_pack)
        return out

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.ds[idx]
        try:
            qry = self._process_side(row["qry"])
            pos = self._process_side(row["pos"])
        except Exception as e:  # robust to corrupted images
            log.warning("sample %d failed: %s — falling back to next", idx, e)
            return self.__getitem__((idx + 1) % len(self.ds))
        return {"qry": qry, "pos": pos, "dataset_name": row["dataset_name"]}
