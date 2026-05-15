"""Pad-and-stack collator for the paired DiffUME SFT dataset."""

from __future__ import annotations

from typing import List

import torch


def _pad_1d(tensors, pad_value, dtype=None):
    L = max(t.size(0) for t in tensors)
    out = torch.full((len(tensors), L), pad_value, dtype=dtype or tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, : t.size(0)] = t
    return out


def _collate_side(side_dicts: List[dict], pad_token_id: int) -> dict:
    input_ids = _pad_1d([d["input_ids"] for d in side_dicts], pad_token_id)
    labels = _pad_1d([d["labels"] for d in side_dicts], -100, dtype=torch.long)
    force_mask = _pad_1d([d["force_mask"] for d in side_dicts], 0, dtype=torch.bool)
    readout_keep = _pad_1d([d["readout_keep"] for d in side_dicts], 0, dtype=torch.bool)
    diffusable = _pad_1d([d["diffusable"] for d in side_dicts], 0, dtype=torch.bool)
    attention_mask = (input_ids != pad_token_id).long()
    asst_starts = torch.tensor([d["assistant_start"] for d in side_dicts], dtype=torch.long)
    asst_ends = torch.tensor([d["assistant_end"] for d in side_dicts], dtype=torch.long)

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "force_mask": force_mask,
        "readout_keep": readout_keep,
        "diffusable": diffusable,
        "assistant_start": asst_starts,
        "assistant_end": asst_ends,
    }

    # Pixel values: concatenate along the patch axis (Qwen2.5-VL convention).
    if any("pixel_values" in d for d in side_dicts):
        pvs, grids = [], []
        for d in side_dicts:
            if "pixel_values" in d:
                pvs.append(d["pixel_values"])
                grids.append(d["image_grid_thw"])
        if pvs:
            out["pixel_values"] = torch.cat(pvs, dim=0)
            out["image_grid_thw"] = torch.cat(grids, dim=0)
    return out


class DiffumeCollator:
    """Collator returning a dict with `qry` and `pos` sub-dicts."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[dict]) -> dict:
        qry = _collate_side([b["qry"] for b in batch], self.pad_token_id)
        pos = _collate_side([b["pos"] for b in batch], self.pad_token_id)
        return {"qry": qry, "pos": pos}
