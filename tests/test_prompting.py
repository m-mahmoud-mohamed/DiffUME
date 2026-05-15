"""Tests for prompting / force-mask logic."""

import torch
from transformers import AutoTokenizer

from diffume.data.prompting import (
    build_force_masks,
    extract_turns,
    replace_visual_placeholders,
    tokenize_with_assistant_span,
)


def _get_tokenizer():
    # Use a tiny tokenizer for CPU speed; we just need encode/decode.
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")


def test_extract_turns_columnar():
    sub = {"conversations": {"from": ["human", "gpt"], "value": ["U", "A"]}}
    u, a = extract_turns(sub)
    assert u == "U" and a == "A"


def test_extract_turns_listed():
    sub = {"conversations": [{"role": "user", "content": "U"},
                              {"role": "assistant", "content": "A"}]}
    u, a = extract_turns(sub)
    assert u == "U" and a == "A"


def test_replace_visual_placeholders_image():
    out = replace_visual_placeholders("hello <image> world", grid_thw_image=[3])
    assert "<|vision_start|>" in out
    assert out.count("<|image_pad|>") == 3
    assert "<|vision_end|>" in out


def test_force_masks_keeps_readout():
    tok = _get_tokenizer()
    user = "Question?"
    asst = "<think>chain</think> answer <gen_emb>"
    out = tokenize_with_assistant_span(tok, user, asst)
    ids = out["input_ids"]
    s, e = out["assistant_start"], out["assistant_end"]
    # We don't know the actual <gen_emb> id in this base tokenizer; pretend it's the last token id.
    gen_id = int(ids[e - 1].item())
    fm, rk, df = build_force_masks(ids, s, e, tok, mask_id=999, readout_ids=[gen_id])
    assert rk.sum().item() >= 1
    # readout positions are NOT diffusable.
    assert (rk & df).sum().item() == 0
    # force-mask positions are NOT diffusable.
    assert (fm & df).sum().item() == 0
