"""Conversation building & label-mask construction for DiffUME.

UME-R1 stores `conversations` in a **columnar** layout:
    {"from": ["human", "gpt"], "value": ["...user...", "...assistant..."]}

The assistant turn already contains the inline reasoning and embedding tag,
e.g.::
    <think>chain-of-thought ...</think>
    <answer>summary ...<|begin_of_box|>concise<|end_of_box|>
    <gen_emb>

Per SDAR-VL §4.1, *the chain-of-thought template tokens (`<think>`, `</think>`)
must always be force-masked during training*.  Additionally we **never** mask
the readout tokens themselves (`<gen_emb>`, `<disc_emb>`) because their final
hidden states are the embeddings we are training.

This module provides:

* :func:`extract_turns` — pull out (user, assistant) strings from the
  columnar conversations dict, tolerant of UME-R1's `from`/`value` and the
  more usual `role`/`content` layouts.
* :func:`build_chatml_text` — assemble the ChatML string the model sees,
  with `<image>`/`<video>` placeholders already replaced by the
  Qwen2.5-VL `<|vision_start|>...<|vision_end|>` blocks.
* :func:`build_label_and_force_masks` — given the tokenised assistant span
  and the special-token id table, return three same-length 1-D tensors:

      labels:           token ids on the **assistant span**, IGNORE elsewhere
      force_mask_ids:   bool, positions that must be masked every step
                        (i.e. ``<think>`` / ``</think>`` template tokens)
      readout_keep_ids: bool, positions that must NEVER be masked
                        (i.e. ``<gen_emb>`` / ``<disc_emb>``)
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import torch

IGNORE_INDEX = -100

# Qwen2.5-VL ChatML template (matches preprocess_qwen_2_visual in UME-R1).
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

SYSTEM_MESSAGE = "You are a helpful assistant."


def _iter_role_value(convs):
    """Yield (role, value) pairs from either columnar or list-of-dict layouts."""
    if isinstance(convs, dict) and "from" in convs and "value" in convs:
        for r, v in zip(convs["from"], convs["value"]):
            yield r, v
    elif isinstance(convs, list):
        for turn in convs:
            r = turn.get("role") or turn.get("from")
            v = turn.get("content") or turn.get("value")
            yield r, v


def extract_turns(sub) -> Tuple[str, str]:
    """Return (user_text, assistant_text) from a qry/pos sub-dict."""
    convs = sub.get("conversations") or {}
    user_text = assistant_text = None
    role_aliases = {"user": "user", "human": "user",
                    "assistant": "assistant", "gpt": "assistant"}
    for r, v in _iter_role_value(convs):
        rr = role_aliases.get(r, r)
        if rr == "user" and user_text is None:
            user_text = v
        elif rr == "assistant" and assistant_text is None:
            assistant_text = v
    if user_text is None or assistant_text is None:
        raise ValueError(f"Could not extract user/assistant turns from {convs!r}")
    return user_text, assistant_text


def replace_visual_placeholders(
    user_text: str,
    grid_thw_image: Iterable[int] = (),
    grid_thw_video: Iterable[int] = (),
) -> str:
    """Replace `<image>` / `<video>` markers with Qwen vision blocks.

    Mirrors :func:`preprocess_qwen_2_visual` from UME-R1 but works on a
    **single** user string instead of looping the whole conversation.
    """
    grid_thw_image = list(grid_thw_image)
    grid_thw_video = list(grid_thw_video)
    if "<image>" in user_text:
        parts = user_text.split("<image>")
        out: List[str] = []
        for i in range(len(parts) - 1):
            out.append(parts[i])
            n = grid_thw_image[i]
            out.append("<|vision_start|>" + "<|image_pad|>" * n + "<|vision_end|>")
        out.append(parts[-1])
        user_text = "".join(out)
    if "<video>" in user_text:
        parts = user_text.split("<video>")
        out = []
        for i in range(len(parts) - 1):
            out.append(parts[i])
            n = grid_thw_video[i]
            out.append("<|vision_start|>" + "<|video_pad|>" * n + "<|vision_end|>")
        out.append(parts[-1])
        user_text = "".join(out)
    return user_text


def build_chatml_text(user_text: str, assistant_text: str,
                      system_message: str = SYSTEM_MESSAGE) -> str:
    """Assemble ChatML string identical to UME-R1's tokenizer.apply_chat_template."""
    return (
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_text}<|im_end|>\n"
    )


def tokenize_with_assistant_span(
    tokenizer,
    user_text: str,
    assistant_text: str,
    system_message: str = SYSTEM_MESSAGE,
) -> dict:
    """Tokenise system+user+assistant and return assistant token boundaries.

    Returns
    -------
    dict with keys
        input_ids:        LongTensor (L,)
        labels:           LongTensor (L,) — assistant tokens only, IGNORE elsewhere
        assistant_start:  int — index of the first assistant content token
                                (just **after** ``<|im_start|>assistant\\n``)
        assistant_end:    int — exclusive end of the assistant content
                                (just **before** the trailing ``<|im_end|>``)
    """
    sys_user_prefix = (
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assistant_with_eot = f"{assistant_text}<|im_end|>\n"

    pre_ids = tokenizer.encode(sys_user_prefix, add_special_tokens=False)
    asst_ids = tokenizer.encode(assistant_with_eot, add_special_tokens=False)

    # The trailing "<|im_end|>\n" — we want the assistant span to STOP before it
    # so the loss is computed only over real content + (optional) <gen_emb>.
    eot_ids = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
    eot_len = len(eot_ids)
    assistant_content_len = max(len(asst_ids) - eot_len, 1)

    input_ids = pre_ids + asst_ids
    labels = [IGNORE_INDEX] * len(pre_ids) + asst_ids[:assistant_content_len] + \
             [IGNORE_INDEX] * (len(asst_ids) - assistant_content_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "assistant_start": len(pre_ids),
        "assistant_end": len(pre_ids) + assistant_content_len,
    }


def build_force_masks(
    input_ids: torch.LongTensor,
    assistant_start: int,
    assistant_end: int,
    tokenizer,
    mask_id: int,
    readout_ids: Iterable[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct the force-mask / readout-keep / diffusable masks.

    Returns three boolean tensors of shape ``input_ids.shape``:

        force_mask:   True ⇢ position MUST be the MASK token at every step.
                      Set for ``<think>`` / ``</think>`` / ``<answer>`` /
                      ``</answer>`` template tokens within the assistant span.
        readout_keep: True ⇢ position must NEVER be masked
                      (the gen_emb / disc_emb readout tokens).
        diffusable:   True ⇢ position is part of the assistant span and is
                      neither force-masked nor a readout token, i.e. it is a
                      candidate for stochastic noising during training.
    """
    L = input_ids.numel()
    force_mask = torch.zeros(L, dtype=torch.bool)
    readout_keep = torch.zeros(L, dtype=torch.bool)
    diffusable = torch.zeros(L, dtype=torch.bool)

    template_strings = ["<think>", "</think>", "<answer>", "</answer>"]
    template_ids = set()
    for s in template_strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        for i in ids:
            template_ids.add(int(i))

    readout_ids = set(int(x) for x in readout_ids)

    for i in range(assistant_start, assistant_end):
        tid = int(input_ids[i].item())
        if tid in readout_ids:
            readout_keep[i] = True
        elif tid in template_ids:
            force_mask[i] = True
        else:
            diffusable[i] = True

    return force_mask, readout_keep, diffusable
