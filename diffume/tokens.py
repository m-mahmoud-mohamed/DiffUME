"""Special-token registration for DiffUME.

We re-use UME-R1 token IDs where possible:
  - GEN_EMB  (id 151657)  — readout token for the **generative** embedding
  - DISC_EMB (id 151658)  — readout token for the **discriminative** embedding

We also need a **MASK token** for discrete diffusion.  Qwen2.5-VL's tokenizer
already exposes `<|object_ref_start|>` etc. as special tokens, but to keep the
diffusion vocabulary clean we add a fresh `<|mask|>` token and mean-init its
embedding from the existing token table (Brown et al. tokenizer-extension
heuristic; gives a much better starting point than random).

Public API:
    register_diffume_special_tokens(model, tokenizer) -> dict
        Adds `<|mask|>` (and the gen/disc tokens if absent), resizes the
        model's input/output embedding tables, mean-initialises the new
        rows, and returns a dict with the token ids.
"""

from __future__ import annotations

import torch

GEN_EMB_TOKEN = "<gen_emb>"
DISC_EMB_TOKEN = "<disc_emb>"
MASK_TOKEN = "<|mask|>"

# UME-R1 fixed IDs (see UME-R1/src/sft-train/qwenvl/train/trainer.py L51-L53).
UMER1_GEN_EMB_ID = 151657
UMER1_DISC_EMB_ID = 151658


def _mean_init_new_rows(embedding: torch.nn.Embedding, n_new: int) -> None:
    """Mean-init the trailing `n_new` rows of an embedding table."""
    with torch.no_grad():
        old_n = embedding.weight.size(0) - n_new
        if old_n <= 0 or n_new <= 0:
            return
        mean = embedding.weight[:old_n].mean(dim=0)
        embedding.weight[old_n:] = mean.unsqueeze(0).expand(n_new, -1).clone()


def register_diffume_special_tokens(model, tokenizer) -> dict:
    """Add `<|mask|>` (+ gen/disc if absent) and resize the model.

    Returns a dict ``{"mask_id": ..., "gen_emb_id": ..., "disc_emb_id": ...}``.
    """
    to_add = []

    if tokenizer.convert_tokens_to_ids(MASK_TOKEN) == tokenizer.unk_token_id:
        to_add.append(MASK_TOKEN)
    for tok in (GEN_EMB_TOKEN, DISC_EMB_TOKEN):
        if tokenizer.convert_tokens_to_ids(tok) == tokenizer.unk_token_id:
            to_add.append(tok)

    n_new = 0
    if to_add:
        n_new = tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        model.resize_token_embeddings(len(tokenizer))

        in_emb = model.get_input_embeddings()
        _mean_init_new_rows(in_emb, n_new)

        out_emb = model.get_output_embeddings()
        if out_emb is not None and out_emb.weight is not in_emb.weight:
            _mean_init_new_rows(out_emb, n_new)

    return {
        "mask_id": tokenizer.convert_tokens_to_ids(MASK_TOKEN),
        "gen_emb_id": tokenizer.convert_tokens_to_ids(GEN_EMB_TOKEN),
        "disc_emb_id": tokenizer.convert_tokens_to_ids(DISC_EMB_TOKEN),
        "n_added": n_new,
    }
