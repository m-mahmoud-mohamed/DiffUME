"""Block-diffusion inference sampler  (SDAR-VL §C.2 + reference ``generate.py``).

Three remasking strategies are supported, mirroring SDAR-VL's
``block_diffusion_generate``:

* ``"sequential"``           — fill the leftmost still-MASK positions first
                               (deterministic left-to-right).
* ``"low_confidence_static"`` — every step, lock-in the top-k most confident
                                still-MASK positions (k from a fixed schedule
                                ``num_transfer_tokens``).
* ``"low_confidence_dynamic"`` — same as static but if the number of
                                positions whose confidence exceeds
                                ``confidence_threshold`` is ≥ k, lock them
                                **all** in this step (early-finish).

Decoding of each block runs for ``T`` denoising steps, then any remaining
MASK position is force-decoded to its argmax.  After every block a
tail-copy repetition check (default ≥10 consecutive repeats) halts
generation early — same heuristic as ``detect_tail_copy_repetition`` in
the SDAR-VL reference repo.

This module is written for **embedding** inference where we just need to
produce a fixed number of blocks of assistant content before reading out
``<gen_emb>`` / ``<disc_emb>`` — no incremental KV cache.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F

from ..models.attention_block import build_block_causal_mask


RemaskStrategy = Literal[
    "sequential", "low_confidence_static", "low_confidence_dynamic"
]


@dataclass
class BlockDecodeConfig:
    block_size: int = 4
    n_blocks: int = 4              # generate block_size * n_blocks tokens total
    n_steps: int = 8               # T denoising steps per block
    temperature: float = 0.0       # 0 ⇒ greedy
    top_k: int = 0
    top_p: float = 1.0
    remasking_strategy: RemaskStrategy = "low_confidence_static"
    confidence_threshold: float = 0.9
    eos_token_id: Optional[int] = None
    repetition_min_repeats: int = 10  # 0 disables tail-copy detection


# ---------------------------------------------------------------------------
# Sampling helpers (port of SDAR-VL generate.py)
# ---------------------------------------------------------------------------
def _top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[..., -1, None]
    return torch.where(
        logits < min_values, torch.full_like(logits, float("-inf")), logits
    )


def _top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(
        torch.full_like(logits, False, dtype=torch.bool),
        -1, sorted_indices, sorted_mask,
    )
    return logits.masked_fill(mask_indices, float("-inf"))


def _sample(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (sampled_ids, original_probs_at_sampled) of shape ``logits.shape[:-1]``."""
    orig_shape = logits.shape[:-1]
    V = logits.shape[-1]
    flat = logits.reshape(-1, V)
    if temperature != 1.0 and temperature > 0.0:
        flat = flat / temperature
    ori_probs = F.softmax(flat, dim=-1)
    if temperature == 0.0:
        ids = ori_probs.argmax(dim=-1, keepdim=True)
    else:
        filtered = flat
        if top_k > 0:
            filtered = _top_k_logits(filtered, top_k)
        if top_p < 1.0:
            filtered = _top_p_logits(filtered, top_p)
        ids = torch.multinomial(F.softmax(filtered, dim=-1), num_samples=1)
    confs = torch.gather(ori_probs, -1, ids)
    return ids.view(*orig_shape), confs.view(*orig_shape)


def _num_transfer_tokens(block_length: int, steps: int) -> torch.Tensor:
    base = block_length // steps
    rem = block_length % steps
    out = torch.full((steps,), base, dtype=torch.int64)
    out[:rem] += 1
    return out


def _detect_tail_copy_repetition(tokens: list[int], min_repeats: int) -> bool:
    """Return True if the tail of ``tokens`` is a unit repeated ≥ min_repeats."""
    n = len(tokens)
    if n < 2 or min_repeats <= 0:
        return False
    for L in range(1, n // 2 + 1):
        unit = tokens[-L:]
        repeats = 1
        idx = n - L
        while idx - L >= 0 and tokens[idx - L: idx] == unit:
            repeats += 1
            idx -= L
        if repeats >= min_repeats:
            return True
    return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
@torch.no_grad()
def block_diffusion_generate(
    model,
    *,
    prefix_input_ids: torch.LongTensor,        # (B, L_pre)
    mask_id: int,
    cfg: BlockDecodeConfig,
    extra_model_kwargs: dict | None = None,    # pixel_values, image_grid_thw, ...
) -> torch.LongTensor:
    """Run block-wise diffusion decoding.

    Returns the full sequence ids (prefix + generated blocks) of shape
    ``(B, L_pre + n_blocks * block_size)``.  Generation may stop early
    on EOS or tail-copy repetition; in that case the trailing positions
    are left as ``mask_id``.
    """
    extra_model_kwargs = extra_model_kwargs or {}
    device = prefix_input_ids.device
    B, L_pre = prefix_input_ids.shape

    n_gen = cfg.n_blocks * cfg.block_size
    full_ids = torch.cat(
        [prefix_input_ids,
         torch.full((B, n_gen), mask_id, dtype=torch.long, device=device)],
        dim=1,
    )
    block_idx = torch.full((B, L_pre + n_gen), -1, dtype=torch.long, device=device)
    pos_asst = torch.arange(n_gen, device=device)
    block_idx[:, L_pre:] = (pos_asst // cfg.block_size).unsqueeze(0).expand(B, -1)

    # Block-causal mask is fixed for the whole sequence; rebuild once.
    attn_mask = build_block_causal_mask(block_idx, dtype=torch.float32, device=device)
    n_transfer = _num_transfer_tokens(cfg.block_size, cfg.n_steps)

    for b in range(cfg.n_blocks):
        block_start = L_pre + b * cfg.block_size
        block_end = block_start + cfg.block_size

        for step in range(cfg.n_steps):
            still_mask = (full_ids[:, block_start:block_end] == mask_id)
            if not still_mask.any():
                break

            out = model(
                input_ids=full_ids,
                attention_mask=attn_mask,
                **extra_model_kwargs,
            )
            logits = out.logits[:, block_start:block_end, :]   # (B, block_size, V)
            x0, x0_p = _sample(
                logits,
                temperature=cfg.temperature if cfg.temperature > 0 else 1.0,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
            )

            n_keep = int(n_transfer[step].item())
            transfer = torch.zeros_like(still_mask, dtype=torch.bool)

            if cfg.remasking_strategy == "sequential":
                for j in range(B):
                    pos = still_mask[j].nonzero(as_tuple=True)[0]
                    if pos.numel() == 0:
                        continue
                    take = pos[:n_keep]
                    transfer[j, take] = True

            elif cfg.remasking_strategy == "low_confidence_static":
                conf = torch.where(still_mask, x0_p, torch.full_like(x0_p, -float("inf")))
                k = min(n_keep, cfg.block_size)
                idx = conf.topk(k, dim=-1).indices
                transfer.scatter_(1, idx, True)
                transfer &= still_mask  # safety

            elif cfg.remasking_strategy == "low_confidence_dynamic":
                conf = torch.where(still_mask, x0_p, torch.full_like(x0_p, -float("inf")))
                for j in range(B):
                    high = conf[j] > cfg.confidence_threshold
                    if int(high.sum().item()) >= n_keep:
                        transfer[j] = high
                    else:
                        k = min(n_keep, cfg.block_size)
                        idx = conf[j].topk(k).indices
                        transfer[j, idx] = True
                transfer &= still_mask
            else:
                raise ValueError(f"Unknown remasking_strategy: {cfg.remasking_strategy}")

            # Lock in the chosen positions.
            cur_block = full_ids[:, block_start:block_end]
            cur_block = torch.where(transfer, x0, cur_block)
            full_ids[:, block_start:block_end] = cur_block

        # Force-decode any leftover MASK to argmax.
        still_mask = (full_ids[:, block_start:block_end] == mask_id)
        if still_mask.any():
            out = model(
                input_ids=full_ids,
                attention_mask=attn_mask,
                **extra_model_kwargs,
            )
            logits = out.logits[:, block_start:block_end, :]
            argmax_ids = logits.argmax(dim=-1)
            full_ids[:, block_start:block_end] = torch.where(
                still_mask, argmax_ids, full_ids[:, block_start:block_end]
            )

        # Tail-copy repetition early stop (port of SDAR-VL detect_tail_copy_repetition).
        if cfg.repetition_min_repeats > 0:
            gen_so_far = full_ids[0, L_pre: block_end].tolist()
            if _detect_tail_copy_repetition(gen_so_far, cfg.repetition_min_repeats):
                break

        # EOS early stop.
        if cfg.eos_token_id is not None:
            done = (full_ids[:, L_pre:block_end] == cfg.eos_token_id).any(dim=-1)
            if done.all():
                break

    return full_ids
