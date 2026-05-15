"""DiffUME SFT training entrypoint (Accelerate + DeepSpeed-friendly).

Usage::

    accelerate launch --config_file configs/accelerate/zero2_8gpu.yaml \
        -m diffume.training.train --config configs/diffume_stage1.yaml
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from ..data.collator import DiffumeCollator
from ..data.sft_dataset import DiffumePairedDataset
from ..diffusion.pbnc import PBNCConfig
from ..models.diffume_model import DiffumeConfig, DiffumeModel
from ..tokens import register_diffume_special_tokens
from .trainer_step import DiffumeStep, LossWeights


def load_yaml(p: str) -> dict:
    with open(p) as f:
        return yaml.safe_load(f)


def build_backbone(name: str, dtype: torch.dtype):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(name)
    mt = getattr(cfg, "model_type", "")
    load_kw = dict(dtype=dtype, attn_implementation="sdpa")
    if mt == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration.from_pretrained(name, **load_kw)
    elif mt == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(name, **load_kw)
    else:
        raise ValueError(
            f"Unsupported backbone model_type '{mt}' at {name}. "
            "Expected 'qwen2_vl' (UME-R1-2B) or 'qwen2_5_vl' (Qwen2.5-VL-3B/7B)."
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))

    accel = Accelerator(
        gradient_accumulation_steps=cfg.get("grad_accum", 1),
        mixed_precision=cfg.get("mixed_precision", "bf16"),
        log_with=cfg.get("log_with"),
    )

    backbone_name = cfg["backbone"]
    dtype = torch.bfloat16 if accel.mixed_precision == "bf16" else torch.float32
    accel.print(f"Loading backbone {backbone_name} ...")
    backbone = build_backbone(backbone_name, dtype)
    if cfg.get("gradient_checkpointing", True):
        backbone.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    processor = AutoProcessor.from_pretrained(backbone_name)

    ids = register_diffume_special_tokens(backbone, tokenizer)
    accel.print(f"Special token ids: {ids}")

    model = DiffumeModel(
        backbone,
        mask_id=ids["mask_id"],
        gen_emb_id=ids["gen_emb_id"],
        disc_emb_id=ids["disc_emb_id"],
        cfg=DiffumeConfig(
            block_size=cfg.get("block_size", 32),
            pbnc=PBNCConfig(**cfg.get("pbnc", {})),
            backbone_name=backbone_name,
        ),
    )

    accel.print("Loading dataset ...")
    hfds = load_from_disk(cfg["dataset_path"])
    if "max_train_samples" in cfg:
        hfds = hfds.select(range(min(cfg["max_train_samples"], len(hfds))))

    ds = DiffumePairedDataset(
        hfds, tokenizer, processor, ids,
        max_length=cfg.get("max_length", 1024),
    )
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    coll = DiffumeCollator(pad_token_id=pad_id)
    loader = DataLoader(
        ds, batch_size=cfg.get("per_device_batch_size", 1),
        shuffle=True, collate_fn=coll,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=True, drop_last=True,
    )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 1e-5),
        betas=(0.9, 0.95),
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    total_steps = cfg.get("max_steps", 10_000)
    warmup = cfg.get("warmup_steps", 200)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return max(0.05, 0.5 * (1 + torch.cos(torch.tensor(prog * 3.14159265)).item()))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    model, optim, loader, sched = accel.prepare(model, optim, loader, sched)

    # IMPORTANT: pass the FSDP/DDP-wrapped `model` (NOT model.module) so that
    # `self.model(...)` inside DiffumeStep triggers the wrapper's pre-forward
    # hook. With FSDP `use_orig_params=True`, that hook is what gathers the
    # sharded flat-param into proper 2-D views for embed_tokens / lm_head.
    # Calling model.module(...) bypasses the hook → 'weight must be 2-D'.
    step_fn = DiffumeStep(
        model,
        weights=LossWeights(**cfg.get("loss_weights", {})),
    )

    out_dir = Path(cfg.get("output_dir", "/mnt/ceph-ssd/workspaces/ws/nii00224/u27702-diffusion-data/diffume-stage1"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── startup summary ────────────────────────────────────────────────
    if accel.is_main_process:
        bc = backbone.config
        total_params = sum(p.numel() for p in backbone.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        pbnc_cfg = cfg.get("pbnc", {})
        lw = cfg.get("loss_weights", {})
        hfds_len = len(hfds)
        effective_batch = (
            cfg.get("per_device_batch_size", 1)
            * cfg.get("grad_accum", 1)
            * accel.num_processes
        )
        print("\n" + "=" * 70)
        print("  DiffUME — training run summary")
        print("=" * 70)
        print(f"  {'Config file':<30} {args.config}")
        print(f"  {'Output dir':<30} {cfg.get('output_dir')}")
        print()
        print("  BACKBONE")
        print(f"    {'Path':<28} {backbone_name}")
        print(f"    {'Architecture':<28} {getattr(bc, 'architectures', [getattr(bc, 'model_type', '?')])[0]}")
        print(f"    {'model_type':<28} {getattr(bc, 'model_type', '?')}")
        # Some VLM configs (e.g. Qwen2-VL) store text-side fields under bc.text_config.
        _tc = getattr(bc, "text_config", None)
        def _bc_get(name, default="?"):
            v = getattr(bc, name, None)
            if v is None and _tc is not None:
                v = getattr(_tc, name, None)
            return default if v is None else v
        _vocab = _bc_get("vocab_size", 0)
        print(f"    {'hidden_size':<28} {_bc_get('hidden_size')}")
        print(f"    {'num_hidden_layers':<28} {_bc_get('num_hidden_layers')}")
        print(f"    {'num_attention_heads':<28} {_bc_get('num_attention_heads')}")
        print(f"    {'vocab_size (after ext)':<28} {_vocab} + {ids['n_added']} → {(_vocab if isinstance(_vocab, int) else 0) + ids['n_added']}")
        print(f"    {'Total params':<28} {total_params / 1e9:.3f}B")
        print(f"    {'Trainable params':<28} {trainable_params / 1e9:.3f}B")
        print(f"    {'dtype':<28} {dtype}")
        print(f"    {'Gradient checkpointing':<28} {cfg.get('gradient_checkpointing', True)}")
        print()
        print("  SPECIAL TOKENS")
        for k, v in ids.items():
            print(f"    {k:<28} {v}")
        print()
        print("  DATASET")
        print(f"    {'Path':<28} {cfg['dataset_path']}")
        print(f"    {'Total samples':<28} {hfds_len}")
        if "max_train_samples" in cfg:
            print(f"    {'max_train_samples':<28} {cfg['max_train_samples']}")
        print(f"    {'max_length':<28} {cfg.get('max_length', 1024)}")
        print()
        print("  DIFFUSION")
        print(f"    {'block_size':<28} {cfg.get('block_size', 4)}")
        print(f"    {'scheduler_type':<28} {pbnc_cfg.get('scheduler_type', 'beta')}")
        print(f"    {'target_mean (μ)':<28} {pbnc_cfg.get('target_mean', 0.8)}")
        print(f"    {'start_C':<28} {pbnc_cfg.get('start_C', 2.0)}")
        print(f"    {'end_C':<28} {pbnc_cfg.get('end_C', 50.0)}")
        print(f"    {'pbnc warmup_steps':<28} {pbnc_cfg.get('warmup_steps', 8090)}")
        if pbnc_cfg.get('scheduler_type') == 'clamp':
            print(f"    {'noise_min':<28} {pbnc_cfg.get('noise_min', 0.0)}")
            print(f"    {'noise_max':<28} {pbnc_cfg.get('noise_max', 1.0)}")
        print()
        print("  LOSS WEIGHTS")
        for k, v in lw.items():
            print(f"    {k:<28} {v}")
        print()
        print("  TRAINING")
        print(f"    {'GPUs':<28} {accel.num_processes}")
        print(f"    {'mixed_precision':<28} {cfg.get('mixed_precision', 'bf16')}")
        print(f"    {'per_device_batch_size':<28} {cfg.get('per_device_batch_size', 1)}")
        print(f"    {'grad_accum':<28} {cfg.get('grad_accum', 1)}")
        print(f"    {'effective_batch_size':<28} {effective_batch}")
        print(f"    {'lr':<28} {cfg.get('lr', 1e-5)}")
        print(f"    {'warmup_steps':<28} {warmup}")
        print(f"    {'max_steps':<28} {total_steps}")
        print(f"    {'grad_clip':<28} {cfg.get('grad_clip', 1.0)}")
        print(f"    {'weight_decay':<28} {cfg.get('weight_decay', 0.0)}")
        print(f"    {'log_every':<28} {cfg.get('log_every', 10)}")
        print(f"    {'save_every':<28} {cfg.get('save_every', 2000)}")
        print("=" * 70 + "\n")
    # ───────────────────────────────────────────────────────────────────
    accel.print(f"Starting training: total_steps={total_steps}")
    log_every  = cfg.get("log_every", 10)
    save_every = cfg.get("save_every", 2000)

    # Rolling window for smoothed-loss display (last 20 steps).
    _loss_window: collections.deque = collections.deque(maxlen=20)

    pbar = tqdm(
        total=total_steps,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
        disable=not accel.is_main_process,
        bar_format=(
            "{desc}: {percentage:3.0f}%|{bar}| "
            "{n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}] "
            "{postfix}"
        ),
    )

    step = 0
    t0 = time.time()
    while step < total_steps:
        for batch in loader:
            with accel.accumulate(model):
                losses = step_fn(batch, step=step)
                accel.backward(losses["loss"])
                if accel.sync_gradients:
                    accel.clip_grad_norm_(model.parameters(),
                                          cfg.get("grad_clip", 1.0))
                optim.step()
                sched.step()
                optim.zero_grad()

            if accel.is_main_process:
                _loss_window.append(float(losses["loss"]))
                smooth_loss = sum(_loss_window) / len(_loss_window)
                dt = time.time() - t0
                steps_done = step + 1
                sps = steps_done / max(dt, 1e-6)          # steps / sec
                remaining = (total_steps - steps_done) / max(sps, 1e-6)  # seconds
                lr_now = sched.get_last_lr()[0]

                # Always update the progress-bar postfix so ETA stays live.
                pbar.set_postfix(
                    loss=f"{smooth_loss:.4f}",
                    diff_q=f"{float(losses.get('loss_diff_q', 0)):.3f}",
                    disc=f"{float(losses.get('loss_disc', 0)):.3f}",
                    gen=f"{float(losses.get('loss_gen', 0)):.3f}",
                    lr=f"{lr_now:.2e}",
                    eta=_fmt_time(remaining),
                    refresh=False,
                )
                pbar.update(1)

                # JSON log line at log_every cadence (machine-readable).
                if step % log_every == 0:
                    payload = {k: float(v) for k, v in losses.items()}
                    payload.update({
                        "step": step,
                        "lr": lr_now,
                        "throughput_steps_per_sec": sps,
                    })
                    tqdm.write(json.dumps(payload))

                if step > 0 and step % save_every == 0:
                    tqdm.write(f"[step {step}] Saving checkpoint ...")
                    accel.save_state(str(out_dir / f"step-{step:07d}"))

            step += 1
            if step >= total_steps:
                break

    pbar.close()
    if accel.is_main_process:
        accel.save_state(str(out_dir / "final"))
        elapsed = time.time() - t0
        print(f"\nTraining complete in {_fmt_time(elapsed)}  "
              f"({total_steps / elapsed:.2f} steps/sec avg)")
    accel.end_training()


def _fmt_time(seconds: float) -> str:
    """Format seconds as Xh Ym Zs (or Ym Zs / Zs for short durations)."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


if __name__ == "__main__":
    main()
