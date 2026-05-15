"""Tiny in-memory retrieval evaluator (cosine top-k).

Loads N qry/pos pairs from `umer1_sft_processed`, embeds each side with
the **clean encoder** (`encode`), and reports Recall@{1,5,10}.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from datasets import load_from_disk

from ..data.prompting import extract_turns
from ..inference.embed_disc import embed_pair, load_diffume


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-VL-3B-Instruct")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, proc, ids = load_diffume(args.checkpoint, args.backbone)
    model.to(device)

    d = load_from_disk(args.dataset)
    n = min(args.n, len(d))

    Z_q, Z_p = [], []
    for i in range(n):
        row = d[i]
        for side, sink in (("qry", Z_q), ("pos", Z_p)):
            sub = row[side]
            user, asst = extract_turns(sub)
            z_d, _ = embed_pair(model, tok, proc, ids,
                                user_text=user.replace("<video>", ""),
                                assistant_text=asst,
                                image=sub.get("image"), device=device)
            sink.append(z_d)

    Q = F.normalize(torch.stack(Z_q), dim=-1)
    P = F.normalize(torch.stack(Z_p), dim=-1)
    sim = Q @ P.T                           # (n, n)
    rank = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(n).unsqueeze(1)
    for k in (1, 5, 10):
        hit = (rank[:, :k] == gt).any(dim=-1).float().mean().item()
        print(f"Recall@{k}: {hit:.4f}")


if __name__ == "__main__":
    main()
