"""Phase 0b — dataset audit.

Loads `umer1_sft_processed`, inspects schema of `qry`/`pos` sub-dicts
(columnar `conversations` = {"from":[...],"value":[...]}), checks
reasoning-span coverage, writes a markdown report.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)


def _turn(sub, role_aliases):
    convs = sub.get("conversations") or {}
    froms = convs.get("from") or []
    values = convs.get("value") or []
    for r, v in zip(froms, values):
        if r in role_aliases:
            return v
    return None


def get_assistant(sub):
    return _turn(sub, {"assistant", "gpt"})


def get_user(sub):
    return _turn(sub, {"user", "human"})


def inspect_side(d, side, n):
    has_think = has_answer = has_gen_emb = has_disc_emb = 0
    has_image = has_video = 0
    sub_keys = Counter()
    sample_assistant = sample_user = None
    for i in range(n):
        sub = d[i][side]
        if not isinstance(sub, dict):
            continue
        for k in sub:
            sub_keys[k] += 1
        if sub.get("image"):
            has_image += 1
        if sub.get("video"):
            has_video += 1
        a = get_assistant(sub)
        u = get_user(sub)
        if a:
            if THINK_RE.search(a):
                has_think += 1
            if ANSWER_RE.search(a):
                has_answer += 1
            if "<gen_emb>" in a:
                has_gen_emb += 1
            if sample_assistant is None:
                sample_assistant = a
        if u:
            if "<disc_emb>" in u:
                has_disc_emb += 1
            if sample_user is None:
                sample_user = u
    return {
        "n": n,
        "sub_keys": dict(sub_keys.most_common()),
        "has_think_pct": 100.0 * has_think / n,
        "has_answer_pct": 100.0 * has_answer / n,
        "has_gen_emb_pct": 100.0 * has_gen_emb / n,
        "has_disc_emb_pct_user": 100.0 * has_disc_emb / n,
        "has_image_pct": 100.0 * has_image / n,
        "has_video_pct": 100.0 * has_video / n,
        "sample_user": sample_user,
        "sample_assistant": sample_assistant,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    d = load_from_disk(args.path)
    n = min(args.n, len(d))

    name_counts = Counter(d[i]["dataset_name"] for i in range(min(20_000, len(d))))
    qry = inspect_side(d, "qry", n)
    pos = inspect_side(d, "pos", n)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("# Dataset report — `umer1_sft_processed`\n\n")
        f.write(f"- Path: `{args.path}`\n")
        f.write(f"- Total rows: **{len(d):,}**\n")
        f.write(f"- Columns: `{d.column_names}`\n")
        f.write(f"- Inspected rows per side: {n}\n\n")
        f.write("## `dataset_name` distribution (first 20k rows)\n\n")
        for name, c in name_counts.most_common(50):
            f.write(f"- `{name}`: {c}\n")
        f.write("\n")
        for label, side in [("QRY", qry), ("POS", pos)]:
            f.write(f"## {label} side\n\n")
            f.write(f"- sub-dict keys: `{side['sub_keys']}`\n")
            f.write(f"- has `<think>` block (assistant): **{side['has_think_pct']:.1f}%**\n")
            f.write(f"- has `<answer>` block (assistant): **{side['has_answer_pct']:.1f}%**\n")
            f.write(f"- has `<gen_emb>` token (assistant): **{side['has_gen_emb_pct']:.1f}%**\n")
            f.write(f"- has `<disc_emb>` token (user): **{side['has_disc_emb_pct_user']:.1f}%**\n")
            f.write(f"- has image: {side['has_image_pct']:.1f}%\n")
            f.write(f"- has video (placeholder): {side['has_video_pct']:.1f}%\n\n")
            f.write("### sample user turn\n\n```\n")
            f.write((side["sample_user"] or "(none)")[:1500])
            f.write("\n```\n\n")
            f.write("### sample assistant turn\n\n```\n")
            f.write((side["sample_assistant"] or "(none)")[:2000])
            f.write("\n```\n\n")

        gate_pass = (
            qry["has_think_pct"] >= 80
            and qry["has_answer_pct"] >= 80
            and pos["has_think_pct"] >= 80
            and pos["has_answer_pct"] >= 80
        )
        f.write("## Phase 0b gate\n\n")
        f.write(
            "- Reasoning-span coverage gate (≥80% on both sides for "
            "`<think>` and `<answer>`): "
            f"**{'PASS' if gate_pass else 'FAIL'}**\n"
        )

    print(f"Wrote {out}")
    summary = {
        "rows": len(d),
        "qry": {k: v for k, v in qry.items() if not k.startswith("sample")},
        "pos": {k: v for k, v in pos.items() if not k.startswith("sample")},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
