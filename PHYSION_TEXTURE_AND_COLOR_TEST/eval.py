#!/usr/bin/env python3
import json
import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = ["internvl", "qwen"]
# TASKS = ["colors", "texture", "randomized_colors"]
TASKS = ["randomized_colors"]

SPLITS = [1, 2, 3]
EPOCHS = [1, 3, 5]

BASE_ROOT = Path("/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST")


def summarize_accuracy(path: Path) -> Optional[Dict[str, float]]:
    """
    Returns dict with keys:
      correct, total, accuracy
    Correct iff (answer == model_output_norm) after stripping + lowercasing.
    Only counts rows where both are in {'yes','no'}.
    """
    if not path.exists():
        return None

    correct = 0
    total = 0

    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue

            gt = (r.get("answer") or "").strip().lower()
            pr = (r.get("model_output_norm") or "").strip().lower()

            if gt not in ("yes", "no"):
                continue
            if pr not in ("yes", "no"):
                continue

            total += 1
            if gt == pr:
                correct += 1

    acc = (correct / total) if total > 0 else None
    return {"correct": correct, "total": total, "accuracy": acc}


def mean_sample_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)  # sample variance
    return m, math.sqrt(var)


def sample_std_err(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    _, sd = mean_sample_std(vals)
    return sd / math.sqrt(len(vals))


def write_results_jsonl(out_path: Path, base_res, split_epoch_res):
    """
    Writes:
      base row
      blank
      for epoch in EPOCHS:
        split rows
        blank
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def dump_row(f, row: Dict):
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with out_path.open("w", encoding="utf-8") as f:
        # base
        if base_res is None:
            dump_row(f, {"tag": "base", "missing": True})
        else:
            dump_row(f, {
                "tag": "base",
                "accuracy": base_res["accuracy"],
                "correct": base_res["correct"],
                "total": base_res["total"],
            })
        f.write("\n")

        # epochs blocks
        for ep in EPOCHS:
            for s in SPLITS:
                r = split_epoch_res.get((s, ep))
                tag = f"SPLIT{s}_epochs{ep}"
                if r is None:
                    dump_row(f, {"tag": tag, "missing": True})
                else:
                    dump_row(f, {
                        "tag": tag,
                        "accuracy": r["accuracy"],
                        "correct": r["correct"],
                        "total": r["total"],
                    })
            f.write("\n")


def save_bar_chart_png(out_png: Path, base_acc: Optional[float], epoch_accs: Dict[int, List[float]]):
    """
    Bars: base, e1, e3, e5
    Error bars: standard error over splits for e1/e3/e5
    """
    labels = ["base"] + [f"e{ep}" for ep in EPOCHS]
    means = []
    errs = []

    # base
    means.append(base_acc)
    errs.append(0.0)

    # epochs
    for ep in EPOCHS:
        m, _ = mean_sample_std(epoch_accs.get(ep, []))
        se = sample_std_err(epoch_accs.get(ep, []))
        means.append(m)
        errs.append(se if se is not None else 0.0)

    # If everything missing, still emit a plot (but it will be empty-ish)
    x = list(range(len(labels)))

    plt.figure(figsize=(8, 4.5))
    plt.bar(x, [0 if v is None else v for v in means], yerr=errs, capsize=6)
    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title(out_png.stem)

    # annotate values
    for i, v in enumerate(means):
        if v is None:
            plt.text(i, 0.02, "NA", ha="center", va="bottom")
        else:
            plt.text(i, min(0.98, v + 0.02), f"{v:.3f}", ha="center", va="bottom")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def run_one(model: str, task: str):
    model_root = BASE_ROOT / model / "results"
    task_dir = model_root / task

    # inputs
    base_path = task_dir / "base_results.jsonl"
    base_res = summarize_accuracy(base_path)

    split_epoch_res = {}
    epoch_accs = {ep: [] for ep in EPOCHS}

    for ep in EPOCHS:
        for s in SPLITS:
            p = task_dir / f"SPLIT{s}_epochs{ep}_results.jsonl"
            r = summarize_accuracy(p)
            if r is not None:
                split_epoch_res[(s, ep)] = r
                if r["accuracy"] is not None:
                    epoch_accs[ep].append(r["accuracy"])

    # outputs exactly where you requested:
    out_jsonl = model_root / f"{task}_results.jsonl"
    out_png  = model_root / f"{task}_results.png"

    write_results_jsonl(out_jsonl, base_res, split_epoch_res)
    save_bar_chart_png(out_png, base_res["accuracy"] if base_res else None, epoch_accs)

    print(f"[OK] {model}/{task}: wrote {out_jsonl}")
    print(f"[OK] {model}/{task}: wrote {out_png}")


def main():
    for model in MODELS:
        for task in TASKS:
            run_one(model, task)


if __name__ == "__main__":
    main()