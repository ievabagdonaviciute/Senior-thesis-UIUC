#!/usr/bin/env python3
import json, math
from pathlib import Path
from collections import defaultdict

TYPES  = ["G","T","C","GC","GT","TC","GTC"]
SPLITS = [1, 2, 3]

# Where your per-split eval outputs live (edit if needed)
MODEL_ROOTS = {
    "internvl": Path("/home/ievab2/run_models/PHYSION_MULTIVIEW4PAIRS_AUXILIARY_TRAIN/internvl/results/epochs5"),
    "qwen":     Path("/home/ievab2/run_models/PHYSION_MULTIVIEW4PAIRS_AUXILIARY_TRAIN/qwen/results/epochs5"),
}

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return (None, None)
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return (m, 0.0)
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)  # sample std
    return (m, math.sqrt(var))

def summarize_one_file(path: Path):
    """
    Returns:
      per_cat: dict[category] -> {"correct": int, "total": int}
    Correct iff answer == model_output_norm (both lowercased).
    """
    per_cat = defaultdict(lambda: {"correct": 0, "total": 0})

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue

            cat = r.get("category")
            gt  = (r.get("answer") or "").strip().lower()
            pr  = (r.get("model_output_norm") or "").strip().lower()

            if not cat:
                continue
            if gt not in ("yes", "no"):
                continue
            if pr not in ("yes", "no"):
                continue

            per_cat[cat]["total"] += 1
            if gt == pr:
                per_cat[cat]["correct"] += 1

    return per_cat

def write_type_summary(model_name: str, root: Path, t: str):
    out_path = root / f"{t}_allsplit_results_myown.jsonl"
    rows_out = []

    # collect split accuracies per category for mean/std
    accs_by_cat = defaultdict(list)

    # also compute ALL per split
    for s in SPLITS:
        in_path = root / f"{t}_SPLIT{s}_out.jsonl"
        per_cat = summarize_one_file(in_path)
        if per_cat is None:
            print(f"[WARN] missing: {in_path}")
            continue

        split_tag = f"SPLIT{s}"

        # per category
        all_correct = 0
        all_total   = 0

        for cat in sorted(per_cat.keys()):
            c = per_cat[cat]["correct"]
            n = per_cat[cat]["total"]
            acc = (c / n) if n > 0 else None

            all_correct += c
            all_total   += n

            rows_out.append({
                "split": split_tag,
                "category": cat,
                "accuracy": acc,
                "correct": c,
                "total": n,
            })
            accs_by_cat[cat].append(acc)

        # ALL
        all_acc = (all_correct / all_total) if all_total > 0 else None
        rows_out.append({
            "split": split_tag,
            "category": "ALL",
            "accuracy": all_acc,
        })
        accs_by_cat["ALL"].append(all_acc)

        # blank line separator (like your example)
        rows_out.append(None)

    # mean/std over splits
    for cat in sorted(accs_by_cat.keys()):
        m, sd = mean_std(accs_by_cat[cat])
        rows_out.append({
            "category": cat,
            "mean_accuracy": m,
            "std_accuracy": sd,
            "type": "mean_std_over_splits",
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows_out:
            if r is None:
                f.write("\n")
            else:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {out_path}")

def main():
    for model_name, root in MODEL_ROOTS.items():
        print(f"\n=== {model_name} @ {root} ===")
        for t in TYPES:
            write_type_summary(model_name, root, t)

if __name__ == "__main__":
    main()
