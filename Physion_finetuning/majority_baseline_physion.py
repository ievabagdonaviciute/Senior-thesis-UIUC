#!/usr/bin/env python3
import json
import os
from collections import Counter

BASE = "/home/ievab2/run_models/Physion_dataset/physion_out_questions"
OUT_DIR = "/home/ievab2/run_models/Physion_finetuning/majority_baseline"

TASKS = [
    ("Dominoes", "dominoes"),
    ("Contain",  "contain"),
    ("Drop",     "drop"),
]

def load_labels(path):
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            lab = str(row.get("ground_truth", "")).strip().lower()
            labels.append(lab)
    return labels

def main(split_kind="past"):
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"majority_{split_kind}.jsonl")

    print(f"[INFO] Using Physion JSONLs for split_kind='{split_kind}':")
    results = []
    all_labels = []

    # ---- per-task stats ----
    for pretty_name, fname_prefix in TASKS:
        path = f"{BASE}/{pretty_name}/{fname_prefix}_{split_kind}.jsonl"
        print("  ", path)
        labs = load_labels(path)
        all_labels.extend(labs)
        cnt = Counter(labs)

        print(f"\n[PER-FILE] {path}")
        print("  label counts:", cnt)

        if cnt:
            maj, maj_n = cnt.most_common(1)[0]
            acc = maj_n / len(labs) * 100
            print(f"  majority='{maj}' → {acc:.2f}% for THIS file")

            results.append({
                "task": pretty_name,
                "split_kind": split_kind,
                "majority_label": maj,
                "accuracy_percent": acc,
                "total": len(labs),
                "label_counts": dict(cnt),
            })
        else:
            print("  [WARN] No labels found for this file!")

    # ---- global stats ----
    global_cnt = Counter(all_labels)
    print("\n[GLOBAL] Across all tasks")
    print("  label counts:", global_cnt)

    if global_cnt:
        maj, maj_n = global_cnt.most_common(1)[0]
        total = len(all_labels)
        acc = maj_n / total * 100
        print(f"  majority='{maj}' → {acc:.2f}% globally")

        results.append({
            "task": "GLOBAL",
            "split_kind": split_kind,
            "majority_label": maj,
            "accuracy_percent": acc,
            "total": total,
            "label_counts": dict(global_cnt),
        })
    else:
        print("  [WARN] No labels found globally!")

    # ---- write JSONL ----
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")

    print(f"\n[INFO] Majority baseline stats written to:\n  {out_path}")
    print(f"[INFO] Output directory: {OUT_DIR}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_kind", choices=["past", "pred"], required=True)
    args = ap.parse_args()
    main(args.split_kind)
