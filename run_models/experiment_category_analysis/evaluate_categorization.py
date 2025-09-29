#!/usr/bin/env python3
# evaluate_categorization.py
import argparse, json, re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

CATEGORIES = ["descriptive", "explanatory", "predictive", "counterfactual"]
LABEL_REGEX = re.compile(r"\b(descriptive|explanatory|predictive|counterfactual)\b", re.IGNORECASE)

def extract_label(text: str) -> str | None:
    if not text:
        return None
    matches = LABEL_REGEX.findall(text)
    # keep unique labels in lowercase
    labels = list({m.lower() for m in matches})
    if len(labels) == 1:
        return labels[0]
    return None  # either no match or multiple matches

def load_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # try to salvage lines with stray commas
                line = line.rstrip(",")
                yield json.loads(line)

def compute_accuracies(file_map: dict[str, Path]) -> dict[str, dict[str, float]]:
    """
    Returns: {model_name: {category: accuracy_float_0to1}}
    """
    # Counters: correct/total per model per category
    correct = defaultdict(lambda: defaultdict(int))
    total   = defaultdict(lambda: defaultdict(int))

    for model_name, path in file_map.items():
        for rec in load_jsonl(path):
            gt = (rec.get("category_answer") or rec.get("category") or "").strip().lower()
            pred = extract_label(str(rec.get("model_output", "")))
            if gt in CATEGORIES:
                total[model_name][gt] += 1
                if pred == gt:
                    correct[model_name][gt] += 1

        # ensure all categories present (even if zero)
        for c in CATEGORIES:
            total[model_name][c] += 0
            correct[model_name][c] += 0

    # compute accuracies
    acc = {}
    for model_name in file_map.keys():
        acc[model_name] = {}
        for c in CATEGORIES:
            t = total[model_name][c]
            acc[model_name][c] = (correct[model_name][c] / t) if t else 0.0
    return acc, correct, total

def plot_grouped_bar(acc: dict[str, dict[str, float]], out_png: Path, title: str = "Question Category Classification Accuracy"):
    models = list(acc.keys())
    n_models = len(models)
    n_cats = len(CATEGORIES)

    # shape data: rows=models, cols=categories
    data = np.array([[acc[m][c] for c in CATEGORIES] for m in models])  # shape (M, C)

    x = np.arange(n_models)
    width = 0.18  # bar width
    offsets = (np.arange(n_cats) - (n_cats - 1) / 2) * (width + 0.02)

    plt.figure(figsize=(12, 6.5))
    for i, cat in enumerate(CATEGORIES):
        plt.bar(x + offsets[i], data[:, i] * 100.0, width, label=cat.capitalize())
        # add value labels
        for j in range(n_models):
            val = data[j, i] * 100.0
            plt.text(x[j] + offsets[i], val + 1.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, models, rotation=0)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 105)
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate category classification and plot accuracies.")
    parser.add_argument("--deepseek", default="/home/ievab2/run_models/experiment_category_analysis/cat_recognition_results/deepseek_tiny_out.jsonl")
    parser.add_argument("--qwen",     default="/home/ievab2/run_models/experiment_category_analysis/cat_recognition_results/qwen_out.jsonl")
    parser.add_argument("--smolvlm",  default="/home/ievab2/run_models/experiment_category_analysis/cat_recognition_results/smolvlm_out.jsonl")
    parser.add_argument("--videollava", default="/home/ievab2/run_models/experiment_category_analysis/cat_recognition_results/videollava_out.jsonl")
    parser.add_argument("--out", default="/home/ievab2/run_models/experiment_category_analysis/cat_recognition_results/categorization_accuracy.png",
                        help="Output PNG path for the chart.")
    args = parser.parse_args()

    file_map = {
        "DeepSeek-tiny": Path(args.deepseek),
        "Qwen": Path(args.qwen),
        "SmolVLM": Path(args.smolvlm),
        "Video-LLaVA": Path(args.videollava),
    }

    # Sanity check files exist
    missing = [str(p) for p in file_map.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input file(s): {missing}")

    acc, correct, total = compute_accuracies(file_map)

    # Print a small summary to stdout
    print("Per-category accuracy (percent):")
    header = ["Model"] + [c.capitalize() for c in CATEGORIES]
    print("\t".join(header))
    for m in file_map.keys():
        row = [m] + [f"{acc[m][c]*100:.2f}" for c in CATEGORIES]
        print("\t".join(row))

    # Plot
    out_png = Path(args.out)
    plot_grouped_bar(acc, out_png)
    print(f"\nSaved grouped bar chart to: {out_png}")

if __name__ == "__main__":
    main()
