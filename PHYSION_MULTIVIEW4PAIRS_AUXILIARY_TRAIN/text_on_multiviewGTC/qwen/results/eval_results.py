#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------
# CONFIG
# ---------------------------------------

ROOT_DIR = "/home/ievab2/run_models/PHYSION_MULTIVIEW4PAIRS_AUXILIARY_TRAIN/text_on_multiviewGTC/qwen/results"

TYPES = ["C", "T", "G", "GT", "GC", "TC", "GTC"]
SPLITS = ["SPLIT1", "SPLIT2", "SPLIT3"]

BASE_PATH_TEMPLATE = os.path.join(ROOT_DIR, "base", "base_results_{dataset}.jsonl")

FILE_TEMPLATE = os.path.join(
    ROOT_DIR,
    "{type}_{split}_{dataset}_out.jsonl"
)


# ---------------------------------------
# HELPERS
# ---------------------------------------

def compute_accuracy(jsonl_path):
    correct = 0
    total = 0

    with open(jsonl_path, "r") as f:
        for line in f:
            item = json.loads(line)
            total += 1

            pred = item.get("model_output_norm", "").strip().lower()
            gt   = item.get("answer", "").strip().lower()

            # unknown counts as incorrect
            if pred == "unknown":
                continue

            if pred == gt:
                correct += 1

    if total == 0:
        return 0.0

    return correct / total


# ---------------------------------------
# MAIN
# ---------------------------------------

def main(dataset):

    if dataset not in ["contact", "geometry", "time"]:
        raise ValueError("Dataset must be 'contact' or 'geometry'")

    means = []
    errors = []
    labels = []

    # -----------------------------------
    # BASE MODEL
    # -----------------------------------

    base_path = BASE_PATH_TEMPLATE.format(dataset=dataset)
    print(f"Reading base model from: {base_path}")

    base_acc = compute_accuracy(base_path)
    means.append(base_acc)
    errors.append(0.0)   # no split variance for base
    labels.append("BASE")

    # -----------------------------------
    # FINETUNED TYPES
    # -----------------------------------

    for t in TYPES:
        split_accs = []

        for s in SPLITS:
            path = FILE_TEMPLATE.format(type=t, split=s, dataset=dataset)

            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")

            acc = compute_accuracy(path)
            split_accs.append(acc)

        mean_acc = np.mean(split_accs)
        std_error = np.std(split_accs, ddof=1) / np.sqrt(len(split_accs))

        means.append(mean_acc)
        errors.append(std_error)
        labels.append(t)

    # -----------------------------------
    # PLOT
    # -----------------------------------

    x = np.arange(len(labels))

    plt.figure(figsize=(10, 6))
    plt.bar(x, means, yerr=errors, capsize=5)
    plt.xticks(x, labels)
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy on {dataset.capitalize()} Dataset")
    plt.ylim(0, 1)

    plt.tight_layout()

    save_path = os.path.join(ROOT_DIR, f"final_results_{dataset}.png")
    plt.savefig(save_path)
    print(f"\nSaved figure to: {save_path}")


# ---------------------------------------
# ENTRY
# ---------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["contact", "geometry", "time"],
        help="Which dataset to evaluate"
    )

    args = parser.parse_args()
    main(args.dataset)