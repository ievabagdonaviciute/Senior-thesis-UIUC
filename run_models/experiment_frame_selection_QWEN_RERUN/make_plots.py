#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # safe backend for cluster
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# ---- INPUT FILES ----
files = [
    "/home/ievab2/run_models/experiment_frame_selection_QWEN_RERUN/total_score_results/deepseek_tiny_final_scores.jsonl",
    "/home/ievab2/run_models/experiment_frame_selection_QWEN_RERUN/total_score_results/qwen_final_scores.jsonl",
    "/home/ievab2/run_models/experiment_frame_selection_QWEN_RERUN/total_score_results/smolvlm_final_scores.jsonl",
    "/home/ievab2/run_models/experiment_frame_selection_QWEN_RERUN/total_score_results/videollava_final_scores.jsonl",
]

CATS = ["Descriptive", "Explanatory", "Predictive", "Counterfactual"]
TOTAL_KEY = "Total"

def load_first_object(path: str):
    """Load either a pretty-printed JSON file or JSONL file."""
    p = Path(path)
    text = p.read_text().strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No valid JSON object found in {path}")

# Collect results
models = []
per_cat = {cat: [] for cat in CATS}
totals = []

for fp in files:
    obj = load_first_object(fp)
    name = obj.get("model_name") or Path(fp).stem
    models.append(name)
    for cat in CATS:
        per_cat[cat].append(float(obj[cat]))
    totals.append(float(obj[TOTAL_KEY]))

# ---- Plot 1: Accuracy by category ----
x = np.arange(len(models))
width = 0.18
fig1, ax1 = plt.subplots(figsize=(10, 8))

bars = []
for i, cat in enumerate(CATS):
    offsets = x + (i - (len(CATS)-1)/2) * width
    b = ax1.bar(offsets, per_cat[cat], width, label=cat)
    bars.extend(b)

# Add percentage labels on each bar
for b in bars:
    height = b.get_height()
    ax1.text(
        b.get_x() + b.get_width() / 2,
        height + 0.01,
        f"{height*100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=8,
        rotation=90
    )

ax1.set_title("Accuracy by Category (CLEVRER): QWEN selects frames")
ax1.set_xlabel("Model")
ax1.set_ylabel("Accuracy")
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=15, ha="right")
ax1.legend()
ax1.grid(True, axis="y", linestyle="--", alpha=0.4)

# Fix y-axis to 0–80%
ax1.set_ylim(0, 0.8)
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

fig1.tight_layout(rect=[0, 0, 1, 0.95])
fig1.savefig("accuracy_by_category.png", dpi=200)

# ---- Plot 2: Total score per model ----
fig2, ax2 = plt.subplots(figsize=(8, 7))
bars2 = ax2.bar(models, totals)

# Add labels on totals
for b in bars2:
    height = b.get_height()
    ax2.text(
        b.get_x() + b.get_width() / 2,
        height + 0.01,
        f"{height*100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9
    )

ax2.set_title("Total Accuracy by Model (CLEVRER): QWEN selects frames")
ax2.set_xlabel("Model")
ax2.set_ylabel("Total Accuracy")
ax2.set_xticks(np.arange(len(models)))
ax2.set_xticklabels(models, rotation=15, ha="right")
ax2.grid(True, axis="y", linestyle="--", alpha=0.4)

# Fix y-axis to 0–70%
ax2.set_ylim(0, 0.7)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

fig2.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig("total_accuracy_by_model.png", dpi=200)

print("Saved plots:")
print("  - accuracy_by_category.png")
print("  - total_accuracy_by_model.png")
