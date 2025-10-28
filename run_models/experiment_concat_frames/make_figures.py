#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# ===================== Paths & Config =====================
BASE = Path("/home/ievab2/run_models/experiment_concat_frames")
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SUB_8  = "experiment_og_concat_8"
SUB_32 = "experiment_og_concat_32"

# (model_key, folder_name, display_name)
MODELS = [
    ("DEEPSEEK_TINY",     "DEEPSEEK",           "DeepSeek-VL2-Tiny"),
    ("QWEN",              "QWEN",               "Qwen2.5-VL-7B-Instruct"),
    ("VIDEOLLAVA_VICUNA", "VIDEOLLAVA_VICUNA",  "llava-v1.6-vicuna-13b-hf"),
    ("MOLMO",             "MOLMO",              "Molmo-7B-D"),
    ("INTERNVL",          "INTERNVL",           "InternVL2-8B"),
]

CATEGORIES = ["Descriptive", "Explanatory", "Predictive", "Counterfactual"]

# Colors (match your template)
COLORS = {
    "Descriptive":    "#1f77b4",  # blue
    "Explanatory":    "#ff7f0e",  # orange
    "Predictive":     "#2ca02c",  # green
    "Counterfactual": "#d62728",  # red
}

# ===================== Helpers =====================
def scores_json_path(folder_name: str, model_key: str, subdir: str) -> Path:
    return BASE / folder_name / subdir / "scores" / f"{model_key.lower()}_final_scores.jsonl"

def per_question_jsonl_path(folder_name: str, model_key: str, subdir: str) -> Path:
    return BASE / folder_name / subdir / "scores" / f"{model_key.lower()}_per_question.jsonl"

def load_final_scores(folder_name: str, model_key: str, subdir: str):
    p = scores_json_path(folder_name, model_key, subdir)
    if not p.exists():
        print(f"[warn] missing final scores: {p}")
        return None
    with p.open() as f:
        data = json.load(f)
    return {cat: float(data.get(cat, 0.0)) for cat in CATEGORIES}

def load_per_question_scores(folder_name: str, model_key: str, subdir: str):
    p = per_question_jsonl_path(folder_name, model_key, subdir)
    out = defaultdict(dict)
    if not p.exists():
        print(f"[warn] missing per-question file: {p}")
        return out
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except Exception:
                continue
            cat = (ex.get("category") or "").strip().lower()
            if   cat == "descriptive":    cat_key = "Descriptive"
            elif cat == "explanatory":    cat_key = "Explanatory"
            elif cat == "predictive":     cat_key = "Predictive"
            elif cat == "counterfactual": cat_key = "Counterfactual"
            else:
                continue
            qid = ex.get("question_id") or ex.get("qid")
            if qid is None:
                continue
            try:
                sc = float(ex.get("score", 0.0))
            except Exception:
                sc = 0.0
            out[cat_key][str(qid)] = sc
    return out

def add_value_labels(ax, bars, fmt="{:.1f}%"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(100*h),
                    xy=(b.get_x() + b.get_width()/2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

def make_bar_figure(title, models_display, cat_means, outfile):
    categories = CATEGORIES
    n_models = len(models_display)
    n_cats = len(categories)

    x = np.arange(n_models)
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, cat in enumerate(categories):
        offsets = x + (i - (n_cats-1)/2)*width
        vals = [cat_means.get(m, {}).get(cat, 0.0) for m in models_display]
        bars = ax.bar(offsets, vals, width, label=cat, color=COLORS[cat])
        add_value_labels(ax, bars)

    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models_display, rotation=15, ha='right')
    ax.set_ylim(0, 0.9)
    ax.legend(loc="upper left")
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"[fig] wrote {outfile}")

def best_model_fail_subanalysis(subdir: str, outfile: Path):
    # Load means
    means = {}
    for model_key, folder_name, disp in MODELS:
        s = load_final_scores(folder_name, model_key, subdir)
        if s is not None:
            means[disp] = s
    if not means:
        print(f"[warn] no means for {subdir}; skipping subanalysis.")
        return

    # Per-question scores
    perq = {}
    for model_key, folder_name, disp in MODELS:
        perq[disp] = load_per_question_scores(folder_name, model_key, subdir)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for idx, cat in enumerate(CATEGORIES):
        ax = axes[idx]
        best_disp = max(means.keys(), key=lambda d: means[d].get(cat, 0.0))
        best_map = perq.get(best_disp, {}).get(cat, {})

        fail_qids = {qid for qid, sc in best_map.items() if sc < 1.0}
        n_fail = len(fail_qids)

        labels, vals = [], []
        for model_key, folder_name, disp in MODELS:
            m_map = perq.get(disp, {}).get(cat, {})
            if n_fail == 0:
                val = 0.0
            else:
                s = sum(float(m_map.get(q, 0.0)) for q in fail_qids)
                val = s / n_fail
            labels.append(disp)
            vals.append(val)

        x = np.arange(len(labels))
        bars = ax.bar(x, vals, color=COLORS[cat])
        ax.set_title(f"{cat}: others when {best_disp} fails (n={n_fail})")
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        add_value_labels(ax, bars, fmt="{:.0f}%")
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"[fig] wrote {outfile}")


# ===== Success-only (0/1) category means from per-question JSONLs =====
def compute_success_only_means(subdir: str):
    """
    For each model and category, compute mean over questions of 1.0 if score==1.0 else 0.0.
    Uses the per-question JSONLs already dumped in .../scores/<model>_per_question.jsonl.
    Returns (models_display, cat_means_dict) where cat_means_dict maps display->category->mean.
    """
    cat_means = {}
    displays = []

    for model_key, folder_name, disp in MODELS:
        perq = load_per_question_scores(folder_name, model_key, subdir)
        if not perq:  # missing file/log → skip this model
            print(f"[warn] no per-question scores for {disp} in {subdir}")
            continue

        per_cat = {}
        for cat in CATEGORIES:
            scores = perq.get(cat, {}).values()
            n = 0
            s = 0
            for sc in scores:
                try:
                    val = float(sc)
                except Exception:
                    val = 0.0
                n += 1
                if val >= 1.0:  # success-only criterion
                    s += 1
            per_cat[cat] = (s / n) if n > 0 else 0.0

        cat_means[disp] = per_cat
        displays.append(disp)

    return displays, cat_means

# ===== Part 5: Delta table (32 frames minus 8 frames) =====
# ===== Part 5: Δ table (32 frames minus 8 frames) =====
def make_delta_table(cat_means_8, displays_8, cat_means_32, displays_32, outfile_png: Path):
    """
    Build a matplotlib table showing Δ(32 - 8) in percentage points
    for each model × category and save to PNG.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # only keep models that exist in both 8 and 32 frame runs
    models = [m for m in displays_8 if m in displays_32]
    categories = ["Descriptive", "Explanatory", "Predictive", "Counterfactual"]

    # compute deltas (in % points)
    table_data = []
    for m in models:
        row = []
        for cat in categories:
            v8  = cat_means_8.get(m, {}).get(cat, None)
            v32 = cat_means_32.get(m, {}).get(cat, None)
            if v8 is None or v32 is None:
                row.append("—")
            else:
                delta_pp = (v32 - v8) * 100
                row.append(f"{delta_pp:+.1f}")
        table_data.append(row)

    # create matplotlib table
    fig, ax = plt.subplots(figsize=(10, 0.7 + 0.45 * len(models)))
    ax.axis("off")

    col_labels = [f"{c} Δ(32–8) pp" for c in categories]
    table = ax.table(
        cellText=table_data,
        rowLabels=models,
        colLabels=col_labels,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)

    # bold header
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold")

    fig.tight_layout()
    fig.savefig(outfile_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[table] wrote {outfile_png}")



# ===================== Build & Plot =====================
def main():
    # ----- 8 frames -----
    cat_means_8 = {}
    displays_8 = []
    for model_key, folder_name, disp in MODELS:
        s = load_final_scores(folder_name, model_key, SUB_8)
        if s:
            displays_8.append(disp)
            cat_means_8[disp] = s
    if cat_means_8:
        make_bar_figure("Accuracy by Category (CLEVRER) - with partial credit [8 frames]",
                        displays_8, cat_means_8,
                        FIG_DIR / "accuracy_by_category_8frames.png")
        best_model_fail_subanalysis(SUB_8, FIG_DIR / "subanalysis_best_fails_8frames.png")
    else:
        print("[warn] no data for 8 frames; skipping.")

    # ----- 32 frames -----
    cat_means_32 = {}
    displays_32 = []
    for model_key, folder_name, disp in MODELS:
        s = load_final_scores(folder_name, model_key, SUB_32)
        if s:
            displays_32.append(disp)
            cat_means_32[disp] = s
    if cat_means_32:
        make_bar_figure("Accuracy by Category (CLEVRER) - with partial credit [32 frames]",
                        displays_32, cat_means_32,
                        FIG_DIR / "accuracy_by_category_32frames.png")
        best_model_fail_subanalysis(SUB_32, FIG_DIR / "subanalysis_best_fails_32frames.png")
    else:
        print("[warn] no data for 32 frames; skipping.")

    # ===== Part 4: Build success-only (0/1) accuracy figures =====
    # ----- 8 frames success-only -----
    displays_succ8, cat_means_succ8 = compute_success_only_means(SUB_8)
    if cat_means_succ8:
        make_bar_figure("Accuracy by Category (CLEVRER) - no partial credit (only score == 1.0) [8 frames]",
                        displays_succ8, cat_means_succ8,
                        FIG_DIR / "accuracy_by_category_success_only_8frames.png")
    else:
        print("[warn] no success-only data for 8 frames; skipping.")

    # ----- 32 frames success-only -----
    displays_succ32, cat_means_succ32 = compute_success_only_means(SUB_32)
    if cat_means_succ32:
        make_bar_figure("Accuracy by Category (CLEVRER) - no partial credit (only score == 1.0) [32 frames]",
                        displays_succ32, cat_means_succ32,
                        FIG_DIR / "accuracy_by_category_success_only_32frames.png")
    else:
        print("[warn] no success-only data for 32 frames; skipping.")

    # ===== Part 5: Delta table (32 - 8) in percentage points =====
    if cat_means_8 and cat_means_32:
        make_delta_table(
            cat_means_8, displays_8,
            cat_means_32, displays_32,
            FIG_DIR / "delta_table_32_minus_8.png"
        )
    else:
        print("[warn] Cannot make Δ table – missing data.")


if __name__ == "__main__":
    main()
