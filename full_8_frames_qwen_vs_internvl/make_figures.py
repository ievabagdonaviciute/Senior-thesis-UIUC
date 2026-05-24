#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# ===================== Paths & Config =====================
BASE = Path("/home/ievab2/run_models/full_8_frames_qwen_vs_internvl")
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
COLORS = {
    "Descriptive":    "#1f77b4",
    "Explanatory":    "#ff7f0e",
    "Predictive":     "#2ca02c",
    "Counterfactual": "#d62728",
}
MODELS = [
    ("QWEN",     "QWEN",     "Qwen2.5-VL-7B-Instruct"),
    ("INTERNVL", "INTERNVL", "InternVL2-8B"),
]
CATEGORIES = ["Descriptive", "Explanatory", "Predictive", "Counterfactual"]

def scores_json_path(folder_name: str, model_key: str) -> Path:
    return BASE / folder_name / "scores" / f"{model_key.lower()}_final_scores.jsonl"

def per_question_jsonl_path(folder_name: str, model_key: str) -> Path:
    return BASE / folder_name / "scores" / f"{model_key.lower()}_per_question.jsonl"

def _load_first_json_line(p: Path):
    if not p.exists():
        return None
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    return json.loads(line)
                except Exception:
                    return None
    return None

def load_final_scores(folder_name: str, model_key: str):
    p = scores_json_path(folder_name, model_key)
    if not p.exists():
        print(f"[warn] missing final scores: {p}")
        return None
    try:
        with p.open() as f:
            data = json.load(f)           # supports pretty-printed single-JSON files
    except Exception:
        # fallback: try first non-empty JSONL line (if you ever change format)
        try:
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        break
                else:
                    return None
        except Exception:
            print(f"[warn] invalid final scores: {p}")
            return None
    return {cat: float(data.get(cat, 0.0)) for cat in CATEGORIES}

def load_per_question_scores(folder_name: str, model_key: str):
    p = per_question_jsonl_path(folder_name, model_key)
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
            cat_map = {"descriptive":"Descriptive","explanatory":"Explanatory","predictive":"Predictive","counterfactual":"Counterfactual"}
            cat_key = cat_map.get(cat)
            if not cat_key:
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
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right")
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"[fig] wrote {outfile}")

    
def best_model_fail_subanalysis_no_subdir(outfile: Path):
    # means by model
    means = {}
    for model_key, folder_name, disp in MODELS:
        s = load_final_scores(folder_name, model_key)
        if s is not None:
            means[disp] = s
    if not means:
        print("[warn] no means; skipping subanalysis.")
        return

    # per-question maps
    perq = {disp: load_per_question_scores(folder_name, model_key)
            for model_key, folder_name, disp in MODELS}

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
            val = (sum(float(m_map.get(q, 0.0)) for q in fail_qids) / n_fail) if n_fail else 0.0
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
def compute_success_only_means_no_subdir():
    cat_means = {}
    displays = []
    for model_key, folder_name, disp in MODELS:
        perq = load_per_question_scores(folder_name, model_key)
        if not perq:
            print(f"[warn] no per-question scores for {disp}")
            continue
        per_cat = {}
        for cat in CATEGORIES:
            scores = perq.get(cat, {}).values()
            n = 0; s = 0
            for sc in scores:
                try:
                    val = float(sc)
                except Exception:
                    val = 0.0
                n += 1
                s += 1 if val >= 1.0 else 0
            per_cat[cat] = (s / n) if n else 0.0
        cat_means[disp] = per_cat
        displays.append(disp)
    return displays, cat_means


# ===================== Build & Plot =====================
def main():
    # Load means
    cat_means = {}
    displays = []
    for model_key, folder_name, disp in MODELS:
        s = load_final_scores(folder_name, model_key)
        if s:
            displays.append(disp)
            cat_means[disp] = s
    if cat_means:
        make_bar_figure("Accuracy by Category (CLEVRER)",
                        displays, cat_means,
                        FIG_DIR / "accuracy_by_category.png")
        # subanalysis without subdir:
        best_model_fail_subanalysis_no_subdir(FIG_DIR / "subanalysis_best_fails.png")

    else:
        print("[warn] no data; skipping.")

    # success-only (0/1) accuracies
    displays_succ, cat_means_succ = compute_success_only_means_no_subdir()
    if cat_means_succ:
        make_bar_figure("Accuracy by Category (CLEVRER) - no partial credit",
                        displays_succ, cat_means_succ,
                        FIG_DIR / "accuracy_by_category_success_only.png")

if __name__ == "__main__":
    main()
