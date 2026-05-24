# excel_analysis.py
import pandas as pd
from upsetplot import UpSet, from_indicators
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------- CONFIG -----------------
XLSX = "/home/ievab2/run_models/analysis/per_question_scores.xlsx"

QWEN_COL = "Qwen2.5-VL-7B-Instruct"
MODELS = ["DeepSeek-VL2-Tiny", "Video-LLaVA-7B-hf", "SmolVLM2-2.2B-Instruct"]

CATEGORIES = ["Descriptive", "Explanatory", "Predictive", "Counterfactual"]
CATEGORY_THRESHOLDS = {
    "Descriptive": 1.0,
    "Explanatory": 1.0,
    "Predictive":   1.0,
    "Counterfactual": 1.0,
}

OUTDIR = Path("/home/ievab2/run_models/analysis/upset_qwen_fail")
OUTDIR.mkdir(parents=True, exist_ok=True)
# ------------------------------------------

# Load & clean
df = pd.read_excel(XLSX)
df.columns = df.columns.astype(str).str.strip()
df = df.dropna(how="all")

if "Category" not in df.columns:
    raise KeyError("Column 'Category' not found. Check header spelling/case.")

# Forward-fill Category (Excel merged cells)
df["Category"] = df["Category"].ffill().astype(str).str.strip()

# Ensure numeric for Qwen + other models
for col in [QWEN_COL] + MODELS:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in sheet.")
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---- Per-category UpSet plots + per-category rescue rates ----
summary_rows = []
total_qwen_fail_rows = 0
total_rescues_per_model = {m: 0 for m in MODELS}

present_cats = [c for c in CATEGORIES if c in set(df["Category"])]

for cat in present_cats:
    thr = CATEGORY_THRESHOLDS.get(cat, 0.5)
    sub = df[df["Category"] == cat].copy()

    # Qwen fails under the category-specific threshold
    subfails = sub[sub[QWEN_COL] < thr].copy()
    n_fails = len(subfails)
    if n_fails == 0:
        print(f"[warn] No Qwen fails in category {cat} (thr={thr}), skipping plot.")
        continue

    # Build boolean success indicators for other models using SAME threshold
    for m in MODELS:
        subfails[m] = (subfails[m] >= thr).fillna(False).astype(bool)

    # Save counts for summary
    row = {"Category": cat, "Qwen_fail_rows": n_fails}
    for m in MODELS:
        # Number of times model succeeded among Qwen-fail rows (this category)
        rescues = int(subfails[m].sum())
        row[m] = rescues / n_fails  # proportion
        total_rescues_per_model[m] += rescues
    summary_rows.append(row)

    total_qwen_fail_rows += n_fails

    # UpSet plot
    inds = subfails[MODELS].apply(lambda s: s.astype(bool)).fillna(False)
    upset_data = from_indicators(MODELS, inds)
    UpSet(upset_data, sort_by="cardinality", show_counts=True).plot()
    plt.suptitle(f"Who Succeeds When Qwen Fails – {cat} (thr={thr})", fontsize=14)

    out_png = OUTDIR / f"qwen_fail_upset_{cat}.png"
    plt.savefig(out_png.as_posix(), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_png}")

# ---- Overall summary (weighted across categories) ----
if total_qwen_fail_rows > 0:
    overall_row = {"Category": "Overall", "Qwen_fail_rows": total_qwen_fail_rows}
    for m in MODELS:
        overall_row[m] = total_rescues_per_model[m] / total_qwen_fail_rows
    summary_rows.append(overall_row)

# Write summary CSV
if summary_rows:
    summary = pd.DataFrame(summary_rows)
    # nice ordering: Category, Qwen_fail_rows, models…
    summary = summary[["Category", "Qwen_fail_rows"] + MODELS]
    out_csv = OUTDIR / "rescue_rates_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")
else:
    print("[warn] No categories had Qwen-fail rows under the specified thresholds.")
