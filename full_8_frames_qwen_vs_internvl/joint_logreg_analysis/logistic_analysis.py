#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint logistic agreement analysis: Qwen vs InternVL
- Train two separate logistic regressions to predict correctness from shared embeddings
- Compare predicted probabilities, agreement, and learned coefficients

Outputs (in OUT_DIR):
  - per_question_probs.csv
  - agreement_by_category.csv
  - heatmap_joint_outcomes.png
  - scatter_prob_agreement.png
  - correlation_summary.png
  - console prints (Cohen's kappa, prob corr, coef corr)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

# ---------------------------- Config ----------------------------
BASE        = Path("/home/ievab2/run_models/full_8_frames_qwen_vs_internvl")
EMB_PATH    = BASE / "QWEN/embeddings/only_prompt/embeddings_eva02_l14.csv.gz"  # shared question embeddings
META_QWEN   = BASE / "QWEN/embeddings/only_prompt/prompts_meta.csv"
META_INTVL  = BASE / "INTERNVL/embeddings/only_prompt/prompts_meta.csv"
JSONL_PATH  = "/home/ievab2/run_models/questions/clevrer_filtered_500.jsonl"
OUT_DIR     = BASE / "joint_logreg_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Choose how to treat correctness labels:
# True  -> strict accuracy (1 only if correct == 1.0)
# False -> use partial credit as target (NOT recommended for logistic; keep True)
BINARY_ACCURACY = True

RANDOM_STATE = 42
N_FOLDS      = 10
CS_GRID      = np.logspace(-2, 2, 9)   # C values for LogisticRegressionCV

# --------------------- Load shared embeddings --------------------
print("Loading shared embeddings ...")
X_all = pd.read_csv(EMB_PATH, compression="gzip").to_numpy()
print(f"Embeddings shape: {X_all.shape}")

# --------------------- Load metas & labels -----------------------
def load_meta(path: Path, binary: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    if binary:
        y = (df["correct"].astype(float) >= 1.0).astype(int)
    else:
        # Using partial as target makes logistic less appropriate; keep binary recommended.
        y = (df["correct"].astype(float) >= 1.0).astype(int)
    out = pd.DataFrame({
        "row_idx": df["row_idx"].astype(int),
        "y": y
    })
    # keep prompt if available (for debugging/inspection)
    if "prompt" in df.columns:
        out["prompt"] = df["prompt"].astype(str)
    return out

mq = load_meta(META_QWEN,  BINARY_ACCURACY).rename(columns={"y":"y_qwen"})
mi = load_meta(META_INTVL, BINARY_ACCURACY).rename(columns={"y":"y_internvl"})

# Strict inner join on row_idx (ensure both models have the question)
merged = mq.merge(mi, on="row_idx", how="inner", suffixes=("_qwen_meta","_internvl_meta"))

# --------------------- Map category from JSONL -------------------
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    jlines = [json.loads(line) for line in f]
# Category is by line index; if JSONL has explicit row_idx, use that; else line-order
if "row_idx" in jlines[0]:
    cat_map = {int(rec["row_idx"]): rec.get("category", "") for rec in jlines}
else:
    cat_map = {i: rec.get("category", "") for i, rec in enumerate(jlines)}

merged["category"] = merged["row_idx"].map(cat_map).astype(str)

# ------------- Align embeddings to merged row order --------------
# Map row_idx -> row position in embeddings using QWEN meta order
mq_full = pd.read_csv(META_QWEN)
order_map = {int(ri): i for i, ri in enumerate(mq_full["row_idx"].tolist())}
pos = [order_map[ri] for ri in merged["row_idx"]]
X = X_all[pos, :]

# Targets
y_q = merged["y_qwen"].to_numpy()
y_i = merged["y_internvl"].to_numpy()

print(f"Merged questions: {X.shape[0]}")

# ----------------------- Train two LR models ---------------------
# Same scaler for both (so coefficients are comparable)
scaler = StandardScaler()
Xz = scaler.fit_transform(X)

# Stratified CV (accuracy)
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Qwen model
lr_qwen = LogisticRegressionCV(
    Cs=CS_GRID, cv=cv, penalty="l2", solver="liblinear",
    scoring="accuracy", max_iter=2000, refit=True
)
lr_qwen.fit(Xz, y_q)

# InternVL model
lr_internvl = LogisticRegressionCV(
    Cs=CS_GRID, cv=cv, penalty="l2", solver="liblinear",
    scoring="accuracy", max_iter=2000, refit=True
)
lr_internvl.fit(Xz, y_i)

# Predicted probabilities (P(correct))
p_qwen     = lr_qwen.predict_proba(Xz)[:, 1]
p_internvl = lr_internvl.predict_proba(Xz)[:, 1]

# -------------------- Agreement & correlations -------------------
# Four-way agreement cases
q = y_q.astype(int)
v = y_i.astype(int)

agreement_case = np.full(len(q), "both_wrong", dtype=object)
agreement_case[(q==1) & (v==1)] = "both_correct"
agreement_case[(q==1) & (v==0)] = "qwen_only"
agreement_case[(q==0) & (v==1)] = "internvl_only"

# Cohen's kappa on outcomes
kappa = cohen_kappa_score(q, v)

# Probability correlation
prob_corr = np.corrcoef(p_qwen, p_internvl)[0,1]

# Coefficient correlation (same scaling & penalty → comparable)
beta_q = lr_qwen.coef_.ravel()
beta_i = lr_internvl.coef_.ravel()
coef_corr = np.corrcoef(beta_q, beta_i)[0,1]

print("\n=== Joint Agreement Summary ===")
print(f"Cohen's kappa (Qwen vs InternVL): {kappa:.3f}")
print(f"Corr(P_qwen, P_internvl):        {prob_corr:.3f}")
print(f"Corr(beta_qwen, beta_internvl):  {coef_corr:.3f}")

# ---------------------- Save per-question CSV --------------------
per_question = pd.DataFrame({
    "row_idx": merged["row_idx"].to_numpy(),
    "category": merged["category"].to_numpy(),
    "y_qwen": q,
    "y_internvl": v,
    "p_qwen": p_qwen,
    "p_internvl": p_internvl,
    "agreement_case": agreement_case
})
if "prompt" in merged.columns:
    per_question["prompt"] = merged["prompt"]

per_question_path = OUT_DIR / "per_question_probs.csv"
per_question.to_csv(per_question_path, index=False)
print(f"Saved: {per_question_path}")

# --------------- Agreement by category (counts & %) --------------
tab = (
    per_question
    .groupby(["category", "agreement_case"])
    .size()
    .unstack(fill_value=0)
    .sort_index()
)
tab["total"] = tab.sum(axis=1)
for col in tab.columns:
    if col != "total":
        tab[col + "_pct"] = (tab[col] / tab["total"]).round(3)

agreement_cat_path = OUT_DIR / "agreement_by_category.csv"
tab.to_csv(agreement_cat_path)
print(f"Saved: {agreement_cat_path}")

# ---------------------- Figure 1: 2x2 Heatmap --------------------
# matrix of joint outcomes normalized by all samples
# rows: Qwen (0/1), cols: InternVL (0/1)
m00 = np.mean((q==0) & (v==0))
m01 = np.mean((q==0) & (v==1))
m10 = np.mean((q==1) & (v==0))
m11 = np.mean((q==1) & (v==1))

fig1 = plt.figure(figsize=(4.6, 4.2))
ax1 = fig1.add_subplot(111)
heat = np.array([[m00, m01],
                 [m10, m11]])
im = ax1.imshow(heat, cmap="Blues", vmin=0, vmax=heat.max())
for i in range(2):
    for j in range(2):
        ax1.text(j, i, f"{heat[i,j]:.2f}", ha="center", va="center", color="black", fontsize=12)
ax1.set_xticks([0,1]); ax1.set_yticks([0,1])
ax1.set_xticklabels(["InternVL=0", "InternVL=1"])
ax1.set_yticklabels(["Qwen=0", "Qwen=1"])
ax1.set_title("Joint outcomes (fraction of all questions)")
plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
plt.tight_layout()
heatmap_path = OUT_DIR / "heatmap_joint_outcomes.png"
plt.savefig(heatmap_path, dpi=200)
plt.close(fig1)
print(f"Saved: {heatmap_path}")

# ---------------- Figure 1B: Category-specific heatmaps ----------------
# Create one 2x2 heatmap per category (normalized within each category),
# plus a combined 2x2 panel for quick comparison.

bycat_dir = OUT_DIR / "by_category"
bycat_dir.mkdir(parents=True, exist_ok=True)

# Preferred order for CLEVRER-style categories; will auto-filter to those present
preferred_order = ["descriptive", "explanatory", "predictive", "counterfactual"]
present_cats = [c for c in preferred_order if c in per_question["category"].unique().tolist()]
if not present_cats:
    present_cats = sorted(per_question["category"].unique().tolist())

# Compute matrices first to share a common color scale (vmax) across all cats
cat_mats = {}
global_max = 0.0

for cat in present_cats:
    sub = per_question[per_question["category"] == cat]
    if len(sub) == 0:
        continue

    q_cat = sub["y_qwen"].to_numpy().astype(int)
    v_cat = sub["y_internvl"].to_numpy().astype(int)

    # Fractions normalized by the size of this category
    m00 = np.mean((q_cat == 0) & (v_cat == 0))
    m01 = np.mean((q_cat == 0) & (v_cat == 1))
    m10 = np.mean((q_cat == 1) & (v_cat == 0))
    m11 = np.mean((q_cat == 1) & (v_cat == 1))
    mat = np.array([[m00, m01],
                    [m10, m11]])

    cat_mats[cat] = mat
    global_max = max(global_max, mat.max())

# Fallback in case of empty (shouldn't happen if data exists)
if global_max == 0:
    global_max = 1.0

# 1) Save one PNG per category
for cat, mat in cat_mats.items():
    fig = plt.figure(figsize=(4.6, 4.2))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=global_max)

    # Annotate cells
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=12)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["InternVL=0", "InternVL=1"])
    ax.set_yticklabels(["Qwen=0", "Qwen=1"])
    ax.set_title(f"Joint outcomes — {cat} (fraction within category)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    cat_fname = f"heatmap_joint_outcomes_{cat.replace(' ', '_')}.png"
    plt.savefig(bycat_dir / cat_fname, dpi=200)
    plt.close(fig)
    print(f"Saved: {bycat_dir / cat_fname}")

# 2) Optional: a single 2×2 panel with all categories (if exactly 4 are present)
if len(cat_mats) == 4:
    panel_order = [c for c in preferred_order if c in cat_mats]
    fig = plt.figure(figsize=(9.6, 8.4))
    axs = fig.subplots(2, 2)
    axs = axs.ravel()

    for ax, cat in zip(axs, panel_order):
        mat = cat_mats[cat]
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=global_max)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="black", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["InternVL=0", "InternVL=1"])
        ax.set_yticklabels(["Qwen=0", "Qwen=1"])
        ax.set_title(cat, fontsize=12)

    # One shared colorbar
    cbar = fig.colorbar(im, ax=axs.tolist(), fraction=0.025, pad=0.02)
    fig.suptitle("Joint outcomes by category (fractions within each category)", fontsize=14)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    panel_path = OUT_DIR / "heatmaps_by_category_2x2.png"
    plt.savefig(panel_path, dpi=220)
    plt.close(fig)
    print(f"Saved: {panel_path}")

# --------------- Figure 2: Scatter of probabilities --------------
fig2 = plt.figure(figsize=(5.8, 5.2))
ax2 = fig2.add_subplot(111)

# Color by agreement case
colors = {
    "both_correct":   "#2ca02c",  # green
    "both_wrong":     "#d62728",  # red
    "qwen_only":      "#1f77b4",  # blue
    "internvl_only":  "#ff7f0e"   # orange
}
for case, col in colors.items():
    mask = (agreement_case == case)
    ax2.scatter(p_qwen[mask], p_internvl[mask], s=14, alpha=0.55, label=case, c=col)

ax2.plot([0,1], [0,1], linestyle="--", color="gray", linewidth=1)  # diagonal
ax2.set_xlim(0,1); ax2.set_ylim(0,1)
ax2.set_xlabel("Qwen  P(correct)")
ax2.set_ylabel("InternVL  P(correct)")
ax2.set_title("Cross-model logistic agreement")
ax2.legend(markerscale=1.3, fontsize=9, frameon=False)
plt.tight_layout()
scatter_path = OUT_DIR / "scatter_prob_agreement.png"
plt.savefig(scatter_path, dpi=200)
plt.close(fig2)
print(f"Saved: {scatter_path}")

# ------ Figure 3: Correlation summary (bars for 3 metrics) -------
fig3 = plt.figure(figsize=(5.2, 3.6))
ax3 = fig3.add_subplot(111)
metrics = ["Cohen κ", "corr(P)", "corr(β)"]
vals    = [kappa, prob_corr, coef_corr]
x = np.arange(len(metrics))
ax3.bar(x, vals, width=0.6)
for i, vval in enumerate(vals):
    ax3.text(i, vval + 0.02, f"{vval:.2f}", ha="center", fontsize=10)
ax3.set_xticks(x); ax3.set_xticklabels(metrics)
ax3.set_ylim(min(0, min(vals)-0.1), 1.05)
ax3.set_ylabel("Value")
ax3.set_title("Agreement metrics summary")
plt.tight_layout()
corrsum_path = OUT_DIR / "correlation_summary.png"
plt.savefig(corrsum_path, dpi=200)
plt.close(fig3)
print(f"Saved: {corrsum_path}")

print("\n✅ Done. Artifacts written to:", OUT_DIR)
