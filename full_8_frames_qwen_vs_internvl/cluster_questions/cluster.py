#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster questions (shared embeddings) + per-cluster VLM performance.

Outputs in OUT_DIR:
  - cluster_stats.csv
  - cluster_pies_all.png   <-- one image with all pie charts (K pies)
  - cluster_accuracy_bars.png
  - cluster_tsne.png       (optional, for preview)
"""

import os, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
BASE = Path("/home/ievab2/run_models/full_8_frames_qwen_vs_internvl")
EMB_PATH   = BASE / "QWEN/embeddings/only_prompt/embeddings_eva02_l14.csv.gz"
META_QWEN  = BASE / "QWEN/embeddings/only_prompt/prompts_meta.csv"
META_INTVL = BASE / "INTERNVL/embeddings/only_prompt/prompts_meta.csv"
JSONL_PATH = "/home/ievab2/run_models/questions/clevrer_filtered_500.jsonl"
OUT_DIR    = BASE / "cluster_questions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# -------------------------------------------------------------------
# 1) Load embeddings (shared EVA02 question embeddings)
# -------------------------------------------------------------------
print("Loading embeddings...")
X = pd.read_csv(EMB_PATH, compression="gzip").to_numpy()
print(f"Embeddings: {X.shape}")

# -------------------------------------------------------------------
# 2) Load metadata for both models
# -------------------------------------------------------------------
def load_meta(path, label_col="correct"):
    df = pd.read_csv(path)
    cols = ["row_idx", label_col] + (["prompt"] if "prompt" in df.columns else [])
    df = df[cols].copy()
    df["y"] = (df[label_col] >= 1.0).astype(int)
    keep = ~df["y"].isna()
    df = df.loc[keep]
    return df

mq = load_meta(META_QWEN)     # has columns: row_idx, correct, (prompt?), y
mi = load_meta(META_INTVL)

# Inner join so both models share the same questions
merged = pd.merge(
    mq[["row_idx", "y"] + (["prompt"] if "prompt" in mq.columns else [])],
    mi[["row_idx", "y"]],
    on="row_idx",
    suffixes=("_qwen", "_internvl")
)
print(f"Joined: {len(merged)} questions")

# -------------------------------------------------------------------
# 3) Map CLEVRER categories via strict 0-based row_idx -> JSONL line
# -------------------------------------------------------------------
with open(JSONL_PATH, "r") as f:
    cats_all = [json.loads(line)["category"] for line in f]
merged["category"] = [cats_all[idx] for idx in merged["row_idx"]]

# -------------------------------------------------------------------
# 4) Align embeddings with merged row_idx order
#     (Map row_idx -> row position in embeddings using QWEN meta order)
# -------------------------------------------------------------------
mq_full = pd.read_csv(META_QWEN)
order_map = {ri: i for i, ri in enumerate(mq_full["row_idx"].tolist())}
pos = [order_map[ri] for ri in merged["row_idx"]]
X_sub = X[pos]

# -------------------------------------------------------------------
# 5) PCA + KMeans clustering
# -------------------------------------------------------------------
print("Running PCA + KMeans...")
pca = PCA(n_components=20, random_state=42)
X_pca = pca.fit_transform(X_sub)

K = 8  # adjust as needed
kmeans = KMeans(n_clusters=K, n_init=20, random_state=42)
clusters = kmeans.fit_predict(X_pca)
merged["cluster"] = clusters

# -------------------------------------------------------------------
# 6) Per-cluster stats
# -------------------------------------------------------------------
def cluster_summary(df):
    out = []
    cats = ["descriptive", "explanatory", "predictive", "counterfactual"]
    for k in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == k]
        stats = {
            "cluster": int(k),
            "n": int(len(sub)),
            "acc_qwen": float(sub["y_qwen"].mean()),
            "acc_internvl": float(sub["y_internvl"].mean()),
            "delta": float(sub["y_qwen"].mean() - sub["y_internvl"].mean()),
        }
        for c in cats:
            stats[c] = float((sub["category"] == c).mean())
        out.append(stats)
    return pd.DataFrame(out).sort_values("cluster").reset_index(drop=True)

cluster_stats = cluster_summary(merged)
cluster_stats.to_csv(OUT_DIR / "cluster_stats.csv", index=False)
print(cluster_stats)

# -------------------------------------------------------------------
# 7) One image with all pie charts (category composition per cluster)
#    Colors:
#      blue -> descriptive
#      orange -> explanatory
#      green -> predictive
#      red -> counterfactual
# -------------------------------------------------------------------
cat_order = ["descriptive", "explanatory", "predictive", "counterfactual"]
colors = {
    "descriptive":    "#1f77b4",  # blue
    "explanatory":    "#ff7f0e",  # orange
    "predictive":     "#2ca02c",  # green
    "counterfactual": "#d62728",  # red
}
color_list = [colors[c] for c in cat_order]

def autopct_hide_small(pct):
    return f"{pct:.1f}%" if pct >= 0.5 else ""

K = len(cluster_stats)
cols = min(4, K)       # up to 4 columns
rows = math.ceil(K / cols)

fig_w = 4 * cols
fig_h = 4 * rows + 0.7
fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))

# Normalize axes to 2D array for indexing
if rows == 1 and cols == 1:
    axes = np.array([[axes]])
elif rows == 1:
    axes = np.array([axes])
elif cols == 1:
    axes = np.array([[ax] for ax in axes])

for i, row in cluster_stats.iterrows():
    r, c = divmod(i, cols)
    ax = axes[r, c]
    vals = [row[col] for col in cat_order]

    # Avoid zero-sum (shouldn't happen, but guard anyway)
    if sum(vals) == 0:
        vals = [0.0001, 0.0001, 0.0001, 0.0001]

    ax.pie(
        vals,
        colors=color_list,
        startangle=90,
        autopct=autopct_hide_small,
        counterclock=False
    )
    ax.axis("equal") 
    ax.set_title(
        f"Cluster {int(row['cluster'])} — n={int(row['n'])}\n"
        f"Qwen={row['acc_qwen']:.2f} | InternVL={row['acc_internvl']:.2f}",
        fontsize=10
    )

# Hide any extra subplots if K not multiple of cols
for j in range(K, rows * cols):
    r, c = divmod(j, cols)
    axes[r, c].axis("off")

# Shared legend (bottom center)
handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=colors[c], markersize=10)
           for c in cat_order]
labels = ["Descriptive", "Explanatory", "Predictive", "Counterfactual"]
fig.legend(handles, labels, loc="lower center", ncol=4, borderaxespad=0.8)

plt.tight_layout(rect=[0, 0.07, 1, 1])
out_pies = OUT_DIR / "cluster_pies_all.png"
plt.savefig(out_pies, dpi=200)
plt.close()
print(f"✅ Saved all pies: {out_pies}")

# -------------------------------------------------------------------
# 8) Bar chart: per-cluster accuracies for both models
# -------------------------------------------------------------------
plt.figure(figsize=(10, 5))
width = 0.38
x = np.arange(len(cluster_stats))
plt.bar(x - width/2, cluster_stats["acc_qwen"], width, label="Qwen")
plt.bar(x + width/2, cluster_stats["acc_internvl"], width, label="InternVL")
for i, (a1, a2) in enumerate(zip(cluster_stats["acc_qwen"], cluster_stats["acc_internvl"])):
    plt.text(i - width/2, a1 + 0.01, f"{a1:.2f}", ha="center", fontsize=8)
    plt.text(i + width/2, a2 + 0.01, f"{a2:.2f}", ha="center", fontsize=8)
plt.xticks(x, [f"C{k}" for k in cluster_stats["cluster"]])
plt.ylim(0, 1.05)
plt.legend()
plt.title("Per-cluster accuracy by model")
plt.tight_layout()
bars_path = OUT_DIR / "cluster_accuracy_bars.png"
plt.savefig(bars_path, dpi=200)
plt.close()
print(f"✅ Saved bars: {bars_path}")

# -------------------------------------------------------------------
# 9) Clustering ONLY non-descriptive questions
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# 9) Clustering ONLY non-descriptive questions
# -------------------------------------------------------------------
import math, json

# Keep only non-descriptive
mask_nd = merged["category"] != "descriptive"
merged_nd = merged.loc[mask_nd].reset_index(drop=True)
X_nd = X_sub[mask_nd.values]
print(f"[ND] Remaining after removing descriptives: {len(merged_nd)} questions")

# PCA + KMeans on the non-descriptive subset
pca_nd = PCA(n_components=20, random_state=42)
X_pca_nd = pca_nd.fit_transform(X_nd)

K_ND = 6  # adjust if you want a different number of clusters for the ND subset
kmeans_nd = KMeans(n_clusters=K_ND, n_init=20, random_state=42)
merged_nd["cluster_nd"] = kmeans_nd.fit_predict(X_pca_nd)

# Per-cluster stats (ND subset)
def cluster_summary_nd(df):
    out = []
    cats = ["descriptive", "explanatory", "predictive", "counterfactual"]
    for k in sorted(df["cluster_nd"].unique()):
        sub = df[df["cluster_nd"] == k]
        stats = {
            "cluster_nd": int(k),
            "n": int(len(sub)),
            "acc_qwen": float(sub["y_qwen"].mean()),
            "acc_internvl": float(sub["y_internvl"].mean()),
            "delta": float(sub["y_qwen"].mean() - sub["y_internvl"].mean()),
        }
        for c in cats:
            stats[c] = float((sub["category"] == c).mean())
        out.append(stats)
    return pd.DataFrame(out).sort_values("cluster_nd").reset_index(drop=True)

cluster_stats_nd = cluster_summary_nd(merged_nd)
cluster_stats_nd.to_csv(OUT_DIR / "nd_cluster_stats.csv", index=False)
print(cluster_stats_nd)

# ---- ONE PNG with all pies for ND clusters ----
# (Reuse your color mapping; descriptive will be 0% here but kept for consistency.)
cat_order = ["descriptive", "explanatory", "predictive", "counterfactual"]
colors = {
    "descriptive":    "#1f77b4",  # blue
    "explanatory":    "#ff7f0e",  # orange
    "predictive":     "#2ca02c",  # green
    "counterfactual": "#d62728",  # red
}
color_list = [colors[c] for c in cat_order]

def autopct_hide_small(pct):
    return f"{pct:.1f}%" if pct >= 0.5 else ""

K = len(cluster_stats_nd)
cols = min(4, K)
rows = math.ceil(K / cols)
fig_w = 4 * cols
fig_h = 4 * rows + 0.7

fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
if rows == 1 and cols == 1:
    axes = np.array([[axes]])
elif rows == 1:
    axes = np.array([axes])
elif cols == 1:
    axes = np.array([[ax] for ax in axes])

for i, row in cluster_stats_nd.iterrows():
    r, c = divmod(i, cols)
    ax = axes[r, c]
    vals = [row[col] for col in cat_order]  # 'descriptive' will be ~0 here
    if sum(vals) == 0:
        vals = [0.0001, 0.0001, 0.0001, 0.0001]

    ax.pie(
        vals,
        colors=color_list,
        startangle=90,
        autopct=autopct_hide_small,
        counterclock=False
    )
    ax.axis("equal")
    ax.set_title(
        f"ND Cluster {int(row['cluster_nd'])} — n={int(row['n'])}\n"
        f"Qwen={row['acc_qwen']:.2f} | InternVL={row['acc_internvl']:.2f}",
        fontsize=10
    )

# Hide empty axes if K not multiple of cols
for j in range(K, rows * cols):
    r, c = divmod(j, cols)
    axes[r, c].axis("off")

# Shared legend
handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=colors[c], markersize=10)
           for c in cat_order]
labels = ["Descriptive", "Explanatory", "Predictive", "Counterfactual"]
fig.legend(handles, labels, loc="lower center", ncol=4, borderaxespad=0.8)

plt.tight_layout(rect=[0, 0.07, 1, 1])
nd_pies_path = OUT_DIR / "nd_cluster_pies_all.png"
plt.savefig(nd_pies_path, dpi=200)
plt.close()
print(f"✅ Saved ND pies (single image): {nd_pies_path}")

# ---- JSONL with the actual questions per ND cluster ----
# One line per question: cluster id + metadata for linguistic inspection
jsonl_path = OUT_DIR / "nd_clusters.jsonl"
with open(jsonl_path, "w", encoding="utf-8") as jf:
    for _, row in merged_nd.sort_values("cluster_nd").iterrows():
        rec = {
            "cluster_nd": int(row["cluster_nd"]),
            "row_idx": int(row["row_idx"]),
            "category": str(row["category"]),
            "qwen_correct": int(row["y_qwen"]),
            "internvl_correct": int(row["y_internvl"]),
            "prompt": (row["prompt"] if "prompt" in merged_nd.columns else None),
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"✅ Saved ND cluster assignments with prompts: {jsonl_path}")


print(f"✅ Done. Results saved in {OUT_DIR}")
