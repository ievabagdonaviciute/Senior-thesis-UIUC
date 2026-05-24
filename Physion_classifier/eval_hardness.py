#!/usr/bin/env python3
"""
Compute per-sample hardness for Physion (Dominoes/Contain/Drop) and QWEN/INTERNVL.

Source = embeddings (only_image | only_prompt | image_and_prompt):
 - Load model's JSONL outputs to get ground_truth vs model_output_norm -> y (correct?).
 - Load matching embeddings + meta (from your embed.py outputs).
 - Fit out-of-fold logistic regression to predict correctness from embeddings.
 - Produce per-sample p_correct, hardness=1-p_correct, and a weight = 1 + alpha*hardness.
"""

import argparse, json, csv, os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ------------------------- CONSTANT PATH ROOTS -------------------------
BASE = Path("/home/ievab2/run_models/experiment_quick_physion")
FINETUNED_BASE = Path("/home/ievab2/run_models/Physion_finetuning")

def jsonl_path(model: str, category: str, subset: str, version: str) -> Path:
    # e.g. /.../INTERNVL/Contain/contain_pred_out.jsonl
    if version == "base":
        return BASE / model / category / f"{category.lower()}_{subset}_out.jsonl"
    elif version == "both_5epochs_4frames":
        return FINETUNED_BASE / model / version / category / f"{category.lower()}_{subset}_out.jsonl"
    else:
        raise ValueError(f"Unknown version: {version}")

def embed_dirs(model: str, category: str, subset: str, feat: str, version: str) -> Path:
    # e.g. /.../INTERNVL/Contain/embeddings/pred/only_image
    if version == "base":
        return BASE / model / category / "embeddings" / subset / feat
    elif version == "both_5epochs_4frames":
        return FINETUNED_BASE / model / version / category / "embeddings" / subset / feat


def meta_filename_for(feat: str) -> str:
    if feat == "only_prompt":       return "prompts_meta.csv"
    if feat == "only_image":        return "images_meta.csv"
    if feat == "image_and_prompt":  return "meta.csv"  # from embed_images_and_questions()
    raise ValueError("feat must be one of: only_prompt, only_image, image_and_prompt")

# overwrite to save under Physion_classifier/image_classifications
def curriculum_out_path(category: str, model: str, subset: str, feat: str, round_idx: int, alpha: float, version: str) -> Path:
    prefix = f"round{round_idx}"
    if version is not None:
        prefix += f"_{version}"

    out_root = Path("/home/ievab2/run_models/Physion_classifier/image_classifications")
    out_dir = out_root / prefix / model / category
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"hardness_{subset}_{feat}_alpha{alpha:g}.csv"

# ------------------------- LOADERS -------------------------
def load_jsonl_qids(jsonl_file: Path) -> pd.DataFrame:
    """
    Returns DataFrame with columns: row_idx, qid, ground_truth, model_output_norm
    row_idx is the 0-based line index in the JSONL.
    """
    rows = []
    with jsonl_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: 
                continue
            d = json.loads(line)
            qid = d.get("qid", i)
            gt  = str(d.get("ground_truth", "")).strip().lower()
            pred= str(d.get("model_output_norm", "")).strip().lower()
            rows.append({"row_idx": i, "qid": qid, "ground_truth": gt, "model_output_norm": pred})
    return pd.DataFrame(rows)

def load_embeddings(meta_p: Path, emb_p: Path) -> pd.DataFrame:
    if not meta_p.exists() or not emb_p.exists():
        raise FileNotFoundError(f"Missing embeddings or meta:\n  {meta_p}\n  {emb_p}")

    meta = pd.read_csv(meta_p)
    if "row_idx" not in meta.columns or "correct" not in meta.columns:
        raise ValueError(f"Meta file missing required columns: {meta_p}")

    X = pd.read_csv(emb_p, compression="gzip").astype("float32")

    if len(meta) != len(X):
        raise ValueError(f"Meta rows ({len(meta)}) != embeddings rows ({len(X)})")

    meta = meta.reset_index(drop=True)
    X    = X.reset_index(drop=True)

    df = meta.copy()
    df["y"] = meta["correct"].astype(int).clip(0, 1)

    # One-shot join (no fragmentation)
    df = df.join(X)

    feature_cols = list(X.columns)
    return df, feature_cols


# ------------------------- HARDNESS MODEL -------------------------
def oof_logistic_proba(X: np.ndarray, y: np.ndarray, n_splits: int = 5, C: float = 1.0, seed: int = 42) -> np.ndarray:
    """
    Compute out-of-fold predicted probabilities P(y=1 | X) via stratified K-fold.
    Returns array of length N with p_correct per sample.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = np.zeros(len(y), dtype=float)

    # Pipeline: Standardize + LogisticRegression (L2, liblinear for stability)
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(solver="liblinear", C=C, max_iter=200, class_weight=None)
    )

    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr = y[tr_idx]
        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_va)[:, 1]
        proba[va_idx] = p
    return proba

# ------------------------- MAIN -------------------------
def main():
    ap = argparse.ArgumentParser(description="Compute hardness (p_correct, 1-p) from embeddings.")
    ap.add_argument("-m", "--model", choices=["INTERNVL"], required=True, help="Which model’s outputs to learn hardness from.")
    ap.add_argument("-v", "--version", choices=["base", "both_5epochs_4frames"], required=True, help="Which model’s outputs to learn hardness from.")

    ap.add_argument("-c", "--category", choices=["Dominoes", "Contain", "Drop"], required=True)
    ap.add_argument("-s", "--subset", choices=["past", "pred"], required=True, help="Subset to use for training hardness.")
    ap.add_argument("-f", "--feat", choices=["only_image", "only_prompt", "image_and_prompt"], default="only_image",
                    help="Which embedding family to train on.")
    ap.add_argument("--round", type=int, default=0, help="Curriculum round index to tag the output.")
    ap.add_argument("--alpha", type=float, default=2.0, help="Weight slope: weight = 1 + alpha * hardness.")
    ap.add_argument("--c", type=float, default=1.0, help="LogReg inverse regularization strength (C).")
    ap.add_argument("--folds", type=int, default=5, help="Number of CV folds for OOF probabilities.")
    args = ap.parse_args()

    # Resolve input paths
    jpath   = jsonl_path(args.model, args.category, args.subset, args.version)
    edir    = embed_dirs(args.model, args.category, args.subset, args.feat, args.version)
    meta_p  = edir / meta_filename_for(args.feat)
    emb_p   = edir / "embeddings.csv.gz"

    if not jpath.exists():
        raise FileNotFoundError(f"Model output JSONL not found: {jpath}")
    if not meta_p.exists() or not emb_p.exists():
        raise FileNotFoundError(f"Embeddings for {args.model}/{args.category}/{args.subset}/{args.feat} not found.\n"
                                f"Expected:\n  {meta_p}\n  {emb_p}")

    print(f"[load] JSONL:     {jpath}")
    print(f"[load] EMBEDS:    {emb_p}")
    print(f"[load] META:      {meta_p}")

    # Load embeddings + labels (y = correct from meta)
    df, feature_cols = load_embeddings(meta_p, emb_p)
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=int)

    print(f"[fit] N={len(y)}, positives={y.sum()}, negatives={len(y)-y.sum()}, dim={X.shape[1]}")
    p_oof = oof_logistic_proba(X, y, n_splits=args.folds, C=args.c)
    hardness = 1.0 - p_oof
    weight = 1.0 + args.alpha * hardness

    # # Attach qid/gt/pred by joining on row_idx
    # Load JSONL (keep as columns)
    jdf = load_jsonl_qids(jpath)

    # Keep only what we need and avoid name clash
    jdf = jdf[["row_idx", "qid", "ground_truth", "model_output_norm"]].rename(
        columns={"ground_truth": "ground_truth_jsonl"}
    )

    # Merge
    df_join = df.merge(jdf, on="row_idx", how="left")


    out_df = pd.DataFrame({
        "row_idx": df_join["row_idx"],
        "qid": df_join["qid"],
        # choose which GT you want to keep; here we keep the JSONL one
        "ground_truth": df_join["ground_truth_jsonl"],
        "model_output_norm": df_join["model_output_norm"],
        "correct": df["y"].astype(int),
        "p_correct": p_oof,
        "hardness": hardness,
        "weight": weight
    })


    out_p = curriculum_out_path(args.category, args.model, args.subset, args.feat, args.round, args.alpha, args.version)
    out_df.to_csv(out_p, index=False)
    print(f"[save] Hardness table -> {out_p}")
    print(out_df.head().to_string(index=False))

if __name__ == "__main__":
    main()
