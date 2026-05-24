#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


ROOT = Path("/home/ievab2/run_models/FULL_PHYSION_FINETUNING")

# Embeddings root: /home/.../FULL_PHYSION_FINETUNING/embeddings/round{round}/
def load_embeddings(round_idx: int):
    emb_root = ROOT / "embeddings" / f"round{round_idx}"
    meta_path = emb_root / "meta.csv" #this is for the base model, but we do NOT use the "model_output_norm,correct" from there
    emb_path = emb_root / "embeddings.csv.gz" #this is for the base model, but the embeddings are just for frames and prompts - so they stay fine for everyhting

    print(f"[EMB] Using round={round_idx}")
    print(f"[EMB] meta: {meta_path}")
    print(f"[EMB] emb : {emb_path}")

    if not meta_path.exists():
        raise FileNotFoundError(f"meta.csv not found: {meta_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"embeddings.csv.gz not found: {emb_path}")

    df_meta = pd.read_csv(meta_path)
    df_emb = pd.read_csv(emb_path, compression="gzip")

    if len(df_meta) != len(df_emb):
        raise RuntimeError(
            f"meta rows ({len(df_meta)}) != emb rows ({len(df_emb)})"
        )

    X = df_emb.to_numpy(dtype=np.float32)

    # Build mapping (category, qid) -> row index
    key_to_idx = {}
    for idx, row in df_meta.iterrows():
        cat = row.get("category")
        qid = row.get("qid")
        if pd.isna(cat) or pd.isna(qid):
            continue
        key = (str(cat), int(qid))
        key_to_idx[key] = idx

    print(f"[EMB] Loaded {X.shape[0]} embeddings, dim={X.shape[1]}")
    print(f"[EMB] Built key->idx for {len(key_to_idx)} (category, qid) pairs")

    return df_meta, X, key_to_idx


def safe_json_lines(path: Path):
    """Yield parsed JSON objects from a JSONL file, skipping blank / invalid lines."""
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode error in {path} line {line_idx}: {e}")
                continue


def process_epoch_split(
    epoch: int,
    split: str,
    X_all: np.ndarray,
    key_to_idx: dict,
    round_idx: int,
):
    """
    For a given (epoch, split):
    - read SPLIT_test_out.jsonl
    - train logistic regression on embeddings for that test set
    - write SPLIT_hardness.jsonl with p_correct & hardness
    """
    test_root = ROOT / "testing" / "round0" / f"epochs{epoch}"
    in_path = test_root / f"{split}_test_out.jsonl"

    out_root = ROOT / "testing" / "round0" / "hardness check" / f"epochs{epoch}"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{split}_hardness.jsonl"

    print(f"\n[EP{epoch}][{split}] IN  = {in_path}")
    print(f"[EP{epoch}][{split}] OUT = {out_path}")

    if not in_path.exists():
        print(f"[WARN] Missing input file, skipping: {in_path}")
        return

    # Collect lines and build training data for logistic regression
    records = []             # original JSON rows
    train_indices = []       # indices into X_all
    y = []                   # correctness labels (0/1)
    line_to_train_idx = {}   # map line index -> position in train_indices

    total_lines = 0
    missing_embed = 0

    for line_idx, row in enumerate(safe_json_lines(in_path)):
        records.append(row)
        total_lines += 1
        local_idx = len(records) - 1

        cat = row.get("category")
        qid = row.get("qid")

        if cat is None or qid is None:
            print(f"[WARN] Missing category/qid in {in_path} line {line_idx}, skipping for training.")
            continue

        try:
            key = (str(cat), int(qid))
        except Exception:
            print(f"[WARN] Unparseable qid in {in_path} line {line_idx}, skipping for training.")
            continue

        emb_idx = key_to_idx.get(key)
        if emb_idx is None:
            missing_embed += 1
            continue

        # correctness: 1 if model_output_norm == answer
        ans = row.get("answer")
        pred = row.get("model_output_norm")
        correct = int(pred == ans)

        train_indices.append(emb_idx)
        y.append(correct)
        line_to_train_idx[local_idx] = len(train_indices) - 1

    print(f"[EP{epoch}][{split}] total lines  = {total_lines}")
    print(f"[EP{epoch}][{split}] train samples= {len(train_indices)} (with embeddings)")
    print(f"[EP{epoch}][{split}] missing emb  = {missing_embed}")

    if len(train_indices) == 0:
        print(f"[EP{epoch}][{split}] No training data with embeddings; skipping.")
        return

    # Build X_sub and y for this split
    X_sub = X_all[np.array(train_indices, dtype=int)]
    y_arr = np.array(y, dtype=int)

    print(f"[EP{epoch}][{split}] Training LogisticRegression on {len(y_arr)} samples")
    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight=None,
        solver="lbfgs",
    )
    clf.fit(X_sub, y_arr)

    p_correct = clf.predict_proba(X_sub)[:, 1]
    hardness = 1.0 - p_correct

    # Write output JSONL: same rows plus p_correct + hardness (if available)
    with out_path.open("w", encoding="utf-8") as fout:
        for i, row in enumerate(records):
            out_row = dict(row)
            t_idx = line_to_train_idx.get(i)

            if t_idx is not None:
                out_row["p_correct"] = float(p_correct[t_idx])
                out_row["hardness"] = float(hardness[t_idx])
            else:
                # no embedding / not used in training
                out_row["p_correct"] = None
                out_row["hardness"] = None

            fout.write(json.dumps(out_row) + "\n")

    print(f"[EP{epoch}][{split}] Wrote hardness-annotated file: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate hardness for finetuned models via logistic regression.")
    ap.add_argument(
        "--round",
        type=int,
        default=0,
        help="Embedding round index (embeddings/round{round}). Default: 0",
    )
    ap.add_argument(
        "--epochs",
        type=str,
        default="1,3,5",
        help="Comma-separated list of epochs to process, e.g. '1,3,5'.",
    )
    ap.add_argument(
        "--splits",
        type=str,
        default="SPLIT1,SPLIT2,SPLIT3",
        help="Comma-separated list of split names, e.g. 'SPLIT1,SPLIT2,SPLIT3'.",
    )
    args = ap.parse_args()

    print(f"[INFO] round={args.round}")
    epoch_list = [int(e.strip()) for e in args.epochs.split(",") if e.strip()]
    split_list = [s.strip() for s in args.splits.split(",") if s.strip()]

    print(f"[INFO] epochs={epoch_list}")
    print(f"[INFO] splits={split_list}")

    # Load global embeddings and mapping
    df_meta, X_all, key_to_idx = load_embeddings(args.round)

    # Process each epoch/split
    for epoch in epoch_list:
        for split in split_list:
            process_epoch_split(
                epoch=epoch,
                split=split,
                X_all=X_all,
                key_to_idx=key_to_idx,
                round_idx=args.round,
            )

    print("\n[DONE] Hardness evaluation completed for all requested epochs/splits.")


if __name__ == "__main__":
    main()
