# #!/usr/bin/env python3
# import argparse
# import json
# from pathlib import Path

# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression


# ROOT = Path("/home/ievab2/run_models/FULL_PHYSION_FINETUNING")
# EMB_ROOT_TEMPLATE = ROOT / "embeddings" / "round{round}"
# CURR_ROOT_TEMPLATE = ROOT / "curriculum" / "round{round}"
# SPLITS_ROOT = Path("/shared/rsaas/ievab2/Physion_full_readout_training/SPLITS")


# def load_embeddings_and_meta(round_idx: int):
#     emb_root = EMB_ROOT_TEMPLATE.with_name(EMB_ROOT_TEMPLATE.name.format(round=round_idx))
#     meta_path = emb_root / "meta.csv"
#     emb_path = emb_root / "embeddings.csv.gz"

#     print(f"[LOAD] meta: {meta_path}")
#     print(f"[LOAD] emb : {emb_path}")

#     if not meta_path.exists():
#         raise FileNotFoundError(f"meta.csv not found: {meta_path}")
#     if not emb_path.exists():
#         raise FileNotFoundError(f"embeddings.csv.gz not found: {emb_path}")

#     df_meta = pd.read_csv(meta_path)
#     df_emb = pd.read_csv(emb_path, compression="gzip")

#     if len(df_meta) != len(df_emb):
#         raise RuntimeError(
#             f"meta rows ({len(df_meta)}) != emb rows ({len(df_emb)})"
#         )

#     # X: embeddings, y: correctness
#     X = df_emb.to_numpy(dtype=np.float32)
#     y = df_meta["correct"].astype(int).to_numpy()

#     print(f"[LOAD] Loaded {X.shape[0]} samples, dim={X.shape[1]}")

#     return df_meta, X, y


# def train_logistic(df_meta: pd.DataFrame, X: np.ndarray, y: np.ndarray):
#     print(f"[CLF] Training LogisticRegression on {len(y)} samples")
#     clf = LogisticRegression(
#         max_iter=1000,
#         n_jobs=-1,
#         class_weight=None,
#         solver="lbfgs",
#     )
#     clf.fit(X, y)

#     # p_correct is prob of class 1 ("correct")
#     p_correct = clf.predict_proba(X)[:, 1]
#     hardness = 1.0 - p_correct

#     df_meta = df_meta.copy()
#     df_meta["p_correct"] = p_correct
#     df_meta["hardness"] = hardness

#     # build lookup table: (category, qid) -> stats
#     table = {}
#     for row in df_meta.itertuples(index=False):
#         cat = getattr(row, "category", None)
#         qid = getattr(row, "qid", None)
#         if cat is None or pd.isna(cat) or qid is None or pd.isna(qid):
#             continue
#         key = (str(cat), int(qid))
#         table[key] = {
#             "p_correct": float(getattr(row, "p_correct")),
#             "hardness": float(getattr(row, "hardness")),
#         }

#     print(f"[CLF] Built hardness table for {len(table)} (category, qid) pairs")
#     return table


# def load_done_keys(out_path: Path):
#     """For resume: read existing curriculum JSONL and return set of (category, qid)."""
#     done = set()
#     if not out_path.exists():
#         return done
#     with out_path.open("r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 rec = json.loads(line)
#             except Exception:
#                 continue
#             cat = rec.get("category")
#             qid = rec.get("qid")
#             if cat is None or qid is None:
#                 continue
#             try:
#                 key = (str(cat), int(qid))
#             except Exception:
#                 continue
#             done.add(key)
#     return done


# def build_curriculum_for_split(
#     split_name: str,
#     subset: str,
#     table: dict,
#     out_root: Path,
#     threshold: float,
#     epsilon: float,
#     resume: bool,
# ):
#     in_path = SPLITS_ROOT / split_name / f"{subset}.jsonl"
#     out_path = out_root / f"curriculum_{split_name}_{subset}.jsonl"

#     print(f"\n[CURR] {split_name} {subset}:")
#     print(f"       IN  = {in_path}")
#     print(f"       OUT = {out_path}")

#     if not in_path.exists():
#         print(f"[WARN] Input split file missing, skipping: {in_path}")
#         return

#     out_root.mkdir(parents=True, exist_ok=True)

#     # resume logic
#     if resume and out_path.exists():
#         done_keys = load_done_keys(out_path)
#         fout = out_path.open("a", encoding="utf-8")
#         print(f"[RESUME] Found {len(done_keys)} already-written rows for {split_name}/{subset}")
#     else:
#         done_keys = set()
#         fout = out_path.open("w", encoding="utf-8")
#         print(f"[CURR] Starting fresh for {split_name}/{subset}")

#     total = 0
#     matched = 0
#     missing = 0

#     with in_path.open("r", encoding="utf-8") as fin:
#         for line_idx, line in enumerate(fin):
#             line = line.strip()
#             if not line:
#                 continue
#             total += 1
#             try:
#                 row = json.loads(line)
#             except Exception as e:
#                 print(f"[WARN] JSON error in {in_path} line {line_idx}: {e}")
#                 continue

#             cat = row.get("category")
#             qid = row.get("qid")
#             if cat is None or qid is None:
#                 print(f"[WARN] Missing category/qid in {in_path} line {line_idx}, skipping.")
#                 continue

#             try:
#                 key = (str(cat), int(qid))
#             except Exception:
#                 print(f"[WARN] Unparseable qid in {in_path} line {line_idx}, skipping.")
#                 continue

#             if key in done_keys:
#                 continue

#             stats = table.get(key)
#             if stats is None:
#                 missing += 1
#                 continue

#             p_correct = stats["p_correct"]
#             hardness = stats["hardness"]
#             weight = 1.0 + epsilon if hardness >= threshold else 1.0

#             out_row = dict(row)
#             out_row["p_correct"] = float(p_correct)
#             out_row["hardness"] = float(hardness)
#             out_row["weight"] = float(weight)

#             fout.write(json.dumps(out_row) + "\n")
#             matched += 1
#             if matched % 500 == 0:
#                 print(f"[CURR] {split_name}/{subset}: matched={matched}, total_seen={total}, missing={missing}")
#                 fout.flush()

#     fout.close()
#     print(f"[CURR] Done {split_name}/{subset}: total={total}, matched={matched}, missing={missing}")


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--round", type=int, default=0, help="Curriculum round index (for paths).")
#     ap.add_argument(
#         "--splits",
#         type=str,
#         default="SPLIT1,SPLIT2,SPLIT3",
#         help="Comma-separated list of split names (e.g. SPLIT1,SPLIT2,SPLIT3).",
#     )
#     ap.add_argument(
#         "--subsets",
#         type=str,
#         default="train,test",
#         help="Comma-separated list of subsets per split (e.g. train,test).",
#     )
#     ap.add_argument(
#         "--threshold",
#         type=float,
#         default=0.5,
#         help="Hardness threshold: hardness>=threshold → hard (gets weight 1+epsilon).",
#     )
#     ap.add_argument(
#         "--epsilon",
#         type=float,
#         default=0.01,
#         help="Extra weight added to hard examples (easy=1.0, hard=1.0+epsilon).",
#     )
#     ap.add_argument(
#         "--resume",
#         action="store_true",
#         help="If set, resume from existing curriculum_* files instead of overwriting.",
#     )
#     args = ap.parse_args()

#     print(f"[INFO] round={args.round}, threshold={args.threshold}, epsilon={args.epsilon}")
#     split_list = [s.strip() for s in args.splits.split(",") if s.strip()]
#     subset_list = [s.strip() for s in args.subsets.split(",") if s.strip()]
#     print(f"[INFO] splits={split_list}, subsets={subset_list}")

#     # 1) load embeddings & meta
#     df_meta, X, y = load_embeddings_and_meta(args.round)

#     # 2) train classifier and build hardness table
#     hardness_table = train_logistic(df_meta, X, y)

#     # 3) build curriculum JSONLs for each split/subset
#     out_root = CURR_ROOT_TEMPLATE.with_name(CURR_ROOT_TEMPLATE.name.format(round=args.round))
#     out_root.mkdir(parents=True, exist_ok=True)
#     print(f"[INFO] Curriculum output root: {out_root}")

#     for split_name in split_list:
#         for subset in subset_list:
#             build_curriculum_for_split(
#                 split_name=split_name,
#                 subset=subset,
#                 table=hardness_table,
#                 out_root=out_root,
#                 threshold=args.threshold,
#                 epsilon=args.epsilon,
#                 resume=args.resume,
#             )

#     print(f"\n[DONE] Curriculum JSONLs saved under: {out_root}")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

SPLITS_ROOT = Path("/shared/rsaas/ievab2/Physion_full_readout_training/SPLITS")


def load_embeddings_and_meta(meta_path: Path, emb_path: Path):
    print(f"[LOAD] meta: {meta_path}")
    print(f"[LOAD] emb : {emb_path}")

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
    y = df_meta["correct"].astype(int).to_numpy()

    print(f"[LOAD] Loaded {X.shape[0]} samples, dim={X.shape[1]}")
    return df_meta, X, y


def train_logistic(df_meta: pd.DataFrame, X: np.ndarray, y: np.ndarray):
    print(f"[CLF] Training LogisticRegression on {len(y)} samples")
    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight=None,
        solver="lbfgs",
    )
    clf.fit(X, y)

    p_correct = clf.predict_proba(X)[:, 1]
    hardness = 1.0 - p_correct

    df_meta = df_meta.copy()
    df_meta["p_correct"] = p_correct
    df_meta["hardness"] = hardness

    table = {}
    for row in df_meta.itertuples(index=False):
        cat = getattr(row, "category", None)
        qid = getattr(row, "qid", None)
        if cat is None or pd.isna(cat) or qid is None or pd.isna(qid):
            continue
        table[(str(cat), int(qid))] = {
            "p_correct": float(row.p_correct),
            "hardness": float(row.hardness),
        }

    print(f"[CLF] Built hardness table for {len(table)} (category, qid) pairs")
    return table


def load_done_keys(out_path: Path):
    done = set()
    if not out_path.exists():
        return done
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add((str(rec["category"]), int(rec["qid"])))
            except Exception:
                pass
    return done


def build_curriculum_for_split(
    split_name: str,
    subset: str,
    table: dict,
    out_root: Path,
    threshold: float,
    epsilon: float,
    resume: bool,
):
    in_path = SPLITS_ROOT / split_name / f"{subset}.jsonl"
    out_path = out_root / f"curriculum_{split_name}_{subset}.jsonl"

    print(f"\n[CURR] {split_name} {subset}")
    print(f"  IN : {in_path}")
    print(f"  OUT: {out_path}")

    if not in_path.exists():
        print("[WARN] Missing split file, skipping.")
        return

    out_root.mkdir(parents=True, exist_ok=True)

    if resume and out_path.exists():
        done_keys = load_done_keys(out_path)
        fout = out_path.open("a", encoding="utf-8")
        print(f"[RESUME] {len(done_keys)} rows already written")
    else:
        done_keys = set()
        fout = out_path.open("w", encoding="utf-8")

    matched = missing = total = 0

    with in_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)

            key = (str(row.get("category")), int(row.get("qid")))
            if key in done_keys:
                continue

            stats = table.get(key)
            if stats is None:
                missing += 1
                continue

            hardness = stats["hardness"]
            weight = 1.0 + epsilon if hardness >= threshold else 1.0

            out_row = dict(row)
            out_row.update(stats)
            out_row["weight"] = float(weight)

            fout.write(json.dumps(out_row) + "\n")
            matched += 1

    fout.close()
    print(f"[DONE] total={total}, matched={matched}, missing={missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", required=True)
    ap.add_argument("--embeddings_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--splits", default="SPLIT1,SPLIT2,SPLIT3")
    ap.add_argument("--subsets", default="train,test")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--epsilon", type=float, default=0.01)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    split_list = [s.strip() for s in args.splits.split(",")]
    subset_list = [s.strip() for s in args.subsets.split(",")]

    df_meta, X, y = load_embeddings_and_meta(
        Path(args.meta_csv), Path(args.embeddings_csv)
    )
    table = train_logistic(df_meta, X, y)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in split_list:
        for subset in subset_list:
            build_curriculum_for_split(
                split, subset, table,
                out_root, args.threshold, args.epsilon, args.resume
            )

    print(f"\n[DONE] Curriculum saved to {out_root}")


if __name__ == "__main__":
    main()
