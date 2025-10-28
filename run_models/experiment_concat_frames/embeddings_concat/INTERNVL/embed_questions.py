#!/usr/bin/env python3
import json, csv, gzip
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

IN_JSONL = Path("/home/ievab2/run_models/experiment_concat_frames/INTERNVL/experiment_og_concat_8/scores/internvl_per_question.jsonl")
OUT_DIR  = Path("/home/ievab2/run_models/experiment_concat_frames/embeddings_concat/INTERNVL/question_embeddings")
OUT_META = OUT_DIR / "questions_meta.csv"        # row_idx, question, correct(0/1), raw_score
OUT_EMB  = OUT_DIR / "embeddings_mpnet.csv.gz"   # 768 columns: e0..e767

rows = []
with IN_JSONL.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        d = json.loads(line)
        q = d.get("question", "")
        s = float(d.get("score", 0.0))
        y = 1 if s == 1.0 else 0   # strict: only 1.0 counts as correct
        rows.append({"row_idx": i, "question": q, "correct": y, "raw_score": s})

df = pd.DataFrame(rows)
print(f"Loaded {len(df)} rows from {IN_JSONL}")

# Load embedder (768-D)
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

emb = model.encode(
    df["question"].fillna("").tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,  # unit vectors
)
emb = np.asarray(emb, dtype=np.float32)
assert emb.shape[0] == len(df) and emb.shape[1] == 768

# Write aligned outputs
df.to_csv(OUT_META, index=False)
colnames = [f"e{i}" for i in range(emb.shape[1])]
with gzip.open(OUT_EMB, "wt", newline="") as gz:
    w = csv.writer(gz)
    w.writerow(colnames)
    w.writerows(emb.tolist())

print(f"Wrote:\n  {OUT_META}\n  {OUT_EMB}\nDone.")
