#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vocabulary–accuracy analysis:
Top-30 frequent words (excluding stopwords/boilerplate/option labels)
and per-word accuracy for Qwen vs InternVL.

Outputs:
  - vocab_accuracy_stats.csv
  - vocab_accuracy.png
"""

import json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# ---------------------------- Config ----------------------------
BASE        = Path("/home/ievab2/run_models/full_8_frames_qwen_vs_internvl")
JSONL_PATH  = "/home/ievab2/run_models/questions/clevrer_filtered_500.jsonl"
META_QWEN   = BASE / "QWEN/embeddings/only_prompt/prompts_meta.csv"
META_INTVL  = BASE / "INTERNVL/embeddings/only_prompt/prompts_meta.csv"
OUT_DIR     = BASE / "vocab_analysis"
TOP_K       = 30

# Choose how to score:
# True  -> strict accuracy (1 only for full credit == 1.0)
# False -> use partial credit (average score in [0,1])
BINARY_ACCURACY = True

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------- Load CLEVRER JSONL ----------------------
rows = [json.loads(line) for line in open(JSONL_PATH, "r", encoding="utf-8")]
dfj = pd.json_normalize(rows)

# Create row_idx from line order if missing
if "row_idx" not in dfj.columns:
    dfj["row_idx"] = np.arange(len(dfj))

# Build unified "prompt" field (question + options when needed)
def build_prompt(row: pd.Series) -> str:
    # 1) direct prompt
    p = row.get("prompt", None)
    if isinstance(p, str) and p.strip():
        return p

    # 2) question text under various keys
    q = None
    for k in ["question", "question_text", "question.prompt", "data.question"]:
        val = row.get(k, None)
        if isinstance(val, str) and val.strip():
            q = val.strip()
            break

    # 3) options/choices
    opts = None
    for k in ["options", "choices", "answers", "data.options"]:
        val = row.get(k, None)
        if isinstance(val, (list, tuple)) and len(val) > 0:
            opts = [str(x) for x in val]
            break

    if q is None and opts is None:
        return str(row)

    if opts is None:
        return q if q is not None else ""

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    opt_str = " ".join(f"{letters[i]}. {opt}" for i, opt in enumerate(opts))
    return f"{q if q is not None else ''} options: {opt_str}"

dfj["prompt"] = dfj.apply(build_prompt, axis=1)
dfq = dfj[["row_idx", "prompt"]].copy()
print(f"Loaded {len(dfq)} prompts")

# ----------------------- Load model metas ------------------------
def load_meta(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if BINARY_ACCURACY:
        y = (df["correct"].astype(float) >= 1.0).astype(float)
    else:
        y = df["correct"].astype(float)  # includes partial credit
    return pd.DataFrame({"row_idx": df["row_idx"].astype(int), "y": y})

mq = load_meta(META_QWEN).rename(columns={"y": "qwen"})
mi = load_meta(META_INTVL).rename(columns={"y": "internvl"})

df = dfq.merge(mq, on="row_idx").merge(mi, on="row_idx")

# ------------------- Tokenization / Cleaning ---------------------
stopwords = set("""
the a an and or of to in on for with from as at by is are was were be been being it its that this those these
do does did doing have has had having than then there here their our your his her its they we you he she i
""".split())

# Option markers / noise
extra_noise = {
    "options","option","option(s)","answer","answers",
    "a","b","c","d","e","f","g",
    "a.","b.","c.","d.","e.","f.","g.",
    "a)","b)","c)","d)","e)","f)","g)"
}

# Boilerplate instructional phrases to strip BEFORE tokenizing
boilerplate_patterns = [
    r"multiple choices may be correct, and possibly none\.?",
    r"if none are correct, answer:?\s*n/?a\.?",
    r"answer with the option text\(s\)\.?",
    r"answer with the option text\(s\)\.?\s*if multiple, separate with ' \|\| '\.?"
]

def clean_and_tokenize(text: str):
    t = text.lower()

    # Remove boilerplate phrases
    for pat in boilerplate_patterns:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE)

    # Remove punctuation except apostrophes
    t = re.sub(r"[^a-z0-9'\s]", " ", t)

    # Tokenize
    toks = re.findall(r"\b[a-z][a-z0-9']*\b", t)

    # Filter stopwords / noise / very short tokens
    toks = [w for w in toks if w not in stopwords and w not in extra_noise and len(w) > 1]
    return toks

df["tokens"] = df["prompt"].map(clean_and_tokenize)

# ---------------------- Word frequencies ------------------------
word_counts = Counter()
for toks in df["tokens"]:
    word_counts.update(toks)

top_words = [w for w, _ in word_counts.most_common(TOP_K)]
print("Top words:", top_words)

# -------------------- Per-word accuracies ------------------------
records = []
for w in top_words:
    mask = df["tokens"].apply(lambda toks, ww=w: ww in toks)
    sub = df.loc[mask]
    if len(sub) == 0:
        continue
    records.append({
        "word": w,
        "freq": int(len(sub)),
        "acc_qwen": float(sub["qwen"].mean()),
        "acc_internvl": float(sub["internvl"].mean())
    })

stats = pd.DataFrame(records).sort_values("freq", ascending=False).reset_index(drop=True)
stats.to_csv(OUT_DIR / "vocab_accuracy_stats.csv", index=False)
print(stats.head(10))

# -------------------------- Plot -------------------------------
plt.figure(figsize=(14, 6))
x = np.arange(len(stats))
width = 0.42

plt.bar(x - width/2, stats["acc_qwen"], width, label="Qwen", alpha=0.9)
plt.bar(x + width/2, stats["acc_internvl"], width, label="InternVL", alpha=0.9)

# Proper word labels on x-axis
plt.xticks(x, stats["word"], rotation=60, ha="right", fontsize=10)

ylabel = "Accuracy (strict 1.0)" if BINARY_ACCURACY else "Average score (incl. partial)"
plt.ylabel(ylabel)
plt.xlabel("Word (top 30 by frequency)")
plt.title("Per-word model accuracy")

# Frequency annotations above each word (small gray text)
for i, freq in enumerate(stats["freq"]):
    y_top = max(stats.loc[i, "acc_qwen"], stats.loc[i, "acc_internvl"])
    plt.text(i, y_top + 0.015, f"{freq}", ha="center", fontsize=8, color="gray")

plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
out_png = OUT_DIR / "vocab_accuracy.png"
plt.savefig(out_png, dpi=250)
plt.close()
print(f"✅ Saved: {out_png}")
