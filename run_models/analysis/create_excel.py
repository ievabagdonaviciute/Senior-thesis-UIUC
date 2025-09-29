#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, Tuple, List, Any
import pandas as pd

# --- Fixed config ---
EVAL_DIR = Path("/home/ievab2/run_models/evaluation/LLM_eval_results")
MCQ_DIR  = Path("/home/ievab2/run_models/evaluation/mcq_eval_results")
OUT_PATH = Path("/home/ievab2/run_models/evaluation/per_question_scores.xlsx")

# Fixed model set (no args)
MODELS = ["QWEN", "VIDEOLLAVA", "SMOLVLM", "DEEPSEEK_TINY"]

# Pretty names for Excel columns
MODEL_DISPLAY_NAMES = {
    "VIDEOLLAVA":    "Video-LLaVA-7B-hf",
    "QWEN":          "Qwen2.5-VL-7B-Instruct",
    "SMOLVLM":       "SmolVLM2-2.2B-Instruct",
    "DEEPSEEK_TINY": "DeepSeek-VL2-Tiny",
}

# Row grouping
CATEGORY_ORDER = ["descriptive", "explanatory", "predictive", "counterfactual"]
CATEGORY_PRETTY = {
    "descriptive": "Descriptive",
    "explanatory": "Explanatory",
    "predictive": "Predictive",
    "counterfactual": "Counterfactual",
}

# Built-in descriptive threshold (not exposed to user)
DESC_THRESHOLD = 0.5

def read_jsonl(path: Path):
    if not path.exists():
        print(f"[warn] Missing file: {path}")
        return
    with path.open() as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue

def load_descriptive_scores(model_key: str) -> Dict[str, float]:
    """
    Reads .../{model}_evaluated_descriptive.jsonl → {question_id: 0/1}
    Fields expected: 'question_id', 'llm_score'
    """
    in_path = EVAL_DIR / f"{model_key.lower()}_evaluated_descriptive.jsonl"
    out: Dict[str, float] = {}
    for ex in read_jsonl(in_path) or []:
        qid = str(ex.get("question_id") or ex.get("id") or "").strip()
        if not qid:
            continue
        try:
            llm_score = float(ex.get("llm_score", 0.0))
        except Exception:
            llm_score = 0.0
        out[qid] = 1.0 if llm_score >= DESC_THRESHOLD else 0.0
    return out

def load_mcq_scores(model_key: str) -> Dict[Tuple[str, str], float]:
    """
    Reads .../{model}_per_question.jsonl → {(category, question_id): score}
    Uses 'score' already computed when you dumped per-question files.
    """
    in_path = MCQ_DIR / f"{model_key.lower()}_per_question.jsonl"
    out: Dict[Tuple[str, str], float] = {}
    for ex in read_jsonl(in_path) or []:
        cat = str(ex.get("category") or "").strip().lower()
        qid = str(ex.get("question_id") or ex.get("id") or "").strip()
        if not cat or not qid:
            continue
        if cat not in {"explanatory", "predictive", "counterfactual"}:
            continue
        try:
            sc = float(ex.get("score", 0.0))
        except Exception:
            sc = 0.0
        out[(cat, qid)] = sc
    return out

def build_matrix(models: List[str]) -> pd.DataFrame:
    """
    Returns DataFrame indexed by (Category, question_id), columns = pretty model names.
    """
    rows_set = set()  # (category, qid)
    per_model_scores: Dict[str, Dict[Tuple[str, str], float]] = {}

    for m in models:
        pretty = MODEL_DISPLAY_NAMES[m]
        desc = load_descriptive_scores(m)
        mcq  = load_mcq_scores(m)

        scores: Dict[Tuple[str, str], float] = {}
        for qid, val in desc.items():
            key = ("descriptive", qid)
            scores[key] = val
            rows_set.add(key)
        for (cat, qid), val in mcq.items():
            key = (cat, qid)
            scores[key] = val
            rows_set.add(key)

        per_model_scores[pretty] = scores

    def row_sort_key(row: Tuple[str, str]):
        cat, qid = row
        return (CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else 999, qid)

    rows_sorted = sorted(rows_set, key=row_sort_key)

    data: Dict[str, List[Any]] = {}
    for pretty in [MODEL_DISPLAY_NAMES[m] for m in models]:
        col_vals: List[Any] = []
        scores = per_model_scores.get(pretty, {})
        for row in rows_sorted:
            col_vals.append(scores.get(row, None))
        data[pretty] = col_vals

    index = pd.MultiIndex.from_tuples(
        [(CATEGORY_PRETTY.get(cat, cat.title()), qid) for cat, qid in rows_sorted],
        names=["Category", "question_id"],
    )
    df = pd.DataFrame(data, index=index)
    df = df[[MODEL_DISPLAY_NAMES[m] for m in models]]
    return df

def write_excel(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="scores")
        ws = writer.sheets["scores"]
        ws.freeze_panes = "C2"  # keep headers + first index visible
        for col in ws.columns:
            max_len = 0
            col_letter = col[0].column_letter
            for cell in col:
                try:
                    val = str(cell.value) if cell.value is not None else ""
                except Exception:
                    val = ""
                max_len = max(max_len, len(val))
            ws.column_dimensions[col_letter].width = min(max(12, max_len + 2), 40)

def main():
    df = build_matrix(MODELS)
    write_excel(df, OUT_PATH)
    print(f"[excel] Wrote matrix → {OUT_PATH}")
    print("[excel] Rows:", len(df.index), "Columns:", len(df.columns))

if __name__ == "__main__":
    main()
