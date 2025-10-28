#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
from typing import Tuple, Dict, Any, List, Set

# ===================== Config =====================
EXPERIMENT_DIR = Path("/home/ievab2/run_models/experiment_concat_frames")

def build_paths(frames: int):
    sub = f"experiment_og_concat_{frames}"
    eval_dir = {
        "DEEPSEEK_TINY":     EXPERIMENT_DIR / "DEEPSEEK"          / sub / "LLM_eval_results",
        "INTERNVL":          EXPERIMENT_DIR / "INTERNVL"          / sub / "LLM_eval_results",
        "MOLMO":             EXPERIMENT_DIR / "MOLMO"             / sub / "LLM_eval_results",
        "QWEN":              EXPERIMENT_DIR / "QWEN"              / sub / "LLM_eval_results",
        "VIDEOLLAVA_VICUNA": EXPERIMENT_DIR / "VIDEOLLAVA_VICUNA" / sub / "LLM_eval_results",
    }
    model_file = {
        "DEEPSEEK_TINY":     EXPERIMENT_DIR / "DEEPSEEK"          / sub / "deepseek_tiny_out.jsonl",
        "INTERNVL":          EXPERIMENT_DIR / "INTERNVL"          / sub / "internvl_out.jsonl",
        "MOLMO":             EXPERIMENT_DIR / "MOLMO"             / sub / "molmo_out.jsonl",
        "QWEN":              EXPERIMENT_DIR / "QWEN"              / sub / "qwen_out.jsonl",
        "VIDEOLLAVA_VICUNA": EXPERIMENT_DIR / "VIDEOLLAVA_VICUNA" / sub / "llava_vicuna_out.jsonl",
    }
    llm_eval_file = {
        "DEEPSEEK_TINY":     EXPERIMENT_DIR / "DEEPSEEK"          / sub / "LLM_eval_results" / "deepseek_tiny_evaluated_descriptive.jsonl",
        "INTERNVL":          EXPERIMENT_DIR / "INTERNVL"          / sub / "LLM_eval_results" / "internvl_evaluated_descriptive.jsonl",
        "MOLMO":             EXPERIMENT_DIR / "MOLMO"             / sub / "LLM_eval_results" / "molmo_evaluated_descriptive.jsonl",
        "QWEN":              EXPERIMENT_DIR / "QWEN"              / sub / "LLM_eval_results" / "qwen_evaluated_descriptive.jsonl",
        "VIDEOLLAVA_VICUNA": EXPERIMENT_DIR / "VIDEOLLAVA_VICUNA" / sub / "LLM_eval_results" / "videollava_vicuna_evaluated_descriptive.jsonl",
    }
    return eval_dir, model_file, llm_eval_file
# ==================================================

# ===== Part 1: Descriptive accuracy from LLM-judged files =====
def descriptive_accuracy(path: Path, threshold: float = 0.5) -> Tuple[float, int, int]:
    total = 0
    correct = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except Exception:
                continue
            # file already filtered to descriptive by your evaluator, but keep the guard:
            if (ex.get("category") or "").strip().lower() != "descriptive":
                continue
            total += 1
            score = ex.get("llm_score", 0.0)
            try:
                score = float(score)
            except Exception:
                score = 0.0
            if score >= threshold:
                correct += 1
    acc = (correct / total) if total > 0 else 0.0
    return acc, correct, total

# ===== Part 2: Deterministic scoring for MCQ categories using Jaccard =====
LETTER_RE = re.compile(r'^\s*([A-Z])\.\s*(.+?)\s*$')
LETTER_ONLY_RE = re.compile(r'^([a-z])\W*$', re.I)
LETTER_PREFIX_RE = re.compile(r'^\s*([a-z])\.\s*(.+)$', re.I)

def parse_options_from_prompt(prompt_text: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for line in (prompt_text or "").splitlines():
        m = LETTER_RE.match(line)
        if m:
            letter = m.group(1).upper()
            text = m.group(2).strip()
            mapping[letter] = text
    return mapping

def split_multi(s: str) -> List[str]:
    return [t.strip() for t in (s or "").split("||") if t.strip()]

def norm_text(s: str) -> str:
    return re.sub(r'\s+', ' ', (s or "").strip().lower())

def map_letters_to_text(items: List[str], letter2text: Dict[str,str]) -> List[str]:
    out: List[str] = []
    for it in items:
        raw = it.strip()
        t = norm_text(raw)

        m_only = LETTER_ONLY_RE.match(t)
        if m_only:
            L = m_only.group(1).upper()
            if L in letter2text:
                out.append(letter2text[L]); continue

        m_prefix = LETTER_PREFIX_RE.match(raw)
        if m_prefix:
            L = m_prefix.group(1).upper()
            if L in letter2text:
                out.append(letter2text[L]); continue

        out.append(raw)
    return out

def normalize_mcq_set(s: str, letter2text: Dict[str, str]) -> Set[str]:
    if not s or norm_text(s) in {"n/a", "na", "none", "no option", "no options"}:
        return set()
    items = split_multi(s)
    items = map_letters_to_text(items, letter2text)
    return {norm_text(x) for x in items}

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    if not u:
        return 1.0
    i = a & b
    return len(i) / len(u)

def deterministic_category_scores(results_path: Path, categories: List[str]) -> Tuple[Dict[str, float], Dict[str, Tuple[int,int]], float, int]:
    cats = {c.lower() for c in categories}
    per_cat_sum: Dict[str, float] = {c: 0.0 for c in cats}
    per_cat_n: Dict[str, int] = {c: 0 for c in cats}
    total_sum = 0.0
    total_n = 0

    with results_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except Exception:
                continue
            cat = (ex.get("category") or "").strip().lower()
            if cat not in cats:
                continue

            prompt = ex.get("prompt", "")
            gt_str = ex.get("ground_truth", "") or ""
            pred_str = ex.get("model_output", "") or ""

            letter2text = parse_options_from_prompt(prompt)
            gt_set   = normalize_mcq_set(gt_str,   letter2text)
            pred_set = normalize_mcq_set(pred_str, letter2text)
            s = jaccard(pred_set, gt_set)

            per_cat_sum[cat] += s
            per_cat_n[cat]   += 1
            total_sum        += s
            total_n          += 1

    per_cat_avg = {c: (per_cat_sum[c] / per_cat_n[c] if per_cat_n[c] > 0 else 0.0) for c in cats}
    per_cat_counts = {c: (per_cat_n[c], 0) for c in cats}
    return per_cat_avg, per_cat_counts, total_sum, total_n

# ===== Extra Part: Dump per-question scores (raw-only; descriptive score assumed to be copied if present) =====
def _load_desc_scores_from_llm_eval(llm_eval_path: Path, threshold: float) -> Dict[str, float]:
    """
    Returns a dict mapping question_id -> 0.0/1.0 for Descriptive,
    using the llm_score from the LLM-evaluated descriptive file.
    """
    out: Dict[str, float] = {}
    with llm_eval_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except Exception:
                continue
            cat = (ex.get("category") or "").strip().lower()
            if cat != "descriptive":
                continue
            qid = ex.get("question_id") or ex.get("qid")
            if qid is None:
                continue
            try:
                raw = float(ex.get("llm_score", 0.0))
            except Exception:
                raw = 0.0
            out[str(qid)] = 1.0 if raw >= threshold else 0.0
    return out

def dump_per_question_scores(raw_results_path: Path, llm_eval_path: Path, out_dir: Path, model_name: str, threshold: float = 0.5):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name.lower()}_per_question.jsonl"

    # NEW: build descriptive scores from LLM-eval file
    desc_scores = _load_desc_scores_from_llm_eval(llm_eval_path, threshold)

    with raw_results_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                ex = json.loads(line)
            except Exception:
                continue

            cat = (ex.get("category") or "").strip().lower()
            qid = ex.get("question_id") or ex.get("qid")
            qid = str(qid) if qid is not None else None

            score = 0.0

            if cat == "descriptive":
                # Use the LLM-eval score (thresholded) by question_id
                if qid is not None and qid in desc_scores:
                    score = desc_scores[qid]
                else:
                    # If missing in the LLM-eval file, mark incorrect (or set to None if you prefer)
                    score = 0.0

            elif cat in {"explanatory", "predictive", "counterfactual"}:
                prompt = ex.get("prompt", "")
                letter2text = parse_options_from_prompt(prompt)
                gt_set   = normalize_mcq_set(ex.get("ground_truth",""), letter2text)
                pred_set = normalize_mcq_set(ex.get("model_output",""), letter2text)
                score = jaccard(pred_set, gt_set)

            ex["score"] = score
            # (Optional) keep original llm_score if present in raw; not needed for correctness
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[per_question] Wrote per-question scores → {out_path}")


# ===================== Main =====================
def main():
    ap = argparse.ArgumentParser(description="Compute final scores combining LLM descriptive and deterministic MCQ (paths via build_paths).")
    ap.add_argument("model_name",
                    choices=["MOLMO","QWEN","INTERNVL","DEEPSEEK_TINY","VIDEOLLAVA_VICUNA"],
                    help="Which model to score")
    ap.add_argument("--frames", type=int, choices=[8, 32], required=True,
                    help="Number of concatenated frames used (8 or 32)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Descriptive score threshold to count as correct (default 0.5)")
    args = ap.parse_args()

    eval_dir_map, model_file_map, llm_eval_file_map = build_paths(args.frames)

    # Paths for this model & frames
    eval_dir_for_model: Path = eval_dir_map[args.model_name]
    raw_results_path: Path   = model_file_map[args.model_name]
    llm_eval_path: Path      = llm_eval_file_map[args.model_name]

    if not raw_results_path.exists():
        raise FileNotFoundError(f"Raw results file not found: {raw_results_path}")
    if not llm_eval_path.exists():
        raise FileNotFoundError(f"LLM-eval descriptive file not found: {llm_eval_path}")

    # --- Part 1: Descriptive from the specific LLM-eval file map ---
    desc_acc, desc_correct, desc_total = descriptive_accuracy(llm_eval_path, threshold=args.threshold)

    # --- Part 2: Deterministic MCQ on original results ---
    mcq_cats = ["explanatory", "predictive", "counterfactual"]
    per_cat_avg, per_cat_counts, mcq_sum, mcq_n = deterministic_category_scores(raw_results_path, mcq_cats)

    # --- Combine for overall weighted total ---
    total_items = desc_total + mcq_n
    total_sum_scores = (desc_acc * desc_total) + mcq_sum
    overall = (total_sum_scores / total_items) if total_items > 0 else 0.0

    # --- Save final scores into .../<MODEL>/<sub>/scores/
    scores_dir_for_model = eval_dir_for_model.parent / "scores"
    scores_dir_for_model.mkdir(parents=True, exist_ok=True)
    out_path = scores_dir_for_model / f"{args.model_name.lower()}_final_scores.jsonl"

    result_obj: Dict[str, Any] = {
        "model_key": args.model_name,
        "dataset": "CLEVRER",
        "frames": args.frames,
        "Descriptive": round(desc_acc, 4),
        "Explanatory": round(per_cat_avg.get("explanatory", 0.0), 4),
        "Predictive":  round(per_cat_avg.get("predictive",  0.0), 4),
        "Counterfactual": round(per_cat_avg.get("counterfactual", 0.0), 4),
        "Total": round(overall, 4),
        "counts": {
            "descriptive_correct": desc_correct,
            "descriptive_total": desc_total,
            "explanatory_total": per_cat_counts.get("explanatory", (0,0))[0],
            "predictive_total":  per_cat_counts.get("predictive",  (0,0))[0],
            "counterfactual_total": per_cat_counts.get("counterfactual", (0,0))[0],
            "all_items_total": total_items,
        },
        "paths": {
            "llm_eval_descriptive_jsonl": str(llm_eval_path),
            "raw_results_jsonl": str(raw_results_path),
            "scores_dir": str(scores_dir_for_model),
        },
    }

    with out_path.open("w") as fout:
        fout.write(json.dumps(result_obj, ensure_ascii=False, indent=2) + "\n")

    # --- Console summary ---
    print(f"[score_eval] {args.model_name} | frames={args.frames}")
    print(f"  Descriptive     : {desc_acc:.4f}  ({desc_correct}/{desc_total})")
    for c in mcq_cats:
        print(f"  {c.capitalize():<14}: {per_cat_avg.get(c, 0.0):.4f}  (n={per_cat_counts.get(c,(0,0))[0]})")
    print(f"  Total           : {overall:.4f}  (items={total_items})")
    print(f"[score_eval] Wrote: {out_path}")

    # --- Extra: per-question scores -> scores/ as well (raw-only; descriptive uses llm_score if baked into raw)
    dump_per_question_scores(raw_results_path, llm_eval_path, scores_dir_for_model, args.model_name, threshold=args.threshold)

if __name__ == "__main__":
    main()
