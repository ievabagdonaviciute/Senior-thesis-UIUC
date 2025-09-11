#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
from typing import Tuple, Dict, Any, List, Set

# --- Folders ---
EVAL_SRC_DIR  = Path("/home/ievab2/run_models/evaluation/LLM_eval_results")
RESULTS_DIR   = Path("/home/ievab2/run_models/results")
SCORE_OUT_DIR = Path("/home/ievab2/run_models/evaluation/score_results")

# --- Short keys you’ll pass on CLI ---
MODEL_CHOICES = ["VIDEOLLAVA","QWEN","SMOLVLM","DEEPSEEK_TINY", "TEST"]

# --- Pretty names you want in final JSONL ---
MODEL_DISPLAY_NAMES = {
    "VIDEOLLAVA":    "Video-LLaVA-7B-hf",
    "QWEN":          "Qwen2.5-7B-Instruct",
    "SMOLVLM":       "SmolVLM2-2.2B-Instruct",
    "DEEPSEEK_TINY": "DeepSeek-VL2-Tiny",
    "TEST":          "test",
}

# --- Raw results filenames per model (the ones with 'prompt','ground_truth','model_output') ---
MODEL_TO_RESULTS_FILE = {
    "VIDEOLLAVA":    RESULTS_DIR / "videollava_out.jsonl",
    "QWEN":          RESULTS_DIR / "qwen_out.jsonl",
    "SMOLVLM":       RESULTS_DIR / "smolvlm_out.jsonl",
    "DEEPSEEK_TINY": RESULTS_DIR / "deepseek_tiny_out.jsonl",
    "TEST":          RESULTS_DIR / "test_results.jsonl",
}

# ===== Part 1 (unchanged): Descriptive accuracy from LLM-judged files =====
def find_eval_file(model_name: str) -> Path:
    expected = EVAL_SRC_DIR / f"{model_name.lower()}_evaluated.jsonl"
    if expected.exists():
        return expected
    # fallback
    key = model_name.lower()
    for p in EVAL_SRC_DIR.glob("*.jsonl"):
        if key in p.stem.lower():
            return p
    raise FileNotFoundError(f"No evaluated JSONL found for {model_name} in {EVAL_SRC_DIR}")

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
    """
    Extract mapping like {'A': 'option text', 'B': 'option text', ...} from the Options: block.
    """
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

        # Case 1: bare letter like "a" or "A." / "A )"
        m_only = LETTER_ONLY_RE.match(t)
        if m_only:
            L = m_only.group(1).upper()
            if L in letter2text:
                out.append(letter2text[L]); continue

        # Case 2: "A. <text>" → use the canonical text for letter A
        m_prefix = LETTER_PREFIX_RE.match(raw)  # use raw to keep original spacing
        if m_prefix:
            L = m_prefix.group(1).upper()
            if L in letter2text:
                out.append(letter2text[L]); continue

        # Else assume it's already option text
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
    """
    Returns:
      per_cat_avg: {cat: mean_jaccard}
      per_cat_counts: {cat: (num_items_contributed, dummy_correct_unused)}
      total_sum_scores: sum of all per-item jaccard across all requested cats
      total_items: count of all requested items
    """
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

            prompt = ex.get("prompt")
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
    # We don’t use “correct” for MCQ Jaccard; keep counts so you can print (n)
    per_cat_counts = {c: (per_cat_n[c], 0) for c in cats}
    return per_cat_avg, per_cat_counts, total_sum, total_n

def main():
    ap = argparse.ArgumentParser(description="Compute final scores combining LLM descriptive and deterministic MCQ.")
    ap.add_argument("model_name", choices=MODEL_CHOICES, help="Which model to score")
    ap.add_argument("--threshold", type=float, default=0.5, help="Descriptive score threshold to count as correct (default 0.5)")
    args = ap.parse_args()

    # --- Part 1: Descriptive from LLM evals ---
    eval_in_path = find_eval_file(args.model_name)
    desc_acc, desc_correct, desc_total = descriptive_accuracy(eval_in_path, threshold=args.threshold)

    # --- Part 2: Deterministic MCQ on original results ---
    results_path = MODEL_TO_RESULTS_FILE[args.model_name]
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    mcq_cats = ["explanatory", "predictive", "counterfactual"]
    per_cat_avg, per_cat_counts, mcq_sum, mcq_n = deterministic_category_scores(results_path, mcq_cats)

    # --- Combine for overall weighted total ---
    # Overall = (sum of binary descriptive per-item scores + sum of MCQ jaccard scores) / total_items
    total_items = desc_total + mcq_n
    total_sum_scores = (desc_acc * desc_total) + mcq_sum
    overall = (total_sum_scores / total_items) if total_items > 0 else 0.0

    # --- Prepare JSON output ---
    SCORE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCORE_OUT_DIR / f"{args.model_name.lower()}_final_scores.jsonl"

    result_obj: Dict[str, Any] = {
        "model_name": MODEL_DISPLAY_NAMES[args.model_name],
        "dataset": "CLEVRER",
        "Descriptive": round(desc_acc, 4),
        "Explanatory": round(per_cat_avg.get("explanatory", 0.0), 4),
        "Predictive":  round(per_cat_avg.get("predictive", 0.0), 4),
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
    }

    # Overwrite with the full combined results
    with out_path.open("w") as fout:
        fout.write(json.dumps(result_obj, ensure_ascii=False, indent=2) + "\n")


    # --- Console summary ---
    print(f"[score_eval] {MODEL_DISPLAY_NAMES[args.model_name]} on CLEVRER")
    print(f"  Descriptive     : {desc_acc:.4f}  ({desc_correct}/{desc_total})")
    for c in mcq_cats:
        print(f"  {c.capitalize():<14}: {per_cat_avg.get(c, 0.0):.4f}  (n={per_cat_counts.get(c,(0,0))[0]})")
    print(f"  Total           : {overall:.4f}  (items={total_items})")
    print(f"[score_eval] Wrote: {out_path}")

if __name__ == "__main__":
    main()
