#!/usr/bin/env python3
import argparse, json, re, sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===================== Config =====================
MODEL_ID = "/shared/rsaas/ievab2/models/Llama-3.1-8B-Instruct"
EXPERIMENT_DIR = Path("/home/ievab2/run_models/experiment_concat_frames")

def build_paths(frames: int):
    sub = f"experiment_og_concat_{frames}"
    eval_dir = {
        "DEEPSEEK_TINY":     EXPERIMENT_DIR / "DEEPSEEK"         / sub / "LLM_eval_results",
        "INTERNVL":          EXPERIMENT_DIR / "INTERNVL"         / sub / "LLM_eval_results",
        "MOLMO":             EXPERIMENT_DIR / "MOLMO"            / sub / "LLM_eval_results",
        "QWEN":              EXPERIMENT_DIR / "QWEN"             / sub / "LLM_eval_results",
        "VIDEOLLAVA_VICUNA": EXPERIMENT_DIR / "VIDEOLLAVA_VICUNA"/ sub / "LLM_eval_results",
    }
    model_file = {
        "DEEPSEEK_TINY":     EXPERIMENT_DIR / "DEEPSEEK"         / sub / "deepseek_tiny_out.jsonl",
        "INTERNVL":          EXPERIMENT_DIR / "INTERNVL"         / sub / "internvl_out.jsonl",
        "MOLMO":             EXPERIMENT_DIR / "MOLMO"            / sub / "molmo_out.jsonl",
        "QWEN":              EXPERIMENT_DIR / "QWEN"             / sub / "qwen_out.jsonl",
        "VIDEOLLAVA_VICUNA": EXPERIMENT_DIR / "VIDEOLLAVA_VICUNA"/ sub / "llava_vicuna_out.jsonl",
    }
    return eval_dir, model_file
# ==================================================

# Generation settings (deterministic)
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.0
TOP_P = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (
    torch.float16 if torch.cuda.is_available() else torch.float32
)
# ===================================================

# ======= Prompts =======

SYSTEM_PROMPT_OPENENDED = """
You are a strict YES/NO grader. Output EXACTLY one number: 1.0 for correct, 0.0 for incorrect. No words or JSON.

Rules:
- Normalize the ground truth (GT) "yes"/"no" to a boolean.
- Decide the ModelAnswer’s polarity using explicit cues only:
  • Positive cues (YES): "yes", "yeah", "yep", "true", "there is", "there are", "at least one", "exists".
  • Negative cues (NO): "no", "none", "false", "not present", "absent", "there is no", "there are no", "zero".
- Ignore extra sentences, object colors, counts, or scene descriptions; judge ONLY the asserted polarity.
- If the ModelAnswer contains BOTH positive and negative cues or expresses uncertainty ("maybe", "not sure"), grade 0.0.
- If there is no recognizable yes/no cue, grade 0.0.

Few-shot:
GroundTruth: No | ModelAnswer: "No, there are none." → 1.0
GroundTruth: Yes | ModelAnswer: "Yes, at least one." → 1.0
GroundTruth: 3 | ModelAnswer: "There are three." → 1.0
GroundTruth: 3 | ModelAnswer: "Four." → 0.0
GroundTruth: Sphere collides with the cube | ModelAnswer: "Sphere collides with the cylinder." → 0.0
GroundTruth: Yellow | ModelAnswer: "The color is black" → 0.0
GroundTruth: Yellow | ModelAnswer: "It is yellow" → 1.0
GroundTruth: Cube | ModelAnswer: "Cylinder" → 0.0
GroundTruth: Cube | ModelAnswer: "The shape is cube" → 1.0
GroundTruth: No | ModelAnswer: "Yes, there are moving cubes" → 0.0
GroundTruth: No | ModelAnswer: "No, there are no moving cubes" → 1.0
GroundTruth: 5 | ModelAnswer: "There are five stationary objects" → 1.0
GroundTruth: 5 | ModelAnswer: "There are six stationary objects" → 0.0
"""


USER_TEMPLATE = """
Question: {prompt}
GroundTruth: {ground_truth}
ModelAnswer: {model_output}

Return ONLY 1.0 for correct, or 0.0 for incorrect.
"""
# ========================================

FLOAT_RE = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available   # <-- add

def load_judge():
    print(f"[llama-judge] Loading {MODEL_ID} …", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    gen_kwargs = {
        "torch_dtype": DTYPE,
        "device_map": "auto",
    }
    # only enable FA2 if the package is installed; else fall back to SDPA
    if is_flash_attn_2_available():
        gen_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        gen_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **gen_kwargs)
    return tok, model


def chat(tokenizer, model, system_prompt: str, user: str) -> str:
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user},
    ]

    try:
        input_ids = tokenizer.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True
        )
        if isinstance(input_ids, dict):
            input_ids = input_ids["input_ids"]
    except Exception:
        # Fallback if template missing (shouldn't happen for Llama-3.1)
        full = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user}\n\n[ASSISTANT]\n"
        input_ids = tokenizer(full, return_tensors="pt")["input_ids"]

    # place inputs on same device as first real param (handles sharded models)
    try:
        first_param_device = next(p.device for p in model.parameters() if p.device.type != "meta")
    except StopIteration:
        first_param_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_ids = input_ids.to(first_param_device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=8,           # only need "0.0"/"0.5"/"1.0"
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_only = out[0, input_ids.shape[-1]:]
    return tokenizer.decode(gen_only, skip_special_tokens=True).strip()

def extract_score(text: str) -> float:
    m = FLOAT_RE.search(text)
    if not m:
        return 0.0
    try:
        val = float(m.group(0))
        return 0.0 if val < 0 else 1.0 if val > 1 else val
    except Exception:
        return 0.0


def build_user_prompt(ex: Dict[str, Any]) -> str:
    category = ex.get("category") or ""
    prompt   = ex.get("prompt")
    gt       = ex.get("ground_truth", "")
    pred     = ex.get("model_output", "")
    return USER_TEMPLATE.format(
        category=category, prompt=prompt, ground_truth=gt, model_output=pred
    )

# ---------- Resume helpers ----------
def load_existing_pairs(out_path: Path) -> Set[Tuple[str, str]]:
    """Return set of (category_lower, question_id) already evaluated."""
    seen: Set[Tuple[str, str]] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                cat = (obj.get("category") or "").strip().lower()
                qid = obj.get("question_id") or ""
                if cat and qid:
                    seen.add((cat, qid))
    return seen

def input_stream_with_resume(fin, resume_cat: Optional[str], resume_qid: Optional[str]) -> Iterable[str]:
    """Yield lines from fin; if resume_* provided, start from the first match (inclusive)."""
    if not resume_cat and not resume_qid:
        yield from fin
        return

    target_cat = (resume_cat or "").strip().lower()
    target_qid = (resume_qid or "").strip()
    started = False

    for line in fin:
        if not line.strip():
            continue
        try:
            ex = json.loads(line)
        except Exception:
            continue

        cat = (ex.get("category") or "").strip().lower()
        qid = (ex.get("question_id") or "").strip()

        if not started:
            cat_ok = (not target_cat) or (cat == target_cat)
            qid_ok = (not target_qid) or (qid == target_qid)
            if cat_ok and qid_ok:
                started = True
                yield json.dumps(ex)
        else:
            yield json.dumps(ex)

# ---------- Main evaluation ----------
def evaluate_file(
    input_jsonl: Path,
    output_jsonl: Path,
    limit: int = 0,
    resume_cat: Optional[str] = None,
    resume_qid: Optional[str] = None,
):
    tokenizer, model = load_judge()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    append_mode = bool(resume_cat or resume_qid)
    fout_mode = "a" if append_mode else "w"

    already: Set[Tuple[str, str]] = load_existing_pairs(output_jsonl) if append_mode else set()
    if append_mode:
        print(f"[llama-judge] Resume mode ON. Already have {len(already)} evaluated items.", flush=True)
        if resume_cat:
            print(f"[llama-judge] Resuming from category='{resume_cat}'", flush=True)
        if resume_qid:
            print(f"[llama-judge] Resuming from question_id='{resume_qid}'", flush=True)

    n_written = 0
    with input_jsonl.open() as fin, output_jsonl.open(fout_mode) as fout:
        stream = input_stream_with_resume(fin, resume_cat, resume_qid)
        for raw_line in stream:
            if limit and n_written >= limit:
                break
            ex: Dict[str, Any] = json.loads(raw_line)

            # ONLY EVALUATING DESCRIPTIVE NOW
            cat_lower = (ex.get("category") or "").strip().lower()
            if cat_lower != "descriptive":
                continue

            key = (cat_lower, (ex.get("question_id") or "").strip())
            if append_mode and key in already:
                continue

            sys_prompt = SYSTEM_PROMPT_OPENENDED
            user_prompt = build_user_prompt(ex)

            raw = chat(tokenizer, model, sys_prompt, user_prompt)
            score = extract_score(raw)

            ex["llm_score"] = score
            ex["llm_system_prompt"] = sys_prompt
            ex["llm_user_prompt"] = user_prompt
            ex["llm_raw_output"] = raw

            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

            n_written += 1
            if n_written % 10 == 0:
                print(f"[llama-judge] Processed {n_written} descriptive examples so far…", flush=True)

    print(f"[llama-judge] Wrote {n_written} lines to {output_jsonl}")

def main():
    ap = argparse.ArgumentParser(description="Evaluate VLM outputs with a local judge (Llama-3.1-8B-Instruct).")
    ap.add_argument("model_name",
        choices=["MOLMO","QWEN","INTERNVL","DEEPSEEK_TINY","VIDEOLLAVA_VICUNA"],
        help="Which VLM's results to evaluate")
    ap.add_argument("--resume-cat", type=str, default=None,
        help="Resume from this category (inclusive).")
    ap.add_argument("--resume-qid", type=str, default=None,
        help="Resume from this question_id (inclusive).")
    ap.add_argument("--frames", type=int, choices=[8, 32], required=True,
        help="Number of concatenated frames used in this experiment (8 or 32).")

    args = ap.parse_args()

    eval_dir_map, model_file_map = build_paths(args.frames)

    in_path = model_file_map[args.model_name]
    if not in_path.exists():
        print(f"ERROR: Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = eval_dir_map[args.model_name]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model_name.lower()}_evaluated_descriptive.jsonl"

    print(f"[llama-judge] Evaluating {args.model_name}  (frames={args.frames})")
    print(f"[llama-judge] Input : {in_path}")
    print(f"[llama-judge] Output: {out_path}")

    LIMIT = 0  # 0 = no limit

    evaluate_file(
        in_path,
        out_path,
        limit=LIMIT,
        resume_cat=args.resume_cat,
        resume_qid=args.resume_qid,
    )

if __name__ == "__main__":
    main()
