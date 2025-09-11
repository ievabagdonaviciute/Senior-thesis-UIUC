# #!/usr/bin/env python3
# import argparse, json, re, sys
# from pathlib import Path
# from typing import Any, Dict

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # ===================== Config =====================
# MODEL_ID = "/home/ievab2/models/Qwen2.5-7B-Instruct"

# RESULTS_DIR = Path("/home/ievab2/run_models/results")
# EVAL_DIR    = Path("/home/ievab2/run_models/evaluation")
# MODEL_NAME_TO_FILE = {
#     "VIDEOLLAVA":    RESULTS_DIR / "videollava_out.jsonl",
#     "QWEN":          RESULTS_DIR / "qwen_out.jsonl",
#     "SMOLVLM":       RESULTS_DIR / "smolvlm_out.jsonl",
#     "DEEPSEEK_TINY": RESULTS_DIR / "deepseek_tiny_out.jsonl",
# }

# # Generation settings (deterministic)
# MAX_NEW_TOKENS = 64
# TEMPERATURE = 0.0
# TOP_P = 1.0

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# # ===================================================


# # ======= Your exact prompts (unchanged) =======
# SYSTEM_PROMPT_OPENENDED = """You are a rigorous grader for short answers to video-QA questions.
# Return ONLY a single floating-point number between 0 and 1 (e.g., 0.0, 0.5, 1.0). No words, no JSON.

# Normalization rules you MUST apply internally before scoring:
# - Convert number words to digits (e.g., "three"→3). Treat "none", "no objects", "0", "zero" as 0.
# - Treat Yes/No as true/False

# Scoring by category: strict exactness after normalization (binary). Output 1.0 for a correct match, otherwise 0.0.

# Output ONLY the number, nothing else.

# Examples:

# [EX1 Descriptive binary]

# Category: descriptive
# Question: How many stationary blue objects are there when the cube enters the scene?
# GroundTruth: 1

# [EX1.1 correct]  
# ModelAnswer: There is one stationary blue object when the cube enters the scene.  
# Score (your output): 1.0  

# [EX1.2 correct]  
# ModelAnswer: One
# Score (your output): 1.0  

# [EX1.3 incorrect]  
# ModelAnswer: 3  
# Score (your output): 0.0  
# """

# SYSTEM_PROMPT_MCQ = """You are a rigorous grader for short answers to video-QA questions.
# Return ONLY a single floating-point number between 0 and 1 (e.g., 0.0, 0.5, 1.0). No words, no JSON.

# Normalization rules you MUST apply internally before scoring:
# - Convert number words to digits (e.g., "three"→3). Treat "none", "no objects", "0", "zero" as 0.
# - Treat Yes/No as true/False
# - Ground truth may have ZERO, ONE, or MULTIPLE correct answers. Multiple correct answers are separated by "||".
# - When both GT and prediction are sets of options, compute **Jaccard score**: |Pred ∩ GT| / |Pred ∪ GT|.
#   - If both sets are empty (no correct answers and model selected nothing), score = 1.0.
#   - If union is non-empty, use the standard ratio.

# Scoring treat as MCQ with possibly multiple correct answers and compute Jaccard.

# Output ONLY the number, nothing else.

# Examples:

# [EX1 MCQ]

# Category: explanatory  
# Question: Which of the following is not responsible for the blue object's colliding with the cyan cylinder?  
# GroundTruth: the presence of the cube || the collision between the cyan cylinder and the purple object || the presence of the rubber sphere  

# [EX1.1 perfect match]  
# ModelAnswer: the presence of the cube || the collision between the cyan cylinder and the purple object || the presence of the rubber sphere  
# Score (your output): 1.0  

# [EX1.2 under-selection]  
# ModelAnswer: the presence of the cube  
# ∩=1, ∪=3 → Score (your output): 0.33  

# [EX1.3 over-selection with extra wrong]  
# ModelAnswer: the presence of the cube || the collision between the cyan cylinder and the purple object || the green object  
# ∩=2, ∪= → Score (your output): 

# [EX1.4 over-selection with 2-extra wrongs]  
# ModelAnswer: the presence of the cube || the green object || the collision between the green sphere and the purple object
# ∩=1, ∪=4 → Score (your output): 0.5  

# """

# USER_TEMPLATE = """Category: {category}
# Question: {prompt}
# GroundTruth (note: may be empty or multiple separated by '||'): {ground_truth}
# ModelAnswer: {model_output}

# Return ONLY the numeric score.
# """
# # ========================================

# FLOAT_RE = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")

# def load_judge():
#     print(f"[qwen-judge] Loading {MODEL_ID} …", flush=True)
#     tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
#     if tok.pad_token is None:
#         tok.pad_token = tok.eos_token  # ensure padding exists

#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_ID,
#         torch_dtype=DTYPE,
#         device_map="auto",
#         trust_remote_code=True
#     )
#     return tok, model

# def chat(tokenizer, model, system_prompt: str, user: str) -> str:
#     msgs = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user",   "content": user},
#     ]
#     try:
#         input_ids = tokenizer.apply_chat_template(
#             msgs, return_tensors="pt", add_generation_prompt=True
#         )
#     except Exception:
#         # Fallback if model lacks chat template
#         full = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user}\n\n[ASSISTANT]\n"
#         input_ids = tokenizer(full, return_tensors="pt")["input_ids"]

#     # figure out a real device for sharded models (device_map="auto")
#     try:
#         first_param_device = next(p.device for p in model.parameters() if p.device.type != "meta")
#     except StopIteration:
#         # extremely rare, but just in case
#         first_param_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     if isinstance(input_ids, dict):
#         # in case the fallback returned a dict
#         input_ids = input_ids.get("input_ids")
#     input_ids = input_ids.to(first_param_device)

#     with torch.no_grad():
#         out = model.generate(
#             input_ids=input_ids,
#             max_new_tokens=MAX_NEW_TOKENS,
#             temperature=TEMPERATURE,
#             top_p=TOP_P,
#             do_sample=False,
#             pad_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#         )

#     # decode only the generated continuation
#     gen_only = out[0, input_ids.shape[-1]:]
#     return tokenizer.decode(gen_only, skip_special_tokens=True).strip()


# def extract_score(text: str) -> float:
#     """Extract the first float-like token and clamp to [0,1]."""
#     m = FLOAT_RE.search(text)
#     if not m:
#         return 0.0
#     try:
#         val = float(m.group(0))
#         if val < 0: return 0.0
#         if val > 1: return 1.0
#         return val
#     except Exception:
#         return 0.0

# def pick_system_prompt(category: str) -> str:
#     if (category or "").strip().lower() == "descriptive":
#         return SYSTEM_PROMPT_OPENENDED
#     return SYSTEM_PROMPT_MCQ

# def build_user_prompt(ex: Dict[str, Any]) -> str:
#     category = ex.get("category") or ""
#     prompt   = ex.get("prompt") or ex.get("question") or ""
#     gt       = ex.get("ground_truth", "")
#     pred     = ex.get("model_output", "")
#     return USER_TEMPLATE.format(
#         category=category, prompt=prompt, ground_truth=gt, model_output=pred
#     )

# def evaluate_file(input_jsonl: Path, output_jsonl: Path, limit: int = 0):
#     tokenizer, model = load_judge()
#     output_jsonl.parent.mkdir(parents=True, exist_ok=True)

#     n_written = 0
#     with input_jsonl.open() as fin, output_jsonl.open("w") as fout:
#         for line in fin:
#             if limit and n_written >= limit:
#                 break
#             if not line.strip():
#                 continue
#             ex: Dict[str, Any] = json.loads(line)

#             sys_prompt = pick_system_prompt(ex.get("category", ""))
#             user_prompt = build_user_prompt(ex)

#             raw = chat(tokenizer, model, sys_prompt, user_prompt)
#             score = extract_score(raw)

#             # Preserve original fields, append only the LLM score
#             ex["llm_score"] = score
#             fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
#             n_written += 1

#     print(f"[qwen-judge] Wrote {n_written} lines to {output_jsonl}")

# def main():
#     ap = argparse.ArgumentParser(description="Evaluate VLM outputs with a local LLaMA judge.")
#     ap.add_argument("model_name",
#                     choices=["VIDEOLLAVA","QWEN","SMOLVLM","DEEPSEEK_TINY"],
#                     help="Which VLM's results to evaluate")
#     args = ap.parse_args()

#     in_path = MODEL_NAME_TO_FILE[args.model_name]
#     if not in_path.exists():
#         print(f"ERROR: Input file not found: {in_path}", file=sys.stderr)
#         sys.exit(1)

#     out_path = EVAL_DIR / f"{args.model_name.lower()}_evaluated.jsonl"

#     print(f"[qwen-judge] Evaluating {args.model_name}")
#     print(f"[qwen-judge] Input : {in_path}")
#     print(f"[qwen-judge] Output: {out_path}")

#     LIMIT = 1 # None for the full file eval

#     evaluate_file(in_path, out_path, limit=LIMIT)

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
import argparse, json, re, sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===================== Config =====================
MODEL_ID = "/home/ievab2/models/Qwen2.5-7B-Instruct"

RESULTS_DIR = Path("/home/ievab2/run_models/results")
EVAL_DIR    = Path("/home/ievab2/run_models/evaluation/LLM_eval_results")

MODEL_NAME_TO_FILE = {
    "VIDEOLLAVA":    RESULTS_DIR / "videollava_out.jsonl",
    "QWEN":          RESULTS_DIR / "qwen_out.jsonl",
    "SMOLVLM":       RESULTS_DIR / "smolvlm_out.jsonl",
    "DEEPSEEK_TINY": RESULTS_DIR / "deepseek_tiny_out.jsonl",
}

# Generation settings (deterministic)
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.0
TOP_P = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
# ===================================================

# ======= Prompts =======
SYSTEM_PROMPT_OPENENDED = """You are a rigorous grader for short answers to video-QA questions.
Return ONLY a single floating-point number between 0 and 1 (e.g., 0.0, 0.5, 1.0). No words, no JSON.

Normalization rules you MUST apply internally before scoring:
- Convert number words to digits (e.g., "three"→3). Treat "none", "no objects", "0", "zero" as 0.
- Treat Yes/No as true/False

Scoring by category: strict exactness after normalization (binary). Output 1.0 for a correct match, otherwise 0.0.

Output ONLY the number, nothing else.

EXAMPLES:

Category: descriptive
Question: How many stationary blue objects are there when the cube enters the scene?
GroundTruth: 1

[EX1.1 correct]  
ModelAnswer: There is one stationary blue object when the cube enters the scene.  
Score (your output): 1.0  

[EX1.2 correct]  
ModelAnswer: One
Score (your output): 1.0  

[EX1.3 incorrect]  
ModelAnswer: 3  
Score (your output): 0.0  
"""

SYSTEM_PROMPT_MCQ = """You are a rigorous grader for short answers to video-QA questions.
Return ONLY a single floating-point number between 0 and 1 (e.g., 0.0, 0.5, 1.0). No words, no JSON.

Normalization rules you MUST apply internally before scoring:
- Convert number words to digits (e.g., "three"→3). Treat "none", "no objects", "0", "zero" as 0.
- Treat Yes/No as true/False
- Ground truth may have ZERO, ONE, or MULTIPLE correct answers. Multiple correct answers are separated by "||".
- When both GT and prediction are sets of options, compute **Jaccard score**: |Pred ∩ GT| / |Pred ∪ GT|.
  - If both sets are empty (no correct answers and model selected nothing), score = 1.0.
  - If union is non-empty, use the standard ratio.
  - If options are listed (A., B., C.…), map letter answers (A/B/…) to their exact option text before scoring.
  - Treat "N/A" as selecting no options (empty set).

Scoring treat as MCQ with possibly multiple correct answers and compute Jaccard.

Output ONLY the number, nothing else.

EXAMPLES:

Category: counterfactual  
Prompt: Which event will happen next?\nOptions:\nA. The red object collides with the cylinder\nB. The green sphere collides with the cube\nC. The green sphere and the red object collide\nMultiple choices may be correct, and possibly none.\nIf none are correct, answer: N/A.\nAnswer with the option text(s). If multiple, separate with ' || '.
GroundTruth: The red object collides with the cylinder || The green sphere collides with the cube

Imply that the GT is A and B

[EX1.1 perfect match]  
ModelAnswer: A || B
Score (your output): 1.0  

[EX1.1 perfect match]  
ModelAnswer: The green sphere collides with the cube || The red object collides with the cylinder
Score (your output): 1.0  

[EX1.2 under-selection]  
ModelAnswer: A
∩=1, ∪=2 → Score (your output): 0.5  

[EX1.3 under-selection]  
ModelAnswer: The green sphere collides with the cube
∩=1, ∪=2 → Score (your output): 0.5 

[EX1.4 over-selection with extra wrong]  
ModelAnswer: The green sphere collides with the cube || The green sphere and the red object collide
∩=1, ∪=3 → Score (your output): 0.33  

[EX1.5 wrong]
ModelAnswer: C
∩=0, ∪=3 → Score (your output): 0.0

[EX1.5 wrong]
ModelAnswer: N/A
∩=0, ∪=2 → Score (your output): 0.0
"""

USER_TEMPLATE = """Category: {category}
Question: {prompt}
GroundTruth (note: may be empty or multiple separated by '||'): {ground_truth}
ModelAnswer: {model_output}

Return ONLY the numeric score according to the calculation rules.
"""
# ========================================

FLOAT_RE = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")

def load_judge():
    print(f"[qwen-judge] Loading {MODEL_ID} …", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # ensure padding exists

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True
    )
    return tok, model

def chat(tokenizer, model, system_prompt: str, user: str) -> str:
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user},
    ]
    # print("[DEBUG] system_prompt length (chars):", len(system_prompt))
    # print("[DEBUG] user_prompt length (chars):", len(user))
    #print("[DEBUG] total input tokens:", input_ids.shape[-1])

    try:
        input_ids = tokenizer.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True
        )
        if isinstance(input_ids, dict):
            input_ids = input_ids["input_ids"]

    except Exception:
        # Fallback if model lacks chat template
        full = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user}\n\n[ASSISTANT]\n"
        input_ids = tokenizer(full, return_tensors="pt")["input_ids"]

    # try:
    #     print("[DEBUG] total input tokens:", input_ids.shape[-1])
    #     print("[DEBUG] First 500 decoded chars:\n", tokenizer.decode(input_ids[0][:500]))
    #     print("[DEBUG] Last 500 decoded chars:\n", tokenizer.decode(input_ids[0][-500:]))
    # except Exception:
    #     pass  

    # find a real device for sharded models
    try:
        first_param_device = next(p.device for p in model.parameters() if p.device.type != "meta")
    except StopIteration:
        first_param_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(input_ids, dict):
        input_ids = input_ids.get("input_ids")
    input_ids = input_ids.to(first_param_device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=8,       # only need "0.0"/"0.5"/"1.0"
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

def pick_system_prompt(category: str) -> str:
    if (category or "").strip().lower() == "descriptive":
        return SYSTEM_PROMPT_OPENENDED
    return SYSTEM_PROMPT_MCQ

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
            # Must satisfy provided filters; both if both set
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

    # If resuming, append; else overwrite
    append_mode = bool(resume_cat or resume_qid)
    fout_mode = "a" if append_mode else "w"

    # If appending and file exists, collect already-done (cat,qid) to dedupe
    already: Set[Tuple[str, str]] = load_existing_pairs(output_jsonl) if append_mode else set()
    if append_mode:
        print(f"[qwen-judge] Resume mode ON. Already have {len(already)} evaluated items.", flush=True)
        if resume_cat:
            print(f"[qwen-judge] Resuming from category='{resume_cat}'", flush=True)
        if resume_qid:
            print(f"[qwen-judge] Resuming from question_id='{resume_qid}'", flush=True)

    n_written = 0
    with input_jsonl.open() as fin, output_jsonl.open(fout_mode) as fout:
        stream = input_stream_with_resume(fin, resume_cat, resume_qid)
        for raw_line in stream:
            if limit and n_written >= limit:
                break
            ex: Dict[str, Any] = json.loads(raw_line)

            # ONLY EVALUATING DESCRIPTIVE NOW
            cat_lower = (ex.get("category") or "").strip().lower()          # NEW
            if cat_lower != "descriptive":                                  # NEW
                continue

            # Dedupe if appending
            key = ((ex.get("category") or "").strip().lower(), (ex.get("question_id") or "").strip())
            if append_mode and key in already:
                continue

            sys_prompt = pick_system_prompt(ex.get("category", ""))
            user_prompt = build_user_prompt(ex)

            raw = chat(tokenizer, model, sys_prompt, user_prompt)
            score = extract_score(raw)

            ex["llm_score"] = score
            ex["llm_system_prompt"] = sys_prompt
            ex["llm_user_prompt"] = user_prompt
            ex["llm_raw_output"] = raw

            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

            n_written += 1
            if n_written % 10 == 0:   # change 10 → 1 if you want every line
                print(f"[qwen-judge] Processed {n_written} descriptive examples so far…", flush=True)


    print(f"[qwen-judge] Wrote {n_written} lines to {output_jsonl}")

def main():
    ap = argparse.ArgumentParser(description="Evaluate VLM outputs with a local judge (Qwen2.5-7B-Instruct).")
    ap.add_argument("model_name",
                    choices=["VIDEOLLAVA","QWEN","SMOLVLM","DEEPSEEK_TINY"],
                    help="Which VLM's results to evaluate")
    ap.add_argument("--resume-cat", type=str, default=None,
                    help="Resume from this category (inclusive).")
    ap.add_argument("--resume-qid", type=str, default=None,
                    help="Resume from this question_id (inclusive).")
    args = ap.parse_args()

    in_path = MODEL_NAME_TO_FILE[args.model_name]
    if not in_path.exists():
        print(f"ERROR: Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_path = EVAL_DIR / f"{args.model_name.lower()}_evaluated_descriptive.jsonl"

    print(f"[qwen-judge] Evaluating {args.model_name}")
    print(f"[qwen-judge] Input : {in_path}")
    print(f"[qwen-judge] Output: {out_path}")

    LIMIT = 0 # 0 MEANS NO LIMIT - TRAVERSES THE WHOLE DESCRIPTIVE FILE

    evaluate_file(
        in_path,
        out_path,
        limit=LIMIT,
        resume_cat=args["resume_cat"] if isinstance(args, dict) else args.resume_cat,
        resume_qid=args["resume_qid"] if isinstance(args, dict) else args.resume_qid,
    )

if __name__ == "__main__":
    main()
