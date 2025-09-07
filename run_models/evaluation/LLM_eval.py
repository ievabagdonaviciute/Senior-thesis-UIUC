

# FIX AND TEST BEFORE FULLY WORKING
# DOWNLOAD LLAMA LOCALLY





#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- Config ----------------
MODEL_ID = "/home/ievab2/models/llama-7b-instruct"  # your local LLaMA 7B
MAX_NEW_TOKENS = 192
TEMPERATURE = 0.0
TOP_P = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
# -----------------------------------------

# === LLM only returns a score ∈ [0,1] ===

SYSTEM_PROMPT = """You are a rigorous grader for short answers to video-QA questions.
Return ONLY a single floating-point number between 0 and 1 (e.g., 0.0, 0.5, 1.0). No words, no JSON.

Normalization rules you MUST apply internally before scoring:
- Convert number words to digits (e.g., "three"→3). Treat "none", "no objects", "0", "zero" as 0.
- Treat Yes/No as true/False
- Ground truth may have ZERO, ONE, or MULTIPLE correct answers. Multiple correct answers are separated by "||".
- When both GT and prediction are sets of options, compute **Jaccard score**: |Pred ∩ GT| / |Pred ∪ GT|.
  - If both sets are empty (no correct answers and model selected nothing), score = 1.0.
  - If union is non-empty, use the standard ratio.

Scoring by category:
- Category "descriptive": strict exactness after normalization (binary). Output 1.0 for a correct match, otherwise 0.0.
- Categories "explanatory", "predictive", "counterfactual": treat as MCQ with possibly multiple correct answers and compute Jaccard.

Output ONLY the number, nothing else.

Examples:

[EX1 MCQ]

Category: explanatory  
Question: Which of the following is not responsible for the blue object's colliding with the cyan cylinder?  
GroundTruth: the presence of the cube || the collision between the cyan cylinder and the purple object || the presence of the rubber sphere  

[EX1.1 perfect match]  
ModelAnswer: the presence of the cube || the collision between the cyan cylinder and the purple object || the presence of the rubber sphere  
Score (your output): 1.0  

[EX1.2 under-selection]  
ModelAnswer: the presence of the cube  
∩=1, ∪=3 → Score (your output): 0.33  

[EX1.3 over-selection with extra wrong]  
ModelAnswer: the presence of the cube || the collision between the cyan cylinder and the purple object || the green object  
∩=2, ∪=4 → Score (your output): 0.5  

---

[EX2 Descriptive binary]

Category: descriptive
Question: How many stationary blue objects are there when the cube enters the scene?
GroundTruth: 1

[EX3.1 correct]  
ModelAnswer: There is one stationary blue object when the cube enters the scene.  
Score (your output): 1.0  

[EX3.2 incorrect]  
ModelAnswer: 3  
Score (your output): 0.0  

"""

USER_TEMPLATE = """Grade this item and return ONLY JSON with "score" (float).

Category: {category}
Question: {prompt}
GroundTruth (note: may be empty or multiple separated by '||'): {ground_truth}
ModelAnswer: {model_output}
"""

def load_model():
    print(f"[llama-judge] Loading {MODEL_ID} …", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto"
    )
    return tok, model

def query_llama(tokenizer, model, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(DEVICE)
    out = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Be robust: extract the first {...} JSON object
    m = re.search(r"\{.*?\}", text, re.S)
    return m.group(0) if m else text

def parse_score(s: str) -> float:
    try:
        obj = json.loads(s)
        score = float(obj.get("score"))
        if score < 0: return 0.0
        if score > 1: return 1.0
        return score
    except Exception:
        return 0.0  # hard-fail safe

def verdict_from_score(score: float) -> str:
    if score >= 1.0 - 1e-9:
        return "Correct"
    if score <= 0.0 + 1e-9:
        return "Incorrect"
    return "Partially Correct"

def main(input_jsonl: str, output_jsonl: str):
    tok, model = load_model()
    out_path = Path(output_jsonl)
    with open(input_jsonl) as f, out_path.open("w") as fout:
        for line in f:
            ex: Dict[str, Any] = json.loads(line)
            prompt = USER_TEMPLATE.format(**ex)
            raw = query_llama(tok, model, prompt)
            score = parse_score(raw)
            ex["judge_score"] = score
            ex["judge_verdict"] = verdict_from_score(score)
            ex["judge_raw"] = raw  # keep raw for auditing; remove if you want minimal output
            fout.write(json.dumps(ex) + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="input JSONL (with category, prompt, ground_truth, model_output)")
    ap.add_argument("output", help="output JSONL with judge_score and judge_verdict")
    args = ap.parse_args()
    main(args.input, args.output)
