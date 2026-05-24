#!/usr/bin/env python3
import os, json, argparse, re
from pathlib import Path
from typing import Optional, Set, List, Dict

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# ================== ENV / CONFIG ==================
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_DIR = "/home/ievab2/models/Qwen2.5-VL-7B-Instruct"

ROOT_JSONL_OFFICIAL = Path("/shared/rsaas/ievab2/Physion_full_readout_training")
OUT_JSONL_OFFICIAL  = Path("/home/ievab2/run_models/FULL_PHYSION_FINETUNING_QWEN/base_model/base_results.jsonl")

ROOT_JSONL_MY = Path("/shared/rsaas/ievab2/my_own_physion_preprocessed")
OUT_DIR_MY    = Path("/home/ievab2/run_models/FULL_PHYSION_FINETUNING_QWEN/base_model/testing_on_my_own_physion")
OUT_JSONL_MY  = OUT_DIR_MY / "base_results.jsonl"
ACC_JSONL_MY  = OUT_DIR_MY / "base_accuracy.jsonl"

CATEGORIES = [
    "Collide",
    "Contain",
    "Dominoes",
    "Drape",
    "Drop",
    "Link",
    "Roll",
    "Support",
]

NUM_FRAMES     = 8
MAX_NEW_TOKENS = 128

_NEG_RE = re.compile(r"\b(?:no|false|not|won't|will not|doesn't|does not|don't|didn't|did not|can't|cannot)\b", re.I)
_POS_RE = re.compile(r"\b(?:yes|true|yep|yeah|y)\b", re.I)

# ================== HELPERS ==================
def normalize_yesno(text: str) -> str:
    t = (text or "").strip().lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    if _NEG_RE.search(t):
        return "no"
    if _POS_RE.search(t):
        return "yes"
    return "unknown"

def _validate_frame_paths(paths: List[str]):
    bad = [p for p in paths if not p or not os.path.isabs(p) or not os.path.exists(p)]
    if bad:
        raise FileNotFoundError(f"Bad/missing frame paths: {bad[:3]}{'...' if len(bad)>3 else ''}")

def _row_key(row: dict) -> str:
    frames = row.get("frames") or row.get("frame_paths") or []
    q = row.get("question") or row.get("prompt") or ""
    first = frames[0] if frames else ""
    last  = frames[-1] if frames else ""
    return f"f::{first}|{last}||q::{q}"

def _load_done_keys(out_path: Path) -> Set[str]:
    done = set()
    if not out_path.exists():
        return done
    with out_path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            frames = rec.get("frame_paths") or rec.get("frames") or []
            q = rec.get("question") or rec.get("prompt") or rec.get("prompt_given_to_model") or ""
            first = frames[0] if frames else ""
            last  = frames[-1] if frames else ""
            done.add(f"f::{first}|{last}||q::{q}")
    return done

# ================== QWEN-SPECIFIC: EXTRACT ASSISTANT ==================
def _extract_assistant(text: str) -> str:
    if not text:
        return ""
    if "<|assistant|>" in text:
        return text.split("<|assistant|>", maxsplit=1)[-1].strip()
    m = re.search(r'(?:^|\n)assistant\s*\n(.*)\Z', text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()

# ================== MODEL LOADING ==================
def _load_model():
    print(f"[qwen] loading model from {MODEL_DIR} …", flush=True)

    if torch.cuda.is_available():
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    local_only = os.path.isdir(MODEL_DIR)
    processor = AutoProcessor.from_pretrained(
        MODEL_DIR, trust_remote_code=True, local_files_only=local_only
    )
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_DIR,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=local_only,
    )
    model.eval()
    print("[qwen] model ready. cuda?", torch.cuda.is_available(), flush=True)
    return processor, model

# ================== INFERENCE (KEEP YOUR LOGIC) ==================
def ask_qwen(processor, model, frame_paths: List[str], question: str) -> str:
    """
    Keep your logic: feed 8 ordered frames as images + text.
    """
    if len(frame_paths) != NUM_FRAMES:
        raise ValueError(f"Expected {NUM_FRAMES} frames, got {len(frame_paths)}")

    # Build chat with 8 images in order + instruction they are sequential frames
    messages = [{
        "role": "user",
        "content": (
            [{"type": "image", "image": p} for p in frame_paths] +
            [{"type": "text",
              "text": "These 8 images are consecutive frames from a single video in time order (000→007). "
                      "Do not explain; just answer the question concisely. "
                      + (question or "")}]
        ),
    }]

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device

    inputs = processor(
        text=[chat_text],
        images=[frame_paths],   # batch size 1, list-of-8 paths
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    return _extract_assistant(text)

# ================== MAIN EVAL LOOP (INTERNVL-STYLE) ==================
def eval_all_categories(
    root_jsonl: Path,
    out_path: Path,
    resume: bool = False,
    limit: Optional[int] = None,
    write_accuracy: bool = False,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processor, model = _load_model()

    done_keys = _load_done_keys(out_path) if resume else set()
    mode = "a" if resume else "w"

    written = 0
    total = 0
    correct = 0
    per_cat: Dict[str, Dict[str, int]] = {c: {"total": 0, "correct": 0} for c in CATEGORIES}

    with out_path.open(mode, encoding="utf-8") as f_out:
        for cat in CATEGORIES:
            # Match your InternVL path logic
            if write_accuracy:
                # MY OWN physion: lowercase directories + lowercase filenames
                cat_dir = root_jsonl / cat.lower()
                cat_file = cat_dir / f"{cat.lower()}_pred.jsonl"
            else:
                # OFFICIAL physion: capitalized directories
                cat_dir = root_jsonl / cat
                cat_file = cat_dir / f"{cat.lower()}_pred.jsonl"

            if not cat_file.exists():
                print(f"[WARN] Missing JSONL for category {cat}: {cat_file}")
                continue

            print(f"[qwen] Evaluating category={cat} from {cat_file}", flush=True)

            with cat_file.open("r", encoding="utf-8") as f_in:
                for i, line in enumerate(f_in):
                    if not line.strip():
                        continue
                    if limit is not None and written >= limit:
                        print("[qwen] Reached global limit; stopping.")
                        return

                    try:
                        row = json.loads(line)

                        frames = row.get("frames") or row.get("frame_paths")
                        q = row.get("question")
                        if not isinstance(frames, list) or len(frames) != NUM_FRAMES:
                            raise ValueError(f"Row must have 'frames' or 'frame_paths' with exactly {NUM_FRAMES} image paths")
                        if not q:
                            raise ValueError("Row missing 'question'")

                        _validate_frame_paths(frames)

                        if resume:
                            key = _row_key(row)
                            if key in done_keys:
                                continue

                        qid = row.get("id") or row.get("qid") or f"{cat}_row{i}"
                        print(f"[qwen] {cat} qid={qid} first={frames[0]} last={frames[-1]}", flush=True)

                        pred = ask_qwen(processor, model, frames, q)

                        out_record = {"category": cat}
                        out_record.update(row)

                        out_record["frame_paths"] = list(map(str, frames))
                        out_record["prompt_given_to_model"] = (
                            "8 images (000→007). Do not explain; just answer concisely. " + (q or "")
                        )
                        out_record["model_output_raw"] = pred
                        out_record["model_output_norm"] = normalize_yesno(pred)
                        out_record["model_dir_used"] = MODEL_DIR

                        # Accuracy tracking (same approach as InternVL)
                        gt = (row.get("answer") or "").strip().lower()
                        pr = out_record["model_output_norm"]
                        if gt in ("yes", "no") and pr in ("yes", "no"):
                            total += 1
                            per_cat[cat]["total"] += 1
                            if gt == pr:
                                correct += 1
                                per_cat[cat]["correct"] += 1

                        f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                        f_out.flush()
                        written += 1
                        print(f"[qwen] wrote {written}", flush=True)

                    except Exception as e:
                        print(f"[qwen][ERROR] {cat} row {i}: {e}", flush=True)

    # Write accuracy JSONL if requested (MY physion)
    if write_accuracy:
        acc_path = out_path.parent / "base_accuracy.jsonl"
        acc_path.parent.mkdir(parents=True, exist_ok=True)

        with acc_path.open("w", encoding="utf-8") as fa:
            for c in CATEGORIES:
                t = per_cat[c]["total"]
                k = per_cat[c]["correct"]
                if t == 0:
                    continue
                fa.write(json.dumps({
                    "category": c,
                    "accuracy": k / t,
                    "correct": k,
                    "total": t,
                }, ensure_ascii=False) + "\n")

            if total > 0:
                fa.write(json.dumps({
                    "category": "ALL",
                    "accuracy": correct / total,
                    "correct": correct,
                    "total": total,
                }, ensure_ascii=False) + "\n")

        print(f"[qwen] Wrote base accuracy to {acc_path}", flush=True)

    print(f"[qwen] Done. Wrote {written} rows to {out_path}", flush=True)

# ================== ENTRY ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="If set, skip already-processed rows and append new ones.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional global max rows to run (across all categories).")
    parser.add_argument("--my_own_physion", action="store_true",
                        help="If set, evaluate on my own Physion dataset and write outputs to testing_on_my_own_physion/")
    args = parser.parse_args()

    if args.my_own_physion:
        root_jsonl = ROOT_JSONL_MY
        out_path = OUT_JSONL_MY
        print(f"[qwen] Using MY Physion root: {root_jsonl}")
        print(f"[qwen] Output JSONL: {out_path}")
    else:
        root_jsonl = ROOT_JSONL_OFFICIAL
        out_path = OUT_JSONL_OFFICIAL
        print(f"[qwen] Using OFFICIAL Physion root: {root_jsonl}")
        print(f"[qwen] Output JSONL: {out_path}")

    eval_all_categories(
        root_jsonl=root_jsonl,
        out_path=out_path,
        resume=args.resume,
        limit=args.limit,
        write_accuracy=args.my_own_physion
    )
