#!/usr/bin/env python3
import os, json, argparse, re
from pathlib import Path
from typing import Optional, Set, List

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# ================== ENV / CONFIG ==================
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_DIR = "/home/ievab2/models/Qwen2.5-VL-7B-Instruct"

# ---------- AUX DATASETS ----------
AUX_CONTACT_JSONL  = Path("/shared/rsaas/ievab2/PHYSION_MULTIVIEW_4PAIRS/contact_dataset.jsonl")
AUX_GEOMETRY_JSONL = Path("/shared/rsaas/ievab2/PHYSION_MULTIVIEW_4PAIRS/geometry_dataset.jsonl")
AUX_TIME_JSONL     = Path("/shared/rsaas/ievab2/PHYSION_MULTIVIEW_4PAIRS/time_dataset.jsonl")

# ---------- OUTPUT ROOT ----------
OUT_ROOT = Path("/home/ievab2/run_models/PHYSION_MULTIVIEW4PAIRS_AUXILIARY_TRAIN/text_on_multiviewGTC/qwen/results/base")

MAX_NEW_TOKENS = 128

_NEG_RE = re.compile(r"\b(?:no|false|not|won't|will not|doesn't|does not|don't|didn't|did not|can't|cannot)\b", re.I)
_POS_RE = re.compile(r"\b(?:yes|true|yep|yeah|y)\b", re.I)

# ================== HELPERS ==================
def normalize_yesno(text: str) -> str:
    t = (text or "").strip().lower()
    if t.startswith("yes"): return "yes"
    if t.startswith("no"):  return "no"
    if _NEG_RE.search(t): return "no"
    if _POS_RE.search(t): return "yes"
    return "unknown"

def normalize_time12(text: str) -> str:
    t = (text or "").strip().lower()
    for ch in t:
        if ch in ("1", "2"):
            return ch
    return "unknown"

def _validate_frame_paths(paths: List[str]):
    bad = [p for p in paths if not p or not os.path.isabs(p) or not os.path.exists(p)]
    if bad:
        raise FileNotFoundError(f"Bad/missing frame paths: {bad[:3]}{'...' if len(bad)>3 else ''}")

def _row_key(row: dict) -> str:
    frames = row.get("frames") or row.get("frame_paths") or []
    q = row.get("question") or ""
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
            q = rec.get("question") or rec.get("prompt_given_to_model") or ""
            first = frames[0] if frames else ""
            last  = frames[-1] if frames else ""
            done.add(f"f::{first}|{last}||q::{q}")
    return done

# ================== QWEN-SPECIFIC ==================
def _extract_assistant(text: str) -> str:
    if not text:
        return ""
    if "<|assistant|>" in text:
        return text.split("<|assistant|>", maxsplit=1)[-1].strip()
    return text.strip()

# ================== MODEL LOADING ==================
def _load_model():
    print(f"[qwen] loading model from {MODEL_DIR} …", flush=True)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(
        MODEL_DIR, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_DIR,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    print("[qwen] model ready.")
    return processor, model

# ================== INFERENCE ==================
def ask_qwen(processor, model, frame_paths: List[str], question: str, dataset_kind: str) -> str:
    # YOUR chosen prompt logic
    if dataset_kind == "time":
        user_text = (
            "You are shown two images that are frames from the same video. "
            "Do not explain; answer with 1 or 2 only. "
        ) + (question or "")
    elif dataset_kind == "contact":
        user_text = (
            "You are shown an image. "
            "Do not explain; just answer the question concisely. "
        ) + (question or "")
    elif dataset_kind == "geometry":
        user_text = (
            "You are shown two images. "
            "Do not explain; just answer the question concisely. "
        ) + (question or "")

    # Build Qwen message format
    messages = [{
        "role": "user",
        "content": (
            [{"type": "image", "image": p} for p in frame_paths] +
            [{"type": "text", "text": user_text}]
        ),
    }]

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device

    inputs = processor(
        text=[chat_text],
        images=[frame_paths],   # variable-length supported
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

# ================== EVAL LOOP ==================
def eval_dataset(in_jsonl: Path, out_path: Path, dataset_kind: str,
                 resume: bool = False, limit: Optional[int] = None):

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    processor, model = _load_model()

    done_keys = _load_done_keys(out_path) if resume else set()
    mode = "a" if resume else "w"
    written = 0

    with in_jsonl.open("r", encoding="utf-8") as f_in, \
         out_path.open(mode, encoding="utf-8") as f_out:

        for i, line in enumerate(f_in):
            if not line.strip():
                continue
            if limit is not None and written >= limit:
                break

            row = json.loads(line)
            frames = row.get("frames") or row.get("frame_paths")
            q = row.get("question")

            if not isinstance(frames, list) or len(frames) == 0:
                raise ValueError("Row must have >=1 image")
            if not q:
                raise ValueError("Row missing question")

            _validate_frame_paths(frames)

            if resume and _row_key(row) in done_keys:
                continue

            pred = ask_qwen(processor, model, frames, q, dataset_kind)

            if dataset_kind == "time":
                pr = normalize_time12(pred)
            else:
                pr = normalize_yesno(pred)

            out_record = dict(row)
            out_record["frame_paths"] = list(map(str, frames))
            out_record["prompt_given_to_model"] = user_text if False else "See code"
            out_record["model_output_raw"] = pred
            out_record["model_output_norm"] = pr
            out_record["model_dir_used"] = MODEL_DIR

            f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            f_out.flush()
            written += 1

    print(f"[qwen] Done. Wrote {written} rows to {out_path}")

# ================== ENTRY ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["geometry", "time", "contact"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.dataset == "geometry":
        in_jsonl = AUX_GEOMETRY_JSONL
        out_jsonl = OUT_ROOT / "base_results_geometry.jsonl"
    elif args.dataset == "time":
        in_jsonl = AUX_TIME_JSONL
        out_jsonl = OUT_ROOT / "base_results_time.jsonl"
    else:
        in_jsonl = AUX_CONTACT_JSONL
        out_jsonl = OUT_ROOT / "base_results_contact.jsonl"

    print(f"[qwen] dataset = {args.dataset}")
    print(f"[qwen] input   = {in_jsonl}")
    print(f"[qwen] output  = {out_jsonl}")

    eval_dataset(
        in_jsonl=in_jsonl,
        out_path=out_jsonl,
        dataset_kind=args.dataset,
        resume=args.resume,
        limit=args.limit
    )

# HOW TOR RUN: 
# python /home/ievab2/run_models/PHYSION_MULTIVIEW4PAIRS_AUXILIARY_TRAIN/text_on_multiviewGTC/qwen/eval_base_qwen_aux.py --dataset geometry