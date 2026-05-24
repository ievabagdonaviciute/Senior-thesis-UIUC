#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path
from typing import Optional, Set, List

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration

# --- env / config ---
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_DIR    = "/home/ievab2/models/Qwen2.5-VL-7B-Instruct"  # adjust if needed
FRAMES_ROOT  = Path("/home/ievab2/run_models/CLEVRER_dataset/validation_frames")
OUT_DIR      = Path("/home/ievab2/run_models/full_8_frames_qwen_vs_internvl/QWEN")
OUT_JSONL    = OUT_DIR / "qwen_out.jsonl"

NUM_FRAMES     = 8
MAX_NEW_TOKENS = 128

# ---------- helpers ----------
def _row_key(row: dict, q: str) -> str:
    qid = row.get("question_id") or row.get("qid")
    if qid is not None:
        return f"id::{qid}"
    vp = row.get("video_path") or ""
    return f"path::{vp}||q::{q or ''}"

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
            qid = rec.get("question_id") or rec.get("qid")
            if qid is not None:
                done.add(f"id::{qid}")
            else:
                vp = rec.get("video_path") or ""
                pq = rec.get("prompt") or rec.get("question") or rec.get("prompt_given_to_model") or ""
                done.add(f"path::{vp}||q::{pq}")
    return done

def rel_chunk_vid_from_row(row: dict) -> Path:
    # e.g., .../video_10000-11000/video_10000.mp4  ->  video_10000-11000/video_10000
    vpath = Path(row["video_path"])
    return Path(vpath.parent.name) / vpath.stem

def frames_dir_path(rel_chunk_vid: Path) -> Path:
    return FRAMES_ROOT / rel_chunk_vid

def list_expected_frame_paths(frames_dir: Path, num_frames: int = NUM_FRAMES) -> List[Path]:
    return [frames_dir / f"{i:03d}.jpg" for i in range(num_frames)]

# ---------- model ----------
def _load_model():
    print(f"[qwen2-vl] loading from {MODEL_DIR} …", flush=True)
    processor = AutoProcessor.from_pretrained(
        MODEL_DIR, trust_remote_code=True, local_files_only=True
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
        device_map="auto",
    ).eval()
    device = next(model.parameters()).device
    print(f"[qwen2-vl] device={device} dtype={next(model.parameters()).dtype}", flush=True)
    return processor, model

# ---------- ask (messages + processor path) ----------
def ask_qwen_messages(processor, model, frames_dir: Path, question: str):
    """
    Build Qwen2-VL 'messages' with 8 images (file:// URIs) + a text query.
    Uses processor.apply_chat_template + processor.process_vision_info + model.generate.
    Returns (prediction, text_prompt, used_frame_paths).
    """
    # 1) Verify exactly 8 frames exist
    frame_paths = list_expected_frame_paths(frames_dir, NUM_FRAMES)
    missing = [p for p in frame_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frame(s): {', '.join(str(p) for p in missing)}")
    uris = [f"file://{p}" for p in frame_paths]

    # 2) Same instruction as InternVL version
    user_text = (
        "You see 8 consecutive frames of a video in temporal order. "
        "Do not explain; just answer the question concisely. "
    ) + (question or "")

    # 3) Messages = 8 images + text (the exact format you referenced)
    messages = [{"role": "user", "content": []}]
    for uri in uris:
        messages[0]["content"].append({"type": "image", "image": uri})
    messages[0]["content"].append({"type": "text", "text": user_text})

    # 4) Template + vision preprocessing
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    pil_images = [Image.open(frames_dir / f"{i:03d}.jpg").convert("RGB") for i in range(8)]
    inputs = processor(text=[text], images=pil_images, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 5) Generate
    with torch.inference_mode():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
        )

    # 6) Trim prompt tokens and decode
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], gen_ids)]
    pred = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return pred, text, [str(p) for p in frame_paths]

# ---------- main loop ----------
def eval_task(task_path: str, out_path: Path, counter_limit: Optional[int] = None, resume: bool = False):
    print("[qwen2-vl] Starting task …", flush=True)
    processor, model = _load_model()

    done_keys = _load_done_keys(out_path) if resume else set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"
    written = 0

    with open(task_path, "r", encoding="utf-8") as f_in, open(out_path, mode, encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            if not line.strip():
                continue
            if counter_limit is not None and written >= counter_limit:
                break
            try:
                row = json.loads(line)
                q = row.get("prompt") or row.get("question")
                if not q:
                    raise ValueError("Row missing 'prompt'/'question'")
                if "video_path" not in row:
                    raise ValueError("Row missing 'video_path'")

                if resume:
                    key = _row_key(row, q)
                    if key in done_keys:
                        continue

                rel = rel_chunk_vid_from_row(row)
                fdir = frames_dir_path(rel)
                if not fdir.exists():
                    raise FileNotFoundError(f"Missing frames folder: {fdir}")

                qid = row.get("question_id") or row.get("qid") or f"row{i}"
                print(f"[qwen2-vl] {qid}  frames_dir={fdir}", flush=True)

                pred, prompt, used_paths = ask_qwen_messages(processor, model, fdir, q)

                out_record = dict(row)
                out_record["frames_dir"] = str(fdir)
                out_record["frame_paths"] = used_paths
                out_record["prompt_given_to_model"] = prompt
                out_record["model_output"] = pred
                f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                f_out.flush()
                written += 1
                print(f"[qwen2-vl] wrote {written}", flush=True)

            except Exception as e:
                print(f"[qwen2-vl][ERROR] row {i}: {e}", flush=True)

    print(f"[qwen2-vl] Done. Wrote {written} rows to {out_path}", flush=True)

# ---------- entry ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Skip entries already in OUT_JSONL and append new results")
    args = parser.parse_args()
    LIMIT = None

    TASK_JSONL = "/home/ievab2/run_models/questions/clevrer_filtered_500.jsonl"
    eval_task(TASK_JSONL, OUT_JSONL, counter_limit=LIMIT, resume=args.resume)
