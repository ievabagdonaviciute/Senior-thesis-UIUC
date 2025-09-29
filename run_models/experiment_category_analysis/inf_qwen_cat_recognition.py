import os, json
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# --- config ---
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

MODEL_ID        = "/home/ievab2/models/Qwen2.5-VL-7B-Instruct"   # local dir (no downloads)
FRAMES_ROOT     = "/home/ievab2/run_models/CLEVRER_dataset/validation_frames"
NUM_FRAMES      = 8
MAX_NEW_TOKENS  = 128
# -------------

import re

def _extract_assistant(text: str) -> str:
    if not text:
        return ""

    if "<|assistant|>" in text:
        return text.split("<|assistant|>", maxsplit=1)[-1].strip()

    m = re.search(r'(?:^|\n)assistant\s*\n(.*)\Z', text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    lines = text.strip().splitlines()
    # drop leading meta/role lines
    drop_prefixes = ('system', 'user', 'assistant')
    cleaned = []
    started = False
    for ln in lines[::-1]:  # scan from bottom up; keep until we hit 'assistant'
        if not started and re.fullmatch(r'\s*assistant\s*', ln, flags=re.IGNORECASE):
            started = True
            continue
        if started:
            cleaned.append(ln)
    if cleaned:
        return "\n".join(cleaned[::-1]).strip()

    return text.strip()


def _load_model():
    print(f"[qwen] loading model from '{MODEL_ID}' …", flush=True)

    if torch.cuda.is_available():
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    local_only = os.path.isdir(MODEL_ID)
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True, local_files_only=local_only
    )
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True, local_files_only=local_only
    )
    model.eval()
    print("[qwen] model loaded. cuda?", torch.cuda.is_available(), flush=True)
    return processor, model

def frames_dir_from_row(row: dict) -> Path:
    vpath = Path(row["video_path"])
    chunk = vpath.parent.name      # e.g., "video_10000-11000"
    vid   = vpath.stem             # e.g., "video_10003"
    return Path(FRAMES_ROOT) / chunk / vid

def read_fixed_8_frames(dir_path: Path):
    """Return absolute paths for exactly 000.jpg..007.jpg (or .png) in order."""
    frames = []
    missing = []
    for i in range(8):
        jpg = dir_path / f"{i:03d}.jpg"
        png = dir_path / f"{i:03d}.png"
        if jpg.exists():
            fp = jpg
        elif png.exists():
            fp = png
        else:
            missing.append(f"{i:03d}.jpg/.png")
            continue
        frames.append(str(fp.resolve()))
    if missing:
        raise FileNotFoundError(f"Missing frames in {dir_path}: {', '.join(missing)}")
    return frames, list(range(8))


def ask_qwen(processor, model, frames_dir: Path, question: str) -> str:
    
    frame_paths, _ = read_fixed_8_frames(frames_dir)
    print(f"[qwen] loaded frames: {[Path(p).name for p in frame_paths]} from {frames_dir}", flush=True)


    prefix = (
        "Task: categorize the QUESTION TYPE (not the video content). "
        "Given 8 ordered frames from a video and a text question about that video, "
        "output EXACTLY one word (lowercase) from: descriptive, explanatory, predictive, counterfactual. "
        "Do NOT answer the question itself; only classify its type. "
        "Output format: Label: <one-word-label>."
        )

    prompt = prefix + "\nQuestion: " + (question or "")

    # Build chat with 8 images in order + instruction that they're sequential frames
    messages = [{
        "role": "user",
        "content": (
            [{"type": "image", "image": p} for p in frame_paths] +
            [{"type": "text",
              "text": prompt}]
        ),
    }]

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device

    # Note: images=[frame_paths] (batch of size 1, list of 8 paths)
    inputs = processor(
        text=[chat_text],
        images=[frame_paths],
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    text = _extract_assistant(text)
    return text, prompt


def eval_task(task_path: str, out_path: str, counter_limit: Optional[int] = None):
    
    print(f"[qwen] Starting task …", flush=True)
    processor, model = _load_model()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(task_path, "r") as f_in, open(out_path, "w") as f_out:
        print("[qwen] opened task and output files", flush=True)
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
                
                frames_dir = frames_dir_from_row(row)
                if not frames_dir.exists():
                    raise FileNotFoundError(f"Missing frames dir: {frames_dir}")

                qid = row.get("question_id", row.get("qid", f"row{i}"))
                print(f"[qwen] Running {qid} …", flush=True)
                print(f"[qwen] using frames {frames_dir} …", flush=True)

                pred, prompt = ask_qwen(processor, model, frames_dir, q)

                out_record = dict(row)
                out_record["model_output"] = pred
                out_record["full_prompt"] = prompt
                out_record["category_answer"] = row.get("category")  # ground truth

                f_out.write(json.dumps(out_record) + "\n")
                f_out.flush()
                written += 1
                print(f"[qwen] wrote row {written}", flush=True)

            except Exception as e:
                print(f"[qwen][ERROR] row {i}: {e}", flush=True)

    print(f"[qwen] wrote {written} rows to {out_path}", flush=True)

if __name__ == "__main__":
    TASK_JSONL = "/home/ievab2/run_models/questions/clevrer_filtered_500.jsonl"
    OUT_JSONL  = "/home/ievab2/run_models/experiment_category_analysis/cat_recognition_results/qwen_out.jsonl"
    # set to 5 for a quick test, or None for all
    LIMIT = None
    eval_task(TASK_JSONL, OUT_JSONL, counter_limit=LIMIT)
