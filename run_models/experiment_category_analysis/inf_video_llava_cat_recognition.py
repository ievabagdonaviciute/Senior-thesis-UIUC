import os, json, glob
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from PIL import Image
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

# --- config ---
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

MODEL_ID     = "/home/ievab2/models/Video-LLaVA-7B-hf"
FRAMES_ROOT  = "/home/ievab2/run_models/CLEVRER_dataset/validation_frames"
NUM_FRAMES   = 8
MAX_NEW_TOKENS = 128
# -------------


def _load_model():
    print(f"[videollava] loading model from '{MODEL_ID}' …", flush=True)

    if torch.cuda.is_available():
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    local_only = os.path.isdir(MODEL_ID)
    processor = VideoLlavaProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True, local_files_only=local_only
    )
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=dtype, device_map="auto",
        trust_remote_code=True, local_files_only=local_only
    )
    model.eval()
    print("[videollava] model loaded. cuda?", torch.cuda.is_available(), flush=True)
    return processor, model

def frames_dir_from_row(row: dict) -> Path:
    vpath = Path(row["video_path"])
    chunk = vpath.parent.name      # e.g., "video_10000-11000"
    vid   = vpath.stem             # e.g., "video_10003"
    return Path(FRAMES_ROOT) / chunk / vid

def read_fixed_8_frames(dir_path: Path):
    """Load exactly 000.jpg .. 007.jpg from dir_path, in order."""
    frames = []
    idxs = list(range(8))
    missing = []
    for i in idxs:
        jpg = dir_path / f"{i:03d}.jpg"
        png = dir_path / f"{i:03d}.png"
        if jpg.exists():
            fp = jpg
        elif png.exists():
            fp = png
        else:
            missing.append(f"{i:03d}.jpg/.png")
            continue
        frames.append(Image.open(fp).convert("RGB"))
    if missing:
        raise FileNotFoundError(f"Missing frames in {dir_path}: {', '.join(missing)}")
    return frames, idxs


def ask_video_llava(processor, model, frames_dir: Path, question: str) -> str:
    frames, _ = read_fixed_8_frames(frames_dir)

    print(f"[videollava] loaded frames: {[f'{i:03d}' for i in range(8)]} from {frames_dir}", flush=True)

    prefix = (
    "Task: categorize the QUESTION TYPE (not the video content). "
    "Given 8 ordered frames from a video and a text question about that video, "
    "output EXACTLY one word (lowercase) from: descriptive, explanatory, predictive, counterfactual. "
    "Do NOT answer the question itself; only classify its type. "
    "Output format: Label: <one-word-label>."
    )

    prompt = f"USER: <video>\n{prefix}\nQuestion: {question}\nASSISTANT:"
    device = next(model.parameters()).device
    inputs = processor(text=prompt, videos=frames, return_tensors="pt").to(device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    decoded = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    if "ASSISTANT:" in decoded:
        decoded = decoded.split("ASSISTANT:", 1)[1]
    return decoded.strip(), prompt

def eval_task(task_path: str, out_path: str, counter_limit: Optional[int] = None):

    print(f"[videollava] Starting task …", flush=True)
    processor, model = _load_model()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(task_path, "r") as f_in, open(out_path, "w") as f_out:
        print("[videollava] opened task and output files", flush=True)
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
                print(f"[videollava] Running {qid} …", flush=True)
                print(f"[videollava] using frames {frames_dir} …", flush=True)

                pred, prompt = ask_video_llava(processor, model, frames_dir, q)

                out_record = dict(row)
                out_record["model_output"] = pred
                out_record["full_prompt"] = prompt
                out_record["category_answer"] = row.get("category")  # ground truth

                f_out.write(json.dumps(out_record) + "\n")
                f_out.flush()
                written += 1
                print(f"[videollava] wrote row {written}", flush=True)

            except Exception as e:
                # Log and continue with the next sample
                print(f"[videollava][ERROR] row {i}: {e}", flush=True)

    print(f"[videollava] wrote {written} rows to {out_path}", flush=True)

if __name__ == "__main__":
    TASK_JSONL = "/home/ievab2/run_models/questions/clevrer_filtered_500.jsonl"
    OUT_JSONL  = "/home/ievab2/run_models/experiment_category_analysis/cat_recognition_results/videollava_out.jsonl"
    # set to 5 for a quick test, or None for all
    LIMIT = None
    eval_task(TASK_JSONL, OUT_JSONL, counter_limit=LIMIT)