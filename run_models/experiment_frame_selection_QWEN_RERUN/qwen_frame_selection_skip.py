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
FRAMES_ROOT     = "/home/ievab2/run_models/experiment_frame_selection/selected_frames"
NUM_FRAMES      = 32
MAX_NEW_TOKENS  = 128
# -------------

import re

def _extract_assistant(text: str) -> str:
    """
    Return only the assistant's final reply from a full chat transcript.
    Handles 'assistant' lines, '<|assistant|>' tokens, and variants.
    """
    if not text:
        return ""

    # 1) Special token form (Qwen often uses this in templates)
    if "<|assistant|>" in text:
        return text.split("<|assistant|>", maxsplit=1)[-1].strip()

    # 2) Role-line form:
    #    system\n...\nuser\n...\nassistant\n<ANSWER>
    m = re.search(r'(?:^|\n)assistant\s*\n(.*)\Z', text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    # 3) Fallback: remove any leading role headers heuristically
    #    (strip starting 'system', 'user', '.', colons, etc.)
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

    # Last resort: return as-is (already trimmed)
    return text.strip()


def _load_model():
    print(f"[qwen] loading model from '{MODEL_ID}' …", flush=True)

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
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
    """
    Same mapping as videollava:
      /.../video_validation/<chunk>/<video_id>.mp4
      -> /.../validation_frames/<chunk>/<video_id>/
    """
    vpath = Path(row["video_path"])
    chunk = vpath.parent.name      # e.g., "video_10000-11000"
    vid   = vpath.stem             # e.g., "video_10003"
    return Path(FRAMES_ROOT) / chunk / vid

def read_fixed_32_frames(dir_path: Path):
    """Return absolute paths for exactly 000.jpg..007.jpg (or .png) in order."""
    frames = []
    missing = []
    for i in range(NUM_FRAMES):
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
    return frames, list(range(NUM_FRAMES))


def ask_qwen(processor, model, frames_dir: Path, question: str, answer: str) -> str:
    """
    Robust path: feed 32 ordered frames as images; instruct Qwen that they're
    consecutive frames from one video. Avoids torchvision/PyAV video pipeline.
    """
    frame_paths, _ = read_fixed_32_frames(frames_dir)
    print(f"[qwen] loaded frames: {[Path(p).name for p in frame_paths]} from {frames_dir}", flush=True)

    # Build chat with 32 images in order + instruction that they're sequential frames
    # messages = [{
    #     "role": "user",
    #     "content": (
    #         [{"type": "image", "image": p} for p in frame_paths] +
    #         [{"type": "text",
    #           "text": #"These 32 images are consecutive frames from a single video in time order (000→031)." +
    #                   #"This is a question about the video: " +
    #                   #(question or "") +
    #                   #"The correct answer to the question is: "
    #                   #(answer or "") +
    #                   #"From the 32 images, index starting at 0, pick 8 video frame indeces from the sequence, that maximize the chance of a model answering the question about the video correctly." +
    #                   #"Example: if you want to pick indeces 0, 3, 4, 5, 12, 14, 15, 30, give the answer in a form: [0, 3, 4, 5, 12, 14, 15, 30]."
    #                     "These are frames of a video."
    #                     #"First, tell me exactly how many frames you see."
    #                     "Tell me what objects (and their colors) you see in frame indexed 10 (indeces starting at 0)."
    #                   }]
    #     ),
    # }]

    content = []
    for i, p in enumerate(frame_paths):
        content.append({"type": "image", "image": p})
        content.append({"type": "text",  "text": f"[FRAME_ID={i}]"})

    content.append({"type": "text", "text": (
        "You will see 32 frames of a video, each immediately followed by a label like [FRAME_ID=i]. "
        "They are in chronological order from [FRAME_ID=0] to [FRAME_ID=31]. "
        f"Question: {(question or '')} "
        f'Ground-truth answer: "{(answer or "")}". '
        "Select exactly 8 UNIQUE frame IDs from this set that, when shown to a VLM, will most help it answer the question with the ground-truth answer. "
        "Prefer frames that capture state changes, interactions/collisions, or key disambiguating views; avoid near-duplicate adjacent frames unless necessary and aim for temporal coverage. "
        "Output format: return ONLY a JSON array of 8 integers in ascending order."
        "Do not include any explanation or extra text. "
        "Any ID outside [0,31] or duplicates make the output invalid."
    )})

    messages = [{"role":"user","content": content}]

    packed = sum(1 for c in messages[0]["content"] if isinstance(c, dict) and c.get("type") == "image")
    print("[DEBUG] packed images in messages:", packed) 

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    device = next(model.parameters()).device

    # Note: images=[frame_paths] (batch of size 1, list of 32 paths)
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
    return text



def eval_task(task_path: str, out_path: str, counter_limit: Optional[int] = None):
    """
    Read the JSONL and run up to `counter_limit` rows (all if None).
    Writes each row + 'model_output' to out_path (overwrites each run).
    Mirrors videollava logging and behavior.
    """
    print(f"[qwen] Starting task …", flush=True)
    processor, model = _load_model()
    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path, "r") as _f:
            for _line in _f:
                try:
                    _j = json.loads(_line)
                    if "question_id" in _j:
                        done_ids.add(_j["question_id"])
                except Exception:
                    pass
    print(f"[qwen] resume: found {len(done_ids)} completed rows", flush=True)
    # -----------------------------------------------------

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0


    with open(task_path, "r") as f_in, open(out_path, "a") as f_out:
        print("[qwen] opened task and output files", flush=True)
        for i, line in enumerate(f_in):
            if not line.strip():
                continue
            if counter_limit is not None and written >= counter_limit:
                break

            try:
                row = json.loads(line)
                if row.get("question_id") in done_ids:
                    print(f"[SKIP] already done: {row['question_id']}", flush=True)
                    continue
                q = row.get("prompt") or row.get("question")
                if not q:
                    raise ValueError("Row missing 'prompt'/'question'")

                frames_dir = frames_dir_from_row(row)
                if not frames_dir.exists():
                    raise FileNotFoundError(f"Missing frames dir: {frames_dir}")

                qid = row.get("question_id", row.get("qid", f"row{i}"))
                print(f"[qwen] Running {qid} …", flush=True)
                print(f"[qwen] using frames {frames_dir} …", flush=True)

                pred = ask_qwen(processor, model, frames_dir, q, answer=row.get("ground_truth") or "")

                out_record = dict(row)
                out_record["model_output"] = pred
                f_out.write(json.dumps(out_record) + "\n")
                f_out.flush()
                done_ids.add(row.get("question_id")) 
                written += 1
                print(f"[qwen] wrote row {written}", flush=True)

            except Exception as e:
                print(f"[qwen][ERROR] row {i}: {e}", flush=True)

    print(f"[qwen] wrote {written} rows to {out_path}", flush=True)

if __name__ == "__main__":
    print("Starting...")
    TASK_JSONL = "/home/ievab2/run_models/questions/clevrer_filtered_500.jsonl"
    OUT_JSONL  = "/home/ievab2/run_models/experiment_frame_selection_QWEN_RERUN/qwen_out_frame_selection_skip.jsonl"
    # set to 5 for a quick test, or None for all
    LIMIT = None
    eval_task(TASK_JSONL, OUT_JSONL, counter_limit=LIMIT)
