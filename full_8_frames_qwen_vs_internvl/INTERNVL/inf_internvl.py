#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path
from typing import Optional, Set, List

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# --- env / config ---
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_DIR    = "/home/ievab2/models/InternVL2-8B"  # local dir for OpenGVLab/InternVL2-8B
FRAMES_ROOT  = Path("/home/ievab2/run_models/CLEVRER_dataset/validation_frames")
NUM_FRAMES   = 8
MAX_NEW_TOKENS = 128
INPUT_SIZE   = 448  # InternVL default tile size

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ---------- helpers ----------
def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def _row_key(row: dict, q: str) -> str:
    qid = row.get("question_id") or row.get("qid")
    if qid is not None:
        return f"id::{qid}"
    vp = row.get("video_path") or ""
    return f"path::{vp}||q::{q or ''}"

def _load_done_keys(out_path: str) -> Set[str]:
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r", encoding="utf-8") as f:
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
    """
    Given row['video_path'] like ".../video_10000-11000/video_10000.mp4",
    return "video_10000-11000/video_10000"
    """
    vpath = Path(row["video_path"])
    return Path(vpath.parent.name) / vpath.stem

def frames_dir_path(rel_chunk_vid: Path) -> Path:
    """
    Map to frames directory under FRAMES_ROOT, e.g.
    FRAMES_ROOT / "video_10000-11000/video_10000"
    """
    return FRAMES_ROOT / rel_chunk_vid

def list_expected_frame_paths(frames_dir: Path, num_frames: int = NUM_FRAMES) -> List[Path]:
    return [frames_dir / f"{i:03d}.jpg" for i in range(num_frames)]

def load_frames_tensor(frames_dir: Path, input_size: int = INPUT_SIZE, num_frames: int = NUM_FRAMES) -> torch.Tensor:
    """
    Load exactly num_frames frames (000.jpg .. 007.jpg), resize/normalize, and stack.
    Returns tensor of shape (num_frames, 3, H, W).
    """
    transform = build_transform(input_size)
    frame_paths = list_expected_frame_paths(frames_dir, num_frames)
    missing = [p for p in frame_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frame(s): {', '.join(str(p) for p in missing)}")

    tensors = []
    for p in frame_paths:
        img = Image.open(p)
        tensors.append(transform(img))
    return torch.stack(tensors, dim=0)  # (K, 3, 448, 448)

# ---------- model ----------
def _load_model():
    print(f"[internvl2] loading from {MODEL_DIR} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
        device_map="auto",
        attn_implementation="eager",
    ).eval()
    device = next(model.parameters()).device
    print(f"[internvl2] device={device} dtype={next(model.parameters()).dtype}", flush=True)
    return tokenizer, model

# ---------- ask ----------
def ask_internvl2(tokenizer, model, frames_dir: Path, question: str) -> tuple[str, str, List[str]]:
    """
    Feed 8 separate frames as K images to InternVL.
    We emit one <image> token per frame, in temporal order 000..007.
    """
    user_text = (
        "You see 8 consecutive frames of a video in temporal order. "
        "Do not explain; just answer the question concisely. "
    ) + (question or "")

    pixel_values = load_frames_tensor(frames_dir, input_size=INPUT_SIZE, num_frames=NUM_FRAMES)  # (K,3,448,448)
    k = pixel_values.shape[0]
    prompt = ("<image>\n" * k) + user_text

    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype
    if device.type == "cpu":
        dtype = torch.float32

    pixel_values = pixel_values.to(device=device, dtype=dtype)

    generation_config = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "temperature": 0.0,
    }

    with torch.inference_mode():
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)

    # return also the concrete frame paths for provenance
    used_paths = [str(p) for p in list_expected_frame_paths(frames_dir, NUM_FRAMES)]
    return response, prompt, used_paths

# ---------- main loop ----------
def eval_task(task_path: str, out_path: str, counter_limit: Optional[int] = None, resume: bool = False):
    print("[internvl2] Starting task …", flush=True)
    tokenizer, model = _load_model()

    done_keys = _load_done_keys(out_path) if resume else set()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
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
                q = row.get("prompt")
                if not q:
                    raise ValueError("Row missing 'prompt'")
                if "video_path" not in row:
                    raise ValueError("Row missing 'video_path'")

                # resume skip
                if resume:
                    key = _row_key(row, q)
                    if key in done_keys:
                        continue

                rel = rel_chunk_vid_from_row(row)                 # e.g., video_10000-11000/video_10000
                fdir = frames_dir_path(rel)                       # …/validation_frames/video_10000-11000/video_10000
                if not fdir.exists():
                    raise FileNotFoundError(f"Missing frames folder: {fdir}")

                qid = row.get("question_id") or row.get("qid") or f"row{i}"
                print(f"[internvl2] {qid}  frames_dir={fdir}", flush=True)

                pred, prompt, used_paths = ask_internvl2(tokenizer, model, fdir, q)

                out_record = dict(row)
                out_record["frames_dir"] = str(fdir)
                out_record["frame_paths"] = used_paths
                out_record["prompt_given_to_model"] = prompt
                out_record["model_output"] = pred
                f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                f_out.flush()
                written += 1
                print(f"[internvl2] wrote {written}", flush=True)

            except Exception as e:
                print(f"[internvl2][ERROR] row {i}: {e}", flush=True)

    print(f"[internvl2] Done. Wrote {written} rows to {out_path}", flush=True)

# ---------- entry ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Skip entries already in OUT_JSONL and append new results")
    args = parser.parse_args()

    TASK_JSONL = "/home/ievab2/run_models/questions/clevrer_filtered_500.jsonl"
    OUT_JSONL  = "/home/ievab2/run_models/full_8_frames_qwen_vs_internvl/INTERNVL/internvl_out.jsonl"

    LIMIT = None

    eval_task(TASK_JSONL, OUT_JSONL, counter_limit=LIMIT, resume=args.resume)
