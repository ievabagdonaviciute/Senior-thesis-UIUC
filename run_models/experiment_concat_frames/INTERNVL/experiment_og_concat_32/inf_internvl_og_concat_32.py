#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path
from typing import Optional, Set

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel, GenerationConfig

import numpy as np
import torchvision.transforms as T
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode

# --- env / config ---
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_DIR   = "/home/ievab2/models/InternVL2-8B"   # local dir for OpenGVLab/InternVL2-8B
CONCAT_ROOT = Path("/home/ievab2/run_models/concatenated_frames/concat_frames_32")
MAX_NEW_TOKENS = 128

# ---------- helpers ----------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    #resized_img = image.resize((target_width, target_height))
    resized_img = image.resize((target_width, target_height), resample=Image.BICUBIC)

    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

# def load_image(image_file, input_size=448, max_num=12):
#     image = Image.open(image_file).convert('RGB')
#     transform = build_transform(input_size=input_size)
#     images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
#     pixel_values = [transform(image) for image in images]
#     pixel_values = torch.stack(pixel_values)
#     return pixel_values

# simplified because i pass only one image
def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)

    # use your existing dynamic_preprocess to generate up to 12 tiles
    images = dynamic_preprocess(
        image,
        min_num=1,
        max_num=max_num,        # try 6, 9, or 12 depending on how dense your concat is
        image_size=input_size,
        use_thumbnail=False      # keep <= max_num
    )
    pixel_values = [transform(im) for im in images]
    return torch.stack(pixel_values)   # shape (K, 3, 448, 448)


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
    vpath = Path(row["video_path"])
    return Path(vpath.parent.name) / vpath.stem  # <chunk>/<video>

def concat_image_path(rel_chunk_vid: Path) -> Path:
    return CONCAT_ROOT / rel_chunk_vid / "concat.jpg"

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
def ask_internvl2(tokenizer, model, concat_img_path: Path, question: str) -> tuple[str, str]:
    user_text = "You see 32 consecutive frames of a video. Do not give any explanation or analysis, just answer the following question. " + (question or "")
    #user_text = "You see 32 consecutive frames of a video. Describe what you see."

    # Build the prompt with one <image> token per tile
    pixel_values = load_image(str(concat_img_path), max_num=12)   # (K, 3, 448, 448)
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

    return response, prompt


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
                q = row.get("prompt") or row.get("question")
                if not q:
                    raise ValueError("Row missing 'prompt'/'question'")
                if "video_path" not in row:
                    raise ValueError("Row missing 'video_path'")

                # resume skip
                if resume:
                    key = _row_key(row, q)
                    if key in done_keys:
                        continue

                rel = rel_chunk_vid_from_row(row)
                img_path = concat_image_path(rel)
                if not img_path.exists():
                    raise FileNotFoundError(f"Missing concatenated image: {img_path}")

                qid = row.get("question_id") or row.get("qid") or f"row{i}"
                print(f"[internvl2] {qid}  image={img_path}", flush=True)

                pred, prompt = ask_internvl2(tokenizer, model, img_path, q)

                out_record = dict(row)
                out_record["image_path"] = str(img_path)
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
    OUT_JSONL  = "/home/ievab2/run_models/experiment_concat_frames/INTERNVL/experiment_og_concat_32/internvl_out.jsonl"

    LIMIT = None

    eval_task(TASK_JSONL, OUT_JSONL, counter_limit=LIMIT, resume=args.resume)
