#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek-VL2-Tiny multi-frame runner (structured like your Qwen script):
- _load_model()
- frames_dir_from_row()
- read_fixed_8_frames()
- ask_deepseek()
- eval_task()
- LIMIT macro in __main__
"""
import os, json
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor
from deepseek_vl2.utils.io import load_pil_images
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from PIL import Image

# --- config ---
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

MODEL_ID        = "/home/ievab2/models/deepseek-vl2-tiny"   # local dir (no downloads)
FRAMES_ROOT     = "/home/ievab2/run_models/CLEVRER_dataset/validation_frames"
NUM_FRAMES      = 8
MAX_NEW_TOKENS  = 128
TEMPERATURE     = 0.0
# -------------

def _pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

def _load_model():
    print(f"[deepseek] loading model from '{MODEL_ID}' …", flush=True)
    processor = DeepseekVLV2Processor.from_pretrained(MODEL_ID)
    model = DeepseekVLV2ForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    dtype = _pick_dtype()
    if torch.cuda.is_available():
        model = model.to(dtype).cuda().eval()
    else:
        model = model.to(torch.float32).eval()
    tokenizer = processor.tokenizer
    print("[deepseek] model loaded. cuda?", torch.cuda.is_available(), flush=True)
    return processor, model, tokenizer


def frames_dir_from_row(row: dict) -> Path:
    """
    Same mapping as your Qwen helper:
      /.../video_validation/<chunk>/<video_id>.mp4
      -> /.../validation_frames/<chunk>/<video_id>/
    """
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

def _build_conversation(question: str, frame_paths: List[str]) -> List[Dict[str, Any]]:
    # Match Qwen phrasing while using DeepSeek’s chat format.
    placeholders = " ".join(["<image_placeholder>"] * len(frame_paths)) if len(frame_paths) > 1 else "<image>"
    # content = (
    #     f"{placeholders} These 8 images are consecutive frames from a single video "
    #     f"in time order (000→007). Use the whole sequence to answer: {question or ''}"
    # )
    
    content = f"{placeholders} These 8 images are consecutive frames from a single video in time order (000→007). Describe what you see in the video, specifically, objects, shapes, colors, movement."
    
    return [
        {"role": "<|User|>", "content": content, "images": frame_paths},
        {"role": "<|Assistant|>", "content": ""},
    ]

def ask_deepseek(processor, model, tokenizer, frames_dir: Path, question: str) -> str:
    """
    DeepSeek path: build conversation with multi-image placeholders,
    use processor -> prepare_inputs_embeds -> language_model.generate.
    """
    frame_paths, _ = read_fixed_8_frames(frames_dir)
    print(f"[deepseek] loaded frames: {[Path(p).name for p in frame_paths]} from {frames_dir}", flush=True)

    conversation = _build_conversation(question, frame_paths)
    #pil_images = load_pil_images(conversation)
    pil_images = [Image.open(p).convert("RGB") for p in frame_paths]


    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(next(model.parameters()).device)

####### Debug: confirm vision tensors exist
    debug_keys = [k for k in dir(prepare_inputs) if not k.startswith("_")]
    print("[deepseek][debug] prepare_inputs attrs:", debug_keys, flush=True)
    for k in ("input_ids", "attention_mask", "pixel_values", "image_pixel_values", "vision_x"):
        if hasattr(prepare_inputs, k):
            v = getattr(prepare_inputs, k)
            try:
                print(f"[deepseek][debug] {k} shape:", tuple(v.shape), flush=True)
            except Exception:
                print(f"[deepseek][debug] {k} present (non-tensor)", flush=True)
    if not any(hasattr(prepare_inputs, k) for k in ("pixel_values", "image_pixel_values", "vision_x")):
        raise RuntimeError("No vision tensor found in prepared inputs — images are NOT getting through.")
#######
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.inference_mode():
        outputs = model.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=pad_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=(TEMPERATURE > 0),
            temperature=max(1e-3, TEMPERATURE),
            use_cache=True,
        )

    return tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

def eval_task(task_path: str, out_path: str, counter_limit: Optional[int] = None):
    """
    Read the JSONL and run up to `counter_limit` rows (all if None).
    Writes each row + 'model_output' to out_path (OVERWRITES each run),
    mirroring your Qwen script’s behavior and logging.
    """
    print(f"[deepseek] Starting task …", flush=True)
    processor, model, tokenizer = _load_model()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(task_path, "r") as f_in, open(out_path, "w") as f_out:
        print("[deepseek] opened task and output files", flush=True)
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
                print(f"[deepseek] Running {qid} …", flush=True)
                print(f"[deepseek] using frames {frames_dir} …", flush=True)

                pred = ask_deepseek(processor, model, tokenizer, frames_dir, q)

                out_record = dict(row)
                out_record["model_output"] = pred
                f_out.write(json.dumps(out_record) + "\n")
                f_out.flush()
                written += 1
                print(f"[deepseek] wrote row {written}", flush=True)

            except Exception as e:
                print(f"[deepseek][ERROR] row {i}: {e}", flush=True)

    print(f"[deepseek] wrote {written} rows to {out_path}", flush=True)

if __name__ == "__main__":
    TASK_JSONL = "/home/ievab2/run_models/questions/clevrer_filtered_500.jsonl"
    OUT_JSONL  = "/home/ievab2/run_models/results/deepseek_vl2_tiny_out.jsonl"
    # set to 5 for a quick test, or None for all
    LIMIT = 2
    eval_task(TASK_JSONL, OUT_JSONL, counter_limit=LIMIT)
