#!/usr/bin/env python3
import os, json, argparse, re
from pathlib import Path
from typing import Optional, Set

import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

# --- env / config ---
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

MODEL_ID        = "/home/ievab2/models/llava-v1.6-vicuna-13b-hf"  # local dir mirror of llava-hf/llava-v1.6-vicuna-13b-hf
CONCAT_ROOT     = Path("/home/ievab2/run_models/concatenated_frames/concat_frames_32")
MAX_NEW_TOKENS  = 128

# ---------- helpers ----------
def _qid_from_row(row: dict) -> Optional[str]:
    return row.get("question_id") or row.get("qid")

def _load_done_ids(out_path: str) -> Set[str]:
    done: Set[str] = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                qid = _qid_from_row(rec)
                if qid is not None:
                    done.add(str(qid))
            except Exception as e:
                print(f"[llava][WARN] couldn't parse existing line {i}: {e}", flush=True)
    return done

def _load_model():
    print(f"[llava] loading model from '{MODEL_ID}' …", flush=True)

    if torch.cuda.is_available():
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    local_only = os.path.isdir(MODEL_ID)

    processor = LlavaNextProcessor.from_pretrained(
        MODEL_ID,
        local_files_only=local_only,
        trust_remote_code=True,
    )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype if dtype != torch.float32 else "auto",
        local_files_only=local_only,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("[llava] model loaded. cuda?", torch.cuda.is_available(), flush=True)
    return processor, model

def rel_chunk_vid_from_row(row: dict) -> Path:
    vpath = Path(row["video_path"])
    return Path(vpath.parent.name) / vpath.stem

def concat_image_path(rel_chunk_vid: Path) -> Path:
    return CONCAT_ROOT / rel_chunk_vid / "concat.jpg"

def ask_llava(processor, model, concat_img_path: Path, question: str) -> tuple[str, str]:
    # Keep your original prompt style
    prompt = "You see 32 consecutive frames of a video. " + (question or "")

    # LLaVA-1.6 chat format: one user message with an image + text
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "url": str(concat_img_path)},
            {"type": "text", "text": prompt},
        ],
    }]
    device = next(model.parameters()).device

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    # Trim the prompt part so we only keep the newly generated tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_texts = processor.tokenizer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_texts[0].strip(), prompt

def eval_task(task_path: str, out_path: str, counter_limit: Optional[int] = None, resume: bool = False):
    print(f"[llava] Starting task …", flush=True)
    done_ids: Set[str] = set()
    if resume:
        done_ids = _load_done_ids(out_path)
        print(f"[llava] resume=True → found {len(done_ids)} completed question_ids", flush=True)

    processor, model = _load_model()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    written = 0
    out_mode = "a" if resume else "w"

    with open(task_path, "r") as f_in, open(out_path, out_mode) as f_out:
        print("[llava] opened task and output files", flush=True)
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

                rel = rel_chunk_vid_from_row(row)
                img_path = concat_image_path(rel)
                if not img_path.exists():
                    raise FileNotFoundError(f"Missing concatenated image: {img_path}")

                qid = row.get("question_id", row.get("qid", f"row{i}"))
                if resume and str(qid) in done_ids:
                    print(f"[llava] skipping already processed {qid}", flush=True)
                    continue

                print(f"[llava] Running {qid} …", flush=True)
                print(f"[llava] using concat image {img_path}", flush=True)

                pred, prompt_used = ask_llava(processor, model, img_path, q)

                out_record = dict(row)
                out_record["image_path"] = str(img_path)
                out_record["prompt_given_to_model"] = prompt_used
                out_record["model_output"] = pred
                if ("question_id" not in out_record) and ("qid" not in out_record):
                    out_record["qid"] = str(qid)

                f_out.write(json.dumps(out_record) + "\n")
                f_out.flush()
                written += 1
                print(f"[llava] wrote row {written}", flush=True)

            except Exception as e:
                print(f"[llava][ERROR] row {i}: {e}", flush=True)

    print(f"[llava] wrote {written} rows to {out_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Skip entries already in OUT_JSONL and append new results")
    args = parser.parse_args()

    TASK_JSONL = "/home/ievab2/run_models/questions/clevrer_filtered_500.jsonl"
    OUT_JSONL  = "/home/ievab2/run_models/experiment_concat_frames/VIDEOLLAVA_VICUNA/experiment_og_concat_32/llava_vicuna_out.jsonl"
    LIMIT = None
    eval_task(TASK_JSONL, OUT_JSONL, counter_limit=LIMIT, resume=args.resume)
