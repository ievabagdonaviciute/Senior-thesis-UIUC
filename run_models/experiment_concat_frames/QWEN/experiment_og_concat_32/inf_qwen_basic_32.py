#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import Qwen2_5_VLForConditionalGeneration


# --- config ---
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


MODEL_ID        = "/home/ievab2/models/Qwen2.5-VL-7B-Instruct"   # local dir (no downloads)
CONCAT_ROOT     = Path("/home/ievab2/run_models/concatenated_frames/concat_frames_32")
MAX_NEW_TOKENS  = 128

import re

def _qid_from_row(row: dict) -> Optional[str]:
    """Return a stable question id from an input/output row."""
    return row.get("question_id") or row.get("qid")

def _load_done_ids(out_path: str) -> set[str]:
    """Scan an existing JSONL and return the set of processed question_ids."""
    done = set()
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
                print(f"[qwen][WARN] couldn't parse existing line {i}: {e}", flush=True)
    return done



def _extract_assistant(text: str) -> str:
    if not text:
        return ""
    if "<|assistant|>" in text:
        return text.split("<|assistant|>", maxsplit=1)[-1].strip()

    m = re.search(r'(?:^|\n)assistant\s*\n(.*)\Z', text, flags=re.IGNORECASE | re.DOTALL)
    
    if m:
        return m.group(1).strip()
    lines = text.strip().splitlines()
    drop_prefixes = ('system', 'user', 'assistant')
    cleaned, started = [], False

    for ln in lines[::-1]:
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
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        #sattn_implementation="flash_attention_2",
        local_files_only=True,
        device_map="auto",
    )

    model.eval()
    print("[qwen] model loaded. cuda?", torch.cuda.is_available(), flush=True)
    return processor, model

def rel_chunk_vid_from_row(row: dict) -> Path:
    vpath = Path(row["video_path"])
    return Path(vpath.parent.name) / vpath.stem

def concat_image_path(rel_chunk_vid: Path) -> Path:
    return CONCAT_ROOT / rel_chunk_vid / "concat.jpg"

def ask_qwen(processor, model, concat_img_path: Path, question: str) -> str:
    # Build chat with a single image + text
    prompt = "You see 32 consecutive frames of a video. " + (question or "")

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image", 
                "image": str(concat_img_path)
            },
            {"type": "text", "text": prompt}
        ],
    }]

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device

    inputs = processor(
        text=[chat_text],
        images=[str(concat_img_path)],
        padding=True,
        return_tensors="pt",
    ).to(device)

    # with torch.inference_mode():
    #     out_ids = model.generate(
    #         **inputs,
    #         max_new_tokens=MAX_NEW_TOKENS,
    #         do_sample=False
    #     )

    # text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
            )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    #return _extract_assistant(text), prompt
    return output_text[0], prompt


def eval_task(task_path: str, out_path: str, counter_limit: Optional[int] = None, resume: bool = False):
    print(f"[qwen] Starting task …", flush=True)
    done_ids = set()
    if resume:
        done_ids = _load_done_ids(out_path)
        print(f"[qwen] resume=True → found {len(done_ids)} completed question_ids", flush=True)

    processor, model = _load_model()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0

    out_mode = "a" if resume else "w"
    with open(task_path, "r") as f_in, open(out_path, out_mode) as f_out:
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
                if "video_path" not in row:
                    raise ValueError("Row missing 'video_path'")

                rel = rel_chunk_vid_from_row(row)
                img_path = concat_image_path(rel)
                if not img_path.exists():
                    raise FileNotFoundError(f"Missing concatenated image: {img_path}")

                qid = row.get("question_id", row.get("qid", f"row{i}"))
                if resume and str(qid) in done_ids:
                    print(f"[qwen] skipping already processed {qid}", flush=True)
                    continue

                print(f"[qwen] Running {qid} …", flush=True)
                print(f"[qwen] using concat image {img_path}", flush=True)

                pred, prompt = ask_qwen(processor, model, img_path, q)

                out_record = dict(row)
                out_record["image_path"] = str(img_path)
                out_record["prompt_given_to_model"] = prompt
                out_record["model_output"] = pred
                # ensure an ID is persisted so future --resume can skip this row
                if ("question_id" not in out_record) and ("qid" not in out_record):
                    out_record["qid"] = str(qid)

                f_out.write(json.dumps(out_record) + "\n")

                f_out.flush()
                written += 1
                print(f"[qwen] wrote row {written}", flush=True)

            except Exception as e:
                print(f"[qwen][ERROR] row {i}: {e}", flush=True)

    print(f"[qwen] wrote {written} rows to {out_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Skip entries already in OUT_JSONL and append new results")
    args = parser.parse_args()

    TASK_JSONL = "/home/ievab2/run_models/questions/clevrer_filtered_500.jsonl"
    OUT_JSONL  = "/home/ievab2/run_models/experiment_concat_frames/QWEN/experiment_og_concat_32/qwen_out.jsonl"
    LIMIT = None  # set e.g. 5 for a quick test
    eval_task(TASK_JSONL, OUT_JSONL, counter_limit=LIMIT, resume=args.resume)
