#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path
from typing import Optional

import torch
from deepseek_vl2.models import DeepseekVLV2Processor
from deepseek_vl2.utils.io import load_pil_images
from transformers import AutoModelForCausalLM

# --- env / config ---
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_DIR        = "/home/ievab2/models/deepseek-vl2-tiny"  # local dir for HF model
CONCAT_ROOT  = Path("/home/ievab2/run_models/concatenated_frames/concat_frames_8")
MAX_NEW_TOKENS = 128

def _load_model():
    print(f"[deepseek-vl2] loading from {MODEL_DIR} …", flush=True)
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(
        MODEL_DIR, trust_remote_code=True, local_files_only=True
    )
    tokenizer = vl_chat_processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


    vl_gpt = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, trust_remote_code=True, local_files_only=True, device_map="auto"
    )
    # prefer BF16 if available on your GPUs; otherwise fallback to FP16/FP32 automatically
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        device = next(vl_gpt.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vl_gpt = vl_gpt.to(dtype).eval()
    print(f"[deepseek-vl2] device={device} dtype={dtype}", flush=True)
    return vl_chat_processor, tokenizer, vl_gpt

def rel_chunk_vid_from_row(row: dict) -> Path:
    vpath = Path(row["video_path"])
    return Path(vpath.parent.name) / vpath.stem  # <chunk>/<video>

def concat_image_path(rel_chunk_vid: Path) -> Path:
    return CONCAT_ROOT / rel_chunk_vid / "concat.jpg"

def ask_deepseek_hf(vl_chat_processor, tokenizer, vl_gpt, concat_img_path: Path, question: str) -> str:
    # HF-format conversation: single image + user text
    prompt = "You see 8 consecutive frames of a video. " + (question or "")

    conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\n" + prompt,
            "images": [str(concat_img_path)],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    pil_images = load_pil_images(conversation)

    device = next(vl_gpt.parameters()).device

    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(device)


    with torch.inference_mode():
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        outputs = vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )

    answer = tokenizer.decode(outputs[0].detach().cpu().tolist(), skip_special_tokens=True).strip()
    return answer, prompt

def eval_task(task_path: str, out_path: str, counter_limit: Optional[int] = None, resume: bool = False):
    print("[deepseek-vl2] Starting task …", flush=True)
    vl_chat_processor, tokenizer, vl_gpt = _load_model()

    # (optional) resume support like your other scripts
    done_keys = set()
    if resume and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for ln in f:
                if not ln.strip(): continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                qid = rec.get("question_id") or rec.get("qid")
                if qid is not None:
                    done_keys.add(f"id::{qid}")
                else:
                    vp = rec.get("video_path") or ""
                    pq = rec.get("prompt") or rec.get("question") or ""
                    done_keys.add(f"path::{vp}||q::{pq}")
        print(f"[deepseek-vl2] resume: loaded {len(done_keys)} completed keys", flush=True)

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

                # resume skip check
                if resume:
                    key = (f"id::{row.get('question_id') or row.get('qid')}"
                           if (row.get('question_id') or row.get('qid')) is not None
                           else f"path::{row.get('video_path') or ''}||q::{q}")
                    if key in done_keys:
                        continue

                rel = rel_chunk_vid_from_row(row)
                img_path = concat_image_path(rel)
                if not img_path.exists():
                    raise FileNotFoundError(f"Missing concatenated image: {img_path}")

                qid = row.get("question_id") or row.get("qid") or f"row{i}"
                print(f"[deepseek-vl2] {qid}  image={img_path}", flush=True)

                pred, prompt = ask_deepseek_hf(vl_chat_processor, tokenizer, vl_gpt, img_path, q)

                out_record = dict(row)

                out_record["image_path"] = str(img_path)
                out_record["prompt_given_to_model"] = prompt
                out_record["model_output"] = pred
                f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                f_out.flush()
                written += 1
                print(f"[deepseek-vl2] wrote {written}", flush=True)

            except Exception as e:
                print(f"[deepseek-vl2][ERROR] row {i}: {e}", flush=True)

    print(f"[deepseek-vl2] Done. Wrote {written} rows to {out_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Skip entries already in OUT_JSONL and append new results")
    args = parser.parse_args()

    TASK_JSONL = "/home/ievab2/run_models/questions/clevrer_filtered_500.jsonl"
    OUT_JSONL  = "/home/ievab2/run_models/experiment_concat_frames/DEEPSEEK/experiment_og_concat_8/deepseek_tiny_out.jsonl"
    LIMIT = None

    eval_task(TASK_JSONL, OUT_JSONL, counter_limit=LIMIT, resume=args.resume)
