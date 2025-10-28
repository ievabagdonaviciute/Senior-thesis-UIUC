# from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
# from PIL import Image
# import torch, requests
# # --- load model & processor ---
# MODEL_PATH = "/home/ievab2/models/Molmo-7B-D-0924"
# # load the processor
# processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     trust_remote_code=True,
#     torch_dtype='auto',
#     device_map="auto"
# )

# # --- open your local image ---
# image_path = "/home/ievab2/run_models/experiment_frame_selection_videollava/concat_frames/video_10000-11000/video_10000/concat.jpg"
# img = Image.open(image_path)
# if img.mode != "RGB":
#     img = img.convert("RGB")   # avoids broadcast errors

# # --- prepare the input ---
# inputs = processor.process(
#     images=[img], 
#     text="You see 32 consecutive frames of a video. Describe what you see."
#     )
# inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

# # --- generate output ---
# output = model.generate_from_batch(
#     inputs,
#     GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
#     tokenizer=processor.tokenizer
# )

# generated_tokens = output[0, inputs['input_ids'].size(1):]
# generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

# print("\nMolmo output:\n", generated_text)


# FULL

#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

# --- env / config ---
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

MODEL_PATH   = "/home/ievab2/models/Molmo-7B-D-0924"  # local dir
CONCAT_ROOT  = Path("/home/ievab2/run_models/concatenated_frames/concat_frames_32")

MAX_NEW_TOKENS = 200

# -------------- helpers --------------
def _qid(row: dict) -> str:
    qid = row.get("question_id") or row.get("qid")
    if qid is None:
        raise ValueError("Row missing 'question_id'/'qid' — cannot resume robustly")
    return str(qid)

def _load_done_keys(out_path: str) -> set[str]:
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r") as f:
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
                done.add(str(qid))
    print(f"[molmo][resume] loaded {len(done)} done qids", flush=True)
    return done

def _load_model():
    print(f"[molmo] loading model from '{MODEL_PATH}' …", flush=True)
    local_only = os.path.isdir(MODEL_PATH)

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=local_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=local_only,
    )
    model.eval()
    print("[molmo] model loaded. cuda?", torch.cuda.is_available(), flush=True)
    return processor, model

def frames_dir_from_row(row: dict) -> Path:
    vpath = Path(row["video_path"])
    chunk = vpath.parent.name   # e.g., "video_10000-11000"
    vid   = vpath.stem          # e.g., "video_10003"
    return Path(chunk) / vid

def concat_image_path(rel_chunk_vid: Path) -> Path:
    return CONCAT_ROOT / rel_chunk_vid / "concat.jpg"

def _sanitize_answer(text: str) -> str:
    if not text:
        return ""
    # Trim common stop string if present
    text = text.replace("<|endoftext|>", "").strip()
    return text

def ask_molmo(processor, model, image_path: Path, question: str) -> str:

    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    prompt = "You see 32 consecutive frames of a video. " + question

    inputs = processor.process(
        images=[img],
        text=prompt,
    )
    # move to device and add batch dimension (from huggingface)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer,
        )

    # Slice off the prompt part and decode
    prompt_len = inputs["input_ids"].size(1)
    generated_tokens = output[0, prompt_len:]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return _sanitize_answer(generated_text), prompt

# -------------- main eval loop --------------

def eval_task(task_path: str, out_path: str, counter_limit: Optional[int] = None, resume: bool = False):

    print("[molmo] Starting task …", flush=True)
    # if --resume : check what's already in the output jsonl
    done_keys = _load_done_keys(out_path) if resume else set()
    seen_this_run = set()

    processor, model = _load_model()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    mode = "a" if resume else "w"
    with open(task_path, "r") as f_in, open(out_path, mode) as f_out:
        print("[molmo] opened task and output files", flush=True)
        for i, line in enumerate(f_in):
            if not line.strip():
                continue
            if counter_limit is not None and written >= counter_limit:
                break

            try:
                row = json.loads(line)
                qid_str = _qid(row)

                q = row.get("prompt") or row.get("question")
                if not q:
                    raise ValueError("Row missing 'prompt'/'question'")
                if "video_path" not in row:
                    raise ValueError("Row missing 'video_path'")
                
                # check if already processed if --resume
                if resume:
                    if (qid_str in done_keys) or (qid_str in seen_this_run):
                        continue
                    seen_this_run.add(qid_str)

                rel = frames_dir_from_row(row)  # <chunk>/<video>
                img_path = concat_image_path(rel)
                if not img_path.exists():
                    raise FileNotFoundError(f"Missing concatenated image: {img_path}")

                qid = row.get("question_id", row.get("qid", f"row{i}"))
                print(f"[molmo] Running {qid} …", flush=True)
                print(f"[molmo] using concat image {img_path}", flush=True)

                pred, prompt = ask_molmo(processor, model, img_path, q)

                out_record = dict(row)
                out_record["image_path"] = str(img_path)
                out_record["prompt_given_to_model"] = prompt
                out_record["model_output"] = pred
                f_out.write(json.dumps(out_record) + "\n")
                f_out.flush()
                written += 1
                print(f"[molmo] wrote row {written}", flush=True)

            except Exception as e:
                print(f"[molmo][ERROR] row {i}: {e}", flush=True)

    print(f"[molmo] wrote {written} rows to {out_path}", flush=True)

# -------------- entrypoint --------------

if __name__ == "__main__":
    # Same task file and style as your Qwen run; just outputs a MOLMO results file.
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Skip entries already in OUT_JSONL and append new results")
    args = parser.parse_args()

    TASK_JSONL = "/home/ievab2/run_models/questions/clevrer_filtered_500.jsonl"
    OUT_JSONL  = "/home/ievab2/run_models/experiment_concat_frames/MOLMO_NEW/experiment_og_concat_32/molmo_out.jsonl"
    LIMIT = None  # set to an int to cap rows for a quick test
    eval_task(TASK_JSONL, OUT_JSONL, counter_limit=LIMIT, resume=args.resume)
