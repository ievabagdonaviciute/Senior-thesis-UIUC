# # bunny exmaple

# import os
# from PIL import Image
# import torch
# from transformers import AutoProcessor, AutoModelForImageTextToText

# MODEL_PATH = "/home/ievab2/models/SmolVLM2-2.2B-Instruct"
# FRAMES_DIR = "/home/ievab2/bunnies-frames"  # 000.jpg … 007.jpg

# print("[smolvlm] loading …", flush=True)
# processor = AutoProcessor.from_pretrained(MODEL_PATH)
# model = AutoModelForImageTextToText.from_pretrained(
#     MODEL_PATH,
#     dtype=torch.bfloat16,              # preferred dtype
#     device_map="auto",
# )

# # build multi-image + text message using the chat template
# content = []
# for i in range(8):
#     content.append({"type": "image", "path": os.path.join(FRAMES_DIR, f"{i:03d}.jpg")})
# content.append({"type": "text", "text": "These are 8 consecutive frames (000→007). Describe what happens over time and name the key objects."})

# messages = [{"role": "user", "content": content}]

# inputs = processor.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt",
# ).to(model.device, dtype=torch.bfloat16)

# with torch.inference_mode():
#     generated_ids = model.generate(
#         **inputs,
#         do_sample=False,
#         max_new_tokens=160,
#     )

# out = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print("\n[smolvlm] OUTPUT:\n", out)


############# FULL

import os, json, math
from pathlib import Path
from typing import Optional, List, Tuple
import torch
from PIL import Image
import argparse
from transformers import AutoProcessor, AutoModelForImageTextToText

# ---- config ----

MODEL_DIR        = "/home/ievab2/models/SmolVLM2-2.2B-Instruct"  # local dir (worked in bunny test)
FRAMES_ROOT      = "/home/ievab2/run_models/CLEVRER_dataset/validation_frames"
MAX_NEW_TOKENS   = 128
REQUIRE_EXACT_8  = True   # enforce 000..007 exist; set False to allow even-sampling up to 8
# ----------------

def _extract_assistant(out: str) -> str:
    if "Assistant:" in out:
        answer = out.split("Assistant:", 1)[1].strip()
    else:
        answer = out.strip()
    return answer

def _load_model() -> Tuple[AutoProcessor, AutoModelForImageTextToText]:
    print(f"[smolvlm] loading from {MODEL_DIR} …", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,              # preferred dtype
        device_map="auto",
    )
    print(f"[smolvlm] device={model.device} dtype=bfloat16", flush=True)
    return processor, model
 

def frames_dir_from_row(row: dict) -> Path:
    vpath = Path(row["video_path"])
    chunk = vpath.parent.name          # e.g., "video_10000-11000"
    vid   = vpath.stem                 # e.g., "video_10003"
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


def ask_smolvlm(processor, model, frames_dir: Path, question: str) -> str:
    frame_paths, _ = read_fixed_8_frames(frames_dir)

    print(f"[smolvlm] frames: {[Path(p).name for p in frame_paths]}  dir={frames_dir}", flush=True)
    # build multi-image + text message using the chat template
    content = [{"type": "image", "path": p} for p in frame_paths]
    
    content.append({
        "type": "text",
        "text": (
            "These 8 images are consecutive frames from one video in time order (000→007). "+ 
            "Use the sequence to answer: " + (question or "")
            #"Describe what you see in the video: objects, shapes, colors, movements."
        )
    })

    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    out = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return out

def eval_task(task_path: str, out_path: str, counter_limit: Optional[int] = None, resume_cat: Optional[str] = None, resume_qid: Optional[str] = None):
    """
    Read CLEVRER-style JSONL:
      - expects 'video_path' to map to frames dir
      - expects 'prompt' or 'question' as the query
    Writes each row + model_output to out_path (JSONL).
    """

    # determine mode: fresh run (truncate) vs resume (append)
    is_resuming = bool(resume_qid or resume_cat)

    already_done: set = set()
    if is_resuming and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f_prev:
            for ln in f_prev:
                try:
                    rec = json.loads(ln)
                    qid_prev = rec.get("question_id") or rec.get("qid")
                    if qid_prev:
                        already_done.add(qid_prev)
                except Exception:
                    continue  # ignore partial/corrupted lines

    # if resuming, we haven't yet passed the last successful item
    # if fresh run, start immediately
    resume_passed = (not is_resuming)

    if is_resuming: print("[smolvlm] Resuming task …", flush=True)
    else: print("[smolvlm] Starting task …", flush=True)

    processor, model = _load_model()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0

    mode = "a" if is_resuming else "w"
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

                frames_dir = frames_dir_from_row(row)
                if not frames_dir.exists():
                    raise FileNotFoundError(f"Missing frames dir: {frames_dir}")

                qid = row.get("question_id") or row.get("qid") or f"row{i}"

                category = row.get("category") or row.get("question_type")

                # 1) De-duplicate when resuming
                if is_resuming and qid in already_done:
                    print(f"[smolvlm][skip-existing] {qid}", flush=True)
                    continue

                # 2) Resume: keep skipping until we pass the specified (cat, qid)
                if not resume_passed:
                    # If a category filter was provided, enforce it while searching
                    if resume_cat is not None and category != resume_cat:
                        continue
                    # If a qid marker was provided and we haven't reached it yet, keep skipping
                    if resume_qid is not None and qid != resume_qid:
                        continue

                    # We have reached the resume marker (by cat/qid conditions)
                    print(f"[smolvlm][resume-hit] reached category={category} qid={qid}; starting HERE", flush=True)
                    resume_passed = True


                print(f"[smolvlm] {qid}", flush=True)

                raw = ask_smolvlm(processor, model, frames_dir, q)
                pred = _extract_assistant(raw)

                out_record = dict(row)
                out_record["model_output"] = pred
                f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                f_out.flush()
                written += 1
                print(f"[smolvlm] wrote {written}", flush=True)

            except Exception as e:
                print(f"[smolvlm][ERROR] row {i}: {e}", flush=True)

    print(f"[smolvlm] Done. Wrote {written} rows to {out_path}", flush=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-cat", default=None,
                        help="e.g., descriptive / explanatory / predictive / counterfactual")
    parser.add_argument("--resume-qid", default=None,
                        help="question_id of the last successful line")
    args = parser.parse_args()

    TASK_JSONL = "/home/ievab2/run_models/questions/clevrer_filtered_500.jsonl"
    OUT_JSONL  = "/home/ievab2/run_models/results/smolvlm_out.jsonl"
    #OUT_JSONL = "/home/ievab2/run_models/results/smolvlm_resume_test.jsonl"

    LIMIT = None   # set to small int for a smoke test
    eval_task(TASK_JSONL, OUT_JSONL, counter_limit=LIMIT, resume_cat=args.resume_cat, resume_qid=args.resume_qid)
