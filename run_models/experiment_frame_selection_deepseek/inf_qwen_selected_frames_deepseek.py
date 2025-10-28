import os, json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# --- config ---
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

MODEL_ID        = "/home/ievab2/models/Qwen2.5-VL-7B-Instruct"   # local dir (no downloads)
FRAMES_ROOT     = "/home/ievab2/run_models/experiment_frame_selection/selected_frames"
SELECTION_JSONL = "/home/ievab2/run_models/experiment_frame_selection_deepseek/deepseek_out_frame_selection.jsonl"

NUM_FRAMES      = 8
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
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True, local_files_only=local_only
    )
    model.eval()
    print("[qwen] model loaded. cuda?", torch.cuda.is_available(), flush=True)
    return processor, model

def frames_dir_from_row(row: dict) -> Path:
    vpath = Path(row["video_path"])
    chunk = vpath.parent.name      # e.g., "video_10000-11000"
    vid   = vpath.stem             # e.g., "video_10003"
    return Path(FRAMES_ROOT) / chunk / vid

def load_selection_map(path: str) -> dict:
    m = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            qid = r.get("question_id")
            mo  = r.get("model_output")
            if isinstance(mo, str):
                try:
                    idxs = json.loads(mo)  # e.g., "[1,10,15,18,20,22,24,28]"
                except json.JSONDecodeError:
                    continue
            else:
                idxs = mo
            if qid and idxs:
                m[qid] = [int(x) for x in idxs]
    return m

def read_selected_frames(dir_path: Path, idxs: List[int]):
    frames, paths, missing = [], [], []
    for i in idxs:
        jpg = dir_path / f"{i:03d}.jpg"
        png = dir_path / f"{i:03d}.png"
        if jpg.exists():
            fp = jpg
        elif png.exists():
            fp = png
        else:
            missing.append(f"{i:03d}")
            continue
        frames.append(Image.open(fp).convert("RGB"))
        paths.append(str(fp))   # <--- keep the file path too
    if missing:
        raise FileNotFoundError(f"Missing frames in {dir_path}: {', '.join(missing)}")
    return frames, paths, idxs


def _bucket(i: int) -> int:
    if   0 <= i <= 7:   return 0
    if   8 <= i <= 15:  return 1
    if  16 <= i <= 23:  return 2
    if  24 <= i <= 31:  return 3
    return -1

def validate_and_sort_qwen_indices(raw_idxs: List[int]) -> Tuple[bool, List[int], str, dict]:
    """
    Returns (ok, sorted_idxs_or_empty, reason, details)
    Enforces:
     - exactly 8 indices
     - all in [0,31]
     - no duplicates
     - exactly 2 per interval bucket (I0..I3)
     - ascending order (we return sorted if ok)
    """
    details = {}
    if not isinstance(raw_idxs, list):
        return False, [], "not_a_list", details
    try:
        idxs = [int(x) for x in raw_idxs]
    except Exception:
        return False, [], "non_integer_values", details

    details["len"] = len(idxs)
    if len(idxs) != 8:
        return False, [], "length_not_8", details

    if any((x < 0 or x > 31) for x in idxs):
        details["out_of_range"] = [x for x in idxs if x < 0 or x > 31]
        return False, [], "out_of_range", details

    if len(set(idxs)) != len(idxs):
        details["duplicates"] = sorted([x for x in idxs if idxs.count(x) > 1])
        return False, [], "duplicates_found", details

    # # bucket counts
    # counts = {0:0,1:0,2:0,3:0}
    # for x in idxs:
    #     counts[_bucket(x)] += 1
    # details["bucket_counts"] = counts

    # # Exactly 2 per bucket (I0..I3)
    # if any(counts[b] != 2 for b in (0,1,2,3)):
    #     return False, [], "bucket_rule_violation(expect_exactly_2_each)", details

    return True, sorted(idxs), "ok", details


def ask_qwen(processor, model, frames_dir: Path, idxs: List[int], question: str):
    """
    Robust path: feed 8 ordered frames as images; instruct Qwen that they're
    consecutive frames from one video. Avoids torchvision/PyAV video pipeline.
    """
    frames, frame_paths, _ = read_selected_frames(frames_dir, idxs)
    print(f"[qwen] loaded frames: {[Path(p).name for p in frame_paths]} from {frames_dir}", flush=True)

    # Build chat with 8 images in order + instruction that they're sequential frames
    messages = [{
        "role": "user",
        "content": (
            [{"type": "image", "image": p} for p in frame_paths] +
            [{"type": "text",
              "text": "These 8 images are frames from a single video in time order. "
                      "Use the whole sequence to answer: " + (question or "")}]
        ),
    }]

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device

    # Note: images=[frame_paths] (batch of size 1, list of 8 paths)
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
    return text, frame_paths



def eval_task(task_path: str, out_path: str, counter_limit: Optional[int] = None):
    print(f"[qwen] Starting task …", flush=True)
    processor, model = _load_model()
    selection_map = load_selection_map(SELECTION_JSONL)

    out_path = Path(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    skipped_path = out_path.with_name("skipped_qwen.jsonl")
    written, skipped = 0,0

    with open(task_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out, open(skipped_path, "w", encoding="utf-8") as f_skip:
        print("[qwen] opened task and output files", flush=True)
        for i, line in enumerate(f_in):
            if not line.strip():
                continue
            if counter_limit is not None and (written + skipped) >= counter_limit:
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
                print(f"[qwen] Running {qid} …", flush=True)
                print(f"[qwen] using frames {frames_dir} …", flush=True)

                qwen_idxs_full = selection_map.get(qid)
                if not qwen_idxs_full:
                    raise ValueError(f"No Qwen-selected indices for {qid}")
                
                ok, idxs_sorted, reason, details = validate_and_sort_qwen_indices(qwen_idxs_full)
                if not ok:
                    skipped += 1
                    f_skip.write(json.dumps({
                        "qid": qid,
                        "reason": reason,
                        "details": details,
                        "raw_indices": qwen_idxs_full
                    }, ensure_ascii=False) + "\n")
                    f_skip.flush()
                    print(f"[qwen][SKIP] {qid}: {reason} {details}", flush=True)
                    continue
                
                pred, frame_paths = ask_qwen(processor, model, frames_dir, idxs_sorted, q)

                out_record = dict(row)
                out_record["model_output"] = pred
                out_record["qwen_selected_idx"] = qwen_idxs_full
                out_record["qwen_selected_idx_sorted"] = idxs_sorted
                out_record["frames"] = frame_paths
                f_out.write(json.dumps(out_record) + "\n")
                f_out.flush()
                written += 1
                print(f"[qwen] wrote row {written}", flush=True)
            
            except Exception as e:
                skipped += 1 
                f_skip.write(json.dumps({
                    "qid": (row.get("question_id") or row.get("qid") or f"row{i}") if 'row' in locals() else f"row{i}",
                    "reason": f"exception:{type(e).__name__}",
                    "details": str(e)
                }, ensure_ascii=False) + "\n")
                f_skip.flush()
                print(f"[qwen][ERROR] row {i}: {e}", flush=True)

    print(f"[qwen] wrote {written} rows to {out_path}", flush=True)

if __name__ == "__main__":
    TASK_JSONL = "/home/ievab2/run_models/questions/clevrer_filtered_500.jsonl"
    OUT_JSONL  = "/home/ievab2/run_models/experiment_frame_selection_deepseek/selected_frames_deepseek_results/qwen_out.jsonl"
    # set to 5 for a quick test, or None for all
    LIMIT = None
    eval_task(TASK_JSONL, OUT_JSONL, counter_limit=LIMIT)
