import os, json, math
from pathlib import Path
from typing import Optional, List, Tuple
import torch
from PIL import Image
import argparse
from transformers import AutoProcessor, AutoModelForImageTextToText

# ---- config ----

MODEL_DIR        = "/home/ievab2/models/SmolVLM2-2.2B-Instruct"  # local dir (worked in bunny test)
FRAMES_ROOT  = "/home/ievab2/run_models/experiment_frame_selection/selected_frames"
SELECTION_JSONL = "/home/ievab2/run_models/experiment_frame_selection_QWEN_RERUN/qwen_out_frame_selection_skip.jsonl"
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
    paths, missing = [], []
    for i in idxs:
        jpg = dir_path / f"{i:03d}.jpg"
        png = dir_path / f"{i:03d}.png"
        fp = jpg if jpg.exists() else (png if png.exists() else None)
        if fp is None:
            missing.append(f"{i:03d}")
            continue
        paths.append(str(fp.resolve()))
    if missing:
        raise FileNotFoundError(f"Missing frames in {dir_path}: {', '.join(missing)}")
    return None, paths, idxs  # keep tuple shape the same

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


# def ask_smolvlm(processor, model, frames_dir: Path, idxs: List[int], question: str):
#     _, frame_paths, _ = read_selected_frames(frames_dir, idxs)

#     print(f"[smolvlm] frames: {[Path(p).name for p in frame_paths]}  dir={frames_dir}", flush=True)
#     # build multi-image + text message using the chat template
#     content = [{"type": "image", "path": p} for p in frame_paths]
    
#     content.append({
#         "type": "text",
#         "text": (
#             "These 8 images are consecutive frames from one video in time order. "+ 
#             "Use the sequence to answer: " + (question or "")
#             #"Describe what you see in the video: objects, shapes, colors, movements."
#         )
#     })

#     messages = [{"role": "user", "content": content}]

#     inputs = processor.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         tokenize=True,
#         return_dict=True,
#         return_tensors="pt",
#     ).to(model.device, dtype=torch.bfloat16)

#     with torch.inference_mode():
#         generated_ids = model.generate(
#             **inputs,
#             do_sample=False,
#             max_new_tokens=MAX_NEW_TOKENS,
#         )

#     out = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     return out, frame_paths

def ask_smolvlm(processor, model, frames_dir: Path, idxs: List[int], question: str):
    _, frame_paths, _ = read_selected_frames(frames_dir, idxs)
    print(f"[smolvlm] frames: {[Path(p).name for p in frame_paths]}  dir={frames_dir}", flush=True)

    # 1) Load the 8 images (PIL)
    frame_images = [Image.open(p).convert("RGB") for p in frame_paths]

    # 2) Build chat: 8 image placeholders + your text
    messages = [{
        "role": "user",
        "content": (
            [{"type": "image"} for _ in frame_images] + 
            [{"type": "text",
              "text": "These 8 images are consecutive frames from one video in time order. "
                      f"Use the sequence to answer: {question or ''}"}]
        )
    }]

    # 3) Get text prompt, then pass BOTH text and images to the processor
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[prompt], images=frame_images, return_tensors="pt").to(model.device)

    # 4) Generate + decode
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
    out = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return out, frame_paths


def eval_task(task_path: str, out_path: str, counter_limit: Optional[int] = None, resume_cat: Optional[str] = None, resume_qid: Optional[str] = None):
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
    selection_map = load_selection_map(SELECTION_JSONL)

    out_path = Path(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    skipped_path = out_path.with_name("skipped_smolvlm.jsonl")
    written, skipped = 0,0

    mode_out  = "a" if is_resuming else "w"
    mode_skip = "a" if is_resuming else "w"

    with open(task_path, "r", encoding="utf-8") as f_in, open(out_path, mode_out, encoding="utf-8") as f_out, open(skipped_path, mode_skip, encoding="utf-8") as f_skip:
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

                qid = row.get("question_id") or row.get("qid") or f"row{i}"
                
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
                    print(f"[smolvlm][SKIP] {qid}: {reason} {details}", flush=True)
                    continue

                idxs = idxs_sorted  # use as-is

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

                raw, frame_paths = ask_smolvlm(processor, model, frames_dir, idxs, q)
                pred = _extract_assistant(raw)

                out_record = dict(row)
                out_record["model_output"] = pred
                out_record["qwen_selected_idx"] = qwen_idxs_full or []
                out_record["qwen_selected_idx_sorted"] = idxs
                out_record["frames"] = frame_paths   
                f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                f_out.flush()
                written += 1
                print(f"[smolvlm] wrote {written}", flush=True)

            except Exception as e:
                skipped += 1
                f_skip.write(json.dumps({
                    "qid": (row.get("question_id") or row.get("qid") or f"row{i}") if 'row' in locals() else f"row{i}",
                    "reason": f"exception:{type(e).__name__}",
                    "details": str(e)
                }, ensure_ascii=False) + "\n")
                f_skip.flush()
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
    OUT_JSONL  = "/home/ievab2/run_models/experiment_frame_selection_QWEN_RERUN/selected_frames_qwen_rerun_results/smolvlm_out.jsonl"

    LIMIT = None   # set to small int for a smoke test
    eval_task(TASK_JSONL, OUT_JSONL, counter_limit=LIMIT, resume_cat=args.resume_cat, resume_qid=args.resume_qid)
