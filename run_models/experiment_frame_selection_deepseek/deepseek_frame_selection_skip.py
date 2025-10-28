import os, json, math
from pathlib import Path
from typing import Optional, List, Tuple

import torch
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
import re
from typing import Dict, Any


# ---- config ----
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_DIR        = "/home/ievab2/models/deepseek-vl2-tiny"  # local dir (worked in bunny test)
FRAMES_ROOT     = "/home/ievab2/run_models/experiment_frame_selection/selected_frames"
NUM_FRAMES      = 32
MAX_NEW_TOKENS   = 128
REQUIRE_EXACT_8  = True   
ARRAY_RE = re.compile(r'\[(.*?)\]', re.S)
# ----------------

def _extract_assistant(raw: str) -> str:
    reply = raw
    if "<|Assistant|>:" in raw:
        reply = raw.split("<|Assistant|>:", 1)[1].strip()
    for tok in ("<｜end▁of▁sentence｜>", "<eos>", "</s>", "<|eot_id|>"):
        reply = reply.replace(tok, "").strip()
    return reply

def _load_model() -> Tuple[DeepseekVLV2Processor, DeepseekVLV2ForCausalLM]:
    print(f"[deepseek tiny] loading from {MODEL_DIR} …", flush=True)
    processor = DeepseekVLV2Processor.from_pretrained(MODEL_DIR)
    model = DeepseekVLV2ForCausalLM.from_pretrained(MODEL_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = model.to(dtype).to(device).eval()
    print(f"[deepseek tiny] device={device} dtype={dtype}", flush=True)
    return processor, model

def frames_dir_from_row(row: dict) -> Path:
    """
    Map videollava/qwen schema to frames location:
      /.../video_validation/<chunk>/<video_id>.mp4
      -> /.../validation_frames/<chunk>/<video_id>/
    """
    vpath = Path(row["video_path"])
    chunk = vpath.parent.name          # e.g., "video_10000-11000"
    vid   = vpath.stem                 # e.g., "video_10003"
    return Path(FRAMES_ROOT) / chunk / vid


def _cast_floats_to(dtype, obj):
    # force any floating tensors inside inputs to match model dtype (fp16 on GPU)
    for name in ("images", "pixel_values", "image_tensors"):
        if hasattr(obj, name):
            val = getattr(obj, name)
            if torch.is_tensor(val) and val.is_floating_point():
                setattr(obj, name, val.to(dtype))
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            val = getattr(obj, name)
        except Exception:
            continue
        if torch.is_tensor(val) and val.is_floating_point():
            setattr(obj, name, val.to(dtype))
    return obj

def read_fixed_32_frames(dir_path: Path) -> Tuple[List[str], List[int]]:
    frames, missing = [], []
    for i in range(NUM_FRAMES):
        jpg = dir_path / f"{i:03d}.jpg"
        png = dir_path / f"{i:03d}.png"
        fp = jpg if jpg.exists() else (png if png.exists() else None)
        if fp is None:
            missing.append(f"{i:03d}.jpg/.png")
            continue
        frames.append(str(fp.resolve()))
    if missing:
        raise FileNotFoundError(f"Missing frames in {dir_path}: {', '.join(missing)}")
    return frames, list(range(NUM_FRAMES))

def _parse_idx_list(text: str) -> Tuple[bool, List[int], str]:
    """
    Best-effort: find the first [...] block, parse numbers, validate 8 unique ints in [0,31].
    If not ascending, we sort instead of skipping.
    Returns (ok, idxs, reason)
    """
    try:
        # Prefer strict JSON parse first
        candidate = text.strip()
        if candidate.startswith('[') and candidate.endswith(']'):
            arr = json.loads(candidate)
        else:
            m = ARRAY_RE.search(candidate)
            if not m:
                return False, [], "no_bracketed_array_found"
            arr = json.loads('[' + m.group(1) + ']')
        if not isinstance(arr, list):
            return False, [], "not_a_list"
        idxs = [int(x) for x in arr]
    except Exception:
        return False, [], "json_parse_failed"

    if len(idxs) != 8:
        return False, [], f"length_{len(idxs)}_not_8"
    if any(x < 0 or x > 31 for x in idxs):
        return False, [], "out_of_range"
    if len(set(idxs)) != 8:
        return False, [], "duplicates_found"

    # if not ascending, just sort instead of failing
    sorted_idxs = sorted(idxs)
    if sorted_idxs != idxs:
        return True, sorted_idxs, "sorted_from_nonascending"

    return True, idxs, "ok"



# def ask_deepseek_select_frames(processor, model, frames_dir: Path, question: str, answer: str):
#     frame_paths, _ = read_fixed_32_frames(frames_dir)
#     print(f"[deepseek tiny] frames: {[Path(p).name for p in frame_paths]}  dir={frames_dir}", flush=True)

#     # ---- single multi-image turn (images list preserves order = frame_id) ----
#     # We explicitly tell the model the mapping: image index == FRAME_ID (0..31).
#     user_text = (
#         "<image>\n"
#         "You are given 32 frames from ONE video in chronological order. "
#         "The first image is [FRAME_ID=0], the second is [FRAME_ID=1], …, the last is [FRAME_ID=31].\n\n"
#         f"Question: {question or ''}\n"
#         f'Ground-truth answer: "{(answer or "")}".\n\n'
#         "Task: Select exactly 8 UNIQUE frame IDs (integers in [0,31]) that would be most helpful for a VLM to answer the question correctly. "
#         "Prefer frames that show key state changes, interactions/collisions, or disambiguating views; "
#         "cover the timeline and avoid near-duplicate adjacent frames.\n\n"
#         "Output ONLY a JSON array of 8 integers in STRICTLY ASCENDING order, e.g., [0,3,4,5,12,14,15,30]. "
#         "The FIRST character must be '[' and the LAST character must be ']'. No explanation."
#         #"Answer in number: How many images are you seeing?"
#     )

#     conversation = [
#         {
#             "role": "<|User|>",
#             "content": user_text,
#             "images": frame_paths,   # IMPORTANT: pass all 32 frames in order
#         },
#         {"role": "<|Assistant|>", "content": ""},
#     ]

#     # ---- preprocess ----
#     pil_images = load_pil_images(conversation)
#     inputs = processor(
#         conversations=conversation,
#         images=pil_images,
#         force_batchify=True,
#         system_prompt=""
#     ).to(model.device)

    # inputs = _cast_floats_to(model.dtype, inputs)
    # tokenizer = processor.tokenizer

    # # ---- generate via multimodal path ----
    
    # with torch.no_grad():
    #     embeds = model.prepare_inputs_embeds(**inputs)
    #     outputs = model.language.generate(
    #         inputs_embeds=embeds,
    #         attention_mask=inputs.attention_mask,
    #         pad_token_id=tokenizer.eos_token_id,
    #         bos_token_id=tokenizer.bos_token_id,
    #         eos_token_id=tokenizer.eos_token_id,
    #         max_new_tokens=MAX_NEW_TOKENS,
    #         do_sample=False,
    #         use_cache=True,
    #     )

    # raw = tokenizer.decode(outputs[0].detach().cpu().tolist(), skip_special_tokens=False)
    # return raw, frame_paths


def ask_deepseek_select_frames(processor, model, frames_dir: Path, question: str, answer: str):
    frame_paths, _ = read_fixed_32_frames(frames_dir)
    print(f"[deepseek tiny] frames: {[Path(p).name for p in frame_paths]}  dir={frames_dir}", flush=True)

    # 1) Build content with 32 placeholders, then your instruction text
    placeholders = "<image_placeholder>" * len(frame_paths)
    user_text = (
        f"{placeholders}"
        "You are given 32 frames from ONE video in chronological order. "
        "The first image is [FRAME_ID=0], the second is [FRAME_ID=1], …, the last is [FRAME_ID=31].\n\n"
        f"Question: {question or ''}\n"
        f'Ground-truth answer: "{(answer or "")}".\n\n'
        "Select exactly 8 UNIQUE frame IDs from this set that, when shown to a VLM, will most help it answer the question with the ground-truth answer. "
        "Prefer frames that capture state changes, interactions/collisions, or key disambiguating views; avoid near-duplicate adjacent frames unless necessary and aim for temporal coverage. "
        "Output format: return ONLY a JSON array of 8 integers in ascending order."
        "Do not include any explanation or extra text. "
        "Any ID outside [0,31] or duplicates make the output invalid."
    )

    # 2) Conversation must use special role tokens
    conversation = [
        {
            "role": "<|User|>",
            "content": user_text,
            "images": frame_paths,  # 32 paths in the SAME order as placeholders
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # 3) Preprocess
    pil_images = load_pil_images(conversation)
    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(model.device)

    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4) Vision encode -> LM generate (note: language_model per HF docs)
    with torch.no_grad():
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
        outputs = model.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=1,
            do_sample=False,
            use_cache=True
        )

    raw = tokenizer.decode(outputs[0].detach().cpu().tolist(), skip_special_tokens=False).strip()

    # 5) Light extractor: cut after the assistant tag if present; drop known end tokens
    ans = raw
    if "<|Assistant|>" in ans:
        ans = ans.split("<|Assistant|>", 1)[1].strip()
    for tok in ("<｜end▁of▁sentence｜>", "<eos>", "</s>", "<|eot_id|>"):
        ans = ans.replace(tok, "").strip()

    return ans, frame_paths


def eval_task(task_path: str, out_path: str,
              counter_limit: Optional[int] = None,
              start_after_qid: Optional[str] = None,
              resume: bool = False):

    """
    Read CLEVRER-style JSONL:
      - expects 'video_path' to map to frames dir
      - expects 'prompt' or 'question' as the query
    Writes each row + model_output to out_path (JSONL).
    """

    done_ids = set()
    if resume and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                try:
                    _j = json.loads(_line)
                    _qid = _j.get("question_id") or _j.get("qid")
                    if _qid is not None:
                        done_ids.add(_qid)
                except Exception:
                    pass
        print(f"[deepseek tiny] resume: found {len(done_ids)} completed rows", flush=True)

    print("[deepseek tiny] Starting task …", flush=True)
    processor, model = _load_model()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0

    seen_start = (start_after_qid is None)

    mode = "a" if resume else "w"
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
                
                # --- start-after guard (if provided) --- 
                if not seen_start:
                    if qid == start_after_qid:
                        seen_start = True
                    else:
                        continue

                # --- skip completed when resuming ---
                if resume and qid in done_ids:
                    print(f"[SKIP existing] {qid}", flush=True)
                    continue

                print(f"[deepseek tiny] {qid}", flush=True)
                raw, frame_paths = ask_deepseek_select_frames(
                    processor, model, frames_dir, q, answer=row.get("ground_truth") or ""
                )
     
                pred = _extract_assistant(raw)

                ok, idxs, reason = _parse_idx_list(pred)

                out_record = dict(row)
                out_record["model_output"] = pred                   # raw text (JSON array expected)
                out_record["parse_ok"] = ok
                out_record["parse_reason"] = reason if not ok else "ok"
                out_record["frames_32"] = frame_paths               # optional: keep for debugging/trace
                f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                f_out.flush()

                written += 1
                print(f"[deepseek tiny] wrote {written}", flush=True)

            except Exception as e:
                print(f"[deepseek tiny][ERROR] row {i}: {e}", flush=True)

    print(f"[deepseek tiny] Done. Wrote {written} rows to {out_path}", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-qid", type=str, default=None,
                        help="question_id/qid to start processing from")
    parser.add_argument("--resume", action="store_true",
                        help="append to OUT_JSONL and skip qids already present")
    
    args = parser.parse_args()

    TASK_JSONL = "/home/ievab2/run_models/questions/clevrer_filtered_500.jsonl"
    OUT_JSONL  = "/home/ievab2/run_models/experiment_frame_selection_deepseek/deepseek_out_frame_selection.jsonl"
    LIMIT = None   # set to small int for a smoke test
    
    eval_task(
        TASK_JSONL,
        OUT_JSONL,
        counter_limit=LIMIT,
        start_after_qid=args.start_qid,
        resume=args.resume
    )


