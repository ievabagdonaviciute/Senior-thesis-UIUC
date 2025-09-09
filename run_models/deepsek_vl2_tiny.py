# ## deepseek bunny test
# import torch
# from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
# from deepseek_vl2.utils.io import load_pil_images

# # paths
# MODEL_DIR = "/home/ievab2/models/deepseek-vl2-tiny"
# FRAMES = [f"/home/ievab2/bunnies-frames/{i:03d}.jpg" for i in range(8)]

# # load
# processor = DeepseekVLV2Processor.from_pretrained(MODEL_DIR)
# tokenizer = processor.tokenizer
# model = DeepseekVLV2ForCausalLM.from_pretrained(MODEL_DIR)
# model = model.to(torch.float16).cuda().eval()

# # prompt for the *video* (all frames)
# conversation = [
#     {
#         "role": "<|User|>",
#         "content": "<image>\nDescribe what happens in this bunny video.",
#         "images": FRAMES,
#     },
#     {"role": "<|Assistant|>", "content": ""},
# ]

# # preprocess
# pil_images = load_pil_images(conversation)
# inputs = processor(
#     conversations=conversation,
#     images=pil_images,
#     force_batchify=True,
#     system_prompt=""
# ).to(model.device)

# # --- IMPORTANT: force any floating tensors to match model dtype (fp16) ---
# def _cast_floats_to(dtype, obj):
#     # handles common fields used by DeepSeek-VL2
#     for name in ("images", "pixel_values", "image_tensors"):
#         if hasattr(obj, name):
#             val = getattr(obj, name)
#             if torch.is_tensor(val) and val.is_floating_point():
#                 setattr(obj, name, val.to(dtype))
#     # also sweep generic attributes just in case
#     for name in dir(obj):
#         if name.startswith("_"):
#             continue
#         try:
#             val = getattr(obj, name)
#         except Exception:
#             continue
#         if torch.is_tensor(val) and val.is_floating_point():
#             setattr(obj, name, val.to(dtype))
#     return obj

# inputs = _cast_floats_to(model.dtype, inputs)

# # generate
# with torch.no_grad():
#     embeds = model.prepare_inputs_embeds(**inputs)
#     outputs = model.language.generate(
#         inputs_embeds=embeds,
#         attention_mask=inputs.attention_mask,
#         pad_token_id=tokenizer.eos_token_id,
#         bos_token_id=tokenizer.bos_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#         max_new_tokens=128,
#         do_sample=False,
#         use_cache=True,
#     )

# raw = tokenizer.decode(outputs[0].detach().cpu().tolist(), skip_special_tokens=False)
# print("=== RAW OUTPUT ===")
# print(raw)

# # cleaned reply
# reply = raw
# if "<|Assistant|>:" in raw:
#     reply = raw.split("<|Assistant|>:", 1)[1].strip()
# for tok in ("<｜end▁of▁sentence｜>", "<eos>", "</s>", "<|eot_id|>"):
#     reply = reply.replace(tok, "").strip()

# print("\n=== ASSISTANT ===")
# print(reply)

############# FULL


# deepseek_clevrer_eval.py
import os, json, math
from pathlib import Path
from typing import Optional, List, Tuple

import torch
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

# ---- config ----
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_DIR        = "/home/ievab2/models/deepseek-vl2-tiny"  # local dir (worked in bunny test)
FRAMES_ROOT      = "/home/ievab2/run_models/CLEVRER_dataset/validation_frames"
MAX_NEW_TOKENS   = 128
REQUIRE_EXACT_8  = True   # enforce 000..007 exist; set False to allow even-sampling up to 8
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

def list_image_files(dir_path: Path) -> List[Path]:
    files = sorted([*dir_path.glob("*.jpg"), *dir_path.glob("*.jpeg"), *dir_path.glob("*.png")])
    return files

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

def even_sample(paths: List[Path], k: int) -> List[str]:
    if len(paths) <= k:
        return [str(p.resolve()) for p in paths]
    n = len(paths)
    idxs = [min(math.floor(i * n / k), n - 1) for i in range(k)]
    seen, picked = set(), []
    for i in idxs:
        if i not in seen:
            picked.append(paths[i])
            seen.add(i)
    i = 0
    while len(picked) < k and i < n:
        if i not in seen:
            picked.append(paths[i]); seen.add(i)
        i += 1
    return [str(p.resolve()) for p in picked]

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

def ask_deepseek(processor, model, frames_dir: Path, question: str) -> str:
    frame_paths, _ = read_fixed_8_frames(frames_dir)

    print(f"[deepseek tiny] frames: {[Path(p).name for p in frame_paths]}  dir={frames_dir}", flush=True)

    # one multi-image turn (treat frames as a video strip)
    conversation = [
        {
            "role": "<|User|>",
            "content": (
                "<image>\nThese 8 images are consecutive frames from one video in time order (000→007). "
                "Use the sequence to answer: " + (question or "")
                #"Answer in detail about what objects, colors and movements you see in the video."
            ),
            "images": frame_paths,   # IMPORTANT: a list, not a generator
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    pil_images = load_pil_images(conversation)
    inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(model.device)

    inputs = _cast_floats_to(model.dtype, inputs)

    tokenizer = processor.tokenizer
    with torch.no_grad():
        embeds = model.prepare_inputs_embeds(**inputs)
        outputs = model.language.generate(
            inputs_embeds=embeds,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )

    raw = tokenizer.decode(outputs[0].detach().cpu().tolist(), skip_special_tokens=False)
    return raw

def eval_task(task_path: str, out_path: str, counter_limit: Optional[int] = None):
    """
    Read CLEVRER-style JSONL:
      - expects 'video_path' to map to frames dir
      - expects 'prompt' or 'question' as the query
    Writes each row + model_output to out_path (JSONL).
    """
    print("[deepseek tiny] Starting task …", flush=True)
    processor, model = _load_model()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(task_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
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
                print(f"[deepseek tiny] {qid}", flush=True)

                raw = ask_deepseek(processor, model, frames_dir, q)
                pred = _extract_assistant(raw)

                out_record = dict(row)
                out_record["model_output"] = pred
                f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                f_out.flush()
                written += 1
                print(f"[deepseek tiny] wrote {written}", flush=True)

            except Exception as e:
                print(f"[deepseek tiny][ERROR] row {i}: {e}", flush=True)

    print(f"[deepseek tiny] Done. Wrote {written} rows to {out_path}", flush=True)

if __name__ == "__main__":
    TASK_JSONL = "/home/ievab2/run_models/questions/clevrer_filtered_500.jsonl"
    OUT_JSONL  = "/home/ievab2/run_models/results/deepseek_tiny_out.jsonl"
    LIMIT = None   # set to small int for a smoke test
    eval_task(TASK_JSONL, OUT_JSONL, counter_limit=LIMIT)
