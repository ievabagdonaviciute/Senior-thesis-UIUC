# #!/usr/bin/env python3
# import os, json, argparse, re
# from pathlib import Path
# from typing import Optional, Set, List, Dict, Tuple

# import torch
# from transformers import AutoProcessor, AutoModelForVision2Seq
# from PIL import Image

# # ================== ENV / CONFIG ==================
# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# MODEL_DIR = "/home/ievab2/models/Qwen2.5-VL-7B-Instruct"

# TASK_JSONL = Path("/shared/rsaas/ievab2/PHYSION_MULTIVIEW/dataset.jsonl")
# OUT_JSONL  = Path("/home/ievab2/run_models/PHYSION_MULTIVIEW_TEST/qwen/results/base_results.jsonl")

# NUM_IMAGES     = 3
# MAX_NEW_TOKENS = 64

# # ================== HELPERS ==================
# def _validate_image_paths(paths: List[str]):
#     bad = [p for p in paths if not p or not os.path.isabs(p) or not os.path.exists(p)]
#     if bad:
#         raise FileNotFoundError(f"Bad/missing image paths: {bad[:3]}{'...' if len(bad)>3 else ''}")

# def normalize_123(text: str) -> str:
#     """
#     Return "1"/"2"/"3" if we can find it, else "unknown".
#     Accepts outputs like "2", "Answer: 2", "image 3", etc.
#     """
#     t = (text or "").strip()
#     if not t:
#         return "unknown"

#     first = t.lstrip()[:1]
#     if first in ("1", "2", "3"):
#         return first

#     m = re.search(r"\b([123])\b", t)
#     if m:
#         return m.group(1)

#     return "unknown"

# def _row_key(row: dict) -> str:
#     imgs = row.get("images") or []
#     prompt = row.get("prompt") or ""
#     a = imgs[0] if imgs else ""
#     c = imgs[-1] if imgs else ""
#     return f"i::{a}|{c}||p::{prompt}"

# def _load_done_keys(out_path: Path) -> Set[str]:
#     done = set()
#     if not out_path.exists():
#         return done
#     with out_path.open("r", encoding="utf-8") as f:
#         for ln in f:
#             ln = ln.strip()
#             if not ln:
#                 continue
#             try:
#                 rec = json.loads(ln)
#             except Exception:
#                 continue
#             imgs = rec.get("images") or rec.get("image_paths") or []
#             prompt = rec.get("prompt") or rec.get("prompt_given_to_model") or ""
#             a = imgs[0] if imgs else ""
#             c = imgs[-1] if imgs else ""
#             done.add(f"i::{a}|{c}||p::{prompt}")
#     return done

# def _extract_assistant(text: str) -> str:
#     if not text:
#         return ""
#     # Keep only content after last assistant tag
#     if "<|assistant|>" in text:
#         text = text.split("<|assistant|>")[-1]
#     # Take the last non-empty line
#     lines = [l.strip() for l in text.splitlines() if l.strip()]
#     return lines[-1] if lines else text.strip()

# def _open_rgb(p: str) -> Image.Image:
#     with Image.open(p) as im:
#         im = im.convert("RGB") if im.mode != "RGB" else im
#         return im.copy()

# # ================== MODEL LOADING ==================
# def _load_model():
#     print(f"[qwen-base] loading model from {MODEL_DIR} …", flush=True)

#     if torch.cuda.is_available():
#         dtype = torch.float16
#     elif torch.cuda.is_bf16_supported():
#         dtype = torch.bfloat16
#     else:
#         dtype = torch.float32

#     local_only = os.path.isdir(MODEL_DIR)

#     processor = AutoProcessor.from_pretrained(
#         MODEL_DIR, trust_remote_code=True, local_files_only=local_only
#     )
#     model = AutoModelForVision2Seq.from_pretrained(
#         MODEL_DIR,
#         torch_dtype=dtype,
#         device_map="auto",
#         trust_remote_code=True,
#         local_files_only=local_only,
#     )
#     model.eval()
#     print("[qwen-base] model ready. cuda?", torch.cuda.is_available(), flush=True)
#     return processor, model

# # ================== INFERENCE ==================
# def ask_qwen(processor, model, image_paths: List[str], question: str) -> Tuple[str, str]:
#     """
#     Multiview task: 3 images + prompt.
#     Model should answer with 1/2/3 (1-based, in the order images are provided).
#     """
#     if len(image_paths) != NUM_IMAGES:
#         raise ValueError(f"Expected {NUM_IMAGES} images, got {len(image_paths)}")

#     imgs = [_open_rgb(p) for p in image_paths]

#     # make sure the model is forced into digits-only behavior
#     prompt_text = (
#         (question or "").strip() +
#         "\nDo not explain. Answer with 1, 2, or 3 only."
#     ).strip()

#     messages = [{
#         "role": "user",
#         "content": (
#             [{"type": "image", "image": im} for im in imgs] +
#             [{"type": "text", "text": prompt_text}]
#         ),
#     }]

#     chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     device = next(model.parameters()).device

#     inputs = processor(
#         text=chat_text,
#         images=imgs,
#         return_tensors="pt",
#     ).to(device)

#     with torch.inference_mode():
#         out_ids = model.generate(
#             **inputs,
#             max_new_tokens=MAX_NEW_TOKENS,
#             do_sample=False
#         )

#     decoded = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
#     pred_raw = _extract_assistant(decoded)
#     return pred_raw, chat_text

# # ================== EVAL LOOP ==================
# def eval_multiview(
#     task_path: Path,
#     out_path: Path,
#     resume: bool = False,
#     limit: Optional[int] = None,
# ):
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     processor, model = _load_model()

#     done_keys = _load_done_keys(out_path) if resume else set()
#     mode = "a" if resume else "w"

#     written = 0
#     total = 0
#     correct = 0

#     with task_path.open("r", encoding="utf-8") as f_in, out_path.open(mode, encoding="utf-8") as f_out:
#         for i, line in enumerate(f_in):
#             if not line.strip():
#                 continue
#             if limit is not None and written >= limit:
#                 break

#             try:
#                 row = json.loads(line)

#                 imgs = row.get("images")
#                 prompt = row.get("prompt")
#                 gt = (row.get("answer") or "").strip()

#                 if not isinstance(imgs, list) or len(imgs) != NUM_IMAGES:
#                     raise ValueError("Row must have 'images' with exactly 3 absolute image paths")
#                 if not isinstance(prompt, str) or not prompt.strip():
#                     raise ValueError("Row missing 'prompt'")
#                 if gt not in ("1", "2", "3"):
#                     raise ValueError("Row missing/invalid 'answer' (must be '1','2','3')")

#                 _validate_image_paths(imgs)

#                 if resume:
#                     key = _row_key(row)
#                     if key in done_keys:
#                         continue

#                 qid = row.get("id") or row.get("qid") or f"row{i}"
#                 print(f"[qwen-base] {qid} img1={imgs[0]} img3={imgs[2]}", flush=True)

#                 pred_raw, chat_text = ask_qwen(processor, model, imgs, prompt)
#                 pred_norm = normalize_123(pred_raw)

#                 is_correct = (pred_norm == gt)
#                 total += 1
#                 correct += int(is_correct)

#                 out_record = {
#                     "qid": qid,
#                     "prompt": prompt,
#                     "images": list(map(str, imgs)),
#                     "answer": gt,
#                     "model_output_raw": pred_raw,
#                     "model_output_norm": pred_norm,
#                     "correct": bool(is_correct),
#                     "prompt_given_to_model": chat_text,
#                     "model_dir_used": MODEL_DIR,
#                 }
#                 if "meta" in row:
#                     out_record["meta"] = row["meta"]

#                 f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
#                 f_out.flush()

#                 if resume:
#                     done_keys.add(_row_key(row))

#                 written += 1
#                 print(f"[qwen-base] wrote {written}  acc={correct}/{total}={correct/total:.3f}", flush=True)

#             except Exception as e:
#                 print(f"[qwen-base][ERROR] row {i}: {e}", flush=True)

#     print(f"[qwen-base] Done. Wrote {written} rows to {out_path}", flush=True)
#     if total > 0:
#         print(f"[qwen-base] Final accuracy: {correct}/{total} = {correct/total:.4f}", flush=True)

# # ================== ENTRY ==================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--resume", action="store_true",
#                         help="If set, skip already-processed rows and append new ones.")
#     parser.add_argument("--limit", type=int, default=None,
#                         help="Optional max rows to run.")
#     args = parser.parse_args()

#     if not TASK_JSONL.exists():
#         raise SystemExit(f"[qwen-base] TASK_JSONL not found: {TASK_JSONL}")

#     print(f"[qwen-base] Using dataset: {TASK_JSONL}")
#     print(f"[qwen-base] Writing outputs to: {OUT_JSONL}")

#     eval_multiview(
#         task_path=TASK_JSONL,
#         out_path=OUT_JSONL,
#         resume=args.resume,
#         limit=args.limit,
#     )

#!/usr/bin/env python3
import os, json, argparse, re
from pathlib import Path
from typing import Optional, Set, List, Tuple

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

# ================== ENV / CONFIG ==================
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_DIR = "/home/ievab2/models/Qwen2.5-VL-7B-Instruct"

TASK_JSONL_TEST1 = Path("/shared/rsaas/ievab2/PHYSION_MULTIVIEW/dataset.jsonl")
TASK_JSONL_TEST2 = Path("/shared/rsaas/ievab2/PHYSION_MULTIVIEW/dataset2.jsonl")

OUT_JSONL_TEST1 = Path("/home/ievab2/run_models/PHYSION_MULTIVIEW_TEST/qwen/results/test1/base_results.jsonl")
OUT_JSONL_TEST2 = Path("/home/ievab2/run_models/PHYSION_MULTIVIEW_TEST/qwen/results/test2/base_results.jsonl")

MAX_NEW_TOKENS = 64

# ================== HELPERS ==================
def _validate_image_paths(paths: List[str]):
    bad = [p for p in paths if not p or not os.path.isabs(p) or not os.path.exists(p)]
    if bad:
        raise FileNotFoundError(f"Bad/missing image paths: {bad[:3]}{'...' if len(bad)>3 else ''}")

def normalize_123(text: str) -> str:
    """
    Return "1"/"2"/"3" if we can find it, else "unknown".
    """
    t = (text or "").strip()
    if not t:
        return "unknown"
    first = t.lstrip()[:1]
    if first in ("1", "2", "3"):
        return first
    m = re.search(r"\b([123])\b", t)
    if m:
        return m.group(1)
    return "unknown"

def normalize_yesno(text: str) -> str:
    """
    Return "yes"/"no" if we can find it, else "unknown".
    """
    t = (text or "").strip().lower()
    if not t:
        return "unknown"
    first = t.split()[0]
    if first in ("yes", "y", "true"):
        return "yes"
    if first in ("no", "n", "false"):
        return "no"
    if re.search(r"\b(yes|true)\b", t):
        return "yes"
    if re.search(r"\b(no|false)\b", t):
        return "no"
    return "unknown"

def _row_key(row: dict) -> str:
    imgs = row.get("images") or []
    prompt = row.get("prompt") or ""
    a = imgs[0] if imgs else ""
    c = imgs[-1] if imgs else ""
    return f"i::{a}|{c}||p::{prompt}"

def _load_done_keys(out_path: Path) -> Set[str]:
    done = set()
    if not out_path.exists():
        return done
    with out_path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            imgs = rec.get("images") or rec.get("image_paths") or []
            prompt = rec.get("prompt") or rec.get("prompt_given_to_model") or ""
            a = imgs[0] if imgs else ""
            c = imgs[-1] if imgs else ""
            done.add(f"i::{a}|{c}||p::{prompt}")
    return done

def _extract_assistant(text: str) -> str:
    if not text:
        return ""
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()

def _open_rgb(p: str) -> Image.Image:
    with Image.open(p) as im:
        im = im.convert("RGB") if im.mode != "RGB" else im
        return im.copy()

# ================== MODEL LOADING ==================
def _load_model():
    print(f"[qwen-base] loading model from {MODEL_DIR} …", flush=True)

    if torch.cuda.is_available():
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    local_only = os.path.isdir(MODEL_DIR)

    processor = AutoProcessor.from_pretrained(
        MODEL_DIR, trust_remote_code=True, local_files_only=local_only
    )
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_DIR,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=local_only,
    )
    model.eval()
    print("[qwen-base] model ready. cuda?", torch.cuda.is_available(), flush=True)
    return processor, model

# ================== INFERENCE ==================
def ask_qwen(processor, model, image_paths: List[str], question: str, test: str) -> Tuple[str, str]:
    """
    - test1: 3 images -> answer 1/2/3
    - test2: 2 images -> answer yes/no
    """
    imgs = [_open_rgb(p) for p in image_paths]

    if test == "test1":
        forced = "\nDo not explain. Answer with 1, 2, or 3 only."
    else:
        forced = "\nDo not explain. Answer with yes or no only."

    prompt_text = ((question or "").strip() + forced).strip()

    messages = [{
        "role": "user",
        "content": (
            [{"type": "image", "image": im} for im in imgs] +
            [{"type": "text", "text": prompt_text}]
        ),
    }]

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device

    inputs = processor(
        text=chat_text,
        images=imgs,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    decoded = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    pred_raw = _extract_assistant(decoded)
    return pred_raw, chat_text

# ================== EVAL LOOP ==================
def eval_multiview(
    task_path: Path,
    out_path: Path,
    test: str,
    resume: bool = False,
    limit: Optional[int] = None,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processor, model = _load_model()

    done_keys = _load_done_keys(out_path) if resume else set()
    mode = "a" if resume else "w"

    written = 0
    total = 0
    correct = 0

    num_images = 3 if test == "test1" else 2

    with task_path.open("r", encoding="utf-8") as f_in, out_path.open(mode, encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            if not line.strip():
                continue
            if limit is not None and written >= limit:
                break

            try:
                row = json.loads(line)

                imgs = row.get("images")
                prompt = row.get("prompt")
                gt = (row.get("answer") or "").strip().lower()

                if not isinstance(imgs, list) or len(imgs) != num_images:
                    raise ValueError(f"Row must have 'images' with exactly {num_images} absolute image paths")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("Row missing 'prompt'")

                if test == "test1":
                    if gt not in ("1", "2", "3"):
                        raise ValueError("Row missing/invalid 'answer' (must be '1','2','3')")
                else:
                    if gt not in ("yes", "no"):
                        raise ValueError("Row missing/invalid 'answer' (must be 'yes' or 'no')")

                _validate_image_paths(imgs)

                if resume:
                    key = _row_key(row)
                    if key in done_keys:
                        continue

                qid = row.get("id") or row.get("qid") or f"row{i}"
                if test == "test1":
                    print(f"[qwen-base] {qid} img1={imgs[0]} img3={imgs[2]}", flush=True)
                else:
                    print(f"[qwen-base] {qid} img1={imgs[0]} img2={imgs[1]}", flush=True)

                pred_raw, chat_text = ask_qwen(processor, model, imgs, prompt, test=test)
                pred_norm = normalize_123(pred_raw) if test == "test1" else normalize_yesno(pred_raw)

                is_correct = (pred_norm == gt)
                total += 1
                correct += int(is_correct)

                out_record = {
                    "qid": qid,
                    "prompt": prompt,
                    "images": list(map(str, imgs)),
                    "answer": gt,
                    "model_output_raw": pred_raw,
                    "model_output_norm": pred_norm,
                    "correct": bool(is_correct),
                    "prompt_given_to_model": chat_text,
                    "model_dir_used": MODEL_DIR,
                }
                if "meta" in row:
                    out_record["meta"] = row["meta"]

                f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                f_out.flush()

                if resume:
                    done_keys.add(_row_key(row))

                written += 1
                print(f"[qwen-base] wrote {written}  acc={correct}/{total}={correct/total:.3f}", flush=True)

            except Exception as e:
                print(f"[qwen-base][ERROR] row {i}: {e}", flush=True)

    print(f"[qwen-base] Done. Wrote {written} rows to {out_path}", flush=True)
    if total > 0:
        print(f"[qwen-base] Final accuracy: {correct}/{total} = {correct/total:.4f}", flush=True)

# ================== ENTRY ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="If set, skip already-processed rows and append new ones.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional max rows to run.")
    parser.add_argument("--test", choices=["test1", "test2"], required=True,
                        help="Which multiview dataset to run: test1 (3 images) or test2 (2 images).")
    args = parser.parse_args()

    if args.test == "test1":
        task_jsonl = TASK_JSONL_TEST1
        out_jsonl = OUT_JSONL_TEST1
    else:
        task_jsonl = TASK_JSONL_TEST2
        out_jsonl = OUT_JSONL_TEST2

    if not task_jsonl.exists():
        raise SystemExit(f"[qwen-base] TASK_JSONL not found: {task_jsonl}")

    print(f"[qwen-base] Using dataset: {task_jsonl}")
    print(f"[qwen-base] Writing outputs to: {out_jsonl}")

    eval_multiview(
        task_path=task_jsonl,
        out_path=out_jsonl,
        test=args.test,
        resume=args.resume,
        limit=args.limit,
    )
