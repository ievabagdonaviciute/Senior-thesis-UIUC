#!/usr/bin/env python3
# import os, json, argparse, re
# from pathlib import Path
# from typing import Set, List

# import torch
# from transformers import AutoProcessor, AutoModelForVision2Seq

# # ================== ENV / CONFIG ==================
# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# MODEL_DIR = "/home/ievab2/models/Qwen2.5-VL-7B-Instruct"

# # ================== DATASETS ==================
# TEXTURE_JSONL = Path("/shared/rsaas/ievab2/PHYSION_TEXTURES/texture_dataset.jsonl")
# COLOR_JSONL   = Path("/shared/rsaas/ievab2/PHYSION_COLORS/color_dataset.jsonl")
# RANDOMIZED_COLORS_JSONL = Path("/shared/rsaas/ievab2/RANDOMIZED_COLORS/randomized_colors_dataset.jsonl")

# # ================== OUTPUTS ==================
# OUT_COLORS  = Path("/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/results/colors/base_results.jsonl")
# OUT_TEXTURE = Path("/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/results/texture/base_results.jsonl")
# OUT_RANDOMIZED_COLORS = Path("/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/results/randomized_colors/base_results.jsonl")

# NUM_FRAMES     = 8
# MAX_NEW_TOKENS = 128

# _NEG_RE = re.compile(r"\b(?:no|false|not|won't|will not|doesn't|does not|don't|didn't|did not|can't|cannot)\b", re.I)
# _POS_RE = re.compile(r"\b(?:yes|true|yep|yeah|y)\b", re.I)

# # ================== HELPERS ==================
# def normalize_yesno(text: str) -> str:
#     t = (text or "").strip().lower()
#     if t.startswith("yes"):
#         return "yes"
#     if t.startswith("no"):
#         return "no"
#     if _NEG_RE.search(t):
#         return "no"
#     if _POS_RE.search(t):
#         return "yes"
#     return "unknown"

# def _validate_frame_paths(paths: List[str]):
#     if not isinstance(paths, list) or len(paths) != NUM_FRAMES:
#         raise ValueError(f"Expected exactly {NUM_FRAMES} frame paths, got {0 if paths is None else len(paths)}")
#     bad = [p for p in paths if not p or not os.path.isabs(p) or not os.path.exists(p)]
#     if bad:
#         raise FileNotFoundError(f"Bad/missing frame paths: {bad[:3]}{'...' if len(bad)>3 else ''}")

# def _row_key(row: dict) -> str:
#     frames = row.get("frames") or row.get("frame_paths") or []
#     q = row.get("question") or row.get("prompt") or ""
#     first = frames[0] if frames else ""
#     last  = frames[-1] if frames else ""
#     tex = row.get("texture") or ""
#     return f"f::{first}|{last}||tex::{tex}||q::{q}"

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
#             frames = rec.get("frame_paths") or rec.get("frames") or []
#             q = rec.get("question") or rec.get("prompt") or rec.get("prompt_given_to_model") or ""
#             first = frames[0] if frames else ""
#             last  = frames[-1] if frames else ""
#             tex = rec.get("texture") or ""
#             done.add(f"f::{first}|{last}||tex::{tex}||q::{q}")
#     return done

# # ================== QWEN-SPECIFIC: EXTRACT ASSISTANT ==================
# def _extract_assistant(text: str) -> str:
#     if not text:
#         return ""
#     if "<|assistant|>" in text:
#         return text.split("<|assistant|>", maxsplit=1)[-1].strip()
#     m = re.search(r"(?:^|\n)assistant\s*\n(.*)\Z", text, flags=re.IGNORECASE | re.DOTALL)
#     if m:
#         return m.group(1).strip()
#     return text.strip()

# # ================== MODEL LOADING ==================
# def _load_model():
#     print(f"[qwen] loading model from {MODEL_DIR} …", flush=True)

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
#     print("[qwen] model ready. cuda?", torch.cuda.is_available(), flush=True)
#     return processor, model

# # ================== INFERENCE ==================
# def ask_qwen(processor, model, frame_paths: List[str], question: str) -> str:
#     if len(frame_paths) != NUM_FRAMES:
#         raise ValueError(f"Expected {NUM_FRAMES} frames, got {len(frame_paths)}")

#     messages = [{
#         "role": "user",
#         "content": (
#             [{"type": "image", "image": p} for p in frame_paths] +
#             [{"type": "text",
#               "text": "You see 8 consecutive frames of a video in temporal order. "
#                       "Do not explain; just answer the question concisely. "
#                       + (question or "")}]
#         ),
#     }]

#     chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     device = next(model.parameters()).device

#     inputs = processor(
#         text=[chat_text],
#         images=[frame_paths],
#         return_tensors="pt",
#     ).to(device)

#     with torch.inference_mode():
#         out_ids = model.generate(
#             **inputs,
#             max_new_tokens=MAX_NEW_TOKENS,
#             do_sample=False,
#         )

#     text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
#     return _extract_assistant(text)

# # ================== MAIN EVAL LOOP ==================
# def eval_dataset(task_jsonl: Path, out_path: Path, resume: bool = False):
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     processor, model = _load_model()

#     done_keys = _load_done_keys(out_path) if resume else set()
#     mode = "a" if resume else "w"

#     written = 0
#     correct = 0

#     with task_jsonl.open("r", encoding="utf-8") as f_in, out_path.open(mode, encoding="utf-8") as f_out:
#         for i, line in enumerate(f_in):
#             if not line.strip():
#                 continue

#             try:
#                 row = json.loads(line)

#                 frames = row.get("frames") or row.get("frame_paths")
#                 q = row.get("question")
#                 gt = (row.get("answer") or "").strip().lower()

#                 if not isinstance(frames, list) or len(frames) != NUM_FRAMES:
#                     raise ValueError(f"Row must have 'frames' or 'frame_paths' with exactly {NUM_FRAMES} image paths")
#                 if not q:
#                     raise ValueError("Row missing 'question'")
#                 if gt not in ("yes", "no"):
#                     raise ValueError("Row missing/invalid 'answer' (must be 'yes' or 'no')")

#                 _validate_frame_paths(frames)

#                 if resume:
#                     key = _row_key(row)
#                     if key in done_keys:
#                         continue

#                 qid = row.get("qid") or row.get("id") or f"row{i}"
#                 name = row.get("name", "unknown")
#                 extra = f" texture={row.get('texture')}" if "texture" in row else ""
#                 print(f"[qwen] qid={qid} name={name}{extra} first={frames[0]} last={frames[-1]}", flush=True)

#                 pred = ask_qwen(processor, model, frames, q)
#                 pred_norm = normalize_yesno(pred)
#                 is_correct = (pred_norm == gt)
#                 correct += int(is_correct)

#                 out_record = {}
#                 out_record.update(row)
#                 out_record["frame_paths"] = list(map(str, frames))
#                 out_record["prompt_given_to_model"] = (
#                     "You see 8 consecutive frames of a video in temporal order. "
#                     "Do not explain; just answer the question concisely. "
#                     + (q or "")
#                 )
#                 out_record["model_output_raw"] = pred
#                 out_record["model_output_norm"] = pred_norm
#                 out_record["correct"] = bool(is_correct)
#                 out_record["model_dir_used"] = MODEL_DIR

#                 f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
#                 f_out.flush()

#                 if resume:
#                     done_keys.add(_row_key(row))

#                 written += 1
#                 if written % 100 == 0:
#                     acc = correct / written if written else 0.0
#                     print(f"[qwen] wrote {written}  acc={correct}/{written}={acc:.3f}", flush=True)

#             except Exception as e:
#                 print(f"[qwen][ERROR] row {i}: {e}", flush=True)

#     print(f"[qwen] Done. Wrote {written} rows to {out_path}", flush=True)
#     if written > 0:
#         print(f"[qwen] Final accuracy: {correct}/{written} = {correct/written:.4f}", flush=True)

# # ================== ENTRY ==================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", choices=["textures", "colors", "randomized_colors"], required=True)
#     parser.add_argument("--resume", action="store_true",
#                         help="If set, skip already-processed rows and append new ones.")
#     args = parser.parse_args()

#     if args.dataset == "textures":
#         task_jsonl = TEXTURE_JSONL
#         out_path = OUT_TEXTURE
#     elif args.dataset == "colors":
#         task_jsonl = COLOR_JSONL
#         out_path = OUT_COLORS
#     else:
#         task_jsonl = RANDOMIZED_COLORS_JSONL
#         out_path = OUT_RANDOMIZED_COLORS

#     if not task_jsonl.exists():
#         raise SystemExit(f"[qwen] TASK_JSONL not found: {task_jsonl}")

#     print(f"[qwen] dataset={args.dataset}", flush=True)
#     print(f"[qwen] reading dataset={task_jsonl}", flush=True)
#     print(f"[qwen] writing out_jsonl={out_path}", flush=True)

#     eval_dataset(
#         task_jsonl=task_jsonl,
#         out_path=out_path,
#         resume=args.resume,
#     )

# # HOW TO RUN:
# #   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_base_qwen.py --dataset colors
# #   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_base_qwen.py --dataset textures
# #   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_base_qwen.py --dataset randomized_colors
# #
# # Resume:
# #   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_base_qwen.py --dataset colors --resume
# #   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_base_qwen.py --dataset randomized_colors --resume

#!/usr/bin/env python3
import os, json, argparse, re
from pathlib import Path
from typing import Set, List

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# ================== ENV / CONFIG ==================
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_DIR = "/home/ievab2/models/Qwen2.5-VL-7B-Instruct"

# ================== DATASETS ==================
TEXTURE_JSONL = Path("/shared/rsaas/ievab2/PHYSION_TEXTURES/texture_dataset.jsonl")
COLOR_JSONL   = Path("/shared/rsaas/ievab2/PHYSION_COLORS/color_dataset.jsonl")
RANDOMIZED_COLORS_JSONL = Path("/shared/rsaas/ievab2/RANDOMIZED_COLORS/randomized_colors_dataset.jsonl")
COLORS_NEW_JSONL = Path("/shared/rsaas/ievab2/RANDOMIZED_COLORS_NEW/randomized_colors_new_dataset.jsonl")
OCCLUDERS_JSONL = Path("/shared/rsaas/ievab2/OCCLUDER_TEST/occluder_dataset.jsonl")

# ================== OUTPUTS ==================
OUT_COLORS  = Path("/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/results/colors/base_results.jsonl")
OUT_TEXTURE = Path("/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/results/texture/base_results.jsonl")
OUT_RANDOMIZED_COLORS = Path("/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/results/randomized_colors/base_results.jsonl")
OUT_COLORS_NEW = Path("/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/results/colors_new/base_results.jsonl")
OUT_OCCLUDERS = Path("/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/results/occluders/base_results.jsonl")

NUM_FRAMES     = 8
MAX_NEW_TOKENS = 128

_NEG_RE = re.compile(r"\b(?:no|false|not|won't|will not|doesn't|does not|don't|didn't|did not|can't|cannot)\b", re.I)
_POS_RE = re.compile(r"\b(?:yes|true|yep|yeah|y)\b", re.I)

# ================== HELPERS ==================
def normalize_yesno(text: str) -> str:
    t = (text or "").strip().lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    if _NEG_RE.search(t):
        return "no"
    if _POS_RE.search(t):
        return "yes"
    return "unknown"

def _validate_frame_paths(paths: List[str]):
    if not isinstance(paths, list) or len(paths) != NUM_FRAMES:
        raise ValueError(f"Expected exactly {NUM_FRAMES} frame paths, got {0 if paths is None else len(paths)}")
    bad = [p for p in paths if not p or not os.path.isabs(p) or not os.path.exists(p)]
    if bad:
        raise FileNotFoundError(f"Bad/missing frame paths: {bad[:3]}{'...' if len(bad)>3 else ''}")

def _row_key(row: dict) -> str:
    frames = row.get("frames") or row.get("frame_paths") or []
    q = row.get("question") or row.get("prompt") or ""
    first = frames[0] if frames else ""
    last  = frames[-1] if frames else ""
    tex = row.get("texture") or ""
    sample_type = row.get("type") or ""
    distr = row.get("distr", "")
    config = row.get("config", "")
    return f"f::{first}|{last}||tex::{tex}||type::{sample_type}||distr::{distr}||config::{config}||q::{q}"

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
            frames = rec.get("frame_paths") or rec.get("frames") or []
            q = rec.get("question") or rec.get("prompt") or rec.get("prompt_given_to_model") or ""
            first = frames[0] if frames else ""
            last  = frames[-1] if frames else ""
            tex = rec.get("texture") or ""
            sample_type = rec.get("type") or ""
            distr = rec.get("distr", "")
            config = rec.get("config", "")
            done.add(f"f::{first}|{last}||tex::{tex}||type::{sample_type}||distr::{distr}||config::{config}||q::{q}")
    return done

# ================== QWEN-SPECIFIC: EXTRACT ASSISTANT ==================
def _extract_assistant(text: str) -> str:
    if not text:
        return ""
    if "<|assistant|>" in text:
        return text.split("<|assistant|>", maxsplit=1)[-1].strip()
    m = re.search(r"(?:^|\n)assistant\s*\n(.*)\Z", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()

# ================== MODEL LOADING ==================
def _load_model():
    print(f"[qwen] loading model from {MODEL_DIR} …", flush=True)

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
    print("[qwen] model ready. cuda?", torch.cuda.is_available(), flush=True)
    return processor, model

# ================== INFERENCE ==================
def ask_qwen(processor, model, frame_paths: List[str], question: str) -> str:
    if len(frame_paths) != NUM_FRAMES:
        raise ValueError(f"Expected {NUM_FRAMES} frames, got {len(frame_paths)}")

    messages = [{
        "role": "user",
        "content": (
            [{"type": "image", "image": p} for p in frame_paths] +
            [{"type": "text",
              "text": "You see 8 consecutive frames of a video in temporal order. "
                      "Do not explain; just answer the question concisely. "
                      + (question or "")}]
        ),
    }]

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device

    inputs = processor(
        text=[chat_text],
        images=[frame_paths],
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    return _extract_assistant(text)

# ================== MAIN EVAL LOOP ==================
def eval_dataset(task_jsonl: Path, out_path: Path, dataset_kind: str, resume: bool = False):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processor, model = _load_model()

    done_keys = _load_done_keys(out_path) if resume else set()
    mode = "a" if resume else "w"

    written = 0
    correct = 0

    with task_jsonl.open("r", encoding="utf-8") as f_in, out_path.open(mode, encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            if not line.strip():
                continue

            try:
                row = json.loads(line)

                frames = row.get("frames") or row.get("frame_paths")
                q = row.get("question")
                gt = (row.get("answer") or "").strip().lower()

                if not isinstance(frames, list) or len(frames) != NUM_FRAMES:
                    raise ValueError(f"Row must have 'frames' or 'frame_paths' with exactly {NUM_FRAMES} image paths")
                if not q:
                    raise ValueError("Row missing 'question'")
                if gt not in ("yes", "no"):
                    raise ValueError("Row missing/invalid 'answer' (must be 'yes' or 'no')")

                _validate_frame_paths(frames)

                if resume:
                    key = _row_key(row)
                    if key in done_keys:
                        continue

                qid = row.get("qid") or row.get("id") or f"row{i}"
                name = row.get("name", "unknown")

                extra = ""
                if dataset_kind == "textures":
                    extra += f" texture={row.get('texture')}"
                if dataset_kind in {"randomized_colors", "colors_new"}:
                    extra += f" distr={row.get('distr')} config={row.get('config')} type={row.get('type')}"
                if dataset_kind == "occluders":
                    extra += f" config={row.get('config')} type={row.get('type')}"

                print(
                    f"[qwen] dataset={dataset_kind} qid={qid} name={name}{extra} "
                    f"first={frames[0]} last={frames[-1]}",
                    flush=True
                )

                pred = ask_qwen(processor, model, frames, q)
                pred_norm = normalize_yesno(pred)
                is_correct = (pred_norm == gt)
                correct += int(is_correct)

                out_record = {}
                out_record.update(row)
                out_record["frame_paths"] = list(map(str, frames))
                out_record["prompt_given_to_model"] = (
                    "You see 8 consecutive frames of a video in temporal order. "
                    "Do not explain; just answer the question concisely. "
                    + (q or "")
                )
                out_record["model_output_raw"] = pred
                out_record["model_output_norm"] = pred_norm
                out_record["correct"] = bool(is_correct)
                out_record["model_dir_used"] = MODEL_DIR
                out_record["dataset_kind"] = dataset_kind

                f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                f_out.flush()

                if resume:
                    done_keys.add(_row_key(row))

                written += 1
                if written % 100 == 0:
                    acc = correct / written if written else 0.0
                    print(f"[qwen] wrote {written}  acc={correct}/{written}={acc:.3f}", flush=True)

            except Exception as e:
                print(f"[qwen][ERROR] dataset={dataset_kind} row {i}: {e}", flush=True)

    print(f"[qwen] Done. Wrote {written} rows to {out_path}", flush=True)
    if written > 0:
        print(f"[qwen] Final accuracy: {correct}/{written} = {correct/written:.4f}", flush=True)

# ================== ENTRY ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["textures", "colors", "randomized_colors", "colors_new", "occluders"],
        required=True
    )
    parser.add_argument("--resume", action="store_true",
                        help="If set, skip already-processed rows and append new ones.")
    args = parser.parse_args()

    if args.dataset == "textures":
        task_jsonl = TEXTURE_JSONL
        out_path = OUT_TEXTURE
    elif args.dataset == "colors":
        task_jsonl = COLOR_JSONL
        out_path = OUT_COLORS
    elif args.dataset == "randomized_colors":
        task_jsonl = RANDOMIZED_COLORS_JSONL
        out_path = OUT_RANDOMIZED_COLORS
    elif args.dataset == "colors_new":
        task_jsonl = COLORS_NEW_JSONL
        out_path = OUT_COLORS_NEW
    else:
        task_jsonl = OCCLUDERS_JSONL
        out_path = OUT_OCCLUDERS

    if not task_jsonl.exists():
        raise SystemExit(f"[qwen] TASK_JSONL not found: {task_jsonl}")

    print(f"[qwen] dataset={args.dataset}", flush=True)
    print(f"[qwen] reading dataset={task_jsonl}", flush=True)
    print(f"[qwen] writing out_jsonl={out_path}", flush=True)

    eval_dataset(
        task_jsonl=task_jsonl,
        out_path=out_path,
        dataset_kind=args.dataset,
        resume=args.resume,
    )

# HOW TO RUN:
#   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_base_qwen.py --dataset colors
#   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_base_qwen.py --dataset textures
#   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_base_qwen.py --dataset randomized_colors
#   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_base_qwen.py --dataset colors_new
#   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_base_qwen.py --dataset occluders
#
# Resume:
#   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_base_qwen.py --dataset colors --resume
#   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_base_qwen.py --dataset randomized_colors --resume
#   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_base_qwen.py --dataset colors_new --resume
#   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_base_qwen.py --dataset occluders --resume