#!/usr/bin/env python3
# import os, json, argparse, re
# from pathlib import Path
# from typing import Set, List, Optional, Dict
# from collections import defaultdict

# import torch
# from transformers import AutoProcessor, AutoModelForVision2Seq
# from peft import PeftModel
# from safetensors.torch import load_file as safetensors_load_file
# from PIL import Image

# # ================== ENV / CONFIG ==================
# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# BASE_MODEL_DIR = "/home/ievab2/models/Qwen2.5-VL-7B-Instruct"

# # ================== DATASETS ==================
# TEXTURE_JSONL = Path("/shared/rsaas/ievab2/PHYSION_TEXTURES/texture_dataset.jsonl")
# COLOR_JSONL   = Path("/shared/rsaas/ievab2/PHYSION_COLORS/color_dataset.jsonl")
# RANDOMIZED_COLORS_JSONL = Path("/shared/rsaas/ievab2/RANDOMIZED_COLORS/randomized_colors_dataset.jsonl")

# # ================== OUTPUT ROOTS ==================
# OUT_ROOT_COLORS  = Path("/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/results/colors")
# OUT_ROOT_TEXTURE = Path("/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/results/texture")
# OUT_ROOT_RANDOMIZED_COLORS = Path("/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/results/randomized_colors")

# # ================== ADAPTER ROOT ==================
# # (same convention as your original qwen finetune checkpoints)
# ADAPTER_ROOT = Path("/shared/rsaas/ievab2/FULL_PHYSION_checkpoints/qwen/round0")

# NUM_FRAMES     = 8
# MAX_NEW_TOKENS = 128

# INSTR = (
#     "You see 8 consecutive frames of a video in temporal order. "
#     "Do not explain; just answer the question concisely. "
# )

# _NEG_RE = re.compile(r"\b(?:no|false|not|won't|will not|doesn't|does not|don't|didn't|did not|can't|cannot)\b", re.I)
# _POS_RE = re.compile(r"\b(?:yes|true|yep|yeah|y)\b", re.I)

# # ================== PATHS ==================

# def build_adapter_dir(split: str, epochs: int) -> str:
#     return f"/shared/rsaas/ievab2/FULL_PHYSION_checkpoints/qwen/round0/{split}_both_{epochs}epochs_4frames"

# def build_out_path(dataset: str, split: str, epochs: int) -> Path:
#     if dataset == "colors":
#         root = OUT_ROOT_COLORS
#     elif dataset == "textures":
#         root = OUT_ROOT_TEXTURE
#     else:
#         root = OUT_ROOT_RANDOMIZED_COLORS
#     return root / f"{split}_epochs{epochs}_results.jsonl"

# # ================== HELPERS ==================
# def normalize_yesno(text: str) -> str:
#     t = (text or "").strip().lower()
#     if not t:
#         return "unknown"
#     first = t.split()[0]
#     if first in ("yes", "y", "true"):
#         return "yes"
#     if first in ("no", "n", "false"):
#         return "no"
#     if _POS_RE.search(t):
#         return "yes"
#     if _NEG_RE.search(t):
#         return "no"
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

# def _extract_assistant(text: str) -> str:
#     if not text:
#         return ""
#     if "<|assistant|>" in text:
#         text = text.split("<|assistant|>")[-1]
#     # Remove stray role words
#     text = re.sub(r"\b(system|user|assistant)\b", "", text, flags=re.I)
#     lines = [l.strip() for l in text.splitlines() if l.strip()]
#     return lines[-1] if lines else text.strip()

# # ================== ADAPTER KEY REMAP + FORCE LOAD ==================
# def _remap_lora_keys_for_runtime(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#     out = {}
#     for k, v in sd.items():
#         k2 = k
#         k2 = k2.replace(".lora_A.weight", ".lora_A.default.weight")
#         k2 = k2.replace(".lora_B.weight", ".lora_B.default.weight")
#         k2 = k2.replace(".model.model.language_model.layers.", ".model.model.layers.")
#         k2 = k2.replace(".model.language_model.layers.", ".model.layers.")
#         k2 = k2.replace(".language_model.layers.", ".layers.")
#         out[k2] = v
#     return out

# def _force_load_adapter_weights_in_memory(peft_model: PeftModel, adapter_dir: str):
#     st_path = Path(adapter_dir) / "adapter_model.safetensors"
#     if not st_path.exists():
#         raise RuntimeError(f"adapter_model.safetensors not found in {adapter_dir}")

#     sd = safetensors_load_file(str(st_path), device="cpu")
#     sd2 = _remap_lora_keys_for_runtime(sd)

#     missing, unexpected = peft_model.load_state_dict(sd2, strict=False)

#     missing_lora = [k for k in missing if "lora_" in k]
#     if missing_lora:
#         print(f"[qwen-ft] WARNING: missing LoRA keys (showing 20): {missing_lora[:20]}", flush=True)
#     else:
#         print("[qwen-ft] LoRA load OK: no missing LoRA keys", flush=True)

#     lora_params = [(n, p) for n, p in peft_model.named_parameters() if "lora_" in n]
#     meta_cnt = sum(int(getattr(p, "is_meta", False)) for _, p in lora_params)
#     print(f"[qwen-ft] manual adapter load: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
#     print(f"[qwen-ft] lora params: {len(lora_params)}  meta={meta_cnt}", flush=True)

#     if meta_cnt > 0:
#         raise RuntimeError(
#             "LoRA params are still on meta after load. "
#             "This usually happens when the base model was initialized with meta tensors. "
#             "Load the base on CPU first (device_map={'': 'cpu'}, low_cpu_mem_usage=False)."
#         )

# # ================== MODEL LOADING ==================
# def _load_model(adapter_dir: str):
#     model_dir = BASE_MODEL_DIR
#     print(f"[qwen-ft] loading base model from {model_dir} …", flush=True)

#     if torch.cuda.is_available():
#         dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
#     else:
#         dtype = torch.float32

#     local_only = os.path.isdir(model_dir)
#     processor = AutoProcessor.from_pretrained(
#         model_dir, trust_remote_code=True, local_files_only=local_only
#     )

#     # IMPORTANT: load on CPU first to avoid meta/offload issues that break adapter loading
#     base = AutoModelForVision2Seq.from_pretrained(
#         model_dir,
#         torch_dtype=dtype,
#         trust_remote_code=True,
#         local_files_only=local_only,
#         low_cpu_mem_usage=False,
#     )
#     base.config.use_cache = False

#     adapter_cfg = Path(adapter_dir) / "adapter_config.json"
#     if not adapter_cfg.exists():
#         raise RuntimeError(f"adapter_config.json not found in {adapter_dir}")

#     print(f"[qwen-ft] attaching LoRA adapter from {adapter_dir} …", flush=True)

#     peft_model = PeftModel.from_pretrained(base, adapter_dir, local_files_only=True)
#     _force_load_adapter_weights_in_memory(peft_model, adapter_dir)
#     model = peft_model

#     if torch.cuda.is_available():
#         model = model.to("cuda")

#     model.eval()
#     print("[qwen-ft] model ready. cuda?", torch.cuda.is_available(), flush=True)
#     return processor, model

# # ================== INFERENCE ==================
# def _open_rgb(p: str) -> Image.Image:
#     with Image.open(p) as im:
#         im = im.convert("RGB") if im.mode != "RGB" else im
#         return im.copy()

# def ask_qwen(processor, model, frame_paths: List[str], question: str):
#     if len(frame_paths) != NUM_FRAMES:
#         raise ValueError(f"Expected {NUM_FRAMES} frames, got {len(frame_paths)}")

#     imgs = [_open_rgb(p) for p in frame_paths]

#     messages = [{
#         "role": "user",
#         "content": (
#             [{"type": "image", "image": im} for im in imgs] +
#             [{"type": "text", "text": INSTR + (question or "")}]
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
#             do_sample=False,
#         )

#     decoded = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
#     pred = _extract_assistant(decoded)
#     return pred, chat_text

# # ================== EVAL LOOP ==================
# def eval_dataset(task_jsonl: Path, out_path: Path, adapter_dir: str, resume: bool):
#     print("[qwen-ft] Starting eval …", flush=True)
#     processor, model = _load_model(adapter_dir)

#     done_keys = _load_done_keys(out_path) if resume else set()
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     mode = "a" if resume else "w"

#     written = 0
#     overall_correct = 0
#     overall_total = 0
#     per_cat = defaultdict(lambda: {"correct": 0, "total": 0})

#     with task_jsonl.open("r", encoding="utf-8") as f_in, out_path.open(mode, encoding="utf-8") as f_out:
#         for i, line in enumerate(f_in):
#             if not line.strip():
#                 continue

#             try:
#                 row = json.loads(line)

#                 frames = row.get("frames") or row.get("frame_paths")
#                 q = row.get("question")
#                 gt = (row.get("answer") or "").strip().lower()
#                 cat = row.get("category", None)

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
#                 print(f"[qwen-ft] qid={qid} name={name}{extra} first={frames[0]} last={frames[-1]}", flush=True)

#                 pred, prompt_text = ask_qwen(processor, model, frames, q)
#                 pr = normalize_yesno(pred)
#                 is_correct = (pr == gt)

#                 overall_total += 1
#                 overall_correct += int(is_correct)
#                 if cat:
#                     per_cat[cat]["total"] += 1
#                     per_cat[cat]["correct"] += int(is_correct)

#                 out_record = {}
#                 out_record.update(row)
#                 out_record["frame_paths"] = list(map(str, frames))
#                 out_record["prompt_given_to_model"] = prompt_text
#                 out_record["model_output_raw"] = pred
#                 out_record["model_output_norm"] = pr
#                 out_record["correct"] = bool(is_correct)
#                 out_record["adapter_dir_used"] = adapter_dir
#                 out_record["base_model_dir_used"] = BASE_MODEL_DIR

#                 f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
#                 f_out.flush()

#                 if resume:
#                     done_keys.add(_row_key(row))

#                 written += 1
#                 if written % 100 == 0:
#                     acc = overall_correct / overall_total if overall_total else 0.0
#                     print(f"[qwen-ft] wrote {written}  acc={overall_correct}/{overall_total}={acc:.3f}", flush=True)

#             except Exception as e:
#                 print(f"[qwen-ft][ERROR] row {i}: {e}", flush=True)

#     print(f"[qwen-ft] Done. Wrote {written} rows to {out_path}", flush=True)
#     if overall_total > 0:
#         print(f"[qwen-ft] Final accuracy: {overall_correct}/{overall_total} = {overall_correct/overall_total:.4f}", flush=True)

# # ================== ENTRY ==================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", choices=["textures", "colors", "randomized_colors"], required=True)
#     parser.add_argument("--split", choices=["SPLIT1", "SPLIT2", "SPLIT3"], required=True)
#     parser.add_argument("--epochs", type=int, choices=[1, 3, 5], required=True)
#     parser.add_argument("--resume", action="store_true",
#                         help="Skip entries already in OUT_JSONL and append new results")
#     args = parser.parse_args()

#     if args.dataset == "textures":
#         task_jsonl = TEXTURE_JSONL
#     elif args.dataset == "colors":
#         task_jsonl = COLOR_JSONL
#     else:
#         task_jsonl = RANDOMIZED_COLORS_JSONL

#     if not task_jsonl.exists():
#         raise SystemExit(f"[qwen-ft] TASK_JSONL not found: {task_jsonl}")

#     adapter_dir = build_adapter_dir(args.split, args.epochs)
#     if not os.path.isdir(adapter_dir):
#         raise SystemExit(f"[qwen-ft] adapter_dir not found: {adapter_dir}")

#     out_path = build_out_path(args.dataset, args.split, args.epochs)

#     print(f"[qwen-ft] dataset={args.dataset} split={args.split} epochs={args.epochs} resume={args.resume}", flush=True)
#     print(f"[qwen-ft] using adapter_dir={adapter_dir}", flush=True)
#     print(f"[qwen-ft] reading dataset={task_jsonl}", flush=True)
#     print(f"[qwen-ft] writing out_jsonl={out_path}", flush=True)

#     eval_dataset(
#         task_jsonl=task_jsonl,
#         out_path=out_path,
#         adapter_dir=adapter_dir,
#         resume=args.resume,
#     )

# # HOW TO RUN:
# #   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_finetuned_qwen.py --dataset colors   --split SPLIT1 --epochs 1
# #   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_finetuned_qwen.py --dataset textures --split SPLIT2 --epochs 3
# #   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_finetuned_qwen.py --dataset randomized_colors --split SPLIT1 --epochs 1
# #
# # Resume:
# #   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_finetuned_qwen.py --dataset colors --split SPLIT1 --epochs 1 --resume
# #   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_finetuned_qwen.py --dataset randomized_colors --split SPLIT1 --epochs 1 --resume

#!/usr/bin/env python3
import os, json, argparse, re
from pathlib import Path
from typing import Set, List, Optional, Dict
from collections import defaultdict

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from safetensors.torch import load_file as safetensors_load_file
from PIL import Image

# ================== ENV / CONFIG ==================
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_MODEL_DIR = "/home/ievab2/models/Qwen2.5-VL-7B-Instruct"

# ================== DATASETS ==================
TEXTURE_JSONL = Path("/shared/rsaas/ievab2/PHYSION_TEXTURES/texture_dataset.jsonl")
COLOR_JSONL   = Path("/shared/rsaas/ievab2/PHYSION_COLORS/color_dataset.jsonl")
RANDOMIZED_COLORS_JSONL = Path("/shared/rsaas/ievab2/RANDOMIZED_COLORS/randomized_colors_dataset.jsonl")
COLORS_NEW_JSONL = Path("/shared/rsaas/ievab2/RANDOMIZED_COLORS_NEW/randomized_colors_new_dataset.jsonl")
OCCLUDERS_JSONL = Path("/shared/rsaas/ievab2/OCCLUDER_TEST/occluder_dataset.jsonl")

# ================== OUTPUT ROOTS ==================
OUT_ROOT_COLORS  = Path("/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/results/colors")
OUT_ROOT_TEXTURE = Path("/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/results/texture")
OUT_ROOT_RANDOMIZED_COLORS = Path("/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/results/randomized_colors")
OUT_ROOT_COLORS_NEW = Path("/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/results/colors_new")
OUT_ROOT_OCCLUDERS = Path("/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/results/occluders")

# ================== ADAPTER ROOT ==================
ADAPTER_ROOT = Path("/shared/rsaas/ievab2/FULL_PHYSION_checkpoints/qwen/round0")

NUM_FRAMES     = 8
MAX_NEW_TOKENS = 128

INSTR = (
    "You see 8 consecutive frames of a video in temporal order. "
    "Do not explain; just answer the question concisely. "
)

_NEG_RE = re.compile(r"\b(?:no|false|not|won't|will not|doesn't|does not|don't|didn't|did not|can't|cannot)\b", re.I)
_POS_RE = re.compile(r"\b(?:yes|true|yep|yeah|y)\b", re.I)

# ================== PATHS ==================

def build_adapter_dir(split: str, epochs: int) -> str:
    return f"/shared/rsaas/ievab2/FULL_PHYSION_checkpoints/qwen/round0/{split}_both_{epochs}epochs_4frames"

def build_out_path(dataset: str, split: str, epochs: int) -> Path:
    if dataset == "colors":
        root = OUT_ROOT_COLORS
    elif dataset == "textures":
        root = OUT_ROOT_TEXTURE
    elif dataset == "randomized_colors":
        root = OUT_ROOT_RANDOMIZED_COLORS
    elif dataset == "colors_new":
        root = OUT_ROOT_COLORS_NEW
    elif dataset == "occluders":
        root = OUT_ROOT_OCCLUDERS
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return root / f"{split}_epochs{epochs}_results.jsonl"

# ================== HELPERS ==================
def normalize_yesno(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return "unknown"
    first = t.split()[0]
    if first in ("yes", "y", "true"):
        return "yes"
    if first in ("no", "n", "false"):
        return "no"
    if _POS_RE.search(t):
        return "yes"
    if _NEG_RE.search(t):
        return "no"
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

def _extract_assistant(text: str) -> str:
    if not text:
        return ""
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1]
    text = re.sub(r"\b(system|user|assistant)\b", "", text, flags=re.I)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()

# ================== ADAPTER KEY REMAP + FORCE LOAD ==================
def _remap_lora_keys_for_runtime(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        k2 = k
        k2 = k2.replace(".lora_A.weight", ".lora_A.default.weight")
        k2 = k2.replace(".lora_B.weight", ".lora_B.default.weight")
        k2 = k2.replace(".model.model.language_model.layers.", ".model.model.layers.")
        k2 = k2.replace(".model.language_model.layers.", ".model.layers.")
        k2 = k2.replace(".language_model.layers.", ".layers.")
        out[k2] = v
    return out

def _force_load_adapter_weights_in_memory(peft_model: PeftModel, adapter_dir: str):
    st_path = Path(adapter_dir) / "adapter_model.safetensors"
    if not st_path.exists():
        raise RuntimeError(f"adapter_model.safetensors not found in {adapter_dir}")

    sd = safetensors_load_file(str(st_path), device="cpu")
    sd2 = _remap_lora_keys_for_runtime(sd)

    missing, unexpected = peft_model.load_state_dict(sd2, strict=False)

    missing_lora = [k for k in missing if "lora_" in k]
    if missing_lora:
        print(f"[qwen-ft] WARNING: missing LoRA keys (showing 20): {missing_lora[:20]}", flush=True)
    else:
        print("[qwen-ft] LoRA load OK: no missing LoRA keys", flush=True)

    lora_params = [(n, p) for n, p in peft_model.named_parameters() if "lora_" in n]
    meta_cnt = sum(int(getattr(p, "is_meta", False)) for _, p in lora_params)
    print(f"[qwen-ft] manual adapter load: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    print(f"[qwen-ft] lora params: {len(lora_params)}  meta={meta_cnt}", flush=True)

    if meta_cnt > 0:
        raise RuntimeError(
            "LoRA params are still on meta after load. "
            "This usually happens when the base model was initialized with meta tensors. "
            "Load the base on CPU first (device_map={'': 'cpu'}, low_cpu_mem_usage=False)."
        )

# ================== MODEL LOADING ==================
def _load_model(adapter_dir: str):
    model_dir = BASE_MODEL_DIR
    print(f"[qwen-ft] loading base model from {model_dir} …", flush=True)

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    local_only = os.path.isdir(model_dir)
    processor = AutoProcessor.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=local_only
    )

    base = AutoModelForVision2Seq.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=local_only,
        low_cpu_mem_usage=False,
    )
    base.config.use_cache = False

    adapter_cfg = Path(adapter_dir) / "adapter_config.json"
    if not adapter_cfg.exists():
        raise RuntimeError(f"adapter_config.json not found in {adapter_dir}")

    print(f"[qwen-ft] attaching LoRA adapter from {adapter_dir} …", flush=True)

    peft_model = PeftModel.from_pretrained(base, adapter_dir, local_files_only=True)
    _force_load_adapter_weights_in_memory(peft_model, adapter_dir)
    model = peft_model

    if torch.cuda.is_available():
        model = model.to("cuda")

    model.eval()
    print("[qwen-ft] model ready. cuda?", torch.cuda.is_available(), flush=True)
    return processor, model

# ================== INFERENCE ==================
def _open_rgb(p: str) -> Image.Image:
    with Image.open(p) as im:
        im = im.convert("RGB") if im.mode != "RGB" else im
        return im.copy()

def ask_qwen(processor, model, frame_paths: List[str], question: str):
    if len(frame_paths) != NUM_FRAMES:
        raise ValueError(f"Expected {NUM_FRAMES} frames, got {len(frame_paths)}")

    imgs = [_open_rgb(p) for p in frame_paths]

    messages = [{
        "role": "user",
        "content": (
            [{"type": "image", "image": im} for im in imgs] +
            [{"type": "text", "text": INSTR + (question or "")}]
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
            do_sample=False,
        )

    decoded = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    pred = _extract_assistant(decoded)
    return pred, chat_text

# ================== EVAL LOOP ==================
def eval_dataset(task_jsonl: Path, out_path: Path, adapter_dir: str, resume: bool, dataset_kind: str):
    print("[qwen-ft] Starting eval …", flush=True)
    processor, model = _load_model(adapter_dir)

    done_keys = _load_done_keys(out_path) if resume else set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"

    written = 0
    overall_correct = 0
    overall_total = 0
    per_cat = defaultdict(lambda: {"correct": 0, "total": 0})

    with task_jsonl.open("r", encoding="utf-8") as f_in, out_path.open(mode, encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            if not line.strip():
                continue

            try:
                row = json.loads(line)

                frames = row.get("frames") or row.get("frame_paths")
                q = row.get("question")
                gt = (row.get("answer") or "").strip().lower()
                cat = row.get("category", None)

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

                print(f"[qwen-ft] dataset={dataset_kind} qid={qid} name={name}{extra} first={frames[0]} last={frames[-1]}", flush=True)

                pred, prompt_text = ask_qwen(processor, model, frames, q)
                pr = normalize_yesno(pred)
                is_correct = (pr == gt)

                overall_total += 1
                overall_correct += int(is_correct)
                if cat:
                    per_cat[cat]["total"] += 1
                    per_cat[cat]["correct"] += int(is_correct)

                out_record = {}
                out_record.update(row)
                out_record["frame_paths"] = list(map(str, frames))
                out_record["prompt_given_to_model"] = prompt_text
                out_record["model_output_raw"] = pred
                out_record["model_output_norm"] = pr
                out_record["correct"] = bool(is_correct)
                out_record["adapter_dir_used"] = adapter_dir
                out_record["base_model_dir_used"] = BASE_MODEL_DIR
                out_record["dataset_kind"] = dataset_kind

                f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                f_out.flush()

                if resume:
                    done_keys.add(_row_key(row))

                written += 1
                if written % 100 == 0:
                    acc = overall_correct / overall_total if overall_total else 0.0
                    print(f"[qwen-ft] wrote {written}  acc={overall_correct}/{overall_total}={acc:.3f}", flush=True)

            except Exception as e:
                print(f"[qwen-ft][ERROR] dataset={dataset_kind} row {i}: {e}", flush=True)

    print(f"[qwen-ft] Done. Wrote {written} rows to {out_path}", flush=True)
    if overall_total > 0:
        print(f"[qwen-ft] Final accuracy: {overall_correct}/{overall_total} = {overall_correct/overall_total:.4f}", flush=True)

# ================== ENTRY ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["textures", "colors", "randomized_colors", "colors_new", "occluders"], required=True)
    parser.add_argument("--split", choices=["SPLIT1", "SPLIT2", "SPLIT3"], required=True)
    parser.add_argument("--epochs", type=int, choices=[1, 3, 5], required=True)
    parser.add_argument("--resume", action="store_true",
                        help="Skip entries already in OUT_JSONL and append new results")
    args = parser.parse_args()

    if args.dataset == "textures":
        task_jsonl = TEXTURE_JSONL
    elif args.dataset == "colors":
        task_jsonl = COLOR_JSONL
    elif args.dataset == "randomized_colors":
        task_jsonl = RANDOMIZED_COLORS_JSONL
    elif args.dataset == "colors_new":
        task_jsonl = COLORS_NEW_JSONL
    else:
        task_jsonl = OCCLUDERS_JSONL

    if not task_jsonl.exists():
        raise SystemExit(f"[qwen-ft] TASK_JSONL not found: {task_jsonl}")

    adapter_dir = build_adapter_dir(args.split, args.epochs)
    if not os.path.isdir(adapter_dir):
        raise SystemExit(f"[qwen-ft] adapter_dir not found: {adapter_dir}")

    out_path = build_out_path(args.dataset, args.split, args.epochs)

    print(f"[qwen-ft] dataset={args.dataset} split={args.split} epochs={args.epochs} resume={args.resume}", flush=True)
    print(f"[qwen-ft] using adapter_dir={adapter_dir}", flush=True)
    print(f"[qwen-ft] reading dataset={task_jsonl}", flush=True)
    print(f"[qwen-ft] writing out_jsonl={out_path}", flush=True)

    eval_dataset(
        task_jsonl=task_jsonl,
        out_path=out_path,
        adapter_dir=adapter_dir,
        resume=args.resume,
        dataset_kind=args.dataset,
    )

# HOW TO RUN:
#   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_finetuned_qwen.py --dataset colors --split SPLIT1 --epochs 1
#   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_finetuned_qwen.py --dataset textures --split SPLIT2 --epochs 3
#   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_finetuned_qwen.py --dataset randomized_colors --split SPLIT1 --epochs 1
#   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_finetuned_qwen.py --dataset colors_new --split SPLIT1 --epochs 1
#   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_finetuned_qwen.py --dataset occluders --split SPLIT1 --epochs 1
#
# Resume:
#   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_finetuned_qwen.py --dataset colors --split SPLIT1 --epochs 1 --resume
#   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_finetuned_qwen.py --dataset randomized_colors --split SPLIT1 --epochs 1 --resume
#   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_finetuned_qwen.py --dataset colors_new --split SPLIT1 --epochs 1 --resume
#   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/qwen/inf_finetuned_qwen.py --dataset occluders --split SPLIT1 --epochs 1 --resume