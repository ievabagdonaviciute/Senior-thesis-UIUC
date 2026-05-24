#!/usr/bin/env python3
# import os, json, argparse, re
# from pathlib import Path
# from typing import Optional, Set, List

# import torch
# from PIL import Image
# from transformers import AutoTokenizer, AutoModel
# import torchvision.transforms as T
# from torchvision.transforms.functional import InterpolationMode
# from peft import PeftModel, LoraConfig

# # ================== ENV / CONFIG ==================
# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# BASE_TOKENIZER = "/home/ievab2/models/InternVL2-8B"
# MODEL_DIR      = "/home/ievab2/models/InternVL2-8B"

# # ================== DATASETS ==================
# TEXTURE_JSONL = Path("/shared/rsaas/ievab2/PHYSION_TEXTURES/texture_dataset.jsonl")
# COLOR_JSONL   = Path("/shared/rsaas/ievab2/PHYSION_COLORS/color_dataset.jsonl")
# RANDOMIZED_COLORS_JSONL = Path("/shared/rsaas/ievab2/RANDOMIZED_COLORS/randomized_colors_dataset.jsonl")

# # ================== RESULTS ROOT ==================
# RESULTS_ROOT = Path("/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/results")

# # ================== ADAPTER ROOT ==================
# ADAPTER_ROOT = Path("/shared/rsaas/ievab2/FULL_PHYSION_checkpoints/internvl/round0")

# # ================== INFERENCE SETTINGS ==================
# NUM_FRAMES     = 8
# MAX_NEW_TOKENS = 128
# INPUT_SIZE     = 448

# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD  = (0.229, 0.224, 0.225)

# _NEG_RE = re.compile(r"\b(?:no|false|not|won't|will not|doesn't|does not|don't|didn't|did not|can't|cannot)\b", re.I)
# _POS_RE = re.compile(r"\b(?:yes|true|yep|yeah|y)\b", re.I)

# # ================== ADAPTER PATHS (MATCH YOUR ORIGINAL LOGIC, curriculum only) ==================
# def build_dir_paths(split: str, epochs: int, dataset: str):
#     adapter_dir = ADAPTER_ROOT / f"{split}_both_{epochs}epochs_4frames"

#     # requested outputs:
#     #   .../results/colors/SPLIT1_epochs3_results.jsonl
#     #   .../results/texture/SPLIT1_epochs3_results.jsonl
#     #   .../results/randomized_colors/SPLIT1_epochs3_results.jsonl
#     if dataset == "colors":
#         out_subdir = "colors"
#     elif dataset == "textures":
#         out_subdir = "texture"
#     else:
#         out_subdir = "randomized_colors"

#     out_dir = RESULTS_ROOT / out_subdir
#     out_jsonl = out_dir / f"{split}_epochs{epochs}_results.jsonl"

#     tag = "curriculum"
#     return str(adapter_dir), out_dir, out_jsonl, tag

# # ================== HELPERS ==================
# def normalize_yesno(text: str) -> str:
#     t = (text or "").strip().lower()
#     if not t:
#         return "unknown"
#     if t.startswith("yes"):
#         return "yes"
#     if t.startswith("no"):
#         return "no"
#     if _NEG_RE.search(t):
#         return "no"
#     if _POS_RE.search(t):
#         return "yes"
#     return "unknown"

# def build_transform(input_size: int):
#     return T.Compose([
#         T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
#         T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
#         T.ToTensor(),
#         T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#     ])

# def _validate_frame_paths(paths: List[str]):
#     if not isinstance(paths, list) or len(paths) != NUM_FRAMES:
#         raise ValueError(f"Row must have 'frame_paths' (or 'frames') with exactly {NUM_FRAMES} absolute paths")
#     bad = [p for p in paths if (not p) or (not os.path.isabs(p)) or (not os.path.exists(p))]
#     if bad:
#         raise FileNotFoundError(f"Bad/missing frame paths: {bad[:2]}{'...' if len(bad)>2 else ''}")

# def load_frames_tensor_from_paths(paths: List[Path], input_size: int = INPUT_SIZE) -> torch.Tensor:
#     transform = build_transform(input_size)
#     tensors = []
#     for p in paths:
#         p = Path(p)
#         if not p.exists():
#             raise FileNotFoundError(f"Missing frame path: {p}")
#         img = Image.open(p)
#         tensors.append(transform(img))
#     return torch.stack(tensors, dim=0)  # [k,3,H,W]

# def _row_key(row: dict) -> str:
#     frames = row.get("frame_paths") or row.get("frames") or []
#     q = row.get("question") or ""
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
#             q = rec.get("question") or rec.get("prompt_given_to_model") or rec.get("prompt") or ""
#             first = frames[0] if frames else ""
#             last  = frames[-1] if frames else ""
#             tex = rec.get("texture") or ""
#             done.add(f"f::{first}|{last}||tex::{tex}||q::{q}")
#     return done

# # ================== MODEL LOADING (ADAPTER ONLY, SAME STYLE AS YOUR SCRIPT) ==================
# def _load_model(adapter_dir: str):
#     print(f"[internvl2] loading adapter from {adapter_dir} …", flush=True)

#     adapter_cfg = os.path.join(adapter_dir, "adapter_config.json")
#     if not os.path.exists(adapter_cfg):
#         raise RuntimeError(
#             f"[ERROR] Expected a LoRA adapter folder at:\n"
#             f"    {adapter_dir}\n"
#             f"but adapter_config.json is missing.\n"
#             f"Refusing to load the unfine-tuned base model."
#         )

#     tokenizer = AutoTokenizer.from_pretrained(
#         BASE_TOKENIZER, trust_remote_code=True, local_files_only=True, use_fast=False
#     )
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     use_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

#     base = AutoModel.from_pretrained(
#         MODEL_DIR,
#         trust_remote_code=True,
#         local_files_only=True,
#         torch_dtype=use_dtype,
#         device_map="auto",
#         attn_implementation="eager",
#     )

#     raw = json.loads(Path(adapter_cfg).read_text(encoding="utf-8"))
#     allowed_keys = set(LoraConfig.__dataclass_fields__.keys())
#     clean = {k: v for k, v in raw.items() if k in allowed_keys}
#     lc = LoraConfig(**clean)

#     model = PeftModel.from_pretrained(base, adapter_dir, config=lc)

#     # Patch missing set_output_embeddings on inner chat model (same as your original)
#     try:
#         inner = model.base_model
#         if hasattr(inner, "model"):
#             inner = inner.model
#         if not hasattr(inner, "set_output_embeddings"):
#             print("[PATCH] Adding dummy set_output_embeddings to", type(inner), flush=True)
#             def _set_output_embeddings(self, new_embeds):
#                 return
#             inner.set_output_embeddings = _set_output_embeddings.__get__(inner, inner.__class__)
#     except Exception as e:
#         print(f"[WARN] could not patch set_output_embeddings: {e}", flush=True)

#     # Ensure embeddings match tokenizer size
#     emb_n = model.get_input_embeddings().weight.shape[0]
#     vs = len(tokenizer)
#     if vs > emb_n:
#         print(f"[PATCH] Resizing token embeddings from {emb_n} → {vs}", flush=True)
#         model.resize_token_embeddings(vs)
#     elif vs < emb_n:
#         print(f"[WARN] tokenizer vocab ({vs}) < model embeddings ({emb_n})", flush=True)

#     model.eval()
#     print("[internvl2] model ready.", flush=True)
#     return tokenizer, model

# # ================== INFERENCE ==================
# def ask_internvl2_8frames(tokenizer, model, frame_paths: List[str], question: str):
#     user_text = (
#         "You see 8 consecutive frames of a video in temporal order. "
#         "Do not explain; just answer the question concisely. "
#     ) + (question or "")

#     pixel_values = load_frames_tensor_from_paths([Path(p) for p in frame_paths], input_size=INPUT_SIZE)
#     k = pixel_values.shape[0]
#     prompt = ("<image>\n" * k) + user_text

#     device = next(model.parameters()).device
#     dtype  = next(model.parameters()).dtype
#     if device.type == "cpu":
#         dtype = torch.float32
#     pixel_values = pixel_values.to(device=device, dtype=dtype)

#     generation_config = {
#         "max_new_tokens": MAX_NEW_TOKENS,
#         "do_sample": False,
#         "temperature": 0.0,
#     }

#     with torch.inference_mode():
#         if torch.cuda.is_available():
#             with torch.cuda.amp.autocast(dtype=torch.float16):
#                 response = model.chat(tokenizer, pixel_values, prompt, generation_config)
#         else:
#             response = model.chat(tokenizer, pixel_values, prompt, generation_config)

#     return response, prompt, list(map(str, frame_paths))

# # ================== EVAL LOOP ==================
# def eval_dataset(task_jsonl: Path, out_jsonl: Path, adapter_dir: str,
#                  resume: bool,
#                  split: str, epochs: int, tag: str, dataset: str):
#     print("[internvl2] Starting eval …", flush=True)
#     tokenizer, model = _load_model(adapter_dir)

#     done_keys = _load_done_keys(out_jsonl) if resume else set()
#     out_jsonl.parent.mkdir(parents=True, exist_ok=True)
#     mode = "a" if resume else "w"

#     written = 0
#     seen = 0
#     correct = 0

#     with task_jsonl.open("r", encoding="utf-8") as f_in, out_jsonl.open(mode, encoding="utf-8") as f_out:
#         for i, line in enumerate(f_in):
#             if not line.strip():
#                 continue

#             seen += 1
#             try:
#                 row = json.loads(line)

#                 frames = row.get("frame_paths") or row.get("frames")
#                 q = row.get("question")
#                 gt = (row.get("answer") or "").strip().lower()

#                 if not isinstance(frames, list) or len(frames) != NUM_FRAMES:
#                     raise ValueError(f"Row must have 'frame_paths' with exactly {NUM_FRAMES} absolute paths")
#                 if not isinstance(q, str) or not q.strip():
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
#                 print(f"[internvl2] qid={qid} split={split} epochs={epochs} dataset={dataset} name={name}{extra}", flush=True)

#                 pred_raw, prompt_used, used_paths = ask_internvl2_8frames(tokenizer, model, frames, q)
#                 pred_norm = normalize_yesno(pred_raw)
#                 is_correct = (pred_norm == gt)
#                 correct += int(is_correct)

#                 out_record = {}
#                 out_record.update(row)
#                 out_record["split"] = split
#                 out_record["epochs"] = int(epochs)
#                 out_record["tag"] = tag
#                 out_record["dataset"] = dataset
#                 out_record["frame_paths"] = used_paths
#                 out_record["prompt_given_to_model"] = prompt_used
#                 out_record["model_output_raw"] = pred_raw
#                 out_record["model_output_norm"] = pred_norm
#                 out_record["correct"] = bool(is_correct)
#                 out_record["adapter_dir_used"] = adapter_dir

#                 f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
#                 f_out.flush()

#                 if resume:
#                     done_keys.add(_row_key(row))

#                 written += 1
#                 if written % 100 == 0:
#                     acc = correct / written if written else 0.0
#                     print(f"[internvl2] wrote {written}  acc={correct}/{written}={acc:.3f}", flush=True)

#             except Exception as e:
#                 print(f"[internvl2][ERROR] row {i}: {e}", flush=True)

#     print(f"[internvl2] Done. Wrote {written} rows to {out_jsonl}", flush=True)
#     if written > 0:
#         print(f"[internvl2] Final accuracy: {correct}/{written} = {correct/written:.4f}", flush=True)

# # ================== ENTRY ==================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", choices=["textures", "colors", "randomized_colors"], required=True)
#     parser.add_argument("--split", choices=["SPLIT1", "SPLIT2", "SPLIT3"], required=True)
#     parser.add_argument("--epochs", type=int, choices=[1, 3, 5], required=True)
#     parser.add_argument("--resume", action="store_true",
#                         help="Skip entries already in output JSONL and append new results")
#     args = parser.parse_args()

#     if args.dataset == "textures":
#         task_jsonl = TEXTURE_JSONL
#     elif args.dataset == "colors":
#         task_jsonl = COLOR_JSONL
#     else:
#         task_jsonl = RANDOMIZED_COLORS_JSONL

#     if not task_jsonl.exists():
#         raise SystemExit(f"[internvl2] TASK_JSONL not found: {task_jsonl}")

#     adapter_dir, out_dir, out_jsonl, tag = build_dir_paths(args.split, args.epochs, args.dataset)

#     # sanity: adapter must exist
#     if not os.path.isdir(adapter_dir):
#         raise SystemExit(f"[internvl2] adapter_dir not found: {adapter_dir}")

#     out_dir.mkdir(parents=True, exist_ok=True)

#     print(f"[internvl2] dataset={args.dataset} split={args.split} epochs={args.epochs}", flush=True)
#     print(f"[internvl2] using adapter_dir={adapter_dir}", flush=True)
#     print(f"[internvl2] reading dataset={task_jsonl}", flush=True)
#     print(f"[internvl2] writing out_jsonl={out_jsonl}", flush=True)

#     eval_dataset(
#         task_jsonl=task_jsonl,
#         out_jsonl=out_jsonl,
#         adapter_dir=adapter_dir,
#         resume=args.resume,
#         split=args.split,
#         epochs=args.epochs,
#         tag=tag,
#         dataset=args.dataset,
#     )

# # HOW TO RUN:
# #   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/inf_finetuned_internvl.py --dataset colors --split SPLIT1 --epochs 3
# #   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/inf_finetuned_internvl.py --dataset textures --split SPLIT2 --epochs 5
# #   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/inf_finetuned_internvl.py --dataset randomized_colors --split SPLIT1 --epochs 3
# #
# # Resume:
# #   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/inf_finetuned_internvl.py --dataset colors --split SPLIT1 --epochs 3 --resume
# #   python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/inf_finetuned_internvl.py --dataset randomized_colors --split SPLIT1 --epochs 3 --resume

#!/usr/bin/env python3
import os, json, argparse, re
from pathlib import Path
from typing import Optional, Set, List

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from peft import PeftModel, LoraConfig

# ================== ENV / CONFIG ==================
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_TOKENIZER = "/home/ievab2/models/InternVL2-8B"
MODEL_DIR      = "/home/ievab2/models/InternVL2-8B"

# ================== DATASETS ==================
TEXTURE_JSONL = Path("/shared/rsaas/ievab2/PHYSION_TEXTURES/texture_dataset.jsonl")
COLOR_JSONL   = Path("/shared/rsaas/ievab2/PHYSION_COLORS/color_dataset.jsonl")
RANDOMIZED_COLORS_JSONL = Path("/shared/rsaas/ievab2/RANDOMIZED_COLORS/randomized_colors_dataset.jsonl")
COLORS_NEW_JSONL = Path("/shared/rsaas/ievab2/RANDOMIZED_COLORS_NEW/randomized_colors_new_dataset.jsonl")
OCCLUDERS_JSONL = Path("/shared/rsaas/ievab2/OCCLUDER_TEST/occluder_dataset.jsonl")

# ================== RESULTS ROOT ==================
RESULTS_ROOT = Path("/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/results")

# ================== ADAPTER ROOT ==================
ADAPTER_ROOT = Path("/shared/rsaas/ievab2/FULL_PHYSION_checkpoints/internvl/round0")

# ================== INFERENCE SETTINGS ==================
NUM_FRAMES     = 8
MAX_NEW_TOKENS = 128
INPUT_SIZE     = 448

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

_NEG_RE = re.compile(r"\b(?:no|false|not|won't|will not|doesn't|does not|don't|didn't|did not|can't|cannot)\b", re.I)
_POS_RE = re.compile(r"\b(?:yes|true|yep|yeah|y)\b", re.I)

# ================== ADAPTER PATHS ==================
def build_dir_paths(split: str, epochs: int, dataset: str):
    adapter_dir = ADAPTER_ROOT / f"{split}_both_{epochs}epochs_4frames"

    if dataset == "colors":
        out_subdir = "colors"
    elif dataset == "textures":
        out_subdir = "texture"
    elif dataset == "randomized_colors":
        out_subdir = "randomized_colors"
    elif dataset == "colors_new":
        out_subdir = "colors_new"
    elif dataset == "occluders":
        out_subdir = "occluders"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    out_dir = RESULTS_ROOT / out_subdir
    out_jsonl = out_dir / f"{split}_epochs{epochs}_results.jsonl"

    tag = "curriculum"
    return str(adapter_dir), out_dir, out_jsonl, tag

# ================== HELPERS ==================
def normalize_yesno(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return "unknown"
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    if _NEG_RE.search(t):
        return "no"
    if _POS_RE.search(t):
        return "yes"
    return "unknown"

def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def _validate_frame_paths(paths: List[str]):
    if not isinstance(paths, list) or len(paths) != NUM_FRAMES:
        raise ValueError(f"Row must have 'frame_paths' (or 'frames') with exactly {NUM_FRAMES} absolute paths")
    bad = [p for p in paths if (not p) or (not os.path.isabs(p)) or (not os.path.exists(p))]
    if bad:
        raise FileNotFoundError(f"Bad/missing frame paths: {bad[:2]}{'...' if len(bad)>2 else ''}")

def load_frames_tensor_from_paths(paths: List[Path], input_size: int = INPUT_SIZE) -> torch.Tensor:
    transform = build_transform(input_size)
    tensors = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Missing frame path: {p}")
        img = Image.open(p)
        tensors.append(transform(img))
    return torch.stack(tensors, dim=0)  # [k,3,H,W]

def _row_key(row: dict) -> str:
    frames = row.get("frame_paths") or row.get("frames") or []
    q = row.get("question") or ""
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
            q = rec.get("question") or rec.get("prompt_given_to_model") or rec.get("prompt") or ""
            first = frames[0] if frames else ""
            last  = frames[-1] if frames else ""
            tex = rec.get("texture") or ""
            sample_type = rec.get("type") or ""
            distr = rec.get("distr", "")
            config = rec.get("config", "")
            done.add(f"f::{first}|{last}||tex::{tex}||type::{sample_type}||distr::{distr}||config::{config}||q::{q}")
    return done

# ================== MODEL LOADING ==================
def _load_model(adapter_dir: str):
    print(f"[internvl2] loading adapter from {adapter_dir} …", flush=True)

    adapter_cfg = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(adapter_cfg):
        raise RuntimeError(
            f"[ERROR] Expected a LoRA adapter folder at:\n"
            f"    {adapter_dir}\n"
            f"but adapter_config.json is missing.\n"
            f"Refusing to load the unfine-tuned base model."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_TOKENIZER, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    base = AutoModel.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=use_dtype,
        device_map="auto",
        attn_implementation="eager",
    )

    raw = json.loads(Path(adapter_cfg).read_text(encoding="utf-8"))
    allowed_keys = set(LoraConfig.__dataclass_fields__.keys())
    clean = {k: v for k, v in raw.items() if k in allowed_keys}
    lc = LoraConfig(**clean)

    model = PeftModel.from_pretrained(base, adapter_dir, config=lc)

    try:
        inner = model.base_model
        if hasattr(inner, "model"):
            inner = inner.model
        if not hasattr(inner, "set_output_embeddings"):
            print("[PATCH] Adding dummy set_output_embeddings to", type(inner), flush=True)
            def _set_output_embeddings(self, new_embeds):
                return
            inner.set_output_embeddings = _set_output_embeddings.__get__(inner, inner.__class__)
    except Exception as e:
        print(f"[WARN] could not patch set_output_embeddings: {e}", flush=True)

    emb_n = model.get_input_embeddings().weight.shape[0]
    vs = len(tokenizer)
    if vs > emb_n:
        print(f"[PATCH] Resizing token embeddings from {emb_n} → {vs}", flush=True)
        model.resize_token_embeddings(vs)
    elif vs < emb_n:
        print(f"[WARN] tokenizer vocab ({vs}) < model embeddings ({emb_n})", flush=True)

    model.eval()
    print("[internvl2] model ready.", flush=True)
    return tokenizer, model

# ================== INFERENCE ==================
def ask_internvl2_8frames(tokenizer, model, frame_paths: List[str], question: str):
    user_text = (
        "You see 8 consecutive frames of a video in temporal order. "
        "Do not explain; just answer the question concisely. "
    ) + (question or "")

    pixel_values = load_frames_tensor_from_paths([Path(p) for p in frame_paths], input_size=INPUT_SIZE)
    k = pixel_values.shape[0]
    prompt = ("<image>\n" * k) + user_text

    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype
    if device.type == "cpu":
        dtype = torch.float32
    pixel_values = pixel_values.to(device=device, dtype=dtype)

    generation_config = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "temperature": 0.0,
    }

    with torch.inference_mode():
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                response = model.chat(tokenizer, pixel_values, prompt, generation_config)
        else:
            response = model.chat(tokenizer, pixel_values, prompt, generation_config)

    return response, prompt, list(map(str, frame_paths))

# ================== EVAL LOOP ==================
def eval_dataset(task_jsonl: Path, out_jsonl: Path, adapter_dir: str,
                 resume: bool,
                 split: str, epochs: int, tag: str, dataset: str,
                 limit: Optional[int] = None):
    print("[internvl2] Starting eval …", flush=True)
    tokenizer, model = _load_model(adapter_dir)

    done_keys = _load_done_keys(out_jsonl) if resume else set()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"

    written = 0
    seen = 0
    correct = 0

    with task_jsonl.open("r", encoding="utf-8") as f_in, out_jsonl.open(mode, encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            if not line.strip():
                continue
            if limit is not None and written >= limit:
                print("[internvl2] Reached global limit; stopping.", flush=True)
                break

            seen += 1
            try:
                row = json.loads(line)

                frames = row.get("frame_paths") or row.get("frames")
                q = row.get("question")
                gt = (row.get("answer") or "").strip().lower()

                if not isinstance(frames, list) or len(frames) != NUM_FRAMES:
                    raise ValueError(f"Row must have 'frame_paths' with exactly {NUM_FRAMES} absolute paths")
                if not isinstance(q, str) or not q.strip():
                    raise ValueError("Row missing 'question'")

                _validate_frame_paths(frames)

                if resume:
                    key = _row_key(row)
                    if key in done_keys:
                        continue

                qid = row.get("qid") if row.get("qid") is not None else row.get("id", f"row{i}")
                cat = row.get("category", "unknown")
                name = row.get("name", "unknown")

                extra = ""
                if dataset == "textures":
                    extra += f" texture={row.get('texture')}"
                if dataset in {"randomized_colors", "colors_new"}:
                    extra += f" distr={row.get('distr')} config={row.get('config')} type={row.get('type')}"
                if dataset == "occluders":
                    extra += f" config={row.get('config')} type={row.get('type')}"

                print(
                    f"[internvl2] split={split} epochs={epochs} dataset={dataset} "
                    f"qid={qid} cat={cat} name={name} k={len(frames)}{extra}",
                    flush=True
                )

                pred, prompt, used_paths = ask_internvl2_8frames(tokenizer, model, frames, q)
                pred_norm = normalize_yesno(pred)

                if gt in {"yes", "no"} and pred_norm == gt:
                    correct += 1

                out_record = {}
                out_record.update(row)
                out_record["frame_paths"] = used_paths
                out_record["prompt_given_to_model"] = prompt
                out_record["model_output_raw"] = pred
                out_record["model_output_norm"] = pred_norm
                out_record["correct"] = (pred_norm == gt) if gt in {"yes", "no"} else None
                out_record["model_dir_used"] = MODEL_DIR
                out_record["adapter_dir_used"] = adapter_dir
                out_record["split"] = split
                out_record["epochs"] = epochs
                out_record["tag"] = tag
                out_record["dataset_kind"] = dataset

                f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                f_out.flush()
                written += 1

                if written % 100 == 0:
                    acc = (correct / written) if written > 0 else 0.0
                    print(f"[internvl2] wrote {written} (seen {seen}) | acc={acc:.4f}", flush=True)

            except Exception as e:
                print(f"[internvl2][ERROR] dataset={dataset} row {i}: {e}", flush=True)

    final_acc = (correct / written) if written > 0 else 0.0
    print(f"[internvl2] Done. Wrote {written} rows to {out_jsonl} | acc={final_acc:.4f}", flush=True)

# ================== ENTRY ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["textures", "colors", "randomized_colors", "colors_new", "occluders"],
                        help="Which dataset to evaluate.")
    parser.add_argument("--split", required=True,
                        choices=["SPLIT1", "SPLIT2", "SPLIT3"],
                        help="Which finetuned split adapter to use.")
    parser.add_argument("--epochs", required=True, type=int,
                        choices=[1, 3, 5],
                        help="How many training epochs the adapter had.")
    parser.add_argument("--resume", action="store_true",
                        help="If set, skip already-processed rows and append new ones.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional global max rows to run.")
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

    adapter_dir, out_dir, out_jsonl, tag = build_dir_paths(args.split, args.epochs, args.dataset)

    print(f"[internvl2] Dataset:     {args.dataset}")
    print(f"[internvl2] Split:       {args.split}")
    print(f"[internvl2] Epochs:      {args.epochs}")
    print(f"[internvl2] Input JSONL:  {task_jsonl}")
    print(f"[internvl2] Adapter dir:  {adapter_dir}")
    print(f"[internvl2] Output JSONL: {out_jsonl}")

    eval_dataset(
        task_jsonl=task_jsonl,
        out_jsonl=out_jsonl,
        adapter_dir=adapter_dir,
        resume=args.resume,
        split=args.split,
        epochs=args.epochs,
        tag=tag,
        dataset=args.dataset,
        limit=args.limit,
    )

# HOW TO RUN:
# python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/inf_finetuned_internvl.py --dataset textures --split SPLIT1 --epochs 3
# python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/inf_finetuned_internvl.py --dataset colors --split SPLIT1 --epochs 3
# python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/inf_finetuned_internvl.py --dataset randomized_colors --split SPLIT1 --epochs 3
# python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/inf_finetuned_internvl.py --dataset colors_new --split SPLIT1 --epochs 3
# python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/inf_finetuned_internvl.py --dataset occluders --split SPLIT1 --epochs 3
#
# Resume:
# python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/inf_finetuned_internvl.py --dataset colors_new --split SPLIT1 --epochs 3 --resume
# python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/inf_finetuned_internvl.py --dataset occluders --split SPLIT1 --epochs 3 --resume