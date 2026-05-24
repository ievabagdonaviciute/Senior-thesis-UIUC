#!/usr/bin/env python3
# import os, json, argparse, re
# from pathlib import Path
# from typing import Optional, Set, List

# import torch
# from PIL import Image
# from transformers import AutoTokenizer, AutoModel
# import torchvision.transforms as T
# from torchvision.transforms.functional import InterpolationMode

# # ================== ENV / CONFIG ==================
# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# MODEL_DIR      = "/home/ievab2/models/InternVL2-8B"
# BASE_TOKENIZER = MODEL_DIR

# # --------- DATASETS (INPUT) ---------
# TEXTURE_JSONL = Path("/shared/rsaas/ievab2/PHYSION_TEXTURES/texture_dataset.jsonl")
# COLOR_JSONL   = Path("/shared/rsaas/ievab2/PHYSION_COLORS/color_dataset.jsonl")
# RANDOMIZED_COLORS_JSONL = Path("/shared/rsaas/ievab2/RANDOMIZED_COLORS/randomized_colors_dataset.jsonl")

# # --------- OUTPUTS (FIXED PATHS) ---------
# OUT_ROOT = Path("/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/results")
# OUT_TEXTURE = OUT_ROOT / "texture" / "base_results.jsonl"
# OUT_COLOR   = OUT_ROOT / "colors"  / "base_results.jsonl"
# OUT_RANDOMIZED_COLORS = OUT_ROOT / "randomized_colors" / "base_results.jsonl"

# MAX_NEW_TOKENS = 128
# INPUT_SIZE     = 448
# N_FRAMES_EXPECTED = 8

# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD  = (0.229, 0.224, 0.225)

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

# def build_transform(input_size: int):
#     return T.Compose([
#         T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
#         T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
#         T.ToTensor(),
#         T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#     ])

# def _validate_frame_paths(paths: List[str], expect_n: int = N_FRAMES_EXPECTED):
#     if not isinstance(paths, list) or len(paths) == 0:
#         raise ValueError("Expected non-empty list of frame paths")
#     if expect_n is not None and len(paths) != expect_n:
#         raise ValueError(f"Expected exactly {expect_n} frames, got {len(paths)}")
#     bad = [p for p in paths if (not p) or (not os.path.isabs(p)) or (not os.path.exists(p))]
#     if bad:
#         raise FileNotFoundError(f"Bad/missing frame paths: {bad[:3]}{'...' if len(bad)>3 else ''}")

# def load_frames_tensor_from_paths(paths: List[Path], input_size: int = INPUT_SIZE) -> torch.Tensor:
#     transform = build_transform(input_size)
#     tensors = []
#     for p in paths:
#         p = Path(p)
#         if not p.exists():
#             raise FileNotFoundError(f"Missing frame path: {p}")
#         img = Image.open(p)
#         tensors.append(transform(img))
#     return torch.stack(tensors, dim=0)

# def _row_key(row: dict) -> str:
#     frames = row.get("frame_paths") or row.get("frames") or []
#     q = row.get("question") or ""
#     first = frames[0] if frames else ""
#     last  = frames[-1] if frames else ""
#     tex = row.get("texture") or ""
#     sample_type = row.get("type") or ""
#     distr = row.get("distr", "")
#     config = row.get("config", "")
#     return f"f::{first}|{last}||tex::{tex}||type::{sample_type}||distr::{distr}||config::{config}||q::{q}"

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
#             sample_type = rec.get("type") or ""
#             distr = rec.get("distr", "")
#             config = rec.get("config", "")
#             done.add(f"f::{first}|{last}||tex::{tex}||type::{sample_type}||distr::{distr}||config::{config}||q::{q}")
#     return done

# # ================== MODEL LOADING ==================
# def _load_model():
#     print(f"[internvl2] loading base model from {MODEL_DIR} …", flush=True)

#     tokenizer = AutoTokenizer.from_pretrained(
#         BASE_TOKENIZER, trust_remote_code=True, local_files_only=True, use_fast=False
#     )
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     use_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

#     model = AutoModel.from_pretrained(
#         MODEL_DIR,
#         trust_remote_code=True,
#         local_files_only=True,
#         torch_dtype=use_dtype,
#         device_map="auto",
#         attn_implementation="eager",
#     )

#     emb_n = model.get_input_embeddings().weight.shape[0]
#     vs = len(tokenizer)
#     if vs > emb_n:
#         print(f"[PATCH] Resizing token embeddings from {emb_n} → {vs}")
#         model.resize_token_embeddings(vs)
#     elif vs < emb_n:
#         print(f"[WARN] tokenizer vocab ({vs}) < model embeddings ({emb_n})")

#     model.eval()
#     print("[internvl2] base model ready.")
#     return tokenizer, model

# # ================== DEBUG VOCAB ==================
# def _debug_vocab(tokenizer, model, prompt: str, k: int):
#     img_cnt = prompt.count("<image>")
#     assert img_cnt == k, f"mismatch: found {img_cnt} <image> tokens but k={k}"

#     enc = tokenizer(prompt, add_special_tokens=False, return_tensors=None)
#     ids = enc["input_ids"]
#     max_id = max(ids) if ids else -1
#     emb_n = model.get_input_embeddings().weight.shape[0]
#     print(f"[DBG] vocab_n={emb_n}  max_token_id_in_prompt={max_id}  n_tokens={len(ids)}")
#     if max_id >= emb_n:
#         raise RuntimeError(f"Token id {max_id} >= vocab size {emb_n} (embedding OOB).")

#     image_id = tokenizer.convert_tokens_to_ids("<image>")
#     print(f"[DBG] '<image>' id: {image_id} (OK if UNK)")

# # ================== INFERENCE ==================
# def ask_internvl2(tokenizer, model, frame_paths: List[str], question: str):
#     user_text = (
#         "You see 8 consecutive frames of a video in temporal order. "
#         "Answer only yes or no. "
#     ) + (question or "")

#     pixel_values = load_frames_tensor_from_paths([Path(p) for p in frame_paths], input_size=INPUT_SIZE)
#     k = pixel_values.shape[0]
#     prompt = ("<image>\n" * k) + user_text

#     _debug_vocab(tokenizer, model, prompt, k)

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
#         response = model.chat(tokenizer, pixel_values, prompt, generation_config)

#     return response, prompt, list(map(str, frame_paths))

# # ================== MAIN EVAL LOOP ==================
# def eval_dataset(in_jsonl: Path, out_path: Path, dataset_kind: str, resume: bool = False, limit: Optional[int] = None):
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     tokenizer, model = _load_model()

#     done_keys = _load_done_keys(out_path) if resume else set()
#     mode = "a" if resume else "w"

#     written = 0
#     seen = 0

#     with out_path.open(mode, encoding="utf-8") as f_out:
#         if not in_jsonl.exists():
#             raise FileNotFoundError(f"Missing input JSONL: {in_jsonl}")

#         print(f"[internvl2] Evaluating dataset={dataset_kind} from {in_jsonl}", flush=True)
#         print(f"[internvl2] Writing to {out_path} (mode={mode})", flush=True)

#         with in_jsonl.open("r", encoding="utf-8") as f_in:
#             for i, line in enumerate(f_in):
#                 if not line.strip():
#                     continue
#                 if limit is not None and written >= limit:
#                     print("[internvl2] Reached global limit; stopping.")
#                     return

#                 seen += 1
#                 try:
#                     row = json.loads(line)

#                     frames = row.get("frame_paths") or row.get("frames")
#                     q = row.get("question")
#                     if not isinstance(frames, list) or len(frames) == 0:
#                         raise ValueError("Row must have 'frame_paths' (or 'frames') with >= 1 image path")
#                     if not q:
#                         raise ValueError("Row missing 'question'")

#                     _validate_frame_paths(frames, expect_n=N_FRAMES_EXPECTED)

#                     if resume:
#                         key = _row_key(row)
#                         if key in done_keys:
#                             continue

#                     qid = row.get("qid") if row.get("qid") is not None else row.get("id", f"row{i}")
#                     cat = row.get("category", "unknown")
#                     name = row.get("name", "unknown")

#                     extra = ""
#                     if dataset_kind == "textures":
#                         extra += f" texture={row.get('texture')}"
#                     if dataset_kind == "randomized_colors":
#                         extra += f" distr={row.get('distr')} config={row.get('config')} type={row.get('type')}"

#                     print(f"[internvl2] {dataset_kind} qid={qid} cat={cat} name={name} k={len(frames)}{extra}", flush=True)

#                     pred, prompt, used_paths = ask_internvl2(tokenizer, model, frames, q)

#                     out_record = {}
#                     out_record.update(row)

#                     out_record["frame_paths"] = used_paths
#                     out_record["prompt_given_to_model"] = prompt
#                     out_record["model_output_raw"] = pred
#                     out_record["model_output_norm"] = normalize_yesno(pred)
#                     out_record["model_dir_used"] = MODEL_DIR
#                     out_record["dataset_kind"] = dataset_kind

#                     f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
#                     f_out.flush()
#                     written += 1
#                     if written % 100 == 0:
#                         print(f"[internvl2] wrote {written} (seen {seen})", flush=True)

#                 except Exception as e:
#                     print(f"[internvl2][ERROR] {dataset_kind} row {i}: {e}", flush=True)

#     print(f"[internvl2] Done. Wrote {written} rows to {out_path}", flush=True)

# # ================== ENTRY ==================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", required=True, choices=["textures", "colors", "randomized_colors"],
#                         help="Which dataset to evaluate.")
#     parser.add_argument("--resume", action="store_true",
#                         help="If set, skip already-processed rows and append new ones.")
#     parser.add_argument("--limit", type=int, default=None,
#                         help="Optional global max rows to run.")
#     args = parser.parse_args()

#     if args.dataset == "textures":
#         in_jsonl = TEXTURE_JSONL
#         out_path = OUT_TEXTURE
#     elif args.dataset == "colors":
#         in_jsonl = COLOR_JSONL
#         out_path = OUT_COLOR
#     else:
#         in_jsonl = RANDOMIZED_COLORS_JSONL
#         out_path = OUT_RANDOMIZED_COLORS

#     print(f"[internvl2] Dataset:     {args.dataset}")
#     print(f"[internvl2] Input JSONL:  {in_jsonl}")
#     print(f"[internvl2] Output JSONL: {out_path}")

#     eval_dataset(in_jsonl, out_path, dataset_kind=args.dataset, resume=args.resume, limit=args.limit)

# # HOW TO RUN:
# # python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/inf_base_internvl.py --dataset textures
# # python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/inf_base_internvl.py --dataset colors
# # python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/inf_base_internvl.py --dataset randomized_colors
# #
# # Resume:
# # python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/inf_base_internvl.py --dataset randomized_colors --resume

#!/usr/bin/env python3
import os, json, argparse, re
from pathlib import Path
from typing import Optional, Set, List

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# ================== ENV / CONFIG ==================
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_DIR      = "/home/ievab2/models/InternVL2-8B"
BASE_TOKENIZER = MODEL_DIR

# --------- DATASETS (INPUT) ---------
TEXTURE_JSONL = Path("/shared/rsaas/ievab2/PHYSION_TEXTURES/texture_dataset.jsonl")
COLOR_JSONL   = Path("/shared/rsaas/ievab2/PHYSION_COLORS/color_dataset.jsonl")
RANDOMIZED_COLORS_JSONL = Path("/shared/rsaas/ievab2/RANDOMIZED_COLORS/randomized_colors_dataset.jsonl")
COLORS_NEW_JSONL = Path("/shared/rsaas/ievab2/RANDOMIZED_COLORS_NEW/randomized_colors_new_dataset.jsonl")
OCCLUDERS_JSONL = Path("/shared/rsaas/ievab2/OCCLUDER_TEST/occluder_dataset.jsonl")

# --------- OUTPUTS (FIXED PATHS) ---------
OUT_ROOT = Path("/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/results")
OUT_TEXTURE = OUT_ROOT / "texture" / "base_results.jsonl"
OUT_COLOR   = OUT_ROOT / "colors"  / "base_results.jsonl"
OUT_RANDOMIZED_COLORS = OUT_ROOT / "randomized_colors" / "base_results.jsonl"
OUT_COLORS_NEW = OUT_ROOT / "colors_new" / "base_results.jsonl"
OUT_OCCLUDERS = OUT_ROOT / "occluders" / "base_results.jsonl"

MAX_NEW_TOKENS = 128
INPUT_SIZE     = 448
N_FRAMES_EXPECTED = 8

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

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

def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def _validate_frame_paths(paths: List[str], expect_n: int = N_FRAMES_EXPECTED):
    if not isinstance(paths, list) or len(paths) == 0:
        raise ValueError("Expected non-empty list of frame paths")
    if expect_n is not None and len(paths) != expect_n:
        raise ValueError(f"Expected exactly {expect_n} frames, got {len(paths)}")
    bad = [p for p in paths if (not p) or (not os.path.isabs(p)) or (not os.path.exists(p))]
    if bad:
        raise FileNotFoundError(f"Bad/missing frame paths: {bad[:3]}{'...' if len(bad)>3 else ''}")

def load_frames_tensor_from_paths(paths: List[Path], input_size: int = INPUT_SIZE) -> torch.Tensor:
    transform = build_transform(input_size)
    tensors = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Missing frame path: {p}")
        img = Image.open(p)
        tensors.append(transform(img))
    return torch.stack(tensors, dim=0)

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
            q = rec.get("question") or rec.get("prompt") or rec.get("prompt_given_to_model") or ""
            first = frames[0] if frames else ""
            last  = frames[-1] if frames else ""
            tex = rec.get("texture") or ""
            sample_type = rec.get("type") or ""
            distr = rec.get("distr", "")
            config = rec.get("config", "")
            done.add(f"f::{first}|{last}||tex::{tex}||type::{sample_type}||distr::{distr}||config::{config}||q::{q}")
    return done

# ================== MODEL LOADING ==================
def _load_model():
    print(f"[internvl2] loading base model from {MODEL_DIR} …", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_TOKENIZER, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModel.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=use_dtype,
        device_map="auto",
        attn_implementation="eager",
    )

    emb_n = model.get_input_embeddings().weight.shape[0]
    vs = len(tokenizer)
    if vs > emb_n:
        print(f"[PATCH] Resizing token embeddings from {emb_n} → {vs}")
        model.resize_token_embeddings(vs)
    elif vs < emb_n:
        print(f"[WARN] tokenizer vocab ({vs}) < model embeddings ({emb_n})")

    model.eval()
    print("[internvl2] base model ready.")
    return tokenizer, model

# ================== DEBUG VOCAB ==================
def _debug_vocab(tokenizer, model, prompt: str, k: int):
    img_cnt = prompt.count("<image>")
    assert img_cnt == k, f"mismatch: found {img_cnt} <image> tokens but k={k}"

    enc = tokenizer(prompt, add_special_tokens=False, return_tensors=None)
    ids = enc["input_ids"]
    max_id = max(ids) if ids else -1
    emb_n = model.get_input_embeddings().weight.shape[0]
    print(f"[DBG] vocab_n={emb_n}  max_token_id_in_prompt={max_id}  n_tokens={len(ids)}")
    if max_id >= emb_n:
        raise RuntimeError(f"Token id {max_id} >= vocab size {emb_n} (embedding OOB).")

    image_id = tokenizer.convert_tokens_to_ids("<image>")
    print(f"[DBG] '<image>' id: {image_id} (OK if UNK)")

# ================== INFERENCE ==================
def ask_internvl2(tokenizer, model, frame_paths: List[str], question: str):
    user_text = (
        "You see 8 consecutive frames of a video in temporal order. "
        "Answer only yes or no. "
    ) + (question or "")

    pixel_values = load_frames_tensor_from_paths([Path(p) for p in frame_paths], input_size=INPUT_SIZE)
    k = pixel_values.shape[0]
    prompt = ("<image>\n" * k) + user_text

    _debug_vocab(tokenizer, model, prompt, k)

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
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)

    return response, prompt, list(map(str, frame_paths))

# ================== MAIN EVAL LOOP ==================
def eval_dataset(in_jsonl: Path, out_path: Path, dataset_kind: str, resume: bool = False, limit: Optional[int] = None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer, model = _load_model()

    done_keys = _load_done_keys(out_path) if resume else set()
    mode = "a" if resume else "w"

    written = 0
    seen = 0

    with out_path.open(mode, encoding="utf-8") as f_out:
        if not in_jsonl.exists():
            raise FileNotFoundError(f"Missing input JSONL: {in_jsonl}")

        print(f"[internvl2] Evaluating dataset={dataset_kind} from {in_jsonl}", flush=True)
        print(f"[internvl2] Writing to {out_path} (mode={mode})", flush=True)

        with in_jsonl.open("r", encoding="utf-8") as f_in:
            for i, line in enumerate(f_in):
                if not line.strip():
                    continue
                if limit is not None and written >= limit:
                    print("[internvl2] Reached global limit; stopping.")
                    return

                seen += 1
                try:
                    row = json.loads(line)

                    frames = row.get("frame_paths") or row.get("frames")
                    q = row.get("question")
                    if not isinstance(frames, list) or len(frames) == 0:
                        raise ValueError("Row must have 'frame_paths' (or 'frames') with >= 1 image path")
                    if not q:
                        raise ValueError("Row missing 'question'")

                    _validate_frame_paths(frames, expect_n=N_FRAMES_EXPECTED)

                    if resume:
                        key = _row_key(row)
                        if key in done_keys:
                            continue

                    qid = row.get("qid") if row.get("qid") is not None else row.get("id", f"row{i}")
                    cat = row.get("category", "unknown")
                    name = row.get("name", "unknown")

                    extra = ""
                    if dataset_kind == "textures":
                        extra += f" texture={row.get('texture')}"
                    if dataset_kind in {"randomized_colors", "colors_new"}:
                        extra += f" distr={row.get('distr')} config={row.get('config')} type={row.get('type')}"
                    if dataset_kind == "occluders":
                        extra += f" config={row.get('config')} type={row.get('type')}"

                    print(f"[internvl2] {dataset_kind} qid={qid} cat={cat} name={name} k={len(frames)}{extra}", flush=True)

                    pred, prompt, used_paths = ask_internvl2(tokenizer, model, frames, q)

                    out_record = {}
                    out_record.update(row)

                    out_record["frame_paths"] = used_paths
                    out_record["prompt_given_to_model"] = prompt
                    out_record["model_output_raw"] = pred
                    out_record["model_output_norm"] = normalize_yesno(pred)
                    out_record["model_dir_used"] = MODEL_DIR
                    out_record["dataset_kind"] = dataset_kind

                    f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                    f_out.flush()
                    written += 1
                    if written % 100 == 0:
                        print(f"[internvl2] wrote {written} (seen {seen})", flush=True)

                except Exception as e:
                    print(f"[internvl2][ERROR] {dataset_kind} row {i}: {e}", flush=True)

    print(f"[internvl2] Done. Wrote {written} rows to {out_path}", flush=True)

# ================== ENTRY ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["textures", "colors", "randomized_colors", "colors_new", "occluders"],
        help="Which dataset to evaluate."
    )
    parser.add_argument("--resume", action="store_true",
                        help="If set, skip already-processed rows and append new ones.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional global max rows to run.")
    args = parser.parse_args()

    if args.dataset == "textures":
        in_jsonl = TEXTURE_JSONL
        out_path = OUT_TEXTURE
    elif args.dataset == "colors":
        in_jsonl = COLOR_JSONL
        out_path = OUT_COLOR
    elif args.dataset == "randomized_colors":
        in_jsonl = RANDOMIZED_COLORS_JSONL
        out_path = OUT_RANDOMIZED_COLORS
    elif args.dataset == "colors_new":
        in_jsonl = COLORS_NEW_JSONL
        out_path = OUT_COLORS_NEW
    else:
        in_jsonl = OCCLUDERS_JSONL
        out_path = OUT_OCCLUDERS

    print(f"[internvl2] Dataset:     {args.dataset}")
    print(f"[internvl2] Input JSONL:  {in_jsonl}")
    print(f"[internvl2] Output JSONL: {out_path}")

    eval_dataset(in_jsonl, out_path, dataset_kind=args.dataset, resume=args.resume, limit=args.limit)

# HOW TO RUN:
# python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/inf_base_internvl.py --dataset textures
# python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/inf_base_internvl.py --dataset colors
# python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/inf_base_internvl.py --dataset randomized_colors
# python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/inf_base_internvl.py --dataset colors_new
# python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/inf_base_internvl.py --dataset occluders
#
# Resume:
# python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/inf_base_internvl.py --dataset colors_new --resume
# python3 /home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/inf_base_internvl.py --dataset occluders --resume