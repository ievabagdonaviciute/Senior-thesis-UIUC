#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path
from typing import Optional, Set, List
import re

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from peft import PeftModel, LoraConfig

# =========================
# ENV / CACHE SETTINGS
# =========================
os.environ.setdefault("HF_HOME", "/shared/rsaas/ievab2/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/shared/rsaas/ievab2/hf_cache/hub")
os.environ.setdefault("TORCH_HOME", "/shared/rsaas/ievab2/hf_cache/torch")

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# =========================
# BASE MODEL CONFIG
# =========================
BASE_MODEL_DIR   = "/home/ievab2/models/InternVL2-8B"
BASE_TOKENIZER   = "/home/ievab2/models/InternVL2-8B"

# Your finetuned adapter root (InternVL)
ADAPTER_ROOT = "/shared/rsaas/ievab2/FULL_PHYSION_checkpoints/internvl/round0/MULTIVIEW4"

MAX_NEW_TOKENS = 128
INPUT_SIZE     = 448

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ---------- AUX dataset inputs ----------
AUX_CONTACT_JSONL  = "/shared/rsaas/ievab2/PHYSION_MULTIVIEW_4PAIRS/contact_dataset.jsonl"
AUX_GEOMETRY_JSONL = "/shared/rsaas/ievab2/PHYSION_MULTIVIEW_4PAIRS/geometry_dataset.jsonl"
AUX_TIME_JSONL     = "/shared/rsaas/ievab2/PHYSION_MULTIVIEW_4PAIRS/time_dataset.jsonl"

# ---------- AUX outputs (fixed base dir) ----------
OUT_ROOT = Path("/home/ievab2/run_models/PHYSION_MULTIVIEW4PAIRS_AUXILIARY_TRAIN/text_on_multiviewGTC/internvl/results/finetuned")

# ---------- helpers ----------
_NEG_RE = re.compile(r"\b(?:no|false|not|won't|will not|doesn't|does not|don't|didn't|did not|can't|cannot)\b", re.I)
_POS_RE = re.compile(r"\b(?:yes|true|yep|yeah|y)\b", re.I)

def normalize_yesno(text: str) -> str:
    t = (text or "").strip().lower()
    if t.startswith("yes"): return "yes"
    if t.startswith("no"):  return "no"
    if _NEG_RE.search(t): return "no"
    if _POS_RE.search(t): return "yes"
    return "unknown"

def normalize_time12(text: str) -> str:
    t = (text or "").strip().lower()
    for ch in t:
        if ch in ("1", "2"):
            return ch
    return "unknown"

def _validate_frame_paths(paths: List[str]):
    bad = [p for p in paths if not p or not os.path.isabs(p) or not os.path.exists(p)]
    if bad:
        raise FileNotFoundError(f"Bad/missing frame paths: {bad[:2]}{'...' if len(bad)>2 else ''}")

def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def _row_key(row: dict) -> str:
    frames = row.get("frames") or row.get("frame_paths") or []
    q = row.get("question") or ""
    first = frames[0] if frames else ""
    last  = frames[-1] if frames else ""
    return f"f::{first}|{last}||q::{q}"

def _load_done_keys(out_path: str) -> Set[str]:
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r", encoding="utf-8") as f:
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
            done.add(f"f::{first}|{last}||q::{q}")
    return done

def load_frames_tensor_from_paths(paths: List[Path], input_size: int = INPUT_SIZE) -> torch.Tensor:
    # NOTE: variable number of images supported
    if not isinstance(paths, list) or len(paths) == 0:
        raise ValueError("Expected non-empty list of frame paths")
    transform = build_transform(input_size)
    tensors = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Missing frame path: {p}")
        img = Image.open(p)
        tensors.append(transform(img))
    return torch.stack(tensors, dim=0)

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

def build_adapter_dir(adapter_type: str, split_num: int, epochs: int, frames_tag: str = "4frames") -> str:
    # Your naming: {TYPE}_SPLIT{split}_{epochs}epochs_4frames
    d = f"{adapter_type}_SPLIT{split_num}_{epochs}epochs_{frames_tag}"
    return str(Path(ADAPTER_ROOT) / d)

# =========================
# MODEL LOADING (adapter-only)
# =========================
def _load_model(adapter_dir: str):
    print(f"[internvl2] loading adapter from {adapter_dir} …", flush=True)

    adapter_cfg = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(adapter_cfg):
        raise RuntimeError(
            f"[ERROR] Expected a LoRA adapter folder at:\n"
            f"    {adapter_dir}\n"
            f"but adapter_config.json is missing."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_TOKENIZER, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    base = AutoModel.from_pretrained(
        BASE_MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=use_dtype,
        device_map="auto",
        attn_implementation="eager",
    )

    # ---- read + sanitize adapter_config.json for LoraConfig ----
    with open(adapter_cfg, "r", encoding="utf-8") as f:
        raw = json.load(f)

    allowed_keys = set(LoraConfig.__dataclass_fields__.keys())
    clean = {k: v for k, v in raw.items() if k in allowed_keys}
    dropped = sorted(set(raw.keys()) - set(clean.keys()))

    print("[internvl2] adapter_config keys:", sorted(raw.keys()))
    print("[internvl2] using LoraConfig keys:", sorted(clean.keys()))
    print("[internvl2] dropping keys:", dropped)

    lc = LoraConfig(**clean)
    model = PeftModel.from_pretrained(base, adapter_dir, config=lc)

    # ---- monkey-patch missing set_output_embeddings on inner chat model ----
    try:
        inner = model.base_model
        if hasattr(inner, "model"):
            inner = inner.model

        if not hasattr(inner, "set_output_embeddings"):
            print("[PATCH] Adding dummy set_output_embeddings to", type(inner))
            def _set_output_embeddings(self, new_embeds):
                return
            inner.set_output_embeddings = _set_output_embeddings.__get__(inner, inner.__class__)
    except Exception as e:
        print(f"[WARN] could not patch set_output_embeddings: {e}")

    # ---- ensure embeddings match tokenizer size ----
    emb_n = model.get_input_embeddings().weight.shape[0]
    vs = len(tokenizer)
    if vs > emb_n:
        print(f"[PATCH] Resizing token embeddings from {emb_n} → {vs}")
        model.resize_token_embeddings(vs)
    elif vs < emb_n:
        print(f"[WARN] tokenizer vocab ({vs}) < model embeddings ({emb_n})")

    model.eval()
    print("[internvl2] ready")
    return tokenizer, model

# =========================
# ASK
# =========================
def ask_internvl2(tokenizer, model, frame_paths: List[str], question: str, dataset_kind: str) -> tuple[str, str, List[str]]:
    # UPDATED: do not assume 8, do not assume consecutive; special text for time.
    if dataset_kind == "time":
        user_text = (
            "You are shown two images that are frames from the same video. "
            "Do not explain; answer with 1 or 2 only. "
        ) + (question or "")
    elif dataset_kind == "contact":
        user_text = (
            "You are shown an image. "
            "Do not explain; just answer the question concisely. "
        ) + (question or "")
    elif dataset_kind == "geometry":
        user_text = (
            "You are shown two images. "
            "Do not explain; just answer the question concisely. "
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
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                response = model.chat(tokenizer, pixel_values, prompt, generation_config)
        else:
            response = model.chat(tokenizer, pixel_values, prompt, generation_config)

    return response, prompt, list(map(str, frame_paths))

# =========================
# EVAL LOOP (AUX DATASETS)
# =========================
def eval_task(task_path: str, out_path: str, adapter_dir: str, dataset_kind: str,
              counter_limit: Optional[int] = None, resume: bool = False):
    print("[internvl2] Starting task …", flush=True)
    tokenizer, model = _load_model(adapter_dir)

    done_keys = _load_done_keys(out_path) if resume else set()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"
    written = 0

    with open(task_path, "r", encoding="utf-8") as f_in, open(out_path, mode, encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            if not line.strip():
                continue
            if counter_limit is not None and written >= counter_limit:
                break

            try:
                row = json.loads(line)

                frames = row.get("frames") or row.get("frame_paths")
                q = row.get("question")

                if not isinstance(frames, list) or len(frames) == 0:
                    raise ValueError("Row must have 'frames' or 'frame_paths' with >= 1 image path")
                if not q:
                    raise ValueError("Row missing 'question'")

                _validate_frame_paths(frames)

                if resume:
                    key = _row_key(row)
                    if key in done_keys:
                        continue

                qid = row.get("id") or row.get("qid") or f"row{i}"
                cat = row.get("category", "unknown")
                print(f"[internvl2] {dataset_kind} qid={qid} cat={cat} k={len(frames)} first={frames[0]} last={frames[-1]}", flush=True)

                pred, prompt, used_paths = ask_internvl2(tokenizer, model, frames, q, dataset_kind=dataset_kind)

                if dataset_kind == "time":
                    pr = normalize_time12(pred)
                else:
                    pr = normalize_yesno(pred)

                out_record = {}
                out_record["qid"] = row.get("qid") or row.get("id") or qid
                if "category" in row:
                    out_record["category"] = row.get("category")
                if "name" in row:
                    out_record["name"] = row["name"]

                for k, v in row.items():
                    if k in ("qid", "id", "category", "name"):
                        continue
                    out_record[k] = v

                out_record["frame_paths"] = used_paths
                out_record["prompt_given_to_model"] = prompt
                out_record["model_output_raw"] = pred
                out_record["model_output_norm"] = pr
                out_record["adapter_dir_used"] = adapter_dir

                f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                if resume:
                    done_keys.add(_row_key(row))

                f_out.flush()
                written += 1
                print(f"[internvl2] wrote {written}", flush=True)

            except Exception as e:
                print(f"[internvl2][ERROR] row {i}: {e}", flush=True)

    print(f"[internvl2] Done. Wrote {written} rows to {out_path}", flush=True)

# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # adapter selection
    parser.add_argument("--type", required=True, choices=["G","T","C","GC","GT","TC","GTC"],
                        help="Which adapter type to load (matches your MULTIVIEW4 naming).")
    parser.add_argument("--split", type=int, required=True, choices=[1,2,3],
                        help="Which split adapter to load (SPLIT1/2/3).")
    parser.add_argument("--epochs", type=int, required=True, choices=[1,3,5],
                        help="Which epoch folder to load (1/3/5).")
    parser.add_argument("--frames_tag", type=str, default="4frames",
                        help="Folder suffix, default matches your training folders: '4frames'.")

    # dataset selection (NEW)
    parser.add_argument("--dataset", required=True, choices=["geometry", "time", "contact"],
                        help="Which AUX dataset to evaluate.")

    # eval controls
    parser.add_argument("--resume", action="store_true",
                        help="Skip entries already in OUT_JSONL and append new results")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional max rows to run")

    args = parser.parse_args()

    adapter_dir = build_adapter_dir(args.type, args.split, args.epochs, frames_tag=args.frames_tag)

    # pick AUX dataset jsonl
    if args.dataset == "geometry":
        task_jsonl = AUX_GEOMETRY_JSONL
    elif args.dataset == "time":
        task_jsonl = AUX_TIME_JSONL
    else:
        task_jsonl = AUX_CONTACT_JSONL

    # output name exactly as requested:
    #   {TYPE}_SPLIT{split}_results_{dataset}.jsonl
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_jsonl = OUT_ROOT / f"{args.type}_SPLIT{args.split}_results_{args.dataset}.jsonl"

    print(f"[internvl2] dataset     = {args.dataset}")
    print(f"[internvl2] adapter_dir  = {adapter_dir}")
    print(f"[internvl2] task_jsonl   = {task_jsonl}")
    print(f"[internvl2] out_jsonl    = {out_jsonl}")

    eval_task(
        task_jsonl,
        str(out_jsonl),
        adapter_dir=adapter_dir,
        dataset_kind=args.dataset,
        counter_limit=args.limit,
        resume=args.resume,
    )

# HOW TO RUN:
# python /home/ievab2/run_models/PHYSION_MULTIVIEW4PAIRS_AUXILIARY_TRAIN/text_on_multiviewGTC/internvl/eval_finetuned_internvl_aux.py --type GTC --split 1 --epochs 3 --dataset time --resume