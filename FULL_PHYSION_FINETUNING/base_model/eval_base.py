# #!/usr/bin/env python3
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

# ROOT_JSONL = Path("/shared/rsaas/ievab2/Physion_full_readout_training")
# OUT_JSONL  = Path("/home/ievab2/run_models/FULL_PHYSION_FINETUNING/base_model/base_results.jsonl")

# CATEGORIES = [
#     "Collide",
#     "Contain",
#     "Dominoes",
#     "Drape",
#     "Drop",
#     "Link",
#     "Roll",
#     "Support",
# ]

# NUM_FRAMES     = 8
# MAX_NEW_TOKENS = 128
# INPUT_SIZE     = 448

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
#         T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
#         T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
#         T.ToTensor(),
#         T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#     ])

# def _validate_frame_paths(paths: List[str]):
#     bad = [p for p in paths if not p or not os.path.isabs(p) or not os.path.exists(p)]
#     if bad:
#         raise FileNotFoundError(f"Bad/missing frame paths: {bad[:3]}{'...' if len(bad)>3 else ''}")

# def load_frames_tensor_from_paths(paths: List[Path], input_size: int = INPUT_SIZE) -> torch.Tensor:
#     if len(paths) != NUM_FRAMES:
#         raise ValueError(f"Expected {NUM_FRAMES} frames, got {len(paths)}")
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
#     frames = row.get("frames") or row.get("frame_paths") or []
#     q = row.get("question") or ""
#     first = frames[0] if frames else ""
#     last  = frames[-1] if frames else ""
#     return f"f::{first}|{last}||q::{q}"

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
#             done.add(f"f::{first}|{last}||q::{q}")
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
#         "Do not explain; just answer the question concisely. "
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
# def eval_all_categories(out_path: Path, resume: bool = False, limit: Optional[int] = None):
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     tokenizer, model = _load_model()

#     done_keys = _load_done_keys(out_path) if resume else set()
#     mode = "a" if resume else "w"

#     written = 0
#     with out_path.open(mode, encoding="utf-8") as f_out:
#         for cat in CATEGORIES:
#             cat_dir = ROOT_JSONL / cat
#             cat_file = cat_dir / f"{cat.lower()}_pred.jsonl"
#             if not cat_file.exists():
#                 print(f"[WARN] Missing JSONL for category {cat}: {cat_file}")
#                 continue

#             print(f"[internvl2] Evaluating category={cat} from {cat_file}", flush=True)

#             with cat_file.open("r", encoding="utf-8") as f_in:
#                 for i, line in enumerate(f_in):
#                     if not line.strip():
#                         continue
#                     if limit is not None and written >= limit:
#                         print("[internvl2] Reached global limit; stopping.")
#                         return

#                     try:
#                         row = json.loads(line)

#                         frames = row.get("frames") or row.get("frame_paths")
#                         q = row.get("question")
#                         if not isinstance(frames, list) or len(frames) != NUM_FRAMES:
#                             raise ValueError("Row must have 'frames' or 'frame_paths' with exactly 8 image paths")
#                         if not q:
#                             raise ValueError("Row missing 'question'")

#                         _validate_frame_paths(frames)

#                         if resume:
#                             key = _row_key(row)
#                             if key in done_keys:
#                                 continue

#                         qid = row.get("id") or row.get("qid") or f"{cat}_row{i}"
#                         print(f"[internvl2] {cat} qid={qid} first={frames[0]} last={frames[-1]}", flush=True)

#                         pred, prompt, used_paths = ask_internvl2(tokenizer, model, frames, q)

#                         out_record = {"category": cat}
#                         out_record.update(row)

#                         out_record["frame_paths"] = used_paths
#                         out_record["prompt_given_to_model"] = prompt
#                         out_record["model_output_raw"] = pred
#                         out_record["model_output_norm"] = normalize_yesno(pred)
#                         out_record["model_dir_used"] = MODEL_DIR

#                         f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
#                         f_out.flush()
#                         written += 1
#                         print(f"[internvl2] wrote {written}", flush=True)

#                     except Exception as e:
#                         print(f"[internvl2][ERROR] {cat} row {i}: {e}", flush=True)

#     print(f"[internvl2] Done. Wrote {written} rows to {out_path}", flush=True)

# # ================== ENTRY ==================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--resume", action="store_true",
#                         help="If set, skip already-processed rows and append new ones.")
#     parser.add_argument("--limit", type=int, default=None,
#                         help="Optional global max rows to run (across all categories).")
#     args = parser.parse_args()

#     print(f"[internvl2] Output JSONL: {OUT_JSONL}")
#     eval_all_categories(OUT_JSONL, resume=args.resume, limit=args.limit)


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

ROOT_JSONL_OFFICIAL = Path("/shared/rsaas/ievab2/Physion_full_readout_training")
OUT_JSONL_OFFICIAL  = Path("/home/ievab2/run_models/FULL_PHYSION_FINETUNING/base_model/base_results.jsonl")

ROOT_JSONL_MY = Path("/shared/rsaas/ievab2/my_own_physion_preprocessed")
OUT_DIR_MY    = Path("/home/ievab2/run_models/FULL_PHYSION_FINETUNING/base_model/testing_on_my_own_physion")
OUT_JSONL_MY  = OUT_DIR_MY / "base_results.jsonl"
ACC_JSONL_MY  = OUT_DIR_MY / "base_accuracy.jsonl"

CATEGORIES = [
    "Collide",
    "Contain",
    "Dominoes",
    "Drape",
    "Drop",
    "Link",
    "Roll",
    "Support",
]

NUM_FRAMES     = 8
MAX_NEW_TOKENS = 128
INPUT_SIZE     = 448

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
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def _validate_frame_paths(paths: List[str]):
    bad = [p for p in paths if not p or not os.path.isabs(p) or not os.path.exists(p)]
    if bad:
        raise FileNotFoundError(f"Bad/missing frame paths: {bad[:3]}{'...' if len(bad)>3 else ''}")

def load_frames_tensor_from_paths(paths: List[Path], input_size: int = INPUT_SIZE) -> torch.Tensor:
    if len(paths) != NUM_FRAMES:
        raise ValueError(f"Expected {NUM_FRAMES} frames, got {len(paths)}")
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
    frames = row.get("frames") or row.get("frame_paths") or []
    q = row.get("question") or ""
    first = frames[0] if frames else ""
    last  = frames[-1] if frames else ""
    return f"f::{first}|{last}||q::{q}"

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
            done.add(f"f::{first}|{last}||q::{q}")
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
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)

    return response, prompt, list(map(str, frame_paths))

# ================== MAIN EVAL LOOP ==================
def eval_all_categories(root_jsonl: Path, out_path: Path, resume: bool = False, limit: Optional[int] = None, write_accuracy: bool = False):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer, model = _load_model()

    done_keys = _load_done_keys(out_path) if resume else set()
    mode = "a" if resume else "w"

    written = 0
    # Accuracy tracking (optional)
    total = 0
    correct = 0
    per_cat = {c: {"total": 0, "correct": 0} for c in CATEGORIES}

    with out_path.open(mode, encoding="utf-8") as f_out:
        for cat in CATEGORIES:
            if write_accuracy:
                # MY OWN physion: lowercase directories + lowercase filenames
                cat_dir = root_jsonl / cat.lower()
                cat_file = cat_dir / f"{cat.lower()}_pred.jsonl"
            else:
                # OFFICIAL physion: capitalized directories
                cat_dir = root_jsonl / cat
                cat_file = cat_dir / f"{cat.lower()}_pred.jsonl"

            if not cat_file.exists():
                print(f"[WARN] Missing JSONL for category {cat}: {cat_file}")
                continue

            print(f"[internvl2] Evaluating category={cat} from {cat_file}", flush=True)

            with cat_file.open("r", encoding="utf-8") as f_in:
                for i, line in enumerate(f_in):
                    if not line.strip():
                        continue
                    if limit is not None and written >= limit:
                        print("[internvl2] Reached global limit; stopping.")
                        return

                    try:
                        row = json.loads(line)

                        frames = row.get("frames") or row.get("frame_paths")
                        q = row.get("question")
                        if not isinstance(frames, list) or len(frames) != NUM_FRAMES:
                            raise ValueError("Row must have 'frames' or 'frame_paths' with exactly 8 image paths")
                        if not q:
                            raise ValueError("Row missing 'question'")

                        _validate_frame_paths(frames)

                        if resume:
                            key = _row_key(row)
                            if key in done_keys:
                                continue

                        qid = row.get("id") or row.get("qid") or f"{cat}_row{i}"
                        print(f"[internvl2] {cat} qid={qid} first={frames[0]} last={frames[-1]}", flush=True)

                        pred, prompt, used_paths = ask_internvl2(tokenizer, model, frames, q)

                        out_record = {"category": cat}
                        out_record.update(row)

                        out_record["frame_paths"] = used_paths
                        out_record["prompt_given_to_model"] = prompt
                        out_record["model_output_raw"] = pred
                        out_record["model_output_norm"] = normalize_yesno(pred)

                        gt = (row.get("answer") or "").strip().lower()
                        pr = out_record["model_output_norm"]

                        if gt in ("yes", "no") and pr in ("yes", "no"):
                            total += 1
                            per_cat[cat]["total"] += 1
                            if gt == pr:
                                correct += 1
                                per_cat[cat]["correct"] += 1


                        out_record["model_dir_used"] = MODEL_DIR

                        f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                        f_out.flush()
                        written += 1
                        print(f"[internvl2] wrote {written}", flush=True)

                    except Exception as e:
                        print(f"[internvl2][ERROR] {cat} row {i}: {e}", flush=True)
    # If evaluating my own physion, write an accuracy summary JSONL next to results
    # ================== WRITE BASE ACCURACY (ONE LINE PER CATEGORY) ==================
    if write_accuracy:
        acc_path = out_path.parent / "base_accuracy.jsonl"
        acc_path.parent.mkdir(parents=True, exist_ok=True)

        with acc_path.open("w", encoding="utf-8") as fa:
            # Per-category accuracy
            for c in CATEGORIES:
                t = per_cat[c]["total"]
                k = per_cat[c]["correct"]
                if t == 0:
                    continue
                rec = {
                    "category": c,
                    "accuracy": k / t,
                    "correct": k,
                    "total": t,
                }
                fa.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # ALL categories
            if total > 0:
                rec_all = {
                    "category": "ALL",
                    "accuracy": correct / total,
                    "correct": correct,
                    "total": total,
                }
                fa.write(json.dumps(rec_all, ensure_ascii=False) + "\n")

        print(f"[internvl2] Wrote base accuracy to {acc_path}", flush=True)

    print(f"[internvl2] Done. Wrote {written} rows to {out_path}", flush=True)

# ================== ENTRY ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="If set, skip already-processed rows and append new ones.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional global max rows to run (across all categories).")
    parser.add_argument("--my_own_physion", action="store_true",
                        help="If set, evaluate on my own Physion dataset and write outputs to testing_on_my_own_physion/")
    args = parser.parse_args()

    if args.my_own_physion:
        root_jsonl = ROOT_JSONL_MY
        out_path = OUT_JSONL_MY
        print(f"[internvl2] Using MY Physion root: {root_jsonl}")
        print(f"[internvl2] Output JSONL: {out_path}")
    else:
        root_jsonl = ROOT_JSONL_OFFICIAL
        out_path = OUT_JSONL_OFFICIAL
        print(f"[internvl2] Using OFFICIAL Physion root: {root_jsonl}")
        print(f"[internvl2] Output JSONL: {out_path}")

    eval_all_categories(root_jsonl, out_path, resume=args.resume, limit=args.limit, write_accuracy=args.my_own_physion)
