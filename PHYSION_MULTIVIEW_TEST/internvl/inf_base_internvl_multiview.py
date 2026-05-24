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

# TASK_JSONL = Path("/shared/rsaas/ievab2/PHYSION_MULTIVIEW/dataset.jsonl")
# OUT_JSONL  = Path("/home/ievab2/run_models/PHYSION_MULTIVIEW_TEST/internvl/results/base_results.jsonl")

# NUM_IMAGES     = 3
# MAX_NEW_TOKENS = 64
# INPUT_SIZE     = 448

# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD  = (0.229, 0.224, 0.225)

# # ================== HELPERS ==================
# def normalize_123(text: str) -> str:
#     t = (text or "").strip()
#     if not t:
#         return "unknown"
#     first = t.lstrip()[:1]
#     if first in ("1", "2", "3"):
#         return first
#     m = re.search(r"\b([123])\b", t)
#     return m.group(1) if m else "unknown"

# def build_transform(input_size: int):
#     return T.Compose([
#         T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
#         T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
#         T.ToTensor(),
#         T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#     ])

# def _validate_image_paths(paths: List[str]):
#     bad = [p for p in paths if not p or not os.path.isabs(p) or not os.path.exists(p)]
#     if bad:
#         raise FileNotFoundError(f"Bad/missing image paths: {bad[:3]}{'...' if len(bad)>3 else ''}")

# def load_images_tensor_from_paths(paths: List[Path], input_size: int = INPUT_SIZE) -> torch.Tensor:
#     if len(paths) != NUM_IMAGES:
#         raise ValueError(f"Expected {NUM_IMAGES} images, got {len(paths)}")
#     transform = build_transform(input_size)
#     tensors = []
#     for p in paths:
#         p = Path(p)
#         if not p.exists():
#             raise FileNotFoundError(f"Missing image path: {p}")
#         img = Image.open(p)
#         tensors.append(transform(img))
#     return torch.stack(tensors, dim=0)  # [k,3,H,W]

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
#             imgs = rec.get("images") or []
#             prompt = rec.get("prompt") or rec.get("prompt_given_to_model") or ""
#             a = imgs[0] if imgs else ""
#             c = imgs[-1] if imgs else ""
#             done.add(f"i::{a}|{c}||p::{prompt}")
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
#         print(f"[PATCH] Resizing token embeddings from {emb_n} → {vs}", flush=True)
#         model.resize_token_embeddings(vs)
#     elif vs < emb_n:
#         print(f"[WARN] tokenizer vocab ({vs}) < model embeddings ({emb_n})", flush=True)

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
#     if max_id >= emb_n:
#         raise RuntimeError(f"Token id {max_id} >= vocab size {emb_n} (embedding OOB).")

# # ================== INFERENCE ==================
# def ask_internvl2(tokenizer, model, image_paths: List[str], prompt_text: str):
#     """
#     Multiview task: 3 images + prompt. Answer should be 1/2/3 (1-based).
#     """
#     user_text = (prompt_text or "").strip()
#     # enforce digit-only even if prompt got edited accidentally
#     if "Answer with 1, 2, or 3 only" not in user_text:
#         user_text = user_text + "\nAnswer with 1, 2, or 3 only."

#     pixel_values = load_images_tensor_from_paths([Path(p) for p in image_paths], input_size=INPUT_SIZE)
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

#     return response, prompt, list(map(str, image_paths))

# # ================== MAIN EVAL LOOP ==================
# def eval_multiview(task_jsonl: Path, out_path: Path, resume: bool = False, limit: Optional[int] = None):
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     tokenizer, model = _load_model()

#     done_keys = _load_done_keys(out_path) if resume else set()
#     mode = "a" if resume else "w"

#     written = 0
#     total = 0
#     correct = 0

#     with task_jsonl.open("r", encoding="utf-8") as f_in, out_path.open(mode, encoding="utf-8") as f_out:
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
#                 print(f"[internvl2] qid={qid} img1={imgs[0]} img3={imgs[2]}", flush=True)

#                 pred_raw, prompt_used, used_paths = ask_internvl2(tokenizer, model, imgs, prompt)
#                 pred_norm = normalize_123(pred_raw)
#                 is_correct = (pred_norm == gt)

#                 total += 1
#                 correct += int(is_correct)

#                 out_record = {
#                     "qid": qid,
#                     "prompt": prompt,
#                     "images": used_paths,
#                     "answer": gt,
#                     "prompt_given_to_model": prompt_used,
#                     "model_output_raw": pred_raw,
#                     "model_output_norm": pred_norm,
#                     "correct": bool(is_correct),
#                     "model_dir_used": MODEL_DIR,
#                 }
#                 if "meta" in row:
#                     out_record["meta"] = row["meta"]

#                 f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
#                 f_out.flush()

#                 if resume:
#                     done_keys.add(_row_key(row))

#                 written += 1
#                 print(f"[internvl2] wrote {written}  acc={correct}/{total}={correct/total:.3f}", flush=True)

#             except Exception as e:
#                 print(f"[internvl2][ERROR] row {i}: {e}", flush=True)

#     print(f"[internvl2] Done. Wrote {written} rows to {out_path}", flush=True)
#     if total > 0:
#         print(f"[internvl2] Final accuracy: {correct}/{total} = {correct/total:.4f}", flush=True)

# # ================== ENTRY ==================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--resume", action="store_true",
#                         help="If set, skip already-processed rows and append new ones.")
#     parser.add_argument("--limit", type=int, default=None,
#                         help="Optional max rows to run.")
#     args = parser.parse_args()

#     if not TASK_JSONL.exists():
#         raise SystemExit(f"[internvl2] TASK_JSONL not found: {TASK_JSONL}")

#     print(f"[internvl2] Using dataset: {TASK_JSONL}")
#     print(f"[internvl2] Writing outputs to: {OUT_JSONL}")

#     eval_multiview(TASK_JSONL, OUT_JSONL, resume=args.resume, limit=args.limit)


#!/usr/bin/env python3
import os, json, argparse, re
from pathlib import Path
from typing import Optional, Set, List, Tuple

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

TASK_JSONL_TEST1 = Path("/shared/rsaas/ievab2/PHYSION_MULTIVIEW/dataset.jsonl")
TASK_JSONL_TEST2 = Path("/shared/rsaas/ievab2/PHYSION_MULTIVIEW/dataset2.jsonl")

# RESULTS (as requested)
OUT_JSONL_TEST1 = Path("/home/ievab2/run_models/PHYSION_MULTIVIEW_TEST/internvl/results/test1/base_results.jsonl")
OUT_JSONL_TEST2 = Path("/home/ievab2/run_models/PHYSION_MULTIVIEW_TEST/internvl/results/test2/base_results.jsonl")

MAX_NEW_TOKENS = 64
INPUT_SIZE     = 448

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ================== HELPERS ==================
_NEG_RE = re.compile(r"\b(?:no|false|not|won't|will not|doesn't|does not|don't|didn't|did not|can't|cannot)\b", re.I)
_POS_RE = re.compile(r"\b(?:yes|true|yep|yeah|y)\b", re.I)

def normalize_123(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "unknown"
    first = t.lstrip()[:1]
    if first in ("1", "2", "3"):
        return first
    m = re.search(r"\b([123])\b", t)
    return m.group(1) if m else "unknown"

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

def _validate_image_paths(paths: List[str]):
    bad = [p for p in paths if not p or not os.path.isabs(p) or not os.path.exists(p)]
    if bad:
        raise FileNotFoundError(f"Bad/missing image paths: {bad[:3]}{'...' if len(bad)>3 else ''}")

def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def load_images_tensor_from_paths(paths: List[Path], input_size: int = INPUT_SIZE) -> torch.Tensor:
    transform = build_transform(input_size)
    tensors = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Missing image path: {p}")
        img = Image.open(p)
        tensors.append(transform(img))
    return torch.stack(tensors, dim=0)  # [k,3,H,W]

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

def _debug_vocab(tokenizer, model, prompt: str, k: int):
    img_cnt = prompt.count("<image>")
    assert img_cnt == k, f"mismatch: found {img_cnt} <image> tokens but k={k}"

    enc = tokenizer(prompt, add_special_tokens=False, return_tensors=None)
    ids = enc["input_ids"]
    max_id = max(ids) if ids else -1
    emb_n = model.get_input_embeddings().weight.shape[0]
    print(f"[DBG] vocab_n={emb_n}  max_token_id_in_prompt={max_id}  n_tokens={len(ids)}", flush=True)
    if max_id >= emb_n:
        raise RuntimeError(f"Token id {max_id} >= vocab size {emb_n} (embedding OOB).")

    image_id = tokenizer.convert_tokens_to_ids("<image>")
    print(f"[DBG] '<image>' id: {image_id} (OK if UNK)", flush=True)

# ================== MODEL LOADING ==================
def _load_model():
    print(f"[internvl2-base-mv] loading base model from {MODEL_DIR} …", flush=True)

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
        print(f"[PATCH] Resizing token embeddings from {emb_n} → {vs}", flush=True)
        model.resize_token_embeddings(vs)
    elif vs < emb_n:
        print(f"[WARN] tokenizer vocab ({vs}) < model embeddings ({emb_n})", flush=True)

    model.eval()
    print("[internvl2-base-mv] base model ready.", flush=True)
    return tokenizer, model

# ================== INFERENCE ==================
def ask_internvl2_multiview(tokenizer, model, image_paths: List[str], prompt_text: str,
                            test: str, debug_vocab: bool) -> Tuple[str, str, List[str]]:
    """
    test1: 3 images -> answer 1/2/3
    test2: 2 images -> answer yes/no
    """
    user_text = (prompt_text or "").strip()

    if test == "test1":
        if "Answer with 1, 2, or 3 only" not in user_text:
            user_text = user_text + "\nDo not explain. Answer with 1, 2, or 3 only."
    else:
        if "Answer with yes or no only" not in user_text:
            user_text = user_text + "\nDo not explain. Answer with yes or no only."

    pixel_values = load_images_tensor_from_paths([Path(p) for p in image_paths], input_size=INPUT_SIZE)
    k = pixel_values.shape[0]
    prompt = ("<image>\n" * k) + user_text

    if debug_vocab:
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

    return response, prompt, list(map(str, image_paths))

# ================== EVAL LOOP ==================
def eval_multiview(task_jsonl: Path, out_jsonl: Path, test: str,
                   resume: bool = False, limit: Optional[int] = None, debug_vocab: bool = False):
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tokenizer, model = _load_model()

    done_keys = _load_done_keys(out_jsonl) if resume else set()
    mode = "a" if resume else "w"

    written = 0
    total = 0
    correct = 0

    num_images = 3 if test == "test1" else 2

    with task_jsonl.open("r", encoding="utf-8") as f_in, out_jsonl.open(mode, encoding="utf-8") as f_out:
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
                    print(f"[internvl2-base-mv] {qid} img1={imgs[0]} img3={imgs[2]}", flush=True)
                else:
                    print(f"[internvl2-base-mv] {qid} img1={imgs[0]} img2={imgs[1]}", flush=True)

                pred_raw, prompt_used, used_paths = ask_internvl2_multiview(
                    tokenizer, model, imgs, prompt, test=test, debug_vocab=debug_vocab
                )
                pred_norm = normalize_123(pred_raw) if test == "test1" else normalize_yesno(pred_raw)
                is_correct = (pred_norm == gt)

                total += 1
                correct += int(is_correct)

                out_record = {
                    "qid": qid,
                    "test": test,
                    "prompt": prompt,
                    "images": used_paths,
                    "answer": gt,
                    "prompt_given_to_model": prompt_used,
                    "model_output_raw": pred_raw,
                    "model_output_norm": pred_norm,
                    "correct": bool(is_correct),
                    "model_dir_used": MODEL_DIR,
                }
                if "meta" in row:
                    out_record["meta"] = row["meta"]

                f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                f_out.flush()

                if resume:
                    done_keys.add(_row_key(row))

                written += 1
                print(f"[internvl2-base-mv] wrote {written}  acc={correct}/{total}={correct/total:.3f}", flush=True)

            except Exception as e:
                print(f"[internvl2-base-mv][ERROR] row {i}: {e}", flush=True)

    print(f"[internvl2-base-mv] Done. Wrote {written} rows to {out_jsonl}", flush=True)
    if total > 0:
        print(f"[internvl2-base-mv] Final accuracy: {correct}/{total} = {correct/total:.4f}", flush=True)

# ================== ENTRY ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug_vocab", action="store_true")
    parser.add_argument("--test", choices=["test1", "test2"], required=True)
    args = parser.parse_args()

    if args.test == "test1":
        task_jsonl = TASK_JSONL_TEST1
        out_jsonl = OUT_JSONL_TEST1
    else:
        task_jsonl = TASK_JSONL_TEST2
        out_jsonl = OUT_JSONL_TEST2

    if not task_jsonl.exists():
        raise SystemExit(f"[internvl2-base-mv] TASK_JSONL not found: {task_jsonl}")

    print(f"[internvl2-base-mv] Using dataset: {task_jsonl}", flush=True)
    print(f"[internvl2-base-mv] Writing outputs to: {out_jsonl}", flush=True)

    eval_multiview(
        task_jsonl=task_jsonl,
        out_jsonl=out_jsonl,
        test=args.test,
        resume=args.resume,
        limit=args.limit,
        debug_vocab=args.debug_vocab,
    )
