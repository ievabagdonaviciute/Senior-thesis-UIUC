# #!/usr/bin/env python3
# import os, json, argparse, re
# from pathlib import Path
# from typing import Optional, Set, List, Dict, Tuple
# from safetensors.torch import load_file as safetensors_load_file

# import torch
# from transformers import AutoProcessor, AutoModelForVision2Seq
# from peft import PeftModel
# from PIL import Image

# # ================== ENV / CONFIG ==================
# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# BASE_MODEL_DIR = "/home/ievab2/models/Qwen2.5-VL-7B-Instruct"

# # Multiview dataset (fixed)
# TASK_JSONL = "/shared/rsaas/ievab2/PHYSION_MULTIVIEW/dataset.jsonl"

# NUM_IMAGES     = 3
# MAX_NEW_TOKENS = 64

# PROMPT_PREFIX = (
#     "Do not explain. Answer with 1, 2, or 3 only.\n"
# )

# # ================== PATHS ==================
# def build_dir_paths(
#     split: str,
#     epochs: int,
#     stage: str,
#     round_idx: int = 0,
#     max_frames: int = 4,
# ):
#     """
#     Keep your existing adapter naming convention:
#       /shared/.../FULL_PHYSION_checkpoints/qwen/round{r}/{split}_{stage}_{epochs}epochs_{max_frames}frames
#     """
#     stage_for_path = "both" if stage == "curriculum" else stage

#     adapter_dir = (
#         f"/shared/rsaas/ievab2/FULL_PHYSION_checkpoints/qwen/round{round_idx}/"
#         f"{split}_{stage_for_path}_{epochs}epochs_{max_frames}frames"
#     )

#     out_dir = Path(
#         f"/home/ievab2/run_models/PHYSION_MULTIVIEW_TEST/qwen/results/"
#         f"{split}_{stage}_epochs{epochs}"
#     )
#     out_jsonl = out_dir / "results.jsonl"
#     return adapter_dir, out_dir, out_jsonl

# # ================== ADAPTER KEY REMAP / LOAD ==================
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
#             "Load base with low_cpu_mem_usage=False and move to cuda only at the end."
#         )

# # ================== MODEL LOADING ==================
# def _load_model(adapter_dir: Optional[str] = None):
#     model_dir = BASE_MODEL_DIR
#     print(f"[qwen] loading base model from {model_dir} …", flush=True)

#     if torch.cuda.is_available():
#         dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
#     else:
#         dtype = torch.float32

#     local_only = os.path.isdir(model_dir)

#     processor = AutoProcessor.from_pretrained(
#         model_dir, trust_remote_code=True, local_files_only=local_only
#     )

#     base = AutoModelForVision2Seq.from_pretrained(
#         model_dir,
#         torch_dtype=dtype,
#         trust_remote_code=True,
#         local_files_only=local_only,
#         low_cpu_mem_usage=False,   # avoid meta tensors
#     )
#     base.config.use_cache = False

#     if adapter_dir is not None:
#         adapter_cfg = Path(adapter_dir) / "adapter_config.json"
#         if not adapter_cfg.exists():
#             raise RuntimeError(f"adapter_config.json not found in {adapter_dir}")

#         print(f"[qwen] attaching LoRA adapter from {adapter_dir} …", flush=True)
#         peft_model = PeftModel.from_pretrained(base, adapter_dir, local_files_only=True)
#         _force_load_adapter_weights_in_memory(peft_model, adapter_dir)
#         model = peft_model
#     else:
#         model = base

#     if torch.cuda.is_available():
#         model = model.to("cuda")

#     model.eval()
#     print("[qwen] model ready. cuda?", torch.cuda.is_available(), flush=True)
#     return processor, model

# # ================== HELPERS ==================
# def _open_rgb(p: str) -> Image.Image:
#     with Image.open(p) as im:
#         im = im.convert("RGB") if im.mode != "RGB" else im
#         return im.copy()

# def _validate_image_paths(paths: List[str]):
#     bad = [p for p in paths if (not p) or (not os.path.isabs(p)) or (not os.path.exists(p))]
#     if bad:
#         raise FileNotFoundError(f"Bad/missing image paths: {bad[:3]}{'...' if len(bad)>3 else ''}")

# def _extract_assistant(text: str) -> str:
#     if not text:
#         return ""
#     if "<|assistant|>" in text:
#         text = text.split("<|assistant|>")[-1]
#     text = re.sub(r"\b(system|user|assistant)\b", "", text, flags=re.I)
#     lines = [l.strip() for l in text.splitlines() if l.strip()]
#     return lines[-1] if lines else text.strip()

# def normalize_123(text: str) -> str:
#     """
#     Return "1"/"2"/"3" if we can find it, else "unknown".
#     Accepts outputs like "2", "Answer: 2", "image 3", etc.
#     """
#     t = (text or "").strip()
#     if not t:
#         return "unknown"

#     # fastest: first non-space char
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
#     b = imgs[-1] if imgs else ""
#     return f"i::{a}|{b}||p::{prompt}"

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
#             b = imgs[-1] if imgs else ""
#             done.add(f"i::{a}|{b}||p::{prompt}")
#     return done

# # ================== INFERENCE ==================
# def ask_qwen(processor, model, image_paths: List[str], question: str) -> Tuple[str, str]:
#     if len(image_paths) != NUM_IMAGES:
#         raise ValueError(f"Expected {NUM_IMAGES} images, got {len(image_paths)}")

#     imgs = [_open_rgb(p) for p in image_paths]

#     # Qwen VL chat format
#     messages = [{
#         "role": "user",
#         "content": (
#             [{"type": "image", "image": im} for im in imgs] +
#             [{"type": "text", "text": PROMPT_PREFIX + (question or "")}]
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
#     task_jsonl: str,
#     out_jsonl: Path,
#     adapter_dir: str,
#     limit: Optional[int] = None,
#     resume: bool = False,
# ):
#     print("[qwen-mv] Starting multiview eval …", flush=True)
#     processor, model = _load_model(adapter_dir)

#     done_keys = _load_done_keys(out_jsonl) if resume else set()
#     out_jsonl.parent.mkdir(parents=True, exist_ok=True)
#     mode = "a" if resume else "w"

#     written = 0
#     correct = 0
#     total = 0

#     with open(task_jsonl, "r", encoding="utf-8") as f_in, out_jsonl.open(mode, encoding="utf-8") as f_out:
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
#                 print(f"[qwen-mv] {qid} img1={imgs[0]} img3={imgs[2]}", flush=True)

#                 pred_raw, prompt_text = ask_qwen(processor, model, imgs, prompt)
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
#                     "prompt_given_to_model": prompt_text,
#                     "model_dir_used": adapter_dir,
#                 }

#                 # keep original metadata if present
#                 if "meta" in row:
#                     out_record["meta"] = row["meta"]

#                 f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
#                 if resume:
#                     done_keys.add(_row_key(row))

#                 f_out.flush()
#                 written += 1
#                 print(f"[qwen-mv] wrote {written}  acc={correct}/{total}={correct/total:.3f}", flush=True)

#             except Exception as e:
#                 print(f"[qwen-mv][ERROR] row {i}: {e}", flush=True)

#     print(f"[qwen-mv] Done. Wrote {written} rows to {out_jsonl}", flush=True)
#     if total > 0:
#         print(f"[qwen-mv] Final accuracy: {correct}/{total} = {correct/total:.4f}", flush=True)

# # ================== ENTRY ==================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--resume", action="store_true",
#                         help="Skip entries already in OUT_JSONL and append new results")
#     parser.add_argument("--limit", type=int, default=None,
#                         help="Optional max rows to run")

#     # Keep these to match your checkpoint naming
#     parser.add_argument("--split", choices=["SPLIT1", "SPLIT2", "SPLIT3"], required=True)
#     parser.add_argument("--epochs", type=int, choices=[1, 3, 5], required=True)
#     parser.add_argument("--stage", choices=["curriculum", "noweights"], default="curriculum")
#     parser.add_argument("--round", type=int, default=0)
#     parser.add_argument("--max_frames", type=int, default=4,
#                         help="Matches the *_4frames part of your adapter directory name")

#     args = parser.parse_args()

#     adapter_dir, out_dir, out_jsonl = build_dir_paths(
#         args.split, args.epochs, args.stage, round_idx=args.round, max_frames=args.max_frames
#     )

#     if not os.path.isdir(adapter_dir):
#         raise SystemExit(f"[qwen-mv] adapter_dir not found: {adapter_dir}")

#     print(f"[qwen-mv] using TASK_JSONL={TASK_JSONL}")
#     print(f"[qwen-mv] using adapter_dir={adapter_dir}")
#     print(f"[qwen-mv] writing OUT_JSONL={out_jsonl}")

#     eval_multiview(
#         TASK_JSONL,
#         out_jsonl,
#         adapter_dir=adapter_dir,
#         limit=args.limit,
#         resume=args.resume,
#     )

#!/usr/bin/env python3
import os, json, argparse, re
from pathlib import Path
from typing import Optional, Set, List, Dict, Tuple

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

# ================== ENV / CONFIG ==================
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_DIR = "/home/ievab2/models/Qwen2.5-VL-7B-Instruct"

# Datasets (selected by --test)
TASK_JSONL_TEST1 = Path("/shared/rsaas/ievab2/PHYSION_MULTIVIEW/dataset.jsonl")
TASK_JSONL_TEST2 = Path("/shared/rsaas/ievab2/PHYSION_MULTIVIEW/dataset2.jsonl")

# Results base (selected by --test/--epochs/--split)
RESULTS_ROOT = Path("/home/ievab2/run_models/PHYSION_MULTIVIEW_TEST/qwen/results")

MAX_NEW_TOKENS = 64

# ================== HELPERS ==================
def _validate_image_paths(paths: List[str]):
    bad = [p for p in paths if not p or not os.path.isabs(p) or not os.path.exists(p)]
    if bad:
        raise FileNotFoundError(f"Bad/missing image paths: {bad[:3]}{'...' if len(bad)>3 else ''}")

def normalize_123(text: str) -> str:
    """
    Return "1"/"2"/"3" if we can find it, else "unknown".
    Accepts outputs like "2", "Answer: 2", "image 3", etc.
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
    # first token heuristic
    first = t.split()[0]
    if first in ("yes", "y", "true"):
        return "yes"
    if first in ("no", "n", "false"):
        return "no"
    # fallback search
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
    # Keep only content after last assistant tag
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1]
    # Take the last non-empty line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()

def _open_rgb(p: str) -> Image.Image:
    with Image.open(p) as im:
        im = im.convert("RGB") if im.mode != "RGB" else im
        return im.copy()

# ================== MODEL LOADING ==================
def _load_model():
    print(f"[qwen-mv] loading model from {MODEL_DIR} …", flush=True)

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
    print("[qwen-mv] model ready. cuda?", torch.cuda.is_available(), flush=True)
    return processor, model

# ================== INFERENCE ==================
def ask_qwen(processor, model, image_paths: List[str], question: str, test: str) -> Tuple[str, str]:
    """
    Multiview task:
      - test1: 3 images + prompt -> answer 1/2/3
      - test2: 2 images + prompt -> answer yes/no
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
                    print(f"[qwen-mv] {qid} img1={imgs[0]} img3={imgs[2]}", flush=True)
                else:
                    print(f"[qwen-mv] {qid} img1={imgs[0]} img2={imgs[1]}", flush=True)

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
                print(f"[qwen-mv] wrote {written}  acc={correct}/{total}={correct/total:.3f}", flush=True)

            except Exception as e:
                print(f"[qwen-mv][ERROR] row {i}: {e}", flush=True)

    print(f"[qwen-mv] Done. Wrote {written} rows to {out_path}", flush=True)
    if total > 0:
        print(f"[qwen-mv] Final accuracy: {correct}/{total} = {correct/total:.4f}", flush=True)

# ================== ENTRY ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="If set, skip already-processed rows and append new ones.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional max rows to run.")
    parser.add_argument("--test", choices=["test1", "test2"], required=True,
                        help="Which multiview dataset to run: test1 (3 images) or test2 (2 images).")
    parser.add_argument("--split", choices=["SPLIT1", "SPLIT2", "SPLIT3"], required=True,
                        help="Split tag used only for output naming (matches your finetune split).")
    parser.add_argument("--epochs", type=int, choices=[1, 3, 5], required=True,
                        help="Epoch tag used only for output naming.")
    parser.add_argument("--stage", choices=["both", "noweights"], default="both",
                        help="Naming only (curriculum flag removed).")

    args = parser.parse_args()

    # Select dataset
    if args.test == "test1":
        task_jsonl = TASK_JSONL_TEST1
    else:
        task_jsonl = TASK_JSONL_TEST2

    if not task_jsonl.exists():
        raise SystemExit(f"[qwen-mv] TASK_JSONL not found: {task_jsonl}")

    # Output path:
    # /home/ievab2/run_models/PHYSION_MULTIVIEW_TEST/qwen/results/test1/epoch1/SPLIT1_results.jsonl
    # /home/ievab2/run_models/PHYSION_MULTIVIEW_TEST/qwen/results/test2/epoch5/SPLIT2_results.jsonl
    out_dir = RESULTS_ROOT / args.test / f"epoch{args.epochs}"
    out_jsonl = out_dir / f"{args.split}_results.jsonl"

    print(f"[qwen-mv] test={args.test} split={args.split} epochs={args.epochs} stage={args.stage}", flush=True)
    print(f"[qwen-mv] Using dataset: {task_jsonl}", flush=True)
    print(f"[qwen-mv] Writing outputs to: {out_jsonl}", flush=True)

    eval_multiview(
        task_path=task_jsonl,
        out_path=out_jsonl,
        test=args.test,
        resume=args.resume,
        limit=args.limit,
    )
