
# #!/usr/bin/env python3
# import os, json, argparse
# from pathlib import Path
# from typing import Optional, Set, List
# import re

# import torch
# from PIL import Image
# from transformers import AutoTokenizer, AutoModel

# import torchvision.transforms as T
# from torchvision.transforms.functional import InterpolationMode
# from peft import PeftModel, LoraConfig

# # --- env / config ---

# def build_dir_paths(split: str, epochs: int, noweights: bool):
#     tag = "noweights" if noweights else "curriculum"  # naming only

#     if noweights:
#         model_dir = (
#             f"/shared/rsaas/ievab2/FULL_PHYSION_checkpoints/internvl/round0/"
#             f"{split}_noweights_both_{epochs}epochs_4frames"
#         )
#     else:
#         model_dir = (
#             f"/shared/rsaas/ievab2/FULL_PHYSION_checkpoints/internvl/round0/"
#             f"{split}_both_{epochs}epochs_4frames"
#         )

#     # default (non-my-own-physion) outputs still go under epochs{epochs}
#     out_root = Path(
#         f"/home/ievab2/run_models/FULL_PHYSION_FINETUNING/testing/round0/epochs{epochs}/{tag}"
#     )
#     return model_dir, out_root, tag


# # ====================================================================================
# BASE_TOKENIZER   = "/home/ievab2/models/InternVL2-8B"  # base tokenizer

# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# MODEL_DIR      = "/home/ievab2/models/InternVL2-8B"
# NUM_FRAMES     = 8
# MAX_NEW_TOKENS = 128
# INPUT_SIZE     = 448

# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD  = (0.229, 0.224, 0.225)

# # ---------- helpers ----------
# _NEG_RE = re.compile(r"\b(?:no|false|not|won't|will not|doesn't|does not|don't|didn't|did not|can't|cannot)\b", re.I)
# _POS_RE = re.compile(r"\b(?:yes|true|yep|yeah|y)\b", re.I)

# def greedy_generate_with_pixels(tokenizer, model, pixel_values, prompt: str, max_new_tokens: int) -> str:
#     """
#     Minimal greedy decoder that does not rely on .generate() or .chat().
#     Works with InternVLChatModel (or LoRA-wrapped) as long as .forward()
#     accepts (input_ids, pixel_values) and returns logits.
#     """
#     device = next(model.parameters()).device

#     enc = tokenizer(prompt, return_tensors="pt")
#     input_ids = enc["input_ids"].to(device)

#     # Make sure pixel_values has batch dimension: (1, T, C, H, W)
#     if pixel_values.dim() == 4:
#         pixel_values = pixel_values.unsqueeze(0)
#     pixel_values = pixel_values.to(device)

#     eos_id = tokenizer.eos_token_id
#     generated_ids = []

#     for _ in range(max_new_tokens):
#         with torch.no_grad():
#             out = model(input_ids=input_ids, pixel_values=pixel_values)
#             logits = out.logits[:, -1, :]  # last token
#             next_id = logits.argmax(dim=-1)  # greedy

#         next_token_id = next_id.item()
#         generated_ids.append(next_token_id)

#         # append to input_ids for next step
#         next_id = next_id.unsqueeze(0)  # (1, 1)
#         input_ids = torch.cat([input_ids, next_id], dim=1)

#         if eos_id is not None and next_token_id == eos_id:
#             break

#     # Decode only the generated tail
#     text = tokenizer.decode(generated_ids, skip_special_tokens=True)
#     return text.strip()


# def _debug_vocab(tokenizer, model, prompt: str, k: int):
#     # 1) the prompt should have k <image> markers
#     img_cnt = prompt.count("<image>")
#     assert img_cnt == k, f"mismatch: found {img_cnt} <image> tokens but k={k}"

#     # 2) sanity-check ids stay within embedding rows
#     enc = tokenizer(prompt, add_special_tokens=False, return_tensors=None)
#     ids = enc["input_ids"]
#     max_id = max(ids) if ids else -1
#     emb_n = model.get_input_embeddings().weight.shape[0]
#     print(f"[DBG] vocab_n={emb_n}  max_token_id_in_prompt={max_id}  n_tokens={len(ids)}")
#     if max_id >= emb_n:
#         raise RuntimeError(f"Token id {max_id} >= vocab size {emb_n} (embedding OOB).")

#     # 3) '<image>' may NOT be a vocab token (that’s OK for InternVL’s chat())
#     image_id = tokenizer.convert_tokens_to_ids("<image>")
#     print(f"[DBG] '<image>' id: {image_id} (OK if UNK)")

# def normalize_yesno(text: str) -> str:
#     t = (text or "").strip().lower()
#     if t.startswith("yes"): return "yes"
#     if t.startswith("no"):  return "no"
#     if _NEG_RE.search(t): return "no"
#     if _POS_RE.search(t): return "yes"
#     return "unknown"

# def _validate_frame_paths(paths: List[str]):
#     bad = [p for p in paths if not p or not os.path.isabs(p) or not os.path.exists(p)]
#     if bad:
#         raise FileNotFoundError(f"Bad/missing frame paths: {bad[:2]}{'...' if len(bad)>2 else ''}")

# def build_transform(input_size: int):
#     return T.Compose([
#         T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
#         T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
#         T.ToTensor(),
#         T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#     ])

# def _row_key(row: dict) -> str:
#     frames = row.get("frames") or row.get("frame_paths") or []
#     q = row.get("question") or ""
#     first = frames[0] if frames else ""
#     last  = frames[-1] if frames else ""
#     return f"f::{first}|{last}||q::{q}"

# def _load_done_keys(out_path: str) -> Set[str]:
#     done = set()
#     if not os.path.exists(out_path):
#         return done
#     with open(out_path, "r", encoding="utf-8") as f:
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

# # ---------- model ----------
# from peft import PeftModel, LoraConfig
# import os
# def _load_model(model_dir: str):
#     """
#     model_dir can be either:
#       - a full InternVL model dir, or
#       - an adapter dir containing adapter_config.json + adapter_model.safetensors
#     """
#     print(f"[internvl2] loading from {model_dir} …", flush=True)

#     adapter_cfg = os.path.join(model_dir, "adapter_config.json")
#     is_adapter = os.path.exists(adapter_cfg)

#     tok_dir = model_dir if is_adapter else BASE_TOKENIZER
#     tokenizer = AutoTokenizer.from_pretrained(
#         tok_dir, trust_remote_code=True, local_files_only=True, use_fast=False
#     )
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     use_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

#     if is_adapter:
#         # ---- load base InternVL chat model ----
#         base = AutoModel.from_pretrained(
#             MODEL_DIR,
#             trust_remote_code=True,
#             local_files_only=True,
#             torch_dtype=use_dtype,
#             device_map="auto",
#             attn_implementation="eager",
#         )

#         # ---- read + sanitize adapter_config.json for LoraConfig ----
#         import json
#         with open(adapter_cfg, "r") as f:
#             raw = json.load(f)

#         print("[internvl2] adapter_config keys:", sorted(raw.keys()))

#         allowed_keys = set(LoraConfig.__dataclass_fields__.keys())
#         clean = {k: v for k, v in raw.items() if k in allowed_keys}
#         dropped = sorted(set(raw.keys()) - set(clean.keys()))

#         print("[internvl2] using LoraConfig keys:", sorted(clean.keys()))
#         print("[internvl2] dropping keys:", dropped)

#         lc = LoraConfig(**clean)

#         # ---- wrap base with LoRA adapters ----
#         model = PeftModel.from_pretrained(base, model_dir, config=lc)

#         # ---- monkey-patch missing set_output_embeddings on inner chat model ----
#         try:
#             inner = model.base_model  # e.g. LoraModel
#             if hasattr(inner, "model"):
#                 inner = inner.model      # InternVLChatModel

#             if not hasattr(inner, "set_output_embeddings"):
#                 print("[PATCH] Adding dummy set_output_embeddings to", type(inner))
#                 def _set_output_embeddings(self, new_embeds):
#                     # We keep the old LM head; we only needed to resize input embeddings
#                     return
#                 inner.set_output_embeddings = _set_output_embeddings.__get__(inner, inner.__class__)
#         except Exception as e:
#             print(f"[WARN] could not patch set_output_embeddings: {e}")

#     else:
#         # full-model case (also chat model)
#         # model = AutoModel.from_pretrained(
#         #     model_dir,
#         #     trust_remote_code=True,
#         #     local_files_only=True,
#         #     torch_dtype=use_dtype,
#         #     device_map="auto",
#         #     attn_implementation="eager",
#         # )
#         raise RuntimeError(
#             f"[ERROR] Expected a LoRA adapter folder at:\n"
#             f"    {model_dir}\n"
#             f"but adapter_config.json is missing.\n"
#             f"Refusing to load the unfine-tuned base model."
#         )

#     # ---- ensure embeddings match tokenizer size ----
#     emb_n = model.get_input_embeddings().weight.shape[0]
#     vs = len(tokenizer)
#     if vs > emb_n:
#         print(f"[PATCH] Resizing token embeddings from {emb_n} → {vs}")
#         # transformers 4.41.2: no mean_resizing kwarg
#         model.resize_token_embeddings(vs)
#     elif vs < emb_n:
#         print(f"[WARN] tokenizer vocab ({vs}) < model embeddings ({emb_n})")

#     model.eval()
#     print(f"[internvl2] ready (adapter={is_adapter})")
#     return tokenizer, model



# # ---------- ask ----------
# def ask_internvl2(tokenizer, model, frame_paths: List[str], question: str) -> tuple[str, str, List[str]]:
#     user_text = (
#         "You see 8 consecutive frames of a video in temporal order. "
#         "Do not explain; just answer the question concisely. "
#     ) + (question or "")

#     pixel_values = load_frames_tensor_from_paths([Path(p) for p in frame_paths], input_size=INPUT_SIZE)
#     k = pixel_values.shape[0]
#     prompt = ("<image>\n" * k) + user_text

#     # Debug; once you're confident, you can comment this out
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

#     # with torch.inference_mode():
#     #     response = model.chat(tokenizer, pixel_values, prompt, generation_config)
#     with torch.inference_mode():
#         if torch.cuda.is_available():
#             with torch.cuda.amp.autocast(dtype=torch.float16):
#                 response = model.chat(tokenizer, pixel_values, prompt, generation_config)
#         else:
#             response = model.chat(tokenizer, pixel_values, prompt, generation_config)

#     return response, prompt, list(map(str, frame_paths))

#     # device = next(model.parameters()).device
#     # dtype  = next(model.parameters()).dtype
#     # if device.type == "cpu":
#     #     dtype = torch.float32

#     # pixel_values = pixel_values.to(device=device, dtype=dtype)

#     # # Use our own greedy decoder instead of model.chat() to avoid .generate() issues
#     # with torch.inference_mode():
#     #     response = greedy_generate_with_pixels(
#     #         tokenizer,
#     #         model,
#     #         pixel_values,
#     #         prompt,
#     #         MAX_NEW_TOKENS,
#     #     )

#     # return response, prompt, list(map(str, frame_paths))




# # ---------- main loop ----------
# def eval_task(task_path: str, out_path: str, counter_limit: Optional[int] = None, resume: bool = False, base_dir: Optional[str] = None):    
#     print("[internvl2] Starting task …", flush=True)
#     tokenizer, model = _load_model(base_dir or MODEL_DIR)

#     done_keys = _load_done_keys(out_path) if resume else set()
#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#     mode = "a" if resume else "w"
#     written = 0
#     from collections import defaultdict
#     per_cat = defaultdict(lambda: {"correct": 0, "total": 0})
#     overall_correct = 0
#     overall_total = 0

#     with open(task_path, "r", encoding="utf-8") as f_in, open(out_path, mode, encoding="utf-8") as f_out:
#         for i, line in enumerate(f_in):
#             if not line.strip():
#                 continue
#             if counter_limit is not None and written >= counter_limit:
#                 break

#             try:
#                 row = json.loads(line)

#                 # ---- ensure category exists (needed for my_own_physion) ----
#                 if "category" not in row:
#                     # infer from filename
#                     if "collide" in task_path.lower():
#                         row["category"] = "Collide"
#                     elif "contain" in task_path.lower():
#                         row["category"] = "Contain"
#                     elif "dominoes" in task_path.lower():
#                         row["category"] = "Dominoes"
#                     elif "drape" in task_path.lower():
#                         row["category"] = "Drape"
#                     elif "drop" in task_path.lower():
#                         row["category"] = "Drop"
#                     elif "link" in task_path.lower():
#                         row["category"] = "Link"
#                     elif "roll" in task_path.lower():
#                         row["category"] = "Roll"
#                     elif "support" in task_path.lower():
#                         row["category"] = "Support"

#                 frames = row.get("frames") or row.get("frame_paths")
#                 q = row.get("question")
#                 if not isinstance(frames, list) or len(frames) != NUM_FRAMES:
#                     raise ValueError("Row must have 'frames' or 'frame_paths' with exactly 8 image paths")
#                 if not q:
#                     raise ValueError("Row missing 'question'")
                
#                 _validate_frame_paths(frames)

#                 if resume:
#                     key = _row_key(row)
#                     if key in done_keys:
#                         continue

#                 qid = row.get("id") or row.get("qid") or f"row{i}"
#                 print(f"[internvl2] {qid}  first={frames[0]}  last={frames[-1]}", flush=True)

#                 pred, prompt, used_paths = ask_internvl2(tokenizer, model, frames, q)
#                 gt = (row.get("answer") or "").strip().lower()
#                 pr = normalize_yesno(pred)

#                 cat = row.get("category")
#                 if gt in ("yes", "no") and pr in ("yes", "no") and cat is not None:
#                     per_cat[cat]["total"] += 1
#                     overall_total += 1
#                     if gt == pr:
#                         per_cat[cat]["correct"] += 1
#                         overall_correct += 1

#                 # Build output in a fixed key order: qid -> category -> name -> (rest...)
#                 out_record = {}

#                 # qid first
#                 out_record["qid"] = row.get("qid") or row.get("id") or qid

#                 # category second (make sure it's there)
#                 out_record["category"] = row.get("category")

#                 # name third (if present)
#                 if "name" in row:
#                     out_record["name"] = row["name"]

#                 # copy the rest of the original row (excluding keys we already placed)
#                 for k, v in row.items():
#                     if k in ("qid", "id", "category", "name"):
#                         continue
#                     out_record[k] = v

#                 # add model outputs at the end
#                 out_record["frame_paths"] = used_paths
#                 out_record["prompt_given_to_model"] = prompt
#                 out_record["model_output_raw"] = pred
#                 out_record["model_output_norm"] = pr
#                 out_record["model_dir_used"] = base_dir

#                 f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
#                 if resume:
#                     done_keys.add(_row_key(row))


#                 f_out.flush()
#                 written += 1
#                 print(f"[internvl2] wrote {written}", flush=True)

#             except Exception as e:
#                 print(f"[internvl2][ERROR] row {i}: {e}", flush=True)

#     print(f"[internvl2] Done. Wrote {written} rows to {out_path}", flush=True)
#     #if "my_own_physion" in str(out_path):
#         # acc_path = Path(out_path).parent / "results_all_splits.jsonl"
#         # with acc_path.open("a", encoding="utf-8") as f:
#         #     for c, v in per_cat.items():
#         #         f.write(json.dumps({
#         #             "split": args.split,
#         #             "category": c,
#         #             "accuracy": (v["correct"] / v["total"]) if v["total"] > 0 else None,
#         #             "correct": v["correct"],
#         #             "total": v["total"],
#         #         }) + "\n")

#         #     f.write(json.dumps({
#         #         "split": args.split,
#         #         "category": "ALL",
#         #         "accuracy": (overall_correct / overall_total) if overall_total > 0 else None,
#         #         "correct": overall_correct,
#         #         "total": overall_total,
#         #     }) + "\n")

# # ---------- entry ----------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--resume", action="store_true",
#                         help="Skip entries already in OUT_JSONL and append new results")
#     parser.add_argument("--limit", type=int, default=None,
#                         help="Optional max rows to run")
#     parser.add_argument("--split", choices=["SPLIT1", "SPLIT2", "SPLIT3"],
#                         required=True,
#                         help="Which split to evaluate (matches train split).")
#     parser.add_argument("--epochs", type=int, choices=[1, 3, 5],
#                         required=True,
#                         help="Which checkpoint to use (1, 3, or 5 epochs).")
    
#     parser.add_argument("--out_of_distribution",     
#                         type=str,
#                         choices=["yes", "no", "my_own_physion"],
#                         default="no",
#                         help="If set to yes, evaluate on left-out Support category instead of the in-split test set.",
#                         )
#     parser.add_argument("--noweights", action="store_true",
#                         help="Load adapters from *_noweights_* folders and save outputs under .../my_own_physion/noweights/")

#     # DO NOT USE RESUME WITH OUT OF DISTRIBUTION+NOWEIGHTS DOESNT WORK
#     args = parser.parse_args()

#     # Map split+epochs -> model_dir and output root
#     model_dir, out_root, stage_tag = build_dir_paths(args.split, args.epochs, args.noweights)

#     # # Input test JSONL for that split
#     if args.out_of_distribution == "my_own_physion":
#         TASK_JSONLS = [
#             "/shared/rsaas/ievab2/my_own_physion_preprocessed/collide/collide_pred.jsonl",
#             "/shared/rsaas/ievab2/my_own_physion_preprocessed/contain/contain_pred.jsonl",
#             "/shared/rsaas/ievab2/my_own_physion_preprocessed/dominoes/dominoes_pred.jsonl",
#             "/shared/rsaas/ievab2/my_own_physion_preprocessed/drop/drop_pred.jsonl",
#             "/shared/rsaas/ievab2/my_own_physion_preprocessed/link/link_pred.jsonl",
#             "/shared/rsaas/ievab2/my_own_physion_preprocessed/roll/roll_pred.jsonl",
#         ]
#     elif args.out_of_distribution == "yes":
#         TASK_JSONL = f"/shared/rsaas/ievab2/Physion_full_readout_training/Support/support_pred.jsonl"
#     else:
#         TASK_JSONL = f"/shared/rsaas/ievab2/Physion_full_readout_training/SPLITS/{args.split}/test.jsonl"

#     # Output JSONL path
#     if args.out_of_distribution == "my_own_physion":
#         OUT_DIR = Path(
#             f"/home/ievab2/run_models/FULL_PHYSION_FINETUNING/"
#             f"testing/round0/my_own_physion/{stage_tag}/epochs_{args.epochs}"
#         )
#         OUT_JSONL = OUT_DIR / f"{args.split}_results.jsonl"
#     else:
#         OUT_DIR = out_root
#         tag = "OOD_SUPPORT" if args.out_of_distribution == "yes" else "test"
#         OUT_JSONL = OUT_DIR / f"{args.split}_{tag}_out.jsonl"


#     OUT_DIR.mkdir(parents=True, exist_ok=True)

#     print(f"[internvl2] split={args.split} epochs={args.epochs}")
#     if args.out_of_distribution == "my_own_physion":
#         print("[internvl2] reading my_own_physion categories:")
#         for p in TASK_JSONLS:
#             print("  -", p)
#     else:
#         print(f"[internvl2] writing:  {OUT_JSONL}")
#         print(f"[internvl2] using model_dir={model_dir}")


#     if args.out_of_distribution == "my_own_physion":
#         # If NOT resuming, wipe outputs ONCE before looping categories
#         if not args.resume:
#             if OUT_JSONL.exists():
#                 OUT_JSONL.unlink()
#             acc_path = OUT_DIR / "results_all_splits.jsonl"
#             if acc_path.exists():
#                 acc_path.unlink()

#         # Always append across categories so earlier categories are preserved
#         for task_jsonl in TASK_JSONLS:
#             if not os.path.exists(task_jsonl):
#                 print(f"[internvl2][WARN] Missing file, skipping: {task_jsonl}")
#                 continue

#             print(f"[internvl2] Evaluating my_own_physion file: {task_jsonl}")

#             eval_task(
#                 task_jsonl,
#                 OUT_JSONL,
#                 counter_limit=args.limit,
#                 resume=True,     
#                 base_dir=model_dir,
#             )

#     else:
#         eval_task(
#             TASK_JSONL,
#             OUT_JSONL,
#             counter_limit=args.limit,
#             resume=args.resume,
#             base_dir=model_dir,
#         )


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

NUM_FRAMES     = 8
MAX_NEW_TOKENS = 128
INPUT_SIZE     = 448

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

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
    """
    Loads base InternVL model + applies LoRA adapter from adapter_dir.
    Refuses to run if adapter_dir is not an adapter folder.
    """
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
def ask_internvl2(tokenizer, model, frame_paths: List[str], question: str) -> tuple[str, str, List[str]]:
    # IMPORTANT: you asked to test on your own physion ONLY; keep prompt text as-is here.
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
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                response = model.chat(tokenizer, pixel_values, prompt, generation_config)
        else:
            response = model.chat(tokenizer, pixel_values, prompt, generation_config)

    return response, prompt, list(map(str, frame_paths))

# =========================
# EVAL LOOP (MY OWN PHYSION ONLY)
# =========================
def eval_task(task_path: str, out_path: str, adapter_dir: str,
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

                # ---- ensure category exists (needed for my_own_physion) ----
                if "category" not in row:
                    tl = task_path.lower()
                    if "collide" in tl:   row["category"] = "Collide"
                    elif "contain" in tl: row["category"] = "Contain"
                    elif "dominoes" in tl:row["category"] = "Dominoes"
                    elif "drape" in tl:   row["category"] = "Drape"
                    elif "drop" in tl:    row["category"] = "Drop"
                    elif "link" in tl:    row["category"] = "Link"
                    elif "roll" in tl:    row["category"] = "Roll"
                    elif "support" in tl: row["category"] = "Support"

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

                qid = row.get("id") or row.get("qid") or f"row{i}"
                print(f"[internvl2] {qid}  first={frames[0]}  last={frames[-1]}", flush=True)

                pred, prompt, used_paths = ask_internvl2(tokenizer, model, frames, q)
                gt = (row.get("answer") or "").strip().lower()
                pr = normalize_yesno(pred)

                out_record = {}
                out_record["qid"] = row.get("qid") or row.get("id") or qid
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

    # adapter selection (NEW)
    parser.add_argument("--type", required=True, choices=["G","T","C","GC","GT","TC","GTC"],
                        help="Which adapter type to load (matches your MULTIVIEW4 naming).")
    parser.add_argument("--split", type=int, required=True, choices=[1,2,3],
                        help="Which split adapter to load (SPLIT1/2/3).")
    parser.add_argument("--epochs", type=int, required=True, choices=[1,3,5],
                        help="Which epoch folder to load (1/3/5).")
    parser.add_argument("--frames_tag", type=str, default="4frames",
                        help="Folder suffix, default matches your training folders: '4frames'.")

    # eval controls
    parser.add_argument("--resume", action="store_true",
                        help="Skip entries already in OUT_JSONL and append new results")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional max rows to run")

    # IMPORTANT: you asked to test ON MY OWN PHYSION ONLY
    parser.add_argument("--out_dir", type=str, default="/home/ievab2/run_models/PHYSION_MULTIVIEW4PAIRS_AUXILIARY_TRAIN/internvl/results",
                        help="Base output directory for results")

    args = parser.parse_args()

    adapter_dir = build_adapter_dir(args.type, args.split, args.epochs, frames_tag=args.frames_tag)

    # my_own_physion inputs (fixed)
    TASK_JSONLS = [
        "/shared/rsaas/ievab2/my_own_physion_preprocessed/collide/collide_pred.jsonl",
        "/shared/rsaas/ievab2/my_own_physion_preprocessed/contain/contain_pred.jsonl",
        "/shared/rsaas/ievab2/my_own_physion_preprocessed/dominoes/dominoes_pred.jsonl",
        "/shared/rsaas/ievab2/my_own_physion_preprocessed/drop/drop_pred.jsonl",
        "/shared/rsaas/ievab2/my_own_physion_preprocessed/link/link_pred.jsonl",
        "/shared/rsaas/ievab2/my_own_physion_preprocessed/roll/roll_pred.jsonl",
    ]

    out_dir = Path(args.out_dir) / f"epochs{args.epochs}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{args.type}_SPLIT{args.split}_out.jsonl"


    print(f"[internvl2] adapter_dir = {adapter_dir}")
    print(f"[internvl2] out_jsonl   = {out_jsonl}")
    print("[internvl2] reading my_own_physion categories:")
    for p in TASK_JSONLS:
        print("  -", p)

    # If NOT resuming, wipe output ONCE before looping categories
    if not args.resume and out_jsonl.exists():
        out_jsonl.unlink()

    # Always append across categories so earlier categories are preserved
    for task_jsonl in TASK_JSONLS:
        if not os.path.exists(task_jsonl):
            print(f"[internvl2][WARN] Missing file, skipping: {task_jsonl}")
            continue

        print(f"[internvl2] Evaluating my_own_physion file: {task_jsonl}")

        eval_task(
            task_jsonl,
            str(out_jsonl),
            adapter_dir=adapter_dir,
            counter_limit=args.limit,
            resume=True,  # always resume across category files in same run
        )



# how to run:

# # Evaluate GT adapter, split 3, 5 epochs on my_own_physion
# python /home/ievab2/run_models/PHYSION_MULTIVIEW4PAIRS_AUXILIARY_TRAIN/internvl/inf_finetuned_internvl_multiview4.py --type GT --split 3 --epochs 5
# python /home/ievab2/run_models/PHYSION_MULTIVIEW4PAIRS_AUXILIARY_TRAIN/internvl/inf_finetuned_internvl_multiview4.py --type T --split 2 --epochs 5

# # Same, but cap to 50 rows total (across all category files)
# python /home/ievab2/run_models/PHYSION_MULTIVIEW4PAIRS_AUXILIARY_TRAIN/internvl/inf_finetuned_internvl_multiview4.py --type GT --split 3 --epochs 5 --limit 50

# # Resume (skip already-written rows)
# python /home/ievab2/run_models/PHYSION_MULTIVIEW4PAIRS_AUXILIARY_TRAIN/internvl/inf_finetuned_internvl_multiview4.py --type GT --split 3 --epochs 5 --resume
