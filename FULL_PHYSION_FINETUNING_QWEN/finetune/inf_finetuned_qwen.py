# #!/usr/bin/env python3
# import os, json, argparse, re
# from pathlib import Path
# from typing import Optional, Set, List, Dict

# import torch
# from transformers import AutoProcessor, AutoModelForVision2Seq
# from peft import PeftModel
# from PIL import Image

# # ================== ENV / CONFIG ==================
# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# BASE_MODEL_DIR = "/home/ievab2/models/Qwen2.5-VL-7B-Instruct"
# ADAPTER_DIR = None  # set via CLI

# NUM_FRAMES     = 8
# MAX_NEW_TOKENS = 128

# INSTR = (
#     "These 8 images are consecutive frames from a single video in time order (000→007). "
#     "Do not explain; just answer the question concisely. "
# )

# _NEG_RE = re.compile(r"\b(?:no|false|not|won't|will not|doesn't|does not|don't|didn't|did not|can't|cannot)\b", re.I)
# _POS_RE = re.compile(r"\b(?:yes|true|yep|yeah|y)\b", re.I)

# # ================== PATHS ==================
# def build_dir_paths(split: str, epochs: int, round_idx: int = 0):
#     adapter_dir = (
#         f"/shared/rsaas/ievab2/FULL_PHYSION_checkpoints/qwen/round{round_idx}/"
#         f"{split}_both_{epochs}epochs_4frames"
#     )
#     out_root = Path(
#         f"/home/ievab2/run_models/FULL_PHYSION_FINETUNING_QWEN/testing/round{round_idx}/"
#         f"{split}_both_epochs{epochs}"
#     )
#     return adapter_dir, out_root


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
#     bad = [p for p in paths if not p or not os.path.isabs(p) or not os.path.exists(p)]
#     if bad:
#         raise FileNotFoundError(f"Bad/missing frame paths: {bad[:3]}{'...' if len(bad)>3 else ''}")

# def _row_key(row: dict) -> str:
#     frames = row.get("frames") or row.get("frame_paths") or []
#     q = row.get("question") or row.get("prompt") or ""
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

# def _extract_assistant(text: str) -> str:
#     """
#     Same idea as your base_qwen.py: strip everything before <|assistant|>.
#     """
#     if not text:
#         return ""
#     if "<|assistant|>" in text:
#         return text.split("<|assistant|>", maxsplit=1)[-1].strip()
#     # fallback
#     return text.strip()

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
#         device_map="auto",
#         trust_remote_code=True,
#         local_files_only=local_only,
#     )
#     base.config.use_cache = False

#     if adapter_dir is not None:
#         adapter_cfg = Path(adapter_dir) / "adapter_config.json"
#         if not adapter_cfg.exists():
#             raise RuntimeError(f"adapter_config.json not found in {adapter_dir}")
#         print(f"[qwen] applying LoRA adapter from {adapter_dir} …", flush=True)
#         base = PeftModel.from_pretrained(base, adapter_dir, local_files_only=True)

#     base.eval()
#     print("[qwen] model ready. cuda?", torch.cuda.is_available(), flush=True)
#     return processor, base


# # ================== INFERENCE ==================
# def _open_rgb(p: str) -> Image.Image:
#     with Image.open(p) as im:
#         im = im.convert("RGB") if im.mode != "RGB" else im
#         return im.copy()   # ensure file handle is closed


# def ask_qwen(processor, model, frame_paths: List[str], question: str) -> str:
#     if len(frame_paths) != NUM_FRAMES:
#         raise ValueError(f"Expected {NUM_FRAMES} frames, got {len(frame_paths)}")

#     imgs = [_open_rgb(p) for p in frame_paths]

#     messages = [{
#         "role": "user",
#         "content": (
#             [{"type": "image", "image": im} for im in imgs] +
#             [{"type": "text",
#               "text": "These 8 images are consecutive frames from a single video in time order (000→007). "
#                       "Do not explain; just answer the question concisely. "
#                       + (question or "")}]
#         ),
#     }]

#     chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     device = next(model.parameters()).device

#     # single-sample form: text=str, images=list[PIL]
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

#     text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
#     pred = _extract_assistant(text)
#     return pred, chat_text



# # ================== EVAL LOOP ==================
# def eval_task(
#     task_path: str,
#     out_path: Path,
#     adapter_dir: str,
#     counter_limit: Optional[int] = None,
#     resume: bool = False,
# ):
#     print("[qwen-ft] Starting task …", flush=True)
#     processor, model = _load_model(adapter_dir)

#     done_keys = _load_done_keys(out_path) if resume else set()
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     mode = "a" if resume else "w"

#     written = 0
#     from collections import defaultdict
#     per_cat = defaultdict(lambda: {"correct": 0, "total": 0})
#     overall_correct = 0
#     overall_total = 0

#     with open(task_path, "r", encoding="utf-8") as f_in, out_path.open(mode, encoding="utf-8") as f_out:
#         for i, line in enumerate(f_in):
#             if not line.strip():
#                 continue
#             if counter_limit is not None and written >= counter_limit:
#                 break

#             try:
#                 row = json.loads(line)

#                 # ---- ensure category exists (needed for my_own_physion) ----
#                 if "category" not in row:
#                     lower = task_path.lower()
#                     if "collide" in lower: row["category"] = "Collide"
#                     elif "contain" in lower: row["category"] = "Contain"
#                     elif "dominoes" in lower: row["category"] = "Dominoes"
#                     elif "drape" in lower: row["category"] = "Drape"
#                     elif "drop" in lower: row["category"] = "Drop"
#                     elif "link" in lower: row["category"] = "Link"
#                     elif "roll" in lower: row["category"] = "Roll"
#                     elif "support" in lower: row["category"] = "Support"

#                 frames = row.get("frames") or row.get("frame_paths")
#                 q = row.get("question")
#                 if not isinstance(frames, list) or len(frames) != NUM_FRAMES:
#                     raise ValueError(f"Row must have 'frames' or 'frame_paths' with exactly {NUM_FRAMES} image paths")
#                 if not q:
#                     raise ValueError("Row missing 'question'")

#                 _validate_frame_paths(frames)

#                 if resume:
#                     key = _row_key(row)
#                     if key in done_keys:
#                         continue

#                 qid = row.get("id") or row.get("qid") or f"row{i}"
#                 print(f"[qwen-ft] {qid} first={frames[0]} last={frames[-1]}", flush=True)

#                 pred, prompt_text = ask_qwen(processor, model, frames, q)
#                 gt = (row.get("answer") or "").strip().lower()
#                 pr = normalize_yesno(pred)
#                 cat = row.get("category")

#                 if gt in ("yes", "no") and pr in ("yes", "no") and cat is not None:
#                     per_cat[cat]["total"] += 1
#                     overall_total += 1
#                     if gt == pr:
#                         per_cat[cat]["correct"] += 1
#                         overall_correct += 1

#                 # InternVL-style fixed key order
#                 out_record = {}
#                 out_record["qid"] = row.get("qid") or row.get("id") or qid
#                 out_record["category"] = row.get("category")
#                 if "name" in row:
#                     out_record["name"] = row["name"]

#                 for k, v in row.items():
#                     if k in ("qid", "id", "category", "name"):
#                         continue
#                     out_record[k] = v

#                 out_record["frame_paths"] = list(map(str, frames))
#                 out_record["prompt_given_to_model"] = prompt_text
#                 out_record["model_output_raw"] = pred
#                 out_record["model_output_norm"] = pr
#                 out_record["model_dir_used"] = adapter_dir

#                 f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
#                 if resume:
#                     done_keys.add(_row_key(row))

#                 f_out.flush()
#                 written += 1
#                 print(f"[qwen-ft] wrote {written}", flush=True)

#             except Exception as e:
#                 print(f"[qwen-ft][ERROR] row {i}: {e}", flush=True)

#     print(f"[qwen-ft] Done. Wrote {written} rows to {out_path}", flush=True)

# # ================== ENTRY ==================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--resume", action="store_true",
#                         help="Skip entries already in OUT_JSONL and append new results")
#     parser.add_argument("--limit", type=int, default=None,
#                         help="Optional max rows to run")

#     parser.add_argument("--split", choices=["SPLIT1", "SPLIT2", "SPLIT3"], required=True)
#     parser.add_argument("--epochs", type=int, choices=[1, 3, 5], required=True)
#     parser.add_argument("--stage", choices=["curriculum", "noweights"], default="curriculum")
#     parser.add_argument("--round", type=int, default=0)

#     parser.add_argument("--out_of_distribution",
#                         type=str,
#                         choices=["yes", "no", "my_own_physion"],
#                         default="no")

#     args = parser.parse_args()

#     # Adapter dir + output dir
#     adapter_dir, out_root = build_dir_paths(args.split, args.epochs, round_idx=args.round)

#     # Input JSONL(s)
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
#         TASK_JSONL = "/shared/rsaas/ievab2/Physion_full_readout_training/Support/support_pred.jsonl"
#     else:
#         TASK_JSONL = f"/shared/rsaas/ievab2/Physion_full_readout_training/SPLITS/{args.split}/test.jsonl"

#     # Output JSONL
#     if args.out_of_distribution == "my_own_physion":
#         OUT_DIR = Path(
#             f"/home/ievab2/run_models/FULL_PHYSION_FINETUNING_QWEN/testing/round{args.round}/my_own_physion/"
#             f"{args.split}_{args.stage}_epochs{args.epochs}"
#         )
#         OUT_JSONL = OUT_DIR / f"{args.split}_results.jsonl"
#     else:
#         OUT_DIR = out_root
#         tag = "OOD_SUPPORT" if args.out_of_distribution == "yes" else "test"
#         OUT_JSONL = OUT_DIR / f"{args.split}_{tag}_out.jsonl"

#     OUT_DIR.mkdir(parents=True, exist_ok=True)

#     print(f"[qwen-ft] split={args.split} stage={args.stage} epochs={args.epochs} round={args.round}")
#     print(f"[qwen-ft] using adapter_dir={adapter_dir}")
#     print(f"[qwen-ft] writing: {OUT_JSONL}")

#     if args.out_of_distribution == "my_own_physion":
#         if not args.resume:
#             if OUT_JSONL.exists():
#                 OUT_JSONL.unlink()

#         for task_jsonl in TASK_JSONLS:
#             if not os.path.exists(task_jsonl):
#                 print(f"[qwen-ft][WARN] Missing file, skipping: {task_jsonl}")
#                 continue

#             print(f"[qwen-ft] Evaluating my_own_physion file: {task_jsonl}")
#             eval_task(
#                 task_jsonl,
#                 OUT_JSONL,
#                 adapter_dir=adapter_dir,
#                 counter_limit=args.limit,
#                 resume=True,
#             )
#     else:
#         eval_task(
#             TASK_JSONL,
#             OUT_JSONL,
#             adapter_dir=adapter_dir,
#             counter_limit=args.limit,
#             resume=args.resume,
#         )

#!/usr/bin/env python3
import os, json, argparse, re
from pathlib import Path
from typing import Optional, Set, List, Dict
from safetensors.torch import load_file as safetensors_load_file

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from PIL import Image

# ================== ENV / CONFIG ==================
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_MODEL_DIR = "/home/ievab2/models/Qwen2.5-VL-7B-Instruct"
ADAPTER_DIR = None  # set via CLI

NUM_FRAMES     = 8
MAX_NEW_TOKENS = 128
TINY_TEST_JSONL = "/shared/rsaas/ievab2/TINY_PHYSION_TEST/tiny_test.jsonl"
TINY_TEST_OUT   = Path("/home/ievab2/run_models/FULL_PHYSION_FINETUNING_QWEN/testing/round0/results_TINY_TEST.jsonl")

INSTR = (
    "These 8 images are consecutive frames from a single video in time order (000→007). "
    "Do not explain; just answer the question concisely. "
)

_NEG_RE = re.compile(r"\b(?:no|false|not|won't|will not|doesn't|does not|don't|didn't|did not|can't|cannot)\b", re.I)
_POS_RE = re.compile(r"\b(?:yes|true|yep|yeah|y)\b", re.I)
TINY_TEST_CKPT_SUFFIX = "_TINY_TEST"

# ================== PATHS ==================
def build_dir_paths(
    split: str,
    epochs: int,
    stage: str,
    round_idx: int = 0,
    max_frames: int = 4,
    tiny_test: bool = False,
):
    suffix = "_TINY_TEST" if tiny_test else ""

    adapter_dir = (
        f"/shared/rsaas/ievab2/FULL_PHYSION_checkpoints/qwen/round{round_idx}/"
        f"{split}_{stage}_{epochs}epochs_{max_frames}frames{suffix}"
    )

    out_root = Path(
        f"/home/ievab2/run_models/FULL_PHYSION_FINETUNING_QWEN/testing/round{round_idx}/"
        f"{split}_{stage}_epochs{epochs}{suffix}"
    )

    return adapter_dir, out_root



# ================== HELPERS ==================
def infer_category_from_frames(frames: List[str]) -> Optional[str]:
    if not frames:
        return None
    p = frames[0].lower()
    if "/collide/" in p: return "Collide"
    if "/contain/" in p: return "Contain"
    if "/dominoes/" in p: return "Dominoes"
    if "/drape/" in p: return "Drape"
    if "/drop/" in p: return "Drop"
    if "/link/" in p: return "Link"
    if "/roll/" in p: return "Roll"
    if "/support/" in p: return "Support"
    return None

def _remap_lora_keys_for_runtime(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remap adapter keys from what PEFT saved to what the *current* runtime modules expect.

    Handles:
      - ...lora_A.weight -> ...lora_A.default.weight (same for lora_B)
      - optional "language_model" path differences seen in Qwen2.5-VL wrappers
    """
    out = {}

    for k, v in sd.items():
        k2 = k

        # 1) Common PEFT suffix mismatch
        k2 = k2.replace(".lora_A.weight", ".lora_A.default.weight")
        k2 = k2.replace(".lora_B.weight", ".lora_B.default.weight")

        # 2) Common Qwen wrapper mismatch (sometimes saved under language_model.layers)
        # Try a few safe normalizations; whichever matches will get loaded.
        # (We keep these as no-ops if not present.)
        k2 = k2.replace(".model.model.language_model.layers.", ".model.model.layers.")
        k2 = k2.replace(".model.language_model.layers.", ".model.layers.")
        k2 = k2.replace(".language_model.layers.", ".layers.")

        out[k2] = v

    return out


def _force_load_adapter_weights_in_memory(peft_model: PeftModel, adapter_dir: str):
    """
    Load adapter_model.safetensors manually (no disk edits) and remap keys so they match runtime.
    """
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

    # Quick sanity prints (these are the same signals you were checking in terminal)
    lora_params = [(n, p) for n, p in peft_model.named_parameters() if "lora_" in n]
    meta_cnt = sum(int(getattr(p, "is_meta", False)) for _, p in lora_params)

    print(f"[qwen-ft] manual adapter load: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    print(f"[qwen-ft] lora params: {len(lora_params)}  meta={meta_cnt}", flush=True)
    if len(lora_params) > 0:
        mean_abs = torch.stack([p.detach().abs().mean().cpu() for _, p in lora_params[: min(10, len(lora_params))]]).mean()
        print(f"[qwen-ft] mean(|lora|) sample={float(mean_abs):.6g}", flush=True)

    # If meta_cnt > 0, adapters still didn't materialize → usually means base loaded with meta/offload.
    if meta_cnt > 0:
        raise RuntimeError(
            "LoRA params are still on meta after load. "
            "This usually happens when the base model was initialized with meta tensors. "
            "Load the base on CPU first (device_map={'': 'cpu'}, low_cpu_mem_usage=False)."
        )


def normalize_yesno(text: str) -> str:
    t = (text or "").strip().lower()

    # hard cut: only first word matters
    first = t.split()[0] if t else ""

    if first in ("yes", "y", "true"):
        return "yes"
    if first in ("no", "n", "false"):
        return "no"

    return "unknown"


def _validate_frame_paths(paths: List[str]):
    bad = [p for p in paths if not p or not os.path.isabs(p) or not os.path.exists(p)]
    if bad:
        raise FileNotFoundError(f"Bad/missing frame paths: {bad[:3]}{'...' if len(bad)>3 else ''}")

def _row_key(row: dict) -> str:
    frames = row.get("frames") or row.get("frame_paths") or []
    q = row.get("question") or row.get("prompt") or ""
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

def _extract_assistant(text: str) -> str:
    if not text:
        return ""

    # Keep only content AFTER the last assistant tag
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1]

    # Remove stray role words
    text = re.sub(r"\b(system|user|assistant)\b", "", text, flags=re.I)

    # Take the last non-empty line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else ""


# ================== MODEL LOADING ==================
def _load_model(adapter_dir: Optional[str] = None):
    model_dir = BASE_MODEL_DIR
    print(f"[qwen] loading base model from {model_dir} …", flush=True)

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    local_only = os.path.isdir(model_dir)

    processor = AutoProcessor.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=local_only
    )

    # IMPORTANT: load on CPU first to avoid meta/offload issues that break adapter loading
    base = AutoModelForVision2Seq.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=local_only,
        low_cpu_mem_usage=False,   # keeps real tensors (not meta)
    )
    base.config.use_cache = False


    if adapter_dir is not None:
        adapter_cfg = Path(adapter_dir) / "adapter_config.json"
        if not adapter_cfg.exists():
            raise RuntimeError(f"adapter_config.json not found in {adapter_dir}")

        print(f"[qwen] attaching LoRA adapter from {adapter_dir} …", flush=True)

        # 1) Attach adapter “structure”
        peft_model = PeftModel.from_pretrained(base, adapter_dir, local_files_only=True)

        # 2) Force-load adapter weights with key remap (no disk changes)
        _force_load_adapter_weights_in_memory(peft_model, adapter_dir)

        model = peft_model
    else:
        model = base

    # Move to GPU at the end (simple + reliable)
    if torch.cuda.is_available():
        model = model.to("cuda")

    model.eval()
    print("[qwen] model ready. cuda?", torch.cuda.is_available(), flush=True)
    return processor, model



# ================== INFERENCE ==================
def _open_rgb(p: str) -> Image.Image:
    with Image.open(p) as im:
        im = im.convert("RGB") if im.mode != "RGB" else im
        return im.copy()   # ensure file handle is closed


def ask_qwen(processor, model, frame_paths: List[str], question: str) -> str:
    if len(frame_paths) != NUM_FRAMES:
        raise ValueError(f"Expected {NUM_FRAMES} frames, got {len(frame_paths)}")

    imgs = [_open_rgb(p) for p in frame_paths]

    messages = [{
        "role": "user",
        "content": (
            [{"type": "image", "image": im} for im in imgs] +
            [{"type": "text",
              "text": "These 8 images are consecutive frames from a single video in time order (000→007). "
                      "Do not explain; just answer the question concisely. "
                      + (question or "")}]
        ),
    }]

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device

    # single-sample form: text=str, images=list[PIL]
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

    text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    pred = _extract_assistant(text)
    return pred, chat_text



# ================== EVAL LOOP ==================
def eval_task(
    task_path: str,
    out_path: Path,
    adapter_dir: str,
    counter_limit: Optional[int] = None,
    resume: bool = False,
):
    print("[qwen-ft] Starting task …", flush=True)
    processor, model = _load_model(adapter_dir)

    done_keys = _load_done_keys(out_path) if resume else set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"

    written = 0
    from collections import defaultdict
    per_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    overall_correct = 0
    overall_total = 0

    with open(task_path, "r", encoding="utf-8") as f_in, out_path.open(mode, encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            if not line.strip():
                continue
            if counter_limit is not None and written >= counter_limit:
                break

            try:
                row = json.loads(line)

                # ---- ensure category exists (needed for my_own_physion) ----
                if "category" not in row:
                    lower = task_path.lower()
                    if "collide" in lower: row["category"] = "Collide"
                    elif "contain" in lower: row["category"] = "Contain"
                    elif "dominoes" in lower: row["category"] = "Dominoes"
                    elif "drape" in lower: row["category"] = "Drape"
                    elif "drop" in lower: row["category"] = "Drop"
                    elif "link" in lower: row["category"] = "Link"
                    elif "roll" in lower: row["category"] = "Roll"
                    elif "support" in lower: row["category"] = "Support"

                frames = row.get("frames") or row.get("frame_paths")
                q = row.get("question")
                if not isinstance(frames, list) or len(frames) != NUM_FRAMES:
                    raise ValueError(f"Row must have 'frames' or 'frame_paths' with exactly {NUM_FRAMES} image paths")
                if not q:
                    raise ValueError("Row missing 'question'")

                # ===== ADD THIS BLOCK EXACTLY HERE =====
                if "category" not in row or row["category"] is None:
                    row["category"] = infer_category_from_frames(frames)
                # ======================================

                _validate_frame_paths(frames)

                if resume:
                    key = _row_key(row)
                    if key in done_keys:
                        continue

                qid = row.get("id") or row.get("qid") or f"row{i}"
                print(f"[qwen-ft] {qid} first={frames[0]} last={frames[-1]}", flush=True)

                pred, prompt_text = ask_qwen(processor, model, frames, q)
                gt = (row.get("answer") or "").strip().lower()
                pr = normalize_yesno(pred)
                cat = row.get("category")

                if gt in ("yes", "no") and pr in ("yes", "no") and cat is not None:
                    per_cat[cat]["total"] += 1
                    overall_total += 1
                    if gt == pr:
                        per_cat[cat]["correct"] += 1
                        overall_correct += 1

                # InternVL-style fixed key order
                out_record = {}
                out_record["qid"] = row.get("qid") or row.get("id") or qid
                out_record["category"] = row.get("category")
                if "name" in row:
                    out_record["name"] = row["name"]

                for k, v in row.items():
                    if k in ("qid", "id", "category", "name"):
                        continue
                    out_record[k] = v

                out_record["frame_paths"] = list(map(str, frames))
                out_record["prompt_given_to_model"] = prompt_text
                out_record["model_output_raw"] = pred
                out_record["model_output_norm"] = pr
                out_record["model_dir_used"] = adapter_dir

                f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                if resume:
                    done_keys.add(_row_key(row))

                f_out.flush()
                written += 1
                print(f"[qwen-ft] wrote {written}", flush=True)

            except Exception as e:
                print(f"[qwen-ft][ERROR] row {i}: {e}", flush=True)

    print(f"[qwen-ft] Done. Wrote {written} rows to {out_path}", flush=True)

# ================== ENTRY ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Skip entries already in OUT_JSONL and append new results")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional max rows to run")

    parser.add_argument("--split", choices=["SPLIT1", "SPLIT2", "SPLIT3"], required=True)
    parser.add_argument("--epochs", type=int, choices=[1, 3, 5], required=True)
    parser.add_argument("--stage", choices=["curriculum", "noweights"], default="curriculum")
    parser.add_argument("--round", type=int, default=0)

    parser.add_argument("--out_of_distribution",
                        type=str,
                        choices=["yes", "no", "my_own_physion"],
                        default="no")
    parser.add_argument("--tiny_test", action="store_true",
                        help="Evaluate on tiny_test.jsonl and write to fixed results_TINY_TEST.jsonl")

    args = parser.parse_args()

    # Adapter dir + output dir
    adapter_dir, out_root = build_dir_paths(
        args.split,
        args.epochs,
        stage=args.stage,
        round_idx=args.round,
        tiny_test=args.tiny_test,
    )


    # Input JSONL(s)
    if args.tiny_test:
        TASK_JSONL = TINY_TEST_JSONL
    elif args.out_of_distribution == "my_own_physion":
        TASK_JSONLS = [
            "/shared/rsaas/ievab2/my_own_physion_preprocessed/collide/collide_pred.jsonl",
            "/shared/rsaas/ievab2/my_own_physion_preprocessed/contain/contain_pred.jsonl",
            "/shared/rsaas/ievab2/my_own_physion_preprocessed/dominoes/dominoes_pred.jsonl",
            "/shared/rsaas/ievab2/my_own_physion_preprocessed/drop/drop_pred.jsonl",
            "/shared/rsaas/ievab2/my_own_physion_preprocessed/link/link_pred.jsonl",
            "/shared/rsaas/ievab2/my_own_physion_preprocessed/roll/roll_pred.jsonl",
        ]
    elif args.out_of_distribution == "yes":
        TASK_JSONL = "/shared/rsaas/ievab2/Physion_full_readout_training/Support/support_pred.jsonl"
    else:
        TASK_JSONL = f"/shared/rsaas/ievab2/Physion_full_readout_training/SPLITS/{args.split}/test.jsonl"

    # Output JSONL
    if args.tiny_test:
        OUT_JSONL = TINY_TEST_OUT
        OUT_DIR = OUT_JSONL.parent
    elif args.out_of_distribution == "my_own_physion":
        base_dir = Path(
            f"/home/ievab2/run_models/FULL_PHYSION_FINETUNING_QWEN/testing/round{args.round}/my_own_physion"
        )

        # If stage == noweights: .../my_own_physion/noweights/epochs{E}/
        # Else:                .../my_own_physion/epochs{E}/
        if args.stage == "noweights":
            OUT_DIR = base_dir / "noweights" / f"epochs{args.epochs}"
        else:
            OUT_DIR = base_dir / f"epochs{args.epochs}"

        OUT_JSONL = OUT_DIR / f"{args.split}_results.jsonl"

    else:
        OUT_DIR = Path(
            f"/home/ievab2/run_models/FULL_PHYSION_FINETUNING_QWEN/testing/round{args.round}/epochs{args.epochs}"
        )
        tag = "OOD_SUPPORT" if args.out_of_distribution == "yes" else "test"
        OUT_JSONL = OUT_DIR / f"{args.split}_{tag}_out.jsonl"

    OUT_DIR.mkdir(parents=True, exist_ok=True)


    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[qwen-ft] split={args.split} stage={args.stage} epochs={args.epochs} round={args.round}")
    print(f"[qwen-ft] using adapter_dir={adapter_dir}")
    print(f"[qwen-ft] writing: {OUT_JSONL}")

    if (not args.tiny_test) and args.out_of_distribution == "my_own_physion":
        if not args.resume:
            if OUT_JSONL.exists():
                OUT_JSONL.unlink()

        for task_jsonl in TASK_JSONLS:
            if not os.path.exists(task_jsonl):
                print(f"[qwen-ft][WARN] Missing file, skipping: {task_jsonl}")
                continue

            print(f"[qwen-ft] Evaluating my_own_physion file: {task_jsonl}")
            eval_task(
                task_jsonl,
                OUT_JSONL,
                adapter_dir=adapter_dir,
                counter_limit=args.limit,
                resume=True,
            )
    else:
        eval_task(
            TASK_JSONL,
            OUT_JSONL,
            adapter_dir=adapter_dir,
            counter_limit=args.limit,
            resume=args.resume,
        )

