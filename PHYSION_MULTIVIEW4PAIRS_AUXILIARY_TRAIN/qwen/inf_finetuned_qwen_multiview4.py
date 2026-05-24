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

NUM_FRAMES     = 8
MAX_NEW_TOKENS = 128

INSTR = (
    "These 8 images are consecutive frames from a single video in time order (000→007). "
    "Do not explain; just answer the question concisely. "
)

_NEG_RE = re.compile(r"\b(?:no|false|not|won't|will not|doesn't|does not|don't|didn't|did not|can't|cannot)\b", re.I)
_POS_RE = re.compile(r"\b(?:yes|true|yep|yeah|y)\b", re.I)

# ======== MY OWN PHYSION (ONLY) ========
MY_OWN_PHYSION_JSONLS = [
    "/shared/rsaas/ievab2/my_own_physion_preprocessed/collide/collide_pred.jsonl",
    "/shared/rsaas/ievab2/my_own_physion_preprocessed/contain/contain_pred.jsonl",
    "/shared/rsaas/ievab2/my_own_physion_preprocessed/dominoes/dominoes_pred.jsonl",
    "/shared/rsaas/ievab2/my_own_physion_preprocessed/drop/drop_pred.jsonl",
    "/shared/rsaas/ievab2/my_own_physion_preprocessed/link/link_pred.jsonl",
    "/shared/rsaas/ievab2/my_own_physion_preprocessed/roll/roll_pred.jsonl",
]

# Results root requested by you
RESULTS_ROOT = Path("/home/ievab2/run_models/PHYSION_MULTIVIEW4PAIRS_AUXILIARY_TRAIN/qwen/results")


# ================== PATHS (NEW MULTIVIEW4 ADAPTERS) ==================
def build_adapter_dir_multiview4(task_type: str, split_idx: int, epochs: int) -> str:
    return (
        f"/shared/rsaas/ievab2/FULL_PHYSION_checkpoints/qwen/round0/"
        f"MULTIVIEW4/{task_type}_SPLIT{split_idx}_{epochs}epochs_4frames"
    )


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

    lora_params = [(n, p) for n, p in peft_model.named_parameters() if "lora_" in n]
    meta_cnt = sum(int(getattr(p, "is_meta", False)) for _, p in lora_params)

    print(f"[qwen-ft] manual adapter load: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    print(f"[qwen-ft] lora params: {len(lora_params)}  meta={meta_cnt}", flush=True)
    if len(lora_params) > 0:
        mean_abs = torch.stack([p.detach().abs().mean().cpu() for _, p in lora_params[: min(10, len(lora_params))]]).mean()
        print(f"[qwen-ft] mean(|lora|) sample={float(mean_abs):.6g}", flush=True)

    if meta_cnt > 0:
        raise RuntimeError(
            "LoRA params are still on meta after load. "
            "This usually happens when the base model was initialized with meta tensors. "
            "Load the base on CPU first (device_map={'': 'cpu'}, low_cpu_mem_usage=False)."
        )


def normalize_yesno(text: str) -> str:
    t = (text or "").strip().lower()
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

    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1]

    text = re.sub(r"\b(system|user|assistant)\b", "", text, flags=re.I)
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
        low_cpu_mem_usage=False,
    )
    base.config.use_cache = False

    if adapter_dir is not None:
        adapter_cfg = Path(adapter_dir) / "adapter_config.json"
        if not adapter_cfg.exists():
            raise RuntimeError(f"adapter_config.json not found in {adapter_dir}")

        print(f"[qwen] attaching LoRA adapter from {adapter_dir} …", flush=True)

        peft_model = PeftModel.from_pretrained(base, adapter_dir, local_files_only=True)
        _force_load_adapter_weights_in_memory(peft_model, adapter_dir)
        model = peft_model
    else:
        model = base

    if torch.cuda.is_available():
        model = model.to("cuda")

    model.eval()
    print("[qwen] model ready. cuda?", torch.cuda.is_available(), flush=True)
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
            [{"type": "text",
              "text": "These 8 images are consecutive frames from a single video in time order (000→007). "
                      "Do not explain; just answer the question concisely. "
                      + (question or "")}]
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

                # ===== keep this block exactly here =====
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

    # NEW: select MULTIVIEW4 adapter by type/split/epochs (like InternVL)
    parser.add_argument("--type", type=str, required=True,
                        choices=["G","T","C","GC","GT","TC","GTC"])
    parser.add_argument("--split", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--epochs", type=int, required=True, choices=[1, 3, 5])

    args = parser.parse_args()

    adapter_dir = build_adapter_dir_multiview4(args.type, args.split, args.epochs)

    # result dir
    out_dir = RESULTS_ROOT / f"epochs{args.epochs}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / f"{args.type}_SPLIT{args.split}_out.jsonl"


    print(f"[qwen-ft] type={args.type} split={args.split} epochs={args.epochs}")
    print(f"[qwen-ft] using adapter_dir={adapter_dir}")
    print(f"[qwen-ft] writing: {out_jsonl}")
    print("[qwen-ft] evaluating ONLY my_own_physion JSONLs:")
    for p in MY_OWN_PHYSION_JSONLS:
        print("  -", p)

    # If NOT resuming, wipe outputs ONCE before looping categories
    if not args.resume:
        if out_jsonl.exists():
            out_jsonl.unlink()

    # Always append across categories so earlier categories are preserved
    for task_jsonl in MY_OWN_PHYSION_JSONLS:
        if not os.path.exists(task_jsonl):
            print(f"[qwen-ft][WARN] Missing file, skipping: {task_jsonl}")
            continue

        print(f"[qwen-ft] Evaluating my_own_physion file: {task_jsonl}")
        eval_task(
            task_jsonl,
            out_jsonl,
            adapter_dir=adapter_dir,
            counter_limit=args.limit,
            resume=True,   # append across categories safely
        )

# example run:
#python /home/ievab2/run_models/PHYSION_MULTIVIEW4PAIRS_AUXILIARY_TRAIN/qwen/qwen_inf_multiview4_myphysion.py \
#   --type GT --split 3 --epochs 5
