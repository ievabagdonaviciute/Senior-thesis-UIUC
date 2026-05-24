#!/usr/bin/env python3
import os, json, argparse, re
from pathlib import Path
from typing import Optional, Set, List, Dict
from safetensors.torch import load_file as safetensors_load_file

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from PIL import Image

# ================== ENV ==================
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_MODEL_DIR = "/home/ievab2/models/Qwen2.5-VL-7B-Instruct"

NUM_FRAMES = 8
MAX_NEW_TOKENS = 128

# ================== AUX DATASETS ==================
DATASET_PATHS = {
    "contact":  "/shared/rsaas/ievab2/PHYSION_MULTIVIEW_4PAIRS/contact_dataset.jsonl",
    "geometry": "/shared/rsaas/ievab2/PHYSION_MULTIVIEW_4PAIRS/geometry_dataset.jsonl",
    "time":     "/shared/rsaas/ievab2/PHYSION_MULTIVIEW_4PAIRS/time_dataset.jsonl",
}

RESULTS_ROOT = Path(
    "/home/ievab2/run_models/"
    "PHYSION_MULTIVIEW4PAIRS_AUXILIARY_TRAIN/"
    "text_on_multiviewGTC/qwen/results"
)

# ================== ADAPTER PATH ==================
def build_adapter_dir_multiview4(task_type: str, split_idx: int, epochs: int) -> str:
    return (
        f"/shared/rsaas/ievab2/FULL_PHYSION_checkpoints/qwen/round0/"
        f"MULTIVIEW4/{task_type}_SPLIT{split_idx}_{epochs}epochs_4frames"
    )

# ================== HELPERS ==================
def _resume_key(row: dict, fallback_i: int) -> str:
    # Prefer true IDs if present
    qid = row.get("qid")
    if qid is not None:
        return f"qid::{qid}"   # preserves int vs str identity in the string

    rid = row.get("id")
    if rid is not None:
        return f"id::{rid}"

    # Fallback ONLY for skipping; do not write this into output
    frames = row.get("frame_paths") or row.get("frames") or []
    q = row.get("question") or ""
    first = frames[0] if frames else ""
    last  = frames[-1] if frames else ""
    return f"f::{first}|{last}||q::{q}"

def _load_done_keys(out_path: Path) -> Set[str]:
    done: Set[str] = set()
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
            done.add(_resume_key(rec, fallback_i=-1))
    return done

def _open_rgb(p: str) -> Image.Image:
    with Image.open(p) as im:
        im = im.convert("RGB") if im.mode != "RGB" else im
        return im.copy()

def normalize_answer(dataset_kind: str, text: str) -> str:
    t = (text or "").strip().lower()
    first = t.split()[0] if t else ""

    if dataset_kind == "time":
        if first in ("1", "2"):
            return first
        return "unknown"

    if first in ("yes", "y", "true"):
        return "yes"
    if first in ("no", "n", "false"):
        return "no"

    return "unknown"

def _validate_frame_paths(paths: List[str]):
    bad = [p for p in paths if not p or not os.path.isabs(p) or not os.path.exists(p)]
    if bad:
        raise FileNotFoundError(f"Bad/missing frame paths: {bad[:3]}")

def _extract_assistant(text: str) -> str:
    if not text:
        return ""
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1]
    text = re.sub(r"\b(system|user|assistant)\b", "", text, flags=re.I)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else ""

# ================== LORA REMAP ==================
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
    print(f"[qwen-ft] adapter load: missing={len(missing)} unexpected={len(unexpected)}")

# ================== MODEL LOAD ==================
def _load_model(adapter_dir: Optional[str] = None):
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True
    )

    base = AutoModelForVision2Seq.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=False,
    )

    base.config.use_cache = False

    if adapter_dir is not None:
        print(f"[qwen-ft] attaching adapter {adapter_dir}")
        peft_model = PeftModel.from_pretrained(base, adapter_dir, local_files_only=True)
        _force_load_adapter_weights_in_memory(peft_model, adapter_dir)
        model = peft_model
    else:
        model = base

    if torch.cuda.is_available():
        model = model.to("cuda")

    model.eval()
    
    if adapter_dir is not None:
        s = 0.0
        n = 0
        for name, p in model.named_parameters():
            if "lora_" in name:
                s += float(p.abs().sum().item())
                n += 1
        print(f"[qwen-ft] lora_params={n} lora_abs_sum={s:.3e}")

    return processor, model

# ================== INFERENCE ==================
def ask_qwen(processor, model, frame_paths: List[str], question: str, dataset_kind: str):
    imgs = [_open_rgb(p) for p in frame_paths]

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
    else:
        raise ValueError(f"Unknown dataset_kind={dataset_kind}")

    messages = [{
        "role": "user",
        "content": (
            [{"type": "image", "image": im} for im in imgs] +
            [{"type": "text", "text": user_text}]
        ),
    }]

    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

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

# ================== EVAL ==================
def eval_task(
    dataset_path: str,
    out_path: Path,
    adapter_dir: str,
    dataset_kind: str,
    limit: Optional[int] = None,
    resume: bool = False,
):
    """
    Resume behavior:
      - If --resume and out_path exists: read existing OUT JSONL and build a set of "done keys"
      - Skip rows whose key is already present
      - Append new outputs (mode='a')
    IMPORTANT: This does NOT modify the JSONL schema:
      - It does NOT overwrite/add "qid" or "id"
      - It writes dict(row) exactly + your added output fields
    """
    processor, model = _load_model(adapter_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"

    done_keys: Set[str] = _load_done_keys(out_path) if resume else set()
    if resume:
        print(f"[qwen-ft] resume=1 found {len(done_keys)} existing entries in {out_path}")

    written = 0
    skipped = 0

    with open(dataset_path, "r", encoding="utf-8") as f_in, \
         out_path.open(mode, encoding="utf-8") as f_out:

        for i, line in enumerate(f_in):
            if not line.strip():
                continue
            if limit is not None and written >= limit:
                break

            row = json.loads(line)

            key = _resume_key(row, i)
            if resume and key in done_keys:
                skipped += 1
                continue

            frames = row.get("frame_paths") or row.get("frames")
            q = row.get("question")

            if not isinstance(frames, list) or len(frames) < 1:
                raise ValueError("Row must have >=1 frame in frame_paths/frames")
            if not q:
                raise ValueError("Row missing question")
            _validate_frame_paths(frames)

            pred, prompt_text = ask_qwen(
                processor,
                model,
                frames,
                q,
                dataset_kind
            )

            # IMPORTANT: keep schema the same — start from row unchanged
            out_record = dict(row)
            out_record["prompt_given_to_model"] = prompt_text
            out_record["model_output_raw"] = pred
            out_record["model_output_norm"] = normalize_answer(dataset_kind, pred)
            out_record["model_dir_used"] = adapter_dir

            f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            f_out.flush()

            written += 1
            if resume:
                done_keys.add(key)

            print(f"[qwen-ft] wrote {written} (skipped {skipped})")

    print(f"[qwen-ft] Done. Wrote {written} new rows (skipped {skipped}) to {out_path}")
# ================== ENTRY ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--type", required=True,
                        choices=["G","T","C","GC","GT","TC","GTC"])
    parser.add_argument("--split", required=True, type=int, choices=[1,2,3])
    parser.add_argument("--epochs", required=True, type=int, choices=[1,3,5])
    parser.add_argument("--dataset_kind", required=True,
                        choices=["contact","geometry","time"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Append to existing out_jsonl and skip qids already present")
    args = parser.parse_args()

    adapter_dir = build_adapter_dir_multiview4(
        args.type,
        args.split,
        args.epochs
    )

    dataset_path = DATASET_PATHS[args.dataset_kind]

    out_dir = RESULTS_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / (
        f"{args.type}_SPLIT{args.split}_{args.dataset_kind}_out.jsonl"
    )

    print("type:", args.type)
    print("split:", args.split)
    print("epochs:", args.epochs)
    print("dataset:", args.dataset_kind)
    print("adapter:", adapter_dir)
    print("writing:", out_jsonl)

    eval_task(
        dataset_path,
        out_jsonl,
        adapter_dir,
        args.dataset_kind,
        args.limit,
        resume=args.resume
    )

# HOW TO RUN:
# python /home/ievab2/run_models/PHYSION_MULTIVIEW4PAIRS_AUXILIARY_TRAIN/text_on_multiviewGTC/qwen/eval_finetuned_qwen_aux.py \
#   --type GTC \
#   --split 1 \
#   --epochs 5 \
#   --dataset_kind geometry