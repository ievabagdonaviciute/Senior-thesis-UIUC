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

# --- env / config ---

def build_dir_paths(eval_type):

# ================================ NOWEIGHTS: ================================

    if eval_type == "base_model":
        base_past_merged = MODEL_DIR
        base_pred_merged = MODEL_DIR
        out_dir = Path("/home/ievab2/run_models/Physion_finetuning/INTERNVL/round0_base_model")

    elif eval_type == "noweights_both_1epochs_4frames":
        base_past_merged = "/shared/rsaas/ievab2/checkpoints/internvl/round0/noweights_both_past_1epochs_4frames"
        base_pred_merged = "/shared/rsaas/ievab2/checkpoints/internvl/round0/noweights_both_pred_1epochs_4frames"
        out_dir = Path("/home/ievab2/run_models/Physion_finetuning/INTERNVL/noweights_both_1epochs_4frames")
    
    elif eval_type == "noweights_both_3epochs_4frames":
        base_past_merged = "/shared/rsaas/ievab2/checkpoints/internvl/round0/noweights_both_past_3epochs_4frames"
        base_pred_merged = "/shared/rsaas/ievab2/checkpoints/internvl/round0/noweights_both_pred_3epochs_4frames"
        out_dir = Path("/home/ievab2/run_models/Physion_finetuning/INTERNVL/noweights_both_3epochs_4frames")
    
    elif eval_type == "noweights_both_5epochs_4frames":
        base_past_merged = "/shared/rsaas/ievab2/checkpoints/internvl/round0/noweights_both_past_5epochs_4frames"
        base_pred_merged = "/shared/rsaas/ievab2/checkpoints/internvl/round0/noweights_both_pred_5epochs_4frames"
        out_dir = Path("/home/ievab2/run_models/Physion_finetuning/INTERNVL/noweights_both_5epochs_4frames")

# ================================ WITH WEIGHT DIFFERENCES: ================================
    elif eval_type == "both_1epochs_4frames":
        base_past_merged = "/shared/rsaas/ievab2/checkpoints/internvl/round0/both_past_1epochs_4frames"
        base_pred_merged = "/shared/rsaas/ievab2/checkpoints/internvl/round0/both_pred_1epochs_4frames"
        out_dir = Path("/home/ievab2/run_models/Physion_finetuning/INTERNVL/both_1epochs_4frames")

    elif eval_type == "both_3epochs_4frames":
        base_past_merged = "/shared/rsaas/ievab2/checkpoints/internvl/round0/both_past_3epochs_4frames"
        base_pred_merged = "/shared/rsaas/ievab2/checkpoints/internvl/round0/both_pred_3epochs_4frames"
        out_dir = Path("/home/ievab2/run_models/Physion_finetuning/INTERNVL/both_3epochs_4frames")

    elif eval_type == "both_5epochs_4frames":
        base_past_merged = "/shared/rsaas/ievab2/checkpoints/internvl/round0/both_past_5epochs_4frames"
        base_pred_merged = "/shared/rsaas/ievab2/checkpoints/internvl/round0/both_pred_5epochs_4frames"
        out_dir = Path("/home/ievab2/run_models/Physion_finetuning/INTERNVL/both_5epochs_4frames")

    return base_past_merged, base_pred_merged, out_dir
# ====================================================================================
BASE_TOKENIZER   = "/home/ievab2/models/InternVL2-8B"  # base tokenizer

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_DIR      = "/home/ievab2/models/InternVL2-8B"
NUM_FRAMES     = 8
MAX_NEW_TOKENS = 128
INPUT_SIZE     = 448

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ---------- helpers ----------
_NEG_RE = re.compile(r"\b(?:no|false|not|won't|will not|doesn't|does not|don't|didn't|did not|can't|cannot)\b", re.I)
_POS_RE = re.compile(r"\b(?:yes|true|yep|yeah|y)\b", re.I)

def greedy_generate_with_pixels(tokenizer, model, pixel_values, prompt: str, max_new_tokens: int) -> str:
    """
    Minimal greedy decoder that does not rely on .generate() or .chat().
    Works with InternVLChatModel (or LoRA-wrapped) as long as .forward()
    accepts (input_ids, pixel_values) and returns logits.
    """
    device = next(model.parameters()).device

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # Make sure pixel_values has batch dimension: (1, T, C, H, W)
    if pixel_values.dim() == 4:
        pixel_values = pixel_values.unsqueeze(0)
    pixel_values = pixel_values.to(device)

    eos_id = tokenizer.eos_token_id
    generated_ids = []

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_ids=input_ids, pixel_values=pixel_values)
            logits = out.logits[:, -1, :]  # last token
            next_id = logits.argmax(dim=-1)  # greedy

        next_token_id = next_id.item()
        generated_ids.append(next_token_id)

        # append to input_ids for next step
        next_id = next_id.unsqueeze(0)  # (1, 1)
        input_ids = torch.cat([input_ids, next_id], dim=1)

        if eos_id is not None and next_token_id == eos_id:
            break

    # Decode only the generated tail
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip()


def _debug_vocab(tokenizer, model, prompt: str, k: int):
    # 1) the prompt should have k <image> markers
    img_cnt = prompt.count("<image>")
    assert img_cnt == k, f"mismatch: found {img_cnt} <image> tokens but k={k}"

    # 2) sanity-check ids stay within embedding rows
    enc = tokenizer(prompt, add_special_tokens=False, return_tensors=None)
    ids = enc["input_ids"]
    max_id = max(ids) if ids else -1
    emb_n = model.get_input_embeddings().weight.shape[0]
    print(f"[DBG] vocab_n={emb_n}  max_token_id_in_prompt={max_id}  n_tokens={len(ids)}")
    if max_id >= emb_n:
        raise RuntimeError(f"Token id {max_id} >= vocab size {emb_n} (embedding OOB).")

    # 3) '<image>' may NOT be a vocab token (that’s OK for InternVL’s chat())
    image_id = tokenizer.convert_tokens_to_ids("<image>")
    print(f"[DBG] '<image>' id: {image_id} (OK if UNK)")

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
    frames = row.get("frames") or []
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

# ---------- model ----------
from peft import PeftModel, LoraConfig
import os
def _load_model(model_dir: str):
    """
    model_dir can be either:
      - a full InternVL model dir, or
      - an adapter dir containing adapter_config.json + adapter_model.safetensors
    """
    print(f"[internvl2] loading from {model_dir} …", flush=True)

    adapter_cfg = os.path.join(model_dir, "adapter_config.json")
    is_adapter = os.path.exists(adapter_cfg)

    tok_dir = model_dir if is_adapter else BASE_TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(
        tok_dir, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if is_adapter:
        # ---- load base InternVL chat model ----
        base = AutoModel.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=use_dtype,
            device_map="auto",
            attn_implementation="eager",
        )

        # ---- read + sanitize adapter_config.json for LoraConfig ----
        import json
        with open(adapter_cfg, "r") as f:
            raw = json.load(f)

        print("[internvl2] adapter_config keys:", sorted(raw.keys()))

        allowed_keys = set(LoraConfig.__dataclass_fields__.keys())
        clean = {k: v for k, v in raw.items() if k in allowed_keys}
        dropped = sorted(set(raw.keys()) - set(clean.keys()))

        print("[internvl2] using LoraConfig keys:", sorted(clean.keys()))
        print("[internvl2] dropping keys:", dropped)

        lc = LoraConfig(**clean)

        # ---- wrap base with LoRA adapters ----
        model = PeftModel.from_pretrained(base, model_dir, config=lc)

        # ---- monkey-patch missing set_output_embeddings on inner chat model ----
        try:
            inner = model.base_model  # e.g. LoraModel
            if hasattr(inner, "model"):
                inner = inner.model      # InternVLChatModel

            if not hasattr(inner, "set_output_embeddings"):
                print("[PATCH] Adding dummy set_output_embeddings to", type(inner))
                def _set_output_embeddings(self, new_embeds):
                    # We keep the old LM head; we only needed to resize input embeddings
                    return
                inner.set_output_embeddings = _set_output_embeddings.__get__(inner, inner.__class__)
        except Exception as e:
            print(f"[WARN] could not patch set_output_embeddings: {e}")

    else:
        # full-model case (also chat model)
        model = AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=use_dtype,
            device_map="auto",
            attn_implementation="eager",
        )

    # ---- ensure embeddings match tokenizer size ----
    emb_n = model.get_input_embeddings().weight.shape[0]
    vs = len(tokenizer)
    if vs > emb_n:
        print(f"[PATCH] Resizing token embeddings from {emb_n} → {vs}")
        # transformers 4.41.2: no mean_resizing kwarg
        model.resize_token_embeddings(vs)
    elif vs < emb_n:
        print(f"[WARN] tokenizer vocab ({vs}) < model embeddings ({emb_n})")

    model.eval()
    print(f"[internvl2] ready (adapter={is_adapter})")
    return tokenizer, model



# ---------- ask ----------
def ask_internvl2(tokenizer, model, frame_paths: List[str], question: str) -> tuple[str, str, List[str]]:
    user_text = (
        "You see 8 consecutive frames of a video in temporal order. "
        "Do not explain; just answer the question concisely. "
    ) + (question or "")

    pixel_values = load_frames_tensor_from_paths([Path(p) for p in frame_paths], input_size=INPUT_SIZE)
    k = pixel_values.shape[0]
    prompt = ("<image>\n" * k) + user_text

    # Debug; once you're confident, you can comment this out
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



# ---------- main loop ----------
def eval_task(task_path: str, out_path: str, counter_limit: Optional[int] = None, resume: bool = False, base_dir: Optional[str] = None):    
    print("[internvl2] Starting task …", flush=True)
    tokenizer, model = _load_model(base_dir or MODEL_DIR)

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

                frames = row.get("frames")
                q = row.get("question")
                if not isinstance(frames, list) or len(frames) != NUM_FRAMES:
                    raise ValueError("Row must have key 'frames' with exactly 8 image paths")
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

                out_record = dict(row)
                out_record["frame_paths"] = used_paths
                out_record["prompt_given_to_model"] = prompt
                out_record["model_output_raw"] = pred
                out_record["model_output_norm"] = normalize_yesno(pred)
                out_record["model_dir_used"] = base_dir

                f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                f_out.flush()
                written += 1
                print(f"[internvl2] wrote {written}", flush=True)

            except Exception as e:
                print(f"[internvl2][ERROR] row {i}: {e}", flush=True)

    print(f"[internvl2] Done. Wrote {written} rows to {out_path}", flush=True)

# ---------- entry ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Skip entries already in OUT_JSONL and append new results")
    parser.add_argument("--task", choices=["pred", "past"], required=True,
                        help="Which task to run: pred / past?")
    parser.add_argument("--category", choices=["Dominoes","Contain","Drop"],
                        required=True, help="Physion category")
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows to run")
    parser.add_argument("--eval_type", choices=["base_model", "noweights_both_1epochs_4frames", "noweights_both_3epochs_4frames", 
                                                "noweights_both_5epochs_4frames", "both_1epochs_4frames", "both_3epochs_4frames", "both_5epochs_4frames"],
                        required=True, help="Tyoe (epochs + frames)")
    args = parser.parse_args()

    base_past_merged, base_pred_merged, out_dir = build_dir_paths(args.eval_type)

    # normalize names
    cat_folder = args.category[0].upper() + args.category[1:].lower()
    cat_lower  = args.category.lower()

    # input jsonl (questions)
    TASK_JSONL = f"/home/ievab2/run_models/Physion_dataset/physion_out_questions/{cat_folder}/{cat_lower}_{args.task}.jsonl"

    # output jsonl (predictions)
    OUT_DIR = out_dir / cat_folder
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSONL = OUT_DIR / f"{cat_lower}_{args.task}_out.jsonl"

    print(f"[internvl2] category={cat_folder} task={args.task}")
    print(f"[internvl2] reading:  {TASK_JSONL}")
    print(f"[internvl2] writing:  {OUT_JSONL}")

    BASE = base_past_merged if args.task == "past" else base_pred_merged
    print(f"[internvl2] using model_dir={BASE}")

    eval_task(TASK_JSONL, OUT_JSONL, counter_limit=args.limit, resume=args.resume, base_dir=BASE)
