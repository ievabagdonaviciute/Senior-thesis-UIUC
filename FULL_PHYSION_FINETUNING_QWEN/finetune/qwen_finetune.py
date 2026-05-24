#!/usr/bin/env python3
import os, json, argparse, re
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- Force caches to shared location (persistent across nodes) ---
os.environ.setdefault("HF_HOME", "/shared/rsaas/ievab2/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/shared/rsaas/ievab2/hf_cache/hub")
os.environ.setdefault("TORCH_HOME", "/shared/rsaas/ievab2/hf_cache/torch")

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
)

# ================== CONFIG ==================
MODEL_DIR  = "/home/ievab2/models/Qwen2.5-VL-7B-Instruct"
NUM_FRAMES = 8
TINY_JSONL = "/shared/rsaas/ievab2/TINY_PHYSION_TEST/tiny_test.jsonl"

INSTR = (
    "These 8 images are consecutive frames from a single video in time order (000→007). "
    "Do not explain; just answer the question concisely. "
)

# Qwen chat markers exist in decoded text; we use processor.apply_chat_template.
_ASSIST_TAG = "<|assistant|>"

# ================== DATA ==================
def _safe_open_rgb(p: str) -> Image.Image:
    im = Image.open(p)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im

class PhysionQwenDataset(Dataset):
    def __init__(self, jsonl_paths: List[str], require_k: int = 8,
                 max_frames: int = 4, noweights: bool = False):
        self.rows = []
        self.require_k = require_k
        self.max_frames = max_frames
        self.noweights = noweights


        for p in jsonl_paths:
            with open(p, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        row = json.loads(ln)
                    except Exception:
                        continue

                    frames = row.get("frames") or row.get("frame_paths") or []
                    q = row.get("question")
                    a = row.get("answer", row.get("ground_truth", ""))

                    if not isinstance(frames, list) or len(frames) != require_k:
                        continue
                    if not q or a is None:
                        continue

                    w = 1.0 if noweights else float(row.get("weight", 1.0))

                    self.rows.append({
                        "frames": frames,
                        "question": str(q),
                        "answer": str(a).strip().lower(),
                        "weight": float(w),
                        "meta": row,  # keep anything else if needed
                    })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        frame_paths = r["frames"]

        K_all = len(frame_paths)
        use_k = min(K_all, self.max_frames)

        if use_k < K_all:
            import numpy as np
            idxs = np.linspace(0, K_all - 1, num=use_k).round().astype(int).tolist()
        else:
            idxs = list(range(K_all))

        imgs = [_safe_open_rgb(frame_paths[i]) for i in idxs]

        return {
            "images": imgs,
            "question": r["question"],
            "answer": r["answer"],
            "weight": float(r["weight"]),
        }


# ================== COLLATOR ==================
class QwenVLMDataCollator:
    """
    Builds:
      - full_text = chat(user(images+text), assistant(answer))
      - prefix_text = chat(user(images+text), assistant(generation prompt))
    Then labels mask out prefix tokens and supervise only the answer tokens.
    """
    def __init__(self, processor: AutoProcessor, max_len: int = 2048):
        self.processor = processor
        self.max_len = int(max_len)

    def _make_user_messages(self, question: str, images: List[Image.Image]):
        # Qwen expects the images attached as "image" items in the user message
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": INSTR + (question or "")})
        return [{"role": "user", "content": content}]

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Build texts and images per sample
        full_texts = []
        prefix_texts = []
        images_batch = []

        for ex in batch:
            images = ex["images"]
            if len(images) < 1:
                raise ValueError("No images found for sample")

            user_msgs = self._make_user_messages(ex["question"], images)

            # prefix = "user ... <assistant>" (generation prompt)
            prefix = self.processor.apply_chat_template(
                user_msgs, tokenize=False, add_generation_prompt=True
            )

            # full = "user ... <assistant> ANSWER"
            full_msgs = user_msgs + [{"role": "assistant", "content": [{"type": "text", "text": ex["answer"]}]}]
            full = self.processor.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False
            )

            prefix_texts.append(prefix)
            full_texts.append(full)
            images_batch.append(images)

        # Tokenize FULL sequences (these produce input_ids + vision tensors)
        full_inputs = self.processor(
            text=full_texts,
            images=images_batch,          # list of list[PIL]
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )

        # Tokenize PREFIX text-only to get prefix lengths (no images needed to compute text token length)
        # But we must use the same tokenizer; processor handles it.
        # prefix_tok = self.processor.tokenizer(
        #     prefix_texts,
        #     return_tensors="pt",
        #     padding=True,
        #     truncation=True,
        #     max_length=self.max_len,
        #     add_special_tokens=False,
        # )
        # prefix_lens = (prefix_tok["attention_mask"].sum(dim=1)).tolist()  # per-sample prefix length
        
        prefix_tok = self.processor.tokenizer(
            prefix_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=False,
        )
        prefix_lens = prefix_tok["attention_mask"].sum(dim=1).tolist()


        input_ids = full_inputs["input_ids"]
        attention_mask = full_inputs["attention_mask"]

        # Labels = input_ids, but mask out everything up through prefix_len
        labels = input_ids.clone()
        for i, L in enumerate(prefix_lens):
            # mask prefix tokens
            labels[i, : int(L)] = -100
        # ===== DEBUG LABEL MASKING (runs once) =====
        if not hasattr(self, "_did_label_debug"):
            self._did_label_debug = True
            for i in range(min(2, labels.size(0))):
                supervised = (labels[i] != -100).sum().item()
                print("[LABELS] supervised tokens:", supervised)

                L = int(prefix_lens[i])
                window_ids = input_ids[i, max(L-20, 0): L+50].tolist()
                window_txt = self.processor.tokenizer.decode(
                    window_ids, skip_special_tokens=False
                )
                print("[LABELS] boundary decode:\n", window_txt)
        # =========================================

        # weights
        weights = torch.tensor([float(ex["weight"]) for ex in batch], dtype=torch.float32)

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sample_weight": weights,
        }

        # include vision tensors that Qwen returns (names can vary; keep all non-text fields)
        for k, v in full_inputs.items():
            if k in ("input_ids", "attention_mask"):
                continue
            out[k] = v

        return out

# ================== TRAINER (WEIGHTED LOSS) ==================
class WeightedLossTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("sample_weight", None)

        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss

        if weights is not None:
            logits = outputs.logits  # (B,T,V)
            labels = inputs["labels"]
            vocab = logits.size(-1)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            token_loss = loss_fct(
                shift_logits.view(-1, vocab),
                shift_labels.view(-1)
            ).view(labels.size(0), -1)  # (B,T-1)

            valid_counts = shift_labels.ne(-100).sum(dim=1).clamp(min=1)
            per_sample = token_loss.sum(dim=1) / valid_counts

            w = weights.to(per_sample.device).to(per_sample.dtype)
            loss = (per_sample * w).mean()

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, *args, **kwargs) -> None:
        if self.is_world_process_zero():
            loss_file = os.path.join(self.args.output_dir, "train_loss.jsonl")
            record = {
                "step": self.state.global_step,
                "epoch": float(self.state.epoch) if self.state.epoch is not None else None,
                **logs,
            }
            with open(loss_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        super().log(logs, *args, **kwargs)

# ================== UTILS ==================
def list_curriculum_jsonls(round_idx: int, split_name: str, root: str):
    base = Path(root) / f"round{round_idx}"
    return [str(base / f"curriculum_{split_name}_train.jsonl")]

def list_raw_split_jsonls(split_name: str):
    base = Path("/shared/rsaas/ievab2/Physion_full_readout_training/SPLITS")
    return [str(base / split_name / "train.jsonl")]

# ================== MAIN ==================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--split", type=str, default="SPLIT3", choices=["SPLIT1", "SPLIT2", "SPLIT3"])
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=8, help="Use at most this many frames per sample (default: 4).")

    ap.add_argument("--noweights", action="store_true", help="Ignore curriculum weights; use raw SPLITS train.jsonl.")
    ap.add_argument("--curriculum_root", type=str,
                    default="/home/ievab2/run_models/FULL_PHYSION_FINETUNING_QWEN/curriculum",
                    help="Root that contains round{round}/curriculum_SPLITX_train.jsonl")
    ap.add_argument(
        "--tiny_test",
        type=bool,
        default=False,
    )

    # LoRA knobs
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.1)

    # model/sequence knobs
    ap.add_argument("--max_len", type=int, default=2048)

    ap.add_argument("--resume_from", type=str, default=None)
    args = ap.parse_args()

    # -------- choose data
    is_tiny = args.tiny_test

    if is_tiny:
        jsonls = [TINY_JSONL]
        stage_tag = "tiny_test"
        print(f"[INFO] tiny_test=True → using tiny JSONL: {jsonls}")
    else:
        if args.noweights:
            jsonls = list_raw_split_jsonls(args.split)
            stage_tag = "noweights"
            print(f"[INFO] noweights=True → using raw split JSONL: {jsonls}")
        else:
            jsonls = list_curriculum_jsonls(args.round, args.split, args.curriculum_root)
            stage_tag = "curriculum"
            print(f"[INFO] noweights=False → using curriculum JSONL: {jsonls}")

    # -------- dataset
    train_ds = PhysionQwenDataset(
        jsonls,
        require_k=NUM_FRAMES,
        max_frames=args.max_frames,
        noweights=args.noweights,
    )

    # -------- load processor + model
    print(f"[INFO] Loading Qwen from {MODEL_DIR}")
    local_only = os.path.isdir(MODEL_DIR)

    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=local_only)

    # dtype
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_DIR,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=local_only,
    )

    model.config.use_cache = False

    # -------- LoRA
    # Qwen2.5-VL language backbone is Llama-like; these are the usual projection module names.
    # If your model uses different names, print(model) and adjust target_modules accordingly.
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    # ===== LoRA SANITY =====
    lora_params = [(n, p) for n, p in model.named_parameters() if "lora_" in n]
    print("[LORA] sample full names:")
    for n, _ in lora_params[:10]:
        print("   ", n)

    print("[LORA] number of LoRA params:", len(lora_params))
    print("[LORA] example params:", [n for n, _ in lora_params[:5]])

    with torch.no_grad():
        lora_norm = torch.stack([p.detach().float().norm().cpu()
                                for _, p in lora_params]).mean()
    print("[LORA] mean param norm (init):", float(lora_norm))
    # ======================

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable params (LoRA): {trainable:,}")

    # -------- output dir (match InternVL naming)
    epochs_tag = str(int(args.epochs)) if abs(args.epochs - int(args.epochs)) < 1e-9 else str(args.epochs)

    suffix = "_TINY_TEST" if args.tiny_test else ""

    # stage tag for naming ONLY
    if args.tiny_test:
        stage_tag = "tiny_test"
    else:
        stage_tag = "noweights" if args.noweights else "curriculum"

    out_dir = (
        f"/shared/rsaas/ievab2/FULL_PHYSION_checkpoints/qwen/"
        f"round{args.round}/{args.split}_{stage_tag}_{epochs_tag}epochs_{args.max_frames}frames{suffix}"
    )

    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Output directory: {out_dir}")



    # -------- collator
    collate = QwenVLMDataCollator(processor=processor, max_len=args.max_len)

    # -------- sanity check batch
    tmp = next(iter(DataLoader(train_ds, batch_size=args.bsz, shuffle=False, collate_fn=collate)))
    print("[SANITY] keys:", list(tmp.keys()))
    print("[SANITY] input_ids:", tmp["input_ids"].shape, "labels:", tmp["labels"].shape)
    
    # ===== SANITY: vision tensors present? =====
    vision_keys = [k for k in tmp.keys()
                if k not in ("input_ids", "attention_mask", "labels", "sample_weight")]
    print("[SANITY] vision keys:", vision_keys)

    for k in vision_keys:
        v = tmp[k]
        if torch.is_tensor(v):
            print(f"[SANITY] {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"[SANITY] {k}: type={type(v)}")
    # =========================================
    # ===== IMAGE ABLATION TEST =====
    model.eval()
    with torch.no_grad():
        # If device_map="auto" shards the model, this test is not valid.
        if hasattr(model, "hf_device_map") and len(set(model.hf_device_map.values())) > 1:
            print("[IMGTEST] Skipping ablation test (model is sharded due to device_map='auto').")
        else:
            dev = next(model.parameters()).device
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in tmp.items()}

            out1 = model(**{k: v for k, v in batch.items() if k != "sample_weight"})
            loss1 = float(out1.loss.detach().cpu())

            batch2 = dict(batch)
            if "pixel_values" in batch2:
                batch2["pixel_values"] = torch.zeros_like(batch2["pixel_values"])
            elif "pixel_values_videos" in batch2:
                batch2["pixel_values_videos"] = torch.zeros_like(batch2["pixel_values_videos"])
            else:
                raise RuntimeError("[IMGTEST] No pixel_values / pixel_values_videos key found")

            out2 = model(**{k: v for k, v in batch2.items() if k != "sample_weight"})
            loss2 = float(out2.loss.detach().cpu())
            print(f"[IMGTEST] loss normal={loss1:.4f} "
                f"loss zeroed_images={loss2:.4f} "
                f"diff={abs(loss2 - loss1):.4f}")
    # ==============================



    # -------- training args
    training_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,

        save_strategy="epoch",
        save_total_limit=2,
        save_safetensors=True,
        logging_dir=f"{out_dir}/logs",
        logging_steps=20,

        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",

        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),

        report_to="none",
        remove_unused_columns=False,
        seed=42,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        data_collator=collate,
        train_dataset=train_ds,
    )

    # save hyperparams
    hyper = {
        "model_dir": MODEL_DIR,
        "lr": args.lr,
        "bsz": args.bsz,
        "grad_accum": args.grad_accum,
        "epochs": args.epochs,
        "split": args.split,
        "round": args.round,
        "stage": stage_tag,
        "max_len": args.max_len,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }
    with open(os.path.join(out_dir, "hyperparams.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps(hyper) + "\n")

    # train
    if args.resume_from:
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # save adapters + processor
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)

    print(f"[DONE] Training complete. Saved to: {out_dir}")

if __name__ == "__main__":
    main()
