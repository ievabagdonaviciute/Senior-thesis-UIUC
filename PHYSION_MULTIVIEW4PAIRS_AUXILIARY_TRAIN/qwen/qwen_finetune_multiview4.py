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

_ASSIST_TAG = "<|assistant|>"

# ================== DATA ==================
def _safe_open_rgb(p: str) -> Image.Image:
    im = Image.open(p)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im

class PhysionQwenDataset(Dataset):
    def __init__(self, jsonl_paths: List[str], max_frames: int = 4):
        self.rows = []
        self.max_frames = max_frames

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

                    # FIX: allow variable number of frames; require at least 1
                    if not isinstance(frames, list) or len(frames) < 1:
                        continue
                    if not q or a is None:
                        continue

                    # Train all with the same weights
                    w = 1.0

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
        content = [{"type": "image", "image": im} for im in images]
        # FIX: remove INSTR; send only the question text
        content.append({"type": "text", "text": (question or "")})
        return [{"role": "user", "content": content}]

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        full_texts = []
        prefix_texts = []
        images_batch = []

        for ex in batch:
            images = ex["images"]
            if len(images) < 1:
                raise ValueError("No images found for sample")

            user_msgs = self._make_user_messages(ex["question"], images)

            prefix = self.processor.apply_chat_template(
                user_msgs, tokenize=False, add_generation_prompt=True
            )

            full_msgs = user_msgs + [{"role": "assistant", "content": [{"type": "text", "text": ex["answer"]}]}]
            full = self.processor.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False
            )

            prefix_texts.append(prefix)
            full_texts.append(full)
            images_batch.append(images)

        full_inputs = self.processor(
            text=full_texts,
            images=images_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )

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

        labels = input_ids.clone()
        for i, L in enumerate(prefix_lens):
            labels[i, : int(L)] = -100

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

        weights = torch.tensor([float(ex["weight"]) for ex in batch], dtype=torch.float32)

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sample_weight": weights,
        }

        for k, v in full_inputs.items():
            if k in ("input_ids", "attention_mask"):
                continue
            out[k] = v

        return out

# ================== TRAINER (WEIGHTED LOSS) ==================
class WeightedLossTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("sample_weight", None)

        outputs = model(**inputs)
        loss = outputs.loss

        if weights is not None:
            logits = outputs.logits
            labels = inputs["labels"]
            vocab = logits.size(-1)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            token_loss = loss_fct(
                shift_logits.view(-1, vocab),
                shift_labels.view(-1)
            ).view(labels.size(0), -1)

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

# ================== MAIN ==================
def main():
    ap = argparse.ArgumentParser()

    # flags + paths to match InternVL changes
    ap.add_argument("--type", type=str, required=True,
                    choices=["G","T","C","GC","GT","TC","GTC", "ALL"],
                    help="Training dataset type")
    ap.add_argument("--split", type=int, required=True, choices=[1,2,3],
                    help="Split number (1/2/3)")
    ap.add_argument("--resume_from", type=str, default=None,
                    help="Optional: checkpoint dir to resume from (e.g., .../checkpoint-500)")

    # keep hyperparams defaults the same
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=1)

    # keep model/sequence knobs (still selecting <=max_frames from however many are present)
    ap.add_argument("--max_frames", type=int, default=4, help="Use at most this many frames per sample (default: 4).")
    ap.add_argument("--max_len", type=int, default=2048)

    # LoRA knobs
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.1)

    args = ap.parse_args()

    # -------- choose data (path logic)
    if args.type == "ALL":
        contact_jsonl  = "/shared/rsaas/ievab2/PHYSION_MULTIVIEW_4PAIRS/contact_dataset.jsonl"
        geometry_jsonl = "/shared/rsaas/ievab2/PHYSION_MULTIVIEW_4PAIRS/geometry_dataset.jsonl"
        time_jsonl     = "/shared/rsaas/ievab2/PHYSION_MULTIVIEW_4PAIRS/time_dataset.jsonl"

        jsonls = [contact_jsonl, geometry_jsonl, time_jsonl]
        train_jsonl = "ALL_3DATASETS"

        print("[INFO] Using JSONLs (ALL):")
        for p in jsonls:
            print(f"       - {p}")
    else:
        train_jsonl = (
            f"/shared/rsaas/ievab2/PHYSION_MULTIVIEW_4PAIRS/SPLITS/"
            f"{args.type}/SPLIT{args.split}_{args.type}.jsonl"
        )
        jsonls = [train_jsonl]
        print(f"[INFO] Using JSONL: {train_jsonl}")

    # -------- dataset (unchanged image passing logic; now supports variable frame counts)
    train_ds = PhysionQwenDataset(
        jsonls,
        max_frames=args.max_frames,
    )

    if len(train_ds) == 0:
        raise RuntimeError(
            f"No training samples loaded from {jsonls}. "
            f"Check file exists and frame_paths lengths >= 1."
        )

    # -------- load processor + model
    print(f"[INFO] Loading Qwen from {MODEL_DIR}")
    local_only = os.path.isdir(MODEL_DIR)

    processor = AutoProcessor.from_pretrained(
        MODEL_DIR, trust_remote_code=True, local_files_only=local_only
    )

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

    # -------- LoRA (unchanged)
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
        lora_norm = torch.stack([p.detach().float().norm().cpu() for _, p in lora_params]).mean()
    print("[LORA] mean param norm (init):", float(lora_norm))
    # ======================

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable params (LoRA): {trainable:,}")

    # -------- output dir (kept matching InternVL naming; still includes max_frames)
    epochs_tag = str(int(args.epochs)) if abs(args.epochs - int(args.epochs)) < 1e-9 else str(args.epochs)
    frames_tag = f"{args.max_frames}frames"

    if args.type == "ALL":
        run_tag = "ALL"
    else:
        run_tag = f"{args.type}_SPLIT{args.split}"

    out_dir = (
        f"/shared/rsaas/ievab2/FULL_PHYSION_checkpoints/qwen/round0/"
        f"MULTIVIEW4/{run_tag}_{epochs_tag}epochs_{frames_tag}"
    )


    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Output directory: {out_dir}")

    # -------- collator
    collate = QwenVLMDataCollator(processor=processor, max_len=args.max_len)

    # -------- sanity check batch
    tmp = next(iter(DataLoader(train_ds, batch_size=args.bsz, shuffle=False, collate_fn=collate)))
    print("[SANITY] keys:", list(tmp.keys()))
    print("[SANITY] input_ids:", tmp["input_ids"].shape, "labels:", tmp["labels"].shape)

    vision_keys = [k for k in tmp.keys() if k not in ("input_ids", "attention_mask", "labels", "sample_weight")]
    print("[SANITY] vision keys:", vision_keys)
    for k in vision_keys:
        v = tmp[k]
        if torch.is_tensor(v):
            print(f"[SANITY] {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"[SANITY] {k}: type={type(v)}")

    # ===== IMAGE ABLATION TEST (unchanged) =====
    model.eval()
    with torch.no_grad():
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
    # =========================================

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
        "type": args.type,
        "split": args.split,
        "train_jsonl": train_jsonl,
        "max_frames": args.max_frames,
        "max_len": args.max_len,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "weights": "uniform_1.0",
        "resume_from": args.resume_from,
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
