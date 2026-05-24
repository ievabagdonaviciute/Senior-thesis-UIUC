#!/usr/bin/env python3
import os, json, argparse, re, glob

# --- Force caches to shared location (persistent across nodes) ---
os.environ["HF_HOME"] = "/shared/rsaas/ievab2/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/shared/rsaas/ievab2/hf_cache/hub"
os.environ["TORCH_HOME"] = "/shared/rsaas/ievab2/hf_cache/torch"

from pathlib import Path
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    AutoConfig,
)

# ---------- config ----------
MODEL_DIR   = "/home/ievab2/models/InternVL2-8B"
INPUT_SIZE  = 448

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform():
    '''
    Ensures RGB
    Resizes each frame to 448x448 --> that's what InternVL expects
    Converts to tensor
    Normalizes with ImageNet mean/std
    '''
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

# --- dataset: all weights fixed to 1.0; accepts variable frame count (>=1) ---
class PhysionVLMDataset(Dataset):
    def __init__(self, jsonl_paths: List[str], max_frames: int = 8):
        self.max_frames = max_frames
        self.rows = []

        for p in jsonl_paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)

                    frames = row.get("frames") or row.get("frame_paths")
                    q = row.get("question")
                    a = row.get("answer", row.get("ground_truth", ""))

                    # minimal: require at least 1 frame
                    if not isinstance(frames, list) or len(frames) < 1:
                        continue

                    self.rows.append({
                        "frames": frames,
                        "question": q,
                        "answer": a,
                        "weight": 1.0,
                    })

        self.tf = build_transform()
        self.num_skipped_samples = 0
        self.num_bad_images = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        max_tries = 10
        last_err = None

        for attempt in range(max_tries):
            r = self.rows[idx]

            frames_list = r["frames"]
            K_all = len(frames_list)
            use_k = min(K_all, self.max_frames)

            if use_k < K_all:
                idxs = torch.linspace(0, K_all - 1, steps=use_k).round().long().tolist()
            else:
                idxs = list(range(K_all))

            imgs = []
            bad_path = None

            try:
                for i in idxs:
                    pth = frames_list[i]
                    bad_path = pth
                    with Image.open(pth) as img:
                        imgs.append(self.tf(img))

                pixel_values = torch.stack(imgs, dim=0)

                return {
                    "pixel_values": pixel_values,
                    "question": r["question"],
                    "answer": r["answer"],
                    "weight": 1.0,
                }

            except Exception as e:
                last_err = e
                self.num_skipped_samples += 1
                self.num_bad_images += 1

                if attempt == 0:
                    print(
                        f"[WARN] Skipping sample due to bad image:\n"
                        f"       path={bad_path}\n"
                        f"       err={repr(e)}"
                    )

                idx = (idx + 1) % len(self.rows)

        raise RuntimeError(
            f"Too many bad samples encountered. "
            f"Last error: {repr(last_err)}"
        )

# ---------- trainer ----------
import torch.nn as nn
from transformers import Trainer

class WeightedLossTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("sample_weight", None)

        allowed = {"input_ids", "attention_mask", "labels", "pixel_values", "image_flags"}
        clean_inputs = {k: v for k, v in inputs.items() if k in allowed}

        for bad_key in ["inputs_embeds", "position_ids"]:
            if bad_key in clean_inputs:
                clean_inputs.pop(bad_key, None)

        outputs = model(**clean_inputs)
        loss = outputs.loss

        if weights is not None:
            logits = outputs.logits
            labels = clean_inputs["labels"]
            vocab = logits.size(-1)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            token_loss = loss_fct(
                shift_logits.view(-1, vocab),
                shift_labels.view(-1)
            )
            token_loss = token_loss.view(labels.size(0), -1)

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

# ---------- collator ----------
class VLMDataCollator:
    def __init__(self, tokenizer, max_len_cap=4096, image_token="<image>", tokens_per_image=256,
                 image_token_id=None, alt_token_id=None, pixel_dtype=None):
        self.tok = tokenizer
        self.hard_cap = int(max_len_cap)
        self.image_token = image_token
        self.tokens_per_image = int(tokens_per_image)
        self.img_id_primary = int(image_token_id) if image_token_id is not None else None
        self.img_id_alt     = int(alt_token_id)   if alt_token_id   is not None else None
        self.pixel_dtype    = pixel_dtype

    def __call__(self, batch):
        img_token_id_t = self.tok.convert_tokens_to_ids(self.image_token)
        img_ids_to_block = set()
        for cand in (self.img_id_primary, self.img_id_alt, img_token_id_t):
            if isinstance(cand, int) and cand >= 0:
                img_ids_to_block.add(cand)
        if not img_ids_to_block:
            raise ValueError("Could not resolve any valid <image> token id.")

        per_sample_imgs = [b["pixel_values"] for b in batch]
        B = len(batch)
        K = per_sample_imgs[0].shape[0]
        required_img_tokens = K * self.tokens_per_image
        safe_id = self.tok.eos_token_id if self.tok.eos_token_id is not None else self.tok.pad_token_id

        per_sample_ids_full, per_sample_att_full, per_sample_lab_full = [], [], []

        for b in batch:
            k = b["pixel_values"].shape[0]
            assert k == K  # unchanged logic

            # (requested change) removed INSTR; prompt is just the question
            prompt_txt = (b["question"] or "")
            ans_txt    = str(b["answer"] or "")

            enc_prompt = self.tok(prompt_txt, add_special_tokens=False, return_tensors=None)
            enc_answer = self.tok(ans_txt,    add_special_tokens=False, return_tensors=None)
            prompt_ids = enc_prompt["input_ids"]
            answer_ids = enc_answer["input_ids"]

            if img_ids_to_block:
                prompt_ids = [(safe_id if tid in img_ids_to_block else tid) for tid in prompt_ids]
                answer_ids = [(safe_id if tid in img_ids_to_block else tid) for tid in answer_ids]

            placeholder_id = self.img_id_primary if self.img_id_primary is not None else img_token_id_t
            image_placeholders = [placeholder_id] * required_img_tokens

            ids_in  = image_placeholders + prompt_ids
            ids_out = answer_ids
            ids_full    = ids_in + ids_out
            att_full    = [1] * len(ids_full)
            labels_full = ([-100] * len(ids_in)) + ids_out

            per_sample_ids_full.append(ids_full)
            per_sample_att_full.append(att_full)
            per_sample_lab_full.append(labels_full)

        max_len_batch = max(max(len(x), required_img_tokens) for x in per_sample_ids_full)
        if max_len_batch > self.hard_cap:
            if required_img_tokens > self.hard_cap:
                raise RuntimeError(
                    f"Need ≥{required_img_tokens} tokens for {K} images "
                    f"({self.tokens_per_image}/img) but hard_cap={self.hard_cap}."
                )
            max_len_batch = self.hard_cap

        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        input_ids_list, attention_list, labels_list = [], [], []
        for ids_full, att_full, lab_full in zip(per_sample_ids_full, per_sample_att_full, per_sample_lab_full):
            ids_full = ids_full[:max_len_batch]
            att_full = att_full[:max_len_batch]
            lab_full = lab_full[:max_len_batch]
            if len(ids_full) < max_len_batch:
                pad_n = max_len_batch - len(ids_full)
                ids_full += [pad_id] * pad_n
                att_full += [0] * pad_n
                lab_full += [-100] * pad_n

            input_ids_list.append(torch.tensor(ids_full, dtype=torch.long))
            attention_list.append(torch.tensor(att_full, dtype=torch.long))
            labels_list.append(torch.tensor(lab_full, dtype=torch.long))

        input_ids = torch.stack(input_ids_list, dim=0)
        attention = torch.stack(attention_list, dim=0)
        labels    = torch.stack(labels_list, dim=0)

        pixels = torch.cat(per_sample_imgs, dim=0)
        if self.pixel_dtype is not None:
            pixels = pixels.to(self.pixel_dtype)
        image_flags = torch.ones((B * K,), dtype=torch.bool)

        weights = torch.ones((B,), dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
            "pixel_values": pixels,
            "image_flags": image_flags,
            "sample_weight": weights,
        }

def main():
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ap = argparse.ArgumentParser()

    ap.add_argument("--type", type=str, required=True,
                    choices=["G","T","C","GC","GT","TC","GTC","ALL"],
                    help="Training dataset type")
    ap.add_argument("--split", type=int, required=True, choices=[1,2,3],
                    help="Split number (1/2/3)")
    ap.add_argument("--resume_from", type=str, default=None,
                    help="Optional: checkpoint dir to resume from (e.g., .../checkpoint-500)")

    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bsz", type=int, default=1)

    ap.add_argument("--lora_cap", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.1)

    ap.add_argument("--max_frames", type=int, default=4, help="Use at most this many frames per sample")

    args = ap.parse_args()

    if args.type == "ALL":
        contact_jsonl  = "/shared/rsaas/ievab2/PHYSION_MULTIVIEW_4PAIRS/contact_dataset.jsonl"
        geometry_jsonl = "/shared/rsaas/ievab2/PHYSION_MULTIVIEW_4PAIRS/geometry_dataset.jsonl"
        time_jsonl     = "/shared/rsaas/ievab2/PHYSION_MULTIVIEW_4PAIRS/time_dataset.jsonl"

        jsonls = [contact_jsonl, geometry_jsonl, time_jsonl]
        print("[INFO] Using JSONLs (ALL):")
        for p in jsonls:
            print(f"       - {p}")

        train_jsonl = "ALL_3DATASETS"
    else:
        train_jsonl = (
            f"/shared/rsaas/ievab2/PHYSION_MULTIVIEW_4PAIRS/SPLITS/"
            f"{args.type}/SPLIT{args.split}_{args.type}.jsonl"
        )
        jsonls = [train_jsonl]
        print(f"[INFO] Using JSONL: {train_jsonl}")

    train_ds = PhysionVLMDataset(
        jsonls,
        max_frames=args.max_frames,                 # still used for non-ALL (and as fallback)
    )


    if len(train_ds) == 0:
        raise RuntimeError(
            f"No training samples loaded from {jsonls}. "
            f"Check file exists and frame paths are valid."
        )

    if train_ds.num_skipped_samples > 0:
        print(
            "\n[DATA WARNING]\n"
            f"  Skipped samples due to bad images: {train_ds.num_skipped_samples}\n"
            f"  Bad image files encountered:      {train_ds.num_bad_images}\n"
            "  Training continued safely.\n"
        )
    else:
        print("\n[DATA CHECK] No corrupted images encountered.\n")

    print(f"[INFO] Loaded {len(train_ds)} samples from JSONLs: {jsonls}")

    cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    IMAGE_TOKEN = getattr(cfg, "image_token", "<image>")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    added_tokens = 0
    if IMAGE_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
        added_tokens = 1

    load_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=load_dtype,
        device_map="auto",
        attn_implementation="eager",
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    if added_tokens:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    # ---- InternVLChatConfig has no 'vocab_size', but PEFT expects it when saving ----
    config_cls = model.config.__class__
    vs = len(tokenizer)
    if not hasattr(model.config, "vocab_size"):
        print(f"[PATCH] Setting model.config.vocab_size = {vs}")
        model.config.vocab_size = vs

    orig_cfg_from_pretrained = config_cls.from_pretrained

    @classmethod
    def patched_from_pretrained(cls, *args, **kwargs):
        cfg2 = orig_cfg_from_pretrained(*args, **kwargs)
        if not hasattr(cfg2, "vocab_size"):
            cfg2.vocab_size = vs
        return cfg2

    if getattr(config_cls, "_vocab_size_patched", False) is not True:
        print(f"[PATCH] Monkey-patching {config_cls.__name__}.from_pretrained to inject vocab_size={vs}")
        config_cls.from_pretrained = patched_from_pretrained
        config_cls._vocab_size_patched = True

    tok_image_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    mdl_image_id = (
        getattr(model, "image_token_index", None)
        or getattr(getattr(model, "config", None), "image_token_index", None)
        or tok_image_id
    )

    TOKENS_PER_IMAGE = (
        getattr(model, "image_token_len", None)
        or getattr(getattr(model, "config", None), "image_token_len", None)
        or getattr(cfg, "image_token_len", None)
        or getattr(cfg, "num_image_tokens", None)
        or getattr(getattr(cfg, "vision_config", {}), "num_image_tokens", None)
        or 256
    )
    print(f"[DEBUG] IMAGE_TOKEN='{IMAGE_TOKEN}', tok_image_id={tok_image_id}, mdl_image_id={mdl_image_id}, TOKENS_PER_IMAGE={TOKENS_PER_IMAGE}")

    lora_cfg = LoraConfig(
        r=args.lora_cap,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["wqkv", "wo"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    base = getattr(model, "base_model", None)
    if base is None:
        raise RuntimeError("Unexpected PEFT structure: model.base_model is None")

    inner = getattr(base, "model", None) or getattr(base, "language_model", None) or base
    print(f"[DEBUG] Patching inner forward on: {inner.__class__.__name__}")

    if hasattr(inner, "img_context_token_id"):
        inner.img_context_token_id = mdl_image_id
        print(f"[PATCH] Set inner.img_context_token_id = {inner.img_context_token_id}")
    else:
        print("[WARN] inner has no img_context_token_id attribute")

    orig_inner_forward = inner.forward

    def inner_patched_forward(*args, **kwargs):
        if "inputs_embeds" in kwargs:
            kwargs = dict(kwargs)
            kwargs.pop("inputs_embeds", None)
        return orig_inner_forward(*args, **kwargs)

    inner.forward = inner_patched_forward

    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("No trainable parameters detected — LoRA not applied correctly")

    print(f"[DEBUG] Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    model.config.use_cache = False

    epochs_tag = str(int(args.epochs)) if abs(args.epochs - int(args.epochs)) < 1e-9 else str(args.epochs)
    frames_tag = f"{args.max_frames}frames"
    if args.type == "ALL":
        out_dir = (
            f"/shared/rsaas/ievab2/FULL_PHYSION_checkpoints/internvl/round0/"
            f"MULTIVIEW4/ALL_{epochs_tag}epochs_{frames_tag}"
        )
    else:
        out_dir = (
            f"/shared/rsaas/ievab2/FULL_PHYSION_checkpoints/internvl/round0/"
            f"MULTIVIEW4/{args.type}_SPLIT{args.split}_{epochs_tag}epochs_{frames_tag}"
        )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Output directory set to {out_dir}")

    model_max = getattr(cfg, "max_position_embeddings", None)
    if not isinstance(model_max, int) or model_max <= 0 or model_max > 32768:
        model_max = 4096
    model_max = min(model_max, 2048)

    model_dtype = next(model.parameters()).dtype

    collate = VLMDataCollator(
        tokenizer,
        max_len_cap=model_max,
        image_token=IMAGE_TOKEN,
        tokens_per_image=TOKENS_PER_IMAGE,
        image_token_id=mdl_image_id,
        alt_token_id=tok_image_id,
        pixel_dtype=model_dtype,
    )

    print(f"[INFO] Starting training --> output dir: {out_dir}")

    tmp_loader = DataLoader(train_ds, batch_size=args.bsz, shuffle=False, collate_fn=collate)
    tmp_batch = next(iter(tmp_loader))

    print("[SANITY] shapes:",
          "input_ids", tmp_batch["input_ids"].shape,
          "pixel_values", tmp_batch["pixel_values"].shape,
          "image_flags", tmp_batch["image_flags"].shape)

    print("[SANITY] Collator OK – image placeholders + pixels + labels built.")

    print("\n[ABLATION TEST] Checking whether images affect loss…")

    batch = tmp_batch
    batch_noimg = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    batch_noimg["pixel_values"].zero_()

    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        out_full = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
            pixel_values=batch["pixel_values"].to(device),
            image_flags=batch["image_flags"].to(device),
        )
        out_noimg = model(
            input_ids=batch_noimg["input_ids"].to(device),
            attention_mask=batch_noimg["attention_mask"].to(device),
            labels=batch_noimg["labels"].to(device),
            pixel_values=batch_noimg["pixel_values"].to(device),
            image_flags=batch_noimg["image_flags"].to(device),
        )

    print(f"[ABLATION] loss WITH images:   {out_full.loss.item():.6f}")
    print(f"[ABLATION] loss WITHOUT images:{out_noimg.loss.item():.6f}")
    print("[ABLATION] Done.\n")

    training_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        learning_rate=args.lr,

        save_strategy="epoch",
        save_total_limit=2,
        save_safetensors=True,
        save_steps=100,
        logging_dir=f"{out_dir}/logs",

        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",

        logging_steps=20,
        fp16=False,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=False,
        gradient_accumulation_steps=1,
        optim="adamw_torch_fused",
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

    hyper = {
        "learning_rate": args.lr,
        "batch_size": args.bsz,
        "epochs": args.epochs,
        "lr_scheduler_type": training_args.lr_scheduler_type,
        "max_frames": args.max_frames,
        "lora_r": args.lora_cap,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "weights": "uniform_1.0",
        "type": args.type,
        "split": args.split,
        "resume_from": args.resume_from,
        "train_jsonl": train_jsonl,
    }
    hyper_path = os.path.join(out_dir, "hyperparams.jsonl")
    with open(hyper_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(hyper) + "\n")
    print(f"[INFO] Saved hyperparameters to {hyper_path}")

    if args.resume_from:
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"[INFO] Training complete; adapters saved to: {out_dir}")

if __name__ == "__main__":
    main()
