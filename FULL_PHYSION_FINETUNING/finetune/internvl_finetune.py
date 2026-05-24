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
from torch.utils.data import random_split
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    AutoConfig,
    EarlyStoppingCallback,
)

# ---------- config ----------
MODEL_DIR   = "/home/ievab2/models/InternVL2-8B"
INPUT_SIZE  = 448
NUM_FRAMES  = 8

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

INSTR = (
    "You see 8 consecutive frames of a video in temporal order. "
    "Do not explain; just answer the question concisely. "
)

def build_transform():
    '''
    Ensures EGB
    Resizes each frame to 448x448 --> that's what InternVl expects
    Converts to tensor
    Normalizes with ImageNet mean/std
    '''
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

# --- list_jsonls now supports "both" ---
def list_jsonls(round_idx: int, split_name: str):
    base = f"/home/ievab2/run_models/FULL_PHYSION_FINETUNING/curriculum/round{round_idx}"
    return [f"{base}/curriculum_{split_name}_train.jsonl"]

# --- dataset: default weight by path (hard > easy), but respect per-row "weight" if present ---
class PhysionVLMDataset(Dataset):
    def __init__(self, jsonl_paths: List[str], max_frames: int = 8,
                 noweights: bool = False):
        self.max_frames = max_frames
        self.noweights = noweights
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

                    if not isinstance(frames, list) or len(frames) != NUM_FRAMES:
                        continue

                    # use JSONL weight unless noweights=True
                    if self.noweights:
                        w = 1.0
                    else:
                        w = float(row.get("weight", 1.0))

                    self.rows.append({
                        "frames": frames,
                        "question": q,
                        "answer": a,
                        "weight": w,
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
                    "weight": float(r["weight"]),
                }

            except Exception as e:
                last_err = e
                self.num_skipped_samples += 1
                self.num_bad_images += 1

                # print only the first time we see a failure for this sample
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
    # bypass HF's _prepare_inputs, which is where inputs_embeds is getting injected
    def _prepare_inputs(self, inputs):
        return inputs #return the batch from the collator unchanged

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss with:
          - sample_weight support
          - hard filtering of kwargs so InternVLChatModel never sees unsupported args
        """
        weights = inputs.pop("sample_weight", None)  # (B,) or None

        # whitelist keys the model supports
        allowed = {"input_ids", "attention_mask", "labels", "pixel_values", "image_flags"}
        clean_inputs = {k: v for k, v in inputs.items() if k in allowed}

        # (extra sefaty )drop any stray 'inputs_embeds' etc if they appear for any reason
        for bad_key in ["inputs_embeds", "position_ids"]:
            if bad_key in clean_inputs: clean_inputs.pop(bad_key, None)

        # Forward pass
        outputs = model(**clean_inputs)
        loss = outputs.loss  # default mean loss from the model

        # Reweight per-sample loss if sample_weight is provided
        if weights is not None:
            # get logits
            logits = outputs.logits # (B, T, vocab)
            labels = clean_inputs["labels"] # (B, T)
            vocab = logits.size(-1)

            #logits[b][t] = the predicted distribution over vocabulary for token t in sample b.
            #labels[b][t] = the correct token or -100 (ignore)

            # when predicting token t, compare it to the next token t+1
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # compute token-level cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            token_loss = loss_fct(
                shift_logits.view(-1, vocab),
                shift_labels.view(-1)
            )  # (B*(T-1),)
            token_loss = token_loss.view(labels.size(0), -1)  # (B, T-1)

            valid_counts = shift_labels.ne(-100).sum(dim=1).clamp(min=1) # how many tokens per sample are valid
            per_sample = token_loss.sum(dim=1) / valid_counts  # (B,); per-sample loss (NOT per-token)
            w = weights.to(per_sample.device).to(per_sample.dtype)
            loss = (per_sample * w).mean()

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, *args, **kwargs) -> None:
        """
        Save training logs (including loss) to a JSONL file in the checkpoint directory.
        Accepts extra HF arguments to avoid TypeError.
        """
        if self.is_world_process_zero():
            loss_file = os.path.join(self.args.output_dir, "train_loss.jsonl")
            record = {
                "step": self.state.global_step,
                "epoch": float(self.state.epoch) if self.state.epoch is not None else None,
                **logs,
            }
            with open(loss_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        # call original HF implementation
        super().log(logs, *args, **kwargs)


# ---------- collator ----------
class VLMDataCollator:
    # Stores tokenizer and a bunch of configuration knobs for building text sequences and image tokens
    def __init__(self, tokenizer, max_len_cap=4096, image_token="<image>", tokens_per_image=256,
                 image_token_id=None, alt_token_id=None, pixel_dtype=None):
        self.tok = tokenizer
        self.hard_cap = int(max_len_cap)
        self.image_token = image_token
        self.tokens_per_image = int(tokens_per_image)
        self.img_id_primary = int(image_token_id) if image_token_id is not None else None
        self.img_id_alt     = int(alt_token_id)   if alt_token_id   is not None else None
        self.pixel_dtype    = pixel_dtype                

    # When Trainer loads a batch from the Dataset, it passes a list of samples to collate(batch).
    def __call__(self, batch):

        # Builds a set of IDs that represent the <image> token in any system.
        # Used to prevent accidently generating these token IDs as normal text tokens.

        # removing <image> token from the prompt itself
        img_token_id_t = self.tok.convert_tokens_to_ids(self.image_token) # converting <image> into its numerid ID
        img_ids_to_block = set()
        for cand in (self.img_id_primary, self.img_id_alt, img_token_id_t):
            if isinstance(cand, int) and cand >= 0: img_ids_to_block.add(cand)
        if not img_ids_to_block: raise ValueError("Could not resolve any valid <image> token id.")

        # per_sample_imgs: list of (K,3,H,W) tensors.
        per_sample_ids_full, per_sample_att_full, per_sample_lab_full = [], [], []
        per_sample_imgs = [b["pixel_values"] for b in batch]
        B = len(batch) # batch size
        K = per_sample_imgs[0].shape[0] # frames per video
        required_img_tokens = K * self.tokens_per_image
        safe_id = self.tok.eos_token_id if self.tok.eos_token_id is not None else self.tok.pad_token_id #safe_id: fallback token used to replace any image IDs found in text.

        # Loop over each sample:
        for b in batch:
            k = b["pixel_values"].shape[0]
            assert k == K
            prompt_txt = INSTR + (b["question"] or "") # Builds the text prompt: fixed instruction + question
            ans_txt    = str(b["answer"] or "") # Answer is separated; we will supervise only on answer tokens.

            # toenizing
            enc_prompt = self.tok(prompt_txt, add_special_tokens=False, return_tensors=None)
            enc_answer = self.tok(ans_txt,    add_special_tokens=False, return_tensors=None)
            prompt_ids = enc_prompt["input_ids"]
            answer_ids = enc_answer["input_ids"]

            # Clean any accidental image token IDs from the text:
            if img_ids_to_block:
                prompt_ids = [
                    (safe_id if tid in img_ids_to_block else tid)
                    for tid in prompt_ids
                ]
                answer_ids = [
                    (safe_id if tid in img_ids_to_block else tid)
                    for tid in answer_ids
                ]

            # Prepend the exact # of image placeholders the model expects
            placeholder_id = (
                self.img_id_primary
                if self.img_id_primary is not None
                else img_token_id_t
            )
            image_placeholders = [placeholder_id] * required_img_tokens

            ids_in  = image_placeholders + prompt_ids
            ids_out = answer_ids
            ids_full    = ids_in + ids_out #input_ids for the whole sample.
            att_full    = [1] * len(ids_full)
            labels_full = ([-100] * len(ids_in)) + ids_out # prompt tokens: -100 (ignored by loss).Answer tokens: equal to token IDs --> teacher forcing for next-token prediction.
            per_sample_ids_full.append(ids_full) # append sequences to lists:
            per_sample_att_full.append(att_full)
            per_sample_lab_full.append(labels_full)

        # Decide max length for the batch; Guarantee there is enough room for the images
        max_len_batch = max(max(len(x), required_img_tokens) for x in per_sample_ids_full)
        if max_len_batch > self.hard_cap:
            if required_img_tokens > self.hard_cap:
                raise RuntimeError(
                    f"Need ≥{required_img_tokens} tokens for {K} images "
                    f"({self.tokens_per_image}/img) but hard_cap={self.hard_cap}."
                )
            max_len_batch = self.hard_cap

        # padding
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

            # stack to tensors
            input_ids_list.append(torch.tensor(ids_full, dtype=torch.long))
            attention_list.append(torch.tensor(att_full, dtype=torch.long))
            labels_list.append(torch.tensor(lab_full, dtype=torch.long))

        # flattening image batch
        input_ids = torch.stack(input_ids_list, dim=0)   # (B, T)
        attention = torch.stack(attention_list, dim=0)   # (B, T)
        labels    = torch.stack(labels_list, dim=0)      # (B, T)

        pixels = torch.cat(per_sample_imgs, dim=0)       # (B*K,3,H,W)
        if self.pixel_dtype is not None: pixels = pixels.to(self.pixel_dtype)
        image_flags = torch.ones((B * K,), dtype=torch.bool)

        weights = torch.tensor([float(b["weight"]) for b in batch], dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
            "pixel_values": pixels,
            "image_flags": image_flags,   # length B*K
            "sample_weight": weights,
        }

def list_physion_jsonls(split_name: str):
    """
    For noweights=True: use raw SPLITX train split.
    """
    base = "/shared/rsaas/ievab2/Physion_full_readout_training/SPLITS"
    return [f"{base}/{split_name}/train.jsonl"]


def main():
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false") #Disable tokenizer multi-process warnings

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["easy","hard","both"], default="both")
    # ap.add_argument("--split_kind", choices=["pred", "past"], required=True)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--easy_weight", type=float, default=1.0)
    ap.add_argument("--hard_weight", type=float, default=1.0)
    ap.add_argument("--noweights", type=bool, default=False) # if noweights=True --> ignore curriculum and hard/easy weights
    
    ap.add_argument("--lora_cap", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.1)

    ap.add_argument("--split", type=str, default="SPLIT3",
                    choices=["SPLIT1", "SPLIT2", "SPLIT3"],
                    help="Which split to use for train curriculum / raw SPLITS.")

    ap.add_argument("--resume_from", type=str, default=None,
                    help="Optional: path to start from (e.g., /shared/.../round0/easy_pred)")
    ap.add_argument("--max_frames", type=int, default=4, help="Use at most this many frames per sample")
    args = ap.parse_args()


    # -------- data
    if args.noweights: # Use original Physion dataset, no curriculum, all weights = 1
        jsonls = list_physion_jsonls(args.split)
        easy_w = 1.0
        hard_w = 1.0
        print(f"[INFO] noweights=True → using original Physion JSONLs (no curriculum): {jsonls}")
    
    else:
        # Use curriculum JSONLs with easy/hard weighting
        jsonls = list_jsonls(args.round, args.split)
        easy_w = args.easy_weight
        hard_w = args.hard_weight

        print(f"[INFO] noweights=False → using curriculum JSONLs: {jsonls}")

    # Building the dataset
    train_ds = PhysionVLMDataset(
        jsonls,
        max_frames=args.max_frames,
        noweights=args.noweights
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

    # -------- tokenizer + model
    cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    IMAGE_TOKEN = getattr(cfg, "image_token", "<image>")

    # tokenizer
    # tok_src = args.resume_from if args.resume_from else MODEL_DIR # if resuming from a checkpoint, get tokenizer from there
    # tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True, use_fast=False)

    # tokenizer: always load from base model dir
    tok_src = MODEL_DIR
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True, use_fast=False)

    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token # enduring there's pad token

    # special tokens
    added_tokens = 0
    if IMAGE_TOKEN not in tokenizer.get_vocab(): # tokenizer might not have <image> in vocab
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
        added_tokens = 1

    # Loading InternVL with: 
    #   bfloat16 if GPU, 
    #   device_map="auto" to place layers on available GPUs,
    #   if resume_from is set, you load a fine-tuned checkpoint; otherwise base InternVL2-8B.
    if torch.cuda.is_available(): load_dtype = torch.bfloat16
    else: load_dtype = torch.float32

    # model = AutoModelForCausalLM.from_pretrained(
    #     args.resume_from if args.resume_from else MODEL_DIR,
    #     trust_remote_code=True,
    #     torch_dtype=load_dtype,
    #     device_map="auto",
    #     attn_implementation="eager",
    # )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=load_dtype,
        device_map="auto",
        attn_implementation="eager",
    )


    model.config.pad_token_id = tokenizer.pad_token_id
    if added_tokens: model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    
    # ---- InternVLChatConfig has no 'vocab_size', but PEFT expects it when saving ----
    config_cls = model.config.__class__
    vs = len(tokenizer)
    # Ensuring current config instance has vocab_size
    if not hasattr(model.config, "vocab_size"):
        print(f"[PATCH] Setting model.config.vocab_size = {vs}")
        model.config.vocab_size = vs
    # ensuring future configs created via from_pretrained also have vocab_size (PEFT calls config_cls.from_pretrained(model_id))
    orig_cfg_from_pretrained = config_cls.from_pretrained

    @classmethod
    def patched_from_pretrained(cls, *args, **kwargs):
        cfg = orig_cfg_from_pretrained(*args, **kwargs)
        if not hasattr(cfg, "vocab_size"): cfg.vocab_size = vs
        return cfg

    # only patching once to avoid wrapping repeatedly if rerun in same process
    # so when PEFT internally calls config_cls.from_pretrained, the resulting config will also have vocab_size
    if getattr(config_cls, "_vocab_size_patched", False) is not True:
        print(f"[PATCH] Monkey-patching {config_cls.__name__}.from_pretrained to inject vocab_size={vs}")
        config_cls.from_pretrained = patched_from_pretrained
        config_cls._vocab_size_patched = True

    # Exact image token id that the MODEL expects (primary)
    tok_image_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)  #ID of <image> in tokenizer.
    mdl_image_id = (    #ID the model itself expects (if it defines one), otherwise fallback to tokenizer’s
        getattr(model, "image_token_index", None)
        or getattr(getattr(model, "config", None), "image_token_index", None)
        or tok_image_id
    )

    # Tokens per image
    TOKENS_PER_IMAGE = (
        getattr(model, "image_token_len", None)
        or getattr(getattr(model, "config", None), "image_token_len", None)
        or getattr(cfg, "image_token_len", None)
        or getattr(cfg, "num_image_tokens", None)
        or getattr(getattr(cfg, "vision_config", {}), "num_image_tokens", None)
        or 256
    )
    print(f"[DEBUG] IMAGE_TOKEN='{IMAGE_TOKEN}', tok_image_id={tok_image_id}, mdl_image_id={mdl_image_id}, TOKENS_PER_IMAGE={TOKENS_PER_IMAGE}")

    # Applying LoRA (PEFT) 
    lora_cfg = LoraConfig( # adds LoRA adapters to the attention blocks (query/key/value and output).
        r=args.lora_cap, 
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["wqkv", "wo"],
        # "wqkv" is where attention input projections happen <-- most important place for adapting model behavior.
        # "wo" is where attention output projection happens.
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # (PEFT sometimes forwards inputs_embeds to the underlying model; InternVLChatModel doesn’t like that) 
    base = getattr(model, "base_model", None) # model is now a PeftModel
    if base is None: raise RuntimeError("Unexpected PEFT structure: model.base_model is None")
    
    inner = getattr(base, "model", None) or getattr(base, "language_model", None) or base
    print(f"[DEBUG] Patching inner forward on: {inner.__class__.__name__}") 
    
    # telling InternVL which token is the image-context token
    if hasattr(inner, "img_context_token_id"):
        inner.img_context_token_id = mdl_image_id  # or tok_image_id
        print(f"[PATCH] Set inner.img_context_token_id = {inner.img_context_token_id}")
    else:
        print("[WARN] inner has no img_context_token_id attribute")

    orig_inner_forward = inner.forward # grab the inner chat model and override its .forward to drop inputs_embeds if PEFT sends them.
    
    def inner_patched_forward(*args, **kwargs):
        if "inputs_embeds" in kwargs:
            kwargs = dict(kwargs)
            kwargs.pop("inputs_embeds", None)
        # if anything else ever shows up that it doesn't like, you strip here too.
        return orig_inner_forward(*args, **kwargs)
    
    inner.forward = inner_patched_forward

    # Fail fast if somehow nothing is trainable
    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("No trainable parameters detected — LoRA not applied correctly")

    print(f"[DEBUG] Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    model.config.use_cache = False

    # -------- build directories
    epochs_tag = str(int(args.epochs)) if abs(args.epochs - int(args.epochs)) < 1e-9 else str(args.epochs)
    frames_tag = f"{args.max_frames}frames"
    stage_tag = "noweights" if args.noweights else args.stage
    if args.noweights: stage_tag = f"noweights_{args.stage}"
    else: stage_tag = args.stage
    out_dir = (
        f"/shared/rsaas/ievab2/FULL_PHYSION_checkpoints/internvl/"
        f"round{args.round}/{args.split}_{stage_tag}_{epochs_tag}epochs_{frames_tag}"
    )

    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Output directory set to {out_dir}")

    # -------- trainer / collator configuration
    model_max = getattr(cfg, "max_position_embeddings", None)
    if not isinstance(model_max, int) or model_max <= 0 or model_max > 32768: model_max = 4096
    model_max = min(model_max, 2048)

    model_dtype = next(model.parameters()).dtype

    collate = VLMDataCollator(
        tokenizer,
        max_len_cap=model_max,
        image_token=IMAGE_TOKEN,
        tokens_per_image=TOKENS_PER_IMAGE,
        image_token_id=mdl_image_id, # model’s exact id
        alt_token_id=tok_image_id, # tokenizer’s id (if different)
        pixel_dtype=model_dtype, 
    )

    print(f"[INFO] Starting training --> output dir: {out_dir}")

    # ---------- sanity check batch
    tmp_loader = DataLoader(train_ds, batch_size=args.bsz, shuffle=False, collate_fn=collate)
    tmp_batch = next(iter(tmp_loader))

    print("[SANITY] shapes:",
          "input_ids", tmp_batch["input_ids"].shape,
          "pixel_values", tmp_batch["pixel_values"].shape,
          "image_flags", tmp_batch["image_flags"].shape)

    print("[SANITY] Collator OK – image placeholders + pixels + labels built.")

    # NEW: IMAGE ABLATION CHECK
    print("\n[ABLATION TEST] Checking whether images affect loss…")

    batch = tmp_batch
    batch_noimg = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    batch_noimg["pixel_values"].zero_()  # remove all image content

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

    # ---------- training
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

    # ---- save hyperparameters for this finetune run ----
    hyper = {
        "learning_rate": args.lr,
        "batch_size": args.bsz,
        "epochs": args.epochs,
        "lr_scheduler_type": training_args.lr_scheduler_type,
        "max_frames": args.max_frames,
        "lora_r": args.lora_cap,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "easy_w": easy_w,
        "hard_w": hard_w,

    }
    hyper_path = os.path.join(out_dir, "hyperparams.jsonl")
    with open(hyper_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(hyper) + "\n")
    print(f"[INFO] Saved hyperparameters to {hyper_path}")

    # ---- training ----

    #trainer.train()
    if args.resume_from:
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()
    # saves adapters only because 'model' is a PeftModel
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"[INFO] Training complete; adapters saved to: {out_dir}")

    # ===============================================

if __name__ == "__main__":
    main()