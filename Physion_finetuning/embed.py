
#!/usr/bin/env python3
import argparse, json, csv, gzip
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
BASE = Path("/home/ievab2/run_models/Physion_finetuning")

def build_paths(model_name: str, category: str, version: str):

    cat_folder = category[0].upper() + category[1:].lower()
    cat_lower  = category.lower()

    in_paths = {
        "past": BASE / model_name / f"{version}" / cat_folder / f"{cat_lower}_past_out.jsonl",
        "pred": BASE / model_name / f"{version}" / cat_folder / f"{cat_lower}_pred_out.jsonl",
    }

    # output roots per subset
    out_roots = {
        "past": BASE / model_name / f"{version}" / cat_folder /  "embeddings" / "past",
        "pred": BASE / model_name / f"{version}" / cat_folder /  "embeddings" / "pred",
    }

    out_dirs = {
        "only_prompt": {s: out_roots[s] / "only_prompt" for s in ["past","pred"]},
        "only_image":  {s: out_roots[s] / "only_image"  for s in ["past","pred"]},
        "both":        {s: out_roots[s] / "image_and_prompt" for s in ["past","pred"]},
    }

    return in_paths, out_dirs

# ---------------------------------------------------------------------
# MODEL INIT
# ---------------------------------------------------------------------
def _init_openclip_model(model_name="EVA02-L-14", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = "merged2b_s4b_b131k"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer, device

def _safe_load_image(path, preprocess):
    im = Image.open(path).convert("RGB")
    return preprocess(im)


# ---------------------------------------------------------------------
# EMBEDDING FUNCTIONS (unchanged logic)
# ---------------------------------------------------------------------
@torch.no_grad()
def embed_only_questions(in_jsonl, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    model, _, tokenizer, device = _init_openclip_model()

    rows = []
    with open(in_jsonl, "r") as f:
        lines = f.readlines()
        print(f"[verify] Loaded {len(lines)} lines from {in_jsonl}")

        for i, line in enumerate(lines):
            if not line.strip(): continue
            d = json.loads(line)
            q = d.get("question", "")
            gt = d.get("ground_truth", "")
            pred = d.get("model_output_norm", "")
            y = 1 if str(gt).strip().lower() == str(pred).strip().lower() else 0
            rows.append({"row_idx": i, "prompt": q, "correct": y, "ground_truth": gt, "predicted": pred})

    df = pd.DataFrame(rows)
    print(f"[questions] {len(df)} samples from {in_jsonl}")

    texts = df["prompt"].fillna("").tolist()
    feats_all = []
    for i in range(0, len(texts), 256):
        toks = tokenizer(texts[i:i+256]).to(device)
        feats = F.normalize(model.encode_text(toks), dim=-1)
        feats_all.append(feats.float().cpu().numpy())
    emb = np.concatenate(feats_all, 0)

    df.to_csv(out_dir / "prompts_meta.csv", index=False)
    with gzip.open(out_dir / "embeddings.csv.gz", "wt", newline="") as gz:
        w = csv.writer(gz)
        w.writerow([f"e{i}" for i in range(emb.shape[1])])
        w.writerows(emb.tolist())
    print(f"[questions] Saved to {out_dir}")


@torch.no_grad()
def embed_only_images(in_jsonl, out_dir, require_k=8):
    out_dir.mkdir(parents=True, exist_ok=True)
    model, preprocess, _, device = _init_openclip_model()

    rows_meta, embs = [], []
    with open(in_jsonl, "r") as f:
        lines = f.readlines()
        print(f"[verify] Loaded {len(lines)} lines from {in_jsonl}")

        for i, line in enumerate(lines):
            if not line.strip(): continue
            d = json.loads(line)
            frames = d.get("frame_paths") or d.get("frames") or []
            if i == 0:
                print(f"[verify] First sample has {len(frames)} frames → {frames[:2]} ...")

            if len(frames) < require_k: continue
            gt = d.get("ground_truth", "")
            pred = d.get("model_output_norm", "")
            correct = 1 if str(gt).strip().lower() == str(pred).strip().lower() else 0
            imgs = [_safe_load_image(p, preprocess) for p in frames]
            batch = torch.stack(imgs).to(device)
            feats = F.normalize(model.encode_image(batch), dim=-1)
            pooled = F.normalize(feats.mean(0), dim=-1)
            embs.append(pooled.cpu().numpy())
            rows_meta.append({"row_idx": i, "correct": correct, "ground_truth": gt, "predicted": pred})

    emb = np.stack(embs)
    pd.DataFrame(rows_meta).to_csv(out_dir / "images_meta.csv", index=False)
    with gzip.open(out_dir / "embeddings.csv.gz", "wt", newline="") as gz:
        w = csv.writer(gz)
        w.writerow([f"e{i}" for i in range(emb.shape[1])])
        w.writerows(emb.tolist())
    print(f"[images] Saved to {out_dir}")


@torch.no_grad()
def embed_images_and_questions(in_jsonl, out_dir, require_k=8):
    out_dir.mkdir(parents=True, exist_ok=True)
    model, preprocess, tokenizer, device = _init_openclip_model()

    texts, frame_lists, meta = [], [], []
    with open(in_jsonl, "r") as f:
        lines = f.readlines()
        print(f"[verify] Loaded {len(lines)} lines from {in_jsonl}")

        for i, line in enumerate(lines):
            if not line.strip(): continue
            d = json.loads(line)
            frames = d.get("frame_paths") or d.get("frames") or []
            if len(frames) < require_k: continue
            q = d.get("question", "")
            gt = d.get("ground_truth", "")
            pred = d.get("model_output_norm", "")
            correct = 1 if str(gt).strip().lower() == str(pred).strip().lower() else 0
            texts.append(q)
            frame_lists.append(frames)
            meta.append({"row_idx": i, "prompt": q, "ground_truth": gt, "predicted": pred, "correct": correct})

    # text embeddings
    t_feats = []
    for i in range(0, len(texts), 256):
        toks = tokenizer(texts[i:i+256]).to(device)
        feats = F.normalize(model.encode_text(toks), dim=-1)
        t_feats.append(feats.float().cpu().numpy())
    text_emb = np.concatenate(t_feats, 0)

    # image embeddings
    img_emb = []
    for frames in frame_lists:
        imgs = [_safe_load_image(p, preprocess) for p in frames]
        batch = torch.stack(imgs).to(device)
        feats = F.normalize(model.encode_image(batch), dim=-1)
        pooled = F.normalize(feats.mean(0), dim=-1)
        img_emb.append(pooled.cpu().numpy())
    img_emb = np.stack(img_emb)

    both = np.concatenate([img_emb, text_emb], axis=1)
    both /= np.linalg.norm(both, axis=1, keepdims=True) + 1e-12

    pd.DataFrame(meta).to_csv(out_dir / "meta.csv", index=False)
    with gzip.open(out_dir / "embeddings.csv.gz", "wt", newline="") as gz:
        w = csv.writer(gz)
        w.writerow([f"e{i}" for i in range(both.shape[1])])
        w.writerows(both.tolist())
    print(f"[both] Saved to {out_dir}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_name", choices=["QWEN", "INTERNVL"])
    ap.add_argument("--category", "-c",
                    choices=["Dominoes","Contain","Drop","dominoes","contain","drop"],
                    required=True)
    ap.add_argument("--version", "-v",
                choices=["both_1epochs_4frames","both_3epochs_4frames","both_5epochs_4frames"],
                required=True)
    args = ap.parse_args()

    category = args.category[0].upper() + args.category[1:].lower()
    print(f"\n=== Building paths for model={args.model_name} category={category} ===")
    in_paths, out_dirs = build_paths(args.model_name, category, args.version)

    for subset in ["past", "pred"]:
        in_jsonl = in_paths[subset]
        print(f"\n=== {args.model_name} — {category} — {subset.upper()} ===")
        print(f"IN:  {in_jsonl}")

        # only_prompt
        out_q = out_dirs["only_prompt"][subset]
        print(f"OUT (only_prompt): {out_q}")
        embed_only_questions(in_jsonl, out_q)
        print("[embed]: Prompt-only embeddings done!")

        # only_image
        out_i = out_dirs["only_image"][subset]
        print(f"OUT (only_image):  {out_i}")
        embed_only_images(in_jsonl, out_i)
        print("[embed]: Image-only embeddings done!")

        # image_and_prompt
        out_b = out_dirs["both"][subset]
        print(f"OUT (both):        {out_b}")
        embed_images_and_questions(in_jsonl, out_b)
        print("[embed]: Prompt-image embeddings done!")

if __name__ == "__main__":
    main()
