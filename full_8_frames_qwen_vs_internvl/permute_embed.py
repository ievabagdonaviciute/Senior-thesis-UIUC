#!/usr/bin/env python3
import argparse, json, csv, gzip
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import open_clip 

EXPERIMENT_DIR = Path("/home/ievab2/run_models/full_8_frames_qwen_vs_internvl")

def _permute_labels_inplace(df: pd.DataFrame, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    df["correct"] = df["correct"].to_numpy()[perm]
    df["raw_score"] = df["raw_score"].to_numpy()[perm]
    df["answer"] = df["answer"].to_numpy()[perm]

def _rng(seed: int):
    return np.random.default_rng(seed)

def _permute_list(lst, rng):
    idx = rng.permutation(len(lst))
    return [lst[i] for i in idx], idx

def build_paths():
    eval_per_question_dir = {
        "INTERNVL": "/home/ievab2/run_models/full_8_frames_qwen_vs_internvl/INTERNVL/scores/internvl_per_question.jsonl",
        "QWEN": "/home/ievab2/run_models/full_8_frames_qwen_vs_internvl/QWEN/scores/qwen_per_question.jsonl",
    }
    out_dir_question_embeds = {
        "INTERNVL":          EXPERIMENT_DIR / "INTERNVL"         / "permuted_embeddings" / "only_prompt",
        "QWEN":              EXPERIMENT_DIR / "QWEN"             / "permuted_embeddings" / "only_prompt",
    }
    out_dir_image_embeds = {
        "INTERNVL":          EXPERIMENT_DIR / "INTERNVL"         / "permuted_embeddings" / "only_image",
        "QWEN":              EXPERIMENT_DIR / "QWEN"             / "permuted_embeddings" / "only_image",
    }
    out_both_embeds = {
        "INTERNVL":          EXPERIMENT_DIR / "INTERNVL"         / "permuted_embeddings" / "image_and_prompt",
        "QWEN":              EXPERIMENT_DIR / "QWEN"             / "permuted_embeddings" / "image_and_prompt",
    }

    return eval_per_question_dir, out_dir_question_embeds, out_dir_image_embeds, out_both_embeds

# ====== embedding ONLY questions ======

def embed_only_questions(in_jsonl_path: str | Path, out_dir_path: str | Path,
                         permute: bool = False, seed: int = 42):
    IN_JSONL = Path(in_jsonl_path)
    OUT_DIR  = Path(out_dir_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_META = OUT_DIR / "prompts_meta.csv"
    OUT_EMB  = OUT_DIR / "embeddings_eva02_l14.csv.gz"

    # ---- load data ----
    rows = []
    with IN_JSONL.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line)
            q = d.get("prompt") or d.get("question") or ""
            try:
                s = float(d.get("score", 0.0))
            except Exception:
                s = 0.0
            y = 1 if s == 1.0 else 0
            rows.append({"row_idx": i, "prompt": q, "correct": y, "raw_score": s})

    df = pd.DataFrame(rows)

    print(f"Loaded {len(df)} rows from {IN_JSONL}")

    # Permute QUESTIONS only (keep labels aligned to original rows)
    if permute and len(df) > 1:
        rng = _rng(seed)
        permuted_questions, src_idx_text = _permute_list(df["prompt"].tolist(), rng)
        df["prompt"] = permuted_questions
        # provenance: where each permuted question came from
        df["src_idx_text"] = src_idx_text


    # ---- OpenCLIP EVA02-CLIP-L/14 (text encoder) ----
    model_name = "EVA02-L-14"
    pretrained = "merged2b_s4b_b131k"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    texts = df["prompt"].fillna("").tolist()
    batch_size = 256

    all_feats = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            tokens = tokenizer(chunk)  # (B, context_len)
            tokens = tokens.to(device)
            # encode_text returns (B, D)
            feats = model.encode_text(tokens)
            feats = F.normalize(feats, dim=-1)  # unit vectors
            all_feats.append(feats.float().cpu().numpy())

    emb = np.concatenate(all_feats, axis=0)
    assert emb.shape[0] == len(df), f"emb rows {emb.shape[0]} != df {len(df)}"
    print(f"Embeddings shape: {emb.shape}")  # e.g., (N, 768 or 1024 depending on ckpt)

    # ---- write aligned outputs ----
    df.to_csv(OUT_META, index=False)
    colnames = [f"e{i}" for i in range(emb.shape[1])]
    with gzip.open(OUT_EMB, "wt", newline="") as gz:
        w = csv.writer(gz)
        w.writerow(colnames)
        w.writerows(emb.tolist())

    print(f"Wrote:\n  {OUT_META}\n  {OUT_EMB}\nDone.")
    
# ====== embedding ONLY images ======

def _init_openclip_image_model(model_name: str = "EVA02-L-14", device: str | None = None):
    import open_clip
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # Pick an available tag for this model (your install exposes 'merged2b_s4b_b131k')
    avail = open_clip.list_pretrained()
    if isinstance(avail, dict): tags = avail.get(model_name, [])
    else: tags = []
    pretrained = tags[0] if tags else "merged2b_s4b_b131k"

    print(f"[open_clip] Using {model_name} with pretrained='{pretrained}' on {device}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    return model, preprocess, device


def _safe_load_image(path: str, preprocess):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    return preprocess(im)  # tensor (3,H,W)


@torch.no_grad()
def embed_only_images(in_jsonl_path: str | Path, out_dir_path: str | Path,
                      require_k: int = 8, pool: str = "mean",
                      permute: bool = False, seed: int = 42):

    IN_JSONL = Path(in_jsonl_path)
    OUT_DIR  = Path(out_dir_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_META = OUT_DIR / "images_meta.csv"
    OUT_EMB  = OUT_DIR / "embeddings_eva02_l14_img.csv.gz"

    model, preprocess, device = _init_openclip_image_model()

    # --------- phase 1: load all valid rows ----------
    raw_rows = []
    with IN_JSONL.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue

            frame_paths = d.get("frame_paths")
            if not frame_paths or len(frame_paths) != require_k:
                continue

            try:
                s = float(d.get("score", 0.0))
            except Exception:
                s = 0.0
            correct01 = 1 if s == 1.0 else 0

            raw_rows.append({
                "_src_idx": i,
                "question_id": d.get("question_id") or d.get("qid") or f"row{i}",
                "frames_dir": d.get("frames_dir", ""),
                "frame_paths": frame_paths,
                "n_frames": require_k,
                "correct": correct01,
                "raw_score": s,
                "category": d.get("category", ""),
                "prompt": d.get("prompt") or d.get("question") or "",
            })

    if not raw_rows:
        print("[embed_only_images] No embeddings produced; nothing to write.")
        return

    # --------- phase 2: permute image paths across rows (if requested) ----------
    rng = _rng(seed)
    if permute and len(raw_rows) > 1:
        frames_list, img_perm_idx = _permute_list([r["frame_paths"] for r in raw_rows], rng)
        for i, r in enumerate(raw_rows):
            r["frame_paths"] = frames_list[i]
            r["src_idx_image"] = int(img_perm_idx[i])  # provenance
    else:
        for r in raw_rows:
            r["src_idx_image"] = r["_src_idx"]

    # --------- phase 3: encode ----------
    rows_meta = []
    embs = []
    for r in raw_rows:
        try:
            imgs = [_safe_load_image(p, preprocess) for p in r["frame_paths"]]
        except Exception as e:
            print(f"[warn] row {r['_src_idx']}: failed to load frames ({e}); skipping")
            continue

        batch = torch.stack(imgs, dim=0).to(device)  # (8,3,H,W)
        feats = model.encode_image(batch)            # (8,D)
        feats = F.normalize(feats, dim=-1)           # per-frame L2

        if pool == "mean":
            pooled = feats.mean(dim=0, keepdim=False)
        elif pool == "max":
            pooled, _ = feats.max(dim=0)
        else:
            pooled = feats.mean(dim=0, keepdim=False)

        pooled = F.normalize(pooled, dim=-1)         # final L2
        embs.append(pooled.float().cpu().numpy())

        rows_meta.append({
            "row_idx": r["_src_idx"],
            "question_id": r["question_id"],
            "frames_dir": r["frames_dir"],
            "n_frames": r["n_frames"],
            "correct": r["correct"],
            "raw_score": r["raw_score"],
            "category": r["category"],
            "prompt": r["prompt"],
            "src_idx_image": r["src_idx_image"],  # provenance
        })

    if not embs:
        print("[embed_only_images] No embeddings produced; nothing to write.")
        return

    emb = np.stack(embs, axis=0)  # (N, D)
    print(f"[embed_only_images] Embeddings shape: {emb.shape}")

    # write meta and embeddings
    pd.DataFrame(rows_meta).to_csv(OUT_META, index=False)
    colnames = [f"e{i}" for i in range(emb.shape[1])]
    with gzip.open(OUT_EMB, "wt", newline="") as gz:
        w = csv.writer(gz)
        w.writerow(colnames)
        w.writerows(emb.tolist())

    print(f"Wrote:\n  {OUT_META}\n  {OUT_EMB}\nDone.")

# ====== embedding images + questions ======

@torch.no_grad()
def embed_images_and_questions(in_jsonl_path: str | Path,
                               out_dir_path: str | Path,
                               require_k: int = 8,
                               pool: str = "mean",
                               batch_size_text: int = 256,
                               permute_all: bool = False,
                               seed: int = 42):

    """
    Produce one embedding per row by encoding:
      - TEXT: question/prompt with OpenCLIP text encoder (unit-norm)
      - IMAGES: 8 frames with OpenCLIP image encoder (per-frame unit-norm, pooled to one, then unit-norm)
    Final embedding = L2-normalized concatenation [img_emb ; text_emb].
    """
    IN_JSONL = Path(in_jsonl_path)
    OUT_DIR  = Path(out_dir_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_META = OUT_DIR / "image_and_prompt_meta.csv"
    OUT_EMB  = OUT_DIR / "embeddings_eva02_l14_both.csv.gz"

    # ---- init model / preprocess / tokenizer ----
    model, preprocess, device = _init_openclip_image_model(model_name="EVA02-L-14")
    tokenizer = open_clip.get_tokenizer("EVA02-L-14")
    model.eval()

    # ---- load rows (collect text + frame lists + labels) ----
    texts: list[str] = []
    frames_list: list[list[str]] = []
    meta_rows = []

    with IN_JSONL.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue

            frame_paths = d.get("frame_paths")
            if not frame_paths or len(frame_paths) != require_k:
                continue

            q_text = d.get("prompt") or d.get("question") or ""
            try:
                s = float(d.get("score", 0.0))
            except Exception:
                s = 0.0
            correct01 = 1 if s == 1.0 else 0
            qid = d.get("question_id") or d.get("qid") or f"row{i}"

            texts.append(q_text)
            frames_list.append(frame_paths)
            meta_rows.append({
                "row_idx": i,
                "question_id": qid,
                "frames_dir": d.get("frames_dir", ""),
                "n_frames": require_k,
                "correct": correct01,
                "raw_score": s,
                "category": d.get("category", ""),
                "prompt": q_text,
                "answer": d.get("answer", "")
            })


    n = len(texts)
    if n == 0:
        print("[embed_images_and_prompts] no valid rows found; nothing to do.")
        return
        
    # --- Permute BOTH questions (texts) and image paths (frames_list), keep labels unchanged ---
    rng = _rng(seed)
    if permute_all and n > 1:
        texts, src_idx_text = _permute_list(texts, rng)
        frames_list, src_idx_image = _permute_list(frames_list, rng)
    else:
        src_idx_text = np.arange(n)
        src_idx_image = np.arange(n)

    # ---- TEXT: encode in batches ----
    text_feats_batches = []
    for b in range(0, n, batch_size_text):
        chunk = texts[b:b+batch_size_text]
        toks = tokenizer(chunk).to(device)
        tfeat = model.encode_text(toks)         # (B, D_t)
        tfeat = F.normalize(tfeat, dim=-1)      # unit vectors
        text_feats_batches.append(tfeat.float().cpu().numpy())
    text_emb = np.concatenate(text_feats_batches, axis=0)  # (n, D_t)

    # ---- IMAGES: encode row-by-row (8 frames each) ----
    img_emb_list = []
    for idx, fpaths in enumerate(frames_list):
        try:
            imgs = [_safe_load_image(p, preprocess) for p in fpaths]
        except Exception as e:
            print(f"[warn] row {idx}: failed to load frames ({e}); using zeros")
            imgs = None

        if imgs is None:
            # fallback zero vector if frames missing/corrupt
            # dimension must match image proj dim
            with torch.no_grad():
                dummy = torch.zeros(1, 3, *preprocess.transforms[-2].size)
                dummy = dummy.to(device)
                dvec = model.encode_image(dummy) * 0.0
                dvec = F.normalize(dvec, dim=-1)
                img_emb_list.append(dvec.squeeze(0).float().cpu().numpy())
            continue

        batch = torch.stack(imgs, dim=0).to(device)  # (8,3,H,W)
        feats = model.encode_image(batch)            # (8, D_i)
        feats = F.normalize(feats, dim=-1)

        if pool == "mean":
            pooled = feats.mean(dim=0)
        elif pool == "max":
            pooled, _ = feats.max(dim=0)
        else:
            pooled = feats.mean(dim=0)

        pooled = F.normalize(pooled, dim=-1)
        img_emb_list.append(pooled.float().cpu().numpy())

    img_emb = np.stack(img_emb_list, axis=0)        # (n, D_i)

    # ---- CONCAT & final L2 ----
    both = np.concatenate([img_emb, text_emb], axis=1)  # (n, D_i + D_t)
    norms = np.linalg.norm(both, axis=1, keepdims=True) + 1e-12
    both = (both / norms).astype(np.float32)

    # ---- write meta + embeddings (labels unchanged; prompt/images are permuted) ----
    meta_df = pd.DataFrame({
        "question_id":  [m["question_id"] for m in meta_rows],
        "prompt":       texts,                               # permuted questions actually encoded
        "answer":       [m["answer"] for m in meta_rows],    # labels unchanged
        "correct":      [m["correct"] for m in meta_rows],
        "raw_score":    [m["raw_score"] for m in meta_rows],
    })
    meta_df.to_csv(OUT_META, index=False)

    colnames = [f"e{i}" for i in range(both.shape[1])]
    with gzip.open(OUT_EMB, "wt", newline="") as gz:
        w = csv.writer(gz)
        w.writerow(colnames)
        w.writerows(both.tolist())

    print(f"[embed_images_and_prompt] Wrote:\n  {OUT_META}\n  {OUT_EMB}\n"
          f"Shapes: text={text_emb.shape}, img={img_emb.shape}, both={both.shape}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate VLM outputs with a local judge (Llama-3.1-8B-Instruct).")
    ap.add_argument("model_name",
        choices=["QWEN","INTERNVL"],
        help="Which VLM's results to evaluate")
    permute_labels = True
    permute_image_paths = True
    permute_all = True
    seed = 42

    args = ap.parse_args()
    model_name = args.model_name 

    eval_per_question_dir, out_dir_question_embeds, out_dir_image_embeds, out_both_embeds = build_paths()
    
    # ====== embedding ONLY questions ======
    embed_only_questions(
        eval_per_question_dir[model_name],
        out_dir_question_embeds[model_name],
        permute=permute_labels, seed=seed
    )
    print("[embed]: Prompt-only embeddings done!")

    embed_only_images(
        eval_per_question_dir[model_name],
        out_dir_image_embeds[model_name],
        permute=permute_image_paths, seed=seed
    )
    print("[embed]: Image-only embeddings done!")

    embed_images_and_questions(
        eval_per_question_dir[model_name],
        out_both_embeds[model_name],
        permute_all=permute_all, seed=seed
    )
    print("[embed]: Prompt-image embeddings done!")

if __name__ == "__main__":
    main()
