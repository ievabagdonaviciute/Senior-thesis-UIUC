#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image


def _init_openclip_model(model_name="EVA02-L-14", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = "merged2b_s4b_b131k"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer, device


def _safe_load_image(path, preprocess):
    im = Image.open(path).convert("RGB")
    return preprocess(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_jsonl",
        required=True,
        help="Base model results JSONL (with frame_paths, question, answer, model_output_norm, category, qid).",
    )
    ap.add_argument(
        "--out_root",
        required=True,
        help="Output directory root, e.g. /home/.../FULL_PHYSION_FINETUNING/embeddings/round0",
    )
    ap.add_argument(
        "--require_k",
        type=int,
        default=8,
        help="Require at least this many frames (default: 8).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="If set, load existing meta.csv + embeddings.csv.gz and skip already embedded (category,qid).",
    )
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    meta_path = out_root / "meta.csv"
    emb_path = out_root / "embeddings.csv.gz"

    # -------------------- resume support --------------------
    meta_rows = []
    emb_rows = []
    done_keys = set()

    if args.resume and meta_path.exists() and emb_path.exists():
        print(f"[RESUME] Loading existing meta + embeddings from {out_root}")
        df_meta_prev = pd.read_csv(meta_path)
        df_emb_prev = pd.read_csv(emb_path, compression="gzip")

        if len(df_meta_prev) != len(df_emb_prev):
            raise RuntimeError(
                f"Mismatch between meta ({len(df_meta_prev)}) and embeddings ({len(df_emb_prev)}) rows."
            )

        meta_rows = df_meta_prev.to_dict("records")
        emb_rows = df_emb_prev.to_numpy().tolist()
        done_keys = set(zip(df_meta_prev["category"], df_meta_prev["qid"]))
        print(f"[RESUME] Found {len(done_keys)} already-embedded examples.")
    else:
        print("[RESUME] Starting fresh (no existing meta/embeddings loaded).")

    # -------------------- init OpenCLIP --------------------
    print("[MODEL] Initializing OpenCLIP EVA02-L-14 …")
    model, preprocess, tokenizer, device = _init_openclip_model()

    # -------------------- main loop --------------------
    print(f"[INPUT] Reading: {in_path}")
    total_seen = 0
    total_skipped_done = 0
    total_skipped_frames = 0
    total_new = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            total_seen += 1

            try:
                d = json.loads(line)
            except Exception as e:
                print(f"[WARN] JSON parse error at line {line_idx}: {e}")
                continue

            cat = d.get("category")
            qid = d.get("qid")

            if cat is None or qid is None:
                print(f"[WARN] Missing category/qid at line {line_idx}, skipping.")
                continue

            key = (cat, qid)
            if key in done_keys:
                total_skipped_done += 1
                continue

            frames = d.get("frame_paths") or d.get("frames") or []
            if len(frames) < args.require_k:
                total_skipped_frames += 1
                continue

            question = d.get("question", "")
            gt = d.get("answer", "")
            pred = d.get("model_output_norm", "")

            correct = 1 if str(gt).strip().lower() == str(pred).strip().lower() else 0

            # ---------- compute text embedding ----------
            with torch.no_grad():
                toks = tokenizer([question]).to(device)
                text_feat = model.encode_text(toks)  # (1, D)
                text_feat = F.normalize(text_feat, dim=-1)[0]  # (D,)

            # ---------- compute image embedding (all frames averaged) ----------
            imgs = []
            for p in frames:
                try:
                    imgs.append(_safe_load_image(p, preprocess))
                except Exception as e:
                    print(f"[WARN] Failed to load image {p} at line {line_idx}: {e}")
                    imgs = []
                    break

            if len(imgs) < args.require_k:
                total_skipped_frames += 1
                continue

            batch = torch.stack(imgs).to(device)  # (K, 3, H, W)
            with torch.no_grad():
                img_feats = model.encode_image(batch)  # (K, D)
                img_feats = F.normalize(img_feats, dim=-1)
                pooled = img_feats.mean(0, keepdim=True)  # (1, D)
                img_vec = F.normalize(pooled, dim=-1)[0]  # (D,)

            # ---------- combine image + text ----------
            both = torch.cat([img_vec, text_feat], dim=-1)  # (2D,)
            both = F.normalize(both, dim=-1)
            emb_rows.append(both.cpu().numpy().tolist())

            meta_rows.append({
                "category": cat,
                "qid": qid,
                "name": d.get("name", ""),
                "question": question,
                "answer": gt,
                "model_output_norm": pred,
                "correct": correct,
            })

            done_keys.add(key)
            total_new += 1

            if total_new % 100 == 0:
                print(f"[PROGRESS] New embedded: {total_new} (total_seen={total_seen})")

    # -------------------- save all to disk --------------------
    if not emb_rows:
        print("[WARN] No new embeddings computed. Nothing to write.")
        return

    print(f"[SAVE] Saving {len(meta_rows)} meta rows and {len(emb_rows)} embedding rows to {out_root}")

    df_meta = pd.DataFrame(meta_rows)
    emb_arr = np.array(emb_rows, dtype=np.float32)
    df_emb = pd.DataFrame(emb_arr, columns=[f"e{i}" for i in range(emb_arr.shape[1])])

    df_meta.to_csv(meta_path, index=False)
    df_emb.to_csv(emb_path, index=False, compression="gzip")

    print("==========================================")
    print(f"Total lines seen in JSONL : {total_seen}")
    print(f"Already embedded (resume) : {total_skipped_done}")
    print(f"Skipped (too few frames)  : {total_skipped_frames}")
    print(f"Newly embedded this run   : {total_new}")
    print(f"Final meta path           : {meta_path}")
    print(f"Final embeddings path     : {emb_path}")
    print("==========================================")


if __name__ == "__main__":
    main()
