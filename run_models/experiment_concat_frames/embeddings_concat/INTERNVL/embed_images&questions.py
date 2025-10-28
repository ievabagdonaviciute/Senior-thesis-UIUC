# -*- coding: utf-8 -*-

#!/usr/bin/env python3
import json, csv, gzip
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import pandas as pd
import cv2

# ==== input & output paths ====
IN_JSONL = Path("/home/ievab2/run_models/experiment_concat_frames/INTERNVL/experiment_og_concat_8/scores/internvl_per_question.jsonl")
OUT_DIR  = Path("/home/ievab2/run_models/experiment_concat_frames/embeddings_concat/INTERNVL/image&question_embeddings")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_META = OUT_DIR / "image_meta.csv"
OUT_EMB  = OUT_DIR / "embeddings_clip.csv.gz"

# ==== load CLIP model ====
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"   # 512-dim embeddings, fast and open
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

rows, embeddings = [], []

# ==== loop over JSONL entries ====
with IN_JSONL.open("r", encoding="utf-8") as f:
    for i, line in enumerate(tqdm(f, desc="Embedding images")):
        if not line.strip():
            continue
        d = json.loads(line)
        q = d.get("question", "")
        s = float(d.get("score", 0.0))
        img_path = d.get("image_path")

        if not img_path or not Path(img_path).exists():
            continue

        y = 1 if s == 1.0 else 0

        try:
            # Read the image efficiently
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError("Could not read image")

            # Convert BGR (OpenCV) → RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize to CLIP input size
            img = cv2.resize(img, (224, 224))

            # Convert to PIL for CLIP processor
            img = Image.fromarray(img)

            # Process and send to GPU
            inputs = processor(images=[img], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                emb = model.get_image_features(**inputs)

            emb = emb[0].cpu().numpy().flatten()
            rows.append({"row_idx": i, "question": q, "correct": y, "image_path": img_path})
            embeddings.append(emb)

        except Exception as e:
            print(f"⚠️ Skipped {img_path}: {e}")
            continue



# ==== save outputs ====
df = pd.DataFrame(rows)
df.to_csv(OUT_META, index=False)

colnames = [f"e{i}" for i in range(len(embeddings[0]))]
with gzip.open(OUT_EMB, "wt", newline="") as gz:
    w = csv.writer(gz)
    w.writerow(colnames)
    w.writerows(embeddings)

print(f"✅ Saved embeddings:")
print(f"  - {OUT_META}")
print(f"  - {OUT_EMB}")
