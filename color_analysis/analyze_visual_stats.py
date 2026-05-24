# comparing my own original dataset (only collide) vs texture vs color datasets
import json
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# -----------------------------
# DATASET PATHS
# -----------------------------

datasets = {
    "original": "/home/ievab2/run_models/FULL_PHYSION_FINETUNING/testing/round0/my_own_physion/epochs_3/SPLIT1_results.jsonl",
    "colors": "/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/results/colors/SPLIT1_epochs1_results.jsonl",
    "textures": "/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST/internvl/results/texture/SPLIT1_epochs1_results.jsonl",
}

output_dir = Path("/home/ievab2/run_models/color_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# IMAGE STATISTICS
# -----------------------------

def compute_frame_stats(img):

    img = img.astype(np.float32) / 255.0

    R = img[:,:,2]
    G = img[:,:,1]
    B = img[:,:,0]

    mean_R = R.mean()
    mean_G = G.mean()
    mean_B = B.mean()

    std_R = R.std()
    std_G = G.std()
    std_B = B.std()

    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    grad = np.sqrt(gx**2 + gy**2)

    grad_mean = grad.mean()
    grad_std = grad.std()

    return np.array([
        mean_R, mean_G, mean_B,
        std_R, std_G, std_B,
        grad_mean, grad_std
    ])


# -----------------------------
# VIDEO STATISTICS
# -----------------------------

def compute_video_stats(frame_paths):

    frame_stats = []

    for p in frame_paths:

        img = cv2.imread(p)

        if img is None:
            continue

        stats = compute_frame_stats(img)
        frame_stats.append(stats)

    frame_stats = np.array(frame_stats)

    return frame_stats.mean(axis=0)


# -----------------------------
# DATASET STATISTICS
# -----------------------------

def analyze_dataset(jsonl_path):

    video_stats = []

    with open(jsonl_path) as f:
        for line in tqdm(f):

            row = json.loads(line)

            cat = row.get("category","").lower()

            if cat != "collide":
                continue

            frame_paths = row["frame_paths"]

            stats = compute_video_stats(frame_paths)

            video_stats.append(stats)

    video_stats = np.array(video_stats)

    dataset_mean = video_stats.mean(axis=0)

    return dataset_mean


# -----------------------------
# RUN ANALYSIS
# -----------------------------

columns = [
    "mean_R","mean_G","mean_B",
    "std_R","std_G","std_B",
    "grad_mean","grad_std"
]

results = {}

for name,path in datasets.items():

    print(f"Processing {name}...")

    stats = analyze_dataset(path)

    results[name] = stats


df = pd.DataFrame(results,index=columns).T

print(df)

# -----------------------------
# SAVE TABLE PNG
# -----------------------------

fig, ax = plt.subplots(figsize=(10,2))

ax.axis("off")

table = ax.table(
    cellText=np.round(df.values,4),
    colLabels=df.columns,
    rowLabels=df.index,
    loc="center"
)

table.scale(1,2)

plt.title("Dataset Visual Statistics (RGB + Gradient)")

out_path = output_dir / "dataset_visual_stats.png"

plt.savefig(out_path, bbox_inches="tight", dpi=300)

print(f"\nSaved table to: {out_path}")