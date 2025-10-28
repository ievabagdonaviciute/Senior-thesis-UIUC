#!/usr/bin/env python3
import os, json, math, argparse
from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# -------- defaults: save next to this script --------
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PNG_DEFAULT       = str(Path(__file__).with_name("picked_frames_histogram.png"))
HEATMAP_PNG_DEFAULT   = str(Path(__file__).with_name("picked_frames_heatmap.png"))

FRAMES_ROOT_DEFAULT     = "/home/ievab2/run_models/experiment_frame_selection/selected_frames"
SELECTION_JSONL_DEFAULT = "/home/ievab2/run_models/experiment_frame_selection_intervals/qwen_out_frame_selection_skip.jsonl"
TASK_JSONL_DEFAULT      = "/home/ievab2/run_models/questions/clevrer_filtered_500.jsonl"
# ----------------------------------------------------

def frames_dir_from_row(row: dict, frames_root: str) -> Path:
    vpath = Path(row["video_path"])
    chunk = vpath.parent.name          # e.g., "video_10000-11000"
    vid   = vpath.stem                 # e.g., "video_10003"
    return Path(frames_root) / chunk / vid

def load_selection_map(path: str) -> dict:
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            qid = r.get("question_id")
            mo  = r.get("model_output")
            if isinstance(mo, str):
                try:
                    idxs = json.loads(mo)
                except json.JSONDecodeError:
                    continue
            else:
                idxs = mo
            if qid and idxs:
                m[qid] = [int(x) for x in idxs]
    return m

def read_selected_frames(dir_path: Path, idxs: List[int]):
    paths, missing = [], []
    for i in idxs:
        jpg = dir_path / f"{i:03d}.jpg"
        png = dir_path / f"{i:03d}.png"
        fp = jpg if jpg.exists() else (png if png.exists() else None)
        if fp is None:
            missing.append(f"{i:03d}")
            continue
        paths.append(str(fp.resolve()))
    if missing:
        raise FileNotFoundError(f"Missing frames in {dir_path}: {', '.join(missing)}")
    return None, paths, idxs

# ====== helper: mirror inference validation ======
def _bucket(i: int) -> int:
    if   0 <= i <= 7:   return 0
    if   8 <= i <= 15:  return 1
    if  16 <= i <= 23:  return 2
    if  24 <= i <= 31:  return 3
    return -1

def validate_and_sort_qwen_indices(raw_idxs: List[int]) -> Tuple[bool, List[int], str, Dict]:
    """
    Returns (ok, sorted_idxs_or_empty, reason, details)
    Enforces:
     - exactly 8 indices
     - all in [0,31]
     - no duplicates
     - exactly 2 per interval bucket (I0..I3)
     - ascending order (we return sorted if ok)
    """
    details = {}
    if not isinstance(raw_idxs, list):
        return False, [], "not_a_list", details
    try:
        idxs = [int(x) for x in raw_idxs]
    except Exception:
        return False, [], "non_integer_values", details

    details["len"] = len(idxs)
    if len(idxs) != 8:
        return False, [], "length_not_8", details

    if any((x < 0 or x > 31) for x in idxs):
        details["out_of_range"] = [x for x in idxs if x < 0 or x > 31]
        return False, [], "out_of_range", details

    if len(set(idxs)) != len(idxs):
        details["duplicates"] = sorted([x for x in idxs if idxs.count(x) > 1])
        return False, [], "duplicates_found", details

    counts = {0:0,1:0,2:0,3:0}
    for x in idxs:
        b = _bucket(x)
        if b == -1:
            details["out_of_range"] = [x for x in idxs if _bucket(x) == -1]
            return False, [], "out_of_range", details
        counts[b] += 1
    details["bucket_counts"] = counts

    if any(counts[b] != 2 for b in (0,1,2,3)):
        return False, [], "bucket_rule_violation(expect_exactly_2_each)", details

    return True, sorted(idxs), "ok", details
# ================================================

def plot_histogram(picked_counts, out_path):
    xs = list(range(32))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.bar(xs, [picked_counts[x] for x in xs])
    plt.title("Qwen-picked frame indices (only rows that passed inference checks)")
    plt.xlabel("Frame index")
    plt.ylabel("Count")
    plt.xticks(xs, rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_heatmap(picked_counts, total_used_rows, out_path):
    probs = np.array([c / total_used_rows if total_used_rows > 0 else 0.0 for c in picked_counts], dtype=float)
    data = probs.reshape(1, -1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 1.25))
    im = plt.imshow(data, aspect="auto", cmap="Reds", vmin=0.0, vmax=max(probs.max(), 1e-9))
    plt.yticks([])
    plt.xticks(range(32), range(32))
    plt.xlabel("Frame index (0–31)")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("P(index is picked)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Replicate Qwen frame selection filtering & plot histogram + heatmap.")
    parser.add_argument("--task-jsonl", default=TASK_JSONL_DEFAULT)
    parser.add_argument("--selection-jsonl", default=SELECTION_JSONL_DEFAULT)
    parser.add_argument("--frames-root", default=FRAMES_ROOT_DEFAULT)
    parser.add_argument("--out-png", default=OUT_PNG_DEFAULT,
                        help="Histogram PNG path (default: next to script).")
    parser.add_argument("--heatmap-png", default=HEATMAP_PNG_DEFAULT,
                        help="Heatmap PNG path (default: next to script).")
    parser.add_argument("--require-exact-8", action="store_true",
                        help="(Deprecated by validator) If set, still enforces 8-length before file checks.")
    args = parser.parse_args()

    selection_map = load_selection_map(args.selection_jsonl)

    picked_counts = [0]*32
    total_in, total_out, total_skipped = 0, 0, 0

    skipped_log_path = Path(args.out_png).with_suffix(".skipped.jsonl")
    with open(args.task_jsonl, "r", encoding="utf-8") as f_in, \
         open(skipped_log_path, "w", encoding="utf-8") as skipped_f:

        for i, line in enumerate(f_in):
            if not line.strip():
                continue
            total_in += 1
            try:
                row = json.loads(line)

                q = row.get("prompt") or row.get("question")
                if not q:
                    raise ValueError("Row missing 'prompt'/'question'")

                frames_dir = frames_dir_from_row(row, args.frames_root)
                if not frames_dir.exists():
                    qid_guess = row.get("question_id") or row.get("qid") or f"row{i}"
                    raise FileNotFoundError(f"Missing frames dir: {frames_dir} (qid={qid_guess})")

                qid = row.get("question_id") or row.get("qid") or f"row{i}"
                qwen_idxs_full = selection_map.get(qid)
                if not qwen_idxs_full:
                    raise ValueError(f"No Qwen-selected indices for {qid}")

                # (Optional pre-check retained, but validator enforces exact-8 anyway)
                if args.require_exact_8 and len(qwen_idxs_full) != 8:
                    raise ValueError(f"{qid}: expected 8 indices, got {len(qwen_idxs_full)}")

                # === mirror inference validation ===
                ok, idxs_sorted, reason, details = validate_and_sort_qwen_indices(qwen_idxs_full)
                if not ok:
                    raise ValueError(f"{qid}: {reason} {details}")

                # Require each requested frame file to exist (same as inference)
                _ = read_selected_frames(frames_dir, idxs_sorted)

                # Count only indices 0..31 (validator ensures this)
                for idx in idxs_sorted:
                    picked_counts[idx] += 1

                total_out += 1

            except Exception as e:
                total_skipped += 1
                try:
                    r = json.loads(line)
                    qid_log = r.get("question_id") or r.get("qid") or f"row{i}"
                except Exception:
                    qid_log = f"row{i}"
                skipped_f.write(json.dumps({"row_index": i, "qid": qid_log, "reason": str(e)}) + "\n")

    print(f"Input rows:   {total_in}")
    print(f"Used rows:    {total_out}")
    print(f"Skipped rows: {total_skipped}")
    print(f"Skip report:  {skipped_log_path}")

    # Save both visualizations
    plot_histogram(picked_counts, args.out_png)
    print(f"Saved histogram: {args.out_png}")

    plot_heatmap(picked_counts, total_out, args.heatmap_png)
    print(f"Saved heatmap: {args.heatmap_png}")

if __name__ == "__main__":
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
