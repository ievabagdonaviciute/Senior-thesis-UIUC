# filter_500.py
import argparse
import json
import random
import re
from pathlib import Path

VID_RE = re.compile(r"video_(\d+)")

def extract_vid_int(video_number: str) -> int | None:
    """Extract integer video id from strings like 'video_13073'."""
    m = VID_RE.fullmatch(video_number.strip())
    return int(m.group(1)) if m else None

def main():
    parser = argparse.ArgumentParser(description="Filter CLEVRER JSONL to 500 random video ids in [10000,14999].")
    parser.add_argument("--in", dest="inp", default="/home/ievab2/run_models/questions/clevrer_all_val_tasks.jsonl",
                        help="Path to input JSONL")
    parser.add_argument("--out", dest="out", default="/home/ievab2/run_models/questions/clevrer_filtered_500.jsonl",
                        help="Path to output JSONL")
    parser.add_argument("--min_id", type=int, default=10000, help="Minimum video id (inclusive)")
    parser.add_argument("--max_id", type=int, default=14999, help="Maximum video id (inclusive)")
    parser.add_argument("--n", type=int, default=500, help="Number of unique video ids to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: collect all eligible unique video ids present in file
    eligible_ids = set()
    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines
                continue
            vn = obj.get("video_number") or ""
            vid = extract_vid_int(vn.replace("video_", "") if vn.startswith("video_") else vn)
            if vid is None:
                # Fallback: try from question_id if needed
                qid = obj.get("question_id", "")
                m = VID_RE.search(qid)
                if m:
                    vid = int(m.group(1))
            if vid is None:
                continue
            if args.min_id <= vid <= args.max_id:
                eligible_ids.add(vid)

    if not eligible_ids:
        print("No eligible video ids found in the specified range. Nothing to write.")
        return

    # Sample desired set of ids
    random.seed(args.seed)
    k = min(args.n, len(eligible_ids))
    selected_ids = set(random.sample(sorted(eligible_ids), k))

    # Pass 2: filter lines whose video_number matches the sampled ids
    total_in = 0
    total_out = 0
    unique_written_ids = set()

    with inp.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            total_in += 1
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            vn = obj.get("video_number") or ""
            vid = extract_vid_int(vn.replace("video_", "") if vn.startswith("video_") else vn)
            if vid is None:
                qid = obj.get("question_id", "")
                m = VID_RE.search(qid)
                if m:
                    vid = int(m.group(1))
            if vid is not None and vid in selected_ids:
                fout.write(s + "\n")
                total_out += 1
                unique_written_ids.add(vid)

    print(f"Input lines scanned: {total_in}")
    print(f"Unique eligible ids in range [{args.min_id},{args.max_id}]: {len(eligible_ids)}")
    print(f"Requested ids: {args.n} | Selected ids: {len(selected_ids)} | Unique ids written: {len(unique_written_ids)}")
    print(f"Total lines written: {total_out}")
    print(f"Output: {out}")

if __name__ == "__main__":
    main()
