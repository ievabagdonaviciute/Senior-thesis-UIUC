#!/usr/bin/env python3
import json, argparse
from pathlib import Path
import pandas as pd

ROOT = Path("/home/ievab2/run_models")
CURR_ROOT = ROOT / "Physion_finetuning" / "curriculum"  # <-- your new folder

def norm_cat(c: str):
    return c[0].upper() + c[1:].lower(), c.lower()

def hardness_csv_path(model: str, category_cap: str, task: str, feat: str, round_idx: int, alpha: float):
    # Matches your eval_hardness.py output pattern
    return ROOT / f"Physion_classifier/image_classifications/round{round_idx}/{model}/{category_cap}/" / \
           f"hardness_{task}_{feat}_round{round_idx}_alpha{alpha:g}.csv"

def questions_jsonl_path(category_cap: str, category_low: str, task: str):
    return ROOT / f"Physion_dataset/physion_out_questions/{category_cap}/{category_low}_{task}.jsonl"

def out_paths(model: str, category_cap: str, task: str, round_idx: int):
    out_dir = CURR_ROOT / model / category_cap
    return (
        out_dir / f"easy_{task}_round{round_idx}.jsonl",
        out_dir / f"hard_{task}_round{round_idx}.jsonl",
        out_dir
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["INTERNVL", "QWEN"])
    ap.add_argument("--category", required=True, choices=["Dominoes","Contain","Drop","dominoes","contain","drop"])
    ap.add_argument("--task", required=True, choices=["past","pred"])
    ap.add_argument("--feat", default="only_image", choices=["only_image","only_prompt","image_and_prompt"])
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=2.0)       # only for locating the CSV
    ap.add_argument("--threshold", type=float, default=0.5)   # hardness split
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--epsilon", type=float, default=0.01,
                help="Small extra weight added to hard examples (easy=1.0, hard=1.0+epsilon)")

    args = ap.parse_args()

    category_cap, category_low = norm_cat(args.category)
    questions_p = questions_jsonl_path(category_cap, category_low, args.task)
    hardness_p  = hardness_csv_path(args.model, category_cap, args.task, args.feat, args.round, args.alpha)
    easy_p, hard_p, out_dir = out_paths(args.model, category_cap, args.task, args.round)

    print(f"[curriculum] model={args.model} cat={category_cap} task={args.task} round={args.round}")
    print(f"  questions: {questions_p}")
    print(f"  hardness : {hardness_p}")
    print(f"  outputs  : {easy_p} | {hard_p}")

    if not questions_p.exists():
        raise FileNotFoundError(f"Questions JSONL missing: {questions_p}")
    if not hardness_p.exists():
        raise FileNotFoundError(f"Hardness CSV missing: {hardness_p}")

    if not args.overwrite and easy_p.exists() and hard_p.exists():
        print("[curriculum] Outputs already exist. Use --overwrite to regenerate.")
        return

    # --- load hardness ---
    h = pd.read_csv(hardness_p)
    if "qid" not in h.columns or "hardness" not in h.columns or "weight" not in h.columns:
        raise ValueError("Hardness CSV must contain columns: qid, hardness, weight, p_correct")
    h["qid_norm"] = h["qid"].astype(str)

    # --- load questions & index by qid ---
    q_by_id = {}
    with questions_p.open("r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip(): continue
            r = json.loads(ln)
            qid = r.get("qid", r.get("id"))
            if qid is None: continue
            q_by_id[str(qid)] = r
    print(f"[curriculum] questions loaded: {len(q_by_id)}")

    # --- join + split ---
    easy, hard = [], []
    missing = 0
    eps = float(args.epsilon)

    for _, row in h.iterrows():
        qid = row["qid_norm"]
        r = q_by_id.get(qid)
        if r is None:
            missing += 1
            continue

        item = dict(r)
        item["p_correct"] = float(row.get("p_correct", 0.0))
        item["hardness"] = float(row["hardness"])

        if item["hardness"] >= args.threshold:
            # HARD example: tiny bump above 1.0
            item["weight"] = 1.0 + eps
            hard.append(item)
        else:
            # EASY example: exactly 1.0
            item["weight"] = 1.0
            easy.append(item)


    out_dir.mkdir(parents=True, exist_ok=True)
    with easy_p.open("w", encoding="utf-8") as fe:
        for r in easy: fe.write(json.dumps(r) + "\n")
    with hard_p.open("w", encoding="utf-8") as fh:
        for r in hard: fh.write(json.dumps(r) + "\n")

    total_matched = len(easy) + len(hard)
    print(f"[curriculum] easy={len(easy)} hard={len(hard)} matched={total_matched} missing_qids={missing}")
    print(f"[curriculum] saved → {out_dir}")

if __name__ == "__main__":
    main()
