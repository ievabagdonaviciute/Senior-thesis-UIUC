# main.py
import argparse
from pathlib import Path
import video_llava

def run_on_dataset(dataset: str, model: str):
    # Hardcoded paths
    TASK_JSONL = "/home/ievab2/run_models/questions/clevrer_filtered_500.jsonl"
    OUT_JSONL  = "/home/ievab2/run_models/results/videollava_out.jsonl"
    LIMIT     = None   # set e.g. 50 to only run first 50 rows\; if None - it will run the whole TASK_JSONL

    if dataset.upper() != "CLEVRER":
        raise ValueError(f"Unknown dataset {dataset}")

    if model.lower() == "videollava":

        print(f"Running Video-LLaVA on {TASK_JSONL} …")

        video_llava.eval_task(
            task_path=TASK_JSONL,
            out_path=OUT_JSONL,
            counter_limit=LIMIT
        )
    elif model.lower() == "qwen":
        print("Qwen not yet wired up.")
    elif model.lower() == "internvl":
        print("InternVL not yet wired up.")
    else:
        raise ValueError(f"Unknown model {model}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset name (e.g., CLEVRER)")
    parser.add_argument("model", help="Model to run (videollava, qwen, internvl)")
    args = parser.parse_args()
    run_on_dataset(args.dataset, args.model)

if __name__ == "__main__":
    main()
