#!/usr/bin/env python3
import json
import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt

CHECK_ROOT_BASE = Path("/shared/rsaas/ievab2/checkpoints/internvl")
OUT_ROOT_BASE   = Path("/home/ievab2/run_models/Physion_finetuning/INTERNVL")

def resolve_model_dir(run: str, flavor: str) -> Path:
    """
    Try to resolve a directory under:
      /shared/rsaas/ievab2/checkpoints/internvl/round0
    given:
      - either an exact subdir (e.g. 'both_past_1epochs_4frames')
      - or a shorthand like 'round0_1epoch_4frames' + flavor ('past'/'pred')
    """
    round_dir = CHECK_ROOT_BASE / "round0"

    if not round_dir.exists():
        raise RuntimeError(f"Round dir does not exist: {round_dir}")

    # 1) Exact match: /round0/<run>
    exact = round_dir / run
    if exact.is_dir():
        print(f"[INFO] Using exact folder: {exact}")
        return exact

    # 2) If run looks like 'round0_1epoch_4frames', map to both_<flavor>_1epochs_4frames
    m = re.match(r"round(\d+)_(\d+)epoch[s]?_(\d+)frames", run)
    if m:
        round_idx, epochs, frames = m.groups()
        dir_name = f"both_{flavor}_{epochs}epochs_{frames}frames"
        candidate = round_dir / dir_name
        if candidate.is_dir():
            print(f"[INFO] Using derived folder: {candidate}")
            return candidate
        else:
            raise RuntimeError(
                f"Derived folder {candidate} does not exist. "
                f"Available dirs: {[d.name for d in round_dir.iterdir() if d.is_dir()]}"
            )

    # 3) Fuzzy: look for any dir containing run as substring
    matches = [d for d in round_dir.iterdir() if d.is_dir() and run in d.name]
    if matches:
        print(f"[INFO] Using first fuzzy match: {matches[0]}")
        return matches[0]

    raise RuntimeError(
        f"No adapter folder found matching '{run}' under {round_dir}.\n"
        f"Available dirs: {[d.name for d in round_dir.iterdir() if d.is_dir()]}"
    )


def find_latest_trainer_state(model_dir: Path) -> Path:
    ckpts = [p for p in model_dir.glob("checkpoint-*/trainer_state.json")]
    if not ckpts:
        raise RuntimeError(f"No trainer_state.json found inside {model_dir}")

    # sort by checkpoint number: checkpoint-1385 -> 1385
    def ckpt_step(p: Path):
        try:
            return int(p.parent.name.split("-")[-1])
        except Exception:
            return 0

    ckpts.sort(key=ckpt_step)
    latest = ckpts[-1]
    print(f"[INFO] Using trainer_state.json from: {latest}")
    return latest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="Run label. Example: 'round0_1epoch_4frames' or 'both_past_1epochs_4frames'")
    ap.add_argument("--flavor", choices=["past", "pred"], default="past",
                    help="Used only when --run is a shorthand like 'round0_1epoch_4frames'.")
    args = ap.parse_args()

    run_label = args.run
    model_dir = resolve_model_dir(run_label, args.flavor)
    trainer_state_path = find_latest_trainer_state(model_dir)

    with open(trainer_state_path, "r") as f:
        state = json.load(f)

    losses = []
    steps  = []
    for entry in state.get("log_history", []):
        if "loss" in entry and "step" in entry:
            losses.append(entry["loss"])
            steps.append(entry["step"])

    if not losses:
        raise RuntimeError("No loss entries found in trainer_state.json")

    # Prepare output directory named exactly as the --run label
    out_root = OUT_ROOT_BASE / run_label
    out_root.mkdir(parents=True, exist_ok=True)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, marker=".", alpha=0.7)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Curve — {run_label}")
    plt.grid(True)

    out_png = out_root / "train_loss.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"[INFO] Saved: {out_png}")


if __name__ == "__main__":
    main()
