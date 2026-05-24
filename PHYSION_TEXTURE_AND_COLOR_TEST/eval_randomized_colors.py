#!/usr/bin/env python3
# import json
# import math
# from pathlib import Path
# from typing import Optional, Tuple, List, Dict

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt


# MODELS = ["internvl", "qwen"]
# SPLITS = [1, 2, 3]
# EPOCHS = [1, 3, 5]
# TYPES = ["yellowz_redt", "redz_yellowt", "randomz_randomt"]

# BASE_ROOT = Path("/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST")


# def normalize_yes_no(x: str) -> str:
#     x = (x or "").strip().lower()
#     if x in ("yes", "no"):
#         return x
#     return ""


# def summarize_accuracy_for_type(path: Path, wanted_type: str) -> Optional[Dict[str, float]]:
#     """
#     Returns dict with keys:
#       correct, total, accuracy

#     Only keeps rows where:
#       - row["type"] == wanted_type
#       - answer and model_output_norm are both in {"yes", "no"}
#     """
#     if not path.exists():
#         return None

#     correct = 0
#     total = 0

#     with path.open("r", encoding="utf-8") as f:
#         for ln in f:
#             ln = ln.strip()
#             if not ln:
#                 continue

#             try:
#                 r = json.loads(ln)
#             except Exception:
#                 continue

#             row_type = (r.get("type") or "").strip()
#             if row_type != wanted_type:
#                 continue

#             gt = normalize_yes_no(r.get("answer"))
#             pr = normalize_yes_no(r.get("model_output_norm"))

#             if gt not in ("yes", "no"):
#                 continue
#             if pr not in ("yes", "no"):
#                 continue

#             total += 1
#             if gt == pr:
#                 correct += 1

#     acc = (correct / total) if total > 0 else None
#     return {"correct": correct, "total": total, "accuracy": acc}


# def mean_sample_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
#     vals = [v for v in vals if v is not None]
#     if not vals:
#         return None, None

#     m = sum(vals) / len(vals)
#     if len(vals) == 1:
#         return m, 0.0

#     var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
#     return m, math.sqrt(var)


# def sample_std_err(vals: List[float]) -> Optional[float]:
#     vals = [v for v in vals if v is not None]
#     if not vals:
#         return None
#     if len(vals) == 1:
#         return 0.0

#     _, sd = mean_sample_std(vals)
#     return sd / math.sqrt(len(vals))


# def write_results_jsonl(
#     out_path: Path,
#     base_res_by_type: Dict[str, Optional[Dict[str, float]]],
#     split_epoch_res_by_type: Dict[str, Dict[Tuple[int, int], Optional[Dict[str, float]]]],
# ):
#     """
#     Writes one block per type:
#       begin marker
#       base row
#       blank
#       rows for epoch 1 splits
#       blank
#       rows for epoch 3 splits
#       blank
#       rows for epoch 5 splits
#       blank
#       end marker
#       blank
#     """
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     def dump_row(f, row: Dict):
#         f.write(json.dumps(row, ensure_ascii=False) + "\n")

#     with out_path.open("w", encoding="utf-8") as f:
#         for t in TYPES:
#             dump_row(f, {"type": t, "section": "begin"})

#             base_res = base_res_by_type.get(t)
#             if base_res is None:
#                 dump_row(f, {"type": t, "tag": "base", "missing": True})
#             else:
#                 dump_row(
#                     f,
#                     {
#                         "type": t,
#                         "tag": "base",
#                         "accuracy": base_res["accuracy"],
#                         "correct": base_res["correct"],
#                         "total": base_res["total"],
#                     },
#                 )

#             f.write("\n")

#             for ep in EPOCHS:
#                 for s in SPLITS:
#                     r = split_epoch_res_by_type[t].get((s, ep))
#                     tag = f"SPLIT{s}_epochs{ep}"

#                     if r is None:
#                         dump_row(f, {"type": t, "tag": tag, "missing": True})
#                     else:
#                         dump_row(
#                             f,
#                             {
#                                 "type": t,
#                                 "tag": tag,
#                                 "accuracy": r["accuracy"],
#                                 "correct": r["correct"],
#                                 "total": r["total"],
#                             },
#                         )
#                 f.write("\n")

#             dump_row(f, {"type": t, "section": "end"})
#             f.write("\n")


# def save_grouped_bar_chart_png(
#     out_png: Path,
#     base_res_by_type: Dict[str, Optional[Dict[str, float]]],
#     epoch_accs_by_type: Dict[str, Dict[int, List[float]]],
# ):
#     """
#     Creates 12 bars total:
#       yellow zone, red target:     base e1 e3 e5
#       red zone, yellow target:     base e1 e3 e5
#       random zone, random target:  base e1 e3 e5

#     Error bars for e1/e3/e5 are standard error across splits.
#     Base bar has no error bar.
#     """
#     pretty_names = {
#         "yellowz_redt": "yellow zone, red target",
#         "redz_yellowt": "red zone, yellow target",
#         "randomz_randomt": "random zone, random target",
#     }

#     color_map = {
#         "yellowz_redt": "goldenrod",
#         "redz_yellowt": "firebrick",
#         "randomz_randomt": "dimgray",
#     }

#     labels = []
#     means = []
#     errs = []
#     colors = []

#     for t in TYPES:
#         base_acc = None
#         if base_res_by_type.get(t) is not None:
#             base_acc = base_res_by_type[t]["accuracy"]

#         labels.append("base")
#         means.append(base_acc)
#         errs.append(0.0)
#         colors.append(color_map[t])

#         for ep in EPOCHS:
#             vals = epoch_accs_by_type[t].get(ep, [])
#             m, _ = mean_sample_std(vals)
#             se = sample_std_err(vals)

#             labels.append(f"e{ep}")
#             means.append(m)
#             errs.append(se if se is not None else 0.0)
#             colors.append(color_map[t])

#     x = list(range(len(labels)))

#     plt.figure(figsize=(14, 5.5))
#     plt.bar(
#         x,
#         [0 if v is None else v for v in means],
#         yerr=errs,
#         capsize=5,
#         color=colors,
#     )

#     plt.xticks(x, labels)
#     plt.ylim(0, 1.0)
#     plt.ylabel("Accuracy")
#     plt.title("Randomized Colors Results (category: collide)", pad=28)

#     # vertical dashed separators between type groups
#     plt.axvline(3.5, linestyle="--", linewidth=1)
#     plt.axvline(7.5, linestyle="--", linewidth=1)

#     # group labels
#     group_centers = [1.5, 5.5, 9.5]
#     for xc, t in zip(group_centers, TYPES):
#         plt.text(
#             xc,
#             1.03,
#             pretty_names[t],
#             ha="center",
#             va="bottom",
#             transform=plt.gca().get_xaxis_transform(),
#             fontsize=11,
#         )

#     # annotate values
#     for i, v in enumerate(means):
#         if v is None:
#             plt.text(i, 0.02, "NA", ha="center", va="bottom", fontsize=9)
#         else:
#             plt.text(i, min(0.98, v + 0.02), f"{v:.3f}", ha="center", va="bottom", fontsize=9)

#     out_png.parent.mkdir(parents=True, exist_ok=True)
#     plt.tight_layout(rect=[0, 0, 1, 0.92])
#     plt.savefig(out_png, dpi=200)
#     plt.close()


# def run_one(model: str):
#     model_root = BASE_ROOT / model / "results"
#     task_dir = model_root / "randomized_colors"

#     base_path = task_dir / "base_results.jsonl"

#     base_res_by_type: Dict[str, Optional[Dict[str, float]]] = {}
#     split_epoch_res_by_type: Dict[str, Dict[Tuple[int, int], Optional[Dict[str, float]]]] = {}
#     epoch_accs_by_type: Dict[str, Dict[int, List[float]]] = {}

#     for t in TYPES:
#         base_res_by_type[t] = summarize_accuracy_for_type(base_path, t)
#         split_epoch_res_by_type[t] = {}
#         epoch_accs_by_type[t] = {ep: [] for ep in EPOCHS}

#         for ep in EPOCHS:
#             for s in SPLITS:
#                 p = task_dir / f"SPLIT{s}_epochs{ep}_results.jsonl"
#                 r = summarize_accuracy_for_type(p, t)

#                 if r is not None:
#                     split_epoch_res_by_type[t][(s, ep)] = r
#                     if r["accuracy"] is not None:
#                         epoch_accs_by_type[t][ep].append(r["accuracy"])

#     out_jsonl = model_root / "randomized_colors_results.jsonl"
#     out_png = model_root / "randomized_colors_results.png"

#     write_results_jsonl(out_jsonl, base_res_by_type, split_epoch_res_by_type)
#     save_grouped_bar_chart_png(out_png, base_res_by_type, epoch_accs_by_type)

#     print(f"[OK] {model}: wrote {out_jsonl}")
#     print(f"[OK] {model}: wrote {out_png}")


# def main():
#     for model in MODELS:
#         run_one(model)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import json
import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = ["internvl", "qwen"]
SPLITS = [1, 2, 3]
EPOCHS = [1, 3, 5]
TYPES = ["yellowz_redt", "redz_yellowt", "randomz_randomt"]

BASE_ROOT = Path("/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST")


def normalize_yes_no(x: str) -> str:
    x = (x or "").strip().lower()
    if x in ("yes", "no"):
        return x
    return ""


def summarize_accuracy_for_type(path: Path, wanted_type: str) -> Optional[Dict[str, float]]:
    """
    Returns dict with keys:
      correct, total, accuracy

    Only keeps rows where:
      - row["type"] == wanted_type
      - answer and model_output_norm are both in {"yes", "no"}
    """
    if not path.exists():
        return None

    correct = 0
    total = 0

    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue

            try:
                r = json.loads(ln)
            except Exception:
                continue

            row_type = (r.get("type") or "").strip()
            if row_type != wanted_type:
                continue

            gt = normalize_yes_no(r.get("answer"))
            pr = normalize_yes_no(r.get("model_output_norm"))

            if gt not in ("yes", "no"):
                continue
            if pr not in ("yes", "no"):
                continue

            total += 1
            if gt == pr:
                correct += 1

    acc = (correct / total) if total > 0 else None
    return {"correct": correct, "total": total, "accuracy": acc}


def mean_sample_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None

    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0

    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def sample_std_err(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0

    _, sd = mean_sample_std(vals)
    return sd / math.sqrt(len(vals))


def write_results_jsonl(
    out_path: Path,
    base_res_by_type: Dict[str, Optional[Dict[str, float]]],
    split_epoch_res_by_type: Dict[str, Dict[Tuple[int, int], Optional[Dict[str, float]]]],
):
    """
    Writes one block per type:
      begin marker
      base row
      blank
      rows for epoch 1 splits
      blank
      rows for epoch 3 splits
      blank
      rows for epoch 5 splits
      blank
      end marker
      blank
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def dump_row(f, row: Dict):
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with out_path.open("w", encoding="utf-8") as f:
        for t in TYPES:
            dump_row(f, {"type": t, "section": "begin"})

            base_res = base_res_by_type.get(t)
            if base_res is None:
                dump_row(f, {"type": t, "tag": "base", "missing": True})
            else:
                dump_row(
                    f,
                    {
                        "type": t,
                        "tag": "base",
                        "accuracy": base_res["accuracy"],
                        "correct": base_res["correct"],
                        "total": base_res["total"],
                    },
                )

            f.write("\n")

            for ep in EPOCHS:
                for s in SPLITS:
                    r = split_epoch_res_by_type[t].get((s, ep))
                    tag = f"SPLIT{s}_epochs{ep}"

                    if r is None:
                        dump_row(f, {"type": t, "tag": tag, "missing": True})
                    else:
                        dump_row(
                            f,
                            {
                                "type": t,
                                "tag": tag,
                                "accuracy": r["accuracy"],
                                "correct": r["correct"],
                                "total": r["total"],
                            },
                        )
                f.write("\n")

            dump_row(f, {"type": t, "section": "end"})
            f.write("\n")


def save_grouped_bar_chart_png(
    out_png: Path,
    base_res_by_type: Dict[str, Optional[Dict[str, float]]],
    epoch_accs_by_type: Dict[str, Dict[int, List[float]]],
):
    """
    Creates 12 bars total:
      yellow zone, red target:     base e1 e3 e5
      red zone, yellow target:     base e1 e3 e5
      random zone, random target:  base e1 e3 e5

    Error bars for e1/e3/e5 are standard error across splits.
    Base bar has no error bar.
    """
    pretty_names = {
        "yellowz_redt": "yellow zone, red target",
        "redz_yellowt": "red zone, yellow target",
        "randomz_randomt": "random zone, random target",
    }

    color_map = {
        "yellowz_redt": "goldenrod",
        "redz_yellowt": "firebrick",
        "randomz_randomt": "dimgray",
    }

    labels = []
    means = []
    errs = []
    colors = []

    for t in TYPES:
        base_acc = None
        if base_res_by_type.get(t) is not None:
            base_acc = base_res_by_type[t]["accuracy"]

        labels.append("base")
        means.append(base_acc)
        errs.append(0.0)
        colors.append(color_map[t])

        for ep in EPOCHS:
            vals = epoch_accs_by_type[t].get(ep, [])
            m, _ = mean_sample_std(vals)
            se = sample_std_err(vals)

            labels.append(f"e{ep}")
            means.append(m)
            errs.append(se if se is not None else 0.0)
            colors.append(color_map[t])

    x = list(range(len(labels)))

    plt.figure(figsize=(14, 5.5))
    plt.bar(
        x,
        [0 if v is None else v for v in means],
        yerr=errs,
        capsize=5,
        color=colors,
    )

    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Colors New Results (category: collide)", pad=28)

    plt.axvline(3.5, linestyle="--", linewidth=1)
    plt.axvline(7.5, linestyle="--", linewidth=1)

    group_centers = [1.5, 5.5, 9.5]
    for xc, t in zip(group_centers, TYPES):
        plt.text(
            xc,
            1.03,
            pretty_names[t],
            ha="center",
            va="bottom",
            transform=plt.gca().get_xaxis_transform(),
            fontsize=11,
        )

    for i, v in enumerate(means):
        if v is None:
            plt.text(i, 0.02, "NA", ha="center", va="bottom", fontsize=9)
        else:
            plt.text(i, min(0.98, v + 0.02), f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_png, dpi=200)
    plt.close()


def run_one(model: str):
    model_root = BASE_ROOT / model / "results"
    task_dir = model_root / "colors_new"

    base_path = task_dir / "base_results.jsonl"

    base_res_by_type: Dict[str, Optional[Dict[str, float]]] = {}
    split_epoch_res_by_type: Dict[str, Dict[Tuple[int, int], Optional[Dict[str, float]]]] = {}
    epoch_accs_by_type: Dict[str, Dict[int, List[float]]] = {}

    for t in TYPES:
        base_res_by_type[t] = summarize_accuracy_for_type(base_path, t)
        split_epoch_res_by_type[t] = {}
        epoch_accs_by_type[t] = {ep: [] for ep in EPOCHS}

        for ep in EPOCHS:
            for s in SPLITS:
                p = task_dir / f"SPLIT{s}_epochs{ep}_results.jsonl"
                r = summarize_accuracy_for_type(p, t)

                split_epoch_res_by_type[t][(s, ep)] = r
                if r is not None and r["accuracy"] is not None:
                    epoch_accs_by_type[t][ep].append(r["accuracy"])

    out_jsonl = model_root / "randomized_colors_new_results.jsonl"
    out_png = model_root / "randomized_colors_new_results.png"

    write_results_jsonl(out_jsonl, base_res_by_type, split_epoch_res_by_type)
    save_grouped_bar_chart_png(out_png, base_res_by_type, epoch_accs_by_type)

    print(f"[OK] {model}: wrote {out_jsonl}")
    print(f"[OK] {model}: wrote {out_png}")


def main():
    for model in MODELS:
        run_one(model)


if __name__ == "__main__":
    main()