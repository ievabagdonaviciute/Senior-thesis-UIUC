#!/usr/bin/env python3
"""
Evaluate InternVL and Qwen on Physion tasks (Dominoes/Contain/Drop), for both
prediction and past. Logic unchanged; only paths are now category-parameterized.

It:
  • Reads 4 JSONL outputs (InternVL/Qwen × past/pred) for a chosen category
  • Compares `ground_truth` vs `model_output_norm`
  • Computes accuracy, TP, TN, FP, FN
  • Saves per-model stats as JSONLs in the same folders as the inputs
  • Generates accuracy bars, confusion matrices, and agreement charts
    into /home/ievab2/run_models/experiment_quick_physion/figures/<Category>/
    with filenames like: "accuracies_past (Drop).png"
"""

import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------- BASE ----------------
BASE = Path("/home/ievab2/run_models/Physion_finetuning")

# ---------------- HELPERS ----------------
TRUE_TOKENS  = {"true","yes","y","1","right","correct","collision","collide","will","will happen","happen"}
FALSE_TOKENS = {"false","no","n","0","wrong","incorrect","no collision","not collide","won't","will not","willn't"}

def to_bool(x):
    if isinstance(x, bool): return x
    if isinstance(x, (int, float)): return x == 1
    if isinstance(x, str):
        s = x.strip().lower().replace(".", "").replace("?", "").replace("!", "")
        if s in TRUE_TOKENS: return True
        if s in FALSE_TOKENS: return False
        if s.startswith("yes"): return True
        if s.startswith("no"): return False
    raise ValueError(f"Cannot convert to bool: {x}")

def load_jsonl(path: Path):
    y_true, y_pred = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            y_true.append(to_bool(obj["ground_truth"]))
            y_pred.append(to_bool(obj["model_output_norm"]))
    return y_true, y_pred

def confusion(y_true, y_pred):
    tp = sum(yt and yp for yt, yp in zip(y_true, y_pred))
    tn = sum((not yt) and (not yp) for yt, yp in zip(y_true, y_pred))
    fp = sum((not yt) and yp for yt, yp in zip(y_true, y_pred))
    fn = sum(yt and (not yp) for yt, yp in zip(y_true, y_pred))
    total = len(y_true)
    return {
        "accuracy_percent": 100*(tp+tn)/total if total else 0,
        "true_positive": tp, "true_negative": tn,
        "false_positive": fp, "false_negative": fn, "total": total
    }

def save_stats(model_file: Path, name: str, stats: dict):
    """Write score_<name>.jsonl in the same directory as model_file."""
    out = model_file.parent / f"score_{name}.jsonl"
    with out.open("w") as f: json.dump(stats, f)
    print(f"Saved {out}")

def plot_accuracy(task, accs, out, category_suffix):
    plt.figure(figsize=(5,4))
    bars = plt.bar(accs.keys(), accs.values())
    plt.ylabel("Accuracy (%)")
    plt.title(f"{task} — Accuracy ({category_suffix})")
    plt.ylim(0,100)
    for b, v in zip(bars, accs.values()):
        plt.text(b.get_x()+b.get_width()/2, v+1, f"{v:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

def plot_confusions(task, stats_dict, out, category_suffix):
    fig, axs = plt.subplots(1, len(stats_dict), figsize=(5*len(stats_dict),4))
    if len(stats_dict)==1: axs=[axs]
    for ax,(name,s) in zip(axs,stats_dict.items()):
        M = np.array([[s["true_negative"], s["false_positive"]],
                      [s["false_negative"], s["true_positive"]]])
        im=ax.imshow(M)
        ax.set_title(f"{name} — {task} ({category_suffix})")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["False","True"]); ax.set_yticklabels(["False","True"])
        for i in range(2):
            for j in range(2):
                ax.text(j,i,int(M[i,j]),ha="center",va="center",
                        color="white" if M[i,j]>M.max()/2 else "black")
        fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    plt.tight_layout(); plt.savefig(out,dpi=200); plt.close()

def plot_agreement(task, yt_q, yp_q, yt_i, yp_i, out, category_suffix):
    n = min(len(yt_q),len(yt_i))
    both_r=both_w=q_r_i_w=i_r_q_w=0
    for i in range(n):
        q_ok = yp_q[i]==yt_q[i]
        i_ok = yp_i[i]==yt_i[i]
        if q_ok and i_ok: both_r+=1
        elif (not q_ok) and (not i_ok): both_w+=1
        elif q_ok: q_r_i_w+=1
        elif i_ok: i_r_q_w+=1
    labels=["Both Right","Both Wrong","Qwen Right, InternVL Wrong","InternVL Right, Qwen Wrong"]
    vals=[both_r,both_w,q_r_i_w,i_r_q_w]
    plt.figure(figsize=(7,4))
    bars=plt.bar(labels,vals)
    plt.title(f"{task} — Agreement ({category_suffix})")
    plt.ylabel("Count")
    plt.xticks(rotation=20,ha="right")
    for b,v in zip(bars,vals):
        plt.text(b.get_x()+b.get_width()/2, v+1, str(v), ha="center")
    plt.tight_layout(); plt.savefig(out,dpi=200); plt.close()

# ---------------- MAIN ----------------
def main():
    ap = argparse.ArgumentParser(description="Score InternVL & Qwen on Physion category with fixed layout.")
    ap.add_argument("--category", "-c",
                    choices=["dominoes","contain","drop","Dominoes","Contain","Drop"],
                    required=True)
    ap.add_argument("--eval_type", choices=["round0_1epoch_4frames","round0_5epochs_4frames", "round0_noweights_1epoch_4frames"],
                    required=True, help="Tyoe (epochs + frames)")
    args = ap.parse_args()

    # Normalize names
    cat_title = args.category[0].upper() + args.category[1:].lower()  # Dominoes / Contain / Drop
    cat_lower = args.category.lower()                                  # dominoes / contain / drop

    # CHANGE:
    folder_name = args.eval_type

    # Input files per model
    internvl_past = BASE / f"INTERNVL/{folder_name}/{cat_title}/{cat_lower}_past_out.jsonl"
    
    internvl_pred = BASE / f"INTERNVL/{folder_name}/{cat_title}/{cat_lower}_pred_out.jsonl"
    # qwen_past     = BASE / f"QWEN/{cat_title}/{cat_lower}_past_out.jsonl"
    # qwen_pred     = BASE / f"QWEN/{cat_title}/{cat_lower}_pred_out.jsonl"

    # Figure dir for this category
    fig_dir = BASE / f"{folder_name}/INTERNVL/figures/{cat_title}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load
    yti_past, ypi_past = load_jsonl(internvl_past)
    yti_pred, ypi_pred = load_jsonl(internvl_pred)
    # ytq_past, ypq_past = load_jsonl(qwen_past)
    # ytq_pred, ypq_pred = load_jsonl(qwen_pred)

    # Stats
    s_i_past = confusion(yti_past, ypi_past)
    # s_q_past = confusion(ytq_past, ypq_past)
    s_i_pred = confusion(yti_pred, ypi_pred)
    # s_q_pred = confusion(ytq_pred, ypq_pred)

    # Save stats next to corresponding model outputs
    save_stats(internvl_past, "past", s_i_past)
    save_stats(internvl_pred, "pred", s_i_pred)
    # save_stats(qwen_past,     "past", s_q_past)
    # save_stats(qwen_pred,     "pred", s_q_pred)

    # Figures (filenames keep original base name + " (Category)")

    #============ REPLACE WHEN RUNNING FULL WITH QWEN
    plot_accuracy("PAST",
                {"InternVL":s_i_past["accuracy_percent"]},
                fig_dir / f"accuracies_past ({cat_title}).png",
                cat_title)

    plot_accuracy("PREDICTION",
                {"InternVL":s_i_pred["accuracy_percent"]},
                fig_dir / f"accuracies_pred ({cat_title}).png",
                cat_title)

    #==================================
    # plot_accuracy("PAST",
    #               {"InternVL":s_i_past["accuracy_percent"],"Qwen":s_q_past["accuracy_percent"]},
    #               fig_dir / f"accuracies_past ({cat_title}).png",
    #               cat_title)

    # plot_accuracy("PREDICTION",
    #               {"InternVL":s_i_pred["accuracy_percent"],"Qwen":s_q_pred["accuracy_percent"]},
    #               fig_dir / f"accuracies_pred ({cat_title}).png",
    #               cat_title)

    # plot_confusions("PAST",
    #                 {"InternVL":s_i_past,"Qwen":s_q_past},
    #                 fig_dir / f"confusions_past ({cat_title}).png",
    #                 cat_title)

    # plot_confusions("PREDICTION",
    #                 {"InternVL":s_i_pred,"Qwen":s_q_pred},
    #                 fig_dir / f"confusions_pred ({cat_title}).png",
    #                 cat_title)

    # plot_agreement("PAST", ytq_past, ypq_past, yti_past, ypi_past,
    #                fig_dir / f"agreement_past ({cat_title}).png",
    #                cat_title)

    # plot_agreement("PREDICTION", ytq_pred, ypq_pred, yti_pred, ypi_pred,
    #                fig_dir / f"agreement_pred ({cat_title}).png",
    #                cat_title)

    print("\n=== SUMMARY ===")
    #print(f"{cat_title} — PAST  — InternVL: {s_i_past['accuracy_percent']:.2f}%, Qwen: {s_q_past['accuracy_percent']:.2f}%")
    #print(f"{cat_title} — PRED  — InternVL: {s_i_pred['accuracy_percent']:.2f}%, Qwen: {s_q_pred['accuracy_percent']:.2f}%")
    print(f"Figures saved to {fig_dir}")

if __name__ == "__main__":
    main()
