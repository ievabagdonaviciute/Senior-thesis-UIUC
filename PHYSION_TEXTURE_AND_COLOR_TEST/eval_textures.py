#!/usr/bin/env python3
import json
import math
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = ["internvl", "qwen"]
SPLITS = [1,2,3]
EPOCHS = [1,3,5]

BASE_ROOT = Path("/home/ievab2/run_models/PHYSION_TEXTURE_AND_COLOR_TEST")


def safe_mean(vals):
    vals=[v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals)/len(vals)


def std_err(vals):
    vals=[v for v in vals if v is not None]
    if len(vals)<=1:
        return 0.0
    m=sum(vals)/len(vals)
    var=sum((v-m)**2 for v in vals)/(len(vals)-1)
    sd=math.sqrt(var)
    return sd/math.sqrt(len(vals))


def load_texture_accuracy(path):
    """
    returns dict texture -> accuracy
    """
    if not path.exists():
        return {}

    stats=defaultdict(lambda:{"correct":0,"total":0})

    with path.open() as f:
        for ln in f:
            r=json.loads(ln)

            tex=r.get("texture")
            if tex is None:
                continue

            gt=(r.get("answer") or "").strip().lower()
            pr=(r.get("model_output_norm") or "").strip().lower()

            if gt not in ("yes","no") or pr not in ("yes","no"):
                continue

            stats[tex]["total"]+=1
            if gt==pr:
                stats[tex]["correct"]+=1

    acc={}
    for t,v in stats.items():
        acc[t]=v["correct"]/v["total"] if v["total"]>0 else None

    return acc


def run_model(model):

    texture_dir = BASE_ROOT/model/"results"/"texture"

    # collect accuracies
    data = defaultdict(lambda: defaultdict(list))
    textures=set()

    # base
    base_file = texture_dir/"base_results.jsonl"
    base_acc = load_texture_accuracy(base_file)

    for tex,v in base_acc.items():
        textures.add(tex)
        data["base"][tex].append(v)

    # splits/epochs
    for ep in EPOCHS:
        for s in SPLITS:

            p = texture_dir / f"SPLIT{s}_epochs{ep}_results.jsonl"
            acc = load_texture_accuracy(p)

            for tex,v in acc.items():
                textures.add(tex)
                data[f"e{ep}"][tex].append(v)

    textures = sorted(textures)

    stages=["base","e1","e3","e5"]

    means={stage:[] for stage in stages}
    errs={stage:[] for stage in stages}

    for tex in textures:
        for stage in stages:

            vals=data[stage][tex]

            m=safe_mean(vals)
            e=std_err(vals)

            means[stage].append(m if m is not None else 0)
            errs[stage].append(e)

    # plotting
    x=np.arange(len(textures))
    width=0.2

    fig,ax=plt.subplots(figsize=(14,6))

    offsets={
        "base":-1.5*width,
        "e1":-0.5*width,
        "e3":0.5*width,
        "e5":1.5*width
    }

    colors={
        "base":"black",
        "e1":"tab:blue",
        "e3":"tab:orange",
        "e5":"tab:green"
    }

    for stage in stages:
        ax.bar(
            x+offsets[stage],
            means[stage],
            width,
            yerr=errs[stage],
            capsize=4,
            label=stage,
            color=colors[stage]
        )

    ax.set_xticks(x)
    ax.set_xticklabels(textures,rotation=45,ha="right")
    ax.set_ylim(0,1)

    ax.set_ylabel("Accuracy")
    ax.set_title(f"{model} texture robustness")
    ax.legend()

    plt.tight_layout()

    out=BASE_ROOT/model/"results"/"texture_material_breakdown.png"
    plt.savefig(out,dpi=200)
    plt.close()

    print("saved",out)


def main():

    for m in MODELS:
        run_model(m)


if __name__=="__main__":
    main()