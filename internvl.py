#!/usr/bin/env python3
import os
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel

MODEL_DIR  = "/home/ievab2/models/InternVL-Chat-V1-5"
FRAMES_DIR = "/home/ievab2/bunnies-frames"
PROMPT     = "These are frames of a video. First tell me how many frames you see. Then, describe what you see in the video."
N_FRAMES   = 8
SIZE       = 448
DTYPE      = torch.bfloat16 if torch.cuda.is_available() else torch.float32

IMAGENET_MEAN, IMAGENET_STD = (0.485,0.456,0.406), (0.229,0.224,0.225)
transform = T.Compose([
    T.Lambda(lambda img: img.convert("RGB")),
    T.Resize((SIZE, SIZE), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def dynamic_preprocess(img, max_num=1, size=SIZE, use_thumbnail=True):
    w,h = img.size
    blocks = 1
    img = img.resize((size, size))
    tiles = [img]
    if use_thumbnail and blocks != 1:
        tiles.append(img.resize((size, size)))
    return tiles

def load_image(path, max_num=1):
    img = Image.open(path).convert("RGB")
    tiles = dynamic_preprocess(img, max_num=max_num, size=SIZE, use_thumbnail=True)
    px = [transform(t) for t in tiles]
    return torch.stack(px)

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=False, local_files_only=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, low_cpu_mem_usage=True,
        local_files_only=True, device_map="auto", torch_dtype=DTYPE
    ).eval()

    frames = [os.path.join(FRAMES_DIR, f"{i:03d}.jpg") for i in range(N_FRAMES)]
    px_list, num_patches = [], []
    for p in frames:
        x = load_image(p, max_num=1)
        num_patches.append(x.shape[0])
        px_list.append(x)
    pixel_values = torch.cat(px_list, dim=0).to(DTYPE).to("cuda:0" if torch.cuda.is_available() else "cpu")

    prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches))])
    question = prefix + PROMPT
    gen = dict(max_new_tokens=128, do_sample=False)

    resp, _ = model.chat(
        tokenizer=tok,
        pixel_values=pixel_values,
        question=question,
        generation_config=gen,
        num_patches_list=num_patches,
        history=None,
        return_history=True,
    )
    print(resp)

if __name__ == "__main__":
    main()
