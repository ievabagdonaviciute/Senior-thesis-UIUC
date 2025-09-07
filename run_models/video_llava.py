# # video_llava.py
# import json
# import torch
# import cv2
# import numpy as np
# from pathlib import Path
# from PIL import Image
# from typing import Optional
# from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

# # --- defaults ---
# MODEL_ID = "LanguageBind/Video-LLaVA-7B-hf"
# NUM_FRAMES = 8
# MAX_NEW_TOKENS = 128
# # ----------------

# def read_video_cv2(path: str, k: int = NUM_FRAMES):
#     cap = cv2.VideoCapture(path)
#     if not cap.isOpened():
#         raise FileNotFoundError(f"Cannot open video: {path}")
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     if total <= 0:
#         cap.release()
#         raise RuntimeError(f"Video {path} has no frames")
#     idxs = np.linspace(0, max(total - 1, 0), num=k, dtype=int).tolist()
#     frames = []
#     for i in idxs:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#         ok, frame = cap.read()
#         if not ok:
#             continue
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frames.append(Image.fromarray(frame))  # PIL frames expected by processor
#     cap.release()
#     if not frames:
#         raise RuntimeError("Failed to extract frames.")
#     return frames, idxs

# def _load_model():
#     print("[videollava] loading model…", flush=True)

#     # Use fp16 on GPU, bf16 if available, else float32 on CPU
#     if torch.cuda.is_available():
#         dtype = torch.float16
#     elif torch.cuda.is_bf16_supported():
#         dtype = torch.bfloat16
#     else:
#         dtype = torch.float32

#     processor = VideoLlavaProcessor.from_pretrained(MODEL_ID)
#     model = VideoLlavaForConditionalGeneration.from_pretrained(
#         MODEL_ID, torch_dtype=dtype, device_map="auto"
#     )
#     model.eval()
#     print("[videollava] model loaded. cuda?", torch.cuda.is_available(), flush=True)

#     return processor, model

# def ask_video_llava(processor, model, video_path: str, question: str) -> str:
#     frames, _ = read_video_cv2(video_path, NUM_FRAMES)
#     prompt = f"USER: <video>\n{question}\nASSISTANT:"
#     #prompt = f"USER: <video>\nDescribe in great detail (number of objects, motion, color, materials, etc.) that you see in the video.\nASSISTANT:"
#     inputs = processor(text=prompt, videos=frames, return_tensors="pt").to(model.device)
#     with torch.inference_mode():
#         output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
#     decoded = processor.batch_decode(
#         output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
#     )[0]
#     if "ASSISTANT:" in decoded:
#         decoded = decoded.split("ASSISTANT:", 1)[1]
#     return decoded.strip()

# def eval_taskfile(task_path: str, out_path: str, counter_limit: Optional[int] = None):
#     """
#     Reads your JSONL with fields like:
#       category, question_id, prompt, ground_truth, video_number, video_path, question
#     Writes the SAME fields + 'model_output'.
#     """
#     print(f"[videollava] Starting task …")

#     processor, model = _load_model()
    
#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#     written = 0
#     with open(task_path, "r") as f_in, open(out_path, "w") as f_out:
#         for i, line in enumerate(f_in):
#             if not line.strip():
#                 continue
#             if counter_limit is not None and written >= counter_limit:
#                 break

#             try:
#                 row = json.loads(line)
#                 q = row.get("prompt") or row.get("question")
#                 vpath = row["video_path"]
#                 qid = row.get("question_id", row.get("qid", f"row{i}"))
#                 print(f"[videollava] Running {qid} …")

#                 pred = ask_video_llava(processor, model, vpath, q)

#                 out_record = dict(row)
#                 out_record["model_output"] = pred
#                 f_out.write(json.dumps(out_record) + "\n")
#                 written += 1

#             except Exception as e:
#                 # Log and continue with the next sample
#                 print(f"[videollava][ERROR] row {i}: {e}")

#     print(f"[videollava] wrote {written} rows to {out_path}")

# video_llava.py

# #################################################

# import json
# import torch
# import cv2
# import numpy as np
# from pathlib import Path
# from PIL import Image
# from typing import Optional
# from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
# import os

# # --- defaults ---
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# MODEL_ID = "/home/ievab2/models/Video-LLaVA-7B-hf"
# NUM_FRAMES = 8
# MAX_NEW_TOKENS = 128
# # ----------------

# def read_video_cv2(path: str, k: int = NUM_FRAMES):
#     cap = cv2.VideoCapture(path)
#     if not cap.isOpened():
#         raise FileNotFoundError(f"Cannot open video: {path}")
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     if total <= 0:
#         cap.release()
#         raise RuntimeError(f"Video {path} has no frames")
#     idxs = np.linspace(0, max(total - 1, 0), num=k, dtype=int).tolist()
#     frames = []
#     for i in idxs:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#         ok, frame = cap.read()
#         if not ok:
#             continue
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frames.append(Image.fromarray(frame))  # PIL frames expected by processor
#     cap.release()
#     if not frames:
#         raise RuntimeError("Failed to extract frames.")
#     return frames, idxs

# def _load_model():
#     print(f"[videollava] loading model from '{MODEL_ID}' …", flush=True)

#     # Use fp16 on GPU, bf16 if available, else float32 on CPU
#     if torch.cuda.is_available():
#         dtype = torch.float16
#     elif torch.cuda.is_bf16_supported():
#         dtype = torch.bfloat16
#     else:
#         dtype = torch.float32

#     local_only = os.path.isdir(MODEL_ID)
#     processor = VideoLlavaProcessor.from_pretrained(
#         MODEL_ID,
#         trust_remote_code=True,
#         local_files_only=local_only,
#     )
#     model = VideoLlavaForConditionalGeneration.from_pretrained(
#         MODEL_ID,
#         torch_dtype=dtype,
#         device_map="auto",
#         trust_remote_code=True,
#         local_files_only=local_only,
#     )
#     model.eval()
#     print("[videollava] model loaded. cuda?", torch.cuda.is_available(), flush=True)

#     return processor, model

# def ask_video_llava(processor, model, video_path: str, question: str) -> str:
#     frames, _ = read_video_cv2(video_path, NUM_FRAMES)
#     prompt = f"USER: <video>\n{question}\nASSISTANT:"
#     #prompt = f"USER: <video>\nDescribe in great detail (number of objects, motion, color, materials, etc.) that you see in the video.\nASSISTANT:"
#     inputs = processor(text=prompt, videos=frames, return_tensors="pt").to(model.device)
#     with torch.inference_mode():
#         output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
#     decoded = processor.batch_decode(
#         output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
#     )[0]
#     if "ASSISTANT:" in decoded:
#         decoded = decoded.split("ASSISTANT:", 1)[1]
#     return decoded.strip()

# def eval_taskfile(task_path: str, out_path: str, counter_limit: Optional[int] = None):
#     """
#     Reads your JSONL with fields like:
#       category, question_id, prompt, ground_truth, video_number, video_path, question
#     Writes the SAME fields + 'model_output'.
#     """
#     print(f"[videollava] Starting task …")

#     processor, model = _load_model()
    
#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#     written = 0
#     with open(task_path, "r") as f_in, open(out_path, "w") as f_out:
#         for i, line in enumerate(f_in):
#             if not line.strip():
#                 continue
#             if counter_limit is not None and written >= counter_limit:
#                 break

#             try:
#                 row = json.loads(line)
#                 q = row.get("prompt") or row.get("question")
#                 vpath = row["video_path"]
#                 qid = row.get("question_id", row.get("qid", f"row{i}"))
#                 print(f"[videollava] Running {qid} …")

#                 pred = ask_video_llava(processor, model, vpath, q)

#                 out_record = dict(row)
#                 out_record["model_output"] = pred
#                 f_out.write(json.dumps(out_record) + "\n")
#                 written += 1

#             except Exception as e:
#                 # Log and continue with the next sample
#                 print(f"[videollava][ERROR] row {i}: {e}")

#     print(f"[videollava] wrote {written} rows to {out_path}")

# ################################################# SINGLE LINE TEST FROM JSONL WITH MP4

# import os, json
# from pathlib import Path
# from typing import Optional

# import cv2
# import numpy as np
# import torch
# from PIL import Image
# from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

# # --- defaults ---
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# MODEL_ID = "/home/ievab2/models/Video-LLaVA-7B-hf"
# NUM_FRAMES = 8
# MAX_NEW_TOKENS = 128
# # ----------------

# def read_video_cv2(path: str, k: int = NUM_FRAMES):
#     cap = cv2.VideoCapture(path)
#     if not cap.isOpened():
#         raise FileNotFoundError(f"Cannot open video: {path}")
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     if total <= 0:
#         cap.release()
#         raise RuntimeError(f"Video {path} has no frames")
#     idxs = np.linspace(0, max(total - 1, 0), num=k, dtype=int).tolist()
#     frames = []
#     for i in idxs:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#         ok, frame = cap.read()
#         if not ok:
#             continue
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frames.append(Image.fromarray(frame))  # PIL frames expected by processor
#     cap.release()
#     if not frames:
#         raise RuntimeError("Failed to extract frames.")
#     return frames, idxs

# def _load_model():
#     print(f"[videollava] loading model from '{MODEL_ID}' …", flush=True)

#     if torch.cuda.is_available():
#         dtype = torch.float16
#     elif torch.cuda.is_bf16_supported():
#         dtype = torch.bfloat16
#     else:
#         dtype = torch.float32

#     local_only = os.path.isdir(MODEL_ID)
#     processor = VideoLlavaProcessor.from_pretrained(
#         MODEL_ID, trust_remote_code=True, local_files_only=local_only
#     )
#     model = VideoLlavaForConditionalGeneration.from_pretrained(
#         MODEL_ID, torch_dtype=dtype, device_map="auto",
#         trust_remote_code=True, local_files_only=local_only
#     )
#     model.eval()
#     print("[videollava] model loaded. cuda?", torch.cuda.is_available(), flush=True)
#     return processor, model

# def ask_video_llava(processor, model, video_path: str, question: str) -> str:
#     frames, _ = read_video_cv2(video_path, NUM_FRAMES)
#     prompt = f"USER: <video>\n{question}\nASSISTANT:"
#     device = next(model.parameters()).device
#     inputs = processor(text=prompt, videos=frames, return_tensors="pt").to(device)
#     with torch.inference_mode():

#         print("[videollava] before generate()")
#         output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
#         print("[videollava] after generate()")

#     print("[videollava] before batch_decode()")
#     decoded = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
#     print("[videollava] after batch_decode()")
    
#     if "ASSISTANT:" in decoded:
#         decoded = decoded.split("ASSISTANT:", 1)[1]
#     return decoded.strip()

# def eval_first_line(task_path: str, out_path: str):
#     """Read ONLY the first non-empty line, run a single fixed prompt, print the answer."""
#     with open(task_path, "r") as f_in:
#         first = None
#         for line in f_in:
#             if line.strip():
#                 first = json.loads(line)
#                 break
#     if first is None:
#         raise RuntimeError("No non-empty lines in task file.")

#     vpath = first["video_path"]
#     print(f"[videollava] Using video: {vpath}", flush=True)
#     cat = first["category"]
#     qid = first["question_id"]

#     print(f"category and question id of this is: {cat}, {qid}")

#     processor, model = _load_model()
#     answer = ask_video_llava(processor, model, vpath, "Describe what you see in the video (objetcs, colors, movement).")
#     print("\n[videollava] Answer:\n" + answer)

#     # --- append result to out JSONL ---
#     out_record = dict(first)
#     out_record["model_output"] = answer

#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#     with open(out_path, "a") as f_out:
#         f_out.write(json.dumps(out_record) + "\n")

#     print(f"[videollava] Appended result to {out_path}", flush=True)
#     # --- end append ---

#     return answer

# if __name__ == "__main__":
#     TASK_JSONL = "/home/ievab2/run_models/questions/clevrer_filtered_500.jsonl"
#     OUT_JSONL = "/home/ievab2/run_models/results/videollava_out.jsonl"
#     open(OUT_JSONL, "w").close() # clearing the jsonl file beforehand

#     eval_first_line(TASK_JSONL, OUT_JSONL)

# ################################################# SINGLE LINE TEST FROM JSONL WITH USING FRAMES INSTEAD OF MP4
# NOT PROPERLY FINISHED YET

# import os, json
# from pathlib import Path
# from typing import Optional

# import cv2
# import numpy as np
# import torch
# from PIL import Image
# from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

# # --- defaults ---
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# MODEL_ID = "/home/ievab2/models/Video-LLaVA-7B-hf"
# NUM_FRAMES = 8
# MAX_NEW_TOKENS = 128
# # ----------------

# FRAMES_ROOT = "/home/ievab2/run_models/CLEVRER_dataset/validation_frames"

# def read_video_frames_from_dir(row: dict, k: int = NUM_FRAMES):
#     frame_dir = frames_dir_from_row(row)
#     if not frame_dir.exists():
#         raise FileNotFoundError(f"No frames found at {frame_dir}")
    
#     frame_files = sorted(frame_dir.glob("*.jpg"))
#     if not frame_files:
#         raise RuntimeError(f"No jpg frames in {frame_dir}")
    
#     idxs = np.linspace(0, len(frame_files) - 1, num=k, dtype=int)
#     selected = [frame_files[i] for i in idxs]
#     frames = [Image.open(f).convert("RGB") for f in selected]
#     return frames, idxs


# def _load_model():
#     print(f"[videollava] loading model from '{MODEL_ID}' …", flush=True)

#     if torch.cuda.is_available():
#         dtype = torch.float16
#     elif torch.cuda.is_bf16_supported():
#         dtype = torch.bfloat16
#     else:
#         dtype = torch.float32

#     local_only = os.path.isdir(MODEL_ID)
#     processor = VideoLlavaProcessor.from_pretrained(
#         MODEL_ID, trust_remote_code=True, local_files_only=local_only
#     )
#     model = VideoLlavaForConditionalGeneration.from_pretrained(
#         MODEL_ID, torch_dtype=dtype, device_map="auto",
#         trust_remote_code=True, local_files_only=local_only
#     )
#     model.eval()
#     print("[videollava] model loaded. cuda?", torch.cuda.is_available(), flush=True)
#     return processor, model

# def ask_video_llava(processor, model, video_path: str, question: str) -> str:
#     frames, _ = read_video_cv2(video_path, NUM_FRAMES)
#     prompt = f"USER: <video>\n{question}\nASSISTANT:"
#     device = next(model.parameters()).device
#     inputs = processor(text=prompt, videos=frames, return_tensors="pt").to(device)
#     with torch.inference_mode():

#         print("[videollava] before generate()")
#         output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
#         print("[videollava] after generate()")

#     print("[videollava] before batch_decode()")
#     decoded = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
#     print("[videollava] after batch_decode()")
    
#     if "ASSISTANT:" in decoded:
#         decoded = decoded.split("ASSISTANT:", 1)[1]
#     return decoded.strip()

# def eval_first_line(task_path: str, out_path: str):
#     """Read ONLY the first non-empty line, run a single fixed prompt, print the answer."""
#     with open(task_path, "r") as f_in:
#         first = None
#         for line in f_in:
#             if line.strip():
#                 first = json.loads(line)
#                 break
#     if first is None:
#         raise RuntimeError("No non-empty lines in task file.")

#     vpath = first["video_path"]
#     print(f"[videollava] Using video: {vpath}", flush=True)
#     cat = first["category"]
#     qid = first["question_id"]

#     print(f"category and question id of this is: {cat}, {qid}")

#     processor, model = _load_model()
#     answer = ask_video_llava(processor, model, vpath, "Describe what you see in the video (objetcs, colors, movement).")
#     print("\n[videollava] Answer:\n" + answer)

#     # --- append result to out JSONL ---
#     out_record = dict(first)
#     out_record["model_output"] = answer

#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#     with open(out_path, "a") as f_out:
#         f_out.write(json.dumps(out_record) + "\n")

#     print(f"[videollava] Appended result to {out_path}", flush=True)
#     # --- end append ---

#     return answer

# if __name__ == "__main__":
#     TASK_JSONL = "/home/ievab2/run_models/questions/clevrer_filtered_500.jsonl"
#     OUT_JSONL = "/home/ievab2/run_models/results/videollava_out.jsonl"
#     open(OUT_JSONL, "w").close() # clearing the jsonl file beforehand

#     eval_first_line(TASK_JSONL, OUT_JSONL)

################### EXMAPLE WITH LIMIT USING FRAMES INSTEAD OF MP4
# WORKED on VISION-23!

import os, json, glob
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from PIL import Image
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

# --- config ---
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

MODEL_ID     = "/home/ievab2/models/Video-LLaVA-7B-hf"
FRAMES_ROOT  = "/home/ievab2/run_models/CLEVRER_dataset/validation_frames"
NUM_FRAMES   = 8
MAX_NEW_TOKENS = 128
# -------------

def _load_model():
    print(f"[videollava] loading model from '{MODEL_ID}' …", flush=True)

    if torch.cuda.is_available():
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    local_only = os.path.isdir(MODEL_ID)
    processor = VideoLlavaProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True, local_files_only=local_only
    )
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=dtype, device_map="auto",
        trust_remote_code=True, local_files_only=local_only
    )
    model.eval()
    print("[videollava] model loaded. cuda?", torch.cuda.is_available(), flush=True)
    return processor, model

def frames_dir_from_row(row: dict) -> Path:
    """
    Map the MP4 path to its frames folder, preserving the subdir:
      /.../video_validation/<chunk>/<video_id>.mp4
      -> /.../validation_frames/<chunk>/<video_id>/
    """
    vpath = Path(row["video_path"])
    chunk = vpath.parent.name      # e.g., "video_10000-11000"
    vid   = vpath.stem             # e.g., "video_10003"
    return Path(FRAMES_ROOT) / chunk / vid

def read_fixed_8_frames(dir_path: Path):
    """Load exactly 000.jpg .. 007.jpg from dir_path, in order."""
    frames = []
    idxs = list(range(8))
    missing = []
    for i in idxs:
        jpg = dir_path / f"{i:03d}.jpg"
        png = dir_path / f"{i:03d}.png"
        if jpg.exists():
            fp = jpg
        elif png.exists():
            fp = png
        else:
            missing.append(f"{i:03d}.jpg/.png")
            continue
        frames.append(Image.open(fp).convert("RGB"))
    if missing:
        raise FileNotFoundError(f"Missing frames in {dir_path}: {', '.join(missing)}")
    return frames, idxs


def ask_video_llava(processor, model, frames_dir: Path, question: str) -> str:
    frames, _ = read_fixed_8_frames(frames_dir)

    print(f"[videollava] loaded frames: {[f'{i:03d}' for i in range(8)]} from {frames_dir}", flush=True)

    prompt = f"USER: <video>\n{question}\nASSISTANT:"
    device = next(model.parameters()).device
    inputs = processor(text=prompt, videos=frames, return_tensors="pt").to(device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    decoded = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    if "ASSISTANT:" in decoded:
        decoded = decoded.split("ASSISTANT:", 1)[1]
    return decoded.strip()

def eval_task(task_path: str, out_path: str, counter_limit: Optional[int] = None):
    """
    Read the JSONL and run up to `counter_limit` rows (all if None).
    Writes each row + 'model_output' to out_path (overwrites each run).
    """
    print(f"[videollava] Starting task …", flush=True)
    processor, model = _load_model()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(task_path, "r") as f_in, open(out_path, "w") as f_out:
        print("[videollava] opened task and output files", flush=True)
        for i, line in enumerate(f_in):
            if not line.strip():
                continue
            if counter_limit is not None and written >= counter_limit:
                break

            try:
                row = json.loads(line)
                q = row.get("prompt") or row.get("question")
                if not q:
                    raise ValueError("Row missing 'prompt'/'question'")

                frames_dir = frames_dir_from_row(row)
                if not frames_dir.exists():
                    raise FileNotFoundError(f"Missing frames dir: {frames_dir}")

                qid = row.get("question_id", row.get("qid", f"row{i}"))
                print(f"[videollava] Running {qid} …", flush=True)
                print(f"[videollava] using frames {frames_dir} …", flush=True)

                pred = ask_video_llava(processor, model, frames_dir, q)

                out_record = dict(row)
                out_record["model_output"] = pred
                f_out.write(json.dumps(out_record) + "\n")
                f_out.flush()
                written += 1
                print(f"[videollava] wrote row {written}", flush=True)

            except Exception as e:
                # Log and continue with the next sample
                print(f"[videollava][ERROR] row {i}: {e}", flush=True)

    print(f"[videollava] wrote {written} rows to {out_path}", flush=True)

if __name__ == "__main__":
    TASK_JSONL = "/home/ievab2/run_models/questions/clevrer_filtered_500.jsonl"
    OUT_JSONL  = "/home/ievab2/run_models/results/videollava_out.jsonl"
    # set to 5 for a quick test, or None for all
    LIMIT = None
    eval_task(TASK_JSONL, OUT_JSONL, counter_limit=LIMIT)