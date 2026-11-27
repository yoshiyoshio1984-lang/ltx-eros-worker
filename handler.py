"""
RunPod serverless worker: LTX-2.3 10Eros (NSFW fine-tune) via ComfyUI.

Same JSON contract as the app's existing LTX (diffusers) worker, so the app
needs no change apart from the endpoint ID:
  Input (job["input"]):
    prompt          str, required
    image_base64    str, optional - start frame; I2V when given, T2V when absent
    negative_prompt str, optional
    width           int, optional - default 768 (rounded to a multiple of 32)
    height          int, optional - default 512 (rounded to a multiple of 32)
    num_frames      int, optional - default 121 (rounded to 8k+1)
    frame_rate      float, optional - default 24
    num_inference_steps int, optional - default 25
    guidance_scale  float, optional - default 3.5 (used as KSampler cfg)
    seed            int, optional - default random
  Output:
    {"video_base64": str, "width": int, "height": int, "fps": 24,
     "num_frames": int, "seed": int, "model": "TenStrip/LTX2.3-10Eros",
     "has_audio": true, "mode": "ltx-i2v" | "ltx-t2v"}

Models (downloaded at worker init, ~42GB total, see download_models.sh):
  - TenStrip/LTX2.3-10Eros v1 FP8 mixed full checkpoint (~29.2GB)
  - Abliterated Gemma 3 12B text encoder, FP8 mixed (~12.8GB)
Workflow: core ComfyUI LTX nodes only, single pass, audio included.
Recommended GPU: 48GB band (FP8 22B checkpoint + 12B text encoder).
"""

import base64
import glob
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

import runpod

COMFYUI_DIR = "/comfyui"
COMFYUI_URL = "http://127.0.0.1:8188"
INPUT_DIR = os.path.join(COMFYUI_DIR, "input")
OUTPUT_DIR = os.path.join(COMFYUI_DIR, "output")

CKPT_NAME = "10Eros_v1_fp8mixed_learned.safetensors"
GEMMA_NAME = "gemma-3-12b-it-ablit-norms-biproj-fp8mixed.safetensors"
MODEL_LABEL = "TenStrip/LTX2.3-10Eros"

DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 512
DEFAULT_FRAMES = 121
DEFAULT_FPS = 24.0
DEFAULT_STEPS = 25
DEFAULT_CFG = 3.5
DEFAULT_NEGATIVE = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "subtitles, captions, text, watermark, still image, bad anatomy"
)
JOB_TIMEOUT_SEC = 1200


def log(*args):
    print(*args, flush=True)


# ------------------------------------------------------------------- init

def download_models():
    log("Downloading models (if needed) ...")
    subprocess.run(["/bin/bash", "/download_models.sh"], check=True)
    log("Models ready.")


def start_comfyui():
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188",
         "--disable-auto-launch"],
        cwd=COMFYUI_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _pump():
        for line in proc.stdout:
            print("[comfyui]", line.rstrip(), flush=True)

    threading.Thread(target=_pump, daemon=True).start()

    deadline = time.time() + 600
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"ComfyUI exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(COMFYUI_URL + "/system_stats", timeout=5) as r:
                if r.status == 200:
                    log("ComfyUI is ready.")
                    return proc
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("ComfyUI did not become ready within 600s")


# ------------------------------------------------------------------- helpers

def save_input_image(b64, name):
    raw = base64.b64decode(b64)
    path = os.path.join(INPUT_DIR, name)
    with open(path, "wb") as f:
        f.write(raw)
    return name


def build_workflow(prompt_text, negative_text, image_name,
                   width, height, num_frames, fps, steps, cfg, seed):
    wf = {}
    wf["1"] = {"class_type": "CheckpointLoaderSimple",
               "inputs": {"ckpt_name": CKPT_NAME}}
    wf["2"] = {"class_type": "LTXAVTextEncoderLoader",
               "inputs": {"text_encoder": GEMMA_NAME, "ckpt_name": CKPT_NAME,
                          "device": "default"}}
    wf["3"] = {"class_type": "LTXVAudioVAELoader",
               "inputs": {"ckpt_name": CKPT_NAME}}
    wf["4"] = {"class_type": "CLIPTextEncode",
               "inputs": {"clip": ["2", 0], "text": prompt_text}}
    wf["5"] = {"class_type": "CLIPTextEncode",
               "inputs": {"clip": ["2", 0], "text": negative_text}}
    wf["6"] = {"class_type": "LTXVConditioning",
               "inputs": {"positive": ["4", 0], "negative": ["5", 0],
                          "frame_rate": float(fps)}}
    wf["7"] = {"class_type": "EmptyLTXVLatentVideo",
               "inputs": {"width": width, "height": height,
                          "length": num_frames, "batch_size": 1}}
    latent_ref = ["7", 0]
    if image_name:
        wf["8"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
        wf["9"] = {"class_type": "ImageScale",
                   "inputs": {"image": ["8", 0], "upscale_method": "lanczos",
                              "width": width, "height": height,
                              "crop": "center"}}
        wf["10"] = {"class_type": "LTXVPreprocess",
                    "inputs": {"image": ["9", 0], "img_compression": 33}}
        wf["11"] = {"class_type": "LTXVImgToVideoInplace",
                    "inputs": {"vae": ["1", 2], "image": ["10", 0],
                               "latent": ["7", 0], "strength": 1.0}}
        latent_ref = ["11", 0]
    wf["12"] = {"class_type": "LTXVEmptyLatentAudio",
                "inputs": {"frames_number": num_frames,
                           "frame_rate": int(round(fps)), "batch_size": 1}}
    wf["13"] = {"class_type": "LTXVConcatAVLatent",
                "inputs": {"video_latent": latent_ref,
                           "audio_latent": ["12", 0]}}
    wf["14"] = {"class_type": "KSampler",
                "inputs": {"model": ["1", 0], "positive": ["6", 0],
                           "negative": ["6", 1], "latent_image": ["13", 0],
                           "seed": seed, "steps": steps, "cfg": cfg,
                           "sampler_name": "euler_ancestral",
                           "scheduler": "simple", "denoise": 1.0}}
    wf["15"] = {"class_type": "LTXVSeparateAVLatent",
                "inputs": {"av_latent": ["14", 0]}}
    wf["16"] = {"class_type": "VAEDecode",
                "inputs": {"samples": ["15", 0], "vae": ["1", 2]}}
    wf["17"] = {"class_type": "LTXVAudioVAEDecode",
                "inputs": {"samples": ["15", 1], "audio_vae": ["3", 0]}}
    wf["18"] = {"class_type": "CreateVideo",
                "inputs": {"images": ["16", 0], "audio": ["17", 0],
                           "fps": float(fps)}}
    wf["19"] = {"class_type": "SaveVideo",
                "inputs": {"video": ["18", 0], "filename_prefix": "ltx_eros",
                           "format": "mp4", "codec": "h264"}}
    return wf


def run_workflow(workflow):
    started = time.time()
    client_id = uuid.uuid4().hex
    payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
    req = urllib.request.Request(COMFYUI_URL + "/prompt", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
    if resp.get("node_errors"):
        raise RuntimeError(f"Workflow validation errors: {resp['node_errors']}")
    prompt_id = resp["prompt_id"]
    log(f"Queued ComfyUI prompt {prompt_id}")

    deadline = time.time() + JOB_TIMEOUT_SEC
    while time.time() < deadline:
        time.sleep(2)
        with urllib.request.urlopen(
                COMFYUI_URL + f"/history/{prompt_id}", timeout=30) as r:
            history = json.loads(r.read())
        if prompt_id not in history:
            continue
        entry = history[prompt_id]
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            msgs = [m for m in status.get("messages", [])
                    if m and m[0] == "execution_error"]
            raise RuntimeError(f"ComfyUI execution error: {msgs}")
        if status.get("completed"):
            # SaveVideo's history format varies by version; scanning the
            # output dir for the newest mp4 with our prefix is robust.
            files = glob.glob(os.path.join(OUTPUT_DIR, "**", "ltx_eros*.mp4"),
                              recursive=True)
            files = [f for f in files if os.path.getmtime(f) >= started - 5]
            if not files:
                raise RuntimeError("Workflow completed but produced no video file")
            newest = max(files, key=os.path.getmtime)
            with open(newest, "rb") as f:
                return f.read()
    raise RuntimeError(f"Workflow timed out after {JOB_TIMEOUT_SEC}s")


# ------------------------------------------------------------------- handler

def handler(job):
    job_input = job.get("input", {}) or {}

    prompt = (job_input.get("prompt") or "").strip()
    if not prompt:
        return {"error": "prompt is required"}
    negative = job_input.get("negative_prompt") or DEFAULT_NEGATIVE
    image_b64 = job_input.get("image_base64")

    width = int(job_input.get("width", DEFAULT_WIDTH))
    height = int(job_input.get("height", DEFAULT_HEIGHT))
    width = max(32, (width // 32) * 32)
    height = max(32, (height // 32) * 32)
    num_frames = int(job_input.get("num_frames", DEFAULT_FRAMES))
    num_frames = ((max(9, num_frames) - 1) // 8) * 8 + 1
    fps = float(job_input.get("frame_rate", DEFAULT_FPS))
    steps = int(job_input.get("num_inference_steps", DEFAULT_STEPS))
    cfg = float(job_input.get("guidance_scale", DEFAULT_CFG))
    seed_in = job_input.get("seed", -1)
    if seed_in is None or int(seed_in) < 0:
        seed = random.randint(0, 2**31 - 1)
    else:
        seed = int(seed_in)

    job_tag = uuid.uuid4().hex[:8]
    image_name = None
    try:
        if image_b64:
            image_name = save_input_image(image_b64, f"{job_tag}_start.png")
        workflow = build_workflow(prompt, negative, image_name,
                                  width, height, num_frames, fps,
                                  steps, cfg, seed)
        video_bytes = run_workflow(workflow)
    finally:
        if image_name:
            try:
                os.remove(os.path.join(INPUT_DIR, image_name))
            except OSError:
                pass

    return {
        "video_base64": base64.b64encode(video_bytes).decode(),
        "width": width,
        "height": height,
        "fps": fps,
        "num_frames": num_frames,
        "seed": seed,
        "model": MODEL_LABEL,
        "has_audio": True,
        "mode": "ltx-i2v" if image_name else "ltx-t2v",
    }


download_models()
start_comfyui()
print("Model loaded.", flush=True)
runpod.serverless.start({"handler": handler})
