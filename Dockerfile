FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Current torch stack, pinned as one matching set (torch/torchvision/torchaudio).
RUN pip install --no-cache-dir \
    torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# ComfyUI pinned (LTX-2.3 nodes are in core: LTXV*, LTXAVTextEncoderLoader,
# CreateVideo / SaveVideo).
ARG COMFYUI_VERSION=v0.46.0
RUN git clone https://github.com/Comfy-Org/ComfyUI.git /comfyui \
    && cd /comfyui && git checkout ${COMFYUI_VERSION} \
    && pip install --no-cache-dir -r /comfyui/requirements.txt

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Bake the models (~42GB) into the image so workers never download at
# startup. Downloading at worker init filled the 50GB container disk and
# killed workers with exit code 1 before they could take a job.
COPY download_models.sh /download_models.sh
RUN chmod +x /download_models.sh && /bin/bash /download_models.sh

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
