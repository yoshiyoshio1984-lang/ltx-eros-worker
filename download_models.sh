#!/bin/bash
# Downloads the model files for the LTX-2.3 10Eros I2V workflow.
# Idempotent: files that already exist are skipped.
# Models live on the container disk (/comfyui/models); a fresh worker
# re-downloads them (~42GB) unless the image is baked later.
set -e

M=/comfyui/models
mkdir -p "$M/checkpoints" "$M/text_encoders"

dl() {
  # $1 = url, $2 = destination
  if [ ! -f "$2" ]; then
    echo "Downloading $2"
    wget -q -O "$2.part" "$1" && mv "$2.part" "$2"
  else
    echo "Already present: $2"
  fi
}

# TenStrip LTX2.3-10Eros v1, FP8 mixed (full checkpoint: transformer + VAEs), ~29.2GB
dl "https://huggingface.co/TenStrip/LTX2.3-10Eros/resolve/main/10Eros_v1_fp8mixed_learned.safetensors" \
   "$M/checkpoints/10Eros_v1_fp8mixed_learned.safetensors"

# Uncensored (abliterated) Gemma 3 12B text encoder for LTX-2.3, FP8 mixed, ~12.8GB
dl "https://huggingface.co/TenStrip/LTX2.3-10Eros/resolve/main/text_encoders/gemma-3-12b-it-ablit-norms-biproj-fp8mixed.safetensors" \
   "$M/text_encoders/gemma-3-12b-it-ablit-norms-biproj-fp8mixed.safetensors"

echo "All models ready."
