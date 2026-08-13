#!/bin/bash
set -e

echo "============================================================"
echo "Setting up Qwen3-TTS Fine-Tuning Environment for ARM64/DGX"
echo "============================================================"

# 1. System dependencies
echo "Checking for ffmpeg..."
if ! command -v ffmpeg &> /dev/null
then
    echo "ffmpeg could not be found. It is required for pydub."
    echo "Please install it manually using: sudo apt-get update && sudo apt-get install -y ffmpeg"
    echo "Or run this script with sudo privileges for apt (not recommended for pip)."
else
    echo "ffmpeg is installed."
fi

# 2. Create Virtual Environment
echo "Creating Python 3.12 virtual environment..."
python3.12 -m venv env
source env/bin/activate

# Upgrade pip
pip install --upgrade pip

# 3. Install PyTorch
echo "Installing PyTorch..."
# Note: You may need to change the torch version/index if your DGX Spark 
# requires a specific Nvidia wheel for CUDA 13 on ARM64. 
# We're using standard torch installation here based on your feedback.
pip install torch torchaudio

# 4. Install Flash-Attention prebuilt wheel
echo "Installing flash-attention prebuilt wheel for ARM64..."
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.48/flash_attn-2.8.3+cu130torch2.13-cp312-cp312-linux_aarch64.whl

# 5. Install the remaining requirements
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "============================================================"
echo "Setup complete! All models/weights will be stored inside ./env/hf_cache"
echo ""
echo "To activate your environment before running the training script, use:"
echo "source env/bin/activate"
echo "============================================================"
