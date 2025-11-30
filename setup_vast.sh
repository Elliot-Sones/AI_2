#!/bin/bash

# Stop on error
set -e

echo "🚀 Starting Setup for UTMIST AI Training..."

# 1. Update System & Install System Dependencies
echo "📦 Installing system dependencies..."
apt-get update
apt-get install -y python3-venv python3-pip ffmpeg libsm6 libxext6 screen htop

# 2. Create Virtual Environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install Python Dependencies
echo "📚 Installing Python requirements..."
# Install torch first (usually pre-installed on Vast.ai images, but good to ensure)
pip install torch torchvision torchaudio

# Install project requirements
cd UTMIST-AI2-main
pip install -r requirements.txt

# Install extra dependencies that might be missing
pip install shimmy>=0.2.1 tensorboard

echo "✅ Setup Complete!"
echo "To start training, run:"
echo "source venv/bin/activate"
echo "export PYTHONPATH=\$PYTHONPATH:\$(pwd)"
echo "python3 ../train_utmist.py --cfgFile ../utmist_config.yaml"
