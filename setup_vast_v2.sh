#!/bin/bash
# UTMIST AI^2 Training Setup for Vast.ai v2
# ==========================================
# Optimized for 4090 GPU training

set -e

echo "🚀 Starting Setup for UTMIST AI Training v2..."
echo "=============================================="

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

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project requirements
cd UTMIST-AI2-main
pip install -r requirements.txt

# Install extra dependencies
pip install shimmy>=0.2.1 tensorboard tqdm
e
echo "✅ Setup Complete!"
echo ""
echo "=============================================="
echo "To start training, run:"
echo ""
echo "  source venv/bin/activate"
echo "  cd .."
echo "  python3 train_utmist_v2.py --cfgFile utmist_config_v2.yaml"
echo ""
echo "To monitor training with TensorBoard:"
echo ""
echo "  tensorboard --logdir ./results/ppo_utmist_v2/tb --port 6006"
echo ""
echo "To run training in background (recommended):"
echo ""
echo "  nohup python3 train_utmist_v2.py --cfgFile utmist_config_v2.yaml > training.log 2>&1 &"
echo "  tail -f training.log"
echo ""
echo "=============================================="
