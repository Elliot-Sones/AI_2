#!/bin/bash
set -euo pipefail
cd /workspace/AI_2/rerun-20260905
mkdir -p artifacts
python -m venv --system-site-packages /workspace/AI_2/venv
PY=/workspace/AI_2/venv/bin/python
"$PY" -m pip install 'numpy==2.1.1' 'gymnasium==1.0.0' 'stable-baselines3==2.5.0' 'sb3-contrib==2.3.0' 'pygame==2.6.1' 'pymunk==6.2.1' 'opencv-python-headless==4.11.0.86' 'scikit-image==0.25.1' scikit-video gdown matplotlib imageio PyYAML pillow ipython pytest wandb psutil > artifacts/setup.log 2>&1
"$PY" -m pip freeze > artifacts/requirements-freeze.txt
nvidia-smi > artifacts/nvidia-smi.txt
lscpu > artifacts/lscpu.txt
"$PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name()); print((torch.ones(4,device="cuda")*2).cpu()); import train' > artifacts/runtime-check.log 2>&1
