#!/usr/bin/env bash
set -e
cd /home/nszb/gym
source unitree-rl/bin/activate
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u tools/urdf4_amp/play_path_tracking.py \
  --vx 0.25 --vy 0 --wz 0 --duration_s 60 \
  --controller path --pose_source oracle \
  --sim_device cuda:0 --rl_device cuda:0 "$@"
