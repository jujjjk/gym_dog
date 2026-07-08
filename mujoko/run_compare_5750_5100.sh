#!/usr/bin/env bash
set -e

cd /home/nszb/gym/mujoko

mkdir -p logs/sim2sim_compare_5750_5100

DURATION=60
MODEL=models/fanfan_scene.xml

run_one () {
  NAME=$1
  POLICY=$2
  VX=$3
  VY=$4
  YAW=$5
  TAG=$6

  echo "========================================"
  echo "RUN: ${NAME}  command=${VX} ${VY} ${YAW}  tag=${TAG}"
  echo "========================================"

  ./.venv/bin/python sim2sim.py \
    --policy "${POLICY}" \
    --model "${MODEL}" \
    --duration "${DURATION}" \
    --command "${VX}" "${VY}" "${YAW}" \
    2>&1 | tee "logs/sim2sim_compare_5750_5100/${NAME}_${TAG}.txt"
}

# 5750
run_one 5750 models/fanfan_desat_torque_5750.onnx 0.20 0.00 0.00 straight_020
run_one 5750 models/fanfan_desat_torque_5750.onnx 0.25 0.00 0.00 straight_025
run_one 5750 models/fanfan_desat_torque_5750.onnx 0.30 0.00 0.00 straight_030
run_one 5750 models/fanfan_desat_torque_5750.onnx 0.35 0.00 0.00 straight_035
run_one 5750 models/fanfan_desat_torque_5750.onnx 0.00 0.03 0.00 left_003
run_one 5750 models/fanfan_desat_torque_5750.onnx 0.00 0.05 0.00 left_005
run_one 5750 models/fanfan_desat_torque_5750.onnx 0.00 0.07 0.00 left_007
run_one 5750 models/fanfan_desat_torque_5750.onnx 0.20 0.07 0.00 diag_020_007

# 5100
run_one 5100 models/fanfan_yaw_clean_5100.onnx 0.20 0.00 0.00 straight_020
run_one 5100 models/fanfan_yaw_clean_5100.onnx 0.25 0.00 0.00 straight_025
run_one 5100 models/fanfan_yaw_clean_5100.onnx 0.30 0.00 0.00 straight_030
run_one 5100 models/fanfan_yaw_clean_5100.onnx 0.35 0.00 0.00 straight_035
run_one 5100 models/fanfan_yaw_clean_5100.onnx 0.00 0.03 0.00 left_003
run_one 5100 models/fanfan_yaw_clean_5100.onnx 0.00 0.05 0.00 left_005
run_one 5100 models/fanfan_yaw_clean_5100.onnx 0.00 0.07 0.00 left_007
run_one 5100 models/fanfan_yaw_clean_5100.onnx 0.20 0.07 0.00 diag_020_007
