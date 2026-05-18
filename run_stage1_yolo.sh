#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/桌面/Spinal-qwen-finetune}"
TRAIN_JSON="${TRAIN_JSON:-${ROOT_DIR}/datasets/train_output/data_detcls_vl.json}"
VAL_JSON="${VAL_JSON:-${ROOT_DIR}/datasets/val_output/data_detcls_vl.json}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/datasets/yolo_spinal_lesion}"
PROJECT_DIR="${PROJECT_DIR:-${ROOT_DIR}/output/yolo_stage1}"
EXP_NAME="${EXP_NAME:-yolov8m_lesion}"
WEIGHTS="${PROJECT_DIR}/${EXP_NAME}/weights/best.pt"

python yolo_stage1_detection.py prepare \
  --train_json "${TRAIN_JSON}" \
  --val_json "${VAL_JSON}" \
  --output_dir "${DATA_DIR}"

python yolo_stage1_detection.py train \
  --data_yaml "${DATA_DIR}/spinal_lesion.yaml" \
  --model ./yolov8m.pt \
  --epochs 150 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --project "${PROJECT_DIR}" \
  --name "${EXP_NAME}"

python yolo_stage1_detection.py val \
  --weights "${WEIGHTS}" \
  --data_yaml "${DATA_DIR}/spinal_lesion.yaml" \
  --imgsz 640 \
  --batch 16 \
  --device 0

python yolo_stage1_detection.py recall \
  --weights "${WEIGHTS}" \
  --val_json "${VAL_JSON}" \
  --conf 0.05 \
  --max_det 10 \
  --thresholds 0.3,0.5 \
  --imgsz 640 \
  --device 0
