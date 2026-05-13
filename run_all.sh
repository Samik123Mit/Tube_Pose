#!/usr/bin/env bash
# End-to-end pipeline for tube pose estimation.
# Run from project root.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

echo "==> 1) Installing requirements"
pip install -q -r requirements.txt

echo "==> 2) Preparing dataset (single split)"
python src/prepare_dataset.py \
    --csv data/annotations.csv \
    --images data/images \
    --out data/yolo \
    --val-frac 0.2 \
    --seed 42 \
    --yaml configs/data.yaml

echo "==> 3) (Optional) Building 5-fold CV splits"
python src/cross_validation.py \
    --csv data/annotations.csv \
    --images data/images \
    --out data/cv \
    --n-folds 5 \
    --seed 42

echo "==> 4) Training YOLOv8s-pose"
python src/train_pose.py \
    --data configs/data.yaml \
    --weights yolov8s-pose.pt \
    --epochs 150 \
    --imgsz 640 \
    --batch 16 \
    --name tube_pose \
    --seed 42 \
    --patience 30

echo "==> 5) Running inference on validation set"
python src/inference.py \
    --weights models/tube_pose_best.pt \
    --images data/yolo/images/val \
    --out outputs/inference \
    --imgsz 640 \
    --conf 0.25 \
    --iou 0.5 \
    --tta \
    --ellipse-refine

echo "==> 6) Evaluating predictions"
python src/evaluate.py \
    --predictions outputs/inference/predictions.json \
    --csv data/annotations.csv \
    --images data/images \
    --out outputs/evaluation

echo "==> 7) Generating visualizations"
python src/visualize.py \
    --predictions outputs/inference/predictions.json \
    --csv data/annotations.csv \
    --images data/images \
    --metrics outputs/evaluation/metrics.json \
    --out outputs/viz \
    --max-images 20

echo "==> Done. See outputs/ for results."
