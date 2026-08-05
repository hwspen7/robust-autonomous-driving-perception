#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${BDD100K_DATA_ROOT:-/kaggle/input/bdd100k-supervisely/bdd100k:-images-100k}"
PROFILE="${PROFILE:-smoke}"
MODEL="${MODEL:-yolo11n.pt}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-2}"

if [ ! -d "$DATA_ROOT/train/ann" ] || [ ! -d "$DATA_ROOT/train/img" ]; then
  echo "BDD100K data root not found or invalid: $DATA_ROOT" >&2
  echo "Set BDD100K_DATA_ROOT to the Kaggle input directory containing train/ann and train/img." >&2
  exit 1
fi

echo "========== Kaggle YOLO Baseline =========="
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_ROOT"
echo "Profile: $PROFILE"
echo "Model: $MODEL"
echo "Device: $DEVICE"

python tools/convert_bdd100k.py \
  --data-root "$DATA_ROOT" \
  --split train \
  --formats yolo coco \
  --image-mode symlink \
  --clean

python tools/convert_bdd100k.py \
  --data-root "$DATA_ROOT" \
  --split val \
  --formats yolo coco \
  --image-mode symlink \
  --clean

TRAIN_ARGS=(
  --profile "$PROFILE"
  --model "$MODEL"
  --device "$DEVICE"
  --workers "$WORKERS"
)

if [ -n "${EPOCHS:-}" ]; then
  TRAIN_ARGS+=(--epochs "$EPOCHS")
fi

if [ -n "${BATCH:-}" ]; then
  TRAIN_ARGS+=(--batch "$BATCH")
fi

python scripts/train_yolo_baseline.py "${TRAIN_ARGS[@]}"
