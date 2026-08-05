#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RAW_DATA_ROOT="${BDD100K_DATA_ROOT:-/kaggle/input/bdd100k-supervisely/bdd100k:-images-100k}"
EXTRACT_ROOT="${BDD100K_EXTRACT_ROOT:-$REPO_ROOT/datasets/downloads/bdd100k}"
PROFILE="${PROFILE:-smoke}"
MODEL="${MODEL:-yolo11n.pt}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-2}"

resolve_data_root() {
  if [ -d "$RAW_DATA_ROOT/train/ann" ] && [ -d "$RAW_DATA_ROOT/train/img" ]; then
    echo "$RAW_DATA_ROOT"
    return
  fi

  local tar_path=""

  if [ -f "$RAW_DATA_ROOT" ]; then
    tar_path="$RAW_DATA_ROOT"
  elif [ -f "$RAW_DATA_ROOT.tar" ]; then
    tar_path="$RAW_DATA_ROOT.tar"
  fi

  if [ -n "$tar_path" ]; then
    mkdir -p "$EXTRACT_ROOT"

    if [ ! -d "$EXTRACT_ROOT/bdd100k:-images-100k/train/ann" ]; then
      echo "Extracting $tar_path to $EXTRACT_ROOT" >&2
      tar -xf "$tar_path" -C "$EXTRACT_ROOT"
    fi

    echo "$EXTRACT_ROOT/bdd100k:-images-100k"
    return
  fi

  echo "BDD100K data root not found or invalid: $RAW_DATA_ROOT" >&2
  echo "Set BDD100K_DATA_ROOT to a directory with train/ann and train/img, or to the uploaded .tar file." >&2
  exit 1
}

DATA_ROOT="$(resolve_data_root)"

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
