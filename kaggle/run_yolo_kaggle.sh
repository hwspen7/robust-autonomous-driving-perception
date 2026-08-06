#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RAW_DATA_ROOT="${BDD100K_DATA_ROOT:-}"
DOWNLOAD_ROOT="${BDD100K_DOWNLOAD_ROOT:-$REPO_ROOT/datasets/downloads/bdd100k}"
EXTRACT_ROOT="${BDD100K_EXTRACT_ROOT:-$DOWNLOAD_ROOT}"
AUTO_DOWNLOAD="${BDD100K_AUTO_DOWNLOAD:-0}"
DATASET_DIR_NAME="bdd100k:-images-100k"
ALT_DATASET_DIR_NAME="bdd100k-images-100k"
PROFILE="${PROFILE:-smoke}"
MODEL="${MODEL:-yolo11n.pt}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-2}"

print_data_root() {
  local root="$1"

  if [ -d "$root/$DATASET_DIR_NAME/train/ann" ] && [ -d "$root/$DATASET_DIR_NAME/train/img" ]; then
    echo "$root/$DATASET_DIR_NAME"
    return 0
  fi

  if [ -d "$root/$ALT_DATASET_DIR_NAME/train/ann" ] && [ -d "$root/$ALT_DATASET_DIR_NAME/train/img" ]; then
    echo "$root/$ALT_DATASET_DIR_NAME"
    return 0
  fi

  return 1
}

extract_archive() {
  local archive_path="$1"

  mkdir -p "$EXTRACT_ROOT"

  if [ -d "$EXTRACT_ROOT/$DATASET_DIR_NAME/train/ann" ] || [ -d "$EXTRACT_ROOT/$ALT_DATASET_DIR_NAME/train/ann" ]; then
    return
  fi

  echo "Extracting $archive_path to $EXTRACT_ROOT" >&2

  case "$archive_path" in
    *.zip)
      unzip -q "$archive_path" -d "$EXTRACT_ROOT"
      ;;
    *.tar|*.tar.gz|*.tgz)
      tar -xf "$archive_path" -C "$EXTRACT_ROOT"
      ;;
    *)
      echo "Unsupported archive type: $archive_path" >&2
      exit 1
      ;;
  esac
}

resolve_data_root() {
  if [ -n "$RAW_DATA_ROOT" ] && [ -d "$RAW_DATA_ROOT/train/ann" ] && [ -d "$RAW_DATA_ROOT/train/img" ]; then
    echo "$RAW_DATA_ROOT"
    return
  fi

  local archive_path=""

  if [ -n "$RAW_DATA_ROOT" ] && [ -f "$RAW_DATA_ROOT" ]; then
    archive_path="$RAW_DATA_ROOT"
  elif [ -n "$RAW_DATA_ROOT" ] && [ -f "$RAW_DATA_ROOT.tar" ]; then
    archive_path="$RAW_DATA_ROOT.tar"
  elif [ -n "$RAW_DATA_ROOT" ] && [ -f "$RAW_DATA_ROOT.zip" ]; then
    archive_path="$RAW_DATA_ROOT.zip"
  fi

  if [ -n "$archive_path" ]; then
    extract_archive "$archive_path"
    if ! print_data_root "$EXTRACT_ROOT"; then
      echo "Archive extraction finished, but BDD100K train/ann and train/img were not found under $EXTRACT_ROOT." >&2
      exit 1
    fi
    return
  fi

  if print_data_root "$DOWNLOAD_ROOT"; then
    return
  fi

  if [ "$AUTO_DOWNLOAD" = "1" ]; then
    python tools/download_bdd100k.py --output-dir "$DOWNLOAD_ROOT"
    if ! print_data_root "$DOWNLOAD_ROOT"; then
      echo "Download finished, but BDD100K train/ann and train/img were not found under $DOWNLOAD_ROOT." >&2
      exit 1
    fi
    return
  fi

  echo "BDD100K data root not found." >&2
  echo "Set BDD100K_DATA_ROOT to a directory/archive." >&2
  echo "For Dataset Ninja runtime download, install requirements-dataset-ninja.txt and set BDD100K_AUTO_DOWNLOAD=1." >&2
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
  --data "$REPO_ROOT/datasets/yolo/bdd100k/data.yaml"
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
