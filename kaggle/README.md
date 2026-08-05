# Kaggle Run Guide

Recommended workflow:

1. GitHub stores code only.
2. Kaggle runtime downloads BDD100K with Dataset Ninja.
3. The helper script converts the data to YOLO/COCO and starts training.

This avoids uploading the full 8GB+ dataset from a local machine.

## 1. Create A Kaggle Notebook

In Kaggle:

1. Create a new Notebook.
2. Enable GPU accelerator.
3. Enable Internet.
4. Clone this repo.

Run:

```bash
!git clone https://github.com/hwspen7/robust-autonomous-driving-perception.git
%cd robust-autonomous-driving-perception
```

Install dependencies:

```bash
!pip install -q -r requirements-kaggle.txt
```

## 2. Smoke Run

This downloads BDD100K into `/kaggle/working`, converts train/val into YOLO and COCO, then runs a short YOLO smoke training.

```bash
!PROFILE=smoke \
  EPOCHS=1 \
  BATCH=8 \
  WORKERS=2 \
  DEVICE=0 \
  bash kaggle/run_yolo_kaggle.sh
```

## 3. Full YOLO Baseline

After smoke succeeds:

```bash
!PROFILE=full \
  EPOCHS=50 \
  BATCH=16 \
  WORKERS=4 \
  DEVICE=0 \
  bash kaggle/run_yolo_kaggle.sh
```

For dual-GPU Kaggle sessions, try:

```bash
DEVICE=0,1 BATCH=32
```

## 4. Optional Existing Dataset Input

If you later attach a Kaggle Dataset or a zip/tar archive instead of downloading at runtime, set `BDD100K_DATA_ROOT`.

Directory input:

```bash
!BDD100K_DATA_ROOT="/kaggle/input/bdd100k-supervisely/bdd100k:-images-100k" \
  PROFILE=smoke \
  bash kaggle/run_yolo_kaggle.sh
```

Archive input:

```bash
!BDD100K_DATA_ROOT="/kaggle/input/bdd100k-supervisely/bdd100k:-images-100k.zip" \
  PROFILE=smoke \
  bash kaggle/run_yolo_kaggle.sh
```

To disable automatic runtime download:

```bash
BDD100K_AUTO_DOWNLOAD=0
```

## 5. Optional Faster R-CNN

Run YOLO first. After data conversion is verified, install OpenMMLab dependencies:

```bash
!pip install -q -U openmim
!mim install -q "mmcv>=2.0.0rc4,<2.2.0"
!pip install -q "mmdet==3.3.0"
```

Smoke:

```bash
!python scripts/train_faster_rcnn_baseline.py \
  --profile smoke \
  --num-smoke-images 200 \
  --batch 1 \
  --workers 2
```

Full:

```bash
!python scripts/train_faster_rcnn_baseline.py \
  --profile full \
  --epochs 12 \
  --batch 2 \
  --workers 4
```

## 6. Outputs

Kaggle input data is read-only under `/kaggle/input`.
Runtime downloads, converted data, checkpoints, logs, and plots are written under `/kaggle/working/robust-autonomous-driving-perception`.

Commit the Notebook when training finishes so Kaggle persists the output files.

After a successful run, revoke any access token pasted into chat and create a new one in Kaggle account settings.
