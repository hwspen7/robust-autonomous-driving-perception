# Kaggle Run Guide

Recommended workflow:

1. GitHub stores code only.
2. Kaggle reads an attached or pre-extracted BDD100K dataset.
3. The helper script converts the data to YOLO/COCO and starts training.

Dataset Ninja runtime download is still available, but it is no longer part of
the default Kaggle dependency stack because it pulls in extra Supervisely
dependencies.

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

If you need conversion utilities outside the helper script:

```bash
!pip install -q -r requirements-data-conversion.txt
```

## 2. Smoke Run

Attach or extract BDD100K first, then point `BDD100K_DATA_ROOT` at the
Supervisely-format directory containing `train/ann` and `train/img`.

```bash
!BDD100K_DATA_ROOT="/kaggle/input/bdd100k-supervisely/bdd100k:-images-100k" \
  PROFILE=smoke \
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

If you attach a Kaggle Dataset or a zip/tar archive, set `BDD100K_DATA_ROOT`.

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

Dataset Ninja runtime download is optional:

```bash
!pip install -q -r requirements-dataset-ninja.txt
BDD100K_AUTO_DOWNLOAD=1
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
  --data datasets/coco/bdd100k \
  --num-smoke-images 200 \
  --batch 1 \
  --workers 2
```

Full:

```bash
!python scripts/train_faster_rcnn_baseline.py \
  --profile full \
  --data datasets/coco/bdd100k \
  --epochs 12 \
  --batch 2 \
  --workers 4
```

## 6. Outputs

Kaggle input data is read-only under `/kaggle/input`.
Runtime downloads, converted data, checkpoints, logs, and plots are written under `/kaggle/working/robust-autonomous-driving-perception`.

Commit the Notebook when training finishes so Kaggle persists the output files.

After a successful run, revoke any access token pasted into chat and create a new one in Kaggle account settings.
