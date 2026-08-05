# Kaggle Run Guide

This project should run on Kaggle as:

1. GitHub stores code only.
2. Kaggle Dataset stores BDD100K data.
3. Kaggle Notebook clones the repo, converts data into `/kaggle/working`, then trains.

## 1. Upload BDD100K As A Kaggle Dataset

Keep the dataset private unless you have confirmed redistribution permissions.

Install and authenticate the Kaggle CLI locally:

```bash
python -m pip install kaggle
```

In Kaggle, open `Account` and create an API token. Put `kaggle.json` under:

```bash
~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Initialize metadata in the local BDD100K download directory:

```bash
cd datasets/downloads/bdd100k
kaggle datasets init -p .
```

Edit `dataset-metadata.json` so the id is:

```json
{
  "id": "hwspen7/bdd100k-supervisely",
  "title": "BDD100K Supervisely Raw"
}
```

Create the private dataset:

```bash
kaggle datasets create -p . --dir-mode tar
```

Expected uploaded structure:

```text
bdd100k:-images-100k/
  train/ann
  train/img
  val/ann
  val/img
  test/ann
  test/img
```

## 2. Create A Kaggle Notebook

In Kaggle:

1. Create a new Notebook.
2. Enable GPU accelerator.
3. Enable Internet for dependency and pretrained weight downloads.
4. Add the private dataset `bdd100k-supervisely` from the right-side Data panel.

Run these cells:

```bash
!git clone https://github.com/hwspen7/robust-autonomous-driving-perception.git
%cd robust-autonomous-driving-perception
```

```bash
!pip install -q -r requirements-kaggle.txt
```

Smoke run:

```bash
!BDD100K_DATA_ROOT="/kaggle/input/bdd100k-supervisely/bdd100k:-images-100k" \
  PROFILE=smoke \
  EPOCHS=1 \
  BATCH=8 \
  WORKERS=2 \
  DEVICE=0 \
  bash kaggle/run_yolo_kaggle.sh
```

Full YOLO baseline:

```bash
!BDD100K_DATA_ROOT="/kaggle/input/bdd100k-supervisely/bdd100k:-images-100k" \
  PROFILE=full \
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

## 3. Optional Faster R-CNN

Run YOLO first. After the data conversion is verified, install OpenMMLab dependencies:

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

## 4. Outputs

Kaggle input data is mounted read-only under `/kaggle/input`.
Converted data, checkpoints, logs, and plots are written under `/kaggle/working/robust-autonomous-driving-perception`.

Commit the Notebook when training finishes so Kaggle persists the output files.
