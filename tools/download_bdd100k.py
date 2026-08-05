"""
使用 Dataset Ninja 下载 BDD100K Images 100K

下载后保存:
datasets/downloads/bdd100k/

格式:
Supervisely

包含:
- train/img
- train/ann
- test/img
- test/ann
- meta.json
"""


from pathlib import Path

import dataset_tools as dtools


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 下载位置
DOWNLOAD_DIR = (
    PROJECT_ROOT
    /
    "datasets"
    /
    "downloads"
    /
    "bdd100k"
)


DOWNLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print(
    "Downloading BDD100K..."
)


dtools.download(
    dataset="BDD100K: Images 100K",
    dst_dir=str(DOWNLOAD_DIR),
)


print(
    "\nBDD100K download finished."
)

print(
    "Location:",
    DOWNLOAD_DIR
)