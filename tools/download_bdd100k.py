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

import argparse
from pathlib import Path

import dataset_tools as dtools


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 下载位置
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用Dataset Ninja下载BDD100K Images 100K。"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "downloads" / "bdd100k",
        help="下载输出目录。",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="BDD100K: Images 100K",
        help="Dataset Ninja数据集名称。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使目标目录已存在，也重新调用下载流程。",
    )
    return parser.parse_args()


def looks_downloaded(
    output_dir: Path,
) -> bool:
    candidates = (
        output_dir / "bdd100k:-images-100k",
        output_dir / "bdd100k-images-100k",
    )

    return any(
        (candidate / "train" / "ann").exists()
        and (candidate / "train" / "img").exists()
        for candidate in candidates
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if looks_downloaded(output_dir) and not args.force:
        print("BDD100K already exists.")
        print("Location:", output_dir)
        return

    print("Downloading BDD100K...")

    dtools.download(
        dataset=args.dataset,
        dst_dir=str(output_dir),
    )

    print("\nBDD100K download finished.")
    print("Location:", output_dir)


if __name__ == "__main__":
    main()
