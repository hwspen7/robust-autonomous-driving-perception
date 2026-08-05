"""
训练BDD100K Faster R-CNN R50-FPN Baseline。

运行模式：
1. smoke：少量数据、1个epoch，只验证训练管线；
2. full：完整BDD100K训练和验证。

Faster R-CNN使用COCO格式数据：
datasets/coco/bdd100k/
"""
import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / "results" / ".cache"

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(CACHE_ROOT / "matplotlib"),
)
os.environ.setdefault(
    "TORCH_HOME",
    str(CACHE_ROOT / "torch"),
)

import torch
from mmengine import Config, init_default_scope
from mmengine.runner import Runner

try:
    from mmdet.utils import register_all_modules
except ModuleNotFoundError as exc:
    if exc.name not in {
        "mmdet",
        "mmcv",
    }:
        raise

    raise SystemExit(
        "MMDetection dependencies are not installed in this environment. "
        "Install mmdet and mmcv before running Faster R-CNN training."
    ) from exc

CONFIG_PATH = PROJECT_ROOT / "configs" / "mmdet" / "faster_rcnn_r50_fpn_bdd100k.py"
DATA_ROOT = PROJECT_ROOT / "datasets" / "coco" / "bdd100k"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "baselines" / "faster_rcnn"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练BDD100K Faster R-CNN Baseline。"
    )

    parser.add_argument(
        "--profile",
        choices=("smoke", "full"),
        default="smoke",
        help="smoke为管线测试，full为正式训练。",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="覆盖默认训练轮数。",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="覆盖默认batch size。",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="覆盖数据加载进程数。",
    )

    parser.add_argument(
        "--num-smoke-images",
        type=int,
        default=200,
        help="smoke模式使用的训练图片数量。",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="实验名称。",
    )

    return parser.parse_args()

def validate_environment(
        args: argparse.Namespace,
) -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"{CONFIG_PATH} does not exist"
        )

    required_paths = (
        DATA_ROOT / "annotations" / "instances_train.json",
        DATA_ROOT / "images" / "train",
    )

    if args.profile == "full":
        required_paths += (
            DATA_ROOT / "annotations" / "instances_val.json",
            DATA_ROOT / "images" / "val",
        )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist"
            )

    if args.profile == "full" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device is required for full Faster R-CNN training."
        )

def apply_data_paths(
        cfg: Config
) -> None:
    data_root = str(DATA_ROOT.resolve()) + "/"

    cfg.train_dataloader.dataset.data_root = data_root
    cfg.val_dataloader.dataset.data_root = data_root
    cfg.test_dataloader.dataset.data_root = data_root

    validation_annotation = str(DATA_ROOT / "annotations" / "instances_val.json")

    cfg.val_evaluator.ann_file = validation_annotation
    cfg.test_evaluator.ann_file = validation_annotation

def configure_smoke(
        cfg: Config,
        args: argparse.Namespace,
) -> None:
    cfg.work_dir = str(OUTPUT_ROOT / (args.name or "faster_rcnn_r50_fpn_smoke"))
    cfg.train_cfg.max_epochs = args.epochs if args.epochs is not None else 1
    cfg.train_dataloader.batch_size = args.batch if args.batch is not None else 1
    cfg.train_dataloader.num_workers = args.workers if args.workers is not None else 0
    cfg.train_dataloader.persistent_workers = False

    # BaseDataset支持indices参数，只读取前N个样本。
    cfg.train_dataloader.dataset.indices = (
        args.num_smoke_images
    )
    cfg.model.backbone.init_cfg = None

    for transform in cfg.train_dataloader.dataset.pipeline:
        if transform.type == "Resize":
            transform.scale = (
                640,
                384,
            )

    # 冒烟测试不运行完整10,000张验证集。
    cfg.val_cfg = None
    cfg.val_dataloader = None
    cfg.val_evaluator = None

    cfg.default_hooks.logger.interval = 10
    cfg.default_hooks.checkpoint.interval = 1
    cfg.default_hooks.checkpoint.save_best = None


def configure_full(
        cfg: Config,
        args: argparse.Namespace,
)->None:
    cfg.work_dir = str(OUTPUT_ROOT / (args.name or "faster_rcnn_r50_fpn_baseline"))
    cfg.train_cfg.max_epochs = args.epochs if args.epochs is not None else 12
    cfg.train_dataloader.batch_size = args.batch if args.batch is not None else 2
    workers = args.workers if args.workers is not None else 4
    cfg.train_dataloader.num_workers = workers
    cfg.val_dataloader.num_workers = workers
    cfg.test_dataloader.num_workers = workers

    persistent_workers = workers > 0

    cfg.train_dataloader.persistent_workers = persistent_workers
    cfg.val_dataloader.persistent_workers = persistent_workers
    cfg.test_dataloader.persistent_workers = persistent_workers

def main():
    args = parse_args()
    validate_environment(args)

    # 注册MMDetection中的模型、数据集和评估器。
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(CONFIG_PATH)
    apply_data_paths(cfg)

    if args.profile == "smoke":
        configure_smoke(
            cfg,
            args,
        )
    else:
        configure_full(
            cfg,
            args,
        )
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)

    print(
        "\n========== Faster R-CNN Baseline =========="
    )
    print(f"Profile：{args.profile}")
    print(f"Config：{CONFIG_PATH}")
    print(f"Data：{DATA_ROOT}")
    print(f"Epochs：{cfg.train_cfg.max_epochs}")
    print(
        "Batch：",
        cfg.train_dataloader.batch_size,
    )
    print(f"Output：{cfg.work_dir}")

    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()
