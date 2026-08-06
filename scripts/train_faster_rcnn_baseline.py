"""
Train the BDD100K Faster R-CNN R50-FPN baseline.

Faster R-CNN uses COCO-format BDD100K data.
"""

import argparse
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / "results" / ".cache"
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "mmdet" / "faster_rcnn_r50_fpn_bdd100k.py"
)
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "coco" / "bdd100k"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "baselines" / "faster_rcnn"

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(CACHE_ROOT / "matplotlib"),
)
os.environ.setdefault(
    "TORCH_HOME",
    str(CACHE_ROOT / "torch"),
)


def project_path(
    path: Path,
) -> Path:
    expanded = path.expanduser()

    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded

    return expanded.resolve(
        strict=False,
    )


def default_data_root() -> Path:
    for env_name in (
        "AUTODRIVE_COCO_DATA_ROOT",
        "BDD100K_COCO_ROOT",
        "COCO_DATA_ROOT",
    ):
        configured_path = os.getenv(
            env_name
        )

        if configured_path:
            return project_path(
                Path(configured_path)
            )

    return DEFAULT_DATA_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the BDD100K Faster R-CNN baseline."
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "full"),
        default="smoke",
        help="smoke validates the pipeline; full runs the baseline.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=default_data_root(),
        help="COCO BDD100K root containing annotations/ and images/.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="MMDetection config path.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from an MMDetection checkpoint.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override default epoch count.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override dataloader worker count.",
    )
    parser.add_argument(
        "--num-smoke-images",
        type=int,
        default=200,
        help="Number of training images used by smoke mode.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output project directory.",
    )
    return parser.parse_args()


def load_mmdet_dependencies() -> tuple[Any, Any, Any]:
    try:
        from mmengine import Config
        from mmengine.runner import Runner
        from mmdet.utils import register_all_modules
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "MMDetection dependencies are not installed in this environment. "
            "Install requirements-mmdet.txt and a compatible mmcv build before "
            "running Faster R-CNN training."
        ) from exc

    return Config, Runner, register_all_modules


def validate_environment(
    args: argparse.Namespace,
    data_root: Path,
    config_path: Path,
) -> Path | None:
    if not config_path.exists():
        raise FileNotFoundError(
            f"MMDetection config does not exist: {config_path}"
        )

    required_paths = [
        data_root / "annotations" / "instances_train.json",
        data_root / "images" / "train",
    ]

    if args.profile == "full":
        required_paths.extend(
            [
                data_root / "annotations" / "instances_val.json",
                data_root / "images" / "val",
            ]
        )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required COCO path does not exist: {path}"
            )

    if args.resume is None:
        return None

    checkpoint_path = project_path(
        args.resume
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Resume checkpoint does not exist: {checkpoint_path}"
        )

    return checkpoint_path


def torch_cuda_available() -> bool:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is not installed. Install requirements-mmdet.txt before "
            "running Faster R-CNN training."
        ) from exc

    return torch.cuda.is_available()


def apply_data_paths(
    cfg: Any,
    data_root: Path,
) -> None:
    resolved_data_root = str(data_root.resolve()) + "/"

    cfg.train_dataloader.dataset.data_root = resolved_data_root
    cfg.val_dataloader.dataset.data_root = resolved_data_root
    cfg.test_dataloader.dataset.data_root = resolved_data_root

    validation_annotation = str(
        data_root / "annotations" / "instances_val.json"
    )

    cfg.val_evaluator.ann_file = validation_annotation
    cfg.test_evaluator.ann_file = validation_annotation


def apply_checkpoint_resume(
    cfg: Any,
    checkpoint_path: Path | None,
) -> None:
    if checkpoint_path is None:
        return

    cfg.resume = True
    cfg.load_from = str(checkpoint_path)


def configure_smoke(
    cfg: Any,
    args: argparse.Namespace,
    output_root: Path,
) -> None:
    cfg.work_dir = str(
        output_root / (args.name or "faster_rcnn_r50_fpn_smoke")
    )
    cfg.train_cfg.max_epochs = args.epochs if args.epochs is not None else 1
    cfg.train_dataloader.batch_size = args.batch if args.batch is not None else 1
    cfg.train_dataloader.num_workers = args.workers if args.workers is not None else 0
    cfg.train_dataloader.persistent_workers = False

    # BaseDataset supports an integer indices value for the first N samples.
    cfg.train_dataloader.dataset.indices = args.num_smoke_images
    cfg.model.backbone.init_cfg = None

    for transform in cfg.train_dataloader.dataset.pipeline:
        if transform.type == "Resize":
            transform.scale = (
                640,
                384,
            )

    cfg.val_cfg = None
    cfg.val_dataloader = None
    cfg.val_evaluator = None

    cfg.default_hooks.logger.interval = 10
    cfg.default_hooks.checkpoint.interval = 1
    cfg.default_hooks.checkpoint.save_best = None


def configure_full(
    cfg: Any,
    args: argparse.Namespace,
    output_root: Path,
) -> None:
    cfg.work_dir = str(
        output_root / (args.name or "faster_rcnn_r50_fpn_baseline")
    )
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


def main() -> None:
    args = parse_args()
    data_root = project_path(
        args.data
    )
    config_path = project_path(
        args.config
    )
    output_root = project_path(
        args.project
    )

    checkpoint_path = validate_environment(
        args=args,
        data_root=data_root,
        config_path=config_path,
    )

    if args.profile == "full" and not torch_cuda_available():
        raise RuntimeError(
            "CUDA device is required for full Faster R-CNN training."
        )

    Config, Runner, register_all_modules = load_mmdet_dependencies()

    register_all_modules(
        init_default_scope=True
    )
    cfg = Config.fromfile(
        config_path
    )
    apply_data_paths(
        cfg=cfg,
        data_root=data_root,
    )

    if args.profile == "smoke":
        configure_smoke(
            cfg=cfg,
            args=args,
            output_root=output_root,
        )
    else:
        configure_full(
            cfg=cfg,
            args=args,
            output_root=output_root,
        )

    apply_checkpoint_resume(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
    )

    Path(cfg.work_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n========== Faster R-CNN Baseline ==========")
    print(f"Profile: {args.profile}")
    print(f"Config: {config_path}")
    print(f"Data: {data_root}")
    print(f"Resume: {checkpoint_path}")
    print(f"Epochs: {cfg.train_cfg.max_epochs}")
    print(f"Batch: {cfg.train_dataloader.batch_size}")
    print(f"Workers: {cfg.train_dataloader.num_workers}")
    print(f"Output: {cfg.work_dir}")

    runner = Runner.from_cfg(
        cfg
    )
    runner.train()


if __name__ == "__main__":
    main()
