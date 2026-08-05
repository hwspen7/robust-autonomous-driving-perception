"""
训练BDD100K RT-DETR Transformer Baseline。

运行模式：
1. smoke：小规模测试数据读取、模型初始化和训练流程。
2. full：完整BDD100K训练和验证。

RT-DETR使用与YOLO11相同的YOLO格式数据：
datasets/yolo/bdd100k/data.yaml

保证：
- 图片一致；
- 标签一致；
- 类别顺序一致；
- train/val划分一致。

支持环境：
- Apple Silicon MPS；
- 单GPU CUDA；
- Kaggle 2×T4 CUDA。
"""

import argparse
from pathlib import Path

import torch
from ultralytics import RTDETR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = PROJECT_ROOT / "datasets" / "yolo" / "bdd100k" / "data.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "baselines" / "rtdetr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练BDD100K RT-DETR Baseline。"
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "full"),
        default="smoke",
        help="smoke测试流程，full正式训练。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rtdetr-l.pt",
        help="RT-DETR预训练权重。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="训练设备，例如0、0,1、mps。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="覆盖默认epoch数量。",
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
        help="数据加载线程数。",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="训练数据比例。",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="实验名称。",
    )
    return parser.parse_args()


def print_gpu_info() -> None:
    """
    输出当前GPU信息。
    用于确认Kaggle是否成功分配双T4。
    """
    if torch.cuda.is_available():
        print(f"CUDA GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(
                f"GPU {i}: "
                f"{torch.cuda.get_device_name(i)}"
            )
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        print("Device: Apple MPS")
    else:
        print("Device: CPU")


def resolve_device(
    requested_device: str | None,
) -> str:
    """
    自动选择训练设备。

    优先级：
    1. 用户指定；
    2. 多GPU CUDA；
    3. 单GPU CUDA；
    4. Apple MPS；
    5. CPU。

    Kaggle双T4返回：
    0,1
    """
    if requested_device is not None:
        return requested_device

    if torch.cuda.is_available():
        if torch.cuda.device_count() >= 2:
            return "0,1"
        return "0"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


def get_default_batch(
    device: str,
) -> int:
    """
    根据硬件设置默认batch。

    RT-DETR-L显存占用较高：
    双T4使用batch=8；
    单GPU使用batch=4；
    MPS使用batch=1。
    """
    if torch.cuda.is_available():
        if torch.cuda.device_count() >= 2:
            return 8
        return 4

    if device == "mps":
        return 1

    return 1


def validate_arguments(
    args: argparse.Namespace,
) -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"BDD100K data.yaml不存在: {DATA_YAML}"
        )

    if (
        args.fraction is not None
        and not 0 < args.fraction <= 1
    ):
        raise ValueError(
            "fraction必须位于0和1之间。"
        )


def build_train_args(
    args: argparse.Namespace,
    device: str,
) -> dict:
    model_name = Path(args.model).stem

    if args.profile == "smoke":
        return {
            "data": str(DATA_YAML),
            "epochs": (
                args.epochs
                if args.epochs is not None
                else 1
            ),
            "imgsz": 640,
            "batch": (
                args.batch
                if args.batch is not None
                else 1
            ),
            "device": device,
            "workers": (
                args.workers
                if args.workers is not None
                else 0
            ),
            "fraction": (
                args.fraction
                if args.fraction is not None
                else 0.005
            ),
            "val": False,
            "plots": False,
            "cache": False,
            "save": True,
            "amp": device not in {"mps", "cpu"},
            "project": str(OUTPUT_ROOT),
            "name": args.name or f"{model_name}_smoke",
            "exist_ok": True,
            "seed": 42,
            "deterministic": True,
            "verbose": True,
        }

    return {
        "data": str(DATA_YAML),
        "epochs": (
            args.epochs
            if args.epochs is not None
            else 50
        ),
        "imgsz": 640,
        "batch": (
            args.batch
            if args.batch is not None
            else get_default_batch(device)
        ),
        "device": device,
        "workers": (
            args.workers
            if args.workers is not None
            else 4
        ),
        "fraction": (
            args.fraction
            if args.fraction is not None
            else 1.0
        ),
        "val": True,
        "plots": True,
        "cache": False,
        "save": True,
        "save_period": 5,
        "amp": device not in {"mps", "cpu"},
        "project": str(OUTPUT_ROOT),
        "name": args.name or f"{model_name}_baseline",
        "exist_ok": False,
        "seed": 42,
        "deterministic": True,
        "patience": 10,
        "verbose": True,
    }


def main() -> None:
    args = parse_args()

    print("\n========== GPU Information ==========")
    print_gpu_info()

    validate_arguments(args)

    device = resolve_device(
        args.device
    )

    train_args = build_train_args(
        args,
        device,
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n========== RT-DETR Baseline ==========")
    print(f"Profile: {args.profile}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Data: {DATA_YAML}")
    print(f"Epochs: {train_args['epochs']}")
    print(f"Batch: {train_args['batch']}")
    print(f"Fraction: {train_args['fraction']}")
    print(f"Validation: {train_args['val']}")
    print(
        "Output:",
        OUTPUT_ROOT / train_args["name"],
    )

    model = RTDETR(
        args.model
    )

    model.train(
        **train_args
    )


if __name__ == "__main__":
    main()