"""
训练BDD100K YOLO11目标检测Baseline。

运行模式：
1. smoke：本地小规模冒烟测试；
2. full：完整Baseline训练。

smoke模式验证：
- 数据是否能够加载；
- 标签是否正确解析；
- 模型是否能够完成训练；
- checkpoint是否正常保存。

full模式用于正式实验结果。
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = PROJECT_ROOT / "datasets" / "yolo" / "bdd100k" / "data.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "training" / "yolo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练BDD100K YOLO11 Baseline。"
    )

    parser.add_argument(
        "--profile",
        choices=("smoke", "full"),
        default="smoke",
        help="smoke为本地冒烟测试，full为完整训练。",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Ultralytics预训练模型。",
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
        help="覆盖数据加载线程数。",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="指定设备，例如mps、cpu、0或0,1。",
    )

    return parser.parse_args()


def print_gpu_info() -> None:
    """
    输出当前GPU环境。

    用于确认Kaggle双T4是否成功加载。
    """
    if torch.cuda.is_available():

        print(
            f"CUDA GPU数量: {torch.cuda.device_count()}"
        )

        for i in range(
            torch.cuda.device_count()
        ):
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

    Kaggle双T4:
        返回0,1
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
    根据硬件自动设置batch。

    YOLO11n较轻：
    双T4使用batch=32；
    单GPU使用batch=16；
    MPS使用batch=4。
    """

    if torch.cuda.is_available():

        if torch.cuda.device_count() >= 2:
            return 32

        return 16

    if device == "mps":
        return 4

    return 2


def main() -> None:
    args = parse_args()

    print(
        "\n========== GPU Information =========="
    )

    print_gpu_info()

    if not DATA_YAML.exists():

        raise FileNotFoundError(
            f"Unable to find configuration file: {DATA_YAML}"
        )

    device = resolve_device(
        args.device
    )

    model = YOLO(
        args.model
    )

    if args.profile == "smoke":

        train_args = {
            "data": str(DATA_YAML),
            "epochs": (
                args.epochs
                if args.epochs is not None
                else 1
            ),
            "imgsz": 640,
            "batch": 4,
            "device": device,
            "workers": 0,
            "fraction": 0.01,
            "val": False,
            "plots": False,
            "cache": False,
            "save": True,
            "amp": device not in {
                "mps",
                "cpu",
            },
            "project": str(OUTPUT_ROOT),
            "name": "yolo11n_smoke",
            "exist_ok": True,
            "seed": 42,
            "deterministic": True,
            "verbose": True,
        }

    else:

        train_args = {
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
                else 8
            ),
            "fraction": 1.0,
            "val": True,
            "plots": True,
            "cache": False,
            "save": True,
            "save_period": 5,
            "amp": device not in {
                "mps",
                "cpu",
            },
            "project": str(OUTPUT_ROOT),
            "name": "yolo11n_baseline",
            "exist_ok": False,
            "seed": 42,
            "deterministic": True,
            "patience": 10,
            "verbose": True,
        }

    print("\n========== YOLO11 Baseline ==========")
    print(f"Profile: {args.profile}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Data: {DATA_YAML}")
    print(f"Epochs: {train_args['epochs']}")
    print(f"Batch: {train_args['batch']}")
    print(f"Output: {OUTPUT_ROOT / train_args['name']}")

    model.train(
        **train_args
    )

if __name__ == "__main__":
    main()