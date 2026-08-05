"""
训练BDD100K YOLO11目标检测Baseline。

运行模式：
1. smoke：本地小规模冒烟测试；
2. full：CUDA设备上的完整Baseline训练。

冒烟测试只验证：
- 数据能否加载；
- 标签能否解析；
- 模型能否完成前向与反向传播；
- checkpoint能否正常保存。

正式实验结果必须使用full模式获得。
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
    if torch.cuda.is_available():
        print(f"CUDA GPU count: {torch.cuda.device_count()}")

        for device_id in range(torch.cuda.device_count()):
            print(f"GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
        return

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        print("Device: Apple MPS")
        return

    print("Device: CPU")

def resolve_device(
    requested_device: str | None,
) -> str:
    """
    自动选择训练设备。

    优先级：
    1. 用户手动指定；
    2. 多GPU CUDA；
    3. 单GPU CUDA；
    4. Apple MPS；
    5. CPU。
    """

    if requested_device is not None:
        return requested_device

    if torch.cuda.is_available():

        gpu_count = torch.cuda.device_count()

        if gpu_count >= 2:
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
    if torch.cuda.is_available():
        if torch.cuda.device_count() >= 2 and device == "0,1":
            return 32

        return 16

    if device == "mps":
        return 4

    return 2

def main():
    args = parse_args()

    print("\n========== GPU Information ==========")
    print_gpu_info()

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Unable to find the configuration file: {DATA_YAML}"
        )

    device = resolve_device(args.device)
    model = YOLO(args.model)

    if args.profile == "smoke":
        epochs = args.epochs or 1

        train_args = {
            "data": str(DATA_YAML),
            "epochs": epochs,
            "imgsz": 640,
            "batch": args.batch if args.batch is not None else 4,
            "device": device,
            "workers": args.workers if args.workers is not None else 0,
            "fraction": 0.01,
            "val": False,
            "plots": False,
            "save": True,
            "project": str(OUTPUT_ROOT),
            "name": "yolo11n_smoke",
            "exist_ok": True,
            "seed": 42,
            "deterministic": True,
            "verbose": True,
        }

    else:
        epochs = args.epochs or 50

        train_args = {
            "data": str(DATA_YAML),
            "epochs": epochs,
            "imgsz": 640,
            "batch": (
                args.batch
                if args.batch is not None
                else get_default_batch(device)
            ),
            "device": device,
            "workers": args.workers if args.workers is not None else 8,
            "fraction": 1.0,
            "val": True,
            "plots": True,
            "save": True,
            "save_period": 5,
            "project": str(OUTPUT_ROOT),
            "name": "yolo11n_baseline",
            "exist_ok": False,
            "seed": 42,
            "deterministic": True,
            "patience": 10,
            "verbose": True,
        }

    print("========== YOLO11 Baseline ==========")
    print(f"Profile：{args.profile}")
    print(f"Model：{args.model}")
    print(f"Device：{device}")
    print(f"Data：{DATA_YAML}")
    print(f"Epochs：{epochs}")
    print(f"Batch：{train_args['batch']}")
    print(f"Workers：{train_args['workers']}")

    model.train(**train_args)

if __name__ == "__main__":
    main()
