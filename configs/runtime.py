import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class RuntimeConfig:
    # 当前运行平台：
    # local      本地Mac/Linux环境
    # kaggle     Kaggle Notebook
    # modelscope ModelScope DSW GPU环境
    platform: str

    # 通用计算设备：
    # cuda:0 / mps / cpu
    device: str

    # MMDetection使用的设备。
    # 本地MPS对部分MMCV算子支持不完整，
    # 因此本地默认使用CPU。
    mmdet_device: str

    # 数据集根目录
    data_root: Path

    # 实验输出根目录
    output_root: Path


def detect_platform() -> str:
    """
    自动检测当前运行平台。

    支持：
    1. Kaggle Notebook；
    2. ModelScope DSW；
    3. 本地环境。
    """

    # Kaggle Notebook通常存在固定输入和工作目录。
    if (
        Path("/kaggle/input").exists()
        and Path("/kaggle/working").exists()
    ):
        return "kaggle"

    # ModelScope DSW通常运行在workspace目录。
    if (
        Path("/mnt/workspace").exists()
        or Path("/home/jovyan").exists()
    ):
        return "modelscope"

    return "local"


def resolve_device() -> str:
    """
    自动选择训练设备。

    优先级：
    1. CUDA GPU；
    2. Apple Silicon MPS；
    3. CPU。
    """

    if torch.cuda.is_available():
        return "cuda:0"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


def resolve_mmdet_device() -> str:
    """
    设置MMDetection设备。

    MMDetection依赖部分MMCV CUDA算子。
    本地MPS对ROI Align、NMS等算子支持不完整，
    因此本地默认使用CPU。
    """

    if torch.cuda.is_available():
        return "cuda:0"

    return "cpu"


def resolve_data_root(
        platform: str
) -> Path:
    """
    根据运行平台确定数据集路径。

    优先读取环境变量：
    AUTODRIVE_DATA_ROOT

    方便用户自定义数据位置。
    """

    custom_root = os.getenv(
        "AUTODRIVE_DATA_ROOT"
    )

    if custom_root:
        data_root = Path(custom_root)

    elif platform == "kaggle":
        data_root = Path(
            "/kaggle/input/autonomous-driving-data"
        )

    elif platform == "modelscope":
        data_root = Path(
            "/mnt/workspace/datasets"
        )

    else:
        data_root = Path("datasets")

    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {data_root}"
        )

    return data_root.resolve()


def resolve_output_root(
        platform: str
) -> Path:
    """
    根据运行平台确定实验输出路径。

    优先读取环境变量：
    AUTODRIVE_OUTPUT_ROOT
    """

    custom_root = os.getenv(
        "AUTODRIVE_OUTPUT_ROOT"
    )

    if custom_root:
        output_root = Path(custom_root)

    elif platform == "kaggle":
        output_root = Path(
            "/kaggle/working/robust-autonomous-driving"
        )

    elif platform == "modelscope":
        output_root = Path(
            "/mnt/workspace/results"
        )

    else:
        output_root = Path("results")

    output_root.mkdir(
        exist_ok=True,
        parents=True,
    )

    return output_root.resolve()


def build_runtime_config() -> RuntimeConfig:
    """
    构建统一运行配置。
    """

    platform = detect_platform()

    return RuntimeConfig(
        platform=platform,
        device=resolve_device(),
        mmdet_device=resolve_mmdet_device(),
        data_root=resolve_data_root(platform),
        output_root=resolve_output_root(platform),
    )


def print_runtime_config() -> None:
    """
    打印当前运行环境。
    """

    print("=" * 50)
    print("Runtime Configuration")
    print(f"Platform      : {RUNTIME.platform}")
    print(f"Device        : {RUNTIME.device}")
    print(f"MMDet Device  : {RUNTIME.mmdet_device}")
    print(f"Data Root     : {RUNTIME.data_root}")
    print(f"Output Root   : {RUNTIME.output_root}")
    print("=" * 50)


RUNTIME = build_runtime_config()