import os
from dataclasses import dataclass
from pathlib import Path

import torch

@dataclass(frozen=True)
class RuntimeConfig:
    # 当前运行平台
    platform: str
    # 通用计算设备
    device: str
    # MMDetection使用的设备 本地MPS对部分MMCV算子支持不完整，因此本地默认使用CPU
    mmdet_device: str
    # 数据集根目录
    data_root: Path
    # 实验输出根目录
    output_root: Path

def detect_platform() -> str:
    # Kaggle Notebook通常存在固定的输入和工作目录。
    if Path("/kaggle/input").exists() and Path("/kaggle/working").exists():
        return "kaggle"

    return "local"

def resolve_device() -> str:
    # 正式Kaggle训练优先使用CUDA
    if torch.cuda.is_available():
        return "cuda:0"

    # Apple Silicon本地环境使用MPS
    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"

def resolve_mmdet_device() -> str:
    # MMDetection依赖部分MMCV自定义算子，本地MPS可能无法运行NMS、ROI Align等操作。
    if torch.cuda.is_available():
        return "cuda:0"

    return "cpu"

def resolve_data_root(
        platform: str
) -> Path:
    custom_root = os.getenv("AUTODRIVE_DATA_ROOT")

    if custom_root:
        data_root = Path(custom_root)
    elif platform == "kaggle":
        data_root = Path("/kaggle/input/autonomous-driving-data")
    else:
        data_root = Path("datasets")

    if not data_root.exists():
        raise FileNotFoundError(
            f"{data_root} does not exist"
        )

    return data_root.resolve()

def resolve_output_root(
        platform: str
) -> Path:
    custom_root = os.getenv("AUTODRIVE_OUTPUT_ROOT")

    if custom_root:
        output_root = Path(custom_root)
    elif platform == "kaggle":
        output_root = Path("/kaggle/working/robust-autonomous-driving")
    else:
        output_root = Path("results")

    output_root.mkdir(exist_ok=True, parents=True)
    return output_root.resolve()

def build_runtime_config() -> RuntimeConfig:
    platform = detect_platform()
    return RuntimeConfig(
        platform=platform,
        device=resolve_device(),
        mmdet_device=resolve_mmdet_device(),
        data_root=resolve_data_root(platform),
        output_root=resolve_output_root(platform)
    )

RUNTIME = build_runtime_config()
