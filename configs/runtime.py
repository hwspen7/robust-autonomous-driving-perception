import os
from dataclasses import dataclass
from pathlib import Path
import sys

try:
    import torch
except ModuleNotFoundError:
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RuntimeConfig:
    # 当前运行平台，仅用于日志和默认行为说明。
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
    自动检测当前运行平台，仅用于标识运行环境。

    路径选择不再依赖平台分支，避免普通Linux GPU服务器
    被错误归类到某个Notebook环境。
    """
    configured_platform = os.getenv(
        "AUTODRIVE_PLATFORM"
    )

    if configured_platform:
        return configured_platform

    if (
        os.getenv("KAGGLE_KERNEL_RUN_TYPE")
        or os.getenv("KAGGLE_URL_BASE")
        or (
            Path("/kaggle/input").exists()
            and Path("/kaggle/working").exists()
        )
    ):
        return "kaggle"

    if sys.platform == "darwin":
        return "macos"

    if sys.platform.startswith("linux"):
        return "linux"

    return sys.platform or "unknown"


def resolve_device() -> str:
    """
    自动选择训练设备。

    优先级：
    1. CUDA GPU；
    2. Apple Silicon MPS；
    3. CPU。
    """

    configured_device = os.getenv(
        "AUTODRIVE_DEVICE"
    )

    if configured_device:
        return configured_device

    if (
        torch is not None
        and torch.cuda.is_available()
    ):
        return "cuda:0"

    if (
        torch is not None
        and hasattr(torch.backends, "mps")
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

    configured_device = os.getenv(
        "AUTODRIVE_MMDET_DEVICE"
    )

    if configured_device:
        return configured_device

    if (
        torch is not None
        and torch.cuda.is_available()
    ):
        return "cuda:0"

    return "cpu"


def resolve_project_path(
    path: Path | str,
) -> Path:
    resolved_path = Path(path).expanduser()

    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path

    return resolved_path.resolve(
        strict=False,
    )


def resolve_data_root(
        platform: str
) -> Path:
    """
    确定数据集路径。

    优先读取环境变量，不在import阶段检查路径是否存在。
    """
    for env_name in (
        "AUTODRIVE_DATA_ROOT",
        "BDD100K_DATA_ROOT",
        "DATA_ROOT",
    ):
        configured_root = os.getenv(
            env_name
        )

        if configured_root:
            return resolve_project_path(
                configured_root
            )

    return resolve_project_path(
        "datasets"
    )


def resolve_output_root(
        platform: str
) -> Path:
    """
    确定实验输出路径。

    优先读取环境变量，不在import阶段创建目录。
    """
    for env_name in (
        "AUTODRIVE_OUTPUT_ROOT",
        "OUTPUT_ROOT",
    ):
        configured_root = os.getenv(
            env_name
        )

        if configured_root:
            return resolve_project_path(
                configured_root
            )

    return resolve_project_path(
        "results"
    )


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


def validate_data_root(
    data_root: Path | None = None,
) -> Path:
    root = data_root or RUNTIME.data_root

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {root}"
        )

    return root


def ensure_output_root(
    output_root: Path | None = None,
) -> Path:
    root = output_root or RUNTIME.output_root
    root.mkdir(
        exist_ok=True,
        parents=True,
    )

    return root


RUNTIME = build_runtime_config()
