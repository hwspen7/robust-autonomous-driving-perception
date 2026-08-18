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
    platform: str
    device: str
    mmdet_device: str
    data_root: Path
    output_root: Path

def detect_platform() -> str:
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
    platform = detect_platform()

    return RuntimeConfig(
        platform=platform,
        device=resolve_device(),
        mmdet_device=resolve_mmdet_device(),
        data_root=resolve_data_root(platform),
        output_root=resolve_output_root(platform),
    )

def print_runtime_config() -> None:
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
