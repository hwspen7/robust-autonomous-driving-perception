from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]



DEFAULT_DFINE_ROOT = PROJECT_ROOT / "source-code" / "third-party" / "D-FINE"
DEFAULT_CONFIG = (
    DEFAULT_DFINE_ROOT
    / "configs"
    / "dfine"
    / "dfine_hgnetv2_m_bdd100k.yml"
)


DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "dfine_m_coco.pth"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start official D-FINE-M baseline training on BDD100K"
    )
    parser.add_argument(
        "--dfine-root",
        type=Path,
        default=DEFAULT_DFINE_ROOT,
        help="Path to the official D-FINE repository",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="BDD100K D-FINE-M training configuration",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="D-FINE-M COCO pretrained checkpoint",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA_VISIBLE_DEVICES value, for example 0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable official AMP training; AMP is enabled by default",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the final command without starting training",
    )
    return parser


def validate_paths(
    dfine_root: Path,
    config: Path,
    checkpoint: Path,
    dry_run: bool,
) -> None:
    required = {
        "D-FINE repo": dfine_root,
        "train.py": dfine_root / "train.py",
        "config": config,
        "checkpoint": checkpoint,
    }

    missing = [
        f"{name}: {path}"
        for name, path in required.items()
        if not path.exists()
    ]

    if not missing:
        return

    message = "The following documents do not yet exist: \n" + "\n".join(
        f"  - {item}" for item in missing
    )



    if dry_run:
        print(f"[DRY-RUN WARNING]\n{message}")
        return

    raise FileNotFoundError(message)


def main() -> None:
    args = build_parser().parse_args()

    dfine_root = args.dfine_root.expanduser().resolve()
    config = args.config.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()

    validate_paths(
        dfine_root=dfine_root,
        config=config,
        checkpoint=checkpoint,
        dry_run=args.dry_run,
    )



    command = [
        sys.executable,
        "train.py",
        "-c",
        str(config),
        f"--seed={args.seed}",
        "-t",
        str(checkpoint),
    ]

    if not args.no_amp:
        command.append("--use-amp")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.device)
    env["OMP_NUM_THREADS"] = "8"
    env["MKL_NUM_THREADS"] = "8"
    env["PYTHONUNBUFFERED"] = "1"

    print("===== D-FINE-M BDD100K Baseline =====")
    print(f"D-FINE repo : {dfine_root}")
    print(f"Config      : {config}")
    print(f"Checkpoint  : {checkpoint}")
    print(f"Device      : cuda:{args.device}")
    print(f"Seed        : {args.seed}")
    print(f"AMP         : {not args.no_amp}")
    print()
    print(
        "Command:",
        " ".join(shlex.quote(item) for item in command),
    )

    if args.dry_run:
        print("\nDry-run only. Training was not started.")
        return

    subprocess.run(
        command,
        cwd=dfine_root,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()