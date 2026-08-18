from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "20260809")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import torch
import yaml
from ultralytics import YOLO


EVAL_ROOT = Path(
    "/root/rivermind-data/autodrive/results/evaluation"
)

PREPARED = (
    EVAL_ROOT
    / "formal_training/prepared_inputs_v1"
)

PROTOCOL = PREPARED / "formal_training_protocol.json"
PREPARED_MANIFEST = PREPARED / "artifact_manifest.json"
TRAIN_LIST = PREPARED / "uniform_full70k_train.txt"
DATA_YAML = PREPARED / "uniform_full70k.yaml"

INITIALIZATION = (
    EVAL_ROOT
    / "architecture_gate/prepared_inputs_v2/"
      "yolo11m_p2_projected_coco_init.pt"
)

TRAINING_ROOT = EVAL_ROOT / "formal_training/training_v1"
RUN_DIR = TRAINING_ROOT / "common_uniform_e1_40"

WEIGHTS_DIR = RUN_DIR / "weights"
LAST = WEIGHTS_DIR / "last.pt"
BEST = WEIGHTS_DIR / "best.pt"
COMMON40_EMA = WEIGHTS_DIR / "common40_ema.pt"

COMPLETION = RUN_DIR / "common40_completion.json"
OUTPUT_MANIFEST = RUN_DIR / "artifact_manifest.json"
STATE = TRAINING_ROOT / "common40_controller_state.json"
LOCK = TRAINING_ROOT / ".common40_controller.lock"

EXPECTED_HASHES = {
    PROTOCOL: (
        "c40ad7567ebddd44d96390aa5e3c0340"
        "a19cbe6f84f83c944d0442ae69205a14"
    ),
    PREPARED_MANIFEST: (
        "ccf01cc4092237665dc4d73a587bd2a"
        "4f9a454cf7fceb0b4f17712ab8c6854aa"
    ),
    INITIALIZATION: (
        "eef43a83adb99ee9b3d9e75c25d345448"
        "8b28ad5f86ac67fcf850351c3e848dd"
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    )
    os.replace(temporary, path)


def verify_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)

    actual = sha256(path)

    print(path)
    print(" expected:", expected)
    print(" actual  :", actual)

    if actual != expected:
        raise AssertionError(path)


def verify_prepared_manifest() -> None:
    data = json.loads(PREPARED_MANIFEST.read_text())

    records = []

    if isinstance(data.get("files"), dict):
        for relative, record in data["files"].items():
            records.append((relative, record))

    elif isinstance(data.get("artifacts"), list):
        for record in data["artifacts"]:
            records.append((record["path"], record))

    else:
        raise KeyError(
            "Prepared manifest has neither files nor artifacts."
        )

    for relative, record in records:
        path = Path(relative)

        if not path.is_absolute():
            path = PREPARED / path

        verify_hash(path, record["sha256"])


def read_train_list() -> list[str]:
    rows = [
        row.strip()
        for row in TRAIN_LIST.read_text().splitlines()
        if row.strip()
    ]

    if len(rows) != 70_000:
        raise AssertionError(
            f"Expected 70000 views, found {len(rows)}"
        )

    if len(set(rows)) != 70_000:
        raise AssertionError("Uniform 70k list contains duplicates.")

    missing = []

    for row in rows:
        if not Path(row).is_file():
            missing.append(row)

            if len(missing) >= 20:
                break

    if missing:
        raise FileNotFoundError(
            "Missing training inputs:\n" + "\n".join(missing)
        )

    return rows


def completed_epochs() -> int:
    results_csv = RUN_DIR / "results.csv"

    if not results_csv.is_file():
        return 0

    with results_csv.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def verify_model_architecture(checkpoint: Path) -> None:
    model = YOLO(str(checkpoint))

    strides = [
        int(value)
        for value in model.model.stride.detach().cpu().tolist()
    ]

    print("model_strides:", strides)
    print("parameters:", sum(
        parameter.numel()
        for parameter in model.model.parameters()
    ))

    if strides != [4, 8, 16, 32]:
        raise AssertionError(strides)


def preflight() -> dict:
    print("=== COMMON-40 FORMAL PREFLIGHT ===")

    for path, expected in EXPECTED_HASHES.items():
        verify_hash(path, expected)

    verify_prepared_manifest()

    protocol = json.loads(PROTOCOL.read_text())
    data_config = yaml.safe_load(DATA_YAML.read_text())

    configured_train = Path(data_config["train"]).resolve()

    if configured_train != TRAIN_LIST.resolve():
        raise AssertionError(
            f"Unexpected train list: {configured_train}"
        )

    names = data_config.get("names")

    if isinstance(names, dict):
        number_of_classes = len(names)
    elif isinstance(names, list):
        number_of_classes = len(names)
    else:
        raise TypeError("Invalid dataset class names.")

    if number_of_classes != 10:
        raise AssertionError(number_of_classes)

    rows = read_train_list()

    verify_model_architecture(INITIALIZATION)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    gpu_properties = torch.cuda.get_device_properties(0)

    print("uniform_views:", len(rows))
    print("classes:", number_of_classes)
    print("gpu:", torch.cuda.get_device_name(0))
    print(
        "gpu_memory_GiB:",
        gpu_properties.total_memory / 1024 ** 3,
    )
    print("protocol_sha256:", sha256(PROTOCOL))
    print("prepared_manifest_sha256:", sha256(PREPARED_MANIFEST))
    print("initialization_sha256:", sha256(INITIALIZATION))

    print("PASS: frozen Common-40 inputs are valid")

    return protocol


def finalize() -> None:
    if not LAST.is_file():
        raise FileNotFoundError(LAST)

    verify_model_architecture(LAST)

    temporary = COMMON40_EMA.with_name(
        COMMON40_EMA.name + ".incomplete"
    )

    shutil.copy2(LAST, temporary)
    os.replace(temporary, COMMON40_EMA)

    last_hash = sha256(LAST)
    common_hash = sha256(COMMON40_EMA)

    if last_hash != common_hash:
        raise AssertionError(
            "Canonical Common-40 checkpoint copy mismatch."
        )

    record = {
        "version": "rebu_yolo_common40_v1",
        "status": "complete",
        "completed_utc": utc_now(),
        "stage": "common_uniform_e1_40",
        "human_epochs": 40,
        "training_views_per_epoch": 70_000,
        "official_validation_used": False,
        "initialization": str(INITIALIZATION),
        "initialization_sha256": sha256(INITIALIZATION),
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256(PROTOCOL),
        "last_checkpoint": str(LAST),
        "last_checkpoint_sha256": last_hash,
        "common40_ema": str(COMMON40_EMA),
        "common40_ema_sha256": common_hash,
        "next_stages": [
            "uniform_e41_100",
            "rebu_risk_e41_100",
        ],
        "branch_optimizer_rule": (
            "Both branches start from this exact EMA with "
            "fresh identical SGD optimizers."
        ),
    }

    if BEST.is_file():
        record["best_checkpoint"] = str(BEST)
        record["best_checkpoint_sha256"] = sha256(BEST)

    atomic_json(COMPLETION, record)

    artifact_paths = [
        LAST,
        COMMON40_EMA,
        COMPLETION,
        RUN_DIR / "args.yaml",
        RUN_DIR / "results.csv",
    ]

    if BEST.is_file():
        artifact_paths.append(BEST)

    files = {}

    for path in artifact_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

        files[str(path.relative_to(RUN_DIR))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    atomic_json(
        OUTPUT_MANIFEST,
        {
            "version": "rebu_yolo_common40_artifacts_v1",
            "created_utc": utc_now(),
            "files": files,
        },
    )

    atomic_json(
        STATE,
        {
            "status": "complete",
            "stage": "common_uniform_e1_40",
            "completed_epochs": 40,
            "checkpoint": str(COMMON40_EMA),
            "checkpoint_sha256": common_hash,
            "artifact_manifest": str(OUTPUT_MANIFEST),
            "artifact_manifest_sha256": sha256(OUTPUT_MANIFEST),
        },
    )

    print("\n" + "=" * 100)
    print("REBU-YOLO COMMON-40 TRAINING COMPLETE")
    print("=" * 100)
    print("checkpoint:", COMMON40_EMA)
    print("checkpoint_sha256:", common_hash)
    print("completion:", COMPLETION)
    print("manifest:", OUTPUT_MANIFEST)
    print("manifest_sha256:", sha256(OUTPUT_MANIFEST))
    print(
        "NEXT: fork this exact EMA into the two "
        "fresh-SGD 60-epoch formal branches"
    )


def start_new_training() -> None:
    if RUN_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing run: {RUN_DIR}\n"
            "Use --execute --resume only if it is incomplete."
        )

    atomic_json(
        STATE,
        {
            "status": "running",
            "stage": "common_uniform_e1_40",
            "started_utc": utc_now(),
            "completed_epochs": 0,
        },
    )

    torch.cuda.empty_cache()

    model = YOLO(str(INITIALIZATION))

    model.train(
        data=str(DATA_YAML),
        epochs=40,
        imgsz=960,
        batch=8,
        nbs=64,
        device=0,
        workers=8,
        cache=False,

        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,

        box=7.5,
        cls=0.5,
        dfl=1.5,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        close_mosaic=10,

        amp=True,
        deterministic=True,
        seed=20260809,
        patience=0,
        val=False,
        plots=False,
        save=True,
        save_period=5,

        project=str(TRAINING_ROOT),
        name=RUN_DIR.name,
        exist_ok=False,
    )


def resume_training() -> None:
    if COMPLETION.is_file():
        print("PASS: Common-40 is already complete.")
        print(COMPLETION)
        return

    if not LAST.is_file():
        raise FileNotFoundError(
            f"No resumable checkpoint: {LAST}"
        )

    epochs = completed_epochs()

    print("completed_epoch_rows:", epochs)
    print("resume_checkpoint:", LAST)
    print("resume_checkpoint_sha256:", sha256(LAST))

    if epochs >= 40:
        print("Training reached 40 epochs; finalizing artifacts.")
        finalize()
        return

    checkpoint = torch.load(LAST, map_location="cpu")

    if checkpoint.get("optimizer") is None:
        raise RuntimeError(
            "Incomplete run checkpoint has no optimizer state."
        )

    atomic_json(
        STATE,
        {
            "status": "resuming",
            "stage": "common_uniform_e1_40",
            "resumed_utc": utc_now(),
            "completed_epoch_rows": epochs,
            "resume_checkpoint": str(LAST),
            "resume_checkpoint_sha256": sha256(LAST),
        },
    )

    torch.cuda.empty_cache()

    model = YOLO(str(LAST))
    model.train(
        resume=str(LAST),
        device=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute formal Common-40 training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted formal run.",
    )
    args = parser.parse_args()

    if args.resume and not args.execute:
        raise ValueError("--resume requires --execute")

    preflight()

    if not args.execute:
        print("\nPASS: preflight only")
        print("NOTHING TRAINED OR MODIFIED")
        return

    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)

    with LOCK.open("w") as lock_handle:
        try:
            fcntl.flock(
                lock_handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            raise RuntimeError(
                "Another Common-40 controller is running."
            ) from error

        try:
            if args.resume:
                resume_training()
            else:
                start_new_training()

            if not COMPLETION.is_file():
                finalize()

        except Exception as error:
            atomic_json(
                STATE,
                {
                    "status": "failed",
                    "stage": "common_uniform_e1_40",
                    "failed_utc": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "completed_epoch_rows": completed_epochs(),
                },
            )
            raise


if __name__ == "__main__":
    main()