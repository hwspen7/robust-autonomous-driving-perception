from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import math
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


EVAL = Path(
    "/root/rivermind-data/autodrive/results/evaluation"
)
FORMAL = EVAL / "formal_training"
PREP = FORMAL / "prepared_inputs_v1"
TRAIN_ROOT = FORMAL / "training_v2"

PROTOCOL = PREP / "formal_training_protocol.json"
PREP_MANIFEST = PREP / "artifact_manifest.json"

COMMON_RUN = (
    FORMAL / "training_v1/common_uniform_e1_40"
)
COMMON = COMMON_RUN / "weights/common40_ema.pt"
COMMON_COMPLETION = (
    COMMON_RUN / "common40_completion.json"
)
COMMON_MANIFEST = (
    COMMON_RUN / "artifact_manifest.json"
)

STATE = TRAIN_ROOT / "controller_state.json"
FINAL_MANIFEST = (
    TRAIN_ROOT / "formal_branches_manifest.json"
)
LOCK = TRAIN_ROOT / ".controller.lock"

EXPECTED = {
    PROTOCOL: (
        "c40ad7567ebddd44d96390aa5e3c0340"
        "a19cbe6f84f83c944d0442ae69205a14"
    ),
    PREP_MANIFEST: (
        "ccf01cc4092237665dc4d73a587bd2a"
        "4f9a454cf7fceb0b4f17712ab8c6854aa"
    ),
    COMMON: (
        "c05ed70a608d026fef6178f40a4dee62"
        "e96323731700b42e426c311dd78b27f3"
    ),
    COMMON_MANIFEST: (
        "5fff0d57aa336cf36074cf8da3aff1b3"
        "c0d58a169f5bc924d5ebaf978a6f77b7"
    ),
}

BRANCHES = [
    {
        "key": "uniform",
        "stage": "uniform_e41_100",
        "yaml": PREP / "uniform_full70k.yaml",
        "list": PREP / "uniform_full70k_train.txt",
        "run": TRAIN_ROOT / "uniform_e41_100",
        "canonical": "uniform100_ema.pt",
        "ram_views": 0,
    },
    {
        "key": "rebu_risk",
        "stage": "rebu_risk_e41_100",
        "yaml": PREP / "rebu_risk_full70k.yaml",
        "list": PREP / "rebu_risk_full70k_train.txt",
        "run": TRAIN_ROOT / "rebu_risk_e41_100",
        "canonical": "rebu_yolo100_ema.pt",
        "ram_views": 19_500,
    },
]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        path.name + ".incomplete"
    )
    temporary.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ) + "\n"
    )
    os.replace(temporary, path)


def verify(path, expected):
    if not path.is_file():
        raise FileNotFoundError(path)

    actual = sha256(path)

    print(path)
    print(" expected:", expected)
    print(" actual  :", actual)

    if actual != expected:
        raise AssertionError(path)


def verify_manifest(path, root):
    data = json.loads(path.read_text())

    if isinstance(data.get("files"), dict):
        records = data["files"].items()
    elif isinstance(data.get("artifacts"), list):
        records = [
            (record["path"], record)
            for record in data["artifacts"]
        ]
    else:
        raise KeyError(path)

    for relative, record in records:
        artifact = Path(relative)

        if not artifact.is_absolute():
            artifact = root / artifact

        verify(artifact, record["sha256"])

        if "bytes" in record:
            assert (
                artifact.stat().st_size
                == record["bytes"]
            )


def check_model(path):
    model = YOLO(str(path))

    strides = [
        int(value)
        for value in
        model.model.stride.detach().cpu().tolist()
    ]

    names = model.model.names
    parameters = sum(
        parameter.numel()
        for parameter in model.model.parameters()
    )

    print("checkpoint:", path)
    print("checkpoint_sha256:", sha256(path))
    print("strides:", strides)
    print("classes:", names)
    print("parameters:", parameters)

    assert strides == [4, 8, 16, 32]
    assert len(names) == 10

    del model
    gc.collect()


def resolve_train_source(data_yaml):
    data = yaml.safe_load(data_yaml.read_text())
    source = Path(data["train"])

    if not source.is_absolute():
        root = Path(
            data.get("path", data_yaml.parent)
        )

        if not root.is_absolute():
            root = data_yaml.parent / root

        source = root / source

    return source.resolve(), data


def check_branch_input(branch):
    source, data = resolve_train_source(
        branch["yaml"]
    )

    assert source == branch["list"].resolve()

    names = data["names"]
    assert len(names) == 10

    rows = [
        row.strip()
        for row in branch["list"].read_text().splitlines()
        if row.strip()
    ]

    assert len(rows) == 70_000
    assert len(set(rows)) == 70_000

    original_root = (
        "/root/rivermind-data/autodrive/datasets/"
        "datasets/bdd100k_final/images/train/"
    )

    object_view_root = (
        "/dev/shm/rebu_yolo/method_gate_v1/"
        "scalar_risk/images/"
    )

    original_views = 0
    object_views = 0
    resolved_targets = []
    missing = []
    unexpected = []

    for row in rows:
        path = Path(row)

        if not path.is_file():
            if len(missing) < 20:
                missing.append(row)
            continue

        target = str(path.resolve())
        resolved_targets.append(target)

        if target.startswith(original_root):
            original_views += 1

        elif target.startswith(object_view_root):
            object_views += 1

        else:
            if len(unexpected) < 20:
                unexpected.append({
                    "listed": row,
                    "resolved": target,
                })

    if missing:
        raise FileNotFoundError(
            "Missing training views:\n"
            + "\n".join(missing)
        )

    if unexpected:
        raise AssertionError(
            "Unexpected resolved training sources:\n"
            + "\n".join(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                for item in unexpected
            )
        )

    expected_object_views = branch["ram_views"]
    expected_original_views = (
        70_000 - expected_object_views
    )

    assert object_views == expected_object_views, (
        branch["stage"],
        "object_views",
        expected_object_views,
        object_views,
    )

    assert original_views == expected_original_views, (
        branch["stage"],
        "original_views",
        expected_original_views,
        original_views,
    )

    assert len(resolved_targets) == 70_000

    assert len(set(resolved_targets)) == 70_000, (
        branch["stage"],
        "duplicate_resolved_sources",
        70_000 - len(set(resolved_targets)),
    )

    print("stage:", branch["stage"])
    print("views:", len(rows))
    print("original_views:", original_views)
    print("object_centric_views:", object_views)
    print(
        "unique_resolved_sources:",
        len(set(resolved_targets)),
    )
    print("yaml_sha256:", sha256(branch["yaml"]))
    print("list_sha256:", sha256(branch["list"]))


def preflight():
    print("=== FORMAL BRANCH V2 PREFLIGHT ===")

    for path, expected in EXPECTED.items():
        verify(path, expected)

    verify_manifest(PREP_MANIFEST, PREP)
    verify_manifest(COMMON_MANIFEST, COMMON_RUN)

    completion = json.loads(
        COMMON_COMPLETION.read_text()
    )

    assert completion["status"] == "complete"
    assert completion["human_epochs"] == 40
    assert (
        completion["common40_ema_sha256"]
        == EXPECTED[COMMON]
    )

    check_model(COMMON)

    for branch in BRANCHES:
        print("\n--- INPUT CHECK ---")
        check_branch_input(branch)

    assert torch.cuda.is_available()

    print("gpu:", torch.cuda.get_device_name(0))
    print(
        "gpu_memory_GiB:",
        torch.cuda.get_device_properties(
            0
        ).total_memory / 1024 ** 3,
    )

    print(
        "PASS: Common-40 and both formal "
        "70k branches are valid"
    )


def paths(branch):
    run = branch["run"]
    weights = run / "weights"

    return {
        "run": run,
        "last": weights / "last.pt",
        "best": weights / "best.pt",
        "canonical": (
            weights / branch["canonical"]
        ),
        "results": run / "results.csv",
        "args": run / "args.yaml",
        "completion": (
            run / "branch_completion.json"
        ),
        "manifest": (
            run / "artifact_manifest.json"
        ),
    }


def update_state(status, stage, **extra):
    write_json(
        STATE,
        {
            "version": (
                "formal_branches_controller_v2"
            ),
            "status": status,
            "stage": stage,
            "updated_utc": now(),
            **extra,
        },
    )


def load_results(path):
    with path.open(newline="") as handle:
        raw_rows = list(csv.DictReader(handle))

    return [
        {
            key.strip(): value.strip()
            for key, value in row.items()
        }
        for row in raw_rows
    ]


def audit_terminal_validation(results_path):
    rows = load_results(results_path)
    assert len(rows) == 60

    metric_keys = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
    ]

    meaningful_epochs = []

    for row in rows:
        values = []

        for key in metric_keys:
            value = row.get(key, "")

            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 0.0

            assert math.isfinite(value)
            values.append(value)

        if any(abs(value) > 1e-12 for value in values):
            meaningful_epochs.append(
                int(float(row["epoch"]))
            )

    if meaningful_epochs != [60]:
        raise AssertionError(
            "Expected terminal-only validation at "
            f"epoch 60, found {meaningful_epochs}"
        )

    last = rows[-1]

    metrics = {
        key: float(last[key])
        for key in metric_keys
    }

    return metrics


def audit_args(branch, args_path):
    data = yaml.safe_load(args_path.read_text())

    expected = {
        "epochs": 60,
        "imgsz": 960,
        "batch": 8,
        "nbs": 64,
        "workers": 8,
        "cache": False,
        "optimizer": "SGD",
        "lr0": 0.001,
        "lrf": 0.05,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 1.0,
        "cos_lr": True,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 0,
        "amp": True,
        "deterministic": True,
        "seed": 20260809,
        "patience": 0,
        "val": False,
        "save_period": 5,
        "fraction": 1.0,
    }

    for key, expected_value in expected.items():
        actual = data.get(key)

        if actual != expected_value:
            raise AssertionError(
                (key, expected_value, actual)
            )

    assert (
        Path(data["data"]).resolve()
        == branch["yaml"].resolve()
    )


def train_new(branch):
    current = paths(branch)

    # Recheck all 70k inputs immediately before each branch.
    # This is especially important for the Rebu views stored in /dev/shm.
    check_branch_input(branch)

    if current["run"].exists():
        raise FileExistsError(current["run"])

    verify(COMMON, EXPECTED[COMMON])

    update_state(
        "running",
        branch["stage"],
        completed_branch_epochs=0,
        common_checkpoint=str(COMMON),
        common_checkpoint_sha256=sha256(COMMON),
    )

    gc.collect()
    torch.cuda.empty_cache()

    model = YOLO(str(COMMON))

    model.train(
        data=str(branch["yaml"]),
        epochs=60,
        imgsz=960,
        batch=8,
        nbs=64,
        device=0,
        workers=8,
        cache=False,

        optimizer="SGD",
        lr0=0.001,
        lrf=0.05,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=1.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
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
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        close_mosaic=0,

        amp=True,
        deterministic=True,
        seed=20260809,
        patience=0,
        val=False,
        plots=False,
        save=True,
        save_period=5,
        fraction=1.0,

        project=str(TRAIN_ROOT),
        name=current["run"].name,
        exist_ok=False,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()


def resume_branch(branch):
    current = paths(branch)

    if not current["last"].is_file():
        raise FileNotFoundError(
            current["last"]
        )

    rows = (
        load_results(current["results"])
        if current["results"].is_file()
        else []
    )

    completed = len(rows)

    print("resume_stage:", branch["stage"])
    print("completed_branch_epochs:", completed)
    print(
        "resume_checkpoint_sha256:",
        sha256(current["last"]),
    )

    if completed >= 60:
        return

    checkpoint = torch.load(
        current["last"],
        map_location="cpu",
    )

    if checkpoint.get("optimizer") is None:
        raise RuntimeError(
            "Incomplete checkpoint has no "
            "optimizer state."
        )

    del checkpoint
    gc.collect()

    update_state(
        "resuming",
        branch["stage"],
        completed_branch_epochs=completed,
        resume_checkpoint=str(current["last"]),
        resume_checkpoint_sha256=sha256(
            current["last"]
        ),
    )

    torch.cuda.empty_cache()

    model = YOLO(str(current["last"]))

    model.train(
        resume=str(current["last"]),
        device=0,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()


def finalize_branch(branch):
    current = paths(branch)

    for required in [
        current["last"],
        current["results"],
        current["args"],
    ]:
        if not required.is_file():
            raise FileNotFoundError(required)

    audit_args(branch, current["args"])

    terminal_metrics = audit_terminal_validation(
        current["results"]
    )

    check_model(current["last"])

    temporary = current["canonical"].with_name(
        current["canonical"].name
        + ".incomplete"
    )

    shutil.copy2(
        current["last"],
        temporary,
    )
    os.replace(
        temporary,
        current["canonical"],
    )

    last_hash = sha256(current["last"])
    canonical_hash = sha256(
        current["canonical"]
    )

    assert last_hash == canonical_hash

    record = {
        "version": "formal_branch_v2",
        "status": "complete",
        "completed_utc": now(),
        "branch": branch["key"],
        "stage": branch["stage"],
        "common_epochs": 40,
        "branch_epochs": 60,
        "global_epoch": 100,
        "views_per_epoch": 70_000,
        "common_checkpoint": str(COMMON),
        "common_checkpoint_sha256": sha256(
            COMMON
        ),
        "optimizer_initialization": "fresh_SGD",
        "validation_during_epochs_1_59": False,
        "terminal_ultralytics_validation": True,
        "official_val_used_for_gradient": False,
        "official_val_used_for_early_stopping": False,
        "checkpoint_selection": (
            "fixed terminal last.pt; "
            "best.pt is not the primary model"
        ),
        "canonical_cocoeval_pending": True,
        "terminal_ultralytics_metrics": (
            terminal_metrics
        ),
        "data_yaml": str(branch["yaml"]),
        "data_yaml_sha256": sha256(
            branch["yaml"]
        ),
        "train_list": str(branch["list"]),
        "train_list_sha256": sha256(
            branch["list"]
        ),
        "canonical_checkpoint": str(
            current["canonical"]
        ),
        "canonical_checkpoint_sha256": (
            canonical_hash
        ),
    }

    if current["best"].is_file():
        record["automatic_best_checkpoint"] = str(
            current["best"]
        )
        record[
            "automatic_best_checkpoint_sha256"
        ] = sha256(current["best"])

    write_json(
        current["completion"],
        record,
    )

    artifacts = [
        current["last"],
        current["canonical"],
        current["results"],
        current["args"],
        current["completion"],
    ]

    if current["best"].is_file():
        artifacts.append(current["best"])

    files = {}

    for artifact in artifacts:
        files[
            str(artifact.relative_to(current["run"]))
        ] = {
            "bytes": artifact.stat().st_size,
            "sha256": sha256(artifact),
        }

    write_json(
        current["manifest"],
        {
            "version": (
                "formal_branch_artifacts_v2"
            ),
            "created_utc": now(),
            "branch": branch["key"],
            "files": files,
        },
    )

    print("\n" + "=" * 100)
    print("FORMAL BRANCH COMPLETE")
    print("stage:", branch["stage"])
    print(
        "terminal_AP:",
        terminal_metrics[
            "metrics/mAP50-95(B)"
        ],
    )
    print(
        "terminal_AP50:",
        terminal_metrics[
            "metrics/mAP50(B)"
        ],
    )
    print("checkpoint:", current["canonical"])
    print("checkpoint_sha256:", canonical_hash)
    print(
        "manifest_sha256:",
        sha256(current["manifest"]),
    )


def branch_complete(branch):
    current = paths(branch)

    if not current["completion"].is_file():
        return False

    if not current["manifest"].is_file():
        raise FileNotFoundError(
            current["manifest"]
        )

    record = json.loads(
        current["completion"].read_text()
    )

    assert record["status"] == "complete"

    verify_manifest(
        current["manifest"],
        current["run"],
    )

    return True


def execute_branch(branch, allow_resume):
    if branch_complete(branch):
        print(
            "PASS: already complete:",
            branch["stage"],
        )
        return

    current = paths(branch)

    if not current["run"].exists():
        train_new(branch)
    else:
        if not allow_resume:
            raise RuntimeError(
                f"Incomplete run exists: "
                f"{current['run']}\n"
                "Use --execute --resume."
            )

        resume_branch(branch)

    finalize_branch(branch)


def finalize_all():
    branch_records = {}

    for branch in BRANCHES:
        assert branch_complete(branch)

        current = paths(branch)
        record = json.loads(
            current["completion"].read_text()
        )

        branch_records[branch["key"]] = {
            "stage": branch["stage"],
            "checkpoint": record[
                "canonical_checkpoint"
            ],
            "checkpoint_sha256": record[
                "canonical_checkpoint_sha256"
            ],
            "terminal_ultralytics_metrics": (
                record[
                    "terminal_ultralytics_metrics"
                ]
            ),
            "manifest": str(current["manifest"]),
            "manifest_sha256": sha256(
                current["manifest"]
            ),
        }

    write_json(
        FINAL_MANIFEST,
        {
            "version": (
                "formal_60_plus_60_complete_v2"
            ),
            "status": "complete",
            "created_utc": now(),
            "protocol_sha256": sha256(PROTOCOL),
            "common_checkpoint_sha256": sha256(
                COMMON
            ),
            "branches": branch_records,
            "primary_checkpoint_rule": (
                "Use fixed terminal checkpoints."
            ),
            "canonical_cocoeval_pending": True,
            "next_stage": (
                "Canonical clean, failure-subset "
                "and corruption evaluation."
            ),
        },
    )

    update_state(
        "complete",
        "formal_branches_complete",
        final_manifest=str(FINAL_MANIFEST),
        final_manifest_sha256=sha256(
            FINAL_MANIFEST
        ),
    )

    print("\n" + "=" * 100)
    print("REBU-YOLO FORMAL TRAINING COMPLETE")
    print("=" * 100)

    for key, record in branch_records.items():
        print(
            key,
            "AP=",
            record[
                "terminal_ultralytics_metrics"
            ]["metrics/mAP50-95(B)"],
            "checkpoint_sha256=",
            record["checkpoint_sha256"],
        )

    print("manifest:", FINAL_MANIFEST)
    print(
        "manifest_sha256:",
        sha256(FINAL_MANIFEST),
    )
    print(
        "NEXT: canonical final evaluation; "
        "do not extend training"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    if args.resume and not args.execute:
        raise ValueError(
            "--resume requires --execute"
        )

    preflight()

    if not args.execute:
        print("\nPASS: preflight only")
        print("NOTHING TRAINED OR MODIFIED")
        return

    TRAIN_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with LOCK.open("w") as handle:
        try:
            fcntl.flock(
                handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            raise RuntimeError(
                "Another formal controller "
                "is running."
            ) from error

        try:
            for branch in BRANCHES:
                execute_branch(
                    branch,
                    allow_resume=args.resume,
                )

            finalize_all()

        except BaseException as error:
            update_state(
                "failed",
                "formal_branches",
                error_type=type(error).__name__,
                error=str(error),
                traceback=traceback.format_exc(),
            )
            raise


if __name__ == "__main__":
    main()