from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import time

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
import ultralytics


REPO = Path(
    "/root/rivermind-data/autodrive/code/"
    "robust-autonomous-driving-perception"
)
ARCH_ROOT = Path(
    "/root/rivermind-data/autodrive/results/evaluation/"
    "architecture_gate"
)
V3 = ARCH_ROOT / "prepared_inputs_v3"
V4 = ARCH_ROOT / "prepared_inputs_v4"
FORMAL_ROOT = ARCH_ROOT / "formal_gate_v4"

V4_MANIFEST = V4 / "artifact_manifest.json"
PROTOCOL_PATH = V4 / "formal_gate_protocol_v4.json"
AMENDMENT_PATH = V4 / "determinism_amendment.json"

# V4 intentionally references the already-frozen V3 data artifacts.
DATA_YAML = V3 / "bdd100k_architecture_gate_v3.yaml"
EFFECTIVE_DEV_COCO = V3 / "instances_train_dev_effective.json"

EXPECTED_FROZEN_HASHES = {
    V4_MANIFEST:
        "809f1266bb7edca4575ae1e5009e8762a96529322a516e81399599311584d4d9",
    PROTOCOL_PATH:
        "7f003f25d4ddb5828933c9fbdb85441bf86fe20e2824495f369baca65765525a",
    AMENDMENT_PATH:
        "323977b86c1a731bcf2c7a92436b8db4b7593ec9c0c292f2477d2d31fd1e3900",
    DATA_YAML:
        "208a57dacf467c4efab4d6d7d687fce63938d24bc4fc2012d551bb49100e808c",
    EFFECTIVE_DEV_COCO:
        "5e13ec651d0582032e5eacc1a46d885279fabe278463bd7c2be58b2429796faf",
}

MODEL_KEYS = ("standard", "p2_projected")
HUMAN_STAGE_EPOCH = 10
FINAL_EPOCH = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)

    actual = sha256(path)
    print(path)
    print(" expected:", expected)
    print(" actual  :", actual)

    if actual != expected:
        raise AssertionError(
            f"SHA256 mismatch:\n{path}\n"
            f"expected={expected}\nactual={actual}"
        )


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.incomplete-{os.getpid()}"
    )

    if temporary.exists():
        raise FileExistsError(temporary)

    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"

    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temporary, path)


def write_compact_json(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_copy(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(target)

    temporary = target.with_name(
        f"{target.name}.incomplete-{os.getpid()}"
    )
    if temporary.exists():
        raise FileExistsError(temporary)

    with source.open("rb") as src, temporary.open("xb") as dst:
        shutil.copyfileobj(
            src,
            dst,
            length=8 * 1024 * 1024,
        )
        dst.flush()
        os.fsync(dst.fileno())

    os.replace(temporary, target)


def torch_load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_resume_checkpoint(
    path: Path,
    expected_human_epoch: int,
    expected_data_yaml: Path,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)

    checkpoint = torch_load_checkpoint(path)
    expected_zero_based = expected_human_epoch - 1

    assert checkpoint["epoch"] == expected_zero_based, (
        checkpoint["epoch"],
        expected_zero_based,
    )
    assert checkpoint.get("ema") is not None
    assert isinstance(checkpoint.get("optimizer"), dict)
    assert isinstance(checkpoint.get("scaler"), dict)
    assert checkpoint.get("updates") is not None

    train_args = checkpoint.get("train_args")
    assert isinstance(train_args, dict)
    assert int(train_args["epochs"]) == FINAL_EPOCH
    assert int(train_args["imgsz"]) == 960
    assert int(train_args["batch"]) == 8
    assert int(train_args["nbs"]) == 64
    assert int(train_args["close_mosaic"]) == 10
    assert int(train_args["seed"]) == 20260809
    assert bool(train_args["deterministic"]) is True

    actual_data = Path(str(train_args["data"])).resolve()
    assert actual_data == expected_data_yaml.resolve(), (
        actual_data,
        expected_data_yaml,
    )

    result = {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "checkpoint_epoch_zero_based":
            int(checkpoint["epoch"]),
        "human_epoch": expected_human_epoch,
        "optimizer_present": True,
        "ema_present": True,
        "scaler_present": True,
        "planned_epochs":
            int(train_args["epochs"]),
    }

    del checkpoint
    gc.collect()

    return result


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for row in handle if row.strip())


def read_nonempty_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            row.strip()
            for row in handle
            if row.strip()
        ]


def release_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def preflight() -> dict:
    print("=== FORMAL GATE V4 PREFLIGHT ===")

    for path, expected in EXPECTED_FROZEN_HASHES.items():
        verify_hash(path, expected)

    manifest = json.loads(V4_MANIFEST.read_text())
    protocol = json.loads(PROTOCOL_PATH.read_text())

    assert manifest["version"] == "prepared_inputs_v4"
    assert manifest["status"] == "frozen"
    assert protocol["version"] == (
        "architecture_gate_protocol_v4"
    )
    assert protocol["status"] == "frozen_before_training"

    for filename, record in manifest["files"].items():
        path = V4 / filename
        verify_hash(path, record["sha256"])
        assert path.stat().st_size == int(record["bytes"])

    assert protocol["training"]["planned_schedule_epochs"] == 20
    assert protocol["training"]["checkpoint_epochs"] == [10, 20]
    assert protocol["training"]["physical_batch"] == 8
    assert protocol["training"]["nominal_batch"] == 64
    assert protocol["training"]["image_size"] == 960
    assert protocol["training"]["close_mosaic"] == 10
    assert protocol["training"]["cache"] is False
    assert protocol["training"]["deterministic"] is True
    assert (
        protocol["amendment"]["record_sha256"]
        == EXPECTED_FROZEN_HASHES[AMENDMENT_PATH]
    )

    assert (
        protocol["state_machine"]["epoch_10"]
        ["p2_can_be_accepted"]
        is False
    )
    assert (
        protocol["state_machine"]["epoch_20"]
        ["extension_to_epoch_30_allowed"]
        is False
    )
    assert (
        protocol["data"]["train_dev_effective_objects"]
        == 92392
    )
    assert (
        protocol["data"]["official_validation_used_for_gate"]
        is False
    )

    train_core = Path(
        protocol["data"]["train_core_list"]
    )
    train_dev = Path(
        protocol["data"]["train_dev_list"]
    )
    assert count_nonempty_lines(train_core) == 65000
    assert count_nonempty_lines(train_dev) == 5000
    verify_hash(
        train_core,
        protocol["data"]["train_core_list_sha256"],
    )
    verify_hash(
        train_dev,
        protocol["data"]["train_dev_list_sha256"],
    )

    for model_key in MODEL_KEYS:
        model_record = protocol["architectures"][model_key]
        verify_hash(
            Path(model_record["config"]),
            model_record["config_sha256"],
        )
        verify_hash(
            Path(model_record["initialization"]),
            model_record["initialization_sha256"],
        )
        assert model_record["historical_bdd_best_used"] is False

    tasks_source = Path(
        protocol["source_state"]["ultralytics_tasks_source"]
    )
    verify_hash(
        tasks_source,
        protocol["source_state"]
        ["ultralytics_tasks_source_sha256"],
    )

    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= 1

    device_name = torch.cuda.get_device_name(0)
    total_memory = (
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3
    )
    assert total_memory >= 20.0

    coco = json.loads(EFFECTIVE_DEV_COCO.read_text())
    assert len(coco["images"]) == 5000
    assert len(coco["annotations"]) == 92392
    assert not any(
        int(row["id"]) == 1005741
        for row in coco["annotations"]
    )

    print("\nGPU:", device_name)
    print("GPU memory GiB:", total_memory)
    print("Ultralytics:", ultralytics.__version__)
    print("PyTorch:", torch.__version__)
    print("PASS: all formal-gate inputs and rules are frozen")

    return protocol


def build_training_arguments(
    protocol: dict,
    model_key: str,
) -> dict:
    training = protocol["training"]
    augmentation = training["augmentation"]

    return {
        "data": str(DATA_YAML),
        "epochs": int(
            training["planned_schedule_epochs"]
        ),
        "imgsz": int(training["image_size"]),
        "batch": int(training["physical_batch"]),
        "nbs": int(training["nominal_batch"]),
        "device": int(training["device"]),
        "workers": int(training["workers"]),
        "cache": training["cache"],
        "amp": bool(training["amp"]),
        "optimizer": training["optimizer"],
        "lr0": float(training["lr0"]),
        "lrf": float(training["lrf"]),
        "momentum": float(training["momentum"]),
        "weight_decay": float(
            training["weight_decay"]
        ),
        "warmup_epochs": float(
            training["warmup_epochs"]
        ),
        "cos_lr": bool(training["cos_lr"]),
        "close_mosaic": int(
            training["close_mosaic"]
        ),
        "seed": int(training["seed"]),
        "deterministic": bool(
            training["deterministic"]
        ),
        "save": True,
        "save_period": int(
            training["save_period"]
        ),
        "val": False,
        "plots": False,
        "patience": 0,
        "fraction": 1.0,
        "rect": False,
        "multi_scale": False,
        "compile": False,
        "verbose": True,
        "project": str(FORMAL_ROOT),
        "name": model_key,
        "exist_ok": False,
        "hsv_h": float(augmentation["hsv_h"]),
        "hsv_s": float(augmentation["hsv_s"]),
        "hsv_v": float(augmentation["hsv_v"]),
        "degrees": float(augmentation["degrees"]),
        "translate": float(augmentation["translate"]),
        "scale": float(augmentation["scale"]),
        "shear": float(augmentation["shear"]),
        "perspective": float(
            augmentation["perspective"]
        ),
        "flipud": float(augmentation["flipud"]),
        "fliplr": float(augmentation["fliplr"]),
        "mosaic": float(augmentation["mosaic"]),
        "mixup": float(augmentation["mixup"]),
        "cutmix": float(augmentation["cutmix"]),
        "copy_paste": float(
            augmentation["copy_paste"]
        ),
    }


def train_to_epoch_10(
    protocol: dict,
    model_key: str,
) -> dict:
    print("\n" + "=" * 100)
    print(f"TRAINING {model_key}: EPOCHS 1-10")
    print("=" * 100)

    initialization = Path(
        protocol["architectures"][model_key]
        ["initialization"]
    )
    run_dir = FORMAL_ROOT / model_key
    resume_checkpoint = (
        run_dir / "weights/resume_epoch10.pt"
    )

    if run_dir.exists():
        raise FileExistsError(run_dir)

    model = YOLO(str(initialization))
    callback_completed = {"value": False}

    def preserve_epoch_10(trainer) -> None:
        completed_epoch = int(trainer.epoch) + 1
        if completed_epoch != HUMAN_STAGE_EPOCH:
            return

        if callback_completed["value"]:
            raise RuntimeError(
                "Epoch-10 checkpoint callback ran twice"
            )

        expected_run_dir = run_dir.resolve()
        actual_run_dir = Path(
            trainer.save_dir
        ).resolve()
        assert actual_run_dir == expected_run_dir, (
            actual_run_dir,
            expected_run_dir,
        )

        atomic_copy(
            Path(trainer.last),
            resume_checkpoint,
        )
        checkpoint_record = validate_resume_checkpoint(
            resume_checkpoint,
            expected_human_epoch=HUMAN_STAGE_EPOCH,
            expected_data_yaml=DATA_YAML,
        )
        atomic_write_json(
            run_dir / "epoch10_checkpoint.json",
            checkpoint_record,
        )

        # Prevent final_eval from validating an arbitrary
        # internal best.pt after this deliberate gate pause.
        trainer.best = (
            trainer.wdir
            / "__formal_gate_final_eval_disabled__.pt"
        )
        trainer.stop = True
        callback_completed["value"] = True

        print("\nFORMAL GATE PAUSE")
        print("model:", model_key)
        print("completed_epoch:", completed_epoch)
        print(
            "resume_checkpoint:",
            resume_checkpoint,
        )
        print(
            "resume_sha256:",
            checkpoint_record["sha256"],
        )

    model.add_callback(
        "on_model_save",
        preserve_epoch_10,
    )

    arguments = build_training_arguments(
        protocol,
        model_key,
    )
    model.train(**arguments)

    if not callback_completed["value"]:
        raise RuntimeError(
            f"{model_key} did not create the epoch-10 "
            "resume checkpoint"
        )

    record = validate_resume_checkpoint(
        resume_checkpoint,
        expected_human_epoch=HUMAN_STAGE_EPOCH,
        expected_data_yaml=DATA_YAML,
    )

    del model
    release_gpu()

    return record


def resume_to_epoch_20(
    protocol: dict,
    model_key: str,
    epoch_10_checkpoint: Path,
) -> dict:
    print("\n" + "=" * 100)
    print(f"RESUMING {model_key}: EPOCHS 11-20")
    print("=" * 100)

    validate_resume_checkpoint(
        epoch_10_checkpoint,
        expected_human_epoch=HUMAN_STAGE_EPOCH,
        expected_data_yaml=DATA_YAML,
    )

    run_dir = FORMAL_ROOT / model_key
    epoch_20_checkpoint = (
        run_dir / "weights/resume_epoch20.pt"
    )
    if epoch_20_checkpoint.exists():
        raise FileExistsError(epoch_20_checkpoint)

    model = YOLO(str(epoch_10_checkpoint))
    callback_completed = {"value": False}

    def preserve_epoch_20(trainer) -> None:
        completed_epoch = int(trainer.epoch) + 1
        if completed_epoch != FINAL_EPOCH:
            return

        if callback_completed["value"]:
            raise RuntimeError(
                "Epoch-20 checkpoint callback ran twice"
            )

        atomic_copy(
            Path(trainer.last),
            epoch_20_checkpoint,
        )
        checkpoint_record = validate_resume_checkpoint(
            epoch_20_checkpoint,
            expected_human_epoch=FINAL_EPOCH,
            expected_data_yaml=DATA_YAML,
        )
        atomic_write_json(
            run_dir / "epoch20_checkpoint.json",
            checkpoint_record,
        )

        # A native final-epoch validation has already happened.
        # Disable the second final_eval(best.pt) validation.
        trainer.best = (
            trainer.wdir
            / "__formal_gate_second_final_eval_disabled__.pt"
        )
        callback_completed["value"] = True

        print("\nFORMAL GATE FINAL CHECKPOINT")
        print("model:", model_key)
        print("completed_epoch:", completed_epoch)
        print(
            "resume_checkpoint:",
            epoch_20_checkpoint,
        )
        print(
            "resume_sha256:",
            checkpoint_record["sha256"],
        )

    model.add_callback(
        "on_model_save",
        preserve_epoch_20,
    )

    model.train(
        resume=str(epoch_10_checkpoint),
        device=int(protocol["training"]["device"]),
        workers=int(protocol["training"]["workers"]),
        cache=protocol["training"]["cache"],
        save_period=int(
            protocol["training"]["save_period"]
        ),
        close_mosaic=int(
            protocol["training"]["close_mosaic"]
        ),
        val=False,
        plots=False,
    )

    if not callback_completed["value"]:
        raise RuntimeError(
            f"{model_key} did not create the epoch-20 "
            "checkpoint"
        )

    record = validate_resume_checkpoint(
        epoch_20_checkpoint,
        expected_human_epoch=FINAL_EPOCH,
        expected_data_yaml=DATA_YAML,
    )

    del model
    release_gpu()

    return record


def evaluate_checkpoint(
    protocol: dict,
    model_key: str,
    human_epoch: int,
    checkpoint_path: Path,
) -> dict:
    print("\n" + "=" * 100)
    print(
        f"EVALUATING {model_key} AT EPOCH "
        f"{human_epoch}"
    )
    print("=" * 100)

    validate_resume_checkpoint(
        checkpoint_path,
        expected_human_epoch=human_epoch,
        expected_data_yaml=DATA_YAML,
    )

    final_dir = (
        FORMAL_ROOT
        / "evaluations"
        / f"{model_key}_epoch{human_epoch}"
    )
    staging_dir = final_dir.with_name(
        f"{final_dir.name}.incomplete-{os.getpid()}"
    )

    if final_dir.exists() or staging_dir.exists():
        raise FileExistsError(
            f"Evaluation output already exists:\n"
            f"{final_dir}\n{staging_dir}"
        )

    staging_dir.mkdir(parents=True, exist_ok=False)

    predictions_path = (
        staging_dir / "predictions.json"
    )
    metrics_path = staging_dir / "metrics.json"
    manifest_path = (
        staging_dir / "artifact_manifest.json"
    )

    coco_data = json.loads(
        EFFECTIVE_DEV_COCO.read_text()
    )
    image_by_name = {}
    image_shape = {}

    for image in coco_data["images"]:
        name = Path(image["file_name"]).name
        if name in image_by_name:
            raise AssertionError(
                f"Duplicate image basename: {name}"
            )
        image_by_name[name] = int(image["id"])
        image_shape[int(image["id"])] = (
            int(image["height"]),
            int(image["width"]),
        )

    dev_list = Path(
        protocol["data"]["train_dev_list"]
    )
    image_paths = read_nonempty_lines(dev_list)

    assert len(image_paths) == 5000
    assert len(image_by_name) == 5000
    assert {
        Path(path).name for path in image_paths
    } == set(image_by_name)

    model = YOLO(str(checkpoint_path))
    predictions = []
    processed_image_ids = set()
    skipped_zero_area = 0

    started = time.time()

    results = model.predict(
        source=image_paths,
        stream=True,
        imgsz=int(
            protocol["evaluation"]
            .get("image_size", 960)
        ),
        batch=1,
        conf=float(
            protocol["evaluation"]
            ["confidence_threshold"]
        ),
        iou=float(
            protocol["evaluation"]
            ["nms_iou_threshold"]
        ),
        max_det=int(
            protocol["evaluation"]
            ["max_detections_per_image"]
        ),
        device=int(protocol["training"]["device"]),
        half=True,
        augment=False,
        agnostic_nms=False,
        save=False,
        verbose=False,
    )

    for index, result in enumerate(results, start=1):
        basename = Path(result.path).name
        if basename not in image_by_name:
            raise KeyError(basename)

        image_id = image_by_name[basename]
        if image_id in processed_image_ids:
            raise AssertionError(
                f"Image processed twice: {image_id}"
            )
        processed_image_ids.add(image_id)

        expected_height, expected_width = (
            image_shape[image_id]
        )
        actual_height, actual_width = (
            int(result.orig_shape[0]),
            int(result.orig_shape[1]),
        )
        assert (
            actual_height,
            actual_width,
        ) == (
            expected_height,
            expected_width,
        )

        boxes = result.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.detach().cpu().tolist()
            scores = (
                boxes.conf.detach().cpu().tolist()
            )
            classes = (
                boxes.cls.detach().cpu().tolist()
            )

            for coords, score, class_id_float in zip(
                xyxy,
                scores,
                classes,
            ):
                x1, y1, x2, y2 = map(float, coords)

                x1 = min(
                    max(x1, 0.0),
                    float(expected_width),
                )
                y1 = min(
                    max(y1, 0.0),
                    float(expected_height),
                )
                x2 = min(
                    max(x2, 0.0),
                    float(expected_width),
                )
                y2 = min(
                    max(y2, 0.0),
                    float(expected_height),
                )

                width = x2 - x1
                height = y2 - y1
                if width <= 0.0 or height <= 0.0:
                    skipped_zero_area += 1
                    continue

                class_id = int(class_id_float)
                assert 0 <= class_id < 10

                predictions.append({
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": [
                        x1,
                        y1,
                        width,
                        height,
                    ],
                    "score": float(score),
                })

        if index % 500 == 0:
            print(
                f"predicted={index}/5000 "
                f"detections={len(predictions)}"
            )

    assert len(processed_image_ids) == 5000
    assert processed_image_ids == set(
        image_by_name.values()
    )
    assert predictions

    write_compact_json(
        predictions_path,
        predictions,
    )

    print("\n=== COCO EVALUATION ===")
    coco_gt = COCO(str(EFFECTIVE_DEV_COCO))
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(
        coco_gt,
        coco_dt,
        "bbox",
    )
    evaluator.params.imgIds = sorted(
        coco_gt.getImgIds()
    )
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = [
        float(value)
        for value in evaluator.stats.tolist()
    ]

    metrics = {
        "model_key": model_key,
        "human_epoch": human_epoch,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256":
            sha256(checkpoint_path),
        "protocol_sha256":
            EXPECTED_FROZEN_HASHES[PROTOCOL_PATH],
        "effective_dev_coco_sha256":
            EXPECTED_FROZEN_HASHES[
                EFFECTIVE_DEV_COCO
            ],
        "images": 5000,
        "effective_gt_objects": 92392,
        "predictions": len(predictions),
        "images_with_predictions": len({
            int(row["image_id"])
            for row in predictions
        }),
        "skipped_zero_area_predictions":
            skipped_zero_area,
        "AP": stats[0],
        "AP50": stats[1],
        "AP75": stats[2],
        "AP_small": stats[3],
        "AP_medium": stats[4],
        "AP_large": stats[5],
        "AR1": stats[6],
        "AR10": stats[7],
        "AR100": stats[8],
        "elapsed_minutes":
            (time.time() - started) / 60.0,
        "prediction_file":
            str(final_dir / predictions_path.name),
    }

    for key in (
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
    ):
        value = metrics[key]
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0

    atomic_write_json(metrics_path, metrics)

    eval_manifest = {
        "version": "formal_gate_evaluation_v4",
        "files": {
            predictions_path.name: {
                "bytes":
                    predictions_path.stat().st_size,
                "sha256":
                    sha256(predictions_path),
            },
            metrics_path.name: {
                "bytes":
                    metrics_path.stat().st_size,
                "sha256":
                    sha256(metrics_path),
            },
        },
    }
    atomic_write_json(
        manifest_path,
        eval_manifest,
    )

    final_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    os.replace(staging_dir, final_dir)

    metrics["prediction_sha256"] = (
        eval_manifest["files"]["predictions.json"]
        ["sha256"]
    )
    metrics["metrics_path"] = str(
        final_dir / "metrics.json"
    )

    del evaluator
    del coco_dt
    del coco_gt
    del predictions
    del model
    release_gpu()

    return metrics


def metric_deltas(
    standard: dict,
    p2: dict,
) -> dict:
    keys = (
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
    )
    return {
        f"delta_{key}":
            float(p2[key]) - float(standard[key])
        for key in keys
    }


def build_epoch_10_decision(
    protocol: dict,
    standard: dict,
    p2: dict,
) -> dict:
    deltas = metric_deltas(standard, p2)
    rules = (
        protocol["state_machine"]["epoch_10"]
        ["early_reject_p2_if_all"]
    )

    early_reject = (
        deltas["delta_AP_small"]
        <= float(rules["delta_AP_small_max"])
        and
        deltas["delta_AP"]
        <= float(rules["delta_AP_max"])
    )

    return {
        "version": "architecture_gate_epoch10_v4",
        "standard_metrics": standard,
        "p2_metrics": p2,
        "deltas_p2_minus_standard": deltas,
        "rule": rules,
        "early_reject_p2": early_reject,
        "action": (
            "select_standard_and_stop_gate"
            if early_reject
            else "resume_both_to_epoch_20"
        ),
    }


def build_epoch_20_decision(
    protocol: dict,
    standard: dict,
    p2: dict,
) -> dict:
    deltas = metric_deltas(standard, p2)
    rules = (
        protocol["decision_rules"]
        ["p2_adoption_epoch_20"]
    )
    latency = (
        protocol["decision_rules"]
        ["frozen_latency_evidence"]
        ["median_increase_fraction"]
    )

    checks = {
        "delta_AP_small":
            deltas["delta_AP_small"]
            >= float(rules["delta_AP_small_min"]),
        "delta_AP":
            deltas["delta_AP"]
            >= float(rules["delta_AP_min"]),
        "delta_AP75":
            deltas["delta_AP75"]
            >= float(rules["delta_AP75_min"]),
        "delta_AP_medium":
            deltas["delta_AP_medium"]
            >= float(
                rules["delta_AP_medium_min"]
            ),
        "delta_AP_large":
            deltas["delta_AP_large"]
            >= float(
                rules["delta_AP_large_min"]
            ),
        "latency":
            float(latency)
            <= float(
                rules["latency_increase_max"]
            ),
    }
    adopt_p2 = all(checks.values())

    return {
        "version": "architecture_gate_final_v4",
        "decision_epoch": 20,
        "standard_metrics": standard,
        "p2_metrics": p2,
        "deltas_p2_minus_standard": deltas,
        "rules": rules,
        "rule_checks": checks,
        "all_p2_adoption_rules_passed":
            adopt_p2,
        "selected_architecture": (
            "p2_projected"
            if adopt_p2
            else "standard"
        ),
        "selected_architecture_name": (
            protocol["architectures"]
            ["p2_projected"]["name"]
            if adopt_p2
            else protocol["architectures"]
            ["standard"]["name"]
        ),
        "extension_to_epoch_30_allowed": False,
        "gate_checkpoint_is_formal_final_model":
            False,
    }


def build_final_manifest() -> Path:
    output = (
        FORMAL_ROOT / "final_artifact_manifest.json"
    )
    if output.exists():
        raise FileExistsError(output)

    files = {}
    for path in sorted(FORMAL_ROOT.rglob("*")):
        if not path.is_file():
            continue
        if path == output:
            continue
        files[str(path.relative_to(FORMAL_ROOT))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    atomic_write_json(
        output,
        {
            "version": "formal_gate_v4",
            "status": "complete",
            "files": files,
        },
    )
    return output


def execute(protocol: dict) -> None:
    if FORMAL_ROOT.exists():
        raise FileExistsError(
            f"Refusing to overwrite formal gate:\n"
            f"{FORMAL_ROOT}"
        )

    FORMAL_ROOT.mkdir(
        parents=False,
        exist_ok=False,
    )

    run_metadata = {
        "version": "formal_gate_run_v4",
        "status": "running",
        "started_at":
            datetime.now().astimezone().isoformat(),
        "controller_script":
            str(Path(__file__).resolve()),
        "controller_script_sha256":
            sha256(Path(__file__).resolve()),
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256":
            EXPECTED_FROZEN_HASHES[PROTOCOL_PATH],
        "v4_manifest_sha256":
            EXPECTED_FROZEN_HASHES[V4_MANIFEST],
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "nvidia_smi": command_output([
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,"
            "driver_version",
            "--format=csv,noheader",
        ]),
    }
    atomic_write_json(
        FORMAL_ROOT / "run_metadata.json",
        run_metadata,
    )
    atomic_write_json(
        FORMAL_ROOT / "controller_state.json",
        {
            "status": "running",
            "stage": "standard_epoch_10",
        },
    )

    standard_epoch10_checkpoint = (
        train_to_epoch_10(protocol, "standard")
    )
    atomic_write_json(
        FORMAL_ROOT / "controller_state.json",
        {
            "status": "running",
            "stage": "standard_evaluation_10",
        },
    )
    standard_metrics_10 = evaluate_checkpoint(
        protocol,
        "standard",
        10,
        Path(standard_epoch10_checkpoint["path"]),
    )

    atomic_write_json(
        FORMAL_ROOT / "controller_state.json",
        {
            "status": "running",
            "stage": "p2_epoch_10",
        },
    )
    p2_epoch10_checkpoint = (
        train_to_epoch_10(
            protocol,
            "p2_projected",
        )
    )
    atomic_write_json(
        FORMAL_ROOT / "controller_state.json",
        {
            "status": "running",
            "stage": "p2_evaluation_10",
        },
    )
    p2_metrics_10 = evaluate_checkpoint(
        protocol,
        "p2_projected",
        10,
        Path(p2_epoch10_checkpoint["path"]),
    )

    epoch10_decision = build_epoch_10_decision(
        protocol,
        standard_metrics_10,
        p2_metrics_10,
    )
    atomic_write_json(
        FORMAL_ROOT / "epoch10_decision.json",
        epoch10_decision,
    )

    print("\n=== EPOCH-10 DECISION ===")
    print(json.dumps(
        epoch10_decision,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ))

    if epoch10_decision["early_reject_p2"]:
        final_decision = {
            "version": "architecture_gate_final_v4",
            "decision_epoch": 10,
            "reason": "p2_early_rejection_rule_passed",
            "selected_architecture": "standard",
            "selected_architecture_name":
                protocol["architectures"]
                ["standard"]["name"],
            "epoch10_decision":
                epoch10_decision,
            "gate_checkpoint_is_formal_final_model":
                False,
            "extension_to_epoch_20_executed":
                False,
            "extension_to_epoch_30_allowed":
                False,
        }
        atomic_write_json(
            FORMAL_ROOT / "final_decision.json",
            final_decision,
        )
        atomic_write_json(
            FORMAL_ROOT / "controller_state.json",
            {
                "status": "complete",
                "stage":
                    "p2_rejected_at_epoch_10",
                "selected_architecture":
                    "standard",
            },
        )
        manifest = build_final_manifest()

        print("\n=== FORMAL GATE COMPLETE ===")
        print("selected_architecture: standard")
        print("decision_epoch: 10")
        print("manifest:", manifest)
        print(
            "manifest_sha256:",
            sha256(manifest),
        )
        return

    atomic_write_json(
        FORMAL_ROOT / "controller_state.json",
        {
            "status": "running",
            "stage": "standard_resume_to_epoch_20",
        },
    )
    standard_epoch20_checkpoint = (
        resume_to_epoch_20(
            protocol,
            "standard",
            Path(
                standard_epoch10_checkpoint["path"]
            ),
        )
    )
    atomic_write_json(
        FORMAL_ROOT / "controller_state.json",
        {
            "status": "running",
            "stage": "standard_evaluation_20",
        },
    )
    standard_metrics_20 = evaluate_checkpoint(
        protocol,
        "standard",
        20,
        Path(standard_epoch20_checkpoint["path"]),
    )

    atomic_write_json(
        FORMAL_ROOT / "controller_state.json",
        {
            "status": "running",
            "stage": "p2_resume_to_epoch_20",
        },
    )
    p2_epoch20_checkpoint = resume_to_epoch_20(
        protocol,
        "p2_projected",
        Path(p2_epoch10_checkpoint["path"]),
    )
    atomic_write_json(
        FORMAL_ROOT / "controller_state.json",
        {
            "status": "running",
            "stage": "p2_evaluation_20",
        },
    )
    p2_metrics_20 = evaluate_checkpoint(
        protocol,
        "p2_projected",
        20,
        Path(p2_epoch20_checkpoint["path"]),
    )

    final_decision = build_epoch_20_decision(
        protocol,
        standard_metrics_20,
        p2_metrics_20,
    )
    atomic_write_json(
        FORMAL_ROOT / "final_decision.json",
        final_decision,
    )
    atomic_write_json(
        FORMAL_ROOT / "controller_state.json",
        {
            "status": "complete",
            "stage": "epoch_20_decision_complete",
            "selected_architecture":
                final_decision[
                    "selected_architecture"
                ],
        },
    )

    manifest = build_final_manifest()

    print("\n=== FORMAL GATE COMPLETE ===")
    print(
        "selected_architecture:",
        final_decision["selected_architecture"],
    )
    print("decision_epoch: 20")
    print("manifest:", manifest)
    print(
        "manifest_sha256:",
        sha256(manifest),
    )
    print(json.dumps(
        final_decision,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen automatic YOLO11m "
            "Standard-960 vs Projected-P2-960 "
            "architecture gate."
        )
    )
    mode = parser.add_mutually_exclusive_group(
        required=True
    )
    mode.add_argument(
        "--preflight-only",
        action="store_true",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
    )
    args = parser.parse_args()

    protocol = preflight()

    if FORMAL_ROOT.exists():
        raise FileExistsError(
            f"Formal output already exists:\n"
            f"{FORMAL_ROOT}"
        )

    if args.preflight_only:
        print("\nPASS: controller preflight only")
        print("NOTHING TRAINED")
        return

    execute(protocol)


if __name__ == "__main__":
    main()