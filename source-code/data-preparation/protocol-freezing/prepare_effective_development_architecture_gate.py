from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform


REPO = Path(
    "/root/rivermind-data/autodrive/code/"
    "robust-autonomous-driving-perception"
)
ARCH_ROOT = Path(
    "/root/rivermind-data/autodrive/results/evaluation/"
    "architecture_gate"
)
V1 = ARCH_ROOT / "prepared_inputs_v1"
V2 = ARCH_ROOT / "prepared_inputs_v2"
DATA_QUALITY = ARCH_ROOT / "data_quality_v1"

OUTPUT_DIR = ARCH_ROOT / "prepared_inputs_v3"
STAGING_DIR = ARCH_ROOT / f"prepared_inputs_v3.incomplete-{os.getpid()}"

STANDARD_INIT = REPO / "yolo11m.pt"
STANDARD_CONFIG = V1 / "yolo11m_standard_gate.yaml"
P2_INIT = V2 / "yolo11m_p2_projected_coco_init.pt"
P2_CONFIG = V2 / "yolo11m_p2_projected_bdd10.yaml"

TRAIN_CORE = V1 / "train_core.txt"
TRAIN_DEV = V1 / "train_dev.txt"
RAW_DEV_COCO = V1 / "instances_train_dev.json"

V1_MANIFEST = V1 / "artifact_manifest.json"
V2_MANIFEST = V2 / "artifact_manifest.json"
EXCEPTION_RECORD = (
    DATA_QUALITY / "duplicate_annotation_exception.json"
)

TASKS_SOURCE = Path(
    "/opt/conda/lib/python3.10/site-packages/"
    "ultralytics/nn/tasks.py"
)

EXPECTED_HASHES = {
    STANDARD_INIT:
        "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95",
    STANDARD_CONFIG:
        "774f9377818db6f2145b1c69967f366f6a00b81bfa7f84372997e9acd2c7e94c",
    P2_INIT:
        "eef43a83adb99ee9b3d9e75c25d3454488b28ad5f86ac67fcf850351c3e848dd",
    P2_CONFIG:
        "bd6bd6cde7f73f54d2e80fd4ad58b44a85d14cb14cb7fb450548b52f5d6ee0ef",
    TRAIN_CORE:
        "7a13091a343ee4f5a3ae20e60e4ab88728d307b29becb8dd26481edda7ea6751",
    TRAIN_DEV:
        "d4e1c82556eb6adb30fa2d29f6d86e838922d70ad701f7403058b49184c0fcf2",
    RAW_DEV_COCO:
        "1cddc4f25f987e7a6887c52b60ff2e5adbf73415d5b8b9f3f50f4fa6719b1be4",
    V1_MANIFEST:
        "3d4ad6089b1e44d95fe05244bfb8c6527425c4847692b3e70c86fe2044e0e43e",
    V2_MANIFEST:
        "b3e47e99c764342a2146a799d65b3939a0e54c471fefcf3880b3d89b8a5e050a",
    EXCEPTION_RECORD:
        "62fa95492d68d2e790a20afc489747acf309f300afe425b816717c491196ec85",
}

DUPLICATE_IMAGE_ID = 54748
CANONICAL_ANNOTATION_ID = 1005729
EXCLUDED_ANNOTATION_ID = 1005741

DATASET_ROOT = Path(
    "/root/rivermind-data/autodrive/datasets/datasets/"
    "bdd100k_final"
)

NAMES = {
    0: "pedestrian",
    1: "rider",
    2: "car",
    3: "truck",
    4: "bus",
    5: "train",
    6: "motorcycle",
    7: "bicycle",
    8: "traffic light",
    9: "traffic sign",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"

    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def verify_input(path: Path, expected: str) -> None:
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


def count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for row in handle if row.strip())


def annotation_signature(row: dict) -> tuple:
    return (
        int(row["image_id"]),
        int(row["category_id"]),
        tuple(float(value) for value in row["bbox"]),
        int(row.get("iscrowd", 0)),
    )


def build_effective_dev_coco() -> dict:
    print("\n=== BUILDING EFFECTIVE TRAIN-DEV COCO ===")
    raw = json.loads(RAW_DEV_COCO.read_text())

    assert len(raw["images"]) == 5000
    assert len(raw["annotations"]) == 92393

    image_ids = {int(row["id"]) for row in raw["images"]}
    assert len(image_ids) == 5000
    assert DUPLICATE_IMAGE_ID in image_ids

    selected = {
        int(row["id"]): row
        for row in raw["annotations"]
        if int(row["id"]) in {
            CANONICAL_ANNOTATION_ID,
            EXCLUDED_ANNOTATION_ID,
        }
    }
    assert set(selected) == {
        CANONICAL_ANNOTATION_ID,
        EXCLUDED_ANNOTATION_ID,
    }

    canonical = selected[CANONICAL_ANNOTATION_ID]
    excluded = selected[EXCLUDED_ANNOTATION_ID]

    assert annotation_signature(canonical) == annotation_signature(excluded)
    assert int(canonical["image_id"]) == DUPLICATE_IMAGE_ID
    assert int(canonical["category_id"]) == 3
    assert [float(v) for v in canonical["bbox"]] == [
        536.0, 339.0, 12.0, 7.0
    ]

    seen = {}
    duplicate_pairs = []

    for annotation in raw["annotations"]:
        signature = annotation_signature(annotation)
        annotation_id = int(annotation["id"])

        if signature in seen:
            duplicate_pairs.append(
                (seen[signature], annotation_id)
            )
        else:
            seen[signature] = annotation_id

    normalized_pairs = {
        tuple(sorted(pair))
        for pair in duplicate_pairs
    }
    assert normalized_pairs == {
        (
            CANONICAL_ANNOTATION_ID,
            EXCLUDED_ANNOTATION_ID,
        )
    }, normalized_pairs

    effective_annotations = [
        row
        for row in raw["annotations"]
        if int(row["id"]) != EXCLUDED_ANNOTATION_ID
    ]
    assert len(effective_annotations) == 92392

    effective = dict(raw)
    effective["annotations"] = effective_annotations

    return effective


def build_dataset_yaml() -> str:
    lines = [
        f"path: {DATASET_ROOT}",
        f"train: {TRAIN_CORE}",
        f"val: {TRAIN_DEV}",
        "",
        "names:",
    ]

    for class_id, class_name in NAMES.items():
        lines.append(f"  {class_id}: {class_name}")

    return "\n".join(lines) + "\n"


def build_protocol(
    effective_coco_path: Path,
    dataset_yaml_path: Path,
) -> dict:
    return {
        "version": "architecture_gate_protocol_v3",
        "status": "frozen_before_training",
        "purpose": (
            "Select one YOLO11m-960 architecture before Uniform "
            "and CAFR formal training."
        ),
        "ancestry": {
            "prepared_inputs_v1_manifest_sha256":
                EXPECTED_HASHES[V1_MANIFEST],
            "prepared_inputs_v2_manifest_sha256":
                EXPECTED_HASHES[V2_MANIFEST],
            "data_quality_exception_sha256":
                EXPECTED_HASHES[EXCEPTION_RECORD],
        },
        "source_state": {
            "builder_script": str(Path(__file__).resolve()),
            "builder_script_sha256":
                sha256(Path(__file__).resolve()),
            "ultralytics_tasks_source": str(TASKS_SOURCE),
            "ultralytics_tasks_source_sha256":
                sha256(TASKS_SOURCE),
            "python": platform.python_version(),
            "ultralytics":
                importlib.metadata.version("ultralytics"),
        },
        "data": {
            "dataset_yaml": str(
                OUTPUT_DIR / dataset_yaml_path.name
            ),
            "dataset_yaml_sha256": sha256(dataset_yaml_path),
            "train_core_list": str(TRAIN_CORE),
            "train_core_list_sha256":
                EXPECTED_HASHES[TRAIN_CORE],
            "train_core_images": 65000,
            "train_core_objects": 1194459,
            "train_dev_list": str(TRAIN_DEV),
            "train_dev_list_sha256":
                EXPECTED_HASHES[TRAIN_DEV],
            "train_dev_images": 5000,
            "train_dev_raw_objects": 92393,
            "train_dev_effective_objects": 92392,
            "effective_dev_coco": str(
                OUTPUT_DIR / effective_coco_path.name
            ),
            "effective_dev_coco_sha256":
                sha256(effective_coco_path),
            "excluded_duplicate_annotation_id":
                EXCLUDED_ANNOTATION_ID,
            "official_validation_used_for_gate": False,
        },
        "architectures": {
            "standard": {
                "name": "YOLO11m-Standard-960",
                "strides": [8, 16, 32],
                "config": str(STANDARD_CONFIG),
                "config_sha256":
                    EXPECTED_HASHES[STANDARD_CONFIG],
                "initialization": str(STANDARD_INIT),
                "initialization_sha256":
                    EXPECTED_HASHES[STANDARD_INIT],
                "historical_bdd_best_used": False,
            },
            "p2_projected": {
                "name": "YOLO11m-P2-Projected-960",
                "strides": [4, 8, 16, 32],
                "config": str(P2_CONFIG),
                "config_sha256":
                    EXPECTED_HASHES[P2_CONFIG],
                "initialization": str(P2_INIT),
                "initialization_sha256":
                    EXPECTED_HASHES[P2_INIT],
                "loaded_parameter_fraction":
                    0.9749652758658931,
                "parameter_increase_fraction":
                    0.02567755463072552,
                "gflops_increase_fraction":
                    0.3871859848104282,
                "historical_bdd_best_used": False,
            },
        },
        "training": {
            "device": 0,
            "seed": 20260809,
            "deterministic": True,
            "image_size": 960,
            "physical_batch": 8,
            "nominal_batch": 64,
            "workers": 8,
            "cache": "ram",
            "amp": True,
            "optimizer": "SGD",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "cos_lr": False,
            "planned_schedule_epochs": 20,
            "checkpoint_epochs": [10, 20],
            "close_mosaic": 10,
            "save_period": 1,
            "validation_during_training": False,
            "plots_during_training": False,
            "augmentation": {
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "degrees": 0.0,
                "translate": 0.1,
                "scale": 0.5,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.5,
                "mosaic": 1.0,
                "mixup": 0.0,
                "cutmix": 0.0,
                "copy_paste": 0.0,
            },
        },
        "evaluation": {
            "split": "frozen_train_dev_5000",
            "coco_annotation_file": str(
                OUTPUT_DIR / effective_coco_path.name
            ),
            "iou_type": "bbox",
            "confidence_threshold": 0.001,
            "nms_iou_threshold": 0.7,
            "max_detections_per_image": 300,
            "metrics": [
                "AP",
                "AP50",
                "AP75",
                "AP_small",
                "AP_medium",
                "AP_large",
            ],
            "evaluate_only_at_epochs": [10, 20],
        },
        "state_machine": {
            "execution_order": [
                "standard_epoch_10",
                "standard_evaluation_10",
                "p2_epoch_10",
                "p2_evaluation_10",
                "epoch_10_decision",
                "conditional_resume_both_to_epoch_20",
                "standard_evaluation_20",
                "p2_evaluation_20",
                "final_architecture_decision",
            ],
            "epoch_10": {
                "p2_can_be_accepted": False,
                "early_reject_p2_if_all": {
                    "delta_AP_small_max": 0.0,
                    "delta_AP_max": -0.005,
                },
                "otherwise": (
                    "Resume both architectures from their own "
                    "epoch-10 last.pt to epoch 20."
                ),
            },
            "epoch_20": {
                "adopt_p2_only_if_all_final_rules_pass": True,
                "fallback_architecture": "standard",
                "extension_to_epoch_30_allowed": False,
            },
            "resume_requirements": [
                "model",
                "optimizer",
                "lr_scheduler",
                "ema",
                "epoch",
                "training_arguments",
            ],
            "resume_checkpoint": "last.pt",
            "best_pt_resume_forbidden": True,
            "cross_architecture_weight_transfer_forbidden": True,
        },
        "decision_rules": {
            "metric_deltas": "p2_minus_standard",
            "primary_metric": "AP_small",
            "p2_adoption_epoch_20": {
                "delta_AP_small_min": 0.010,
                "delta_AP_min": -0.002,
                "delta_AP75_min": -0.003,
                "delta_AP_medium_min": -0.003,
                "delta_AP_large_min": -0.005,
                "latency_increase_max": 0.30,
            },
            "frozen_latency_evidence": {
                "precision": "FP16",
                "batch": 1,
                "image_size": 960,
                "median_increase_fraction":
                    0.11336106078953367,
                "p90_increase_fraction":
                    0.10668939473765038,
                "threshold": 0.30,
                "passed": True,
            },
            "absolute_final_AP_target_used_for_gate": False,
            "official_validation_used_for_gate": False,
            "tie_or_failed_p2_rule": "select_standard",
        },
        "failure_policy": {
            "input_hash_mismatch": "abort",
            "standard_training_invalid": "abort",
            "p2_training_invalid": "abort_and_diagnose",
            "evaluation_invalid": "abort",
            "automatic_threshold_change": "forbidden",
            "manual_metric_override": "forbidden",
        },
    }


def main() -> None:
    print("=== V3 FROZEN-INPUT VERIFICATION ===")

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {OUTPUT_DIR}"
        )

    incomplete = sorted(
        ARCH_ROOT.glob("prepared_inputs_v3.incomplete-*")
    )
    if incomplete:
        raise FileExistsError(
            f"Incomplete V3 directories already exist: {incomplete}"
        )

    for path, expected in EXPECTED_HASHES.items():
        verify_input(path, expected)

    assert TASKS_SOURCE.is_file(), TASKS_SOURCE
    assert count_nonempty_lines(TRAIN_CORE) == 65000
    assert count_nonempty_lines(TRAIN_DEV) == 5000

    exception = json.loads(EXCEPTION_RECORD.read_text())
    assert exception["status"] == "frozen"
    assert (
        exception["duplicate"]["canonical_annotation_id"]
        == CANONICAL_ANNOTATION_ID
    )
    assert (
        exception["duplicate"]["excluded_annotation_id"]
        == EXCLUDED_ANNOTATION_ID
    )

    STAGING_DIR.mkdir(parents=False, exist_ok=False)

    effective_coco_path = (
        STAGING_DIR / "instances_train_dev_effective.json"
    )
    dataset_yaml_path = (
        STAGING_DIR / "bdd100k_architecture_gate_v3.yaml"
    )
    protocol_path = (
        STAGING_DIR / "formal_gate_protocol_v3.json"
    )
    manifest_path = (
        STAGING_DIR / "artifact_manifest.json"
    )

    effective_coco = build_effective_dev_coco()
    write_json(effective_coco_path, effective_coco)

    dataset_yaml = build_dataset_yaml()
    with dataset_yaml_path.open(
        "x", encoding="utf-8"
    ) as handle:
        handle.write(dataset_yaml)
        handle.flush()
        os.fsync(handle.fileno())

    protocol = build_protocol(
        effective_coco_path=effective_coco_path,
        dataset_yaml_path=dataset_yaml_path,
    )
    write_json(protocol_path, protocol)

    artifact_files = [
        effective_coco_path,
        dataset_yaml_path,
        protocol_path,
    ]
    manifest = {
        "version": "prepared_inputs_v3",
        "status": "frozen",
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifact_files
        },
    }
    write_json(manifest_path, manifest)

    for path in [
        effective_coco_path,
        dataset_yaml_path,
        protocol_path,
        manifest_path,
    ]:
        path.chmod(0o444)

    os.replace(STAGING_DIR, OUTPUT_DIR)

    final_manifest = OUTPUT_DIR / "artifact_manifest.json"

    print("\n=== V3 PREPARATION COMPLETE ===")
    print("output:", OUTPUT_DIR)
    print(
        "effective_dev_coco:",
        OUTPUT_DIR / effective_coco_path.name,
    )
    print(
        "dataset_yaml:",
        OUTPUT_DIR / dataset_yaml_path.name,
    )
    print(
        "protocol:",
        OUTPUT_DIR / protocol_path.name,
    )
    print(
        "manifest:",
        final_manifest,
    )
    print(
        "manifest_sha256:",
        sha256(final_manifest),
    )
    print("train_core_images: 65000")
    print("train_dev_images: 5000")
    print("train_dev_raw_objects: 92393")
    print("train_dev_effective_objects: 92392")
    print("stage_1: both architectures to epoch 10")
    print("stage_2: conditional resume to epoch 20")
    print("extension_to_epoch_30_allowed: False")
    print(
        "PASS: projected-P2 adaptive gate protocol "
        "is frozen before training"
    )


if __name__ == "__main__":
    main()