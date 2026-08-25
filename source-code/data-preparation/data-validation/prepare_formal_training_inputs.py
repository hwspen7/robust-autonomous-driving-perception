from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil

from ultralytics.data.utils import img2label_paths


ROOT = Path(
    "/root/rivermind-data/autodrive/results/evaluation"
)
METHOD = ROOT / "method_gate"
ARCH = ROOT / "architecture_gate"

OUTPUT = (
    ROOT / "formal_training/prepared_inputs_v1"
)
ANCHORS = Path(
    "/dev/shm/rebu_yolo/formal_training_v1/cache_anchors"
)

METHOD_PROTOCOL = (
    METHOD / "frozen_manifests/method_gate_protocol_v1.json"
)
ARCH_SELECTION = (
    ARCH / "frozen_manifests/final_architecture_selection_v1.json"
)
INITIALIZATION = (
    ARCH / "prepared_inputs_v2/"
    "yolo11m_p2_projected_coco_init.pt"
)

SOURCE_CACHE = (
    METHOD / "materialization_v2_cache_isolated"
)
SOURCE_CACHE_MANIFEST = (
    SOURCE_CACHE / "artifact_manifest.json"
)
SOURCE_MATERIALIZATION_MANIFEST = (
    METHOD / "materialization_v1/artifact_manifest.json"
)
SAMPLING_ROOT = METHOD / "sampling_audit_v3"
SAMPLING_MANIFEST = (
    SAMPLING_ROOT / "artifact_manifest.json"
)

TRAIN_CORE = (
    ARCH / "prepared_inputs_v1/train_core.txt"
)
TRAIN_DEV = (
    ARCH / "prepared_inputs_v1/train_dev.txt"
)
TRAIN_COCO = Path(
    "/root/rivermind-data/autodrive/datasets/datasets/"
    "bdd100k_final/coco/annotations/instances_train.json"
)

CLEAN_ROOT = METHOD / "evaluation_v2_clean"
CLEAN_MANIFEST = CLEAN_ROOT / "clean_artifact_manifest.json"
CLEAN_RESULT = CLEAN_ROOT / "clean_comparison.json"

FAILURE_ROOT = METHOD / "evaluation_v3_failures"
FAILURE_MANIFEST = FAILURE_ROOT / "artifact_manifest.json"
FAILURE_RESULT = FAILURE_ROOT / "failure_evaluation.json"

CORRUPTION_ROOT = METHOD / "evaluation_v4_corruptions"
CORRUPTION_MANIFEST = CORRUPTION_ROOT / "artifact_manifest.json"
CORRUPTION_RESULT = CORRUPTION_ROOT / "corruption_metrics.json"
V1_DECISION = CORRUPTION_ROOT / "final_gate_decision.json"

V2_CLEAN_ROOT = METHOD / "evaluation_v5_cafr_v2_clean"
V2_CLEAN_MANIFEST = (
    V2_CLEAN_ROOT / "clean_artifact_manifest.json"
)
V2_CLEAN_RESULT = (
    V2_CLEAN_ROOT / "clean_comparison.json"
)

SCALAR_GATE_CHECKPOINT = (
    METHOD / "training_v1/scalar_risk/weights/last.pt"
)

SHM_V1 = Path(
    "/dev/shm/rebu_yolo/method_gate_v1"
)

EXPECTED = {
    METHOD_PROTOCOL:
        "6be7ae25550360e38195bb785a064bd19cb9c4a65b4625b30b5698861cc1b00a",
    ARCH_SELECTION:
        "b3a20fc148a508334268aa4d43bc251219244dffba134173dee3ac1648de29f0",
    INITIALIZATION:
        "eef43a83adb99ee9b3d9e75c25d3454488b28ad5f86ac67fcf850351c3e848dd",
    SOURCE_CACHE_MANIFEST:
        "434df46b40c443505d68d4f3b55929ffc488afa4c8ab76ac3d8618d7bf2eb21d",
    SOURCE_MATERIALIZATION_MANIFEST:
        "74f26c5cc14a2f5bea497bea921966d5c12cbf81a43ac19c39fe71d226e18c06",
    SAMPLING_MANIFEST:
        "7c96b15970798025e2482613c6fb0304d972e480a7166c5f2e8817b1e7ec93a9",
    TRAIN_CORE:
        "7a13091a343ee4f5a3ae20e60e4ab88728d307b29becb8dd26481edda7ea6751",
    TRAIN_DEV:
        "d4e1c82556eb6adb30fa2d29f6d86e838922d70ad701f7403058b49184c0fcf2",
    TRAIN_COCO:
        "a8e9c188e58a328901af6e3a93b747d517da354e2dab3292ff26d7c0fbd32cd2",
    CLEAN_MANIFEST:
        "29dc8882d383c37c6e9eb5a8cf826b9d562a3c48e9d0ef8d5a23be5b78350f97",
    FAILURE_MANIFEST:
        "2e57f0e7cb31731362511ca8d29fa9fa52328336241aea696a322eee9f5c2457",
    CORRUPTION_MANIFEST:
        "4acd7c2bebb826dfcfe7f59ee2cc753644e63419c147dfc8ea1c84d278381ce8",
    V2_CLEAN_MANIFEST:
        "c55efe0d1faa28a2dbd7e8d1bfd2fe8c3e9bf9e3cc38c5b63e4501982a571432",
    SCALAR_GATE_CHECKPOINT:
        "3093b249d8052924d8324c826731a77b82ab98327a41ff68ab3a2ad615f74b9d",
}

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
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)

    actual = sha256(path)
    print(path)
    print(" expected:", expected)
    print(" actual  :", actual)

    if actual != expected:
        raise AssertionError(path)


def load_json(path: Path):
    return json.loads(path.read_text())


def read_paths(path: Path) -> list[str]:
    values = [
        row.strip()
        for row in path.read_text().splitlines()
        if row.strip()
    ]
    if len(values) != len(set(values)):
        raise AssertionError(f"Duplicate paths: {path}")
    return values


def verify_manifest_artifact(
    root: Path,
    manifest_path: Path,
    relative: str,
) -> None:
    manifest = load_json(manifest_path)
    record = manifest["files"][relative]
    path = root / relative
    verify(path, record["sha256"])
    assert path.stat().st_size == int(record["bytes"])


def predicted_cache(image_path: str) -> Path:
    label_path = img2label_paths([image_path])[0]
    return Path(label_path).parent.with_suffix(".cache")


def create_anchor(
    staging_root: Path,
    final_root: Path,
    name: str,
    source_image: Path,
) -> str:
    source_image = source_image.resolve()
    source_label = Path(
        img2label_paths([str(source_image)])[0]
    ).resolve()

    assert source_image.is_file()
    assert source_label.is_file()

    image_dir = staging_root / name / "images"
    label_dir = staging_root / name / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    image_link = image_dir / f"cache_anchor{source_image.suffix}"
    label_link = label_dir / "cache_anchor.txt"

    os.symlink(source_image, image_link)
    os.symlink(source_label, label_link)

    final_image = (
        final_root / name / "images" / image_link.name
    )
    return str(final_image)


def yaml_text(train_list: Path) -> str:
    lines = [
        "path: /root/rivermind-data/autodrive/"
        "datasets/datasets/bdd100k_final",
        f"train: {train_list}",
        "val: images/val",
        "",
        "names:",
    ]
    for class_id, name in NAMES.items():
        lines.append(f"  {class_id}: {name}")
    return "\n".join(lines) + "\n"


def preflight():
    print("=== FORMAL REBU-YOLO PREPARATION PREFLIGHT ===")

    for path, expected in EXPECTED.items():
        verify(path, expected)

    verify_manifest_artifact(
        CLEAN_ROOT,
        CLEAN_MANIFEST,
        "clean_comparison.json",
    )
    verify_manifest_artifact(
        FAILURE_ROOT,
        FAILURE_MANIFEST,
        "failure_evaluation.json",
    )
    verify_manifest_artifact(
        CORRUPTION_ROOT,
        CORRUPTION_MANIFEST,
        "corruption_metrics.json",
    )
    verify_manifest_artifact(
        CORRUPTION_ROOT,
        CORRUPTION_MANIFEST,
        "final_gate_decision.json",
    )
    verify_manifest_artifact(
        V2_CLEAN_ROOT,
        V2_CLEAN_MANIFEST,
        "clean_comparison.json",
    )

    cache_manifest = load_json(SOURCE_CACHE_MANIFEST)
    for relative, record in cache_manifest["files"].items():
        path = SOURCE_CACHE / relative
        verify(path, record["sha256"])
        assert path.stat().st_size == int(record["bytes"])

    sampling_manifest = load_json(SAMPLING_MANIFEST)
    for relative in (
        "scalar_risk_selection.csv",
        "common_base_image_ids.json",
    ):
        record = sampling_manifest["files"][relative]
        verify(
            SAMPLING_ROOT / relative,
            record["sha256"],
        )

    clean = load_json(CLEAN_RESULT)
    failure = load_json(FAILURE_RESULT)
    corruption = load_json(CORRUPTION_RESULT)
    v1_decision = load_json(V1_DECISION)
    v2_clean = load_json(V2_CLEAN_RESULT)

    scalar_clean = clean["comparisons"][
        "scalar_risk_minus_uniform"
    ]
    scalar_failure = failure["comparisons"][
        "scalar_risk_minus_uniform"
    ]
    scalar_corruption = corruption["comparisons"][
        "scalar_risk_minus_uniform"
    ]

    assert scalar_clean["delta_AP"] > 0.0
    assert scalar_clean["delta_AP75"] > 0.0
    assert (
        scalar_failure["delta_failure_macro_AR100"]
        >= 0.005
    )
    assert (
        scalar_failure[
            "delta_small_joint_not_detected_recall50"
        ] >= 0.005
    )
    assert (
        scalar_corruption[
            "delta_mean_corruption_AP_small"
        ] >= -0.002
    )
    assert v1_decision["gate_result"] == "FAIL"
    assert v2_clean["clean_gate_passed"] is False

    uniform_core = read_paths(
        SOURCE_CACHE / "uniform_train.txt"
    )
    scalar_core = read_paths(
        SOURCE_CACHE / "scalar_risk_train.txt"
    )
    train_dev = read_paths(TRAIN_DEV)

    assert len(uniform_core) == 65000
    assert len(scalar_core) == 65000
    assert len(train_dev) == 5000

    scalar_prefix = (
        str(SHM_V1 / "scalar_risk/images") + "/"
    )
    scalar_crop_count = sum(
        path.startswith(scalar_prefix)
        for path in scalar_core
    )
    assert scalar_crop_count == 19500

    assert len(list(
        (SHM_V1 / "scalar_risk/images").glob("*.jpg")
    )) == 19500
    assert len(list(
        (SHM_V1 / "scalar_risk/labels").glob("*.txt")
    )) == 19500

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if ANCHORS.exists():
        raise FileExistsError(ANCHORS)

    disk_free = (
        shutil.disk_usage("/root/rivermind-data").free
        / 1024**3
    )
    shm_free = (
        shutil.disk_usage("/dev/shm").free
        / 1024**3
    )

    print("disk_free_GiB:", round(disk_free, 3))
    print("shm_free_GiB:", round(shm_free, 3))
    assert disk_free >= 8.0
    assert shm_free >= 8.0

    evidence = {
        "scalar_minus_uniform_clean_AP":
            scalar_clean["delta_AP"],
        "scalar_minus_uniform_clean_AP75":
            scalar_clean["delta_AP75"],
        "scalar_minus_uniform_failure_macro_AR100":
            scalar_failure[
                "delta_failure_macro_AR100"
            ],
        "scalar_minus_uniform_small_joint_recall50":
            scalar_failure[
                "delta_small_joint_not_detected_recall50"
            ],
        "scalar_minus_uniform_corruption_AP_small":
            scalar_corruption[
                "delta_mean_corruption_AP_small"
            ],
        "typed_v1_gate_result":
            v1_decision["gate_result"],
        "typed_v2_clean_gate_passed":
            v2_clean["clean_gate_passed"],
    }

    print("PASS: Scalar-Risk fallback evidence is valid")
    print("PASS: frozen 65k source lists and 5k dev are intact")
    print("NOTHING WRITTEN OR TRAINED")

    return uniform_core, scalar_core, train_dev, evidence


def execute():
    (
        uniform_core,
        scalar_core,
        train_dev,
        evidence,
    ) = preflight()

    token = str(os.getpid())
    output_staging = Path(
        str(OUTPUT) + f".incomplete-{token}"
    )
    anchor_staging = Path(
        str(ANCHORS) + f".incomplete-{token}"
    )

    output_staging.mkdir(parents=True)
    anchor_staging.mkdir(parents=True)

    uniform_alias = create_anchor(
        anchor_staging,
        ANCHORS,
        "uniform",
        Path(uniform_core[0]),
    )
    rebu_alias = create_anchor(
        anchor_staging,
        ANCHORS,
        "rebu_risk",
        Path(scalar_core[0]),
    )

    uniform_full = (
        [uniform_alias]
        + uniform_core[1:]
        + train_dev
    )
    rebu_full = (
        [rebu_alias]
        + scalar_core[1:]
        + train_dev
    )

    assert len(uniform_full) == 70000
    assert len(rebu_full) == 70000
    assert len(set(uniform_full)) == 70000
    assert len(set(rebu_full)) == 70000

    uniform_list = (
        output_staging / "uniform_full70k_train.txt"
    )
    rebu_list = (
        output_staging / "rebu_risk_full70k_train.txt"
    )

    uniform_list.write_text(
        "\n".join(uniform_full) + "\n"
    )
    rebu_list.write_text(
        "\n".join(rebu_full) + "\n"
    )

    final_uniform_list = (
        OUTPUT / uniform_list.name
    )
    final_rebu_list = (
        OUTPUT / rebu_list.name
    )

    uniform_yaml = (
        output_staging / "uniform_full70k.yaml"
    )
    rebu_yaml = (
        output_staging / "rebu_risk_full70k.yaml"
    )

    uniform_yaml.write_text(
        yaml_text(final_uniform_list)
    )
    rebu_yaml.write_text(
        yaml_text(final_rebu_list)
    )

    cache_routes = {
        "uniform": str(
            predicted_cache(uniform_alias)
        ),
        "rebu_risk": str(
            predicted_cache(rebu_alias)
        ),
    }
    assert len(set(cache_routes.values())) == 2
    assert all(
        route.startswith("/dev/shm/")
        for route in cache_routes.values()
    )

    selection = {
        "version": "formal_method_selection_v1",
        "status": "frozen",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "selected_method": "Rebu-YOLO-Risk",
        "selected_development_arm": "scalar_risk",
        "selection_timing": "post_typed_gate_fallback",
        "primary_theme":
            "object-level failure diagnosis and targeted repair",
        "typed_cafr_v1_passed": False,
        "typed_cafr_v2_passed": False,
        "further_typed_tuning_allowed": False,
        "selection_evidence": evidence,
        "official_validation_used_for_method_selection": False,
        "gate_checkpoint_used_as_formal_initialization": False,
        "formal_initialization": str(INITIALIZATION),
        "formal_initialization_sha256":
            EXPECTED[INITIALIZATION],
    }

    selection_path = (
        output_staging / "formal_method_selection.json"
    )
    selection_path.write_text(
        json.dumps(
            selection,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
    )

    protocol = {
        "version": "rebu_yolo_formal_training_protocol_v1",
        "status": "frozen_before_formal_training",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "architecture": "YOLO11m-P2-Projected-960",
        "method": "Rebu-YOLO-Risk",
        "initialization": {
            "path": str(INITIALIZATION),
            "sha256": EXPECTED[INITIALIZATION],
            "source": "frozen projected-P2 COCO initialization",
        },
        "data": {
            "official_train_images": 70000,
            "uniform_views_per_epoch": 70000,
            "rebu_views_per_epoch": 70000,
            "rebu_original_views": 50500,
            "rebu_object_centric_views": 19500,
            "uniform_yaml": str(
                OUTPUT / uniform_yaml.name
            ),
            "rebu_yaml": str(
                OUTPUT / rebu_yaml.name
            ),
            "uniform_list_sha256":
                sha256(uniform_list),
            "rebu_list_sha256":
                sha256(rebu_list),
            "former_train_dev_rejoined_as_originals": True,
            "official_val_used_for_training": False,
            "cache_routes": cache_routes,
        },
        "schedule": {
            "common_uniform_epochs": 40,
            "branch_epochs": 60,
            "effective_epochs_per_final_arm": 100,
            "common_checkpoint_is_final_model": False,
            "branch_initialization":
                "exact common-epoch40 EMA weights",
            "branch_optimizer_resumed": False,
            "branch_optimizer":
                "fresh identical SGD for both branches",
            "formal_endpoints": [
                "Uniform-Control-E100",
                "Rebu-YOLO-Risk-E100",
            ],
        },
        "common_training": {
            "epochs": 40,
            "imgsz": 960,
            "batch": 8,
            "nbs": 64,
            "optimizer": "SGD",
            "lr0": 0.01,
            "lrf": 0.01,
            "cos_lr": True,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "mosaic": 1.0,
            "close_mosaic": 10,
        },
        "branch_training": {
            "epochs": 60,
            "imgsz": 960,
            "batch": 8,
            "nbs": 64,
            "optimizer": "SGD",
            "lr0": 0.001,
            "lrf": 0.05,
            "cos_lr": True,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 1.0,
            "mosaic": 0.0,
            "close_mosaic": 0,
        },
        "shared": {
            "device": 0,
            "workers": 8,
            "amp": True,
            "deterministic": True,
            "seed": 20260809,
            "validation_during_training": False,
            "patience": 0,
            "save_period": 5,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "translate": 0.1,
            "scale": 0.5,
            "fliplr": 0.5,
            "mixup": 0.0,
            "cutmix": 0.0,
            "copy_paste": 0.0,
        },
        "evaluation": {
            "official_val_evaluated_only_after_training": True,
            "clean_and_failure_metrics": True,
            "controlled_corruptions": True,
            "full_multi_seed_training": False,
            "paired_bootstrap_resamples": 1000,
        },
        "stopping": {
            "fixed_schedule": True,
            "manual_early_stopping_allowed": False,
            "extension_beyond_100_allowed": False,
        },
    }

    protocol_path = (
        output_staging / "formal_training_protocol.json"
    )
    protocol_path.write_text(
        json.dumps(
            protocol,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
    )

    runtime = {
        "version": "formal_training_runtime_anchors_v1",
        "anchors_root": str(ANCHORS),
        "cache_routes": cache_routes,
        "uniform_anchor_source":
            str(Path(uniform_core[0]).resolve()),
        "rebu_anchor_source":
            str(Path(scalar_core[0]).resolve()),
    }
    runtime_path = (
        output_staging / "runtime_anchors.json"
    )
    runtime_path.write_text(
        json.dumps(
            runtime,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
    )

    files = {}
    for path in sorted(output_staging.iterdir()):
        if path.is_file():
            files[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }

    manifest = {
        "version": "rebu_yolo_formal_inputs_v1",
        "status": "frozen",
        "files": files,
    }
    manifest_path = (
        output_staging / "artifact_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
    )

    os.replace(anchor_staging, ANCHORS)
    os.replace(output_staging, OUTPUT)

    print("\n" + "=" * 100)
    print("REBU-YOLO FORMAL TRAINING INPUTS FROZEN")
    print("=" * 100)
    print("uniform_views:", len(uniform_full))
    print("rebu_views:", len(rebu_full))
    print("rebu_object_centric_views: 19500")
    print(
        "protocol_sha256:",
        sha256(OUTPUT / "formal_training_protocol.json"),
    )
    print(
        "manifest_sha256:",
        sha256(OUTPUT / "artifact_manifest.json"),
    )
    print("PASS: two equal-budget 70k formal arms are frozen")
    print("NOTHING TRAINED")
    print("NEXT: start common Uniform epochs 1-40")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.execute:
        execute()
    else:
        preflight()


if __name__ == "__main__":
    main()