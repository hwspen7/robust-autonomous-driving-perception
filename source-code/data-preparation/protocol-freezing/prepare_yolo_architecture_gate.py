from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

SEED = 20260809

ROOT = Path("/root/rivermind-data/autodrive")
REPO = ROOT / "code/robust-autonomous-driving-perception"
DATASET = ROOT / "datasets/datasets/bdd100k_final"
RESULTS = ROOT / "results"

OFFICIAL_WEIGHT = REPO / "yolo11m.pt"
SOURCE_YOLO_YAML = Path(
    "/opt/conda/lib/python3.10/site-packages/"
    "ultralytics/cfg/models/11/yolo11.yaml"
)
TRAIN_COCO = (
        DATASET
        / "coco/annotations/instances_train.json"
)
SPLIT_DIR = (
        RESULTS
        / "evaluation/train_risk_labels/calibration_split"
)

CORE_IDS_PATH = SPLIT_DIR / "train_core_image_ids.json"
DEV_IDS_PATH = SPLIT_DIR / "train_dev_image_ids.json"
SPLIT_METADATA_PATH = SPLIT_DIR / "split_metadata.json"

OUTPUT_DIR = (
        RESULTS
        / "evaluation/architecture_gate/prepared_inputs_v1"
)

EXPECTED_HASHES = {
    OFFICIAL_WEIGHT:
        "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95",
    SOURCE_YOLO_YAML:
        "43d8a7c86acc77282ddf4966d6526e091e5b064c140ed4a6e4c0b68ecbc3784c",
    CORE_IDS_PATH:
        "9996d854b06ddefd12dde1949ecae30c6d08230eb10003e7ec1863a3386ee1a3",
    DEV_IDS_PATH:
        "961ddf514525223aa08599774c167ffcf81752551cdc6619fc6febb1184ada8c",
    SPLIT_METADATA_PATH:
        "cfd4c2cc92dd0206d106431dc0c4a9b682b86b88e9ab5ce2862559a0add5fa89",
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
        while True:
            block = handle.read(8 * 1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()

def write_json(
        path: Path,
        value,
) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

def load_image_ids(path: Path) -> set[int]:
    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if isinstance(value, list):
        raw_ids = value
    elif isinstance(value, dict):
        raw_ids = None

        for key in (
                "image_ids",
                "ids",
                "train_core_image_ids",
                "train_dev_image_ids",
        ):
            candidate = value.get(key)

            if isinstance(candidate, list):
                raw_ids = candidate
                break

        if raw_ids is None:
            lists = [
                item
                for item in value.values()
                if isinstance(item, list)
            ]

            if len(lists) != 1:
                raise ValueError(
                    f"Unable to identify image IDs in {path}"
                )

            raw_ids = lists[0]
    else:
        raise TypeError(
            f"Unsupported split format: {path}"
        )

    result = set()

    for item in raw_ids:
        if isinstance(item, dict):
            item = item["id"]

        result.add(int(item))

    if len(result) != len(raw_ids):
        raise ValueError(
            f"Duplicate image IDs in {path}"
        )

    return result

def resolve_image_path(
        image_record: dict,
) -> Path:
    file_name = Path(image_record["file_name"])

    candidates = [
        DATASET / file_name,
        DATASET / "images/train" / file_name.name,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Image not found for COCO record: {image_record}"
    )

def image_to_label(
        image_path: Path,
        label_root: Path,
) -> Path:
    relative = image_path.relative_to(
        DATASET / "images"
    )

    return (
            label_root
            / relative
    ).with_suffix(".txt")

def validate_split(
        split_name: str,
        image_ids: set[int],
        image_by_id: dict[int, dict],
) -> tuple[list[str], int, str]:
    paths = []
    object_count = 0
    digest = hashlib.sha256()

    standard_label_root = DATASET / "labels"
    duplicate_label_root = DATASET / "yolo/labels"

    for index, image_id in enumerate(
            sorted(image_ids),
            start=1,
    ):
        image_record = image_by_id[image_id]
        image_path = resolve_image_path(image_record)

        standard_label = image_to_label(
            image_path,
            standard_label_root,
        )
        duplicate_label = image_to_label(
            image_path,
            duplicate_label_root,
        )

        if not standard_label.is_file():
            raise FileNotFoundError(standard_label)

        if not duplicate_label.is_file():
            raise FileNotFoundError(duplicate_label)

        standard_bytes = standard_label.read_bytes()
        duplicate_bytes = duplicate_label.read_bytes()

        if standard_bytes != duplicate_bytes:
            raise AssertionError(
                "The two label trees differ:\n"
                f"{standard_label}\n"
                f"{duplicate_label}"
            )

        digest.update(
            f"{image_id}\0".encode("utf-8")
        )
        digest.update(standard_bytes)
        digest.update(b"\0")

        text = standard_bytes.decode(
            "utf-8",
            errors="strict",
        )

        for line_number, line in enumerate(
                text.splitlines(),
                start=1,
        ):
            if not line.strip():
                continue

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"Invalid YOLO row: "
                    f"{standard_label}:{line_number}"
                )

            class_value = float(parts[0])
            class_id = int(class_value)

            if class_value != class_id:
                raise ValueError(
                    f"Non-integer class ID: "
                    f"{standard_label}:{line_number}"
                )

            if class_id not in NAMES:
                raise ValueError(
                    f"Out-of-range class ID: "
                    f"{standard_label}:{line_number}"
                )

            x, y, width, height = map(
                float,
                parts[1:],
            )

            tolerance = 1e-6

            if not (
                    -tolerance <= x <= 1 + tolerance
                    and -tolerance <= y <= 1 + tolerance
                    and 0 < width <= 1 + tolerance
                    and 0 < height <= 1 + tolerance
            ):
                raise ValueError(
                    f"Invalid normalized box: "
                    f"{standard_label}:{line_number}"
                )

            object_count += 1

        paths.append(str(image_path))

        if index % 10_000 == 0:
            print(
                f"{split_name}: "
                f"validated={index}/{len(image_ids)}"
            )

    return paths, object_count, digest.hexdigest()

def build_configs(
        temporary_dir: Path,
) -> dict:
    source_config = yaml.safe_load(
        SOURCE_YOLO_YAML.read_text(
            encoding="utf-8"
        )
    )

    standard_config = copy.deepcopy(
        source_config
    )
    standard_config["nc"] = 10
    standard_config["scale"] = "m"

    original_head = standard_config["head"]

    if original_head[-1][2] != "Detect":
        raise AssertionError(
            "Unexpected YOLO11 detection head"
        )

    p2_config = copy.deepcopy(
        standard_config
    )


    p2_config["head"] = (
            copy.deepcopy(original_head[:-1])
            + [
                [16, 1, "nn.Upsample", [None, 2, "nearest"]],
                [[-1, 2], 1, "Concat", [1]],
                [-1, 2, "C3k2", [128, False]],
                [[25, 16, 19, 22], 1, "Detect", ["nc"]],
            ]
    )

    standard_path = (
            temporary_dir
            / "yolo11m_standard_gate.yaml"
    )
    p2_path = (
            temporary_dir
            / "yolo11m_p2_gate.yaml"
    )

    standard_path.write_text(
        yaml.safe_dump(
            standard_config,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    p2_path.write_text(
        yaml.safe_dump(
            p2_config,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    torch.manual_seed(SEED)

    standard_model = YOLO(
        str(standard_path),
        task="detect",
    )
    p2_model = YOLO(
        str(p2_path),
        task="detect",
    )

    standard_strides = [
        int(value)
        for value in standard_model.model.stride.tolist()
    ]
    p2_strides = [
        int(value)
        for value in p2_model.model.stride.tolist()
    ]

    if standard_strides != [8, 16, 32]:
        raise AssertionError(
            f"Unexpected standard strides: "
            f"{standard_strides}"
        )

    if p2_strides != [4, 8, 16, 32]:
        raise AssertionError(
            f"Unexpected P2 strides: {p2_strides}"
        )

    standard_parameters = sum(
        parameter.numel()
        for parameter in standard_model.model.parameters()
    )
    p2_parameters = sum(
        parameter.numel()
        for parameter in p2_model.model.parameters()
    )

    return {
        "standard_config": standard_path.name,
        "p2_config": p2_path.name,
        "standard_strides": standard_strides,
        "p2_strides": p2_strides,
        "standard_parameters": standard_parameters,
        "p2_parameters": p2_parameters,
        "parameter_increase_fraction": (
                p2_parameters / standard_parameters - 1
        ),
    }

def main() -> None:
    print("=== VERIFYING FROZEN INPUTS ===")

    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file():
            raise FileNotFoundError(path)

        actual = sha256(path)

        print(path)
        print(" expected:", expected)
        print(" actual  :", actual)

        if actual != expected:
            raise AssertionError(
                f"SHA256 mismatch: {path}"
            )

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {OUTPUT_DIR}"
        )

    OUTPUT_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    core_ids = load_image_ids(CORE_IDS_PATH)
    dev_ids = load_image_ids(DEV_IDS_PATH)

    if len(core_ids) != 65_000:
        raise AssertionError(len(core_ids))

    if len(dev_ids) != 5_000:
        raise AssertionError(len(dev_ids))

    if core_ids & dev_ids:
        raise AssertionError(
            "train-core and train-dev overlap"
        )

    print("=== LOADING TRAIN COCO ===")

    coco = json.loads(
        TRAIN_COCO.read_text(encoding="utf-8")
    )

    image_by_id = {
        int(image["id"]): image
        for image in coco["images"]
    }

    all_image_ids = set(image_by_id)

    if core_ids | dev_ids != all_image_ids:
        raise AssertionError(
            "Frozen split does not equal COCO train images"
        )

    if len(coco["annotations"]) != 1_286_852:
        raise AssertionError(
            len(coco["annotations"])
        )

    with tempfile.TemporaryDirectory(
            prefix="prepared_inputs_v1.incomplete-",
            dir=OUTPUT_DIR.parent,
    ) as temporary_name:
        temporary_dir = Path(temporary_name)

        print("=== VALIDATING TRAIN-CORE ===")
        core_paths, core_objects, core_label_hash = (
            validate_split(
                "train_core",
                core_ids,
                image_by_id,
            )
        )

        print("=== VALIDATING TRAIN-DEV ===")
        dev_paths, dev_objects, dev_label_hash = (
            validate_split(
                "train_dev",
                dev_ids,
                image_by_id,
            )
        )

        if core_objects != 1_194_459:
            raise AssertionError(core_objects)

        if dev_objects != 92_393:
            raise AssertionError(dev_objects)

        if core_objects + dev_objects != 1_286_852:
            raise AssertionError(
                core_objects + dev_objects
            )

        core_list_path = (
                temporary_dir / "train_core.txt"
        )
        dev_list_path = (
                temporary_dir / "train_dev.txt"
        )

        core_list_path.write_text(
            "\n".join(core_paths) + "\n",
            encoding="utf-8",
        )
        dev_list_path.write_text(
            "\n".join(dev_paths) + "\n",
            encoding="utf-8",
        )

        dev_images = [
            image
            for image in coco["images"]
            if int(image["id"]) in dev_ids
        ]
        dev_annotations = [
            annotation
            for annotation in coco["annotations"]
            if int(annotation["image_id"]) in dev_ids
        ]

        if len(dev_images) != 5_000:
            raise AssertionError(len(dev_images))

        if len(dev_annotations) != 92_393:
            raise AssertionError(
                len(dev_annotations)
            )

        dev_coco = {
            key: copy.deepcopy(value)
            for key, value in coco.items()
            if key not in {
                "images",
                "annotations",
            }
        }
        dev_coco["images"] = dev_images
        dev_coco["annotations"] = dev_annotations

        dev_coco_path = (
                temporary_dir
                / "instances_train_dev.json"
        )
        write_json(
            dev_coco_path,
            dev_coco,
        )

        data_yaml_path = (
                temporary_dir
                / "bdd100k_architecture_gate.yaml"
        )
        data_yaml_path.write_text(
            yaml.safe_dump(
                {
                    "path": str(DATASET),
                    "train": str(core_list_path),
                    "val": str(dev_list_path),
                    "names": NAMES,
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

        architecture = build_configs(
            temporary_dir
        )

        gate_rules = {
            "version": "architecture_gate_v1",
            "seed": SEED,
            "train_images": 65_000,
            "dev_images": 5_000,
            "epochs": 20,
            "image_size": 960,
            "primary_metric": "COCO_AP_small",
            "pass_rules": {
                "delta_AP_small_min": 0.010,
                "delta_AP_min": -0.002,
                "delta_AP_medium_min": -0.003,
                "delta_AP_large_min": -0.005,
                "delta_AP75_min": -0.003,
                "inference_cost_increase_max": 0.30,
            },
            "borderline_AP_small": [
                0.006,
                0.010,
            ],
            "borderline_action": (
                "Extend both architectures to 30 epochs "
                "under the same protocol."
            ),
            "formal_initialization": {
                "path": str(OFFICIAL_WEIGHT),
                "sha256": sha256(OFFICIAL_WEIGHT),
                "historical_bdd_best_used": False,
            },
            "architecture": architecture,
            "inputs": {
                "train_coco": str(TRAIN_COCO),
                "train_coco_sha256": sha256(TRAIN_COCO),
                "core_split_sha256": sha256(
                    CORE_IDS_PATH
                ),
                "dev_split_sha256": sha256(
                    DEV_IDS_PATH
                ),
                "core_label_tree_sha256": (
                    core_label_hash
                ),
                "dev_label_tree_sha256": (
                    dev_label_hash
                ),
            },
        }

        write_json(
            temporary_dir / "gate_rules.json",
            gate_rules,
        )

        builder_path = Path(__file__).resolve()

        metadata = {
            "builder_script": str(builder_path),
            "builder_script_sha256": sha256(
                builder_path
            ),
            "core_images": len(core_paths),
            "dev_images": len(dev_paths),
            "core_objects": core_objects,
            "dev_objects": dev_objects,
            "architecture": architecture,
        }

        write_json(
            temporary_dir / "metadata.json",
            metadata,
        )

        artifact_files = sorted(
            path
            for path in temporary_dir.iterdir()
            if path.is_file()
        )

        manifest = {
            "version": "prepared_inputs_v1",
            "files": {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in artifact_files
            },
        }

        write_json(
            temporary_dir / "artifact_manifest.json",
            manifest,
        )

        os.replace(
            temporary_dir,
            OUTPUT_DIR,
        )

    print("\n=== ARCHITECTURE-GATE INPUTS COMPLETE ===")
    print("output:", OUTPUT_DIR)
    print("core_images:", 65_000)
    print("dev_images:", 5_000)
    print("core_objects:", 1_194_459)
    print("dev_objects:", 92_393)
    print(
        "manifest_sha256:",
        sha256(
            OUTPUT_DIR
            / "artifact_manifest.json"
        ),
    )
    print(
        "standard_config:",
        OUTPUT_DIR
        / "yolo11m_standard_gate.yaml",
    )
    print(
        "p2_config:",
        OUTPUT_DIR
        / "yolo11m_p2_gate.yaml",
    )
    print(
        "PASS: manifests, labels, split and "
        "P2 strides are valid"
    )

if __name__ == "__main__":
    main()
