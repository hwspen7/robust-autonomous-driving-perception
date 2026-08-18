from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/root/rivermind-data/autodrive")
DATASET = ROOT / "datasets/datasets/bdd100k_final"
RESULTS = ROOT / "results"

COCO_PATH = (
    DATASET
    / "coco/annotations/instances_train.json"
)

IMAGE_ID = 14125
FILE_NAME = "images/train/23e6e46f-d1ed450d.jpg"

STANDARD_LABEL = (
    DATASET
    / "labels/train/23e6e46f-d1ed450d.txt"
)
DUPLICATE_LABEL = (
    DATASET
    / "yolo/labels/train/23e6e46f-d1ed450d.txt"
)

OUTPUT_DIR = (
    RESULTS
    / "evaluation/architecture_gate/"
    "label_repairs_v1/image_14125"
)

EXPECTED = {
    261470: {
        "category_id": 3,
        "bbox": [204.0, 179.0, 90.0, 54.0],
    },
    261471: {
        "category_id": 3,
        "bbox": [373.0, 209.0, 34.0, 30.0],
    },
    261472: {
        "category_id": 3,
        "bbox": [1156.0, 214.0, 123.0, 42.0],
    },
    261473: {
        "category_id": 10,
        "bbox": [1178.0, 154.0, 21.0, 8.0],
    },
    261474: {
        "category_id": 10,
        "bbox": [1093.0, 174.0, 26.0, 12.0],
    },
    261475: {
        "category_id": 10,
        "bbox": [1096.0, 189.0, 23.0, 25.0],
    },
    261476: {
        "category_id": 3,
        "bbox": [560.0, 199.0, 54.0, 37.0],
    },
    261477: {
        "category_id": 3,
        "bbox": [600.0, 187.0, 38.0, 27.0],
    },
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


def atomic_write(
    path: Path,
    content: bytes,
) -> None:
    temporary = path.with_name(
        f".{path.name}.repair-{os.getpid()}.tmp"
    )

    if temporary.exists():
        raise FileExistsError(temporary)

    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(
            temporary,
            path,
        )
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    print("=== YOLO LABEL REPAIR PREFLIGHT ===")

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite repair record: "
            f"{OUTPUT_DIR}"
        )

    for path in (
        COCO_PATH,
        STANDARD_LABEL,
        DUPLICATE_LABEL,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    standard_before = STANDARD_LABEL.read_bytes()
    duplicate_before = DUPLICATE_LABEL.read_bytes()

    if standard_before.strip():
        raise AssertionError(
            f"Expected empty label: {STANDARD_LABEL}"
        )

    if duplicate_before.strip():
        raise AssertionError(
            f"Expected empty label: {DUPLICATE_LABEL}"
        )

    if standard_before != duplicate_before:
        raise AssertionError(
            "The two empty label files differ"
        )

    print("standard_label:", STANDARD_LABEL)
    print("duplicate_label:", DUPLICATE_LABEL)
    print(
        "before_sha256:",
        hashlib.sha256(
            standard_before
        ).hexdigest(),
    )

    print("=== LOADING SOURCE COCO ===")

    coco = json.loads(
        COCO_PATH.read_text(encoding="utf-8")
    )

    images = {
        int(image["id"]): image
        for image in coco["images"]
    }

    image = images.get(IMAGE_ID)

    if image is None:
        raise KeyError(IMAGE_ID)

    if image["file_name"] != FILE_NAME:
        raise AssertionError(image)

    width = float(image["width"])
    height = float(image["height"])

    if width != 1280.0 or height != 720.0:
        raise AssertionError(
            (width, height)
        )

    image_path = DATASET / FILE_NAME

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    annotations = sorted(
        (
            annotation
            for annotation in coco["annotations"]
            if int(annotation["image_id"])
            == IMAGE_ID
        ),
        key=lambda annotation: int(
            annotation["id"]
        ),
    )

    actual_ids = {
        int(annotation["id"])
        for annotation in annotations
    }

    if actual_ids != set(EXPECTED):
        raise AssertionError(
            {
                "expected": sorted(EXPECTED),
                "actual": sorted(actual_ids),
            }
        )

    if len(annotations) != 8:
        raise AssertionError(
            len(annotations)
        )

    lines = []
    source_annotations = []

    for annotation in annotations:
        annotation_id = int(annotation["id"])
        category_id = int(
            annotation["category_id"]
        )
        bbox = [
            float(value)
            for value in annotation["bbox"]
        ]

        expected = EXPECTED[annotation_id]

        if category_id != expected["category_id"]:
            raise AssertionError(annotation)

        if bbox != expected["bbox"]:
            raise AssertionError(annotation)

        if int(annotation.get("iscrowd", 0)):
            raise AssertionError(annotation)

        x, y, box_width, box_height = bbox

        if box_width <= 0 or box_height <= 0:
            raise AssertionError(annotation)

        if not (
            0 <= x < width
            and 0 <= y < height
            and x + box_width <= width
            and y + box_height <= height
        ):
            raise AssertionError(annotation)


        # COCO category 1..10 -> YOLO class 0..9。
        class_id = category_id - 1

        center_x = (
            x + box_width / 2
        ) / width
        center_y = (
            y + box_height / 2
        ) / height
        normalized_width = box_width / width
        normalized_height = (
            box_height / height
        )

        values = (
            center_x,
            center_y,
            normalized_width,
            normalized_height,
        )

        if not all(
            0 < value <= 1
            for value in values
        ):
            raise AssertionError(
                {
                    "annotation": annotation,
                    "normalized": values,
                }
            )

        line = (
            f"{class_id} "
            f"{center_x:.10f} "
            f"{center_y:.10f} "
            f"{normalized_width:.10f} "
            f"{normalized_height:.10f}"
        )

        lines.append(line)

        source_annotations.append(
            {
                "annotation_id": annotation_id,
                "category_id": category_id,
                "class_id": class_id,
                "bbox": bbox,
                "area": annotation.get("area"),
                "iscrowd": annotation.get(
                    "iscrowd"
                ),
                "yolo_line": line,
            }
        )

    corrected_content = (
        "\n".join(lines) + "\n"
    ).encode("utf-8")

    if len(
        corrected_content.decode(
            "utf-8"
        ).splitlines()
    ) != 8:
        raise AssertionError(
            "Corrected label must contain 8 rows"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    (OUTPUT_DIR / "original_standard_label.txt").write_bytes(
        standard_before
    )
    (OUTPUT_DIR / "original_duplicate_label.txt").write_bytes(
        duplicate_before
    )
    (OUTPUT_DIR / "corrected_label.txt").write_bytes(
        corrected_content
    )

    repair_plan = {
        "status": "prepared",
        "reason": (
            "One derived YOLO label file was empty "
            "while its source COCO image contained "
            "eight valid in-bounds annotations."
        ),
        "source_of_truth": str(COCO_PATH),
        "source_coco_sha256": sha256(
            COCO_PATH
        ),
        "image_id": IMAGE_ID,
        "file_name": FILE_NAME,
        "width": width,
        "height": height,
        "targets": [
            str(STANDARD_LABEL),
            str(DUPLICATE_LABEL),
        ],
        "before": {
            "standard_sha256":
                hashlib.sha256(
                    standard_before
                ).hexdigest(),
            "duplicate_sha256":
                hashlib.sha256(
                    duplicate_before
                ).hexdigest(),
            "rows": 0,
        },
        "after": {
            "sha256":
                hashlib.sha256(
                    corrected_content
                ).hexdigest(),
            "rows": 8,
        },
        "annotations": source_annotations,
    }

    write_json(
        OUTPUT_DIR / "repair_plan.json",
        repair_plan,
    )

    print("=== APPLYING ATOMIC LABEL REPAIR ===")

    atomic_write(
        STANDARD_LABEL,
        corrected_content,
    )
    atomic_write(
        DUPLICATE_LABEL,
        corrected_content,
    )

    standard_after = STANDARD_LABEL.read_bytes()
    duplicate_after = DUPLICATE_LABEL.read_bytes()

    if standard_after != corrected_content:
        raise AssertionError(
            "Standard label verification failed"
        )

    if duplicate_after != corrected_content:
        raise AssertionError(
            "Duplicate label verification failed"
        )

    repair_record = {
        **repair_plan,
        "status": "applied_and_verified",
        "builder_script": str(
            Path(__file__).resolve()
        ),
        "builder_script_sha256": sha256(
            Path(__file__).resolve()
        ),
        "verified_standard_sha256": (
            sha256(STANDARD_LABEL)
        ),
        "verified_duplicate_sha256": (
            sha256(DUPLICATE_LABEL)
        ),
    }

    write_json(
        OUTPUT_DIR / "repair_record.json",
        repair_record,
    )

    artifact_files = sorted(
        path
        for path in OUTPUT_DIR.iterdir()
        if path.is_file()
    )

    manifest = {
        "version": "bdd100k_yolo_label_repair_v1",
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifact_files
        },
    }

    write_json(
        OUTPUT_DIR / "artifact_manifest.json",
        manifest,
    )

    print("\n=== LABEL REPAIR COMPLETE ===")
    print("image_id:", IMAGE_ID)
    print("repaired_rows:", 8)
    print(
        "corrected_sha256:",
        sha256(STANDARD_LABEL),
    )
    print("repair_record:", OUTPUT_DIR)
    print(
        "manifest_sha256:",
        sha256(
            OUTPUT_DIR
            / "artifact_manifest.json"
        ),
    )
    print(
        "PASS: both derived YOLO label trees "
        "now match the source COCO annotations"
    )


if __name__ == "__main__":
    main()