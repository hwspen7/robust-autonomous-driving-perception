from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import os
import random
import shutil
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 20260809
EXCLUDED_ANNOTATION_ID = 1005741
FORMAL_SLOTS_PER_ARM = 19_500
MINIMUM_FREE_SHM_GIB = 12.0

BDD_ROOT = Path(
    "/root/rivermind-data/autodrive/datasets/datasets/bdd100k_final"
)
TRAIN_COCO = BDD_ROOT / "coco/annotations/instances_train.json"

RISK_ROOT = Path(
    "/root/rivermind-data/autodrive/results/evaluation/"
    "train_risk_labels/object_risk_v1"
)
RISK_TABLE = RISK_ROOT / "object_risk_table.csv"
RISK_FREEZE = Path(
    "/root/rivermind-data/autodrive/results/evaluation/"
    "train_risk_labels/frozen_manifests/object_risk_v1_freeze.json"
)
DUPLICATE_EXCEPTION = Path(
    "/root/rivermind-data/autodrive/results/evaluation/"
    "architecture_gate/data_quality_v1/"
    "duplicate_annotation_exception.json"
)

SHM_OUTPUT = Path(
    "/dev/shm/rebu_yolo/object_view_smoke_v1"
)
PERSISTENT_OUTPUT = Path(
    "/root/rivermind-data/autodrive/results/evaluation/"
    "method_gate/object_view_smoke_v1"
)

EXPECTED_HASHES = {
    RISK_TABLE: (
        "83e03b29ff5b796257fed1a3fb284fc22597e6e748315169e25fa481f9b1ba38"
    ),
    RISK_FREEZE: (
        "b5e1934ee9108e29548288e6f98cc611bb424c2f6d5b0ab99329b490699ed9d6"
    ),
    DUPLICATE_EXCEPTION: (
        "62fa95492d68d2e790a20afc489747acf309f300afe425b816717c491196ec85"
    ),
}


SMOKE_QUOTAS = {
    "small_joint_not_detected": 80,
    "dfine_supported_yolo_failure": 40,
    "yolo_low_confidence": 20,
    "small_high_or_critical": 20,
    "yolo_geometric_failure": 16,
    "yolo_classification_failure": 10,
    "other_joint_not_detected": 10,
    "other_high_or_critical": 4,
}



CROP_POLICIES = {
    "small_joint_not_detected": (144, 400, 9.0, 16.0),
    "dfine_supported_yolo_failure": (192, 560, 7.0, 14.0),
    "yolo_low_confidence": (192, 560, 8.0, 14.0),
    "small_high_or_critical": (144, 400, 9.0, 16.0),
    "yolo_geometric_failure": (256, 640, 6.0, 12.0),
    "yolo_classification_failure": (288, 680, 7.0, 12.0),
    "other_joint_not_detected": (256, 640, 6.0, 12.0),
    "other_high_or_critical": (192, 560, 7.0, 13.0),
}

REQUIRED_RISK_COLUMNS = {
    "annotation_id",
    "image_id",
    "split",
    "category_id",
    "scale",
    "yolo_status",
    "yolo_detected",
    "dfine_detected",
    "risk_score",
    "risk_tier",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def gib(value: int | float) -> float:
    return float(value) / (1024 ** 3)


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {
        "1", "true", "yes", "y"
    }


def stable_rank(group: str, annotation_id: int) -> int:
    payload = f"{SEED}:{group}:{annotation_id}".encode()
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        byteorder="big",
        signed=False,
    )


def primary_failure_group(row: dict[str, str]) -> str | None:
    small = row["scale"].strip().lower() == "small"
    tier = row["risk_tier"].strip().lower()
    yolo_status = row["yolo_status"].strip().lower()
    yolo_detected = as_bool(row["yolo_detected"])
    dfine_detected = as_bool(row["dfine_detected"])

    if small and not yolo_detected and not dfine_detected:
        return "small_joint_not_detected"

    if dfine_detected and not yolo_detected:
        return "dfine_supported_yolo_failure"

    if yolo_status == "low_confidence_candidate":
        return "yolo_low_confidence"

    if small and tier in {"high", "critical"}:
        return "small_high_or_critical"

    if yolo_status in {
        "localization_error_candidate",
        "assignment_conflict_candidate",
    }:
        return "yolo_geometric_failure"

    if yolo_status == "classification_error_candidate":
        return "yolo_classification_failure"

    if not yolo_detected and not dfine_detected:
        return "other_joint_not_detected"

    if tier in {"high", "critical"}:
        return "other_high_or_critical"

    return None


def choose_crop_window(
    *,
    image_width: int,
    image_height: int,
    bbox: list[float],
    group: str,
    annotation_id: int,
) -> tuple[int, int, int, int]:
    x, y, width, height = map(float, bbox)
    min_height, max_height, factor_low, factor_high = (
        CROP_POLICIES[group]
    )

    rng = random.Random(
        SEED * 1_000_003 + annotation_id
    )
    factor = rng.uniform(factor_low, factor_high)

    crop_height = max(
        float(min_height),
        height * factor,
        width * factor * 9.0 / 16.0,
    )
    crop_height = min(
        crop_height,
        float(max_height),
        float(image_height),
    )
    crop_width = crop_height * 16.0 / 9.0

    if crop_width > image_width:
        crop_width = float(image_width)
        crop_height = min(
            float(image_height),
            crop_width * 9.0 / 16.0,
        )

    crop_width = max(
        min(float(image_width), crop_width),
        min(float(image_width), width),
    )
    crop_height = max(
        min(float(image_height), crop_height),
        min(float(image_height), height),
    )

    crop_width_i = max(1, int(round(crop_width)))
    crop_height_i = max(1, int(round(crop_height)))

    crop_width_i = min(crop_width_i, image_width)
    crop_height_i = min(crop_height_i, image_height)

    target_cx = x + width / 2.0
    target_cy = y + height / 2.0

    jitter_x = rng.uniform(-0.08, 0.08) * crop_width_i
    jitter_y = rng.uniform(-0.08, 0.08) * crop_height_i

    desired_x0 = target_cx - crop_width_i / 2.0 + jitter_x
    desired_y0 = target_cy - crop_height_i / 2.0 + jitter_y

    feasible_x_low = max(
        0.0,
        x + width - crop_width_i,
    )
    feasible_x_high = min(
        x,
        float(image_width - crop_width_i),
    )
    feasible_y_low = max(
        0.0,
        y + height - crop_height_i,
    )
    feasible_y_high = min(
        y,
        float(image_height - crop_height_i),
    )

    if feasible_x_low <= feasible_x_high:
        x0 = min(
            max(desired_x0, feasible_x_low),
            feasible_x_high,
        )
    else:
        x0 = min(
            max(desired_x0, 0.0),
            float(image_width - crop_width_i),
        )

    if feasible_y_low <= feasible_y_high:
        y0 = min(
            max(desired_y0, feasible_y_low),
            feasible_y_high,
        )
    else:
        y0 = min(
            max(desired_y0, 0.0),
            float(image_height - crop_height_i),
        )

    x0_i = int(round(x0))
    y0_i = int(round(y0))

    x0_i = min(
        max(0, x0_i),
        image_width - crop_width_i,
    )
    y0_i = min(
        max(0, y0_i),
        image_height - crop_height_i,
    )

    return (
        x0_i,
        y0_i,
        x0_i + crop_width_i,
        y0_i + crop_height_i,
    )


def transformed_labels(
    *,
    annotations: list[dict],
    crop_box: tuple[int, int, int, int],
    target_annotation_id: int,
) -> tuple[list[str], int, float]:
    x0, y0, x1, y1 = crop_box
    crop_width = x1 - x0
    crop_height = y1 - y0

    rows: list[str] = []
    target_rows = 0
    target_projected_short_side = 0.0

    for annotation in annotations:
        if int(annotation.get("iscrowd", 0)) != 0:
            continue

        category_id = int(annotation["category_id"])
        if category_id < 1 or category_id > 10:
            continue

        bx, by, bw, bh = map(
            float,
            annotation["bbox"],
        )
        if bw <= 0.0 or bh <= 0.0:
            continue

        ix0 = max(bx, float(x0))
        iy0 = max(by, float(y0))
        ix1 = min(bx + bw, float(x1))
        iy1 = min(by + bh, float(y1))

        iw = max(0.0, ix1 - ix0)
        ih = max(0.0, iy1 - iy0)
        intersection_area = iw * ih
        original_area = bw * bh

        annotation_id = int(annotation["id"])
        is_target = annotation_id == target_annotation_id

        center_inside = (
            x0 <= bx + bw / 2.0 <= x1
            and y0 <= by + bh / 2.0 <= y1
        )
        visible_fraction = (
            intersection_area / original_area
            if original_area > 0.0
            else 0.0
        )

        if not is_target:
            if (
                not center_inside
                or visible_fraction < 0.50
                or iw < 2.0
                or ih < 2.0
            ):
                continue

        if iw <= 0.0 or ih <= 0.0:
            continue

        relative_x = ix0 - x0
        relative_y = iy0 - y0

        center_x = (
            relative_x + iw / 2.0
        ) / crop_width
        center_y = (
            relative_y + ih / 2.0
        ) / crop_height
        normalized_width = iw / crop_width
        normalized_height = ih / crop_height

        values = (
            center_x,
            center_y,
            normalized_width,
            normalized_height,
        )
        if not all(
            math.isfinite(value) for value in values
        ):
            continue

        if not all(
            0.0 <= value <= 1.0 for value in values
        ):
            raise AssertionError(
                (annotation_id, values, crop_box)
            )

        class_id = category_id - 1
        rows.append(
            f"{class_id} "
            f"{center_x:.8f} "
            f"{center_y:.8f} "
            f"{normalized_width:.8f} "
            f"{normalized_height:.8f}"
        )

        if is_target:
            target_rows += 1
            letterbox_scale = min(
                960.0 / crop_width,
                960.0 / crop_height,
            )
            target_projected_short_side = (
                min(iw, ih) * letterbox_scale
            )

    return (
        rows,
        target_rows,
        target_projected_short_side,
    )


print("=== VERIFYING FROZEN INPUTS ===")
for path, expected in EXPECTED_HASHES.items():
    if not path.is_file():
        raise FileNotFoundError(path)

    actual = sha256_file(path)
    print(path)
    print(" expected:", expected)
    print(" actual  :", actual)

    if actual != expected:
        raise AssertionError(
            f"SHA256 mismatch: {path}"
        )

exception_text = DUPLICATE_EXCEPTION.read_text()
if str(EXCLUDED_ANNOTATION_ID) not in exception_text:
    raise AssertionError(
        "Frozen duplicate annotation ID not found"
    )

shm_usage = shutil.disk_usage("/dev/shm")
print()
print("=== MEMORY-DISK STATE ===")
print(f"shm_free_GiB: {gib(shm_usage.free):.3f}")

if gib(shm_usage.free) < MINIMUM_FREE_SHM_GIB:
    raise RuntimeError(
        "Insufficient /dev/shm free capacity"
    )

for formal_path in (
    SHM_OUTPUT,
    PERSISTENT_OUTPUT,
):
    if formal_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {formal_path}"
        )

run_token = f"{os.getpid()}"
shm_incomplete = Path(
    str(SHM_OUTPUT) + f".incomplete-{run_token}"
)
persistent_incomplete = Path(
    str(PERSISTENT_OUTPUT) + f".incomplete-{run_token}"
)

for incomplete in (
    shm_incomplete,
    persistent_incomplete,
):
    if incomplete.exists():
        raise FileExistsError(incomplete)

print()
print("=== SELECTING REPRESENTATIVE CORE OBJECTS ===")

heaps: dict[str, list[tuple[int, int, float]]] = {
    group: []
    for group in SMOKE_QUOTAS
}
eligible_counts = Counter()

with RISK_TABLE.open(
    "r",
    encoding="utf-8",
    newline="",
) as handle:
    reader = csv.DictReader(handle)

    if reader.fieldnames is None:
        raise AssertionError("Missing CSV header")

    missing_columns = (
        REQUIRED_RISK_COLUMNS - set(reader.fieldnames)
    )
    if missing_columns:
        raise AssertionError(
            sorted(missing_columns)
        )

    for index, row in enumerate(reader, start=1):
        if row["split"] != "train_core":
            continue

        annotation_id = int(row["annotation_id"])
        if annotation_id == EXCLUDED_ANNOTATION_ID:
            continue

        group = primary_failure_group(row)
        if group is None:
            continue

        eligible_counts[group] += 1
        quota = SMOKE_QUOTAS[group]
        rank = stable_rank(
            group,
            annotation_id,
        )
        risk_score = float(row["risk_score"])

        item = (
            -rank,
            annotation_id,
            risk_score,
        )
        heap = heaps[group]

        if len(heap) < quota:
            heapq.heappush(heap, item)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, item)

        if index % 200_000 == 0:
            print(f"risk_rows_scanned={index}")

selected: list[dict] = []

for group, quota in SMOKE_QUOTAS.items():
    heap = heaps[group]
    if len(heap) != quota:
        raise AssertionError(
            f"{group}: selected={len(heap)} "
            f"expected={quota}"
        )

    ordered = sorted(
        heap,
        key=lambda item: -item[0],
    )

    for negative_rank, annotation_id, risk_score in ordered:
        selected.append({
            "group": group,
            "annotation_id": annotation_id,
            "risk_score": risk_score,
            "selection_rank": -negative_rank,
        })

print("eligible_counts:", dict(eligible_counts))
print("selected_counts:", dict(Counter(
    item["group"] for item in selected
)))
print("selected_total:", len(selected))

if len(selected) != sum(SMOKE_QUOTAS.values()):
    raise AssertionError(len(selected))

if len({
    item["annotation_id"] for item in selected
}) != len(selected):
    raise AssertionError(
        "Duplicate selected annotation IDs"
    )

print()
print("=== LOADING TRAIN COCO ===")
train_coco_sha256 = sha256_file(TRAIN_COCO)
coco = json.loads(TRAIN_COCO.read_text())

if len(coco["images"]) != 70_000:
    raise AssertionError(len(coco["images"]))
if len(coco["annotations"]) != 1_286_852:
    raise AssertionError(len(coco["annotations"]))

image_by_id = {
    int(image["id"]): image
    for image in coco["images"]
}
annotation_by_id = {
    int(annotation["id"]): annotation
    for annotation in coco["annotations"]
}
annotations_by_image: dict[int, list[dict]] = defaultdict(list)

for annotation in coco["annotations"]:
    annotations_by_image[
        int(annotation["image_id"])
    ].append(annotation)

print("train_coco_sha256:", train_coco_sha256)
print("images:", len(image_by_id))
print("annotations:", len(annotation_by_id))

shm_image_dir = shm_incomplete / "images"
shm_label_dir = shm_incomplete / "labels"
shm_image_dir.mkdir(parents=True)
shm_label_dir.mkdir(parents=True)
persistent_incomplete.mkdir(parents=True)

print()
print("=== GENERATING 200 RAM-DISK VIEWS ===")

samples = []
payload_bytes = 0
target_pixel_sizes = []
objects_per_view = []
class_counts = Counter()
group_counts = Counter()
rolling_digest = hashlib.sha256()

for index, selection in enumerate(selected, start=1):
    annotation_id = int(
        selection["annotation_id"]
    )
    group = selection["group"]

    target_annotation = annotation_by_id.get(
        annotation_id
    )
    if target_annotation is None:
        raise KeyError(annotation_id)

    image_id = int(
        target_annotation["image_id"]
    )
    image_record = image_by_id[image_id]

    source_path = (
        BDD_ROOT / image_record["file_name"]
    )
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    image_width = int(image_record["width"])
    image_height = int(image_record["height"])

    crop_box = choose_crop_window(
        image_width=image_width,
        image_height=image_height,
        bbox=target_annotation["bbox"],
        group=group,
        annotation_id=annotation_id,
    )

    output_stem = (
        f"{index:04d}_{group}_{annotation_id}"
    )
    image_output = (
        shm_image_dir / f"{output_stem}.jpg"
    )
    label_output = (
        shm_label_dir / f"{output_stem}.txt"
    )

    with Image.open(source_path) as source_image:
        source_image = source_image.convert("RGB")

        if source_image.size != (
            image_width,
            image_height,
        ):
            raise AssertionError(
                (
                    source_path,
                    source_image.size,
                    (image_width, image_height),
                )
            )

        cropped = source_image.crop(crop_box)
        cropped.save(
            image_output,
            format="JPEG",
            quality=90,
            subsampling=2,
            optimize=False,
        )

    (
        label_rows,
        target_rows,
        projected_short_side,
    ) = transformed_labels(
        annotations=annotations_by_image[image_id],
        crop_box=crop_box,
        target_annotation_id=annotation_id,
    )

    if target_rows != 1:
        raise AssertionError(
            (
                annotation_id,
                target_rows,
                crop_box,
            )
        )
    if not label_rows:
        raise AssertionError(
            f"No labels: {annotation_id}"
        )

    label_output.write_text(
        "\n".join(label_rows) + "\n",
        encoding="utf-8",
    )

    image_sha256 = sha256_file(image_output)
    label_sha256 = sha256_file(label_output)

    image_bytes = image_output.stat().st_size
    label_bytes = label_output.stat().st_size
    payload_bytes += image_bytes + label_bytes

    rolling_digest.update(
        output_stem.encode()
    )
    rolling_digest.update(
        image_sha256.encode()
    )
    rolling_digest.update(
        label_sha256.encode()
    )

    target_pixel_sizes.append(
        projected_short_side
    )
    objects_per_view.append(
        len(label_rows)
    )
    group_counts[group] += 1

    for row in label_rows:
        class_counts[int(row.split()[0])] += 1

    x0, y0, x1, y1 = crop_box

    samples.append({
        **selection,
        "image_id": image_id,
        "source_file": image_record["file_name"],
        "target_category_id": int(
            target_annotation["category_id"]
        ),
        "crop_box_xyxy": [
            x0, y0, x1, y1
        ],
        "crop_width": x1 - x0,
        "crop_height": y1 - y0,
        "objects_in_view": len(label_rows),
        "target_projected_short_side_at_960": (
            projected_short_side
        ),
        "ephemeral_image": (
            f"images/{image_output.name}"
        ),
        "ephemeral_label": (
            f"labels/{label_output.name}"
        ),
        "image_bytes": image_bytes,
        "label_bytes": label_bytes,
        "image_sha256": image_sha256,
        "label_sha256": label_sha256,
    })

    if index % 25 == 0:
        print(
            f"generated={index}/{len(selected)}"
        )

average_payload_bytes = (
    payload_bytes / len(samples)
)
estimated_one_arm_bytes = (
    average_payload_bytes
    * FORMAL_SLOTS_PER_ARM
    * 1.10
)
estimated_two_arms_bytes = (
    estimated_one_arm_bytes * 2
)

sorted_pixels = sorted(target_pixel_sizes)

def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Empty values")

    position = (len(values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))

    if lower == upper:
        return values[lower]

    weight = position - lower
    return (
        values[lower] * (1.0 - weight)
        + values[upper] * weight
    )

summary = {
    "version": "object_view_smoke_v1",
    "status": "technical_smoke_not_formal_protocol",
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "seed": SEED,
    "script_sha256": sha256_file(
        Path(__file__)
    ),
    "inputs": {
        "train_coco": str(TRAIN_COCO),
        "train_coco_sha256": train_coco_sha256,
        "risk_table": str(RISK_TABLE),
        "risk_table_sha256": EXPECTED_HASHES[
            RISK_TABLE
        ],
        "risk_freeze_sha256": EXPECTED_HASHES[
            RISK_FREEZE
        ],
        "duplicate_exception_sha256": (
            EXPECTED_HASHES[DUPLICATE_EXCEPTION]
        ),
        "excluded_annotation_id": (
            EXCLUDED_ANNOTATION_ID
        ),
    },
    "smoke": {
        "views": len(samples),
        "quotas": SMOKE_QUOTAS,
        "group_counts": dict(group_counts),
        "payload_bytes": payload_bytes,
        "payload_MiB": (
            payload_bytes / (1024 ** 2)
        ),
        "average_payload_KiB": (
            average_payload_bytes / 1024
        ),
        "ephemeral_set_sha256": (
            rolling_digest.hexdigest()
        ),
    },
    "geometry": {
        "target_projected_short_side_at_960": {
            "minimum": min(target_pixel_sizes),
            "p10": percentile(
                sorted_pixels, 0.10
            ),
            "median": statistics.median(
                target_pixel_sizes
            ),
            "p90": percentile(
                sorted_pixels, 0.90
            ),
            "maximum": max(target_pixel_sizes),
            "below_8px": sum(
                value < 8.0
                for value in target_pixel_sizes
            ),
            "below_16px": sum(
                value < 16.0
                for value in target_pixel_sizes
            ),
        },
        "objects_per_view": {
            "minimum": min(objects_per_view),
            "median": statistics.median(
                objects_per_view
            ),
            "maximum": max(objects_per_view),
            "mean": statistics.mean(
                objects_per_view
            ),
        },
        "label_class_counts_0_to_9": {
            str(class_id): class_counts[class_id]
            for class_id in range(10)
        },
    },
    "capacity_estimate": {
        "formal_slots_per_arm": (
            FORMAL_SLOTS_PER_ARM
        ),
        "estimated_one_arm_GiB_with_10pct_margin": (
            gib(estimated_one_arm_bytes)
        ),
        "estimated_two_arms_GiB_with_10pct_margin": (
            gib(estimated_two_arms_bytes)
        ),
        "execution_plan": (
            "materialize and train one arm at a time"
        ),
        "current_shm_free_GiB": (
            gib(shm_usage.free)
        ),
    },
    "ephemeral_output": str(SHM_OUTPUT),
    "persistent_output": str(
        PERSISTENT_OUTPUT
    ),
    "formal_training_rules_frozen": False,
}

samples_path = (
    persistent_incomplete
    / "sample_manifest.json"
)
summary_path = (
    persistent_incomplete
    / "summary.json"
)

samples_path.write_text(
    json.dumps(
        samples,
        indent=2,
        ensure_ascii=False,
    ) + "\n",
    encoding="utf-8",
)
summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ) + "\n",
    encoding="utf-8",
)

artifact_manifest = {
    "version": "object_view_smoke_v1",
    "status": "technical_smoke",
    "files": {
        "sample_manifest.json": {
            "bytes": samples_path.stat().st_size,
            "sha256": sha256_file(samples_path),
        },
        "summary.json": {
            "bytes": summary_path.stat().st_size,
            "sha256": sha256_file(summary_path),
        },
    },
    "ephemeral_set_sha256": (
        rolling_digest.hexdigest()
    ),
}

artifact_path = (
    persistent_incomplete
    / "artifact_manifest.json"
)
artifact_path.write_text(
    json.dumps(
        artifact_manifest,
        indent=2,
        ensure_ascii=False,
    ) + "\n",
    encoding="utf-8",
)

shm_incomplete.rename(SHM_OUTPUT)
persistent_incomplete.rename(
    PERSISTENT_OUTPUT
)

print()
print("=" * 88)
print("OBJECT-CENTRIC VIEW SMOKE COMPLETE")
print("=" * 88)
print("views:", len(samples))
print(
    "group_counts:",
    dict(group_counts),
)
print(
    "payload_MiB:",
    round(payload_bytes / (1024 ** 2), 3),
)
print(
    "average_payload_KiB:",
    round(average_payload_bytes / 1024, 3),
)
print(
    "estimated_one_arm_GiB:",
    round(gib(estimated_one_arm_bytes), 3),
)
print(
    "estimated_two_arms_GiB:",
    round(gib(estimated_two_arms_bytes), 3),
)
print(
    "target_short_side_p10:",
    round(percentile(sorted_pixels, 0.10), 3),
)
print(
    "target_short_side_median:",
    round(statistics.median(target_pixel_sizes), 3),
)
print(
    "target_short_side_p90:",
    round(percentile(sorted_pixels, 0.90), 3),
)
print(
    "objects_per_view_mean:",
    round(statistics.mean(objects_per_view), 3),
)
print("ephemeral_output:", SHM_OUTPUT)
print("persistent_output:", PERSISTENT_OUTPUT)
print(
    "artifact_manifest_sha256:",
    sha256_file(
        PERSISTENT_OUTPUT
        / "artifact_manifest.json"
    ),
)

if gib(estimated_one_arm_bytes) >= 12.0:
    raise RuntimeError(
        "Estimated one-arm payload exceeds "
        "the conservative RAM-disk budget"
    )

print()
print("PASS: crop geometry, transformed labels and RAM capacity are valid")
print("NOTE: final method quotas and augmentation rules are not frozen yet")
