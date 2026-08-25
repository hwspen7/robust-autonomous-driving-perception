from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SEED = 20260809

TOTAL_CORE_IMAGES = 65_000
COMMON_BASE_IMAGES = 45_500
CROP_SLOTS = 19_500
MIN_CLASS_SUPPORT = 500
MAX_TARGET_CLASS_FRACTION = 0.35
RISK_BIN_COUNT = 20
MAX_MATCHED_MEAN_RISK_GAP = 0.03
EXCLUDED_DUPLICATE_ANNOTATION_ID = 1005741

RISK_ROOT = Path(
    "/root/rivermind-data/autodrive/results/evaluation/"
    "train_risk_labels/object_risk_v1"
)
RISK_TABLE = RISK_ROOT / "object_risk_table.csv"
RISK_FREEZE = Path(
    "/root/rivermind-data/autodrive/results/evaluation/"
    "train_risk_labels/frozen_manifests/object_risk_v1_freeze.json"
)
CORE_IDS_PATH = Path(
    "/root/rivermind-data/autodrive/results/evaluation/"
    "train_risk_labels/calibration_split/train_core_image_ids.json"
)
DUPLICATE_EXCEPTION = Path(
    "/root/rivermind-data/autodrive/results/evaluation/"
    "architecture_gate/data_quality_v1/"
    "duplicate_annotation_exception.json"
)

OUTPUT_DIR = Path(
    "/root/rivermind-data/autodrive/results/evaluation/"
    "method_gate/sampling_audit_v3"
)

EXPECTED_HASHES = {
    RISK_TABLE: (
        "83e03b29ff5b796257fed1a3fb284fc22597e6e748315169e25fa481f9b1ba38"
    ),
    RISK_FREEZE: (
        "b5e1934ee9108e29548288e6f98cc611bb424c2f6d5b0ab99329b490699ed9d6"
    ),
    CORE_IDS_PATH: (
        "9996d854b06ddefd12dde1949ecae30c6d08230eb10003e7ec1863a3386ee1a3"
    ),
    DUPLICATE_EXCEPTION: (
        "62fa95492d68d2e790a20afc489747acf309f300afe425b816717c491196ec85"
    ),
}

CATEGORY_NAMES = {
    1: "pedestrian",
    2: "rider",
    3: "car",
    4: "truck",
    5: "bus",
    6: "train",
    7: "motorcycle",
    8: "bicycle",
    9: "traffic light",
    10: "traffic sign",
}

TYPED_QUOTAS = {
    "small_joint_not_detected": 7_800,
    "dfine_supported_yolo_failure": 3_900,
    "yolo_geometric_failure": 1_950,
    "yolo_low_confidence": 1_950,
    "yolo_classification_failure": 1_365,
    "other_joint_not_detected": 975,
    "small_high_or_critical": 975,
    "other_high_or_critical": 585,
}


TYPED_SELECTION_ORDER = [
    "yolo_geometric_failure",
    "yolo_classification_failure",
    "other_joint_not_detected",
    "other_high_or_critical",
    "dfine_supported_yolo_failure",
    "yolo_low_confidence",
    "small_high_or_critical",
    "small_joint_not_detected",
]

REQUIRED_COLUMNS = {
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


@dataclass(frozen=True, slots=True)
class Candidate:
    annotation_id: int
    image_id: int
    category_id: int
    scale: str
    risk_score: float
    risk_tier: str
    yolo_status: str
    yolo_detected: bool
    dfine_detected: bool
    diagnostic_group: str
    priority: float
    weight: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {
        "1", "true", "yes", "y"
    }


def stable_uniform(namespace: str, annotation_id: int) -> float:
    payload = f"{SEED}:{namespace}:{annotation_id}".encode()
    value = int.from_bytes(
        hashlib.sha256(payload).digest()[:16],
        byteorder="big",
        signed=False,
    )
    return (value + 1.0) / ((1 << 128) + 1.0)


def risk_weight(risk_score: float) -> float:
    if not 0.0 <= risk_score <= 1.0:
        raise AssertionError(risk_score)
    return 0.05 + 0.95 * (risk_score ** 2)


def weighted_priority(
    namespace: str,
    annotation_id: int,
    risk_score: float,
) -> tuple[float, float]:
    weight = risk_weight(risk_score)
    uniform = stable_uniform(namespace, annotation_id)
    return -math.log(uniform) / weight, weight


def stable_image_rank(namespace: str, image_id: int) -> int:
    payload = f"{SEED}:{namespace}:{image_id}".encode()
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:16],
        byteorder="big",
        signed=False,
    )


def primary_failure_group(row: dict[str, str]) -> str:
    small = row["scale"].strip().lower() == "small"
    tier = row["risk_tier"].strip().lower()
    status = row["yolo_status"].strip().lower()
    yolo_detected = as_bool(row["yolo_detected"])
    dfine_detected = as_bool(row["dfine_detected"])

    if small and not yolo_detected and not dfine_detected:
        return "small_joint_not_detected"
    if dfine_detected and not yolo_detected:
        return "dfine_supported_yolo_failure"
    if status == "low_confidence_candidate":
        return "yolo_low_confidence"
    if small and tier in {"high", "critical"}:
        return "small_high_or_critical"
    if status in {
        "localization_error_candidate",
        "assignment_conflict_candidate",
    }:
        return "yolo_geometric_failure"
    if status == "classification_error_candidate":
        return "yolo_classification_failure"
    if not yolo_detected and not dfine_detected:
        return "other_joint_not_detected"
    if tier in {"high", "critical"}:
        return "other_high_or_critical"
    return "standard_support"


def parse_core_ids(path: Path) -> list[int]:
    data = json.loads(path.read_text())

    if isinstance(data, list):
        values = data
    elif isinstance(data, dict):
        values = None
        for key in (
            "image_ids",
            "train_core_image_ids",
            "ids",
        ):
            if key in data and isinstance(data[key], list):
                values = data[key]
                break
        if values is None:
            raise KeyError(
                f"Cannot locate image IDs in {path}"
            )
    else:
        raise TypeError(type(data).__name__)

    result = [int(value) for value in values]
    if len(result) != TOTAL_CORE_IMAGES:
        raise AssertionError(len(result))
    if len(set(result)) != TOTAL_CORE_IMAGES:
        raise AssertionError("Duplicate core image IDs")
    return result


def candidate_from_row(
    row: dict[str, str],
    *,
    namespace: str,
    diagnostic_group: str,
) -> Candidate:
    annotation_id = int(row["annotation_id"])
    risk_score = float(row["risk_score"])
    if namespace.startswith("scalar_matched:"):

        priority = stable_uniform(
            namespace,
            annotation_id,
        )
        weight = 1.0
    else:
        priority, weight = weighted_priority(
            namespace,
            annotation_id,
            risk_score,
        )

    return Candidate(
        annotation_id=annotation_id,
        image_id=int(row["image_id"]),
        category_id=int(row["category_id"]),
        scale=row["scale"].strip().lower(),
        risk_score=risk_score,
        risk_tier=row["risk_tier"].strip().lower(),
        yolo_status=row["yolo_status"].strip().lower(),
        yolo_detected=as_bool(row["yolo_detected"]),
        dfine_detected=as_bool(row["dfine_detected"]),
        diagnostic_group=diagnostic_group,
        priority=priority,
        weight=weight,
    )


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    fraction = position - low
    return (
        ordered[low] * (1.0 - fraction)
        + ordered[high] * fraction
    )


def risk_bin_index(risk_score: float) -> int:
    if not 0.0 <= risk_score <= 1.0:
        raise AssertionError(risk_score)

    return min(
        RISK_BIN_COUNT - 1,
        max(0, int(risk_score * RISK_BIN_COUNT)),
    )


def summarize(candidates: list[Candidate]) -> dict:
    risk_values = [
        candidate.risk_score
        for candidate in candidates
    ]
    return {
        "count": len(candidates),
        "unique_annotations": len({
            candidate.annotation_id
            for candidate in candidates
        }),
        "unique_source_images": len({
            candidate.image_id
            for candidate in candidates
        }),
        "target_classes": dict(sorted(Counter(
            candidate.category_id
            for candidate in candidates
        ).items())),
        "scales": dict(sorted(Counter(
            candidate.scale
            for candidate in candidates
        ).items())),
        "risk_tiers": dict(sorted(Counter(
            candidate.risk_tier
            for candidate in candidates
        ).items())),
        "diagnostic_groups": dict(sorted(Counter(
            candidate.diagnostic_group
            for candidate in candidates
        ).items())),
        "yolo_status": dict(sorted(Counter(
            candidate.yolo_status
            for candidate in candidates
        ).items())),
        "risk_score": {
            "mean": statistics.mean(risk_values),
            "p10": percentile(risk_values, 0.10),
            "median": statistics.median(risk_values),
            "p90": percentile(risk_values, 0.90),
        },
    }


def write_selection_csv(
    path: Path,
    *,
    arm: str,
    candidates: list[Candidate],
) -> None:
    fields = [
        "selection_arm",
        "slot_index",
        "annotation_id",
        "image_id",
        "target_category_id",
        "target_category",
        "scale",
        "risk_score",
        "risk_tier",
        "yolo_status",
        "yolo_detected",
        "dfine_detected",
        "diagnostic_group",
        "crop_policy",
        "selection_priority",
        "risk_weight",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()

        for index, candidate in enumerate(
            candidates,
            start=1,
        ):
            crop_policy = (
                candidate.diagnostic_group
                if arm == "typed_cafr"
                else "generic_scalar_context"
            )
            writer.writerow({
                "selection_arm": arm,
                "slot_index": index,
                "annotation_id": candidate.annotation_id,
                "image_id": candidate.image_id,
                "target_category_id": candidate.category_id,
                "target_category": CATEGORY_NAMES[
                    candidate.category_id
                ],
                "scale": candidate.scale,
                "risk_score": (
                    f"{candidate.risk_score:.10f}"
                ),
                "risk_tier": candidate.risk_tier,
                "yolo_status": candidate.yolo_status,
                "yolo_detected": int(
                    candidate.yolo_detected
                ),
                "dfine_detected": int(
                    candidate.dfine_detected
                ),
                "diagnostic_group": (
                    candidate.diagnostic_group
                ),
                "crop_policy": crop_policy,
                "selection_priority": (
                    f"{candidate.priority:.12f}"
                ),
                "risk_weight": (
                    f"{candidate.weight:.12f}"
                ),
            })


print("=== VERIFYING FROZEN INPUTS ===")
for path, expected in EXPECTED_HASHES.items():
    if not path.is_file():
        raise FileNotFoundError(path)

    actual = sha256_file(path)
    print(path)
    print(" expected:", expected)
    print(" actual  :", actual)

    if actual != expected:
        raise AssertionError(path)

if str(EXCLUDED_DUPLICATE_ANNOTATION_ID) not in (
    DUPLICATE_EXCEPTION.read_text()
):
    raise AssertionError(
        "Duplicate exception record mismatch"
    )

if sum(TYPED_QUOTAS.values()) != CROP_SLOTS:
    raise AssertionError(TYPED_QUOTAS)

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Refusing to overwrite: {OUTPUT_DIR}"
    )

incomplete = Path(
    str(OUTPUT_DIR) + f".incomplete-{os.getpid()}"
)
if incomplete.exists():
    raise FileExistsError(incomplete)

core_ids = parse_core_ids(CORE_IDS_PATH)
core_set = set(core_ids)

print()
print("=== PASS 1: CORE CLASS SUPPORT ===")
core_class_counts = Counter()
core_object_count = 0

with RISK_TABLE.open(
    "r",
    encoding="utf-8",
    newline="",
) as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        raise AssertionError("Missing CSV header")

    missing = REQUIRED_COLUMNS - set(reader.fieldnames)
    if missing:
        raise AssertionError(sorted(missing))

    for index, row in enumerate(reader, start=1):
        if row["split"] != "train_core":
            continue

        image_id = int(row["image_id"])
        if image_id not in core_set:
            raise AssertionError(image_id)

        annotation_id = int(row["annotation_id"])
        if annotation_id == EXCLUDED_DUPLICATE_ANNOTATION_ID:
            continue

        category_id = int(row["category_id"])
        core_class_counts[category_id] += 1
        core_object_count += 1

        if index % 250_000 == 0:
            print(f"pass1_rows={index}")

excluded_target_categories = {
    category_id
    for category_id, count in core_class_counts.items()
    if count < MIN_CLASS_SUPPORT
}

print("core_objects_effective:", core_object_count)
print("core_class_counts:", dict(sorted(
    core_class_counts.items()
)))
print(
    "excluded_target_categories:",
    sorted(excluded_target_categories),
)

if excluded_target_categories != {6}:
    raise AssertionError(
        excluded_target_categories
    )

print()
print("=== PASS 2: TYPED-CAFR CANDIDATE POOLS ===")
typed_pools: dict[str, list[Candidate]] = {
    group: []
    for group in TYPED_QUOTAS
}
typed_pool_counts = Counter()

with RISK_TABLE.open(
    "r",
    encoding="utf-8",
    newline="",
) as handle:
    reader = csv.DictReader(handle)

    for index, row in enumerate(reader, start=1):
        if row["split"] != "train_core":
            continue

        annotation_id = int(row["annotation_id"])
        if annotation_id == EXCLUDED_DUPLICATE_ANNOTATION_ID:
            continue

        category_id = int(row["category_id"])
        if category_id in excluded_target_categories:
            continue

        group = primary_failure_group(row)
        if group not in TYPED_QUOTAS:
            continue

        candidate = candidate_from_row(
            row,
            namespace=f"typed:{group}",
            diagnostic_group=group,
        )
        typed_pools[group].append(candidate)
        typed_pool_counts[group] += 1

        if index % 250_000 == 0:
            print(f"pass2_rows={index}")

for group in typed_pools:
    typed_pools[group].sort(
        key=lambda candidate: (
            candidate.priority,
            candidate.annotation_id,
        )
    )

print(
    "typed_pool_counts:",
    dict(typed_pool_counts),
)

class_cap = int(
    math.floor(
        CROP_SLOTS
        * MAX_TARGET_CLASS_FRACTION
    )
)

typed_selected: list[Candidate] = []
typed_used_images: set[int] = set()
typed_class_counts = Counter()
typed_group_counts = Counter()
typed_skip_counts = Counter()

print()
print("=== SELECTING TYPED-CAFR SLOTS ===")

for group in TYPED_SELECTION_ORDER:
    quota = TYPED_QUOTAS[group]

    for candidate in typed_pools[group]:
        if typed_group_counts[group] >= quota:
            break

        if candidate.image_id in typed_used_images:
            typed_skip_counts[
                f"{group}:duplicate_image"
            ] += 1
            continue

        if typed_class_counts[
            candidate.category_id
        ] >= class_cap:
            typed_skip_counts[
                f"{group}:class_cap"
            ] += 1
            continue

        typed_selected.append(candidate)
        typed_used_images.add(candidate.image_id)
        typed_class_counts[
            candidate.category_id
        ] += 1
        typed_group_counts[group] += 1

    if typed_group_counts[group] != quota:
        raise RuntimeError({
            "group": group,
            "selected": typed_group_counts[group],
            "quota": quota,
            "pool": len(typed_pools[group]),
            "class_counts": dict(typed_class_counts),
        })

if len(typed_selected) != CROP_SLOTS:
    raise AssertionError(len(typed_selected))

if len(typed_used_images) != CROP_SLOTS:
    raise AssertionError(
        len(typed_used_images)
    )

print(
    "typed_group_counts:",
    dict(typed_group_counts),
)
print(
    "typed_class_counts:",
    dict(sorted(typed_class_counts.items())),
)
print(
    "typed_skip_counts:",
    dict(typed_skip_counts),
)


del typed_pools

print()
print("=== PASS 3: MATCHED SCALAR-RISK POOLS ===")

typed_stratum_counts = Counter(
    (
        candidate.category_id,
        candidate.scale,
        risk_bin_index(candidate.risk_score),
    )
    for candidate in typed_selected
)

typed_scale_counts = Counter(
    candidate.scale
    for candidate in typed_selected
)
typed_risk_bin_counts = Counter(
    risk_bin_index(candidate.risk_score)
    for candidate in typed_selected
)

scalar_pools: dict[
    tuple[int, str, int],
    list[Candidate],
] = {
    stratum: []
    for stratum in typed_stratum_counts
}

with RISK_TABLE.open(
    "r",
    encoding="utf-8",
    newline="",
) as handle:
    reader = csv.DictReader(handle)

    for index, row in enumerate(reader, start=1):
        if row["split"] != "train_core":
            continue

        annotation_id = int(row["annotation_id"])
        if annotation_id == EXCLUDED_DUPLICATE_ANNOTATION_ID:
            continue

        category_id = int(row["category_id"])
        if category_id in excluded_target_categories:
            continue

        scale = row["scale"].strip().lower()
        risk_score = float(row["risk_score"])
        risk_bin = risk_bin_index(risk_score)

        stratum = (
            category_id,
            scale,
            risk_bin,
        )
        if stratum not in scalar_pools:
            continue

        group = primary_failure_group(row)
        namespace = (
            f"scalar_matched:"
            f"{category_id}:{scale}:{risk_bin}"
        )

        candidate = candidate_from_row(
            row,
            namespace=namespace,
            diagnostic_group=group,
        )
        scalar_pools[stratum].append(candidate)

        if index % 250_000 == 0:
            print(f"pass3_rows={index}")

for stratum in scalar_pools:
    scalar_pools[stratum].sort(
        key=lambda candidate: (
            candidate.priority,
            candidate.annotation_id,
        )
    )


scalar_stratum_order = sorted(
    typed_stratum_counts,
    key=lambda stratum: (
        len(scalar_pools[stratum])
        / typed_stratum_counts[stratum],
        stratum,
    ),
)

scalar_selected: list[Candidate] = []
scalar_used_images: set[int] = set()
scalar_stratum_counts = Counter()
scalar_skip_counts = Counter()

print()
print("=== SELECTING MATCHED SCALAR-RISK SLOTS ===")
print(
    "matched_strata:",
    len(scalar_stratum_order),
)

for stratum in scalar_stratum_order:
    required = typed_stratum_counts[stratum]

    for candidate in scalar_pools[stratum]:
        if scalar_stratum_counts[stratum] >= required:
            break

        if candidate.image_id in scalar_used_images:
            scalar_skip_counts[
                "duplicate_source_image"
            ] += 1
            continue

        scalar_selected.append(candidate)
        scalar_used_images.add(candidate.image_id)
        scalar_stratum_counts[stratum] += 1

    if scalar_stratum_counts[stratum] != required:
        raise RuntimeError({
            "stratum": stratum,
            "selected": scalar_stratum_counts[stratum],
            "required": required,
            "pool": len(scalar_pools[stratum]),
        })

if len(scalar_selected) != CROP_SLOTS:
    raise AssertionError(len(scalar_selected))

if len(scalar_used_images) != CROP_SLOTS:
    raise AssertionError(
        len(scalar_used_images)
    )

if scalar_stratum_counts != typed_stratum_counts:
    raise AssertionError(
        "Joint category/scale/risk-bin mismatch"
    )

scalar_class_counts = Counter(
    candidate.category_id
    for candidate in scalar_selected
)
scalar_scale_counts = Counter(
    candidate.scale
    for candidate in scalar_selected
)
scalar_risk_bin_counts = Counter(
    risk_bin_index(candidate.risk_score)
    for candidate in scalar_selected
)

if scalar_class_counts != typed_class_counts:
    raise AssertionError({
        "scalar_classes": dict(scalar_class_counts),
        "typed_classes": dict(typed_class_counts),
    })

if scalar_scale_counts != typed_scale_counts:
    raise AssertionError({
        "scalar_scales": dict(scalar_scale_counts),
        "typed_scales": dict(typed_scale_counts),
    })

if scalar_risk_bin_counts != typed_risk_bin_counts:
    raise AssertionError({
        "scalar_risk_bins": dict(scalar_risk_bin_counts),
        "typed_risk_bins": dict(typed_risk_bin_counts),
    })

print(
    "scalar_class_counts:",
    dict(sorted(scalar_class_counts.items())),
)
print(
    "matched_scale_counts:",
    dict(sorted(scalar_scale_counts.items())),
)
print(
    "matched_risk_bin_counts:",
    dict(sorted(scalar_risk_bin_counts.items())),
)
print(
    "scalar_skip_counts:",
    dict(scalar_skip_counts),
)

print()
print("=== SELECTING COMMON ORIGINAL BASE ===")

ranked_core_ids = sorted(
    core_ids,
    key=lambda image_id: (
        stable_image_rank(
            "common_original_base",
            image_id,
        ),
        image_id,
    ),
)

common_base_ids = ranked_core_ids[
    :COMMON_BASE_IMAGES
]
uniform_supplement_ids = ranked_core_ids[
    COMMON_BASE_IMAGES:
]

if len(common_base_ids) != COMMON_BASE_IMAGES:
    raise AssertionError(len(common_base_ids))
if len(uniform_supplement_ids) != CROP_SLOTS:
    raise AssertionError(
        len(uniform_supplement_ids)
    )
if set(common_base_ids) & set(
    uniform_supplement_ids
):
    raise AssertionError("Base split overlap")
if (
    set(common_base_ids)
    | set(uniform_supplement_ids)
) != core_set:
    raise AssertionError(
        "Base split does not reconstruct train-core"
    )

typed_summary = summarize(typed_selected)
scalar_summary = summarize(scalar_selected)

mean_risk_gap = abs(
    typed_summary["risk_score"]["mean"]
    - scalar_summary["risk_score"]["mean"]
)
if mean_risk_gap > MAX_MATCHED_MEAN_RISK_GAP:
    raise AssertionError({
        "mean_risk_gap": mean_risk_gap,
        "maximum": MAX_MATCHED_MEAN_RISK_GAP,
    })

annotation_overlap = len(
    {
        candidate.annotation_id
        for candidate in typed_selected
    }
    & {
        candidate.annotation_id
        for candidate in scalar_selected
    }
)
image_overlap = len(
    typed_used_images & scalar_used_images
)

print()
print("=== WRITING AUDIT ARTIFACTS ===")
incomplete.mkdir(parents=True)

typed_csv = incomplete / "typed_cafr_selection.csv"
scalar_csv = incomplete / "scalar_risk_selection.csv"
base_json = incomplete / "common_base_image_ids.json"
uniform_json = (
    incomplete
    / "uniform_supplement_image_ids.json"
)
summary_json = incomplete / "sampling_summary.json"

write_selection_csv(
    typed_csv,
    arm="typed_cafr",
    candidates=typed_selected,
)
write_selection_csv(
    scalar_csv,
    arm="scalar_risk",
    candidates=scalar_selected,
)

base_json.write_text(
    json.dumps(
        common_base_ids,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
uniform_json.write_text(
    json.dumps(
        uniform_supplement_ids,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)

summary = {
    "version": "cafr_sampling_audit_v3",
    "status": "audit_not_frozen_protocol",
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "seed": SEED,
    "script_sha256": sha256_file(
        Path(__file__)
    ),
    "input_hashes": {
        str(path): expected
        for path, expected in EXPECTED_HASHES.items()
    },
    "design": {
        "train_core_images": TOTAL_CORE_IMAGES,
        "uniform_original_views": TOTAL_CORE_IMAGES,
        "common_original_views": COMMON_BASE_IMAGES,
        "object_centric_views_per_arm": CROP_SLOTS,
        "views_per_arm": TOTAL_CORE_IMAGES,
        "one_crop_target_per_source_image": True,
        "scalar_matches_typed_target_classes": True,
        "minimum_target_class_support": MIN_CLASS_SUPPORT,
        "excluded_target_categories": sorted(
            excluded_target_categories
        ),
        "excluded_duplicate_annotation_id": (
            EXCLUDED_DUPLICATE_ANNOTATION_ID
        ),
        "maximum_target_class_fraction": (
            MAX_TARGET_CLASS_FRACTION
        ),
        "typed_quotas": TYPED_QUOTAS,
        "typed_selection_order": (
            TYPED_SELECTION_ORDER
        ),
        "typed_risk_weight_formula": (
            "0.05 + 0.95 * risk_score^2"
        ),
        "typed_sampling_priority": (
            "-log(deterministic_uniform)/risk_weight"
        ),
        "scalar_matching_strata": (
            "target_category_id x scale x fixed_0.05_risk_bin"
        ),
        "scalar_sampling_priority": (
            "deterministic uniform selection inside matched strata"
        ),
        "risk_bin_count": RISK_BIN_COUNT,
        "maximum_matched_mean_risk_gap": (
            MAX_MATCHED_MEAN_RISK_GAP
        ),
    },
    "core_class_counts": dict(sorted(
        core_class_counts.items()
    )),
    "typed": typed_summary,
    "scalar": scalar_summary,
    "target_class_counts_identical": (
        scalar_class_counts
        == typed_class_counts
    ),
    "target_scale_counts_identical": (
        scalar_scale_counts
        == typed_scale_counts
    ),
    "target_risk_bins_identical": (
        scalar_risk_bin_counts
        == typed_risk_bin_counts
    ),
    "typed_risk_bin_counts": dict(sorted(
        typed_risk_bin_counts.items()
    )),
    "scalar_risk_bin_counts": dict(sorted(
        scalar_risk_bin_counts.items()
    )),
    "matched_mean_risk_gap": mean_risk_gap,
    "cross_arm_overlap": {
        "annotation_ids": annotation_overlap,
        "source_image_ids": image_overlap,
        "annotation_fraction": (
            annotation_overlap / CROP_SLOTS
        ),
        "image_fraction": (
            image_overlap / CROP_SLOTS
        ),
    },
    "typed_pool_counts": dict(
        typed_pool_counts
    ),
    "typed_skip_counts": dict(
        typed_skip_counts
    ),
    "scalar_skip_counts": dict(
        scalar_skip_counts
    ),
    "training_view_accounting": {
        "uniform": TOTAL_CORE_IMAGES,
        "scalar_risk": (
            COMMON_BASE_IMAGES + CROP_SLOTS
        ),
        "typed_cafr": (
            COMMON_BASE_IMAGES + CROP_SLOTS
        ),
    },
    "formal_training_rules_frozen": False,
    "images_generated": 0,
    "training_started": False,
}

summary_json.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ) + "\n",
    encoding="utf-8",
)

artifact_files = {}
for path in (
    typed_csv,
    scalar_csv,
    base_json,
    uniform_json,
    summary_json,
):
    artifact_files[path.name] = {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }

artifact_manifest = {
    "version": "cafr_sampling_audit_v3",
    "status": "audit",
    "files": artifact_files,
}

artifact_path = (
    incomplete / "artifact_manifest.json"
)
artifact_path.write_text(
    json.dumps(
        artifact_manifest,
        indent=2,
        ensure_ascii=False,
    ) + "\n",
    encoding="utf-8",
)

incomplete.rename(OUTPUT_DIR)

print()
print("=" * 96)
print("CAFR SAMPLING AUDIT COMPLETE")
print("=" * 96)
print("common_base_images:", len(common_base_ids))
print(
    "uniform_supplement_images:",
    len(uniform_supplement_ids),
)
print(
    "typed_views:",
    typed_summary["count"],
)
print(
    "typed_unique_images:",
    typed_summary["unique_source_images"],
)
print(
    "scalar_views:",
    scalar_summary["count"],
)
print(
    "scalar_unique_images:",
    scalar_summary["unique_source_images"],
)
print(
    "target_class_counts:",
    typed_summary["target_classes"],
)
print(
    "typed_groups:",
    typed_summary["diagnostic_groups"],
)
print(
    "scalar_groups:",
    scalar_summary["diagnostic_groups"],
)
print(
    "typed_scales:",
    typed_summary["scales"],
)
print(
    "scalar_scales:",
    scalar_summary["scales"],
)
print(
    "typed_mean_risk:",
    typed_summary["risk_score"]["mean"],
)
print(
    "scalar_mean_risk:",
    scalar_summary["risk_score"]["mean"],
)
print(
    "annotation_overlap:",
    annotation_overlap,
)
print(
    "source_image_overlap:",
    image_overlap,
)
print("output:", OUTPUT_DIR)
print(
    "artifact_manifest_sha256:",
    sha256_file(
        OUTPUT_DIR / "artifact_manifest.json"
    ),
)
print()
print("PASS: equal-budget Scalar and Typed selections are auditable")
print("NO IMAGES GENERATED")
print("NOTHING TRAINED")
print("FORMAL METHOD PROTOCOL IS NOT FROZEN YET")
