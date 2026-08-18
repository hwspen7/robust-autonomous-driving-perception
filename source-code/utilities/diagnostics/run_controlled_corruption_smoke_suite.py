"""
Run all 15 controlled-corruption smoke tests for YOLO11m and D-FINE-M.

Run this script from the BASE conda environment.

It:
- uses the existing controlled-corruption exporters;
- runs 100 images by default;
- skips an already completed model/variant unless --force-inference is set;
- validates that both models processed the same fixed BDD100K subset;
- runs COCO bbox evaluation after each inference;
- progressively saves CSV/JSON metrics so interrupted work is not lost.

Smoke results are pipeline diagnostics only, not formal 10,000-image benchmark results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


PROJECT_ROOT = Path(
    "/root/rivermind-data/autodrive/code/robust-autonomous-driving-perception"
)

YOLO_EXPORTER = (
    PROJECT_ROOT
    / "scripts/export_yolo11m_controlled_corruptions.py"
)
DFINE_EXPORTER = (
    PROJECT_ROOT
    / "scripts/export_dfine_m_controlled_corruptions.py"
)
CONTROLLED_MODULE = (
    PROJECT_ROOT
    / "scripts/controlled_corruptions.py"
)

YOLO_MODEL = Path(
    "/root/rivermind-data/autodrive/results/detection/"
    "yolo11m_bdd100k_main/weights/best.pt"
)

DFINE_ROOT = PROJECT_ROOT / "third_party/D-FINE"
DFINE_CONFIG = (
    DFINE_ROOT
    / "configs/dfine/dfine_hgnetv2_m_bdd100k.yml"
)
DFINE_CHECKPOINT = Path(
    "/root/rivermind-data/autodrive/results/final/"
    "dfine_m_bdd100k_60epoch_final/checkpoints/"
    "dfine_m_bdd100k_best.pth"
)
DFINE_ENV = Path(
    "/root/rivermind-data/autodrive/conda_envs/dfine"
)

DATASET_ROOT = Path(
    "/root/rivermind-data/autodrive/datasets/datasets/bdd100k_final"
)
GT_PATH = (
    DATASET_ROOT
    / "coco/annotations/instances_val.json"
)

DEFAULT_OUTPUT_ROOT = Path(
    "/root/rivermind-data/autodrive/results/evaluation/"
    "controlled_corruption/smoke"
)

VARIANTS = [
    ("low_light", 1),
    ("low_light", 2),
    ("low_light", 3),
    ("fog", 1),
    ("fog", 2),
    ("fog", 3),
    ("rain", 1),
    ("rain", 2),
    ("rain", 3),
    ("blur", 1),
    ("blur", 2),
    ("blur", 3),
    ("noise", 1),
    ("noise", 2),
    ("noise", 3),
]

METRIC_FIELDS = [
    "model",
    "variant",
    "corruption",
    "severity",
    "num_images",
    "num_predictions",
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
    "prediction_json",
    "prediction_sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--save-vis",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--force-inference",
        action="store_true",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Example: --only fog_s1 rain_s3",
    )
    return parser.parse_args()


def variant_name(
    corruption: str,
    severity: int,
) -> str:
    return f"{corruption}_s{severity}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def require_file(
    path: Path,
    label: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} missing:\n{path}"
        )


def require_dir(
    path: Path,
    label: str,
) -> None:
    if not path.is_dir():
        raise FileNotFoundError(
            f"{label} missing:\n{path}"
        )


def get_variants(
    only: list[str] | None,
) -> list[tuple[str, int]]:
    if not only:
        return list(VARIANTS)

    mapping = {
        variant_name(c, s): (c, s)
        for c, s in VARIANTS
    }

    bad = [
        name
        for name in only
        if name not in mapping
    ]

    if bad:
        raise ValueError(
            f"Unknown variants: {bad}"
        )

    return [
        mapping[name]
        for name in only
    ]


def build_subset_gt(
    *,
    limit: int,
    output_path: Path,
) -> tuple[set[int], int]:
    with GT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        gt = json.load(f)

    images = sorted(
        gt["images"],
        key=lambda item: int(item["id"]),
    )

    if limit <= 0:
        raise ValueError(
            "--limit must be > 0."
        )

    selected_images = images[:limit]

    if len(selected_images) != limit:
        raise RuntimeError(
            "Not enough GT images."
        )

    selected_ids = {
        int(item["id"])
        for item in selected_images
    }

    selected_annotations = [
        ann
        for ann in gt["annotations"]
        if int(ann["image_id"])
        in selected_ids
    ]

    subset = {
        key: value
        for key, value in gt.items()
        if key not in {
            "images",
            "annotations",
        }
    }

    subset["images"] = selected_images
    subset["annotations"] = (
        selected_annotations
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            subset,
            f,
            ensure_ascii=False,
        )

    return (
        selected_ids,
        len(selected_annotations),
    )


def run_command(
    command: list[str],
    label: str,
) -> None:
    print()
    print("=" * 110)
    print(label)
    print("=" * 110)

    subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        check=True,
    )


def validate_existing_metadata(
    metadata_path: Path,
    *,
    corruption: str,
    severity: int,
) -> None:
    ""







    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    expected_variant = (
        f"{corruption}_s{severity}"
    )

    # --------------------------------------------------------

    # --------------------------------------------------------
    found_corruption = metadata.get(
        "corruption"
    )

    found_severity = metadata.get(
        "severity"
    )

    found_variant = metadata.get(
        "variant"
    )

    # --------------------------------------------------------

    # --------------------------------------------------------
    controlled = metadata.get(
        "controlled_corruption"
    )

    if isinstance(
        controlled,
        dict,
    ):
        if found_corruption is None:
            found_corruption = (
                controlled.get(
                    "corruption"
                )
            )

        if found_severity is None:
            found_severity = (
                controlled.get(
                    "severity"
                )
            )

        if found_variant is None:
            found_variant = (
                controlled.get(
                    "variant"
                )
            )

    # --------------------------------------------------------

    # --------------------------------------------------------
    if found_variant is not None:
        if str(found_variant) != expected_variant:
            raise RuntimeError(
                "Existing output metadata variant mismatch.\n"
                f"Expected : {expected_variant}\n"
                f"Found    : {found_variant}\n"
                f"Metadata : {metadata_path}"
            )

        return

    # --------------------------------------------------------

    # --------------------------------------------------------
    if found_corruption is None or found_severity is None:


        print(
            "[WARNING] Existing metadata does not contain "
            "explicit corruption/severity fields."
        )
        print(
            f"          Reusing output directory: "
            f"{metadata_path.parent}"
        )
        return

    if (
        str(found_corruption)
        != corruption
        or int(found_severity)
        != severity
    ):
        raise RuntimeError(
            "Existing output metadata does not match "
            "requested corruption variant.\n"
            f"Expected : {corruption}, severity={severity}\n"
            f"Found    : {found_corruption}, "
            f"severity={found_severity}\n"
            f"Metadata : {metadata_path}"
        )


def validate_image_summary(
    summary_path: Path,
    *,
    expected_ids: set[int],
    limit: int,
) -> None:
    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        summary = json.load(f)

    if len(summary) != limit:
        raise RuntimeError(
            f"Image count mismatch: "
            f"{summary_path}"
        )

    actual_ids = {
        int(item["image_id"])
        for item in summary
    }

    if actual_ids != expected_ids:
        raise RuntimeError(
            "Exporter did not process the fixed "
            f"{limit}-image subset:\n"
            f"{summary_path}"
        )


def run_yolo(
    *,
    corruption: str,
    severity: int,
    output_dir: Path,
    limit: int,
    save_vis: int,
    force: bool,
) -> None:
    prediction_path = (
        output_dir
        / "predictions.json"
    )
    metadata_path = (
        output_dir
        / "metadata.json"
    )

    if (
        prediction_path.is_file()
        and metadata_path.is_file()
        and not force
    ):
        validate_existing_metadata(
            metadata_path,
            corruption=corruption,
            severity=severity,
        )
        print(
            f"[SKIP] YOLO11m "
            f"{variant_name(corruption, severity)}"
        )
        return

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        sys.executable,
        str(YOLO_EXPORTER),
        "--model",
        str(YOLO_MODEL),
        "--dataset-root",
        str(DATASET_ROOT),
        "--annotation",
        str(GT_PATH),
        "--output-dir",
        str(output_dir),
        "--corruption",
        corruption,
        "--severity",
        str(severity),
        "--device",
        "0",
        "--imgsz",
        "640",
        "--batch",
        "16",
        "--conf",
        "0.001",
        "--iou",
        "0.7",
        "--max-det",
        "300",
        "--limit",
        str(limit),
        "--save-vis",
        str(save_vis),
        "--overwrite",
    ]

    run_command(
        command,
        (
            "YOLO11m | "
            f"{variant_name(corruption, severity)}"
        ),
    )


def run_dfine(
    *,
    conda_executable: str,
    corruption: str,
    severity: int,
    output_dir: Path,
    limit: int,
    save_vis: int,
    force: bool,
) -> None:
    prediction_path = (
        output_dir
        / "predictions.json"
    )
    metadata_path = (
        output_dir
        / "metadata.json"
    )

    if (
        prediction_path.is_file()
        and metadata_path.is_file()
        and not force
    ):
        validate_existing_metadata(
            metadata_path,
            corruption=corruption,
            severity=severity,
        )
        print(
            f"[SKIP] D-FINE-M "
            f"{variant_name(corruption, severity)}"
        )
        return

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        conda_executable,
        "run",
        "--no-capture-output",
        "-p",
        str(DFINE_ENV),
        "python",
        str(DFINE_EXPORTER),
        "--dfine-root",
        str(DFINE_ROOT),
        "--config",
        str(DFINE_CONFIG),
        "--checkpoint",
        str(DFINE_CHECKPOINT),
        "--dataset-root",
        str(DATASET_ROOT),
        "--annotation",
        str(GT_PATH),
        "--output-dir",
        str(output_dir),
        "--corruption",
        corruption,
        "--severity",
        str(severity),
        "--device",
        "cuda:0",
        "--limit",
        str(limit),
        "--save-vis",
        str(save_vis),
        "--vis-conf",
        "0.25",
        "--overwrite",
    ]

    run_command(
        command,
        (
            "D-FINE-M | "
            f"{variant_name(corruption, severity)}"
        ),
    )


def load_predictions(
    prediction_path: Path,
    expected_ids: set[int],
) -> list[dict[str, Any]]:
    with prediction_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        predictions = json.load(f)

    if not isinstance(
        predictions,
        list,
    ):
        raise TypeError(
            "predictions.json must be a list."
        )

    bad_images = {
        int(item["image_id"])
        for item in predictions
        if int(item["image_id"])
        not in expected_ids
    }

    if bad_images:
        raise RuntimeError(
            "Prediction contains unexpected image IDs."
        )

    bad_categories = {
        int(item["category_id"])
        for item in predictions
        if int(item["category_id"])
        not in range(1, 11)
    }

    if bad_categories:
        raise RuntimeError(
            "Prediction contains invalid "
            "BDD category IDs."
        )

    return predictions


def evaluate(
    *,
    model: str,
    corruption: str,
    severity: int,
    prediction_path: Path,
    subset_gt_path: Path,
    expected_ids: set[int],
    limit: int,
) -> dict[str, Any]:
    predictions = load_predictions(
        prediction_path,
        expected_ids,
    )

    print()
    print(
        f"COCOeval | {model} | "
        f"{variant_name(corruption, severity)}"
    )

    coco_gt = COCO(
        str(subset_gt_path)
    )
    coco_dt = coco_gt.loadRes(
        predictions
    )

    evaluator = COCOeval(
        coco_gt,
        coco_dt,
        "bbox",
    )
    evaluator.params.imgIds = sorted(
        expected_ids
    )

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = [
        float(value)
        for value in evaluator.stats
    ]

    return {
        "model": model,
        "variant": variant_name(
            corruption,
            severity,
        ),
        "corruption": corruption,
        "severity": severity,
        "num_images": limit,
        "num_predictions": len(
            predictions
        ),
        "AP": stats[0],
        "AP50": stats[1],
        "AP75": stats[2],
        "AP_small": stats[3],
        "AP_medium": stats[4],
        "AP_large": stats[5],
        "AR_1": stats[6],
        "AR_10": stats[7],
        "AR_100": stats[8],
        "AR_small": stats[9],
        "AR_medium": stats[10],
        "AR_large": stats[11],
        "prediction_json": str(
            prediction_path
        ),
        "prediction_sha256": (
            sha256_file(
                prediction_path
            )
        ),
    }


def replace_row(
    rows: list[dict[str, Any]],
    new_row: dict[str, Any],
) -> list[dict[str, Any]]:
    key = (
        new_row["model"],
        new_row["variant"],
    )

    rows = [
        row
        for row in rows
        if (
            row.get("model"),
            row.get("variant"),
        )
        != key
    ]

    rows.append(
        new_row
    )
    return rows


def save_metrics(
    rows: list[dict[str, Any]],
    csv_path: Path,
    json_path: Path,
) -> None:
    rows = sorted(
        rows,
        key=lambda row: (
            row["variant"],
            row["model"],
        ),
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=METRIC_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            rows,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def load_old_metrics(
    json_path: Path,
) -> list[dict[str, Any]]:
    if not json_path.is_file():
        return []

    with json_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    return (
        data
        if isinstance(data, list)
        else []
    )


def main() -> None:
    args = parse_args()
    start = time.time()

    try:
        import ultralytics  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Run this script from BASE:\n"
            "conda activate base"
        ) from exc

    for path, label in [
        (
            YOLO_EXPORTER,
            "YOLO controlled exporter",
        ),
        (
            DFINE_EXPORTER,
            "D-FINE controlled exporter",
        ),
        (
            CONTROLLED_MODULE,
            "controlled_corruptions.py",
        ),
        (
            YOLO_MODEL,
            "YOLO checkpoint",
        ),
        (
            DFINE_CONFIG,
            "D-FINE config",
        ),
        (
            DFINE_CHECKPOINT,
            "D-FINE checkpoint",
        ),
        (
            GT_PATH,
            "BDD100K GT",
        ),
    ]:
        require_file(
            path,
            label,
        )

    require_dir(
        DFINE_ENV,
        "D-FINE conda environment",
    )
    require_dir(
        DATASET_ROOT,
        "BDD100K dataset root",
    )

    conda = shutil.which(
        "conda"
    )

    if conda is None:
        raise RuntimeError(
            "`conda` executable not found."
        )

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )
    eval_dir = (
        output_root
        / "suite_evaluation"
    )
    eval_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    subset_gt = (
        eval_dir
        / f"smoke_gt_{args.limit}.json"
    )
    metrics_csv = (
        eval_dir
        / "smoke_suite_metrics.csv"
    )
    metrics_json = (
        eval_dir
        / "smoke_suite_metrics.json"
    )
    metadata_json = (
        eval_dir
        / "smoke_suite_metadata.json"
    )

    expected_ids, num_gt = (
        build_subset_gt(
            limit=args.limit,
            output_path=subset_gt,
        )
    )

    variants = get_variants(
        args.only
    )
    rows = load_old_metrics(
        metrics_json
    )

    print("=" * 110)
    print(
        "CONTROLLED CORRUPTION SMOKE SUITE"
    )
    print("=" * 110)
    print(
        f"Images   : {args.limit}"
    )
    print(
        f"GT       : {num_gt}"
    )
    print(
        f"Variants : {len(variants)}"
    )
    print(
        "Models   : YOLO11m + D-FINE-M"
    )
    print(
        f"Output   : {output_root}"
    )
    print("=" * 110)

    for index, (
        corruption,
        severity,
    ) in enumerate(
        variants,
        start=1,
    ):
        variant = variant_name(
            corruption,
            severity,
        )

        print()
        print("#" * 110)
        print(
            f"{index}/{len(variants)} | "
            f"{variant}"
        )
        print("#" * 110)

        yolo_dir = (
            output_root
            / f"yolo11m_{variant}"
        )
        dfine_dir = (
            output_root
            / f"dfine_m_{variant}"
        )

        run_yolo(
            corruption=corruption,
            severity=severity,
            output_dir=yolo_dir,
            limit=args.limit,
            save_vis=args.save_vis,
            force=args.force_inference,
        )

        validate_image_summary(
            yolo_dir
            / "image_summary.json",
            expected_ids=expected_ids,
            limit=args.limit,
        )

        yolo_row = evaluate(
            model="YOLO11m",
            corruption=corruption,
            severity=severity,
            prediction_path=(
                yolo_dir
                / "predictions.json"
            ),
            subset_gt_path=subset_gt,
            expected_ids=expected_ids,
            limit=args.limit,
        )

        rows = replace_row(
            rows,
            yolo_row,
        )
        save_metrics(
            rows,
            metrics_csv,
            metrics_json,
        )

        run_dfine(
            conda_executable=conda,
            corruption=corruption,
            severity=severity,
            output_dir=dfine_dir,
            limit=args.limit,
            save_vis=args.save_vis,
            force=args.force_inference,
        )

        validate_image_summary(
            dfine_dir
            / "image_summary.json",
            expected_ids=expected_ids,
            limit=args.limit,
        )

        dfine_row = evaluate(
            model="D-FINE-M",
            corruption=corruption,
            severity=severity,
            prediction_path=(
                dfine_dir
                / "predictions.json"
            ),
            subset_gt_path=subset_gt,
            expected_ids=expected_ids,
            limit=args.limit,
        )

        rows = replace_row(
            rows,
            dfine_row,
        )
        save_metrics(
            rows,
            metrics_csv,
            metrics_json,
        )

        print()
        print(
            f"[DONE] {variant} | "
            f"YOLO AP={yolo_row['AP']:.4f} | "
            f"D-FINE AP={dfine_row['AP']:.4f}"
        )

    elapsed = (
        time.time()
        - start
    )

    metadata = {
        "stage": (
            "controlled_corruption_smoke_suite"
        ),
        "purpose": (
            "Pipeline validation only; "
            "not formal benchmark results."
        ),
        "num_images": args.limit,
        "num_gt_annotations": num_gt,
        "variants": [
            variant_name(c, s)
            for c, s in variants
        ],
        "models": [
            "YOLO11m",
            "D-FINE-M",
        ],
        "gt": str(GT_PATH),
        "subset_gt": str(
            subset_gt
        ),
        "yolo_exporter_sha256": (
            sha256_file(
                YOLO_EXPORTER
            )
        ),
        "dfine_exporter_sha256": (
            sha256_file(
                DFINE_EXPORTER
            )
        ),
        "controlled_corruptions_sha256": (
            sha256_file(
                CONTROLLED_MODULE
            )
        ),
        "elapsed_seconds": round(
            elapsed,
            3,
        ),
        "metrics_csv": str(
            metrics_csv
        ),
        "metrics_json": str(
            metrics_json
        ),
    }

    with metadata_json.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    expected_keys = {
        (
            model,
            variant_name(c, s),
        )
        for c, s in variants
        for model in [
            "YOLO11m",
            "D-FINE-M",
        ]
    }

    actual_keys = {
        (
            row["model"],
            row["variant"],
        )
        for row in rows
    }

    missing = sorted(
        expected_keys
        - actual_keys
    )

    if missing:
        raise RuntimeError(
            f"Missing metrics: {missing}"
        )

    print()
    print("=" * 110)
    print(
        "CONTROLLED CORRUPTION SMOKE SUITE COMPLETE"
    )
    print("=" * 110)
    print(
        f"Variants    : {len(variants)}"
    )
    print(
        f"Evaluations : {len(expected_keys)}"
    )
    print(
        f"Time        : {elapsed / 60.0:.2f} min"
    )
    print(
        f"CSV         : {metrics_csv}"
    )
    print(
        f"JSON        : {metrics_json}"
    )
    print(
        f"Metadata    : {metadata_json}"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()
