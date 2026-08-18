""



















from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


EXPECTED_IMAGES = 10_000
EXPECTED_ANNOTATIONS = 185_523

BDD100K_CATEGORIES = {
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

METRIC_NAMES = [
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "AR1",
    "AR10",
    "AR100",
    "ARs",
    "ARm",
    "ARl",
]


@dataclass
class Subset:
    dimension: str
    value: str
    role: str
    image_ids: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BDD100K detector robustness by condition."
    )
    parser.add_argument(
        "--gt",
        type=Path,
        required=True,
        help="BDD100K original val COCO annotation.",
    )
    parser.add_argument(
        "--subsets",
        type=Path,
        required=True,
        help="Output directory produced by build_condition_subsets.py.",
    )
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="MODEL=/absolute/path/to/predictions.json; repeat for each model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_prediction_args(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(
                "--prediction must use MODEL=/path/predictions.json"
            )
        model, path_string = item.split("=", 1)
        model = model.strip()
        path = Path(path_string).expanduser().resolve()
        if not model:
            raise ValueError("Prediction model name cannot be empty.")
        if model in result:
            raise ValueError(f"Duplicate model name: {model}")
        if not path.is_file():
            raise FileNotFoundError(path)
        result[model] = path
    return result


def normalize_name(value: Any) -> str:
    return str(value).strip().lower()


def load_gt(path: Path) -> tuple[dict[str, Any], COCO]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if len(data.get("images", [])) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} images, "
            f"found {len(data.get('images', []))}."
        )

    if len(data.get("annotations", [])) != EXPECTED_ANNOTATIONS:
        raise ValueError(
            f"Expected {EXPECTED_ANNOTATIONS} annotations, "
            f"found {len(data.get('annotations', []))}."
        )

    actual_categories = {
        int(category["id"]): normalize_name(category["name"])
        for category in data["categories"]
    }
    expected_categories = {
        category_id: normalize_name(name)
        for category_id, name in BDD100K_CATEGORIES.items()
    }

    if actual_categories != expected_categories:
        raise ValueError(
            "BDD100K category definition mismatch.\n"
            f"Expected: {expected_categories}\n"
            f"Found   : {actual_categories}"
        )


    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(path))

    return data, coco_gt


def load_subsets(
    subset_root: Path,
    all_image_ids: set[int],
) -> list[Subset]:
    image_id_root = subset_root / "image_ids"

    if not image_id_root.is_dir():
        raise FileNotFoundError(
            f"Missing condition subset directory: {image_id_root}"
        )

    subsets = [
        Subset(
            dimension="overall",
            value="all",
            role="primary",
            image_ids=sorted(all_image_ids),
        )
    ]

    paths = sorted(image_id_root.glob("*/*.json"))

    if not paths:
        raise RuntimeError(
            f"No condition subset JSON files found in {image_id_root}"
        )

    seen = set()

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        dimension = str(data["dimension"])
        value = str(data["value"])
        role = str(data["role"])
        image_ids = [int(x) for x in data["image_ids"]]

        key = (dimension, value)

        if key in seen:
            raise ValueError(f"Duplicate subset: {key}")

        seen.add(key)

        if len(image_ids) != int(data["num_images"]):
            raise ValueError(
                f"Subset count mismatch: {dimension}/{value}"
            )

        if len(image_ids) != len(set(image_ids)):
            raise ValueError(
                f"Duplicate image IDs in subset: {dimension}/{value}"
            )

        unknown_ids = set(image_ids) - all_image_ids
        if unknown_ids:
            raise ValueError(
                f"Unknown image IDs in {dimension}/{value}: "
                f"{sorted(unknown_ids)[:10]}"
            )

        if not image_ids:
            raise ValueError(
                f"Empty subset: {dimension}/{value}"
            )

        subsets.append(
            Subset(
                dimension=dimension,
                value=value,
                role=role,
                image_ids=sorted(image_ids),
            )
        )

    dimension_order = {
        "overall": 0,
        "timeofday": 1,
        "weather": 2,
        "scene": 3,
    }

    subsets.sort(
        key=lambda x: (
            dimension_order.get(x.dimension, 99),
            x.value,
        )
    )

    return subsets


def validate_predictions(
    predictions: list[dict[str, Any]],
    valid_image_ids: set[int],
) -> dict[str, int]:
    invalid_image_ids = 0
    invalid_categories = 0
    invalid_numbers = 0
    zero_area = 0
    negative_size = 0

    for index, prediction in enumerate(predictions):
        try:
            image_id = int(prediction["image_id"])
            category_id = int(prediction["category_id"])
            bbox = prediction["bbox"]
            score = float(prediction["score"])
        except Exception as exc:
            raise RuntimeError(
                f"Malformed prediction at index {index}"
            ) from exc

        if image_id not in valid_image_ids:
            invalid_image_ids += 1

        if category_id not in BDD100K_CATEGORIES:
            invalid_categories += 1

        if not isinstance(bbox, list) or len(bbox) != 4:
            raise RuntimeError(
                f"Prediction {index} has invalid bbox: {bbox}"
            )

        values = [float(x) for x in bbox]

        if not all(math.isfinite(x) for x in values):
            invalid_numbers += 1

        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            invalid_numbers += 1

        width = values[2]
        height = values[3]

        if width == 0 or height == 0:
            zero_area += 1

        if width < 0 or height < 0:
            negative_size += 1

    if invalid_image_ids:
        raise RuntimeError(
            f"Predictions contain {invalid_image_ids} invalid image IDs."
        )

    if invalid_categories:
        raise RuntimeError(
            f"Predictions contain {invalid_categories} invalid category IDs."
        )

    if invalid_numbers:
        raise RuntimeError(
            f"Predictions contain {invalid_numbers} non-finite/invalid values."
        )

    return {
        "num_predictions": len(predictions),
        "zero_area_bbox_count": zero_area,
        "negative_size_bbox_count": negative_size,
    }


def run_coco_eval(
    coco_gt: COCO,
    coco_dt: COCO,
    image_ids: list[int],
) -> COCOeval:
    evaluator = COCOeval(
        coco_gt,
        coco_dt,
        "bbox",
    )

    evaluator.params.imgIds = sorted(image_ids)
    evaluator.params.catIds = sorted(BDD100K_CATEGORIES)


    evaluator.params.iouThrs = np.arange(
        0.50,
        0.96,
        0.05,
    )
    evaluator.params.recThrs = np.linspace(
        0.0,
        1.0,
        101,
    )
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.params.areaRng = [
        [0 ** 2, 1e5 ** 2],
        [0 ** 2, 32 ** 2],
        [32 ** 2, 96 ** 2],
        [96 ** 2, 1e5 ** 2],
    ]
    evaluator.params.areaRngLbl = [
        "all",
        "small",
        "medium",
        "large",
    ]


    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    return evaluator


def stats_dict(evaluator: COCOeval) -> dict[str, float | None]:
    result = {}

    for index, name in enumerate(METRIC_NAMES):
        value = float(evaluator.stats[index])
        result[name] = (
            value
            if value >= 0 and math.isfinite(value)
            else None
        )

    return result


def mean_valid(values: np.ndarray) -> float | None:
    valid = values[values > -1]

    if valid.size == 0:
        return None

    return float(np.mean(valid))


def find_iou_index(
    iou_thresholds: np.ndarray,
    target: float,
) -> int:
    matches = np.where(
        np.isclose(
            iou_thresholds,
            target,
            atol=1e-8,
        )
    )[0]

    if len(matches) != 1:
        raise RuntimeError(
            f"Cannot locate IoU={target} in COCOeval."
        )

    return int(matches[0])


def per_class_metrics(
    evaluator: COCOeval,
) -> dict[int, dict[str, float | None]]:
    """
    precision shape:
        [T, R, K, A, M]

    recall shape:
        [T, K, A, M]

    A=0 -> all area
    M=2 -> maxDets=100
    """
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]

    iou50 = find_iou_index(
        evaluator.params.iouThrs,
        0.50,
    )
    iou75 = find_iou_index(
        evaluator.params.iouThrs,
        0.75,
    )

    metrics = {}

    for category_index, category_id in enumerate(
        evaluator.params.catIds
    ):
        metrics[int(category_id)] = {
            "AP": mean_valid(
                precision[
                    :,
                    :,
                    category_index,
                    0,
                    2,
                ]
            ),
            "AP50": mean_valid(
                precision[
                    iou50,
                    :,
                    category_index,
                    0,
                    2,
                ]
            ),
            "AP75": mean_valid(
                precision[
                    iou75,
                    :,
                    category_index,
                    0,
                    2,
                ]
            ),
            "AR100": mean_valid(
                recall[
                    :,
                    category_index,
                    0,
                    2,
                ]
            ),
        }

    return metrics


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    data: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def main() -> None:
    args = parse_args()

    gt_path = args.gt.expanduser().resolve()
    subset_root = args.subsets.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)

    if not subset_root.is_dir():
        raise FileNotFoundError(subset_root)

    prediction_paths = parse_prediction_args(
        args.prediction
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gt_data, coco_gt = load_gt(gt_path)

    all_image_ids = {
        int(image["id"])
        for image in gt_data["images"]
    }

    subsets = load_subsets(
        subset_root,
        all_image_ids,
    )

    # --------------------------------------------------------

    # --------------------------------------------------------

    gt_count_by_image = Counter()
    gt_category_count_by_image: dict[
        int,
        Counter,
    ] = defaultdict(Counter)

    for annotation in gt_data["annotations"]:
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])

        gt_count_by_image[image_id] += 1
        gt_category_count_by_image[
            image_id
        ][category_id] += 1

    model_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    summary_models: dict[str, Any] = {}
    prediction_diagnostics: dict[str, Any] = {}

    start_time = time.time()

    for model_name, prediction_path in prediction_paths.items():
        print()
        print("=" * 80)
        print(f"MODEL: {model_name}")
        print("=" * 80)
        print(f"Loading: {prediction_path}")

        with prediction_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            predictions = json.load(f)

        diagnostics = validate_predictions(
            predictions,
            all_image_ids,
        )

        prediction_diagnostics[
            model_name
        ] = {
            **diagnostics,
            "path": str(prediction_path),
            "sha256": sha256_file(
                prediction_path
            ),
        }

        print(
            f"Predictions: {len(predictions)}"
        )

        with contextlib.redirect_stdout(
            io.StringIO()
        ):
            coco_dt = coco_gt.loadRes(
                predictions
            )


        del predictions

        model_results: dict[str, Any] = {}
        overall_ap: float | None = None

        for subset_index, subset in enumerate(
            subsets,
            start=1,
        ):
            num_gt = sum(
                gt_count_by_image[
                    image_id
                ]
                for image_id in subset.image_ids
            )

            category_counts = Counter()

            for image_id in subset.image_ids:
                category_counts.update(
                    gt_category_count_by_image[
                        image_id
                    ]
                )

            subset_label = (
                f"{subset.dimension}/{subset.value}"
            )

            print(
                f"[{subset_index:02d}/{len(subsets):02d}] "
                f"{subset_label:<35} "
                f"images={len(subset.image_ids):5d} "
                f"GT={num_gt:6d}",
                end="",
                flush=True,
            )

            evaluator = run_coco_eval(
                coco_gt,
                coco_dt,
                subset.image_ids,
            )

            metrics = stats_dict(
                evaluator
            )

            if (
                subset.dimension == "overall"
                and subset.value == "all"
            ):
                overall_ap = metrics["AP"]

            if overall_ap is None:
                raise RuntimeError(
                    "Overall subset must be evaluated first."
                )

            ap = metrics["AP"]

            delta_vs_overall = (
                ap - overall_ap
                if ap is not None
                else None
            )

            retention = (
                ap / overall_ap
                if (
                    ap is not None
                    and overall_ap > 0
                )
                else None
            )

            row = {
                "model": model_name,
                "dimension": subset.dimension,
                "value": subset.value,
                "role": subset.role,
                "num_images": len(
                    subset.image_ids
                ),
                "num_annotations": num_gt,
                **metrics,
                "delta_AP_vs_overall": (
                    delta_vs_overall
                ),
                "AP_retention": retention,
            }

            model_rows.append(row)

            class_metrics = per_class_metrics(
                evaluator
            )

            for category_id, category_name in BDD100K_CATEGORIES.items():
                per_class_rows.append(
                    {
                        "model": model_name,
                        "dimension": subset.dimension,
                        "value": subset.value,
                        "role": subset.role,
                        "num_images": len(
                            subset.image_ids
                        ),
                        "category_id": category_id,
                        "category": category_name,
                        "gt_count": int(
                            category_counts[
                                category_id
                            ]
                        ),
                        **class_metrics[
                            category_id
                        ],
                    }
                )

            model_results[
                f"{subset.dimension}/{subset.value}"
            ] = {
                "role": subset.role,
                "num_images": len(
                    subset.image_ids
                ),
                "num_annotations": num_gt,
                **metrics,
                "delta_AP_vs_overall": (
                    delta_vs_overall
                ),
                "AP_retention": retention,
            }

            print(
                f"  AP={metrics['AP']:.4f}"
                if metrics["AP"] is not None
                else "  AP=N/A"
            )

        summary_models[
            model_name
        ] = model_results

        del coco_dt

    # --------------------------------------------------------


    # model B - model A
    # --------------------------------------------------------

    comparison_rows = []

    model_names = list(
        prediction_paths.keys()
    )

    if len(model_names) == 2:
        model_a = model_names[0]
        model_b = model_names[1]

        rows_by_key = {
            (
                row["model"],
                row["dimension"],
                row["value"],
            ): row
            for row in model_rows
        }

        for subset in subsets:
            row_a = rows_by_key[
                (
                    model_a,
                    subset.dimension,
                    subset.value,
                )
            ]

            row_b = rows_by_key[
                (
                    model_b,
                    subset.dimension,
                    subset.value,
                )
            ]

            comparison = {
                "dimension": subset.dimension,
                "value": subset.value,
                "role": subset.role,
                "num_images": len(
                    subset.image_ids
                ),
                "model_a": model_a,
                "model_b": model_b,
            }

            for metric in (
                "AP",
                "AP50",
                "AP75",
                "APs",
                "APm",
                "APl",
                "AR100",
            ):
                value_a = row_a[metric]
                value_b = row_b[metric]

                comparison[
                    f"{model_a}_{metric}"
                ] = value_a

                comparison[
                    f"{model_b}_{metric}"
                ] = value_b

                comparison[
                    f"delta_{model_b}_minus_{model_a}_{metric}"
                ] = (
                    value_b - value_a
                    if (
                        value_a is not None
                        and value_b is not None
                    )
                    else None
                )

            comparison_rows.append(
                comparison
            )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    condition_metric_fields = [
        "model",
        "dimension",
        "value",
        "role",
        "num_images",
        "num_annotations",
        *METRIC_NAMES,
        "delta_AP_vs_overall",
        "AP_retention",
    ]

    write_csv(
        output_dir
        / "condition_metrics.csv",
        model_rows,
        condition_metric_fields,
    )

    write_csv(
        output_dir
        / "condition_per_class_metrics.csv",
        per_class_rows,
        [
            "model",
            "dimension",
            "value",
            "role",
            "num_images",
            "category_id",
            "category",
            "gt_count",
            "AP",
            "AP50",
            "AP75",
            "AR100",
        ],
    )

    if comparison_rows:
        write_csv(
            output_dir
            / "condition_model_comparison.csv",
            comparison_rows,
            list(
                comparison_rows[0].keys()
            ),
        )

    elapsed = (
        time.time()
        - start_time
    )

    summary = {
        "dataset": "BDD100K",
        "split": "val",
        "models": summary_models,
    }

    write_json(
        output_dir
        / "summary.json",
        summary,
    )

    metadata = {
        "dataset": "BDD100K",
        "split": "val",
        "gt": str(gt_path),
        "gt_sha256": sha256_file(
            gt_path
        ),
        "subset_root": str(
            subset_root
        ),
        "subset_manifest": str(
            subset_root
            / "manifest.json"
        ),
        "prediction_diagnostics": (
            prediction_diagnostics
        ),
        "num_subsets_including_overall": len(
            subsets
        ),
        "evaluation": {
            "type": "COCO bbox",
            "iou_thresholds": (
                "0.50:0.05:0.95"
            ),
            "max_dets": [1, 10, 100],
            "area_ranges": {
                "small": "area < 32^2",
                "medium": (
                    "32^2 <= area < 96^2"
                ),
                "large": "area >= 96^2",
            },
            "gt_modified": False,
            "predictions_modified": False,
            "subset_mechanism": (
                "COCOeval.params.imgIds"
            ),
        },
        "elapsed_seconds": round(
            elapsed,
            3,
        ),
    }

    write_json(
        output_dir
        / "evaluation_metadata.json",
        metadata,
    )

    # --------------------------------------------------------
    # Console final summary
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("CONDITION ROBUSTNESS RESULTS")
    print("=" * 100)

    primary_rows = [
        row
        for row in model_rows
        if row["role"] == "primary"
    ]

    for model_name in model_names:
        print()
        print(model_name)
        print("-" * 100)

        rows = [
            row
            for row in primary_rows
            if row["model"] == model_name
        ]

        for row in rows:
            label = (
                "Overall"
                if row["dimension"] == "overall"
                else (
                    f"{row['dimension']}:"
                    f"{row['value']}"
                )
            )

            print(
                f"{label:<35} "
                f"N={row['num_images']:5d}  "
                f"AP={row['AP']:.4f}  "
                f"AP50={row['AP50']:.4f}  "
                f"APs={row['APs']:.4f}  "
                f"APm={row['APm']:.4f}  "
                f"APl={row['APl']:.4f}  "
                f"ΔAP={row['delta_AP_vs_overall']:+.4f}"
            )

    print()
    print(f"Results saved to: {output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()