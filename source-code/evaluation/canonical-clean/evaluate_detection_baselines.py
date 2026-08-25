""



































from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ============================================================

#



# ============================================================

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

EXPECTED_BDD100K_VAL_IMAGES = 10_000
EXPECTED_BDD100K_VAL_ANNOTATIONS = 185_523

OVERALL_METRIC_NAMES = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified COCO bbox evaluation for BDD100K detection baselines."
        )
    )

    parser.add_argument(
        "--gt",
        type=Path,
        required=True,
        help=(
            "Original BDD100K COCO validation annotation JSON. "
            "Must use category IDs 1~10."
        ),
    )

    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help=(
            "Prediction specification: MODEL_NAME=/path/to/predictions.json "
            "Can be supplied multiple times."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory used to store all unified evaluation results.",
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    ""






    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def safe_package_version(package_name: str) -> str:
    ""





    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def sanitize_model_name(name: str) -> str:
    ""






    name = name.strip()

    if not name:
        raise ValueError("Model name cannot be empty.")

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

    if not safe_name:
        raise ValueError(
            f"Invalid model name: {name!r}"
        )

    return safe_name


def parse_prediction_specs(
    specs: list[str],
) -> list[tuple[str, Path]]:
    ""











    parsed: list[tuple[str, Path]] = []
    names_seen: set[str] = set()

    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                "Each --prediction must use: "
                "MODEL_NAME=/path/to/predictions.json"
            )

        model_name, file_path = spec.split("=", 1)

        model_name = model_name.strip()
        path = Path(file_path.strip()).expanduser().resolve()

        if not model_name:
            raise ValueError(
                f"Missing model name in prediction specification: {spec}"
            )

        if model_name in names_seen:
            raise ValueError(
                f"Duplicate model name: {model_name}"
            )

        if not path.is_file():
            raise FileNotFoundError(
                f"Prediction file does not exist: {path}"
            )

        names_seen.add(model_name)
        parsed.append((model_name, path))

    return parsed


def load_raw_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def validate_bdd100k_gt(
    gt_path: Path,
) -> dict[str, Any]:
    ""










    raw_gt = load_raw_json(gt_path)

    if not isinstance(raw_gt, dict):
        raise TypeError(
            "Ground-truth JSON must be a COCO dictionary."
        )

    for key in ("images", "annotations", "categories"):
        if key not in raw_gt:
            raise KeyError(
                f"Ground-truth JSON missing key: {key}"
            )

    images = raw_gt["images"]
    annotations = raw_gt["annotations"]
    categories = raw_gt["categories"]

    if len(images) != EXPECTED_BDD100K_VAL_IMAGES:
        raise ValueError(
            "Wrong BDD100K validation image count.\n"
            f"Expected: {EXPECTED_BDD100K_VAL_IMAGES}\n"
            f"Found   : {len(images)}\n"
            "Check whether the correct original val annotation was used."
        )

    if len(annotations) != EXPECTED_BDD100K_VAL_ANNOTATIONS:
        raise ValueError(
            "Wrong BDD100K validation annotation count.\n"
            f"Expected: {EXPECTED_BDD100K_VAL_ANNOTATIONS}\n"
            f"Found   : {len(annotations)}"
        )

    # --------------------------------------------------------

    # --------------------------------------------------------

    image_ids = [int(image["id"]) for image in images]

    if len(image_ids) != len(set(image_ids)):
        raise ValueError(
            "Ground truth contains duplicate image IDs."
        )

    image_id_set = set(image_ids)

    # --------------------------------------------------------

    # --------------------------------------------------------

    actual_categories = {
        int(category["id"]): str(category["name"])
        for category in categories
    }

    if actual_categories != BDD100K_CATEGORIES:
        raise ValueError(
            "BDD100K category definition mismatch.\n"
            f"Expected: {BDD100K_CATEGORIES}\n"
            f"Found   : {actual_categories}\n\n"
            "Do NOT use the contiguous 0~9 training annotation "
            "for final evaluation."
        )

    # --------------------------------------------------------

    # --------------------------------------------------------

    annotation_ids = [
        int(annotation["id"])
        for annotation in annotations
    ]

    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError(
            "Ground truth contains duplicate annotation IDs."
        )

    if any(annotation_id <= 0 for annotation_id in annotation_ids):
        raise ValueError(
            "Ground-truth annotation IDs must be positive."
        )

    # --------------------------------------------------------

    # --------------------------------------------------------

    category_ids = set(BDD100K_CATEGORIES)

    gt_counts = Counter()

    for index, annotation in enumerate(annotations):
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])

        if image_id not in image_id_set:
            raise ValueError(
                f"GT annotation {index} references unknown image_id "
                f"{image_id}."
            )

        if category_id not in category_ids:
            raise ValueError(
                f"GT annotation {index} contains invalid category_id "
                f"{category_id}."
            )

        bbox = annotation.get("bbox")

        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(
                f"GT annotation {index} has invalid bbox: {bbox}"
            )

        x, y, w, h = map(float, bbox)

        if not all(
            math.isfinite(value)
            for value in (x, y, w, h)
        ):
            raise ValueError(
                f"GT annotation {index} contains non-finite bbox."
            )

        if w <= 0 or h <= 0:
            raise ValueError(
                f"GT annotation {index} has non-positive bbox "
                f"size: {bbox}"
            )

        gt_counts[category_id] += 1

    # --------------------------------------------------------
    # image size lookup：

    #


    # --------------------------------------------------------

    image_sizes = {}

    for image in images:
        image_id = int(image["id"])

        image_sizes[image_id] = (
            int(image["width"]),
            int(image["height"]),
        )

    return {
        "num_images": len(images),
        "num_annotations": len(annotations),
        "image_ids": image_id_set,
        "image_sizes": image_sizes,
        "category_ids": category_ids,
        "gt_counts": dict(gt_counts),
    }


def load_prediction_list(
    prediction_path: Path,
) -> list[dict[str, Any]]:
    ""















    raw = load_raw_json(prediction_path)

    if isinstance(raw, dict):
        if "predictions" not in raw:
            raise ValueError(
                f"Prediction JSON dictionary does not contain "
                f"'predictions': {prediction_path}"
            )

        raw = raw["predictions"]

    if not isinstance(raw, list):
        raise TypeError(
            f"Prediction JSON must contain a list: {prediction_path}"
        )

    if len(raw) == 0:
        raise ValueError(
            f"Prediction file contains zero detections: "
            f"{prediction_path}"
        )

    return raw


def strict_int(
    value: Any,
    field_name: str,
    index: int,
) -> int:
    ""











    if isinstance(value, bool):
        raise ValueError(
            f"Prediction {index}: {field_name} cannot be bool."
        )

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Prediction {index}: invalid {field_name}: {value}"
        ) from exc

    if not math.isfinite(numeric_value):
        raise ValueError(
            f"Prediction {index}: non-finite {field_name}: {value}"
        )

    if not numeric_value.is_integer():
        raise ValueError(
            f"Prediction {index}: {field_name} must be integer, "
            f"got {value}"
        )

    return int(numeric_value)


def validate_and_canonicalize_predictions(
    raw_predictions: list[dict[str, Any]],
    gt_info: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ""











    valid_image_ids = gt_info["image_ids"]
    valid_category_ids = gt_info["category_ids"]
    image_sizes = gt_info["image_sizes"]

    canonical: list[dict[str, Any]] = []

    class_counts = Counter()
    image_counts = Counter()

    out_of_bounds_count = 0
    negative_xy_count = 0
    negative_size_bbox_count = 0
    zero_area_bbox_count = 0

    score_min = math.inf
    score_max = -math.inf

    for index, prediction in enumerate(raw_predictions):
        if not isinstance(prediction, dict):
            raise TypeError(
                f"Prediction {index} is not a dictionary."
            )

        required_fields = {
            "image_id",
            "category_id",
            "bbox",
            "score",
        }

        missing_fields = (
            required_fields - prediction.keys()
        )

        if missing_fields:
            raise KeyError(
                f"Prediction {index} missing fields: "
                f"{sorted(missing_fields)}"
            )

        image_id = strict_int(
            prediction["image_id"],
            "image_id",
            index,
        )

        category_id = strict_int(
            prediction["category_id"],
            "category_id",
            index,
        )

        if image_id not in valid_image_ids:
            raise ValueError(
                f"Prediction {index}: image_id={image_id} "
                "does not exist in the evaluation GT."
            )

        # ----------------------------------------------------

        #


        # ----------------------------------------------------

        if category_id not in valid_category_ids:
            raise ValueError(
                f"Prediction {index}: invalid BDD100K "
                f"category_id={category_id}.\n"
                "Final evaluation requires original IDs 1~10."
            )

        bbox = prediction["bbox"]

        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
        ):
            raise ValueError(
                f"Prediction {index}: bbox must be "
                f"[x, y, width, height], got {bbox}"
            )

        try:
            x, y, width, height = (
                float(value)
                for value in bbox
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Prediction {index}: invalid bbox {bbox}"
            ) from exc

        if not all(
            math.isfinite(value)
            for value in (
                x,
                y,
                width,
                height,
            )
        ):
            raise ValueError(
                f"Prediction {index}: bbox contains NaN/Inf."
            )



        #


        if width < 0 or height < 0:
            negative_size_bbox_count += 1

        if width == 0 or height == 0:
            zero_area_bbox_count += 1

        try:
            score = float(prediction["score"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Prediction {index}: invalid score "
                f"{prediction['score']}"
            ) from exc

        if not math.isfinite(score):
            raise ValueError(
                f"Prediction {index}: score is NaN/Inf."
            )

        if score < 0.0 or score > 1.0:
            raise ValueError(
                f"Prediction {index}: score must be in [0, 1], "
                f"got {score}"
            )

        # ----------------------------------------------------

        #


        # ----------------------------------------------------

        image_width, image_height = image_sizes[image_id]

        if x < 0 or y < 0:
            negative_xy_count += 1

        if (
            x < 0
            or y < 0
            or x + width > image_width
            or y + height > image_height
        ):
            out_of_bounds_count += 1

        canonical_prediction = {
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [
                x,
                y,
                width,
                height,
            ],
            "score": score,
        }

        canonical.append(
            canonical_prediction
        )

        class_counts[category_id] += 1
        image_counts[image_id] += 1

        score_min = min(
            score_min,
            score,
        )

        score_max = max(
            score_max,
            score,
        )

    validation_info = {
        "num_predictions": len(canonical),
        "num_images_with_predictions": len(image_counts),
        "prediction_counts_by_category": {
            str(category_id): class_counts.get(
                category_id,
                0,
            )
            for category_id in sorted(
                BDD100K_CATEGORIES
            )
        },
        "out_of_bounds_bbox_count": out_of_bounds_count,
        "negative_xy_bbox_count": negative_xy_count,
        "negative_size_bbox_count": negative_size_bbox_count,
        "zero_area_bbox_count": zero_area_bbox_count,
        "minimum_score": score_min,
        "maximum_score": score_max,
    }

    return canonical, validation_info


def save_json(
    data: Any,
    path: Path,
    indent: int | None = 2,
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
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )


def mean_valid(values: np.ndarray) -> float:
    ""




    valid = values[values > -1]

    if valid.size == 0:
        return -1.0

    return float(np.mean(valid))


def extract_per_class_metrics(
    coco_eval: COCOeval,
    coco_gt: COCO,
    canonical_predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ""














    precision = coco_eval.eval["precision"]
    recall = coco_eval.eval["recall"]

    params = coco_eval.params

    cat_ids = list(params.catIds)

    area_all_index = params.areaRngLbl.index(
        "all"
    )

    max_det_100_index = params.maxDets.index(
        100
    )

    iou_50_indices = np.where(
        np.isclose(
            params.iouThrs,
            0.50,
        )
    )[0]

    iou_75_indices = np.where(
        np.isclose(
            params.iouThrs,
            0.75,
        )
    )[0]

    if len(iou_50_indices) != 1:
        raise RuntimeError(
            "COCO evaluator does not contain exactly one IoU=0.50."
        )

    if len(iou_75_indices) != 1:
        raise RuntimeError(
            "COCO evaluator does not contain exactly one IoU=0.75."
        )

    iou_50_index = int(iou_50_indices[0])

    iou_75_index = int(iou_75_indices[0])

    prediction_counts = Counter(
        prediction["category_id"]
        for prediction in canonical_predictions
    )

    gt_counts = Counter(
        annotation["category_id"]
        for annotation in coco_gt.dataset["annotations"]
    )

    gt_images_by_class: dict[int, set[int]] = defaultdict(set)
    pred_images_by_class: dict[int, set[int]] = defaultdict(set)

    for annotation in coco_gt.dataset["annotations"]:
        gt_images_by_class[
            int(annotation["category_id"])
        ].add(
            int(annotation["image_id"])
        )

    for prediction in canonical_predictions:
        pred_images_by_class[
            int(prediction["category_id"])
        ].add(
            int(prediction["image_id"])
        )

    rows: list[dict[str, Any]] = []

    for category_index, category_id in enumerate(cat_ids):
        category = coco_gt.loadCats(
            [category_id]
        )[0]

        # AP@[.50:.95]
        ap_values = precision[
            :,
            :,
            category_index,
            area_all_index,
            max_det_100_index,
        ]

        # AP50
        ap50_values = precision[
            iou_50_index,
            :,
            category_index,
            area_all_index,
            max_det_100_index,
        ]

        # AP75
        ap75_values = precision[
            iou_75_index,
            :,
            category_index,
            area_all_index,
            max_det_100_index,
        ]

        # AR100：

        ar100_values = recall[
            :,
            category_index,
            area_all_index,
            max_det_100_index,
        ]

        rows.append(
            {
                "category_id": int(category_id),
                "class_name": category["name"],
                "AP": mean_valid(ap_values),
                "AP50": mean_valid(ap50_values),
                "AP75": mean_valid(ap75_values),
                "AR100": mean_valid(ar100_values),
                "num_gt": int(
                    gt_counts[category_id]
                ),
                "num_predictions": int(
                    prediction_counts[category_id]
                ),
                "num_images_with_gt": len(
                    gt_images_by_class[category_id]
                ),
                "num_images_with_predictions": len(
                    pred_images_by_class[category_id]
                ),
            }
        )

    return rows


def run_coco_evaluation(
    coco_gt: COCO,
    canonical_predictions: list[dict[str, Any]],
) -> tuple[
    COCOeval,
    dict[str, float],
    list[dict[str, Any]],
    str,
]:
    ""


    # --------------------------------------------------------

    # --------------------------------------------------------

    coco_dt = coco_gt.loadRes(
        canonical_predictions
    )

    coco_eval = COCOeval(
        coco_gt,
        coco_dt,
        iouType="bbox",
    )

    # --------------------------------------------------------

    #



    # --------------------------------------------------------

    coco_eval.params.imgIds = sorted(
        coco_gt.getImgIds()
    )

    coco_eval.params.catIds = sorted(
        BDD100K_CATEGORIES.keys()
    )

    coco_eval.params.iouThrs = np.linspace(
        0.50,
        0.95,
        10,
    )

    coco_eval.params.recThrs = np.linspace(
        0.00,
        1.00,
        101,
    )

    coco_eval.params.maxDets = [
        1,
        10,
        100,
    ]

    coco_eval.params.areaRng = [
        [0**2, 100000**2],
        [0**2, 32**2],
        [32**2, 96**2],
        [96**2, 100000**2],
    ]

    coco_eval.params.areaRngLbl = [
        "all",
        "small",
        "medium",
        "large",
    ]

    coco_eval.params.useCats = 1

    # --------------------------------------------------------

    # evaluate -> accumulate -> summarize
    # --------------------------------------------------------

    coco_eval.evaluate()
    coco_eval.accumulate()

    summary_buffer = io.StringIO()

    with contextlib.redirect_stdout(
        summary_buffer
    ):
        coco_eval.summarize()

    summary_text = summary_buffer.getvalue()


    print(summary_text, end="")

    if len(coco_eval.stats) != 12:
        raise RuntimeError(
            f"Unexpected COCO stats length: "
            f"{len(coco_eval.stats)}"
        )

    overall_metrics = {
        metric_name: float(metric_value)
        for metric_name, metric_value in zip(
            OVERALL_METRIC_NAMES,
            coco_eval.stats,
        )
    }

    per_class_metrics = (
        extract_per_class_metrics(
            coco_eval,
            coco_gt,
            canonical_predictions,
        )
    )

    return (
        coco_eval,
        overall_metrics,
        per_class_metrics,
        summary_text,
    )


def write_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    ""


    if not rows:
        raise ValueError(
            f"Cannot write empty CSV: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0].keys()
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


def save_eval_arrays(
    coco_eval: COCOeval,
    path: Path,
) -> None:
    ""









    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,
        precision=coco_eval.eval["precision"],
        recall=coco_eval.eval["recall"],
        scores=coco_eval.eval["scores"],
        iou_thresholds=coco_eval.params.iouThrs,
        recall_thresholds=coco_eval.params.recThrs,
        category_ids=np.asarray(
            coco_eval.params.catIds,
            dtype=np.int64,
        ),
        max_dets=np.asarray(
            coco_eval.params.maxDets,
            dtype=np.int64,
        ),
        area_ranges=np.asarray(
            coco_eval.params.areaRng,
            dtype=np.float64,
        ),
    )


def main() -> None:
    args = parse_args()

    gt_path = args.gt.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not gt_path.is_file():
        raise FileNotFoundError(
            f"GT annotation does not exist: {gt_path}"
        )

    prediction_specs = parse_prediction_specs(
        args.prediction
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    canonical_dir = (
        output_dir
        / "canonical_predictions"
    )

    model_result_dir = (
        output_dir
        / "models"
    )

    canonical_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("BDD100K UNIFIED DETECTION EVALUATION")
    print("=" * 78)
    print(f"GT          : {gt_path}")
    print(f"Models      : {len(prediction_specs)}")
    print(f"Output      : {output_dir}")
    print("=" * 78)

    # ========================================================

    # ========================================================

    print()
    print("[1/3] Validating original BDD100K validation GT...")

    gt_info = validate_bdd100k_gt(
        gt_path
    )

    print(
        f"GT images       : {gt_info['num_images']}"
    )

    print(
        f"GT annotations  : "
        f"{gt_info['num_annotations']}"
    )

    print(
        "GT categories   : "
        f"{sorted(gt_info['category_ids'])}"
    )

    print(
        "GT validation   : OK"
    )

    # ========================================================

    #

    # ========================================================

    coco_gt = COCO(
        str(gt_path)
    )

    comparison_rows: list[dict[str, Any]] = []
    all_per_class_rows: list[dict[str, Any]] = []

    model_summaries: dict[str, Any] = {}
    prediction_hashes: dict[str, str] = {}

    # ========================================================


    # ========================================================

    for model_index, (
        model_name,
        prediction_path,
    ) in enumerate(
        prediction_specs,
        start=1,
    ):
        print()
        print("=" * 78)
        print(
            f"[2/3] Model {model_index}/"
            f"{len(prediction_specs)}: {model_name}"
        )
        print("=" * 78)
        print(
            f"Prediction file : {prediction_path}"
        )

        safe_model_name = sanitize_model_name(
            model_name
        )

        this_model_dir = (
            model_result_dir
            / safe_model_name
        )

        this_model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        raw_predictions = load_prediction_list(
            prediction_path
        )

        (
            canonical_predictions,
            validation_info,
        ) = validate_and_canonicalize_predictions(
            raw_predictions,
            gt_info,
        )

        print(
            f"Predictions     : "
            f"{validation_info['num_predictions']}"
        )

        print(
            f"Images predicted: "
            f"{validation_info['num_images_with_predictions']}"
        )

        print(
            f"Score range     : "
            f"{validation_info['minimum_score']:.6f} "
            f"~ "
            f"{validation_info['maximum_score']:.6f}"
        )

        print(
            f"Out-of-bounds   : "
            f"{validation_info['out_of_bounds_bbox_count']}"
        )

        print(
            f"Zero-area bbox  : "
            f"{validation_info['zero_area_bbox_count']}"
        )

        # ----------------------------------------------------

        #


        # ----------------------------------------------------

        canonical_prediction_path = (
            canonical_dir
            / f"{safe_model_name}.json"
        )

        save_json(
            canonical_predictions,
            canonical_prediction_path,
            indent=None,
        )

        prediction_hash = sha256_file(
            prediction_path
        )

        prediction_hashes[
            model_name
        ] = prediction_hash

        # ----------------------------------------------------

        # ----------------------------------------------------

        (
            coco_eval,
            overall_metrics,
            per_class_metrics,
            coco_summary_text,
        ) = run_coco_evaluation(
            coco_gt,
            canonical_predictions,
        )

        # ----------------------------------------------------

        # ----------------------------------------------------

        with (
            this_model_dir
            / "coco_summary.txt"
        ).open(
            "w",
            encoding="utf-8",
        ) as f:
            f.write(
                coco_summary_text
            )

        # ----------------------------------------------------

        # ----------------------------------------------------

        overall_row = {
            "model": model_name,
            **overall_metrics,
            "num_predictions": (
                validation_info[
                    "num_predictions"
                ]
            ),
            "num_images_with_predictions": (
                validation_info[
                    "num_images_with_predictions"
                ]
            ),
        }

        comparison_rows.append(
            overall_row
        )

        write_csv(
            [overall_row],
            this_model_dir
            / "overall_metrics.csv",
        )

        # ----------------------------------------------------

        # ----------------------------------------------------

        model_per_class_rows = []

        for row in per_class_metrics:
            model_row = {
                "model": model_name,
                **row,
            }

            model_per_class_rows.append(
                model_row
            )

            all_per_class_rows.append(
                model_row
            )

        write_csv(
            model_per_class_rows,
            this_model_dir
            / "per_class_metrics.csv",
        )

        # ----------------------------------------------------

        # ----------------------------------------------------

        save_eval_arrays(
            coco_eval,
            this_model_dir
            / "coco_eval_arrays.npz",
        )

        # ----------------------------------------------------

        # ----------------------------------------------------

        model_summary = {
            "model": model_name,
            "prediction_source": str(
                prediction_path
            ),
            "prediction_sha256": (
                prediction_hash
            ),
            "canonical_prediction": str(
                canonical_prediction_path
            ),
            "canonical_prediction_sha256": (
                sha256_file(
                    canonical_prediction_path
                )
            ),
            "validation": validation_info,
            "overall_metrics": (
                overall_metrics
            ),
            "per_class_metrics": (
                per_class_metrics
            ),
        }

        model_summaries[
            model_name
        ] = model_summary

        save_json(
            model_summary,
            this_model_dir
            / "summary.json",
        )

    # ========================================================

    # ========================================================

    print()
    print("=" * 78)
    print("[3/3] Writing unified comparison...")
    print("=" * 78)

    write_csv(
        comparison_rows,
        output_dir
        / "model_comparison.csv",
    )

    write_csv(
        all_per_class_rows,
        output_dir
        / "per_class_metrics.csv",
    )

    # ========================================================
    # Metadata：

    # ========================================================

    evaluation_metadata = {
        "evaluation_name": (
            "BDD100K Unified Detection Evaluation"
        ),
        "task": "bbox_detection",
        "evaluation_protocol": (
            "COCO bbox evaluation"
        ),
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "ground_truth": {
            "path": str(gt_path),
            "sha256": sha256_file(
                gt_path
            ),
            "num_images": (
                gt_info["num_images"]
            ),
            "num_annotations": (
                gt_info[
                    "num_annotations"
                ]
            ),
            "categories": (
                BDD100K_CATEGORIES
            ),
        },
        "predictions": {
            model_name: {
                "path": str(
                    prediction_path
                ),
                "sha256": (
                    prediction_hashes[
                        model_name
                    ]
                ),
            }
            for (
                model_name,
                prediction_path,
            ) in prediction_specs
        },
        "evaluator_parameters": {
            "iou_type": "bbox",
            "iou_thresholds": [
                round(
                    float(value),
                    2,
                )
                for value in np.linspace(
                    0.50,
                    0.95,
                    10,
                )
            ],
            "recall_threshold_count": 101,
            "max_dets": [
                1,
                10,
                100,
            ],
            "area_ranges": {
                "all": [
                    0,
                    100000**2,
                ],
                "small": [
                    0,
                    32**2,
                ],
                "medium": [
                    32**2,
                    96**2,
                ],
                "large": [
                    96**2,
                    100000**2,
                ],
            },
            "use_categories": True,
        },
        "software": {
            "python": (
                platform.python_version()
            ),
            "numpy": (
                np.__version__
            ),
            "pycocotools": (
                safe_package_version(
                    "pycocotools"
                )
            ),
        },
        "models": model_summaries,
    }

    save_json(
        evaluation_metadata,
        output_dir
        / "evaluation_metadata.json",
    )

    save_json(
        {
            "models": model_summaries,
        },
        output_dir
        / "summary.json",
    )

    # ========================================================

    # ========================================================

    print()
    print("=" * 78)
    print("FINAL UNIFIED RESULTS")
    print("=" * 78)

    for row in comparison_rows:
        print(
            f"{row['model']:<20} "
            f"AP={row['AP']:.4f}  "
            f"AP50={row['AP50']:.4f}  "
            f"AP75={row['AP75']:.4f}  "
            f"APs={row['APs']:.4f}  "
            f"APm={row['APm']:.4f}  "
            f"APl={row['APl']:.4f}"
        )

    print("=" * 78)
    print(
        f"Results saved to: {output_dir}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
