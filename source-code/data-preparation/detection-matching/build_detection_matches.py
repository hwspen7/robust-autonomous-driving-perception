from __future__ import annotations





















































import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


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

EXPECTED_IMAGES = 10_000
EXPECTED_ANNOTATIONS = 185_523

COCO_SMALL_MAX = 32 ** 2
COCO_MEDIUM_MAX = 96 ** 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build YOLO11m / D-FINE-M object-level "
            "detection matches for BDD100K failure diagnosis."
        )
    )

    parser.add_argument(
        "--gt",
        type=Path,
        required=True,
        help="BDD100K original val COCO GT.",
    )

    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help=(
            "MODEL=/path/to/canonical_predictions.json. "
            "Repeat for YOLO11m and D-FINE-M."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.25,
        help=(
            "Common operating-point confidence threshold used only "
            "for object-level failure diagnosis. Default: 0.25."
        ),
    )

    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.50,
        help="IoU threshold for a correct detection. Default: 0.50.",
    )

    parser.add_argument(
        "--localization-iou",
        type=float,
        default=0.10,
        help=(
            "Minimum IoU used to call an unmatched same-class "
            "prediction a localization-error candidate. Default: 0.10."
        ),
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


def normalize_name(value: Any) -> str:
    return str(value).strip().lower()


def classify_scale(area: float) -> str:
    if area < COCO_SMALL_MAX:
        return "small"

    if area < COCO_MEDIUM_MAX:
        return "medium"

    return "large"


def xywh_to_xyxy(
    bbox: list[float],
) -> np.ndarray:
    x, y, width, height = bbox

    return np.asarray(
        [
            x,
            y,
            x + width,
            y + height,
        ],
        dtype=np.float64,
    )


def bbox_iou(
    box_a: np.ndarray,
    box_b: np.ndarray,
) -> float:
    ""








    width_a = box_a[2] - box_a[0]
    height_a = box_a[3] - box_a[1]

    width_b = box_b[2] - box_b[0]
    height_b = box_b[3] - box_b[1]

    if (
        width_a <= 0
        or height_a <= 0
        or width_b <= 0
        or height_b <= 0
    ):
        return 0.0

    inter_x1 = max(
        box_a[0],
        box_b[0],
    )

    inter_y1 = max(
        box_a[1],
        box_b[1],
    )

    inter_x2 = min(
        box_a[2],
        box_b[2],
    )

    inter_y2 = min(
        box_a[3],
        box_b[3],
    )

    inter_width = max(
        0.0,
        inter_x2 - inter_x1,
    )

    inter_height = max(
        0.0,
        inter_y2 - inter_y1,
    )

    intersection = (
        inter_width
        * inter_height
    )

    area_a = (
        width_a
        * height_a
    )

    area_b = (
        width_b
        * height_b
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return float(
        intersection / union
    )


def parse_prediction_args(
    values: list[str],
) -> dict[str, Path]:
    result = {}

    for item in values:
        if "=" not in item:
            raise ValueError(
                "--prediction must use MODEL=/path/predictions.json"
            )

        model, path_string = item.split(
            "=",
            1,
        )

        model = model.strip()

        path = (
            Path(path_string)
            .expanduser()
            .resolve()
        )

        if not model:
            raise ValueError(
                "Model name cannot be empty."
            )

        if model in result:
            raise ValueError(
                f"Duplicate model: {model}"
            )

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        result[model] = path

    if set(result) != {
        "YOLO11m",
        "D-FINE-M",
    }:
        raise ValueError(
            "This experiment requires exactly:\n"
            "YOLO11m and D-FINE-M\n"
            f"Found: {sorted(result)}"
        )

    return result


def load_gt(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if len(data["images"]) != EXPECTED_IMAGES:
        raise ValueError(
            "Unexpected GT image count."
        )

    if len(
        data["annotations"]
    ) != EXPECTED_ANNOTATIONS:
        raise ValueError(
            "Unexpected GT annotation count."
        )

    actual_categories = {
        int(category["id"]): normalize_name(
            category["name"]
        )
        for category in data[
            "categories"
        ]
    }

    expected_categories = {
        category_id: normalize_name(name)
        for category_id, name
        in BDD100K_CATEGORIES.items()
    }

    if actual_categories != expected_categories:
        raise ValueError(
            "BDD100K category definition mismatch."
        )

    return data


def build_gt_records(
    gt: dict[str, Any],
) -> tuple[
    dict[int, dict[str, Any]],
    dict[int, list[dict[str, Any]]],
]:
    image_by_id = {
        int(image["id"]): image
        for image in gt["images"]
    }

    gt_records: dict[
        int,
        dict[str, Any],
    ] = {}

    gt_by_image: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for annotation in gt[
        "annotations"
    ]:
        annotation_id = int(
            annotation["id"]
        )

        image_id = int(
            annotation["image_id"]
        )

        category_id = int(
            annotation["category_id"]
        )

        bbox = [
            float(value)
            for value
            in annotation["bbox"]
        ]

        area = float(
            annotation["area"]
        )

        image = image_by_id[
            image_id
        ]

        attributes = image.get(
            "attributes",
            {},
        )

        record = {
            "annotation_id": (
                annotation_id
            ),
            "image_id": image_id,
            "file_name": str(
                image["file_name"]
            ),
            "category_id": (
                category_id
            ),
            "category": (
                BDD100K_CATEGORIES[
                    category_id
                ]
            ),
            "gt_x": bbox[0],
            "gt_y": bbox[1],
            "gt_width": bbox[2],
            "gt_height": bbox[3],
            "gt_area": area,
            "scale": classify_scale(
                area
            ),
            "weather": normalize_name(
                attributes.get(
                    "weather",
                    "undefined",
                )
            ),
            "scene": normalize_name(
                attributes.get(
                    "scene",
                    "undefined",
                )
            ),
            "timeofday": normalize_name(
                attributes.get(
                    "timeofday",
                    "undefined",
                )
            ),
        }

        gt_records[
            annotation_id
        ] = record

        gt_by_image[
            image_id
        ].append(
            {
                **record,
                "_box_xyxy": xywh_to_xyxy(
                    bbox
                ),
            }
        )

    return (
        gt_records,
        gt_by_image,
    )


def load_predictions_by_image(
    path: Path,
) -> tuple[
    dict[int, list[dict[str, Any]]],
    int,
]:
    print(
        f"Loading predictions: {path}"
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        predictions = json.load(f)

    grouped: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for prediction_index, prediction in enumerate(
        predictions
    ):
        image_id = int(
            prediction["image_id"]
        )

        category_id = int(
            prediction["category_id"]
        )

        score = float(
            prediction["score"]
        )

        bbox = [
            float(value)
            for value
            in prediction["bbox"]
        ]

        if category_id not in BDD100K_CATEGORIES:
            raise RuntimeError(
                f"Invalid category_id={category_id}"
            )

        if not math.isfinite(
            score
        ):
            raise RuntimeError(
                "Non-finite prediction score."
            )

        grouped[
            image_id
        ].append(
            {
                "prediction_index": (
                    prediction_index
                ),
                "image_id": image_id,
                "category_id": (
                    category_id
                ),
                "score": score,
                "bbox": bbox,
                "_box_xyxy": (
                    xywh_to_xyxy(
                        bbox
                    )
                ),
            }
        )

    total = len(
        predictions
    )

    del predictions

    return (
        grouped,
        total,
    )


def best_candidate_for_gt(
    gt: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    category_id: int | None = None,
    exclude_category_id: int | None = None,
    minimum_score: float | None = None,
) -> tuple[
    dict[str, Any] | None,
    float,
]:
    ""







    best_prediction = None
    best_iou = 0.0

    for prediction in predictions:
        if (
            category_id is not None
            and prediction[
                "category_id"
            ] != category_id
        ):
            continue

        if (
            exclude_category_id
            is not None
            and prediction[
                "category_id"
            ] == exclude_category_id
        ):
            continue

        if (
            minimum_score is not None
            and prediction[
                "score"
            ] < minimum_score
        ):
            continue

        iou = bbox_iou(
            gt["_box_xyxy"],
            prediction[
                "_box_xyxy"
            ],
        )

        if iou > best_iou:
            best_iou = iou
            best_prediction = (
                prediction
            )

    return (
        best_prediction,
        best_iou,
    )


def process_model(
    model_name: str,
    prediction_path: Path,
    gt_records: dict[
        int,
        dict[str, Any],
    ],
    gt_by_image: dict[
        int,
        list[dict[str, Any]],
    ],
    *,
    score_threshold: float,
    match_iou: float,
    localization_iou: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    predictions_by_image, total_predictions = (
        load_predictions_by_image(
            prediction_path
        )
    )

    print()
    print(
        f"Matching {model_name}: "
        f"{total_predictions} predictions"
    )

    # annotation_id -> detector result
    detector_results: dict[
        int,
        dict[str, Any],
    ] = {}

    false_positive_rows = []

    summary_counter = Counter()

    processed_images = 0

    for image_id in sorted(
        gt_by_image
    ):
        image_gt = gt_by_image[
            image_id
        ]

        image_predictions = (
            predictions_by_image.get(
                image_id,
                [],
            )
        )

        # ----------------------------------------------------

        # ----------------------------------------------------

        operating_predictions = [
            prediction
            for prediction
            in image_predictions
            if prediction[
                "score"
            ] >= score_threshold
        ]

        # ----------------------------------------------------

        #



        # ----------------------------------------------------

        matched_prediction_indices = set()
        matched_annotation_ids = set()

        for category_id in BDD100K_CATEGORIES:
            category_gt = [
                gt
                for gt in image_gt
                if gt[
                    "category_id"
                ] == category_id
            ]

            category_predictions = [
                prediction
                for prediction
                in operating_predictions
                if prediction[
                    "category_id"
                ] == category_id
            ]

            category_predictions.sort(
                key=lambda prediction: (
                    prediction["score"]
                ),
                reverse=True,
            )

            for prediction in category_predictions:
                best_gt = None
                best_iou = 0.0

                for gt in category_gt:
                    annotation_id = (
                        gt[
                            "annotation_id"
                        ]
                    )

                    if (
                        annotation_id
                        in matched_annotation_ids
                    ):
                        continue

                    iou = bbox_iou(
                        gt[
                            "_box_xyxy"
                        ],
                        prediction[
                            "_box_xyxy"
                        ],
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt

                if (
                    best_gt is not None
                    and best_iou >= match_iou
                ):
                    annotation_id = int(
                        best_gt[
                            "annotation_id"
                        ]
                    )

                    prediction_index = int(
                        prediction[
                            "prediction_index"
                        ]
                    )

                    matched_annotation_ids.add(
                        annotation_id
                    )

                    matched_prediction_indices.add(
                        prediction_index
                    )

                    detector_results[
                        annotation_id
                    ] = {
                        "status": "detected",
                        "detected": True,
                        "matched_iou": (
                            best_iou
                        ),
                        "matched_score": (
                            prediction[
                                "score"
                            ]
                        ),
                        "matched_pred_category_id": (
                            prediction[
                                "category_id"
                            ]
                        ),
                        "matched_pred_bbox": (
                            prediction[
                                "bbox"
                            ]
                        ),
                    }

        # ----------------------------------------------------

        # ----------------------------------------------------

        for gt in image_gt:
            annotation_id = int(
                gt["annotation_id"]
            )

            if annotation_id in detector_results:
                summary_counter[
                    "detected"
                ] += 1
                continue

            category_id = int(
                gt["category_id"]
            )


            best_same_all, best_same_all_iou = (
                best_candidate_for_gt(
                    gt,
                    image_predictions,
                    category_id=category_id,
                )
            )


            best_same_op, best_same_op_iou = (
                best_candidate_for_gt(
                    gt,
                    image_predictions,
                    category_id=category_id,
                    minimum_score=score_threshold,
                )
            )


            best_wrong, best_wrong_iou = (
                best_candidate_for_gt(
                    gt,
                    image_predictions,
                    exclude_category_id=category_id,
                    minimum_score=score_threshold,
                )
            )

            status = "missed"

            # ------------------------------------------------

            #


            # ------------------------------------------------

            if (
                best_same_all is not None
                and best_same_all_iou
                >= match_iou
                and best_same_all[
                    "score"
                ] < score_threshold
            ):
                status = (
                    "low_confidence_candidate"
                )

            # ------------------------------------------------

            #

            # ------------------------------------------------

            elif (
                best_wrong is not None
                and best_wrong_iou
                >= match_iou
            ):
                status = (
                    "class_confusion_candidate"
                )

            # ------------------------------------------------

            #


            # ------------------------------------------------

            elif (
                best_same_op is not None
                and localization_iou
                <= best_same_op_iou
                < match_iou
            ):
                status = (
                    "localization_candidate"
                )

            detector_results[
                annotation_id
            ] = {
                "status": status,
                "detected": False,
                "matched_iou": None,
                "matched_score": None,
                "matched_pred_category_id": None,
                "matched_pred_bbox": None,

                "best_same_class_iou_all": (
                    best_same_all_iou
                ),
                "best_same_class_score_all": (
                    (
                        best_same_all[
                            "score"
                        ]
                    )
                    if best_same_all
                    is not None
                    else None
                ),

                "best_same_class_iou_op": (
                    best_same_op_iou
                ),
                "best_same_class_score_op": (
                    (
                        best_same_op[
                            "score"
                        ]
                    )
                    if best_same_op
                    is not None
                    else None
                ),

                "best_wrong_class_iou": (
                    best_wrong_iou
                ),
                "best_wrong_class_score": (
                    (
                        best_wrong[
                            "score"
                        ]
                    )
                    if best_wrong
                    is not None
                    else None
                ),
                "best_wrong_class_category_id": (
                    (
                        best_wrong[
                            "category_id"
                        ]
                    )
                    if best_wrong
                    is not None
                    else None
                ),
                "best_wrong_class_category": (
                    BDD100K_CATEGORIES[
                        best_wrong[
                            "category_id"
                        ]
                    ]
                    if best_wrong
                    is not None
                    else None
                ),
            }

            summary_counter[
                status
            ] += 1

        # ----------------------------------------------------
        # False Positive：

        # ----------------------------------------------------

        for prediction in operating_predictions:
            prediction_index = int(
                prediction[
                    "prediction_index"
                ]
            )

            if (
                prediction_index
                in matched_prediction_indices
            ):
                continue




            best_gt = None
            best_gt_iou = 0.0

            for gt in image_gt:
                iou = bbox_iou(
                    gt[
                        "_box_xyxy"
                    ],
                    prediction[
                        "_box_xyxy"
                    ],
                )

                if iou > best_gt_iou:
                    best_gt_iou = iou
                    best_gt = gt

            fp_type = (
                "background_or_unmatched"
            )

            if (
                best_gt is not None
                and best_gt_iou >= match_iou
            ):
                if (
                    prediction[
                        "category_id"
                    ]
                    != best_gt[
                        "category_id"
                    ]
                ):
                    fp_type = (
                        "class_confusion_fp"
                    )
                else:
                    fp_type = (
                        "duplicate_fp"
                    )

            elif (
                best_gt is not None
                and best_gt_iou
                >= localization_iou
            ):
                fp_type = (
                    "localization_fp"
                )

            false_positive_rows.append(
                {
                    "model": model_name,
                    "image_id": image_id,
                    "prediction_index": (
                        prediction_index
                    ),
                    "category_id": (
                        prediction[
                            "category_id"
                        ]
                    ),
                    "category": (
                        BDD100K_CATEGORIES[
                            prediction[
                                "category_id"
                            ]
                        ]
                    ),
                    "score": (
                        prediction[
                            "score"
                        ]
                    ),
                    "pred_x": (
                        prediction[
                            "bbox"
                        ][0]
                    ),
                    "pred_y": (
                        prediction[
                            "bbox"
                        ][1]
                    ),
                    "pred_width": (
                        prediction[
                            "bbox"
                        ][2]
                    ),
                    "pred_height": (
                        prediction[
                            "bbox"
                        ][3]
                    ),
                    "fp_type": fp_type,
                    "best_gt_iou": (
                        best_gt_iou
                    ),
                    "best_gt_annotation_id": (
                        (
                            best_gt[
                                "annotation_id"
                            ]
                        )
                        if best_gt
                        is not None
                        else None
                    ),
                    "best_gt_category_id": (
                        (
                            best_gt[
                                "category_id"
                            ]
                        )
                        if best_gt
                        is not None
                        else None
                    ),
                }
            )

        processed_images += 1

        if (
            processed_images
            % 1000
            == 0
        ):
            print(
                f"[{processed_images:5d}/10000] "
                f"{model_name}"
            )

    # --------------------------------------------------------
    # Merge detector result onto GT records。
    # --------------------------------------------------------

    model_rows = []

    prefix = (
        "yolo"
        if model_name
        == "YOLO11m"
        else "dfine"
    )

    for annotation_id in sorted(
        gt_records
    ):
        result = detector_results[
            annotation_id
        ]

        row = {
            "annotation_id": (
                annotation_id
            ),
            f"{prefix}_status": (
                result[
                    "status"
                ]
            ),
            f"{prefix}_detected": (
                result[
                    "detected"
                ]
            ),
            f"{prefix}_matched_iou": (
                result.get(
                    "matched_iou"
                )
            ),
            f"{prefix}_matched_score": (
                result.get(
                    "matched_score"
                )
            ),
            f"{prefix}_best_same_class_iou_all": (
                result.get(
                    "best_same_class_iou_all"
                )
            ),
            f"{prefix}_best_same_class_score_all": (
                result.get(
                    "best_same_class_score_all"
                )
            ),
            f"{prefix}_best_same_class_iou_op": (
                result.get(
                    "best_same_class_iou_op"
                )
            ),
            f"{prefix}_best_same_class_score_op": (
                result.get(
                    "best_same_class_score_op"
                )
            ),
            f"{prefix}_best_wrong_class_iou": (
                result.get(
                    "best_wrong_class_iou"
                )
            ),
            f"{prefix}_best_wrong_class_score": (
                result.get(
                    "best_wrong_class_score"
                )
            ),
            f"{prefix}_best_wrong_class_category_id": (
                result.get(
                    "best_wrong_class_category_id"
                )
            ),
            f"{prefix}_best_wrong_class_category": (
                result.get(
                    "best_wrong_class_category"
                )
            ),
        }

        model_rows.append(
            row
        )

    model_summary = {
        "model": model_name,
        "prediction_file": str(
            prediction_path
        ),
        "prediction_sha256": (
            sha256_file(
                prediction_path
            )
        ),
        "num_predictions_total": (
            total_predictions
        ),
        "score_threshold": (
            score_threshold
        ),
        "match_iou": (
            match_iou
        ),
        "localization_iou": (
            localization_iou
        ),
        "gt_status_counts": dict(
            summary_counter
        ),
        "false_positive_count": len(
            false_positive_rows
        ),
    }


    del predictions_by_image

    return (
        model_rows,
        false_positive_rows,
        model_summary,
    )


def main() -> None:
    args = parse_args()

    if not (
        0.0
        <= args.score_threshold
        <= 1.0
    ):
        raise ValueError(
            "--score-threshold must be in [0,1]."
        )

    if not (
        0.0
        < args.match_iou
        <= 1.0
    ):
        raise ValueError(
            "--match-iou must be in (0,1]."
        )

    if not (
        0.0
        <= args.localization_iou
        < args.match_iou
    ):
        raise ValueError(
            "--localization-iou must be >=0 "
            "and < --match-iou."
        )

    gt_path = (
        args.gt
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    prediction_paths = (
        parse_prediction_args(
            args.prediction
        )
    )

    if not gt_path.is_file():
        raise FileNotFoundError(
            gt_path
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gt = load_gt(
        gt_path
    )

    (
        gt_records,
        gt_by_image,
    ) = build_gt_records(
        gt
    )

    start_time = time.time()

    model_match_rows = {}
    model_summaries = {}

    all_false_positive_rows = []

    # --------------------------------------------------------



    # --------------------------------------------------------

    for model_name in (
        "YOLO11m",
        "D-FINE-M",
    ):
        print()
        print("=" * 80)
        print(
            f"BUILDING MATCHES: {model_name}"
        )
        print("=" * 80)

        (
            rows,
            fp_rows,
            summary,
        ) = process_model(
            model_name=model_name,
            prediction_path=(
                prediction_paths[
                    model_name
                ]
            ),
            gt_records=gt_records,
            gt_by_image=gt_by_image,
            score_threshold=(
                args.score_threshold
            ),
            match_iou=(
                args.match_iou
            ),
            localization_iou=(
                args.localization_iou
            ),
        )

        model_match_rows[
            model_name
        ] = {
            int(row["annotation_id"]): row
            for row in rows
        }

        model_summaries[
            model_name
        ] = summary

        all_false_positive_rows.extend(
            fp_rows
        )

    # --------------------------------------------------------

    #
    # GT info
    # + YOLO detector result
    # + D-FINE detector result
    #

    # --------------------------------------------------------

    combined_rows = []

    for annotation_id in sorted(
        gt_records
    ):
        base = dict(
            gt_records[
                annotation_id
            ]
        )

        base.update(
            model_match_rows[
                "YOLO11m"
            ][
                annotation_id
            ]
        )

        base.update(
            model_match_rows[
                "D-FINE-M"
            ][
                annotation_id
            ]
        )

        combined_rows.append(
            base
        )

    combined_fields = [
        "annotation_id",
        "image_id",
        "file_name",
        "category_id",
        "category",
        "gt_x",
        "gt_y",
        "gt_width",
        "gt_height",
        "gt_area",
        "scale",
        "weather",
        "scene",
        "timeofday",

        "yolo_status",
        "yolo_detected",
        "yolo_matched_iou",
        "yolo_matched_score",
        "yolo_best_same_class_iou_all",
        "yolo_best_same_class_score_all",
        "yolo_best_same_class_iou_op",
        "yolo_best_same_class_score_op",
        "yolo_best_wrong_class_iou",
        "yolo_best_wrong_class_score",
        "yolo_best_wrong_class_category_id",
        "yolo_best_wrong_class_category",

        "dfine_status",
        "dfine_detected",
        "dfine_matched_iou",
        "dfine_matched_score",
        "dfine_best_same_class_iou_all",
        "dfine_best_same_class_score_all",
        "dfine_best_same_class_iou_op",
        "dfine_best_same_class_score_op",
        "dfine_best_wrong_class_iou",
        "dfine_best_wrong_class_score",
        "dfine_best_wrong_class_category_id",
        "dfine_best_wrong_class_category",
    ]

    write_csv(
        output_dir
        / "gt_detection_matches.csv",
        combined_rows,
        combined_fields,
    )

    write_csv(
        output_dir
        / "false_positives.csv",
        all_false_positive_rows,
        [
            "model",
            "image_id",
            "prediction_index",
            "category_id",
            "category",
            "score",
            "pred_x",
            "pred_y",
            "pred_width",
            "pred_height",
            "fp_type",
            "best_gt_iou",
            "best_gt_annotation_id",
            "best_gt_category_id",
        ],
    )

    # --------------------------------------------------------

    #



    # --------------------------------------------------------

    architecture_matrix = Counter()

    for row in combined_rows:
        yolo = bool(
            row[
                "yolo_detected"
            ]
        )

        dfine = bool(
            row[
                "dfine_detected"
            ]
        )

        if yolo and dfine:
            key = "YOLO_OK__DFINE_OK"
        elif yolo and not dfine:
            key = "YOLO_OK__DFINE_FAIL"
        elif not yolo and dfine:
            key = "YOLO_FAIL__DFINE_OK"
        else:
            key = "YOLO_FAIL__DFINE_FAIL"

        architecture_matrix[
            key
        ] += 1

    elapsed = (
        time.time()
        - start_time
    )

    metadata = {
        "dataset": "BDD100K",
        "split": "val",
        "gt": str(
            gt_path
        ),
        "gt_sha256": (
            sha256_file(
                gt_path
            )
        ),
        "num_images": (
            EXPECTED_IMAGES
        ),
        "num_gt_objects": (
            EXPECTED_ANNOTATIONS
        ),
        "score_threshold": (
            args.score_threshold
        ),
        "match_iou": (
            args.match_iou
        ),
        "localization_iou": (
            args.localization_iou
        ),
        "models": (
            model_summaries
        ),
        "architecture_detection_matrix": dict(
            architecture_matrix
        ),
        "policy": {
            "matching": (
                "greedy one-to-one within image and category; "
                "predictions sorted by descending confidence"
            ),
            "detected": (
                "score >= operating threshold and same-class IoU >= match_iou"
            ),
            "low_confidence_candidate": (
                "same-class IoU >= match_iou but score < operating threshold"
            ),
            "class_confusion_candidate": (
                "different-class prediction with IoU >= match_iou"
            ),
            "localization_candidate": (
                "same-class operating-point prediction with "
                "localization_iou <= IoU < match_iou"
            ),
            "causal_claim": False,
            "segmentation_included": False,
            "note": (
                "This is Detection Matching only. "
                "YOLO11m-Seg will be joined in the next diagnosis stage."
            ),
        },
        "elapsed_seconds": round(
            elapsed,
            3,
        ),
    }

    write_json(
        output_dir
        / "metadata.json",
        metadata,
    )

    print()
    print("=" * 80)
    print(
        "DETECTION MATCHING COMPLETE"
    )
    print("=" * 80)

    print(
        f"GT objects: {len(combined_rows)}"
    )

    print()

    for model_name in (
        "YOLO11m",
        "D-FINE-M",
    ):
        summary = (
            model_summaries[
                model_name
            ]
        )

        print(
            model_name
        )

        for status, count in sorted(
            summary[
                "gt_status_counts"
            ].items()
        ):
            print(
                f"  {status:<30} {count}"
            )

        print(
            "  false_positive_count"
            f"{'':<10} "
            f"{summary['false_positive_count']}"
        )

        print()

    print(
        "Cross-architecture GT matrix:"
    )

    for key, count in sorted(
        architecture_matrix.items()
    ):
        print(
            f"  {key:<30} {count}"
        )

    print()

    print(
        "Next step:"
    )

    print(
        "  Join YOLO11m-Seg predictions to these exact "
        "GT annotation_id/image_id records."
    )

    print()

    print(
        f"Results: {output_dir}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()