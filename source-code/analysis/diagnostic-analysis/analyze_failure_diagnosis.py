""

















































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


SUPPORTED_SEG_CATEGORIES = {
    1: "pedestrian",
    3: "car",
    4: "truck",
    5: "bus",
    6: "train",
    7: "motorcycle",
    8: "bicycle",
    9: "traffic light",
}

UNSUPPORTED_SEG_CATEGORIES = {
    2: "rider",
    10: "traffic sign",
}

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

EXPECTED_GT_OBJECTS = 185_523


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join YOLO11m-Seg with YOLO11m/D-FINE-M "
            "GT-level detection matches."
        )
    )

    parser.add_argument(
        "--detection-matches",
        type=Path,
        required=True,
        help="gt_detection_matches.csv.",
    )

    parser.add_argument(
        "--seg-predictions",
        type=Path,
        required=True,
        help="YOLO11m-Seg predictions.json.",
    )

    parser.add_argument(
        "--seg-metadata",
        type=Path,
        default=None,
        help="Optional YOLO11m-Seg metadata.json.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--seg-score-threshold",
        type=float,
        default=0.25,
        help="Seg diagnostic operating threshold.",
    )

    parser.add_argument(
        "--seg-match-iou",
        type=float,
        default=0.50,
        help="BBox IoU required for GT ↔ Seg spatial match.",
    )

    parser.add_argument(
        "--seg-localization-iou",
        type=float,
        default=0.10,
        help=(
            "Minimum IoU used to mark an unmatched Seg prediction "
            "as a localization candidate."
        ),
    )

    parser.add_argument(
        "--min-mask-inside-ratio",
        type=float,
        default=0.50,
        help=(
            "Minimum fraction of predicted polygon area lying inside "
            "the GT bbox required for Segmentation Support."
        ),
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def parse_bool(value: Any) -> bool:
    if isinstance(
        value,
        bool,
    ):
        return value

    text = str(
        value
    ).strip().lower()

    if text in {
        "true",
        "1",
        "yes",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
    }:
        return False

    raise ValueError(
        f"Cannot parse bool: {value}"
    )


def safe_float(
    value: Any,
) -> float | None:
    if value is None:
        return None

    text = str(
        value
    ).strip()

    if text == "":
        return None

    number = float(
        text
    )

    if not math.isfinite(
        number
    ):
        return None

    return number


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

        writer.writerows(
            rows
        )


def xywh_to_xyxy(
    bbox: list[float],
) -> tuple[
    float,
    float,
    float,
    float,
]:
    x, y, width, height = bbox

    return (
        x,
        y,
        x + width,
        y + height,
    )


def bbox_iou(
    bbox_a: list[float],
    bbox_b: list[float],
) -> float:
    ""






    ax1, ay1, ax2, ay2 = (
        xywh_to_xyxy(
            bbox_a
        )
    )

    bx1, by1, bx2, by2 = (
        xywh_to_xyxy(
            bbox_b
        )
    )

    aw = ax2 - ax1
    ah = ay2 - ay1

    bw = bx2 - bx1
    bh = by2 - by1

    if (
        aw <= 0
        or ah <= 0
        or bw <= 0
        or bh <= 0
    ):
        return 0.0

    ix1 = max(
        ax1,
        bx1,
    )

    iy1 = max(
        ay1,
        by1,
    )

    ix2 = min(
        ax2,
        bx2,
    )

    iy2 = min(
        ay2,
        by2,
    )

    iw = max(
        0.0,
        ix2 - ix1,
    )

    ih = max(
        0.0,
        iy2 - iy1,
    )

    intersection = (
        iw
        * ih
    )

    area_a = (
        aw
        * ah
    )

    area_b = (
        bw
        * bh
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return float(
        intersection
        / union
    )


def polygon_area(
    points: list[
        tuple[
            float,
            float,
        ]
    ],
) -> float:
    """
    Shoelace formula。
    """
    if len(
        points
    ) < 3:
        return 0.0

    total = 0.0

    for index in range(
        len(points)
    ):
        x1, y1 = points[
            index
        ]

        x2, y2 = points[
            (
                index + 1
            )
            % len(points)
        ]

        total += (
            x1 * y2
            - x2 * y1
        )

    return abs(
        total
    ) * 0.5


def intersect_vertical(
    start: tuple[
        float,
        float,
    ],
    end: tuple[
        float,
        float,
    ],
    x_value: float,
) -> tuple[
    float,
    float,
]:
    x1, y1 = start
    x2, y2 = end

    dx = (
        x2 - x1
    )

    if abs(dx) < 1e-12:
        return (
            x_value,
            y1,
        )

    t = (
        x_value - x1
    ) / dx

    return (
        x_value,
        y1
        + t
        * (
            y2 - y1
        ),
    )


def intersect_horizontal(
    start: tuple[
        float,
        float,
    ],
    end: tuple[
        float,
        float,
    ],
    y_value: float,
) -> tuple[
    float,
    float,
]:
    x1, y1 = start
    x2, y2 = end

    dy = (
        y2 - y1
    )

    if abs(dy) < 1e-12:
        return (
            x1,
            y_value,
        )

    t = (
        y_value - y1
    ) / dy

    return (
        x1
        + t
        * (
            x2 - x1
        ),
        y_value,
    )


def clip_polygon(
    polygon: list[
        tuple[
            float,
            float,
        ]
    ],
    *,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> list[
    tuple[
        float,
        float,
    ]
]:
    ""






    def clip_against(
        points: list[
            tuple[
                float,
                float,
            ]
        ],
        inside,
        intersection,
    ) -> list[
        tuple[
            float,
            float,
        ]
    ]:
        if not points:
            return []

        result = []

        previous = points[
            -1
        ]

        previous_inside = (
            inside(
                previous
            )
        )

        for current in points:
            current_inside = (
                inside(
                    current
                )
            )

            if current_inside:
                if not previous_inside:
                    result.append(
                        intersection(
                            previous,
                            current,
                        )
                    )

                result.append(
                    current
                )

            elif previous_inside:
                result.append(
                    intersection(
                        previous,
                        current,
                    )
                )

            previous = current
            previous_inside = (
                current_inside
            )

        return result

    points = polygon

    points = clip_against(
        points,
        lambda p: p[0] >= xmin,
        lambda a, b: intersect_vertical(
            a,
            b,
            xmin,
        ),
    )

    points = clip_against(
        points,
        lambda p: p[0] <= xmax,
        lambda a, b: intersect_vertical(
            a,
            b,
            xmax,
        ),
    )

    points = clip_against(
        points,
        lambda p: p[1] >= ymin,
        lambda a, b: intersect_horizontal(
            a,
            b,
            ymin,
        ),
    )

    points = clip_against(
        points,
        lambda p: p[1] <= ymax,
        lambda a, b: intersect_horizontal(
            a,
            b,
            ymax,
        ),
    )

    return points


def polygon_support_metrics(
    segmentation: Any,
    gt_bbox: list[float],
) -> dict[str, Any]:
    ""















    gx, gy, gw, gh = (
        gt_bbox
    )

    xmin = gx
    ymin = gy
    xmax = gx + gw
    ymax = gy + gh

    total_polygon_area = 0.0
    inside_area = 0.0
    valid_polygon_count = 0

    if not isinstance(
        segmentation,
        list,
    ):
        return {
            "polygon_count": 0,
            "polygon_area": 0.0,
            "polygon_inside_gt_area": 0.0,
            "mask_inside_gt_ratio": None,
            "gt_bbox_coverage_ratio": None,
        }

    for flat_polygon in segmentation:
        if (
            not isinstance(
                flat_polygon,
                list,
            )
            or len(
                flat_polygon
            ) < 6
            or len(
                flat_polygon
            ) % 2
            != 0
        ):
            continue

        points = []

        valid = True

        for index in range(
            0,
            len(flat_polygon),
            2,
        ):
            x = float(
                flat_polygon[
                    index
                ]
            )

            y = float(
                flat_polygon[
                    index + 1
                ]
            )

            if not (
                math.isfinite(x)
                and math.isfinite(y)
            ):
                valid = False
                break

            points.append(
                (
                    x,
                    y,
                )
            )

        if not valid:
            continue

        area = polygon_area(
            points
        )

        if area <= 0:
            continue

        clipped = clip_polygon(
            points,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )

        clipped_area = (
            polygon_area(
                clipped
            )
            if len(clipped)
            >= 3
            else 0.0
        )

        total_polygon_area += (
            area
        )

        inside_area += (
            clipped_area
        )

        valid_polygon_count += 1

    if total_polygon_area > 0:
        inside_ratio = (
            inside_area
            / total_polygon_area
        )
    else:
        inside_ratio = None

    gt_area = (
        gw
        * gh
    )

    if gt_area > 0:
        gt_coverage = (
            inside_area
            / gt_area
        )
    else:
        gt_coverage = None

    return {
        "polygon_count": (
            valid_polygon_count
        ),
        "polygon_area": (
            total_polygon_area
        ),
        "polygon_inside_gt_area": (
            inside_area
        ),
        "mask_inside_gt_ratio": (
            inside_ratio
        ),
        "gt_bbox_coverage_ratio": (
            gt_coverage
        ),
    }


def load_detection_matches(
    path: Path,
) -> tuple[
    list[dict[str, Any]],
    list[str],
]:
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(
            f
        )

        original_fields = list(
            reader.fieldnames
            or []
        )

        for row in reader:
            row["annotation_id"] = int(
                row[
                    "annotation_id"
                ]
            )

            row["image_id"] = int(
                row[
                    "image_id"
                ]
            )

            row["category_id"] = int(
                row[
                    "category_id"
                ]
            )

            row["gt_x"] = float(
                row[
                    "gt_x"
                ]
            )

            row["gt_y"] = float(
                row[
                    "gt_y"
                ]
            )

            row["gt_width"] = float(
                row[
                    "gt_width"
                ]
            )

            row["gt_height"] = float(
                row[
                    "gt_height"
                ]
            )

            row["gt_area"] = float(
                row[
                    "gt_area"
                ]
            )

            row["yolo_detected"] = (
                parse_bool(
                    row[
                        "yolo_detected"
                    ]
                )
            )

            row["dfine_detected"] = (
                parse_bool(
                    row[
                        "dfine_detected"
                    ]
                )
            )

            rows.append(
                row
            )

    if len(
        rows
    ) != EXPECTED_GT_OBJECTS:
        raise RuntimeError(
            "Detection match row count mismatch.\n"
            f"Expected: {EXPECTED_GT_OBJECTS}\n"
            f"Found   : {len(rows)}"
        )

    annotation_ids = [
        int(
            row[
                "annotation_id"
            ]
        )
        for row in rows
    ]

    if len(
        annotation_ids
    ) != len(
        set(
            annotation_ids
        )
    ):
        raise RuntimeError(
            "Duplicate annotation_id in detection matches."
        )

    return (
        rows,
        original_fields,
    )


def load_seg_predictions(
    path: Path,
    *,
    score_threshold: float,
) -> tuple[
    dict[
        int,
        dict[
            int,
            list[
                dict[
                    str,
                    Any,
                ]
            ],
        ],
    ],
    dict[str, Any],
]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        predictions = json.load(
            f
        )

    grouped: dict[
        int,
        dict[
            int,
            list[
                dict[
                    str,
                    Any,
                ]
            ],
        ],
    ] = defaultdict(
        lambda: defaultdict(
            list
        )
    )

    category_counter = Counter()

    total_above_threshold = 0

    for prediction_index, prediction in enumerate(
        predictions
    ):
        image_id = int(
            prediction[
                "image_id"
            ]
        )

        category_id = int(
            prediction[
                "category_id"
            ]
        )

        score = float(
            prediction[
                "score"
            ]
        )

        if (
            category_id
            not in SUPPORTED_SEG_CATEGORIES
        ):
            raise RuntimeError(
                "Unexpected YOLO11m-Seg BDD category.\n"
                f"category_id={category_id}"
            )

        category_counter[
            category_id
        ] += 1

        if score < score_threshold:
            continue

        bbox = [
            float(value)
            for value
            in prediction[
                "bbox"
            ]
        ]

        if len(
            bbox
        ) != 4:
            raise RuntimeError(
                "Invalid Seg bbox."
            )

        grouped[
            image_id
        ][
            category_id
        ].append(
            {
                "prediction_index": (
                    prediction_index
                ),
                "score": score,
                "bbox": bbox,
                "segmentation": (
                    prediction.get(
                        "segmentation",
                        [],
                    )
                ),
                "file_name": (
                    prediction.get(
                        "file_name"
                    )
                ),
            }
        )

        total_above_threshold += 1

    for image_dict in grouped.values():
        for category_predictions in image_dict.values():
            category_predictions.sort(
                key=lambda item: (
                    item[
                        "score"
                    ]
                ),
                reverse=True,
            )

    diagnostics = {
        "num_predictions_raw": len(
            predictions
        ),
        "num_predictions_above_threshold": (
            total_above_threshold
        ),
        "num_images_with_predictions_above_threshold": len(
            grouped
        ),
        "category_counts_raw": {
            str(category_id): int(
                category_counter[
                    category_id
                ]
            )
            for category_id
            in sorted(
                category_counter
            )
        },
    }

    del predictions

    return (
        grouped,
        diagnostics,
    )


def build_seg_matches(
    gt_rows: list[
        dict[
            str,
            Any,
        ]
    ],
    seg_by_image: dict[
        int,
        dict[
            int,
            list[
                dict[
                    str,
                    Any,
                ]
            ],
        ],
    ],
    *,
    match_iou: float,
    localization_iou: float,
    min_mask_inside_ratio: float,
) -> tuple[
    dict[
        int,
        dict[
            str,
            Any,
        ]
    ],
    dict[str, Any],
]:
    ""















    gt_by_image_category: dict[
        tuple[
            int,
            int,
        ],
        list[
            dict[
                str,
                Any,
            ]
        ],
    ] = defaultdict(
        list
    )

    results: dict[
        int,
        dict[
            str,
            Any,
        ]
    ] = {}

    # --------------------------------------------------------

    # --------------------------------------------------------

    for row in gt_rows:
        annotation_id = int(
            row[
                "annotation_id"
            ]
        )

        category_id = int(
            row[
                "category_id"
            ]
        )

        if (
            category_id
            in UNSUPPORTED_SEG_CATEGORIES
        ):
            results[
                annotation_id
            ] = {
                "seg_applicable": False,
                "seg_status": (
                    "not_applicable"
                ),
                "seg_support": None,
                "seg_prediction_index": None,
                "seg_score": None,
                "seg_bbox_iou": None,
                "seg_bbox": None,
                "seg_polygon_count": None,
                "seg_polygon_area": None,
                "seg_polygon_inside_gt_area": None,
                "seg_mask_inside_gt_ratio": None,
                "seg_gt_bbox_coverage_ratio": None,
                "seg_best_same_class_iou": None,
                "seg_best_same_class_score": None,
            }

            continue

        gt_by_image_category[
            (
                int(
                    row[
                        "image_id"
                    ]
                ),
                category_id,
            )
        ].append(
            row
        )

    matched_prediction_count = 0
    supported_count = 0
    weak_region_count = 0
    unmatched_count = 0
    localization_candidate_count = 0

    # --------------------------------------------------------
    # Supported 8 classes。
    # --------------------------------------------------------

    for (
        image_id,
        category_id,
    ), category_gt in gt_by_image_category.items():

        predictions = (
            seg_by_image
            .get(
                image_id,
                {},
            )
            .get(
                category_id,
                [],
            )
        )

        matched_annotation_ids = set()

        # ----------------------------------------------------
        # Standard score-ordered greedy one-to-one matching。
        # ----------------------------------------------------

        for prediction in predictions:
            best_gt = None
            best_iou = 0.0

            for gt in category_gt:
                annotation_id = int(
                    gt[
                        "annotation_id"
                    ]
                )

                if (
                    annotation_id
                    in matched_annotation_ids
                ):
                    continue

                gt_bbox = [
                    float(
                        gt[
                            "gt_x"
                        ]
                    ),
                    float(
                        gt[
                            "gt_y"
                        ]
                    ),
                    float(
                        gt[
                            "gt_width"
                        ]
                    ),
                    float(
                        gt[
                            "gt_height"
                        ]
                    ),
                ]

                iou = bbox_iou(
                    gt_bbox,
                    prediction[
                        "bbox"
                    ],
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            if (
                best_gt is None
                or best_iou
                < match_iou
            ):
                continue

            annotation_id = int(
                best_gt[
                    "annotation_id"
                ]
            )

            matched_annotation_ids.add(
                annotation_id
            )

            gt_bbox = [
                float(
                    best_gt[
                        "gt_x"
                    ]
                ),
                float(
                    best_gt[
                        "gt_y"
                    ]
                ),
                float(
                    best_gt[
                        "gt_width"
                    ]
                ),
                float(
                    best_gt[
                        "gt_height"
                    ]
                ),
            ]

            polygon_metrics = (
                polygon_support_metrics(
                    prediction[
                        "segmentation"
                    ],
                    gt_bbox,
                )
            )

            inside_ratio = (
                polygon_metrics[
                    "mask_inside_gt_ratio"
                ]
            )

            region_supported = (
                inside_ratio
                is not None
                and inside_ratio
                >= min_mask_inside_ratio
            )

            if region_supported:
                status = (
                    "region_supported"
                )

                supported_count += 1

            else:
                status = (
                    "bbox_matched_weak_region_overlap"
                )

                weak_region_count += 1

            results[
                annotation_id
            ] = {
                "seg_applicable": True,
                "seg_status": status,
                "seg_support": (
                    region_supported
                ),
                "seg_prediction_index": (
                    prediction[
                        "prediction_index"
                    ]
                ),
                "seg_score": (
                    prediction[
                        "score"
                    ]
                ),
                "seg_bbox_iou": (
                    best_iou
                ),
                "seg_bbox": (
                    prediction[
                        "bbox"
                    ]
                ),
                "seg_polygon_count": (
                    polygon_metrics[
                        "polygon_count"
                    ]
                ),
                "seg_polygon_area": (
                    polygon_metrics[
                        "polygon_area"
                    ]
                ),
                "seg_polygon_inside_gt_area": (
                    polygon_metrics[
                        "polygon_inside_gt_area"
                    ]
                ),
                "seg_mask_inside_gt_ratio": (
                    polygon_metrics[
                        "mask_inside_gt_ratio"
                    ]
                ),
                "seg_gt_bbox_coverage_ratio": (
                    polygon_metrics[
                        "gt_bbox_coverage_ratio"
                    ]
                ),
                "seg_best_same_class_iou": (
                    best_iou
                ),
                "seg_best_same_class_score": (
                    prediction[
                        "score"
                    ]
                ),
            }

            matched_prediction_count += 1

        # ----------------------------------------------------

        # ----------------------------------------------------

        for gt in category_gt:
            annotation_id = int(
                gt[
                    "annotation_id"
                ]
            )

            if annotation_id in results:
                continue

            gt_bbox = [
                float(
                    gt[
                        "gt_x"
                    ]
                ),
                float(
                    gt[
                        "gt_y"
                    ]
                ),
                float(
                    gt[
                        "gt_width"
                    ]
                ),
                float(
                    gt[
                        "gt_height"
                    ]
                ),
            ]

            best_prediction = None
            best_iou = 0.0

            for prediction in predictions:
                iou = bbox_iou(
                    gt_bbox,
                    prediction[
                        "bbox"
                    ],
                )

                if iou > best_iou:
                    best_iou = iou
                    best_prediction = (
                        prediction
                    )

            if (
                best_prediction
                is not None
                and best_iou
                >= localization_iou
            ):
                status = (
                    "localization_candidate"
                )

                localization_candidate_count += 1

            else:
                status = (
                    "no_spatial_support"
                )

                unmatched_count += 1

            results[
                annotation_id
            ] = {
                "seg_applicable": True,
                "seg_status": status,
                "seg_support": False,
                "seg_prediction_index": None,
                "seg_score": None,
                "seg_bbox_iou": None,
                "seg_bbox": None,
                "seg_polygon_count": None,
                "seg_polygon_area": None,
                "seg_polygon_inside_gt_area": None,
                "seg_mask_inside_gt_ratio": None,
                "seg_gt_bbox_coverage_ratio": None,
                "seg_best_same_class_iou": (
                    best_iou
                ),
                "seg_best_same_class_score": (
                    (
                        best_prediction[
                            "score"
                        ]
                    )
                    if best_prediction
                    is not None
                    else None
                ),
            }

    if len(
        results
    ) != EXPECTED_GT_OBJECTS:
        raise RuntimeError(
            "Seg diagnosis result count mismatch.\n"
            f"Expected: {EXPECTED_GT_OBJECTS}\n"
            f"Found   : {len(results)}"
        )

    diagnostics = {
        "matched_seg_predictions": (
            matched_prediction_count
        ),
        "region_supported": (
            supported_count
        ),
        "bbox_matched_weak_region_overlap": (
            weak_region_count
        ),
        "localization_candidate": (
            localization_candidate_count
        ),
        "no_spatial_support": (
            unmatched_count
        ),
        "not_applicable": sum(
            1
            for result
            in results.values()
            if not result[
                "seg_applicable"
            ]
        ),
    }

    return (
        results,
        diagnostics,
    )


def classify_failure_type(
    *,
    yolo_detected: bool,
    dfine_detected: bool,
    seg_applicable: bool,
    seg_support: bool | None,
) -> str:
    ""







    if not seg_applicable:
        if (
            yolo_detected
            and dfine_detected
        ):
            return (
                "stable_detection_no_seg_evidence"
            )

        if (
            not yolo_detected
            and dfine_detected
        ):
            return (
                "yolo_architecture_sensitive_candidate_no_seg"
            )

        if (
            not yolo_detected
            and not dfine_detected
        ):
            return (
                "shared_detection_difficulty_no_seg_evidence"
            )

        return (
            "dfine_architecture_sensitive_candidate_no_seg"
        )

    assert (
        seg_support
        is not None
    )

    # --------------------------------------------------------

    # --------------------------------------------------------

    if (
        yolo_detected
        and dfine_detected
        and seg_support
    ):
        return "stable"

    if (
        yolo_detected
        and dfine_detected
        and not seg_support
    ):
        return (
            "fine_grained_perception_degradation_candidate"
        )

    # --------------------------------------------------------

    # --------------------------------------------------------

    if (
        not yolo_detected
        and dfine_detected
        and seg_support
    ):
        return (
            "yolo_specific_detection_failure_candidate"
        )

    if (
        not yolo_detected
        and dfine_detected
        and not seg_support
    ):
        return (
            "architecture_sensitive_yolo_failure_mixed_seg_evidence"
        )

    # --------------------------------------------------------

    # --------------------------------------------------------

    if (
        not yolo_detected
        and not dfine_detected
        and seg_support
    ):
        return (
            "shared_detection_difficulty_candidate"
        )

    if (
        not yolo_detected
        and not dfine_detected
        and not seg_support
    ):
        return (
            "shared_visual_hard_sample_candidate"
        )

    # --------------------------------------------------------

    # --------------------------------------------------------

    if (
        yolo_detected
        and not dfine_detected
        and seg_support
    ):
        return (
            "dfine_specific_detection_failure_candidate"
        )

    return (
        "architecture_sensitive_dfine_failure_mixed_seg_evidence"
    )


def build_group_specs(
    rows: list[
        dict[
            str,
            Any,
        ]
    ],
) -> list[
    tuple[
        str,
        str,
    ]
]:
    specs = [
        (
            "overall",
            "all",
        )
    ]

    for dimension in (
        "timeofday",
        "weather",
        "scene",
        "scale",
    ):
        values = sorted(
            {
                str(
                    row[
                        dimension
                    ]
                )
                for row in rows
            }
        )

        for value in values:
            specs.append(
                (
                    dimension,
                    value,
                )
            )

    return specs


def row_in_group(
    row: dict[
        str,
        Any,
    ],
    dimension: str,
    value: str,
) -> bool:
    if dimension == "overall":
        return True

    return str(
        row[
            dimension
        ]
    ) == value


def main() -> None:
    args = parse_args()

    if not (
        0.0
        <= args.seg_score_threshold
        <= 1.0
    ):
        raise ValueError(
            "--seg-score-threshold must be in [0,1]."
        )

    if not (
        0.0
        < args.seg_match_iou
        <= 1.0
    ):
        raise ValueError(
            "--seg-match-iou must be in (0,1]."
        )

    if not (
        0.0
        <= args.seg_localization_iou
        < args.seg_match_iou
    ):
        raise ValueError(
            "--seg-localization-iou must be >=0 "
            "and < seg-match-iou."
        )

    if not (
        0.0
        <= args.min_mask_inside_ratio
        <= 1.0
    ):
        raise ValueError(
            "--min-mask-inside-ratio must be in [0,1]."
        )

    detection_match_path = (
        args.detection_matches
        .expanduser()
        .resolve()
    )

    seg_prediction_path = (
        args.seg_predictions
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if not detection_match_path.is_file():
        raise FileNotFoundError(
            detection_match_path
        )

    if not seg_prediction_path.is_file():
        raise FileNotFoundError(
            seg_prediction_path
        )

    if args.seg_metadata is not None:
        seg_metadata_path = (
            args.seg_metadata
            .expanduser()
            .resolve()
        )
    else:
        seg_metadata_path = (
            seg_prediction_path
            .parent
            / "metadata.json"
        )

    if not seg_metadata_path.is_file():
        seg_metadata_path = None

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_time = time.time()

    # ========================================================
    # Detection matching table
    # ========================================================

    print("=" * 80)
    print(
        "LOADING DETECTION MATCHES"
    )
    print("=" * 80)

    (
        gt_rows,
        original_fields,
    ) = load_detection_matches(
        detection_match_path
    )

    print(
        f"GT objects: {len(gt_rows)}"
    )

    # ========================================================
    # Seg metadata cross-check
    # ========================================================

    seg_export_metadata = None

    if seg_metadata_path is not None:
        with seg_metadata_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            seg_export_metadata = (
                json.load(
                    f
                )
            )

        export_threshold = (
            seg_export_metadata.get(
                "confidence_threshold"
            )
        )

        if (
            export_threshold
            is not None
        ):
            export_threshold = float(
                export_threshold
            )

            print(
                "Seg export confidence threshold:",
                export_threshold,
            )



            if (
                args.seg_score_threshold
                < export_threshold
                - 1e-12
            ):
                raise RuntimeError(
                    "Requested Seg score threshold is below "
                    "the prediction export threshold.\n"
                    f"Export threshold   : {export_threshold}\n"
                    f"Requested threshold: {args.seg_score_threshold}\n"
                    "Lower-score predictions no longer exist."
                )

    # ========================================================
    # Seg predictions
    # ========================================================

    print()
    print("=" * 80)
    print(
        "LOADING YOLO11m-SEG PREDICTIONS"
    )
    print("=" * 80)

    (
        seg_by_image,
        seg_prediction_diagnostics,
    ) = load_seg_predictions(
        seg_prediction_path,
        score_threshold=(
            args.seg_score_threshold
        ),
    )

    print(
        "Raw predictions:",
        seg_prediction_diagnostics[
            "num_predictions_raw"
        ],
    )

    print(
        "Above threshold:",
        seg_prediction_diagnostics[
            "num_predictions_above_threshold"
        ],
    )

    print(
        "Images with predictions:",
        seg_prediction_diagnostics[
            "num_images_with_predictions_above_threshold"
        ],
    )

    # ========================================================
    # GT ↔ Seg matching
    # ========================================================

    print()
    print("=" * 80)
    print(
        "BUILDING GT <-> SEG MATCHES"
    )
    print("=" * 80)

    (
        seg_results,
        seg_match_diagnostics,
    ) = build_seg_matches(
        gt_rows,
        seg_by_image,
        match_iou=(
            args.seg_match_iou
        ),
        localization_iou=(
            args.seg_localization_iou
        ),
        min_mask_inside_ratio=(
            args.min_mask_inside_ratio
        ),
    )

    del seg_by_image

    # ========================================================
    # Join Detection + Seg
    # ========================================================

    diagnosis_rows = []

    failure_type_counter = Counter()

    for row in gt_rows:
        annotation_id = int(
            row[
                "annotation_id"
            ]
        )

        seg = seg_results[
            annotation_id
        ]

        yolo_detected = bool(
            row[
                "yolo_detected"
            ]
        )

        dfine_detected = bool(
            row[
                "dfine_detected"
            ]
        )

        failure_type = classify_failure_type(
            yolo_detected=(
                yolo_detected
            ),
            dfine_detected=(
                dfine_detected
            ),
            seg_applicable=(
                bool(
                    seg[
                        "seg_applicable"
                    ]
                )
            ),
            seg_support=(
                seg[
                    "seg_support"
                ]
            ),
        )

        merged = dict(
            row
        )

        merged.update(
            {
                "seg_applicable": (
                    seg[
                        "seg_applicable"
                    ]
                ),
                "seg_status": (
                    seg[
                        "seg_status"
                    ]
                ),
                "seg_support": (
                    seg[
                        "seg_support"
                    ]
                ),
                "seg_prediction_index": (
                    seg[
                        "seg_prediction_index"
                    ]
                ),
                "seg_score": (
                    seg[
                        "seg_score"
                    ]
                ),
                "seg_bbox_iou": (
                    seg[
                        "seg_bbox_iou"
                    ]
                ),
                "seg_bbox": (
                    (
                        json.dumps(
                            seg[
                                "seg_bbox"
                            ],
                            separators=(
                                ",",
                                ":",
                            ),
                        )
                    )
                    if seg[
                        "seg_bbox"
                    ]
                    is not None
                    else None
                ),
                "seg_polygon_count": (
                    seg[
                        "seg_polygon_count"
                    ]
                ),
                "seg_polygon_area": (
                    seg[
                        "seg_polygon_area"
                    ]
                ),
                "seg_polygon_inside_gt_area": (
                    seg[
                        "seg_polygon_inside_gt_area"
                    ]
                ),
                "seg_mask_inside_gt_ratio": (
                    seg[
                        "seg_mask_inside_gt_ratio"
                    ]
                ),
                "seg_gt_bbox_coverage_ratio": (
                    seg[
                        "seg_gt_bbox_coverage_ratio"
                    ]
                ),
                "seg_best_same_class_iou": (
                    seg[
                        "seg_best_same_class_iou"
                    ]
                ),
                "seg_best_same_class_score": (
                    seg[
                        "seg_best_same_class_score"
                    ]
                ),
                "failure_type": (
                    failure_type
                ),
                "primary_yolo_failure": (
                    not yolo_detected
                ),
            }
        )

        diagnosis_rows.append(
            merged
        )

        failure_type_counter[
            failure_type
        ] += 1

    # ========================================================
    # Full GT diagnosis table
    # ========================================================

    new_fields = [
        "seg_applicable",
        "seg_status",
        "seg_support",
        "seg_prediction_index",
        "seg_score",
        "seg_bbox_iou",
        "seg_bbox",
        "seg_polygon_count",
        "seg_polygon_area",
        "seg_polygon_inside_gt_area",
        "seg_mask_inside_gt_ratio",
        "seg_gt_bbox_coverage_ratio",
        "seg_best_same_class_iou",
        "seg_best_same_class_score",
        "failure_type",
        "primary_yolo_failure",
    ]

    output_fields = (
        original_fields
        + [
            field
            for field in new_fields
            if field
            not in original_fields
        ]
    )

    write_csv(
        output_dir
        / "gt_failure_diagnosis.csv",
        diagnosis_rows,
        output_fields,
    )

    # ========================================================
    # Failure matrix summaries
    # ========================================================

    group_specs = build_group_specs(
        diagnosis_rows
    )

    matrix_summary_rows = []

    yolo_failure_category_rows = []

    for (
        dimension,
        value,
    ) in group_specs:

        group_rows = [
            row
            for row
            in diagnosis_rows
            if row_in_group(
                row,
                dimension,
                value,
            )
        ]

        num_gt = len(
            group_rows
        )

        yolo_fail_rows = [
            row
            for row
            in group_rows
            if not bool(
                row[
                    "yolo_detected"
                ]
            )
        ]

        yolo_failure_count = len(
            yolo_fail_rows
        )

        matrix_counter = Counter(
            row[
                "failure_type"
            ]
            for row
            in group_rows
        )

        for (
            failure_type,
            count,
        ) in sorted(
            matrix_counter.items()
        ):
            matrix_summary_rows.append(
                {
                    "dimension": (
                        dimension
                    ),
                    "value": value,
                    "num_gt": (
                        num_gt
                    ),
                    "yolo_failure_count": (
                        yolo_failure_count
                    ),
                    "failure_type": (
                        failure_type
                    ),
                    "count": count,
                    "share_of_gt": (
                        count
                        / num_gt
                        if num_gt > 0
                        else None
                    ),
                }
            )

        # ----------------------------------------------------
        # YOLO failures:
        # condition × class × YOLO error mode × three-model type
        # ----------------------------------------------------

        category_counter = Counter()

        for row in yolo_fail_rows:
            category_counter[
                (
                    int(
                        row[
                            "category_id"
                        ]
                    ),
                    str(
                        row[
                            "category"
                        ]
                    ),
                    str(
                        row[
                            "yolo_status"
                        ]
                    ),
                    str(
                        row[
                            "failure_type"
                        ]
                    ),
                )
            ] += 1

        for (
            (
                category_id,
                category,
                yolo_status,
                failure_type,
            ),
            count,
        ) in sorted(
            category_counter.items()
        ):
            yolo_failure_category_rows.append(
                {
                    "dimension": (
                        dimension
                    ),
                    "value": value,
                    "num_gt": (
                        num_gt
                    ),
                    "num_yolo_failures": (
                        yolo_failure_count
                    ),
                    "category_id": (
                        category_id
                    ),
                    "category": (
                        category
                    ),
                    "yolo_status": (
                        yolo_status
                    ),
                    "failure_type": (
                        failure_type
                    ),
                    "count": count,
                    "share_of_yolo_failures": (
                        count
                        / yolo_failure_count
                        if yolo_failure_count
                        > 0
                        else None
                    ),
                }
            )

    write_csv(
        output_dir
        / "failure_matrix_summary.csv",
        matrix_summary_rows,
        [
            "dimension",
            "value",
            "num_gt",
            "yolo_failure_count",
            "failure_type",
            "count",
            "share_of_gt",
        ],
    )

    write_csv(
        output_dir
        / "yolo_failure_category_summary.csv",
        yolo_failure_category_rows,
        [
            "dimension",
            "value",
            "num_gt",
            "num_yolo_failures",
            "category_id",
            "category",
            "yolo_status",
            "failure_type",
            "count",
            "share_of_yolo_failures",
        ],
    )

    # ========================================================
    # Metadata
    # ========================================================

    elapsed = (
        time.time()
        - start_time
    )

    metadata = {
        "dataset": "BDD100K",
        "split": "val",

        "detection_matches": str(
            detection_match_path
        ),
        "detection_matches_sha256": (
            sha256_file(
                detection_match_path
            )
        ),

        "seg_predictions": str(
            seg_prediction_path
        ),
        "seg_predictions_sha256": (
            sha256_file(
                seg_prediction_path
            )
        ),

        "seg_metadata": (
            str(
                seg_metadata_path
            )
            if seg_metadata_path
            is not None
            else None
        ),

        "num_gt_objects": (
            len(
                diagnosis_rows
            )
        ),

        "supported_seg_categories": {
            str(
                category_id
            ): name
            for category_id, name
            in SUPPORTED_SEG_CATEGORIES.items()
        },

        "unsupported_seg_categories": {
            str(
                category_id
            ): name
            for category_id, name
            in UNSUPPORTED_SEG_CATEGORIES.items()
        },

        "thresholds": {
            "seg_score_threshold": (
                args.seg_score_threshold
            ),
            "seg_match_iou": (
                args.seg_match_iou
            ),
            "seg_localization_iou": (
                args.seg_localization_iou
            ),
            "min_mask_inside_ratio": (
                args.min_mask_inside_ratio
            ),
        },

        "seg_prediction_diagnostics": (
            seg_prediction_diagnostics
        ),

        "seg_match_diagnostics": (
            seg_match_diagnostics
        ),

        "failure_type_counts": dict(
            failure_type_counter
        ),

        "research_boundary": {
            "has_instance_mask_gt": False,
            "seg_interpretation": (
                "Segmentation Support / Region Recovered"
            ),
            "mask_correctness_claim": False,
            "internal_causal_claim": False,
            "diagnostic_evidence_only": True,
            "mask_inside_gt_ratio_definition": (
                "fraction of predicted polygon area "
                "lying inside the BDD100K detection GT bbox"
            ),
            "rider_seg_status": (
                "N/A"
            ),
            "traffic_sign_seg_status": (
                "N/A"
            ),
        },

        "elapsed_seconds": round(
            elapsed,
            3,
        ),
    }

    if seg_export_metadata is not None:
        metadata[
            "seg_export_metadata"
        ] = (
            seg_export_metadata
        )

    write_json(
        output_dir
        / "metadata.json",
        metadata,
    )

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print("=" * 100)
    print(
        "THREE-MODEL FAILURE DIAGNOSIS COMPLETE"
    )
    print("=" * 100)

    print(
        f"GT objects: {len(diagnosis_rows)}"
    )

    print()

    print(
        "Seg matching:"
    )

    for key, value in (
        seg_match_diagnostics.items()
    ):
        print(
            f"  {key:<40} {value}"
        )

    print()

    print(
        "Overall three-model matrix:"
    )

    for (
        failure_type,
        count,
    ) in sorted(
        failure_type_counter.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    ):
        print(
            f"  {failure_type:<60} {count}"
        )

    # --------------------------------------------------------
    # Key research subsets。
    # --------------------------------------------------------

    key_subsets = [
        (
            "overall",
            "all",
        ),
        (
            "timeofday",
            "night",
        ),
        (
            "scene",
            "highway",
        ),
        (
            "scale",
            "small",
        ),
    ]

    print()

    print("=" * 100)
    print(
        "YOLO11m FAILURE DIAGNOSIS — KEY SUBSETS"
    )
    print("=" * 100)

    for (
        dimension,
        value,
    ) in key_subsets:

        subset_rows = [
            row
            for row
            in diagnosis_rows
            if (
                row_in_group(
                    row,
                    dimension,
                    value,
                )
                and not bool(
                    row[
                        "yolo_detected"
                    ]
                )
            )
        ]

        counter = Counter(
            row[
                "failure_type"
            ]
            for row
            in subset_rows
        )

        print()

        print(
            f"{dimension}:{value} "
            f"YOLO failures={len(subset_rows)}"
        )

        for (
            failure_type,
            count,
        ) in sorted(
            counter.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        ):
            percentage = (
                100.0
                * count
                / len(
                    subset_rows
                )
                if subset_rows
                else 0.0
            )

            print(
                f"  {failure_type:<60} "
                f"{count:7d}  "
                f"{percentage:6.2f}%"
            )

    print()

    print(
        f"Results: {output_dir}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()