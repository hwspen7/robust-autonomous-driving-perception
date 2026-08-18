from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


MODELS = ("YOLO11m", "D-FINE-M")
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

G_ID, G_CAT, G_BBOX, G_BOX, G_AREA, G_CROWD = range(6)
P_ID, P_CAT, P_SCORE, P_BBOX, P_BOX = range(5)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build full-train object-level matches using independently "
            "calibrated detector operating points."
        )
    )
    parser.add_argument("--gt", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--expected-calibration-sha256", required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="MODEL=/path/predictions.json",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        required=True,
        help="MODEL=FLOAT",
    )
    parser.add_argument(
        "--expected-predictions",
        action="append",
        required=True,
        help="MODEL=COUNT",
    )
    parser.add_argument(
        "--prediction-sha256",
        action="append",
        required=True,
        help="MODEL=SHA256",
    )
    parser.add_argument("--expected-images", type=int, default=70000)
    parser.add_argument(
        "--expected-annotations",
        type=int,
        default=1286852,
    )
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument(
        "--localization-iou",
        type=float,
        default=0.10,
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_map(values, caster, name):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{name} requires MODEL=VALUE: {value}")
        model, raw = value.split("=", 1)
        model = model.strip()
        if model in result:
            raise ValueError(f"Duplicate {name} model: {model}")
        result[model] = caster(raw.strip())

    if set(result) != set(MODELS):
        raise ValueError(
            f"{name} must contain exactly {MODELS}; got {sorted(result)}"
        )
    return result


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    os.replace(temporary, path)


def bbox_xyxy(bbox):
    x, y, width, height = bbox
    return (x, y, x + width, y + height)


def scale_name(area):
    if area < 32.0**2:
        return "small"
    if area < 96.0**2:
        return "medium"
    return "large"


def clean_attribute(value):
    if value is None:
        return "undefined"
    value = str(value).strip()
    return value if value else "undefined"


def image_attributes(image):
    attrs = image.get("attributes")
    if not isinstance(attrs, dict):
        attrs = {}

    return {
        "width": int(image["width"]),
        "height": int(image["height"]),
        "timeofday": clean_attribute(
            attrs.get("timeofday", image.get("timeofday"))
        ),
        "weather": clean_attribute(attrs.get("weather", image.get("weather"))),
        "scene": clean_attribute(attrs.get("scene", image.get("scene"))),
    }


def load_gt(path, expected_images, expected_annotations):
    print("=== LOADING TRAIN GT ===", flush=True)

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    images = payload.get("images")
    annotations = payload.get("annotations")
    categories = payload.get("categories")

    if not isinstance(images, list):
        raise TypeError("GT images must be a list.")
    if not isinstance(annotations, list):
        raise TypeError("GT annotations must be a list.")
    if not isinstance(categories, list):
        raise TypeError("GT categories must be a list.")

    if len(images) != expected_images:
        raise ValueError(
            f"Unexpected image count: {len(images)} != {expected_images}"
        )
    if len(annotations) != expected_annotations:
        raise ValueError(
            "Unexpected annotation count: "
            f"{len(annotations)} != {expected_annotations}"
        )

    category_names = {
        int(category["id"]): str(category["name"]) for category in categories
    }
    if set(category_names) != set(CATEGORY_NAMES):
        raise ValueError(f"Unexpected category IDs: {sorted(category_names)}")

    image_meta = {}
    for image in images:
        image_id = int(image["id"])
        if image_id in image_meta:
            raise ValueError(f"Duplicate image_id: {image_id}")
        meta = image_attributes(image)
        if meta["width"] <= 0 or meta["height"] <= 0:
            raise ValueError(f"Invalid image dimensions: {image_id}")
        image_meta[image_id] = meta

    gt_by_image = defaultdict(list)
    annotation_ids = set()
    gt_per_class = Counter()
    invalid_gt = 0

    for annotation in annotations:
        annotation_id = int(annotation["id"])
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])

        if annotation_id in annotation_ids:
            raise ValueError(f"Duplicate annotation_id: {annotation_id}")
        annotation_ids.add(annotation_id)

        if image_id not in image_meta:
            raise ValueError(f"Unknown GT image_id: {image_id}")
        if category_id not in category_names:
            raise ValueError(f"Unknown GT category_id: {category_id}")

        bbox = tuple(float(x) for x in annotation["bbox"])
        if len(bbox) != 4 or not all(math.isfinite(x) for x in bbox):
            raise ValueError(f"Invalid GT bbox: annotation_id={annotation_id}")

        if bbox[2] <= 0.0 or bbox[3] <= 0.0:
            invalid_gt += 1

        area = float(annotation.get("area", max(0.0, bbox[2] * bbox[3])))
        iscrowd = int(annotation.get("iscrowd", 0))

        gt_by_image[image_id].append(
            (
                annotation_id,
                category_id,
                bbox,
                bbox_xyxy(bbox),
                area,
                iscrowd,
            )
        )
        gt_per_class[category_id] += 1

    for records in gt_by_image.values():
        records.sort(key=lambda record: record[G_ID])

    if invalid_gt:
        raise ValueError(f"GT contains {invalid_gt} non-positive boxes.")

    image_ids = sorted(image_meta)

    print(f"images: {len(image_ids)}", flush=True)
    print(f"annotations: {len(annotation_ids)}", flush=True)
    print("gt_per_class:", dict(sorted(gt_per_class.items())), flush=True)

    del payload, images, annotations, categories
    gc.collect()

    return image_ids, image_meta, gt_by_image, category_names


def load_predictions(
    path,
    model,
    image_meta,
    category_names,
    expected_count,
    expected_sha256,
):
    print(f"Verifying SHA256: {model}", flush=True)
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{model} prediction SHA256 mismatch:\n"
            f"expected={expected_sha256}\nactual={actual_sha256}"
        )

    print(f"Loading predictions: {model}", flush=True)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        predictions = payload.get("predictions")
    else:
        predictions = payload

    if not isinstance(predictions, list):
        raise TypeError(f"{model} predictions must be a list.")

    if len(predictions) != expected_count:
        raise ValueError(
            f"{model} prediction count mismatch: "
            f"{len(predictions)} != {expected_count}"
        )

    grouped = defaultdict(list)
    negative_size = 0
    zero_area = 0
    out_of_bounds = 0
    minimum_score = math.inf
    maximum_score = -math.inf

    for prediction_index, prediction in enumerate(predictions):
        image_id = int(prediction["image_id"])
        category_id = int(prediction["category_id"])
        score = float(prediction["score"])
        bbox = tuple(float(x) for x in prediction["bbox"])

        if image_id not in image_meta:
            raise ValueError(f"{model}: unknown image_id={image_id}")
        if category_id not in category_names:
            raise ValueError(f"{model}: unknown category_id={category_id}")
        if len(bbox) != 4 or not all(math.isfinite(x) for x in bbox):
            raise ValueError(
                f"{model}: invalid bbox at prediction {prediction_index}"
            )
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(
                f"{model}: invalid score at prediction {prediction_index}"
            )

        width, height = bbox[2], bbox[3]
        if width < 0.0 or height < 0.0:
            negative_size += 1
        if width == 0.0 or height == 0.0:
            zero_area += 1

        meta = image_meta[image_id]
        if (
            bbox[0] < 0.0
            or bbox[1] < 0.0
            or bbox[0] + width > meta["width"]
            or bbox[1] + height > meta["height"]
        ):
            out_of_bounds += 1

        grouped[image_id].append(
            (
                prediction_index,
                category_id,
                score,
                bbox,
                bbox_xyxy(bbox),
            )
        )

        minimum_score = min(minimum_score, score)
        maximum_score = max(maximum_score, score)

        if (
            (prediction_index + 1) % 5_000_000 == 0
            or prediction_index + 1 == expected_count
        ):
            print(
                f"loaded={prediction_index + 1}/{expected_count}",
                flush=True,
            )

    del payload, predictions
    gc.collect()

    per_image = [len(grouped.get(image_id, ())) for image_id in image_meta]

    audit = {
        "prediction_file": str(path),
        "prediction_sha256": actual_sha256,
        "prediction_count": expected_count,
        "num_images_with_predictions": sum(count > 0 for count in per_image),
        "minimum_predictions_per_image": min(per_image),
        "maximum_predictions_per_image": max(per_image),
        "minimum_score": minimum_score,
        "maximum_score": maximum_score,
        "negative_size_bbox": negative_size,
        "zero_area_bbox": zero_area,
        "out_of_bounds_bbox": out_of_bounds,
    }

    print(json.dumps(audit, indent=2), flush=True)
    return grouped, audit


def pairwise_iou(gt_boxes, pred_boxes):
    gt_count = len(gt_boxes)
    pred_count = len(pred_boxes)

    if gt_count == 0 or pred_count == 0:
        return np.zeros((gt_count, pred_count), dtype=np.float32)

    gt = np.asarray(gt_boxes, dtype=np.float32)
    pred = np.asarray(pred_boxes, dtype=np.float32)

    left = np.maximum(gt[:, None, 0], pred[None, :, 0])
    top = np.maximum(gt[:, None, 1], pred[None, :, 1])
    right = np.minimum(gt[:, None, 2], pred[None, :, 2])
    bottom = np.minimum(gt[:, None, 3], pred[None, :, 3])

    intersection = np.maximum(0.0, right - left) * np.maximum(
        0.0, bottom - top
    )

    gt_area = np.maximum(0.0, gt[:, 2] - gt[:, 0]) * np.maximum(
        0.0, gt[:, 3] - gt[:, 1]
    )
    pred_area = np.maximum(0.0, pred[:, 2] - pred[:, 0]) * np.maximum(
        0.0, pred[:, 3] - pred[:, 1]
    )

    union = gt_area[:, None] + pred_area[None, :] - intersection

    result = np.zeros_like(intersection, dtype=np.float32)
    np.divide(intersection, union, out=result, where=union > 0.0)
    return result


def best_position(iou_matrix, gt_position, pred_positions):
    if not pred_positions:
        return None, 0.0

    values = iou_matrix[gt_position, pred_positions]
    local_index = int(np.argmax(values))
    pred_position = int(pred_positions[local_index])
    return pred_position, float(values[local_index])


def float_cell(value):
    if value is None:
        return ""
    return format(float(value), ".10g")


OBJECT_FIELDS = [
    "model",
    "annotation_id",
    "image_id",
    "category_id",
    "category",
    "gt_x",
    "gt_y",
    "gt_width",
    "gt_height",
    "gt_area",
    "scale",
    "iscrowd",
    "image_width",
    "image_height",
    "timeofday",
    "weather",
    "scene",
    "status",
    "detected",
    "matched_prediction_index",
    "matched_iou",
    "matched_score",
    "best_same_all_prediction_index",
    "best_same_all_iou",
    "best_same_all_score",
    "best_same_op_prediction_index",
    "best_same_op_iou",
    "best_same_op_score",
    "best_wrong_prediction_index",
    "best_wrong_category_id",
    "best_wrong_iou",
    "best_wrong_score",
    "num_predictions_image",
    "num_operating_predictions_image",
    "num_same_class_operating_predictions",
]

FP_FIELDS = [
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
    "best_gt_annotation_id",
    "best_gt_category_id",
    "best_gt_iou",
]

IMAGE_FIELDS = [
    "model",
    "image_id",
    "num_gt",
    "num_predictions",
    "num_operating_predictions",
    "num_detected_gt",
    "num_false_positives",
    "num_missed",
    "num_low_confidence_candidate",
    "num_assignment_conflict_candidate",
    "num_classification_error_candidate",
    "num_localization_error_candidate",
]


def process_model(
    model,
    prediction_path,
    prediction_sha256,
    expected_predictions,
    threshold,
    image_ids,
    image_meta,
    gt_by_image,
    category_names,
    match_iou,
    localization_iou,
    model_dir,
):
    model_start = time.time()

    prediction_by_image, prediction_audit = load_predictions(
        path=prediction_path,
        model=model,
        image_meta=image_meta,
        category_names=category_names,
        expected_count=expected_predictions,
        expected_sha256=prediction_sha256,
    )

    object_path = model_dir / "object_matches.csv"
    fp_path = model_dir / "false_positives.csv"
    image_path = model_dir / "image_summary.csv"

    status_counts = Counter()
    fp_type_counts = Counter()
    per_class_status = defaultdict(Counter)

    object_count = 0
    operating_prediction_count = 0
    matched_prediction_count = 0
    false_positive_count = 0

    with (
        object_path.open("w", encoding="utf-8", newline="") as object_file,
        fp_path.open("w", encoding="utf-8", newline="") as fp_file,
        image_path.open("w", encoding="utf-8", newline="") as image_file,
    ):
        object_writer = csv.DictWriter(object_file, fieldnames=OBJECT_FIELDS)
        fp_writer = csv.DictWriter(fp_file, fieldnames=FP_FIELDS)
        image_writer = csv.DictWriter(image_file, fieldnames=IMAGE_FIELDS)

        object_writer.writeheader()
        fp_writer.writeheader()
        image_writer.writeheader()

        for image_number, image_id in enumerate(image_ids, 1):
            gt_records = gt_by_image.get(image_id, ())
            predictions = prediction_by_image.pop(image_id, ())

            gt_count = len(gt_records)
            prediction_count = len(predictions)

            gt_boxes = [record[G_BOX] for record in gt_records]
            pred_boxes = [record[P_BOX] for record in predictions]
            iou_matrix = pairwise_iou(gt_boxes, pred_boxes)

            gt_categories = np.asarray(
                [record[G_CAT] for record in gt_records],
                dtype=np.int16,
            )
            pred_categories = np.asarray(
                [record[P_CAT] for record in predictions],
                dtype=np.int16,
            )
            pred_scores = np.asarray(
                [record[P_SCORE] for record in predictions],
                dtype=np.float32,
            )

            operating_positions = np.flatnonzero(
                pred_scores >= threshold
            ).tolist()

            operating_prediction_count += len(operating_positions)

            all_by_category = {}
            operating_by_category = {}

            for category_id in category_names:
                all_by_category[category_id] = np.flatnonzero(
                    pred_categories == category_id
                ).tolist()
                operating_by_category[category_id] = [
                    position
                    for position in operating_positions
                    if predictions[position][P_CAT] == category_id
                ]

            matches = {}
            matched_prediction_positions = set()

            for category_id in category_names:
                category_gt_positions = np.flatnonzero(
                    gt_categories == category_id
                ).tolist()

                category_prediction_positions = sorted(
                    operating_by_category[category_id],
                    key=lambda position: (
                        -predictions[position][P_SCORE],
                        predictions[position][P_ID],
                    ),
                )

                unmatched_gt_positions = set(category_gt_positions)

                for prediction_position in category_prediction_positions:
                    if not unmatched_gt_positions:
                        break

                    candidates = sorted(unmatched_gt_positions)
                    values = iou_matrix[candidates, prediction_position]
                    best_local = int(np.argmax(values))
                    gt_position = candidates[best_local]
                    best_iou = float(values[best_local])

                    if best_iou >= match_iou:
                        matches[gt_position] = (
                            prediction_position,
                            best_iou,
                        )
                        matched_prediction_positions.add(prediction_position)
                        unmatched_gt_positions.remove(gt_position)

            matched_prediction_count += len(matches)

            image_status_counts = Counter()

            for gt_position, gt in enumerate(gt_records):
                annotation_id = gt[G_ID]
                category_id = gt[G_CAT]

                same_all_position, same_all_iou = best_position(
                    iou_matrix,
                    gt_position,
                    all_by_category[category_id],
                )
                same_op_position, same_op_iou = best_position(
                    iou_matrix,
                    gt_position,
                    operating_by_category[category_id],
                )

                wrong_positions = [
                    position
                    for position in operating_positions
                    if predictions[position][P_CAT] != category_id
                ]
                wrong_position, wrong_iou = best_position(
                    iou_matrix,
                    gt_position,
                    wrong_positions,
                )

                matched_position = None
                matched_iou = None
                matched_score = None

                if gt_position in matches:
                    matched_position, matched_iou = matches[gt_position]
                    matched_score = predictions[matched_position][P_SCORE]
                    status = "detected"
                elif same_op_position is not None and same_op_iou >= match_iou:
                    status = "assignment_conflict_candidate"
                elif (
                    same_all_position is not None
                    and same_all_iou >= match_iou
                    and predictions[same_all_position][P_SCORE] < threshold
                ):
                    status = "low_confidence_candidate"
                elif wrong_position is not None and wrong_iou >= match_iou:
                    status = "classification_error_candidate"
                elif (
                    same_op_position is not None
                    and localization_iou <= same_op_iou < match_iou
                ):
                    status = "localization_error_candidate"
                else:
                    status = "missed"

                status_counts[status] += 1
                image_status_counts[status] += 1
                per_class_status[category_id][status] += 1
                object_count += 1

                meta = image_meta[image_id]
                bbox = gt[G_BBOX]

                same_all = (
                    predictions[same_all_position]
                    if same_all_position is not None
                    else None
                )
                same_op = (
                    predictions[same_op_position]
                    if same_op_position is not None
                    else None
                )
                wrong = (
                    predictions[wrong_position]
                    if wrong_position is not None
                    else None
                )

                object_writer.writerow(
                    {
                        "model": model,
                        "annotation_id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "category": category_names[category_id],
                        "gt_x": float_cell(bbox[0]),
                        "gt_y": float_cell(bbox[1]),
                        "gt_width": float_cell(bbox[2]),
                        "gt_height": float_cell(bbox[3]),
                        "gt_area": float_cell(gt[G_AREA]),
                        "scale": scale_name(gt[G_AREA]),
                        "iscrowd": gt[G_CROWD],
                        "image_width": meta["width"],
                        "image_height": meta["height"],
                        "timeofday": meta["timeofday"],
                        "weather": meta["weather"],
                        "scene": meta["scene"],
                        "status": status,
                        "detected": int(status == "detected"),
                        "matched_prediction_index": (
                            predictions[matched_position][P_ID]
                            if matched_position is not None
                            else ""
                        ),
                        "matched_iou": float_cell(matched_iou),
                        "matched_score": float_cell(matched_score),
                        "best_same_all_prediction_index": (
                            same_all[P_ID] if same_all else ""
                        ),
                        "best_same_all_iou": float_cell(same_all_iou),
                        "best_same_all_score": (
                            float_cell(same_all[P_SCORE]) if same_all else ""
                        ),
                        "best_same_op_prediction_index": (
                            same_op[P_ID] if same_op else ""
                        ),
                        "best_same_op_iou": float_cell(same_op_iou),
                        "best_same_op_score": (
                            float_cell(same_op[P_SCORE]) if same_op else ""
                        ),
                        "best_wrong_prediction_index": (
                            wrong[P_ID] if wrong else ""
                        ),
                        "best_wrong_category_id": (
                            wrong[P_CAT] if wrong else ""
                        ),
                        "best_wrong_iou": float_cell(wrong_iou),
                        "best_wrong_score": (
                            float_cell(wrong[P_SCORE]) if wrong else ""
                        ),
                        "num_predictions_image": prediction_count,
                        "num_operating_predictions_image": len(
                            operating_positions
                        ),
                        "num_same_class_operating_predictions": len(
                            operating_by_category[category_id]
                        ),
                    }
                )

            image_fp_count = 0

            for prediction_position in operating_positions:
                if prediction_position in matched_prediction_positions:
                    continue

                prediction = predictions[prediction_position]

                if gt_count:
                    values = iou_matrix[:, prediction_position]
                    gt_position = int(np.argmax(values))
                    best_gt_iou = float(values[gt_position])
                    best_gt = gt_records[gt_position]
                    same_category = prediction[P_CAT] == best_gt[G_CAT]

                    if best_gt_iou >= match_iou and same_category:
                        fp_type = "duplicate_candidate"
                    elif best_gt_iou >= match_iou:
                        fp_type = "classification_candidate"
                    elif best_gt_iou >= localization_iou and same_category:
                        fp_type = "localization_candidate"
                    elif best_gt_iou >= localization_iou:
                        fp_type = "overlap_candidate"
                    else:
                        fp_type = "background_candidate"
                else:
                    best_gt = None
                    best_gt_iou = 0.0
                    fp_type = "background_candidate"

                bbox = prediction[P_BBOX]

                fp_writer.writerow(
                    {
                        "model": model,
                        "image_id": image_id,
                        "prediction_index": prediction[P_ID],
                        "category_id": prediction[P_CAT],
                        "category": category_names[prediction[P_CAT]],
                        "score": float_cell(prediction[P_SCORE]),
                        "pred_x": float_cell(bbox[0]),
                        "pred_y": float_cell(bbox[1]),
                        "pred_width": float_cell(bbox[2]),
                        "pred_height": float_cell(bbox[3]),
                        "fp_type": fp_type,
                        "best_gt_annotation_id": (
                            best_gt[G_ID] if best_gt else ""
                        ),
                        "best_gt_category_id": (
                            best_gt[G_CAT] if best_gt else ""
                        ),
                        "best_gt_iou": float_cell(best_gt_iou),
                    }
                )

                fp_type_counts[fp_type] += 1
                false_positive_count += 1
                image_fp_count += 1

            if len(matches) + image_fp_count != len(operating_positions):
                raise RuntimeError(
                    f"Operating prediction accounting failed: {image_id}"
                )

            image_writer.writerow(
                {
                    "model": model,
                    "image_id": image_id,
                    "num_gt": gt_count,
                    "num_predictions": prediction_count,
                    "num_operating_predictions": len(operating_positions),
                    "num_detected_gt": image_status_counts["detected"],
                    "num_false_positives": image_fp_count,
                    "num_missed": image_status_counts["missed"],
                    "num_low_confidence_candidate": image_status_counts[
                        "low_confidence_candidate"
                    ],
                    "num_assignment_conflict_candidate": image_status_counts[
                        "assignment_conflict_candidate"
                    ],
                    "num_classification_error_candidate": image_status_counts[
                        "classification_error_candidate"
                    ],
                    "num_localization_error_candidate": image_status_counts[
                        "localization_error_candidate"
                    ],
                }
            )

            if image_number % 1000 == 0 or image_number == len(image_ids):
                elapsed_minutes = (time.time() - model_start) / 60.0
                print(
                    f"[{image_number}/{len(image_ids)}] "
                    f"{model} elapsed={elapsed_minutes:.2f} min",
                    flush=True,
                )

    del prediction_by_image
    gc.collect()

    if object_count != sum(len(gt_by_image.get(x, ())) for x in image_ids):
        raise RuntimeError(f"{model}: object row count mismatch.")
    if matched_prediction_count + false_positive_count != (
        operating_prediction_count
    ):
        raise RuntimeError(f"{model}: prediction accounting mismatch.")

    summary = {
        "model": model,
        "threshold": threshold,
        "match_iou": match_iou,
        "localization_iou": localization_iou,
        "num_gt_objects": object_count,
        "num_operating_predictions": operating_prediction_count,
        "num_matched_predictions": matched_prediction_count,
        "num_false_positives": false_positive_count,
        "status_counts": dict(sorted(status_counts.items())),
        "false_positive_type_counts": dict(sorted(fp_type_counts.items())),
        "per_class_status": {
            str(category_id): dict(
                sorted(per_class_status[category_id].items())
            )
            for category_id in sorted(category_names)
        },
        "prediction_audit": prediction_audit,
        "elapsed_minutes": (time.time() - model_start) / 60.0,
    }

    write_json(model_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    args = parse_args()

    if not 0.0 <= args.localization_iou < args.match_iou <= 1.0:
        raise ValueError("Require 0 <= localization_iou < match_iou <= 1.")

    prediction_paths = parse_map(
        args.prediction,
        lambda value: Path(value).expanduser().resolve(),
        "--prediction",
    )
    thresholds = parse_map(args.threshold, float, "--threshold")
    expected_predictions = parse_map(
        args.expected_predictions,
        int,
        "--expected-predictions",
    )
    prediction_hashes = parse_map(
        args.prediction_sha256,
        str,
        "--prediction-sha256",
    )

    for model in MODELS:
        if not 0.0 <= thresholds[model] <= 1.0:
            raise ValueError(f"Invalid threshold for {model}.")
        if expected_predictions[model] <= 0:
            raise ValueError(f"Invalid prediction count for {model}.")
        if len(prediction_hashes[model]) != 64:
            raise ValueError(f"Invalid prediction SHA256 for {model}.")
        if not prediction_paths[model].is_file():
            raise FileNotFoundError(prediction_paths[model])

    gt_path = Path(args.gt).expanduser().resolve()
    calibration_path = Path(args.calibration).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite output directory: {output_dir}"
        )

    calibration_sha256 = sha256_file(calibration_path)
    if calibration_sha256 != args.expected_calibration_sha256:
        raise ValueError(
            "Calibration SHA256 mismatch:\n"
            f"expected={args.expected_calibration_sha256}\n"
            f"actual={calibration_sha256}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    staging_dir = output_dir.with_name(
        output_dir.name + f".incomplete-{int(time.time())}-{os.getpid()}"
    )
    if staging_dir.exists():
        raise FileExistsError(staging_dir)
    staging_dir.mkdir(parents=False)

    start_time = time.time()

    try:
        gt_sha256 = sha256_file(gt_path)

        image_ids, image_meta, gt_by_image, category_names = load_gt(
            gt_path,
            args.expected_images,
            args.expected_annotations,
        )

        summaries = {}

        for model in MODELS:
            print("\n" + "=" * 80, flush=True)
            print(f"PROCESSING {model}", flush=True)
            print("=" * 80, flush=True)

            model_dir = staging_dir / model
            model_dir.mkdir()

            summaries[model] = process_model(
                model=model,
                prediction_path=prediction_paths[model],
                prediction_sha256=prediction_hashes[model],
                expected_predictions=expected_predictions[model],
                threshold=thresholds[model],
                image_ids=image_ids,
                image_meta=image_meta,
                gt_by_image=gt_by_image,
                category_names=category_names,
                match_iou=args.match_iou,
                localization_iou=args.localization_iou,
                model_dir=model_dir,
            )

        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "BDD100K",
            "split": "official_training",
            "gt_file": str(gt_path),
            "gt_sha256": gt_sha256,
            "num_images": len(image_ids),
            "num_gt_objects": sum(
                len(gt_by_image.get(x, ())) for x in image_ids
            ),
            "calibration_file": str(calibration_path),
            "calibration_sha256": calibration_sha256,
            "thresholds": thresholds,
            "match_iou": args.match_iou,
            "localization_iou": args.localization_iou,
            "models": summaries,
            "matching_method": (
                "Per-image, per-class, score-descending greedy "
                "one-to-one matching using raw canonical xywh boxes."
            ),
            "diagnostic_statuses_are_candidates": True,
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "elapsed_minutes": (time.time() - start_time) / 60.0,
        }
        write_json(staging_dir / "metadata.json", metadata)

        manifest_entries = []
        for artifact in sorted(staging_dir.rglob("*")):
            if artifact.is_file():
                manifest_entries.append(
                    {
                        "path": str(artifact.relative_to(staging_dir)),
                        "bytes": artifact.stat().st_size,
                        "sha256": sha256_file(artifact),
                    }
                )

        write_json(
            staging_dir / "artifact_manifest.json",
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "artifacts": manifest_entries,
            },
        )

        os.replace(staging_dir, output_dir)

    except Exception as exc:
        try:
            (staging_dir / "FAILED.txt").write_text(
                f"{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        raise

    print("\n" + "=" * 80)
    print("TRAIN DETECTION MATCHING COMPLETE")
    print(f"Output: {output_dir}")
    print(f"Metadata: {output_dir / 'metadata.json'}")
    print(f"Manifest: {output_dir / 'artifact_manifest.json'}")
    print(f"Elapsed minutes: {(time.time() - start_time) / 60.0:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
