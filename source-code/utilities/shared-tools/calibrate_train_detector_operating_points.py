from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import time
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate independent detector operating points on "
            "the frozen BDD100K train-dev split."
        )
    )
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="MODEL=/absolute/path/to/predictions.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--min-class-gt", type=int, default=100)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            data,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_prediction_specs(
    specifications: list[str],
) -> list[tuple[str, Path]]:
    parsed = []
    names = set()

    for specification in specifications:
        if "=" not in specification:
            raise ValueError(
                "--prediction must use MODEL=/path/predictions.json"
            )

        model, path_string = specification.split("=", 1)
        model = model.strip()
        path = Path(path_string).expanduser().resolve()

        if not model:
            raise ValueError("Prediction model name cannot be empty")
        if model in names:
            raise ValueError(f"Duplicate model name: {model}")
        if not path.is_file():
            raise FileNotFoundError(path)

        names.add(model)
        parsed.append((model, path))

    return parsed


def bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second

    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    intersection_width = max(
        0.0, min(ax2, bx2) - max(ax, bx)
    )
    intersection_height = max(
        0.0, min(ay2, by2) - max(ay, by)
    )
    intersection = intersection_width * intersection_height

    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def summarize_counts(
    tp: dict[int, int],
    fp: dict[int, int],
    gt_counts: dict[int, int],
    eligible: list[int],
    categories: dict[int, str],
) -> dict[str, Any]:
    per_class = {}
    macro_precision = []
    macro_recall = []
    macro_f1 = []

    total_tp = 0
    total_fp = 0
    total_gt = sum(gt_counts.values())

    for category_id in sorted(categories):
        true_positive = tp.get(category_id, 0)
        false_positive = fp.get(category_id, 0)
        gt_count = gt_counts.get(category_id, 0)
        false_negative = gt_count - true_positive

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )
        recall = (
            true_positive / gt_count
            if gt_count > 0
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        per_class[str(category_id)] = {
            "class_name": categories[category_id],
            "eligible_for_macro_f1": category_id in eligible,
            "gt": gt_count,
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        if category_id in eligible:
            macro_precision.append(precision)
            macro_recall.append(recall)
            macro_f1.append(f1)

        total_tp += true_positive
        total_fp += false_positive

    micro_precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )
    micro_recall = (
        total_tp / total_gt if total_gt > 0 else 0.0
    )
    micro_f1 = (
        2.0 * micro_precision * micro_recall
        / (micro_precision + micro_recall)
        if micro_precision + micro_recall > 0
        else 0.0
    )

    return {
        "macro_precision": sum(macro_precision) / len(eligible),
        "macro_recall": sum(macro_recall) / len(eligible),
        "macro_f1": sum(macro_f1) / len(eligible),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_gt - total_tp,
        "per_class": per_class,
    }


def exact_threshold_sweep(
    class_events: dict[int, list[tuple[float, int]]],
    gt_counts: dict[int, int],
    eligible: list[int],
    categories: dict[int, str],
) -> tuple[float, dict[str, Any]]:
    combined = []

    for category_id, events in class_events.items():
        combined.extend(
            (score, category_id, is_tp)
            for score, is_tp in events
        )

    combined.sort(key=lambda row: row[0], reverse=True)

    tp = Counter()
    fp = Counter()

    best_threshold = 1.0
    best_summary = summarize_counts(
        tp, fp, gt_counts, eligible, categories
    )

    index = 0

    while index < len(combined):
        score = combined[index][0]
        end = index

        while (
            end < len(combined)
            and combined[end][0] == score
        ):
            _, category_id, is_tp = combined[end]

            if is_tp:
                tp[category_id] += 1
            else:
                fp[category_id] += 1

            end += 1

        summary = summarize_counts(
            tp, fp, gt_counts, eligible, categories
        )

        better_f1 = (
            summary["macro_f1"]
            > best_summary["macro_f1"] + 1e-15
        )
        equal_f1_better_recall = (
            abs(
                summary["macro_f1"]
                - best_summary["macro_f1"]
            ) <= 1e-15
            and summary["macro_recall"]
            > best_summary["macro_recall"] + 1e-15
        )

        if better_f1 or equal_f1_better_recall:
            best_threshold = score
            best_summary = summary

        index = end

    return best_threshold, best_summary


def metrics_at_threshold(
    threshold: float,
    class_arrays: dict[int, dict[str, Any]],
    gt_counts: dict[int, int],
    eligible: list[int],
    categories: dict[int, str],
) -> dict[str, Any]:
    tp = {}
    fp = {}

    for category_id in categories:
        arrays = class_arrays[category_id]
        count = bisect_right(
            arrays["negative_scores"],
            -threshold,
        )

        true_positive = (
            arrays["cumulative_tp"][count - 1]
            if count > 0
            else 0
        )

        tp[category_id] = true_positive
        fp[category_id] = count - true_positive

    return summarize_counts(
        tp, fp, gt_counts, eligible, categories
    )


def calibrate_model(
    model: str,
    prediction_path: Path,
    dev_ids: set[int],
    gt_groups: dict[
        tuple[int, int],
        list[tuple[float, float, float, float]],
    ],
    gt_counts: dict[int, int],
    categories: dict[int, str],
    match_iou: float,
    min_class_gt: int,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.time()

    print("\n" + "=" * 80)
    print("CALIBRATING:", model)
    print("prediction:", prediction_path)

    prediction_sha256 = sha256_file(prediction_path)

    with prediction_path.open("r", encoding="utf-8") as handle:
        raw_predictions = json.load(handle)

    if not isinstance(raw_predictions, list):
        raise TypeError("Prediction JSON must be a list")

    total_predictions = len(raw_predictions)
    prediction_groups = defaultdict(list)
    dev_prediction_count = 0

    for row in raw_predictions:
        image_id = int(row["image_id"])

        if image_id not in dev_ids:
            continue

        category_id = int(row["category_id"])
        score = float(row["score"])
        bbox = tuple(float(value) for value in row["bbox"])

        if category_id not in categories:
            raise ValueError(
                f"Unexpected category_id: {category_id}"
            )
        if len(bbox) != 4:
            raise ValueError("Prediction bbox must contain four values")
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"Invalid score: {score}")
        if not all(math.isfinite(value) for value in bbox):
            raise ValueError("Non-finite prediction bbox")

        prediction_groups[
            (image_id, category_id)
        ].append((score, bbox))
        dev_prediction_count += 1

    del raw_predictions
    gc.collect()

    class_events = {
        category_id: []
        for category_id in categories
    }

    keys = set(gt_groups)
    keys.update(prediction_groups)

    for processed, key in enumerate(sorted(keys), 1):
        category_id = key[1]
        ground_truth = gt_groups.get(key, [])
        predictions = sorted(
            prediction_groups.get(key, []),
            key=lambda row: row[0],
            reverse=True,
        )

        matched_gt = [False] * len(ground_truth)

        for score, prediction_bbox in predictions:
            best_index = -1
            best_iou = 0.0

            for gt_index, gt_bbox in enumerate(ground_truth):
                if matched_gt[gt_index]:
                    continue

                overlap = bbox_iou(
                    prediction_bbox,
                    gt_bbox,
                )

                if overlap > best_iou:
                    best_iou = overlap
                    best_index = gt_index

            is_true_positive = int(
                best_index >= 0 and best_iou >= match_iou
            )

            if is_true_positive:
                matched_gt[best_index] = True

            class_events[category_id].append(
                (score, is_true_positive)
            )

        if processed % 10_000 == 0:
            print(
                f"matched_groups={processed}/{len(keys)}"
            )

    del prediction_groups
    gc.collect()

    eligible = [
        category_id
        for category_id in sorted(categories)
        if gt_counts.get(category_id, 0) >= min_class_gt
    ]

    if not eligible:
        raise RuntimeError("No class satisfies --min-class-gt")

    threshold, best_summary = exact_threshold_sweep(
        class_events,
        gt_counts,
        eligible,
        categories,
    )

    class_arrays = {}

    for category_id in sorted(categories):
        events = sorted(
            class_events[category_id],
            key=lambda row: row[0],
            reverse=True,
        )
        negative_scores = []
        cumulative_tp = []
        running_tp = 0

        for score, is_tp in events:
            negative_scores.append(-score)
            running_tp += is_tp
            cumulative_tp.append(running_tp)

        class_arrays[category_id] = {
            "negative_scores": negative_scores,
            "cumulative_tp": cumulative_tp,
        }

    sensitivity_thresholds = {
        round(step / 100.0, 2)
        for step in range(0, 101)
    }
    sensitivity_thresholds.add(threshold)

    safe_name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        model,
    )

    sensitivity_path = (
        output_dir
        / f"{safe_name}_threshold_sensitivity.csv"
    )

    with sensitivity_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "threshold",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "micro_precision",
                "micro_recall",
                "micro_f1",
                "total_tp",
                "total_fp",
                "total_fn",
            ],
        )
        writer.writeheader()

        for candidate in sorted(
            sensitivity_thresholds,
            reverse=True,
        ):
            summary = metrics_at_threshold(
                candidate,
                class_arrays,
                gt_counts,
                eligible,
                categories,
            )
            writer.writerow({
                "threshold": candidate,
                **{
                    key: summary[key]
                    for key in writer.fieldnames
                    if key != "threshold"
                },
            })

    result = {
        "model": model,
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_sha256,
        "total_predictions": total_predictions,
        "dev_predictions": dev_prediction_count,
        "match_iou": match_iou,
        "selection_objective": (
            "exact-score maximum macro-F1; "
            "tie-break by higher macro recall"
        ),
        "min_class_gt_for_macro": min_class_gt,
        "eligible_category_ids": eligible,
        "excluded_category_ids": [
            category_id
            for category_id in sorted(categories)
            if category_id not in eligible
        ],
        "operating_threshold": threshold,
        "metrics": best_summary,
        "threshold_sensitivity_csv": str(
            sensitivity_path
        ),
        "elapsed_minutes": (
            time.time() - started
        ) / 60.0,
    }

    result_path = (
        output_dir
        / f"{safe_name}_calibration.json"
    )
    atomic_json(result_path, result)

    print("total_predictions:", total_predictions)
    print("dev_predictions:", dev_prediction_count)
    print("eligible_categories:", eligible)
    print(
        "excluded_categories:",
        result["excluded_category_ids"],
    )
    print("operating_threshold:", threshold)
    print("macro_precision:", best_summary["macro_precision"])
    print("macro_recall:", best_summary["macro_recall"])
    print("macro_f1:", best_summary["macro_f1"])
    print("micro_f1:", best_summary["micro_f1"])
    print("result:", result_path)
    print(
        "elapsed_minutes:",
        round(result["elapsed_minutes"], 2),
    )

    del class_events
    del class_arrays
    gc.collect()

    return result


def main() -> None:
    args = parse_args()

    if not 0.0 < args.match_iou <= 1.0:
        raise ValueError("--match-iou must be in (0,1]")
    if args.min_class_gt <= 0:
        raise ValueError("--min-class-gt must be positive")

    gt_path = args.gt.expanduser().resolve()
    split_path = args.split.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite output directory: {output_dir}"
        )

    prediction_specs = parse_prediction_specs(
        args.prediction
    )

    with split_path.open("r", encoding="utf-8") as handle:
        split = json.load(handle)

    dev_ids = {
        int(image_id)
        for image_id in split["image_ids"]
    }

    if len(dev_ids) != 5_000:
        raise ValueError(
            f"Expected 5000 dev images, found {len(dev_ids)}"
        )

    print("=== LOADING TRAIN-DEV GT ===")

    with gt_path.open("r", encoding="utf-8") as handle:
        gt = json.load(handle)

    categories = {
        int(row["id"]): str(row["name"])
        for row in gt["categories"]
    }
    image_ids = {
        int(row["id"]) for row in gt["images"]
    }

    if not dev_ids <= image_ids:
        raise ValueError("Split contains unknown image IDs")

    gt_groups = defaultdict(list)
    gt_counts = Counter()

    for annotation in gt["annotations"]:
        image_id = int(annotation["image_id"])

        if image_id not in dev_ids:
            continue

        category_id = int(annotation["category_id"])
        bbox = tuple(
            float(value)
            for value in annotation["bbox"]
        )

        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError(
                f"Invalid GT bbox: {annotation.get('id')}"
            )

        gt_groups[(image_id, category_id)].append(bbox)
        gt_counts[category_id] += 1

    del gt
    gc.collect()

    output_dir.mkdir(parents=True, exist_ok=False)

    print("dev_images:", len(dev_ids))
    print("dev_gt:", sum(gt_counts.values()))
    print("gt_per_class:", dict(sorted(gt_counts.items())))

    results = {}

    for model, prediction_path in prediction_specs:
        result = calibrate_model(
            model=model,
            prediction_path=prediction_path,
            dev_ids=dev_ids,
            gt_groups=gt_groups,
            gt_counts=gt_counts,
            categories=categories,
            match_iou=args.match_iou,
            min_class_gt=args.min_class_gt,
            output_dir=output_dir,
        )
        results[model] = {
            "operating_threshold":
                result["operating_threshold"],
            "macro_f1":
                result["metrics"]["macro_f1"],
            "macro_precision":
                result["metrics"]["macro_precision"],
            "macro_recall":
                result["metrics"]["macro_recall"],
            "eligible_category_ids":
                result["eligible_category_ids"],
            "prediction_sha256":
                result["prediction_sha256"],
        }

    combined = {
        "protocol": (
            "independent detector operating-point calibration "
            "on frozen BDD100K train-dev"
        ),
        "gt": str(gt_path),
        "gt_sha256": sha256_file(gt_path),
        "split": str(split_path),
        "split_sha256": sha256_file(split_path),
        "num_dev_images": len(dev_ids),
        "num_dev_gt": sum(gt_counts.values()),
        "match_iou": args.match_iou,
        "min_class_gt_for_macro": args.min_class_gt,
        "models": results,
    }

    combined_path = output_dir / "operating_points.json"
    atomic_json(combined_path, combined)

    print("\n" + "=" * 80)
    print("CALIBRATION COMPLETE")
    for model, result in results.items():
        print(
            model,
            "threshold=",
            result["operating_threshold"],
            "macro_f1=",
            result["macro_f1"],
        )
    print("combined:", combined_path)


if __name__ == "__main__":
    main()
