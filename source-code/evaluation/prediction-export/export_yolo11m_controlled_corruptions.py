""
























from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import torch
import ultralytics
from PIL import Image
from ultralytics import YOLO

from controlled_corruptions import (
    CORRUPTION_CONFIG,
    CORRUPTION_NAMES,
    GLOBAL_SEED,
    SEVERITIES,
    apply_corruption,
    deterministic_seed,
    variant_name,
)


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

EXPECTED_VAL_IMAGES = 10_000
EXPECTED_VAL_ANNOTATIONS = 185_523


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export YOLO11m predictions on deterministic on-the-fly "
            "BDD100K controlled corruptions."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="YOLO11m best.pt checkpoint.",
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="BDD100K dataset root.",
    )

    parser.add_argument(
        "--annotation",
        type=Path,
        required=True,
        help="Original BDD100K COCO val annotation JSON.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--corruption",
        type=str,
        required=True,
        choices=list(CORRUPTION_NAMES),
        help="Controlled corruption type.",
    )

    parser.add_argument(
        "--severity",
        type=int,
        required=True,
        choices=list(SEVERITIES),
        help="Controlled corruption severity: 1, 2, or 3.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Inference device.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help=(
            "Number of source images loaded, corrupted, and inferred per batch. "
            "This bounds RAM usage because corrupted images are never stored as a dataset."
        ),
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Low confidence floor used for COCO-style evaluation export.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="YOLO native NMS IoU threshold.",
    )

    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Maximum YOLO detections retained per image before COCOeval.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = all images; positive value = smoke test only.",
    )

    parser.add_argument(
        "--save-vis",
        type=int,
        default=20,
        help=(
            "Number of prediction visualizations to save. "
            "These are the only corrupted-image-derived files saved by this exporter."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing predictions.json.",
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


def normalize_class_name(name: str) -> str:
    ""


    name = str(name).strip().lower()
    name = re.sub(r"[_-]+", " ", name)
    name = re.sub(r"\s+", " ", name)

    return name


def load_and_validate_gt(
    annotation_path: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    ""





    with annotation_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError(
            "BDD100K annotation must be a COCO dictionary."
        )

    for key in (
        "images",
        "annotations",
        "categories",
    ):
        if key not in data:
            raise KeyError(
                f"Annotation missing key: {key}"
            )

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    if len(images) != EXPECTED_VAL_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_VAL_IMAGES} val images, "
            f"found {len(images)}."
        )

    if len(annotations) != EXPECTED_VAL_ANNOTATIONS:
        raise ValueError(
            f"Expected {EXPECTED_VAL_ANNOTATIONS} annotations, "
            f"found {len(annotations)}."
        )

    actual_categories = {
        int(category["id"]): normalize_class_name(
            category["name"]
        )
        for category in categories
    }

    expected_categories = {
        category_id: normalize_class_name(name)
        for category_id, name in BDD100K_CATEGORIES.items()
    }

    if actual_categories != expected_categories:
        raise ValueError(
            "BDD100K category mapping mismatch.\n"
            f"Expected: {expected_categories}\n"
            f"Found   : {actual_categories}\n"
            "Do not use contiguous 0~9 GT for final evaluation."
        )

    image_ids = [
        int(image["id"])
        for image in images
    ]

    if len(image_ids) != len(set(image_ids)):
        raise ValueError(
            "Duplicate image IDs found in GT."
        )

    return images, data


def build_model_class_mapping(
    model_names: dict[int, str] | list[str],
) -> dict[int, int]:
    ""




    if isinstance(model_names, dict):
        normalized_model_names = {
            int(class_id): normalize_class_name(name)
            for class_id, name in model_names.items()
        }
    else:
        normalized_model_names = {
            class_id: normalize_class_name(name)
            for class_id, name in enumerate(model_names)
        }

    expected_name_to_bdd_id = {
        normalize_class_name(name): category_id
        for category_id, name in BDD100K_CATEGORIES.items()
    }

    if len(normalized_model_names) != 10:
        raise ValueError(
            "YOLO checkpoint must contain exactly 10 BDD100K classes.\n"
            f"Found: {normalized_model_names}"
        )

    actual_name_set = set(
        normalized_model_names.values()
    )

    expected_name_set = set(
        expected_name_to_bdd_id.keys()
    )

    if actual_name_set != expected_name_set:
        raise ValueError(
            "YOLO checkpoint class names do not match BDD100K.\n"
            f"Expected: {sorted(expected_name_set)}\n"
            f"Found   : {sorted(actual_name_set)}"
        )

    mapping: dict[int, int] = {}

    for yolo_class_id, class_name in normalized_model_names.items():
        mapping[yolo_class_id] = (
            expected_name_to_bdd_id[class_name]
        )

    return mapping


def load_and_corrupt_image(
    *,
    dataset_root: Path,
    image_info: dict[str, Any],
    corruption: str,
    severity: int,
) -> tuple[Image.Image, int]:
    ""





    file_name = str(
        image_info["file_name"]
    )

    image_path = (
        dataset_root
        / file_name
    ).resolve()

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image does not exist: {image_path}"
        )

    expected_size = (
        int(image_info["width"]),
        int(image_info["height"]),
    )

    with Image.open(image_path) as source:
        source_rgb = source.convert("RGB")

        if source_rgb.size != expected_size:
            raise RuntimeError(
                "BDD100K source image size mismatch.\n"
                f"image_id={int(image_info['id'])}\n"
                f"file_name={file_name}\n"
                f"GT      ={expected_size}\n"
                f"Source  ={source_rgb.size}"
            )

        corrupted = apply_corruption(
            source_rgb,
            image_key=file_name,
            corruption=corruption,
            severity=severity,
        )

    if corrupted.mode != "RGB":
        corrupted = corrupted.convert("RGB")

    if corrupted.size != expected_size:
        raise RuntimeError(
            "Controlled corruption changed image geometry.\n"
            f"image_id={int(image_info['id'])}\n"
            f"file_name={file_name}\n"
            f"Expected ={expected_size}\n"
            f"Found    ={corrupted.size}"
        )

    seed = deterministic_seed(
        image_key=file_name,
        corruption=corruption,
        severity=severity,
    )

    return corrupted, seed


def iter_batches(
    items: list[dict[str, Any]],
    batch_size: int,
):
    ""





    for start in range(0, len(items), batch_size):
        yield items[
            start : start + batch_size
        ]


def main() -> None:
    args = parse_args()

    model_path = args.model.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    annotation_path = args.annotation.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    corruption_module_path = (
        Path(__file__).resolve().parents[3]
        / "source-code"
        / "utilities"
        / "shared-components"
        / "controlled_corruptions.py"
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint does not exist: {model_path}"
        )

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root does not exist: {dataset_root}"
        )

    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"Annotation does not exist: {annotation_path}"
        )

    if not corruption_module_path.is_file():
        raise FileNotFoundError(
            "controlled_corruptions.py must be in the same scripts directory.\n"
            f"Expected: {corruption_module_path}"
        )

    if not 0.0 <= args.conf <= 1.0:
        raise ValueError(
            "--conf must be in [0, 1]."
        )

    if not 0.0 <= args.iou <= 1.0:
        raise ValueError(
            "--iou must be in [0, 1]."
        )

    if args.batch <= 0:
        raise ValueError(
            "--batch must be positive."
        )

    if args.max_det <= 0:
        raise ValueError(
            "--max-det must be positive."
        )

    if args.limit < 0:
        raise ValueError(
            "--limit must be >= 0."
        )

    if args.save_vis < 0:
        raise ValueError(
            "--save-vis must be >= 0."
        )

    variant = variant_name(
        args.corruption,
        args.severity,
    )

    corruption_parameters = (
        CORRUPTION_CONFIG[
            args.corruption
        ][
            args.severity
        ]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    visualization_dir = (
        output_dir
        / "visualizations"
    )

    visualization_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_path = (
        output_dir
        / "predictions.json"
    )

    temp_prediction_path = (
        output_dir
        / "predictions.json.tmp"
    )

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    image_summary_path = (
        output_dir
        / "image_summary.json"
    )

    if prediction_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{prediction_path} already exists.\n"
            "Use --overwrite only if you intentionally want to replace it."
        )

    if args.overwrite:
        temp_prediction_path.unlink(
            missing_ok=True
        )

    # --------------------------------------------------------

    # --------------------------------------------------------

    images, gt_data = load_and_validate_gt(
        annotation_path
    )


    images = sorted(
        images,
        key=lambda image: int(image["id"]),
    )

    if args.limit > 0:
        images = images[
            : args.limit
        ]

    if not images:
        raise RuntimeError(
            "No validation images selected."
        )

    # --------------------------------------------------------

    # --------------------------------------------------------

    seen_basenames: set[str] = set()

    for image_info in images:
        file_name = str(
            image_info["file_name"]
        )

        image_path = (
            dataset_root
            / file_name
        ).resolve()

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image does not exist: {image_path}"
            )

        basename = image_path.name

        if basename in seen_basenames:
            raise ValueError(
                f"Duplicate image basename detected: {basename}"
            )

        seen_basenames.add(
            basename
        )

    # --------------------------------------------------------

    # --------------------------------------------------------

    model = YOLO(
        str(model_path)
    )

    class_mapping = (
        build_model_class_mapping(
            model.names
        )
    )

    print("=" * 90)
    print("YOLO11m CONTROLLED-CORRUPTION PREDICTION EXPORT")
    print("=" * 90)
    print(f"Model       : {model_path}")
    print(f"Checkpoint  : {sha256_file(model_path)}")
    print(f"Images      : {len(images)}")
    print(f"Variant     : {variant}")
    print(f"Corruption  : {args.corruption}")
    print(f"Severity    : {args.severity}")
    print(f"Parameters  : {corruption_parameters}")
    print(f"Global seed : {GLOBAL_SEED}")
    print(f"imgsz       : {args.imgsz}")
    print(f"batch       : {args.batch}")
    print(f"conf        : {args.conf}")
    print(f"NMS IoU     : {args.iou}")
    print(f"max_det     : {args.max_det}")
    print(f"device      : {args.device}")
    print(f"Output      : {output_dir}")
    print()
    print("YOLO -> BDD category mapping:")

    for yolo_id in sorted(
        class_mapping
    ):
        bdd_id = class_mapping[
            yolo_id
        ]

        print(
            f"  YOLO {yolo_id:2d} "
            f"{model.names[yolo_id]:<15} "
            f"-> BDD {bdd_id:2d} "
            f"{BDD100K_CATEGORIES[bdd_id]}"
        )

    print("=" * 90)

    # --------------------------------------------------------

    #

    # clean exporter：file path -> YOLO preprocessing

    #

    # conf=0.001, iou=0.7, max_det=300, rect=True,
    # augment=False, agnostic_nms=False。
    # --------------------------------------------------------

    start_time = time.time()

    total_predictions = 0
    processed_images = 0
    visualized_images = 0
    zero_area_bbox_count = 0

    prediction_counts_by_class = Counter()

    image_summaries: list[
        dict[str, Any]
    ] = []

    seen_image_ids: set[int] = set()

    first_prediction = True

    try:
        with temp_prediction_path.open(
            "w",
            encoding="utf-8",
        ) as prediction_file:
            prediction_file.write("[")

            for batch_infos in iter_batches(
                images,
                args.batch,
            ):
                corrupted_images: list[
                    Image.Image
                ] = []

                corruption_seeds: list[int] = []

                # --------------------------------------------


                # --------------------------------------------

                for image_info in batch_infos:
                    corrupted_image, seed = (
                        load_and_corrupt_image(
                            dataset_root=dataset_root,
                            image_info=image_info,
                            corruption=args.corruption,
                            severity=args.severity,
                        )
                    )

                    corrupted_images.append(
                        corrupted_image
                    )

                    corruption_seeds.append(
                        seed
                    )

                # --------------------------------------------

                #

                # YOLO resize / letterbox / tensor conversion / NMS

                # --------------------------------------------

                batch_results = model.predict(
                    source=corrupted_images,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    device=args.device,
                    stream=False,
                    rect=True,
                    augment=False,
                    agnostic_nms=False,
                    save=False,
                    verbose=False,
                )

                if len(batch_results) != len(batch_infos):
                    raise RuntimeError(
                        "YOLO batch output count mismatch.\n"
                        f"Input batch : {len(batch_infos)}\n"
                        f"Result batch: {len(batch_results)}"
                    )

                for result, image_info, corruption_seed in zip(
                    batch_results,
                    batch_infos,
                    corruption_seeds,
                    strict=True,
                ):
                    processed_images += 1

                    image_id = int(
                        image_info["id"]
                    )

                    file_name = str(
                        image_info["file_name"]
                    )

                    result_filename = (
                        Path(file_name).name
                    )

                    image_width = int(
                        image_info["width"]
                    )

                    image_height = int(
                        image_info["height"]
                    )

                    if image_id in seen_image_ids:
                        raise RuntimeError(
                            f"Duplicate inference result for image_id={image_id}"
                        )

                    seen_image_ids.add(
                        image_id
                    )

                    # ----------------------------------------


                    # ----------------------------------------

                    result_height, result_width = map(
                        int,
                        result.orig_shape,
                    )

                    if (
                        result_width != image_width
                        or result_height != image_height
                    ):
                        raise RuntimeError(
                            "Image size mismatch after controlled corruption.\n"
                            f"image_id={image_id}\n"
                            f"GT      ={image_width}x{image_height}\n"
                            f"Result  ={result_width}x{result_height}"
                        )

                    image_prediction_count = 0

                    boxes = result.boxes

                    if (
                        boxes is not None
                        and len(boxes) > 0
                    ):
                        xyxy = (
                            boxes.xyxy
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        scores = (
                            boxes.conf
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        classes = (
                            boxes.cls
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(int)
                        )

                        for detection_index in range(
                            len(boxes)
                        ):
                            yolo_class_id = int(
                                classes[
                                    detection_index
                                ]
                            )

                            if yolo_class_id not in class_mapping:
                                raise RuntimeError(
                                    "Unexpected YOLO class ID: "
                                    f"{yolo_class_id}"
                                )

                            category_id = (
                                class_mapping[
                                    yolo_class_id
                                ]
                            )

                            score = float(
                                scores[
                                    detection_index
                                ]
                            )

                            if not math.isfinite(
                                score
                            ):
                                raise RuntimeError(
                                    "Non-finite YOLO score detected."
                                )

                            if not 0.0 <= score <= 1.0:
                                raise RuntimeError(
                                    f"Invalid YOLO score: {score}"
                                )

                            x1, y1, x2, y2 = map(
                                float,
                                xyxy[
                                    detection_index
                                ],
                            )

                            width = x2 - x1
                            height = y2 - y1

                            if not all(
                                math.isfinite(value)
                                for value in (
                                    x1,
                                    y1,
                                    width,
                                    height,
                                )
                            ):
                                raise RuntimeError(
                                    "Non-finite YOLO bbox detected."
                                )



                            if width < 0 or height < 0:
                                raise RuntimeError(
                                    "YOLO produced negative-size bbox.\n"
                                    f"image_id={image_id}\n"
                                    f"bbox={[x1, y1, width, height]}"
                                )

                            if width == 0 or height == 0:
                                zero_area_bbox_count += 1

                            prediction = {
                                "image_id": image_id,
                                "category_id": category_id,
                                "bbox": [
                                    x1,
                                    y1,
                                    width,
                                    height,
                                ],
                                "score": score,
                            }

                            if not first_prediction:
                                prediction_file.write(
                                    ","
                                )

                            json.dump(
                                prediction,
                                prediction_file,
                                separators=(
                                    ",",
                                    ":",
                                ),
                                allow_nan=False,
                            )

                            first_prediction = False

                            total_predictions += 1
                            image_prediction_count += 1

                            prediction_counts_by_class[
                                category_id
                            ] += 1

                    image_summaries.append(
                        {
                            "image_id": image_id,
                            "file_name": file_name,
                            "variant": variant,
                            "corruption_seed": int(
                                corruption_seed
                            ),
                            "prediction_count": (
                                image_prediction_count
                            ),
                        }
                    )

                    # ----------------------------------------



                    # ----------------------------------------

                    if visualized_images < args.save_vis:
                        plotted = result.plot()

                        visualization_path = (
                            visualization_dir
                            / result_filename
                        )

                        if not cv2.imwrite(
                            str(visualization_path),
                            plotted,
                        ):
                            raise RuntimeError(
                                "Failed to save visualization: "
                                f"{visualization_path}"
                            )

                        visualized_images += 1

                    if (
                        processed_images % 500 == 0
                        or processed_images == len(images)
                    ):
                        elapsed_now = (
                            time.time()
                            - start_time
                        )

                        speed = (
                            processed_images / elapsed_now
                            if elapsed_now > 0
                            else 0.0
                        )

                        print(
                            f"[{processed_images:5d}/{len(images)}] "
                            f"predictions={total_predictions:8d} "
                            f"speed={speed:.2f} img/s"
                        )


                corrupted_images.clear()

            prediction_file.write("]")

    except Exception:
        if temp_prediction_path.exists():
            temp_prediction_path.unlink()

        raise

    # --------------------------------------------------------

    # --------------------------------------------------------

    if processed_images != len(images):
        temp_prediction_path.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "Inference image count mismatch.\n"
            f"Expected : {len(images)}\n"
            f"Processed: {processed_images}"
        )

    if len(seen_image_ids) != len(images):
        temp_prediction_path.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "Unique image_id count mismatch."
        )


    os.replace(
        temp_prediction_path,
        prediction_path,
    )

    elapsed = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------

    # --------------------------------------------------------

    with image_summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            image_summaries,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------

    # --------------------------------------------------------

    metadata = {
        "model": "YOLO11m",
        "task": "bbox_detection",
        "evaluation_branch": "controlled_corruption",
        "checkpoint": str(
            model_path
        ),
        "checkpoint_sha256": (
            sha256_file(
                model_path
            )
        ),
        "dataset": "BDD100K",
        "split": "official_validation",
        "annotation": str(
            annotation_path
        ),
        "annotation_sha256": (
            sha256_file(
                annotation_path
            )
        ),
        "num_gt_images": len(
            gt_data["images"]
        ),
        "num_gt_annotations": len(
            gt_data["annotations"]
        ),
        "num_images_exported": (
            processed_images
        ),
        "num_predictions": (
            total_predictions
        ),
        "num_images_with_predictions": sum(
            1
            for item in image_summaries
            if item["prediction_count"] > 0
        ),
        "controlled_corruption": {
            "variant": variant,
            "corruption": args.corruption,
            "severity": args.severity,
            "parameters": corruption_parameters,
            "global_seed": GLOBAL_SEED,
            "seed_rule": (
                "SHA256(GLOBAL_SEED|image_key|corruption|severity) "
                "first 8 bytes little-endian unsigned"
            ),
            "module": str(
                corruption_module_path
            ),
            "module_sha256": (
                sha256_file(
                    corruption_module_path
                )
            ),
            "applied_before_model_preprocessing": True,
            "geometry_changed": False,
            "gt_modified": False,
            "full_corrupted_images_saved": False,
            "same_function_intended_for_all_models": True,
        },
        "class_mapping": {
            str(yolo_class_id): {
                "yolo_name": str(
                    model.names[
                        yolo_class_id
                    ]
                ),
                "bdd_category_id": (
                    class_mapping[
                        yolo_class_id
                    ]
                ),
                "bdd_name": (
                    BDD100K_CATEGORIES[
                        class_mapping[
                            yolo_class_id
                        ]
                    ]
                ),
            }
            for yolo_class_id
            in sorted(
                class_mapping
            )
        },
        "prediction_counts_by_category": {
            str(category_id): int(
                prediction_counts_by_class[
                    category_id
                ]
            )
            for category_id in BDD100K_CATEGORIES
        },
        "inference": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "conf": args.conf,
            "nms_iou": args.iou,
            "max_det": args.max_det,
            "rect": True,
            "test_time_augmentation": False,
            "agnostic_nms": False,
            "device": args.device,
            "source_type": "in_memory_PIL_RGB_batches",
        },
        "software": {
            "python_torch": (
                torch.__version__
            ),
            "ultralytics": (
                ultralytics.__version__
            ),
            "cuda_available": (
                torch.cuda.is_available()
            ),
        },
        "elapsed_seconds": round(
            elapsed,
            3,
        ),
        "images_per_second": (
            round(
                processed_images / elapsed,
                3,
            )
            if elapsed > 0
            else None
        ),
        "prediction_file": str(
            prediction_path
        ),
        "prediction_sha256": (
            sha256_file(
                prediction_path
            )
        ),
        "image_summary_file": str(
            image_summary_path
        ),
        "zero_area_bbox_count": (
            zero_area_bbox_count
        ),
    }

    with metadata_path.open(
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

    print()
    print("=" * 90)
    print("YOLO11m CONTROLLED-CORRUPTION EXPORT COMPLETE")
    print("=" * 90)
    print(f"Variant     : {variant}")
    print(f"Images      : {processed_images}")
    print(f"Predictions : {total_predictions}")
    print(f"Visualized  : {visualized_images}")
    print(f"Time        : {elapsed / 60:.2f} min")
    print(f"Predictions : {prediction_path}")
    print(f"SHA256      : {sha256_file(prediction_path)}")
    print(f"Metadata    : {metadata_path}")
    print(f"Image summary: {image_summary_path}")
    print(f"Zero-area bbox: {zero_area_bbox_count}")
    print("=" * 90)


if __name__ == "__main__":
    main()
