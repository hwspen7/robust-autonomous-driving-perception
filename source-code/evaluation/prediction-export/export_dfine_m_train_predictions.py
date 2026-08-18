# TRAIN-SPLIT EXPORT COPY
# Source: export_dfine_m_predictions.py
# Original validation exporter remains unchanged.

""





























from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import torch

# ============================================================
# BDD100K original detection categories。

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

# ============================================================

#

#   instances_train_contiguous.json
#   instances_val_contiguous.json
#

# ============================================================

DFINE_CONTIGUOUS_CATEGORIES = {
    0: "pedestrian",
    1: "rider",
    2: "car",
    3: "truck",
    4: "bus",
    5: "train",
    6: "motorcycle",
    7: "bicycle",
    8: "traffic light",
    9: "traffic sign",
}

# D-FINE contiguous ID -> BDD100K original category ID
DFINE_TO_BDD = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
}

EXPECTED_TRAIN_IMAGES = 70_000
EXPECTED_TRAIN_ANNOTATIONS = 1_286_852

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export D-FINE-M predictions for "
            "BDD100K unified COCO evaluation."
        )
    )

    parser.add_argument(
        "--dfine-root",
        type=Path,
        required=True,
        help="Root directory of third_party/D-FINE.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="BDD100K D-FINE-M training config.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="D-FINE-M checkpoint to evaluate/export.",
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
        help=(
            "Original BDD100K validation annotation "
            "with category IDs 1~10."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory used to save prediction export.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Inference device, e.g. cuda:0 or cpu.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Maximum number of images to export. "
            "0 = complete validation set."
        ),
    )

    parser.add_argument(
        "--save-vis",
        type=int,
        default=20,
        help="Number of visualization images to save.",
    )

    parser.add_argument(
        "--vis-conf",
        type=float,
        default=0.25,
        help=(
            "Visualization-only score threshold. "
            "Does NOT affect predictions.json."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing predictions.json to be replaced.",
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

def normalize_name(name: str) -> str:
    ""





    return (
        str(name)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )

def load_original_gt(
        annotation_path: Path,
) -> tuple[
    dict[int, dict[str, Any]],
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
            "BDD100K GT must be a COCO dictionary."
        )

    for key in (
            "images",
            "annotations",
            "categories",
    ):
        if key not in data:
            raise KeyError(
                f"GT missing key: {key}"
            )

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    if len(images) != EXPECTED_TRAIN_IMAGES:
        raise ValueError(
            "Unexpected BDD100K val image count.\n"
            f"Expected: {EXPECTED_TRAIN_IMAGES}\n"
            f"Found   : {len(images)}"
        )

    if len(annotations) != EXPECTED_TRAIN_ANNOTATIONS:
        raise ValueError(
            "Unexpected BDD100K val annotation count.\n"
            f"Expected: {EXPECTED_TRAIN_ANNOTATIONS}\n"
            f"Found   : {len(annotations)}"
        )

    actual_categories = {
        int(category["id"]): normalize_name(
            category["name"]
        )
        for category in categories
    }

    expected_categories = {
        category_id: normalize_name(name)
        for category_id, name
        in BDD100K_CATEGORIES.items()
    }

    if actual_categories != expected_categories:
        raise ValueError(
            "Original BDD100K category definition mismatch.\n"
            f"Expected: {expected_categories}\n"
            f"Found   : {actual_categories}\n\n"
            "Final evaluation must use original category IDs 1~10."
        )

    image_by_id: dict[
        int,
        dict[str, Any],
    ] = {}

    for image in images:
        image_id = int(
            image["id"]
        )

        if image_id in image_by_id:
            raise ValueError(
                f"Duplicate GT image_id: {image_id}"
            )

        image_by_id[
            image_id
        ] = image

    return image_by_id, data

def find_coco_categories(
        dataset: Any,
) -> dict[int, str] | None:
    ""





    current = dataset

    for _ in range(8):
        coco = getattr(
            current,
            "coco",
            None,
        )

        if (
                coco is not None
                and hasattr(coco, "cats")
        ):
            return {
                int(category_id): normalize_name(
                    category["name"]
                )
                for category_id, category
                in coco.cats.items()
            }

        current = getattr(
            current,
            "dataset",
            None,
        )

        if current is None:
            break

    return None

def validate_dfine_dataset(
        val_dataloader: Any,
) -> dict[int, str]:
    ""






    dataset = val_dataloader.dataset

    dataset_size = len(
        dataset
    )

    if dataset_size != EXPECTED_TRAIN_IMAGES:
        raise ValueError(
            "D-FINE validation dataset size mismatch.\n"
            f"Expected: {EXPECTED_TRAIN_IMAGES}\n"
            f"Found   : {dataset_size}"
        )

    categories = find_coco_categories(
        dataset
    )

    if categories is None:
        raise RuntimeError(
            "Cannot inspect D-FINE validation COCO categories.\n"
            "Exporter refuses to guess category mapping."
        )

    expected = {
        category_id: normalize_name(name)
        for category_id, name
        in DFINE_CONTIGUOUS_CATEGORIES.items()
    }

    if categories != expected:
        raise ValueError(
            "D-FINE validation category definition mismatch.\n"
            f"Expected contiguous categories: {expected}\n"
            f"Found                         : {categories}\n\n"
            "Do not continue until category mapping is confirmed."
        )

    return categories

def draw_predictions(
        dataset_root: Path,
        image_info: dict[str, Any],
        predictions: list[dict[str, Any]],
        output_path: Path,
        confidence_threshold: float,
) -> None:
    ""






    image_path = (
            dataset_root
            / str(image_info["file_name"])
    ).resolve()

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise RuntimeError(
            f"Cannot read visualization image: {image_path}"
        )

    for prediction in predictions:
        score = float(
            prediction["score"]
        )

        if score < confidence_threshold:
            continue

        x, y, width, height = (
            prediction["bbox"]
        )

        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + width))
        y2 = int(round(y + height))

        category_id = int(
            prediction["category_id"]
        )

        class_name = (
            BDD100K_CATEGORIES[
                category_id
            ]
        )

        label = (
            f"{class_name} "
            f"{score:.2f}"
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            image,
            label,
            (
                x1,
                max(y1 - 5, 15),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
            str(output_path),
            image,
    ):
        raise RuntimeError(
            f"Failed to save visualization: {output_path}"
        )

def main() -> None:
    args = parse_args()

    dfine_root = (
        args.dfine_root
        .expanduser()
        .resolve()
    )

    config_path = (
        args.config
        .expanduser()
        .resolve()
    )

    checkpoint_path = (
        args.checkpoint
        .expanduser()
        .resolve()
    )

    dataset_root = (
        args.dataset_root
        .expanduser()
        .resolve()
    )

    annotation_path = (
        args.annotation
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    # --------------------------------------------------------

    # --------------------------------------------------------

    if not dfine_root.is_dir():
        raise FileNotFoundError(
            f"D-FINE root does not exist: {dfine_root}"
        )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"D-FINE config does not exist: {config_path}"
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"D-FINE checkpoint does not exist: {checkpoint_path}"
        )

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"BDD100K dataset root does not exist: {dataset_root}"
        )

    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"BDD100K annotation does not exist: {annotation_path}"
        )

    if args.limit < 0:
        raise ValueError(
            "--limit cannot be negative."
        )

    if args.save_vis < 0:
        raise ValueError(
            "--save-vis cannot be negative."
        )

    if not 0.0 <= args.vis_conf <= 1.0:
        raise ValueError(
            "--vis-conf must be in [0, 1]."
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

    if (
            prediction_path.exists()
            and not args.overwrite
    ):
        raise FileExistsError(
            f"{prediction_path} already exists.\n"
            "Use --overwrite only when replacement is intentional."
        )

    # --------------------------------------------------------

    # --------------------------------------------------------

    image_by_id, gt_data = (
        load_original_gt(
            annotation_path
        )
    )

    # --------------------------------------------------------

    #


    # --------------------------------------------------------

    sys.path.insert(
        0,
        str(dfine_root),
    )


    from src.core import YAMLConfig
    from src.misc import dist_utils
    from src.solver import TASKS

    # --------------------------------------------------------


    # --------------------------------------------------------

    dist_utils.setup_distributed(
        print_rank=0,
        print_method="builtin",
        seed=0,
    )

    try:
        runtime_output_dir = (
                output_dir
                / "_dfine_runtime"
        )

        # ----------------------------------------------------

        #
        # resume：

        #
        # output_dir：

        # ----------------------------------------------------

        cfg = YAMLConfig(
            str(config_path),
            resume=str(
                checkpoint_path
            ),
            device=args.device,
            output_dir=str(
                runtime_output_dir
            ),
            use_wandb=False,
        )

        # ----------------------------------------------------


        # ----------------------------------------------------

        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg[
                "HGNetv2"
            ][
                "pretrained"
            ] = False

        if cfg.yaml_cfg.get(
                "task"
        ) != "detection":
            raise ValueError(
                "D-FINE config task must be detection."
            )

        num_classes = int(
            cfg.yaml_cfg.get(
                "num_classes",
                -1,
            )
        )

        if num_classes != 10:
            raise ValueError(
                "D-FINE config must contain exactly 10 classes.\n"
                f"Found num_classes={num_classes}"
            )

        remap = cfg.yaml_cfg.get(
            "remap_mscoco_category",
            None,
        )

        if remap is not False:
            raise ValueError(
                "Expected remap_mscoco_category=False "
                "for BDD100K contiguous training.\n"
                f"Found: {remap}"
            )

        # ----------------------------------------------------

        #

        #   model
        #   checkpoint
        #   EMA
        #   postprocessor
        #   validation dataloader
        #


        # ----------------------------------------------------

        solver_class = TASKS[
            cfg.yaml_cfg["task"]
        ]

        solver = solver_class(
            cfg
        )

        solver.eval()

        # ----------------------------------------------------

        #

        #     solver.ema.module
        #

        #     solver.model
        #

        # ----------------------------------------------------

        module = (
            solver.ema.module
            if solver.ema is not None
            else solver.model
        )

        module.eval()

        if hasattr(
                solver.postprocessor,
                "eval",
        ):
            solver.postprocessor.eval()

        # ----------------------------------------------------

        # ----------------------------------------------------

        postprocessor_remap = getattr(
            solver.postprocessor,
            "remap_mscoco_category",
            None,
        )

        if postprocessor_remap is not False:
            raise ValueError(
                "D-FINE postprocessor must use "
                "remap_mscoco_category=False.\n"
                f"Found: {postprocessor_remap}"
            )

        # ----------------------------------------------------

        #
        # 0 pedestrian
        # ...
        # 9 traffic sign
        #

        # ----------------------------------------------------

        dfine_categories = (
            validate_dfine_dataset(
                solver.val_dataloader
            )
        )

        actual_batch_size = getattr(
            solver.val_dataloader,
            "batch_size",
            None,
        )

        checkpoint_sha256 = (
            sha256_file(
                checkpoint_path
            )
        )

        config_sha256 = (
            sha256_file(
                config_path
            )
        )

        gt_sha256 = (
            sha256_file(
                annotation_path
            )
        )

        using_ema = (
                solver.ema is not None
        )

        print("=" * 78)
        print(
            "D-FINE-M BDD100K PREDICTION EXPORT"
        )
        print("=" * 78)
        print(
            f"D-FINE root : {dfine_root}"
        )
        print(
            f"Config      : {config_path}"
        )
        print(
            f"Checkpoint  : {checkpoint_path}"
        )
        print(
            f"Checkpoint SHA256: {checkpoint_sha256}"
        )
        print(
            f"Device      : {solver.device}"
        )
        print(
            f"Val images  : {len(solver.val_dataloader.dataset)}"
        )
        print(
            f"Batch       : {actual_batch_size}"
        )
        print(
            f"EMA model   : {using_ema}"
        )
        print(
            f"remap COCO  : {postprocessor_remap}"
        )
        print(
            f"Limit       : {args.limit if args.limit > 0 else 'ALL'}"
        )
        print(
            f"Output      : {output_dir}"
        )
        print()
        print(
            "D-FINE contiguous -> BDD original:"
        )

        for dfine_id in sorted(
                DFINE_TO_BDD
        ):
            bdd_id = (
                DFINE_TO_BDD[
                    dfine_id
                ]
            )

            print(
                f"  D-FINE {dfine_id:2d} "
                f"{DFINE_CONTIGUOUS_CATEGORIES[dfine_id]:<15} "
                f"-> BDD {bdd_id:2d} "
                f"{BDD100K_CATEGORIES[bdd_id]}"
            )

        print("=" * 78)

        # ----------------------------------------------------

        # ----------------------------------------------------

        processed_images = 0
        total_predictions = 0
        visualized_images = 0

        seen_image_ids: set[
            int
        ] = set()

        prediction_counts = Counter()

        out_of_bounds_count = 0
        negative_size_bbox_count = 0
        zero_area_bbox_count = 0

        minimum_score = math.inf
        maximum_score = -math.inf

        image_summaries: list[
            dict[str, Any]
        ] = []

        start_time = time.time()

        first_prediction = True
        stop_export = False

        try:
            with temp_prediction_path.open(
                    "w",
                    encoding="utf-8",
            ) as prediction_file:

                prediction_file.write(
                    "["
                )

                # ------------------------------------------------

                #
                # samples -> device
                # targets -> device
                # outputs = model(samples)
                # orig_target_sizes = target["orig_size"]
                # results = postprocessor(outputs, orig_target_sizes)
                # ------------------------------------------------

                with torch.no_grad():

                    for (
                            samples,
                            targets,
                    ) in solver.val_dataloader:

                        samples = samples.to(
                            solver.device
                        )

                        device_targets = [
                            {
                                key: (
                                    value.to(
                                        solver.device
                                    )
                                    if isinstance(
                                        value,
                                        torch.Tensor,
                                    )
                                    else value
                                )
                                for key, value
                                in target.items()
                            }
                            for target in targets
                        ]

                        outputs = module(
                            samples
                        )

                        orig_target_sizes = (
                            torch.stack(
                                [
                                    target[
                                        "orig_size"
                                    ]
                                    for target
                                    in device_targets
                                ],
                                dim=0,
                            )
                        )

                        results = (
                            solver.postprocessor(
                                outputs,
                                orig_target_sizes,
                            )
                        )

                        if len(results) != len(
                                device_targets
                        ):
                            raise RuntimeError(
                                "D-FINE result/target batch size mismatch."
                            )

                        for (
                                target,
                                result,
                        ) in zip(
                            device_targets,
                            results,
                        ):

                            if (
                                    args.limit > 0
                                    and processed_images
                                    >= args.limit
                            ):
                                stop_export = True
                                break

                            image_id = int(
                                target[
                                    "image_id"
                                ].item()
                            )

                            if image_id not in image_by_id:
                                raise RuntimeError(
                                    "D-FINE returned image_id "
                                    "not present in original BDD100K GT:\n"
                                    f"{image_id}"
                                )

                            if image_id in seen_image_ids:
                                raise RuntimeError(
                                    "Duplicate D-FINE inference result "
                                    f"for image_id={image_id}"
                                )

                            seen_image_ids.add(
                                image_id
                            )

                            image_info = (
                                image_by_id[
                                    image_id
                                ]
                            )

                            # ------------------------------------


                            #

                            # target["orig_size"] = [width, height]
                            #

                            # orig_size = [1280, 720]
                            # ------------------------------------

                            target_width = int(
                                target[
                                    "orig_size"
                                ][0].item()
                            )

                            target_height = int(
                                target[
                                    "orig_size"
                                ][1].item()
                            )

                            gt_width = int(
                                image_info[
                                    "width"
                                ]
                            )

                            gt_height = int(
                                image_info[
                                    "height"
                                ]
                            )

                            if (
                                    target_width
                                    != gt_width
                                    or target_height
                                    != gt_height
                            ):
                                raise RuntimeError(
                                    "Original image size mismatch.\n"
                                    f"image_id={image_id}\n"
                                    f"D-FINE target: "
                                    f"{target_width}x{target_height}\n"
                                    f"BDD GT       : "
                                    f"{gt_width}x{gt_height}"
                                )

                            boxes = (
                                result[
                                    "boxes"
                                ]
                                .detach()
                                .cpu()
                            )

                            labels = (
                                result[
                                    "labels"
                                ]
                                .detach()
                                .cpu()
                            )

                            scores = (
                                result[
                                    "scores"
                                ]
                                .detach()
                                .cpu()
                            )

                            if not (
                                    len(boxes)
                                    == len(labels)
                                    == len(scores)
                            ):
                                raise RuntimeError(
                                    "D-FINE boxes/labels/scores "
                                    "length mismatch."
                                )

                            image_predictions: list[
                                dict[str, Any]
                            ] = []

                            for detection_index in range(
                                    len(boxes)
                            ):
                                dfine_label = int(
                                    labels[
                                        detection_index
                                    ].item()
                                )

                                if dfine_label not in DFINE_TO_BDD:
                                    raise RuntimeError(
                                        "Unexpected D-FINE label.\n"
                                        f"image_id={image_id}\n"
                                        f"label={dfine_label}\n"
                                        "Expected contiguous label 0~9."
                                    )

                                category_id = (
                                    DFINE_TO_BDD[
                                        dfine_label
                                    ]
                                )

                                score = float(
                                    scores[
                                        detection_index
                                    ].item()
                                )

                                if not math.isfinite(
                                        score
                                ):
                                    raise RuntimeError(
                                        "Non-finite D-FINE score."
                                    )

                                if not (
                                        0.0
                                        <= score
                                        <= 1.0
                                ):
                                    raise RuntimeError(
                                        "D-FINE score outside [0,1]: "
                                        f"{score}"
                                    )

                                x1, y1, x2, y2 = [
                                    float(value)
                                    for value
                                    in boxes[
                                        detection_index
                                    ].tolist()
                                ]

                                width = (
                                        x2 - x1
                                )

                                height = (
                                        y2 - y1
                                )

                                if not all(
                                        math.isfinite(
                                            value
                                        )
                                        for value in (
                                                x1,
                                                y1,
                                                width,
                                                height,
                                        )
                                ):
                                    raise RuntimeError(
                                        "Non-finite D-FINE bbox."
                                    )




                                #


                                if (
                                        width < 0
                                        or height < 0
                                ):
                                    negative_size_bbox_count += 1

                                if (
                                        width == 0
                                        or height == 0
                                ):
                                    zero_area_bbox_count += 1

                                # --------------------------------
                                # D-FINE postprocessor boxes:
                                # [x1, y1, x2, y2]
                                #
                                # COCO prediction:
                                # [x, y, width, height]
                                #

                                # --------------------------------

                                prediction = {
                                    "image_id": (
                                        image_id
                                    ),
                                    "category_id": (
                                        category_id
                                    ),
                                    "bbox": [
                                        x1,
                                        y1,
                                        width,
                                        height,
                                    ],
                                    "score": (
                                        score
                                    ),
                                }

                                # --------------------------------


                                # --------------------------------

                                if (
                                        x1 < 0
                                        or y1 < 0
                                        or x1 + width
                                        > gt_width
                                        or y1 + height
                                        > gt_height
                                ):
                                    out_of_bounds_count += 1

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

                                image_predictions.append(
                                    prediction
                                )

                                total_predictions += 1

                                prediction_counts[
                                    category_id
                                ] += 1

                                minimum_score = min(
                                    minimum_score,
                                    score,
                                )

                                maximum_score = max(
                                    maximum_score,
                                    score,
                                )

                            image_summaries.append(
                                {
                                    "image_id": (
                                        image_id
                                    ),
                                    "file_name": (
                                        image_info[
                                            "file_name"
                                        ]
                                    ),
                                    "prediction_count": (
                                        len(
                                            image_predictions
                                        )
                                    ),
                                    "max_score": (
                                        max(
                                            (
                                                item[
                                                    "score"
                                                ]
                                                for item
                                                in image_predictions
                                            ),
                                            default=None,
                                        )
                                    ),
                                }
                            )

                            if (
                                    visualized_images
                                    < args.save_vis
                            ):
                                visualization_path = (
                                        visualization_dir
                                        / (
                                            f"{image_id}_"
                                            f"{Path(image_info['file_name']).name}"
                                        )
                                )

                                draw_predictions(
                                    dataset_root=dataset_root,
                                    image_info=image_info,
                                    predictions=image_predictions,
                                    output_path=visualization_path,
                                    confidence_threshold=args.vis_conf,
                                )

                                visualized_images += 1

                            processed_images += 1

                            if (
                                    processed_images
                                    % 500
                                    == 0
                            ):
                                elapsed = (
                                        time.time()
                                        - start_time
                                )

                                speed = (
                                    processed_images
                                    / elapsed
                                    if elapsed > 0
                                    else 0.0
                                )

                                print(
                                    f"[{processed_images:5d}/"
                                    f"{args.limit if args.limit > 0 else EXPECTED_TRAIN_IMAGES}] "
                                    f"predictions={total_predictions:8d} "
                                    f"speed={speed:.2f} img/s"
                                )

                        if stop_export:
                            break

                prediction_file.write(
                    "]"
                )

        except Exception:
            if temp_prediction_path.exists():
                temp_prediction_path.unlink()

            raise

        # ----------------------------------------------------

        # ----------------------------------------------------

        expected_images = (
            args.limit
            if args.limit > 0
            else EXPECTED_TRAIN_IMAGES
        )

        if processed_images != expected_images:
            temp_prediction_path.unlink(
                missing_ok=True
            )

            raise RuntimeError(
                "D-FINE export image count mismatch.\n"
                f"Expected : {expected_images}\n"
                f"Processed: {processed_images}"
            )

        if len(
                seen_image_ids
        ) != processed_images:
            temp_prediction_path.unlink(
                missing_ok=True
            )

            raise RuntimeError(
                "Unique image_id count mismatch."
            )

        # ----------------------------------------------------

        # ----------------------------------------------------

        os.replace(
            temp_prediction_path,
            prediction_path,
        )

        elapsed = (
                time.time()
                - start_time
        )

        # ----------------------------------------------------

        # ----------------------------------------------------

        image_summary_path = (
                output_dir
                / "image_summary.json"
        )

        with image_summary_path.open(
                "w",
                encoding="utf-8",
        ) as f:
            json.dump(
                image_summaries,
                f,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        # ----------------------------------------------------

        #



        # ----------------------------------------------------

        metadata = {
            "model": "D-FINE-M",
            "task": "bbox_detection",
            "dataset": "BDD100K",
            "split": "official_training",

            "dfine_root": str(
                dfine_root
            ),

            "config": str(
                config_path
            ),
            "config_sha256": (
                config_sha256
            ),

            "checkpoint": str(
                checkpoint_path
            ),
            "checkpoint_sha256": (
                checkpoint_sha256
            ),

            "annotation": str(
                annotation_path
            ),
            "annotation_sha256": (
                gt_sha256
            ),

            "num_gt_images": len(
                gt_data[
                    "images"
                ]
            ),

            "num_gt_annotations": len(
                gt_data[
                    "annotations"
                ]
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
                if item[
                    "prediction_count"
                ] > 0
            ),

            "using_ema_model": (
                using_ema
            ),

            "device": str(
                solver.device
            ),

            "validation_batch_size": (
                actual_batch_size
            ),

            "postprocessor": {
                "class": (
                    solver.postprocessor
                    .__class__
                    .__name__
                ),
                "remap_mscoco_category": (
                    postprocessor_remap
                ),
                "num_top_queries": getattr(
                    solver.postprocessor,
                    "num_top_queries",
                    None,
                ),
            },

            "dfine_contiguous_categories": {
                str(category_id): name
                for category_id, name
                in dfine_categories.items()
            },

            "dfine_to_bdd_mapping": {
                str(dfine_id): {
                    "dfine_name": (
                        DFINE_CONTIGUOUS_CATEGORIES[
                            dfine_id
                        ]
                    ),
                    "bdd_category_id": (
                        bdd_id
                    ),
                    "bdd_name": (
                        BDD100K_CATEGORIES[
                            bdd_id
                        ]
                    ),
                }
                for dfine_id, bdd_id
                in DFINE_TO_BDD.items()
            },

            "prediction_counts_by_category": {
                str(category_id): int(
                    prediction_counts[
                        category_id
                    ]
                )
                for category_id
                in BDD100K_CATEGORIES
            },

            "out_of_bounds_bbox_count": (
                out_of_bounds_count
            ),

            "negative_size_bbox_count": (
                negative_size_bbox_count
            ),

            "zero_area_bbox_count": (
                zero_area_bbox_count
            ),

            "minimum_score": (
                minimum_score
                if total_predictions > 0
                else None
            ),

            "maximum_score": (
                maximum_score
                if total_predictions > 0
                else None
            ),

            "elapsed_seconds": round(
                elapsed,
                3,
            ),

            "images_per_second": (
                round(
                    processed_images
                    / elapsed,
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

            "software": {
                "python": (
                    sys.version
                    .split()[0]
                ),
                "torch": (
                    torch.__version__
                ),
                "cuda_available": (
                    torch.cuda.is_available()
                ),
            },
        }

        metadata_path = (
                output_dir
                / "metadata.json"
        )

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
        print("=" * 78)
        print("D-FINE-M PREDICTION EXPORT COMPLETE")
        print("=" * 78)
        print(f"Images: {processed_images}")
        print(f"Predictions: {total_predictions}")
        print(f"Visualized: {visualized_images}")
        print(f"EMA model: {using_ema}")
        print(f"Out-of-bounds: {out_of_bounds_count}")
        print(
            f"Negative-size bbox: {negative_size_bbox_count}"
        )
        print(f"Zero-area bbox: {zero_area_bbox_count}")
        print(f"Time: {elapsed / 60:.2f} min")
        print(f"Predictions  : {prediction_path}")
        print(f"SHA256: {sha256_file(prediction_path)}")
        print(f"Metadata: {metadata_path}")
        print("=" * 78)

    finally:
        dist_utils.cleanup()

if __name__ == "__main__":
    main()
