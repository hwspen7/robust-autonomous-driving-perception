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
from types import MethodType
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from controlled_corruptions import (
    GLOBAL_SEED,
    CORRUPTION_CONFIG,
    apply_corruption,
    deterministic_seed,
    image_sha256,
    validate_corruption,
    variant_name,
)

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

EXPECTED_VAL_IMAGES = 10_000
EXPECTED_VAL_ANNOTATIONS = 185_523

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
        "--corruption",
        type=str,
        required=True,
        choices=sorted(CORRUPTION_CONFIG.keys()),
        help="Controlled corruption applied before D-FINE preprocessing.",
    )

    parser.add_argument(
        "--severity",
        type=int,
        required=True,
        choices=(1, 2, 3),
        help="Controlled corruption severity: 1 / 2 / 3.",
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

    if len(images) != EXPECTED_VAL_IMAGES:
        raise ValueError(
            "Unexpected BDD100K val image count.\n"
            f"Expected: {EXPECTED_VAL_IMAGES}\n"
            f"Found   : {len(images)}"
        )

    if len(annotations) != EXPECTED_VAL_ANNOTATIONS:
        raise ValueError(
            "Unexpected BDD100K val annotation count.\n"
            f"Expected: {EXPECTED_VAL_ANNOTATIONS}\n"
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

    if dataset_size != EXPECTED_VAL_IMAGES:
        raise ValueError(
            "D-FINE validation dataset size mismatch.\n"
            f"Expected: {EXPECTED_VAL_IMAGES}\n"
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

def controlled_load_item(
        self: Any,
        idx: int,
) -> tuple[Image.Image, dict[str, Any]]:
    ""















    original_fn = getattr(
        self,
        "_controlled_original_load_item_fn",
        None,
    )

    if original_fn is None:
        raise RuntimeError(
            "Controlled D-FINE dataset is missing original load_item function."
        )

    image, target = original_fn(
        self,
        idx,
    )

    if not isinstance(image, Image.Image):
        raise TypeError(
            "D-FINE load_item must return a PIL image before preprocessing. "
            f"Found: {type(image)}"
        )

    image_id_value = target.get(
        "image_id"
    )

    if isinstance(
            image_id_value,
            torch.Tensor,
    ):
        image_id = int(
            image_id_value.item()
        )
    else:
        image_id = int(
            image_id_value
        )

    coco_image = self.coco.loadImgs(
        image_id
    )[0]

    image_key = str(
        coco_image[
            "file_name"
        ]
    )

    source_size = image.size

    corrupted = apply_corruption(
        image,
        image_key=image_key,
        corruption=self._controlled_corruption,
        severity=self._controlled_severity,
    )

    if corrupted.size != source_size:
        raise RuntimeError(
            "Controlled corruption changed image geometry.\n"
            f"image_id={image_id}\n"
            f"source={source_size}\n"
            f"corrupted={corrupted.size}"
        )

    return corrupted, target


def install_controlled_corruption(
        dataset: Any,
        *,
        corruption: str,
        severity: int,
) -> dict[str, Any]:
    ""






    validate_corruption(
        corruption,
        severity,
    )

    if not hasattr(
            dataset,
            "load_item",
    ):
        raise RuntimeError(
            "D-FINE validation dataset has no load_item method."
        )

    if not hasattr(
            dataset,
            "_transforms",
    ):
        raise RuntimeError(
            "D-FINE validation dataset has no _transforms attribute."
        )

    if getattr(
            dataset,
            "_controlled_original_load_item_fn",
            None,
    ) is not None:
        raise RuntimeError(
            "Controlled corruption has already been installed on this dataset."
        )

    original_fn = dataset.__class__.load_item

    dataset._controlled_original_load_item_fn = (
        original_fn
    )
    dataset._controlled_corruption = (
        corruption
    )
    dataset._controlled_severity = (
        severity
    )

    dataset.load_item = MethodType(
        controlled_load_item,
        dataset,
    )

    # --------------------------------------------------------





    # --------------------------------------------------------
    original_image, original_target = (
        original_fn(
            dataset,
            0,
        )
    )

    if not isinstance(
            original_image,
            Image.Image,
    ):
        raise TypeError(
            "Original D-FINE load_item is not returning PIL image."
        )

    image_id = int(
        original_target[
            "image_id"
        ].item()
    )

    image_key = str(
        dataset.coco.loadImgs(
            image_id
        )[0][
            "file_name"
        ]
    )

    expected_corrupted = (
        apply_corruption(
            original_image,
            image_key=image_key,
            corruption=corruption,
            severity=severity,
        )
    )

    patched_corrupted, patched_target = (
        dataset.load_item(
            0
        )
    )

    expected_hash = image_sha256(
        expected_corrupted
    )
    patched_hash = image_sha256(
        patched_corrupted
    )

    if expected_hash != patched_hash:
        raise RuntimeError(
            "D-FINE controlled load_item does not reproduce the public "
            "controlled corruption function exactly."
        )

    if (
            patched_corrupted.size
            != original_image.size
    ):
        raise RuntimeError(
            "D-FINE controlled dataset changed original geometry."
        )

    transformed_sample, transformed_target = (
        dataset[
            0
        ]
    )

    transformed_shape = getattr(
        transformed_sample,
        "shape",
        None,
    )

    if transformed_shape is None:
        raise RuntimeError(
            "D-FINE transformed sample has no shape."
        )

    if int(
            transformed_target[
                "image_id"
            ].item()
    ) != image_id:
        raise RuntimeError(
            "D-FINE target image_id changed after controlled corruption."
        )

    return {
        "image_id": image_id,
        "image_key": image_key,
        "raw_original_size": list(
            original_image.size
        ),
        "raw_corrupted_sha256": (
            patched_hash
        ),
        "corruption_seed": (
            deterministic_seed(
                image_key=image_key,
                corruption=corruption,
                severity=severity,
            )
        ),
        "transformed_sample_shape": [
            int(value)
            for value in transformed_shape
        ],
        "transformed_sample_dtype": str(
            getattr(
                transformed_sample,
                "dtype",
                None,
            )
        ),
        "geometry_changed": False,
        "gt_modified": False,
    }


def draw_predictions(
        dataset_root: Path,
        image_info: dict[str, Any],
        predictions: list[dict[str, Any]],
        output_path: Path,
        confidence_threshold: float,
        corruption: str,
        severity: int,
) -> None:
    ""






    image_path = (
            dataset_root
            / str(image_info["file_name"])
    ).resolve()

    if not image_path.is_file():
        raise RuntimeError(
            f"Cannot read visualization image: {image_path}"
        )

    with Image.open(
            image_path
    ) as source_image:
        corrupted_rgb = apply_corruption(
            source_image.convert(
                "RGB"
            ),
            image_key=str(
                image_info[
                    "file_name"
                ]
            ),
            corruption=corruption,
            severity=severity,
        )

    image = cv2.cvtColor(
        np.asarray(
            corrupted_rgb,
            dtype=np.uint8,
        ),
        cv2.COLOR_RGB2BGR,
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

    validate_corruption(
        args.corruption,
        args.severity,
    )

    corruption_variant = variant_name(
        args.corruption,
        args.severity,
    )

    corruption_parameters = dict(
        CORRUPTION_CONFIG[
            args.corruption
        ][
            args.severity
        ]
    )

    controlled_module_path = (
        Path(__file__)
        .resolve()
        .parent
        / "controlled_corruptions.py"
    )

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

    if not controlled_module_path.is_file():
        raise FileNotFoundError(
            "controlled_corruptions.py does not exist: "
            f"{controlled_module_path}"
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

        corruption_runtime_check = (
            install_controlled_corruption(
                solver.val_dataloader.dataset,
                corruption=args.corruption,
                severity=args.severity,
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

        controlled_module_sha256 = (
            sha256_file(
                controlled_module_path
            )
        )

        using_ema = (
                solver.ema is not None
        )

        print("=" * 78)
        print(
            "D-FINE-M BDD100K CONTROLLED-CORRUPTION EXPORT"
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
            f"Variant     : {corruption_variant}"
        )
        print(
            f"Corruption  : {args.corruption}"
        )
        print(
            f"Severity    : {args.severity}"
        )
        print(
            f"Parameters  : {corruption_parameters}"
        )
        print(
            f"Global seed : {GLOBAL_SEED}"
        )
        print(
            f"Corruption module SHA256: {controlled_module_sha256}"
        )
        print(
            "Injection    : raw PIL RGB -> corruption -> D-FINE _transforms"
        )
        print(
            f"Runtime check: {corruption_runtime_check}"
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
                                    "corruption_seed": (
                                        deterministic_seed(
                                            image_key=str(
                                                image_info[
                                                    "file_name"
                                                ]
                                            ),
                                            corruption=args.corruption,
                                            severity=args.severity,
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
                                    corruption=args.corruption,
                                    severity=args.severity,
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
                                    f"{args.limit if args.limit > 0 else EXPECTED_VAL_IMAGES}] "
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
            else EXPECTED_VAL_IMAGES
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
            "split": "official_validation",

            "controlled_corruption": {
                "mode": "on_the_fly",
                "variant": (
                    corruption_variant
                ),
                "corruption": (
                    args.corruption
                ),
                "severity": (
                    args.severity
                ),
                "parameters": (
                    corruption_parameters
                ),
                "global_seed": (
                    GLOBAL_SEED
                ),
                "module_path": str(
                    controlled_module_path
                ),
                "module_sha256": (
                    controlled_module_sha256
                ),
                "runtime_check": (
                    corruption_runtime_check
                ),
                "applied_before_model_preprocessing": True,
                "injection_point": (
                    "CocoDetection.load_item output PIL RGB "
                    "before dataset._transforms"
                ),
                "geometry_changed": False,
                "gt_modified": False,
                "full_corrupted_images_saved": False,
                "visualizations_are_regenerated_from_same_seed_rule": True,
            },

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

            "validation_num_workers": getattr(
                solver.val_dataloader,
                "num_workers",
                None,
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
        print("D-FINE-M CONTROLLED-CORRUPTION EXPORT COMPLETE")
        print("=" * 78)
        print(f"Variant: {corruption_variant}")
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
