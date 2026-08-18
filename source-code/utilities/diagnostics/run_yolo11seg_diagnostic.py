""

















from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
from pycocotools.coco import COCO
from ultralytics import YOLO


#
# BDD100K:
# 1 pedestrian
# 2 rider
# 3 car
# 4 truck
# 5 bus
# 6 train
# 7 motorcycle
# 8 bicycle
# 9 traffic light
# 10 traffic sign
#
# COCO:
# 0 person
# 1 bicycle
# 2 car
# 3 motorcycle
# 5 bus
# 6 train
# 7 truck
# 9 traffic light
COCO_TO_BDD = {
    0: (1, "pedestrian"),
    1: (8, "bicycle"),
    2: (3, "car"),
    3: (7, "motorcycle"),
    5: (5, "bus"),
    6: (6, "train"),
    7: (4, "truck"),
    9: (9, "traffic light"),
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO11m-Seg diagnostic inference on BDD100K."
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
        help="Original BDD100K COCO validation annotation JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11m-seg.pt",
        help="Ultralytics segmentation checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Inference device, e.g. 0 or cpu.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help="Prediction confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of images. 0 = all images.",
    )
    parser.add_argument(
        "--save-vis",
        type=int,
        default=50,
        help="Number of annotated images to save.",
    )

    return parser.parse_args()

def polygon_to_list(polygon: Any) -> list[float]:
    ""











    if polygon is None or len(polygon) < 3:
        return []

    return [
        round(float(value), 2)
        for point in polygon
        for value in point
    ]

def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.resolve()
    annotation_path = args.annotation.resolve()
    output_dir = args.output_dir.resolve()

    vis_dir = output_dir / "visualizations"

    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {dataset_root}"
        )

    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Annotation does not exist: {annotation_path}"
        )

    # ------------------------------------------------------------

    #


    # ------------------------------------------------------------
    coco = COCO(str(annotation_path))

    image_ids = sorted(coco.getImgIds())

    if args.limit > 0:
        image_ids = image_ids[: args.limit]

    image_infos = coco.loadImgs(image_ids)

    image_paths: list[Path] = []

    for info in image_infos:
        image_path = (dataset_root / info["file_name"]).resolve()

        if not image_path.exists():
            raise FileNotFoundError(
                f"BDD100K image does not exist: {image_path}"
            )

        image_paths.append(image_path)

    print("=" * 70)
    print("YOLO11m-Seg BDD100K Diagnostic")
    print("=" * 70)
    print(f"Model       : {args.model}")
    print(f"Images      : {len(image_paths)}")
    print(f"Image size  : {args.imgsz}")
    print(f"Batch       : {args.batch}")
    print(f"Confidence  : {args.conf}")
    print(f"Device      : {args.device}")
    print(f"Output      : {output_dir}")
    print("=" * 70)

    # ------------------------------------------------------------

    #


    # ------------------------------------------------------------
    model = YOLO(args.model)

    start_time = time.time()

    predictions: list[dict[str, Any]] = []
    image_summaries: list[dict[str, Any]] = []

    total_objects = 0
    saved_vis = 0

    # ------------------------------------------------------------
    # stream=True：

    #
    # retina_masks=True：

    # ------------------------------------------------------------
    # ------------------------------------------------------------

    #



    #


    # ------------------------------------------------------------
    source_txt = output_dir / "inference_paths.txt"

    with source_txt.open("w", encoding="utf-8") as f:
        for image_path in image_paths:
            f.write(f"{image_path}\n")

    results = model.predict(
        source=str(source_txt),
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        stream=True,
        retina_masks=True,
        verbose=False,
    )

    for index, result in enumerate(results, start=1):



        info = image_infos[index - 1]
        image_path = image_paths[index - 1]

        image_id = int(info["id"])
        file_name = info["file_name"]
        image_width = int(info["width"])
        image_height = int(info["height"])

        boxes = result.boxes
        masks = result.masks

        image_prediction_count = 0

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confidences = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)

            mask_polygons = (
                masks.xy
                if masks is not None
                else [None] * len(boxes)
            )

            for detection_index in range(len(boxes)):
                coco_class_id = int(classes[detection_index])


                if coco_class_id not in COCO_TO_BDD:
                    continue

                bdd_category_id, bdd_class_name = (
                    COCO_TO_BDD[coco_class_id]
                )

                x1, y1, x2, y2 = xyxy[detection_index]

                bbox_xywh = [
                    round(float(x1), 2),
                    round(float(y1), 2),
                    round(float(x2 - x1), 2),
                    round(float(y2 - y1), 2),
                ]

                polygon = []

                if (
                        masks is not None
                        and detection_index < len(mask_polygons)
                ):
                    polygon = polygon_to_list(
                        mask_polygons[detection_index]
                    )

                prediction = {
                    "image_id": image_id,
                    "file_name": file_name,

                    # BDD100K canonical category
                    "category_id": bdd_category_id,
                    "class_name": bdd_class_name,


                    "coco_class_id": coco_class_id,
                    "coco_class_name": model.names[coco_class_id],

                    "score": round(
                        float(confidences[detection_index]),
                        6,
                    ),


                    "bbox": bbox_xywh,

                    # COCO-style polygon list
                    "segmentation": (
                        [polygon]
                        if polygon
                        else []
                    ),

                    "image_width": image_width,
                    "image_height": image_height,
                }

                predictions.append(prediction)

                image_prediction_count += 1
                total_objects += 1

        image_summaries.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "prediction_count": image_prediction_count,
            }
        )

        # --------------------------------------------------------

        #


        # --------------------------------------------------------
        if saved_vis < args.save_vis:
            plotted = result.plot(
                boxes=True,
                masks=True,
                labels=True,
                conf=True,
            )

            output_image = (
                    vis_dir
                    / f"{image_id:07d}_{Path(file_name).name}"
            )

            cv2.imwrite(
                str(output_image),
                plotted,
            )

            saved_vis += 1

        if index % 100 == 0 or index == len(image_paths):
            elapsed = time.time() - start_time
            speed = index / elapsed if elapsed > 0 else 0.0

            print(
                f"[{index:5d}/{len(image_paths)}] "
                f"objects={total_objects:7d} "
                f"speed={speed:.2f} img/s"
            )

    elapsed = time.time() - start_time

    # ------------------------------------------------------------

    #

    # YOLO11m detection
    # D-FINE detection
    # YOLO11m-Seg
    #

    # ------------------------------------------------------------
    prediction_path = output_dir / "predictions.json"

    with prediction_path.open(
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(
            predictions,
            f,
            ensure_ascii=False,
        )

    summary_path = output_dir / "image_summary.json"

    with summary_path.open(
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(
            image_summaries,
            f,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------

    # ------------------------------------------------------------
    metadata = {
        "model": args.model,
        "task": "instance_segmentation_diagnostic",
        "dataset": "BDD100K",
        "dataset_root": str(dataset_root),
        "annotation": str(annotation_path),
        "num_images": len(image_paths),
        "num_predictions": len(predictions),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "device": args.device,
        "mapped_bdd_categories": {
            str(coco_id): {
                "bdd_category_id": bdd_id,
                "bdd_class_name": bdd_name,
            }
            for coco_id, (
                bdd_id,
                bdd_name,
            ) in COCO_TO_BDD.items()
        },
        "unmapped_bdd_categories": {
            "2": "rider",
            "10": "traffic sign",
        },
        "elapsed_seconds": round(elapsed, 2),
        "images_per_second": round(
            len(image_paths) / elapsed,
            2,
        ) if elapsed > 0 else None,
    }

    metadata_path = output_dir / "metadata.json"

    with metadata_path.open(
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("YOLO11m-Seg diagnostic completed")
    print("=" * 70)
    print(f"Images       : {len(image_paths)}")
    print(f"Predictions  : {len(predictions)}")
    print(f"Visualized   : {saved_vis}")
    print(f"Time         : {elapsed / 60:.2f} min")
    print(f"Predictions  : {prediction_path}")
    print(f"Metadata     : {metadata_path}")
    print(f"Visuals      : {vis_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
