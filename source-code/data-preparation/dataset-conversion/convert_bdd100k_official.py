"""
Convert official BDD100K detection JSON labels to YOLO and COCO.

Expected official layout:
- images/100k/train/*.jpg
- images/100k/val/*.jpg
- labels/det_20/det_train.json
- labels/det_20/det_val.json

The script keeps the same 10-class taxonomy used by the existing
Dataset Ninja/Supervisely converter.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, TextIO

import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.data_taxonomy import (
    COCO_CLASS_TO_ID,
    DETECTION_CLASSES,
    SOURCE_TO_CANONICAL,
    YOLO_CLASS_TO_ID,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert official BDD100K JSON labels to YOLO/COCO."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets/downloads/bdd100k"),
        help="BDD100K root containing images/ and labels/.",
    )
    parser.add_argument(
        "--label-file",
        type=Path,
        default=None,
        help="Explicit official det_{split}.json path.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help=(
            "Explicit image root. Can be images/100k, images/100k/train, "
            "or a directory containing split subdirectories."
        ),
    )
    parser.add_argument(
        "--yolo-output-root",
        type=Path,
        default=Path("datasets/yolo/bdd100k"),
        help="YOLO output root.",
    )
    parser.add_argument(
        "--coco-output-root",
        type=Path,
        default=Path("datasets/coco/bdd100k"),
        help="COCO output root.",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Dataset split, for example train or val.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("yolo", "coco"),
        default=("yolo", "coco"),
        help="Output formats to generate.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Link or copy images into output datasets.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=1280,
        help="Fallback image width when labels do not include size.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=720,
        help="Fallback image height when labels do not include size.",
    )
    parser.add_argument(
        "--read-image-size",
        action="store_true",
        help="Read actual image dimensions with Pillow.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Convert at most this many images.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing outputs for this split before converting.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    expanded = path.expanduser()

    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded

    return expanded.resolve(
        strict=False,
    )


def dedupe_paths(
    paths: list[Path],
) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()

    for path in paths:
        key = str(path)

        if key in seen:
            continue

        result.append(path)
        seen.add(key)

    return result


def resolve_label_file(
    data_root: Path,
    label_file: Path | None,
    split: str,
) -> Path:
    if label_file is not None:
        resolved_label_file = project_path(label_file)

        if resolved_label_file.exists():
            return resolved_label_file

        raise FileNotFoundError(
            f"Label file does not exist: {resolved_label_file}"
        )

    candidates = dedupe_paths(
        [
            data_root / "labels" / "det_20" / f"det_{split}.json",
            data_root / "bdd100k" / "labels" / "det_20" / f"det_{split}.json",
            data_root / "det_20" / f"det_{split}.json",
            data_root / "labels" / f"det_{split}.json",
            data_root / f"det_{split}.json",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find official BDD100K label file for split={split}.\n"
        "Searched:\n"
        + "\n".join(f"  - {path}" for path in candidates)
    )


def resolve_image_dir(
    data_root: Path,
    image_root: Path | None,
    split: str,
) -> Path:
    if image_root is not None:
        root = project_path(image_root)
        candidates = [
            root,
            root / split,
            root / "images" / "100k" / split,
            root / "100k" / split,
            root / "images" / split,
        ]
    else:
        candidates = [
            data_root / "images" / "100k" / split,
            data_root / "bdd100k" / "images" / "100k" / split,
            data_root / "100k" / split,
            data_root / "images" / split,
            data_root / split,
        ]

    candidates = dedupe_paths(candidates)

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"Could not find BDD100K images for split={split}.\n"
        "Searched:\n"
        + "\n".join(f"  - {path}" for path in candidates)
    )


def load_official_labels(
    label_file: Path,
) -> list[dict[str, Any]]:
    with label_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise TypeError(
            f"Official BDD100K label file must contain a list: {label_file}"
        )

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(
                f"Label item {index} is not an object in {label_file}"
            )

    return data


def resolve_image_path(
    image_dir: Path,
    image_name: str,
) -> Path:
    image_path = image_dir / image_name

    if image_path.exists():
        return image_path

    stem = Path(image_name).stem
    candidates = [
        path
        for path in image_dir.glob(f"{stem}.*")
        if path.is_file()
    ]

    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not find image for label entry: {image_name}"
    )


def read_image_size(
    image_path: Path,
) -> tuple[int, int]:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Pillow is required for --read-image-size. "
            "Install it with: pip install Pillow"
        ) from exc

    with Image.open(image_path) as image:
        return image.size


def resolve_image_size(
    record: dict[str, Any],
    image_path: Path,
    args: argparse.Namespace,
) -> tuple[int, int]:
    width = (
        record.get("width")
        or record.get("image_width")
        or record.get("imageWidth")
    )
    height = (
        record.get("height")
        or record.get("image_height")
        or record.get("imageHeight")
    )

    if width is not None and height is not None:
        return int(width), int(height)

    if args.read_image_size:
        return read_image_size(
            image_path
        )

    return args.image_width, args.image_height


def parse_bbox(
    label: dict[str, Any],
    image_width: int,
    image_height: int,
) -> list[float] | None:
    box = label.get("box2d")

    if not isinstance(box, dict):
        return None

    try:
        x1 = float(box["x1"])
        y1 = float(box["y1"])
        x2 = float(box["x2"])
        y2 = float(box["y2"])
    except (KeyError, TypeError, ValueError):
        return None

    left = max(0.0, min(x1, x2))
    top = max(0.0, min(y1, y2))
    right = min(float(image_width), max(x1, x2))
    bottom = min(float(image_height), max(y1, y2))

    width = right - left
    height = bottom - top

    if width <= 0 or height <= 0:
        return None

    return [
        left,
        top,
        width,
        height,
    ]


def coco_bbox_to_yolo(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    left, top, width, height = bbox

    return (
        (left + width / 2) / image_width,
        (top + height / 2) / image_height,
        width / image_width,
        height / image_height,
    )


def parse_detection_objects(
    record: dict[str, Any],
    image_width: int,
    image_height: int,
) -> tuple[list[dict[str, Any]], int, int]:
    detection_objects: list[dict[str, Any]] = []
    invalid_boxes = 0
    skipped_objects = 0

    labels = record.get("labels", [])

    if labels is None:
        labels = []

    if not isinstance(labels, list):
        raise TypeError(
            f"labels must be a list for image {record.get('name')}"
        )

    for label in labels:
        if not isinstance(label, dict):
            skipped_objects += 1
            continue

        source_class = str(
            label.get("category", "")
        )
        canonical_class = SOURCE_TO_CANONICAL.get(
            source_class
        )

        if canonical_class is None:
            skipped_objects += 1
            continue

        bbox = parse_bbox(
            label=label,
            image_width=image_width,
            image_height=image_height,
        )

        if bbox is None:
            invalid_boxes += 1
            continue

        attributes = label.get("attributes", {})

        if not isinstance(attributes, dict):
            attributes = {}

        detection_objects.append(
            {
                "class_name": canonical_class,
                "yolo_class_id": YOLO_CLASS_TO_ID[
                    canonical_class
                ],
                "coco_category_id": COCO_CLASS_TO_ID[
                    canonical_class
                ],
                "bbox": bbox,
                "attributes": {
                    "occluded": bool(
                        attributes.get(
                            "occluded",
                            False,
                        )
                    ),
                    "truncated": bool(
                        attributes.get(
                            "truncated",
                            False,
                        )
                    ),
                    "traffic_light_color": attributes.get(
                        "trafficLightColor",
                        "none",
                    ),
                },
            }
        )

    return detection_objects, invalid_boxes, skipped_objects


def prepare_image(
    source_path: Path,
    destination_path: Path,
    mode: str,
) -> None:
    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination_path.exists() or destination_path.is_symlink():
        return

    if mode == "copy":
        shutil.copy2(
            source_path,
            destination_path,
        )
        return

    destination_path.symlink_to(
        source_path.resolve()
    )


def clean_split_outputs(
    split: str,
    formats: set[str],
    yolo_output_root: Path,
    coco_output_root: Path,
) -> None:
    if "yolo" in formats:
        for path in (
            yolo_output_root / "images" / split,
            yolo_output_root / "labels" / split,
        ):
            if path.exists() or path.is_symlink():
                shutil.rmtree(path)

        for path in (
            yolo_output_root / "metadata" / f"{split}.jsonl",
            yolo_output_root / "metadata" / f"stats_{split}.json",
        ):
            path.unlink(missing_ok=True)

    if "coco" in formats:
        image_dir = coco_output_root / "images" / split

        if image_dir.exists() or image_dir.is_symlink():
            shutil.rmtree(image_dir)

        annotation_dir = coco_output_root / "annotations"

        for path in (
            annotation_dir / f"instances_{split}.json",
            annotation_dir / f"stats_{split}.json",
            annotation_dir / f".{split}_annotations.jsonl",
        ):
            path.unlink(missing_ok=True)


def write_yolo_label(
    label_path: Path,
    detection_objects: list[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> None:
    lines: list[str] = []

    for obj in detection_objects:
        center_x, center_y, width, height = coco_bbox_to_yolo(
            bbox=obj["bbox"],
            image_width=image_width,
            image_height=image_height,
        )

        lines.append(
            f"{obj['yolo_class_id']} "
            f"{center_x:.8f} "
            f"{center_y:.8f} "
            f"{width:.8f} "
            f"{height:.8f}"
        )

    label_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    label_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_yolo_yaml(
    output_root: Path,
) -> None:
    if (output_root / "images" / "val").exists():
        val_path = "images/val"
    elif (output_root / "images" / "test").exists():
        val_path = "images/test"
    else:
        val_path = "images/train"

    yaml_data = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": val_path,
        "test": "images/test",
        "names": {
            class_id: class_name
            for class_id, class_name in enumerate(DETECTION_CLASSES)
        },
    }

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (output_root / "data.yaml").open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            yaml_data,
            file,
            allow_unicode=True,
            sort_keys=False,
        )


def write_coco_annotation_line(
    file: TextIO,
    annotation: dict[str, Any],
) -> None:
    file.write(
        json.dumps(
            annotation,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )


def build_coco_json(
    output_path: Path,
    temporary_annotation_path: Path,
    images: list[dict[str, Any]],
) -> None:
    categories = [
        {
            "id": COCO_CLASS_TO_ID[class_name],
            "name": class_name,
            "supercategory": "road_object",
        }
        for class_name in DETECTION_CLASSES
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        output_file.write('{"info":')
        json.dump(
            {
                "description": "Official BDD100K to COCO",
                "version": "1.0",
            },
            output_file,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        output_file.write(',"licenses":[],"images":')
        json.dump(
            images,
            output_file,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        output_file.write(',"annotations":[')

        first_annotation = True

        with temporary_annotation_path.open(
            "r",
            encoding="utf-8",
        ) as annotation_file:
            for line in annotation_file:
                line = line.strip()

                if not line:
                    continue

                if not first_annotation:
                    output_file.write(",")

                output_file.write(line)
                first_annotation = False

        output_file.write('],"categories":')
        json.dump(
            categories,
            output_file,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        output_file.write("}")

    temporary_annotation_path.unlink(
        missing_ok=True,
    )


def get_record_attributes(
    record: dict[str, Any],
) -> dict[str, str]:
    attributes = record.get("attributes", {})

    if not isinstance(attributes, dict):
        attributes = {}

    return {
        "weather": str(
            attributes.get(
                "weather",
                "undefined",
            )
        ),
        "scene": str(
            attributes.get(
                "scene",
                "undefined",
            )
        ),
        "timeofday": str(
            attributes.get(
                "timeofday",
                "undefined",
            )
        ),
    }


def convert_dataset(
    args: argparse.Namespace,
) -> dict[str, Any]:
    formats = set(args.formats)
    data_root = project_path(args.data_root)
    yolo_output_root = project_path(args.yolo_output_root)
    coco_output_root = project_path(args.coco_output_root)

    label_file = resolve_label_file(
        data_root=data_root,
        label_file=args.label_file,
        split=args.split,
    )
    image_dir = resolve_image_dir(
        data_root=data_root,
        image_root=args.image_root,
        split=args.split,
    )

    records = load_official_labels(
        label_file
    )

    if args.max_images is not None:
        records = records[: args.max_images]

    if args.clean:
        clean_split_outputs(
            split=args.split,
            formats=formats,
            yolo_output_root=yolo_output_root,
            coco_output_root=coco_output_root,
        )

    if "yolo" in formats:
        yolo_image_dir = yolo_output_root / "images" / args.split
        yolo_label_dir = yolo_output_root / "labels" / args.split
        yolo_metadata_dir = yolo_output_root / "metadata"

        yolo_image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        yolo_label_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        yolo_metadata_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        metadata_file = (
            yolo_metadata_dir / f"{args.split}.jsonl"
        ).open(
            "w",
            encoding="utf-8",
        )
    else:
        yolo_image_dir = None
        yolo_label_dir = None
        metadata_file = None

    if "coco" in formats:
        coco_image_dir = coco_output_root / "images" / args.split
        coco_annotation_dir = coco_output_root / "annotations"

        coco_image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        coco_annotation_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_annotation_path = (
            coco_annotation_dir / f".{args.split}_annotations.jsonl"
        )
        temporary_annotation_file = temporary_annotation_path.open(
            "w",
            encoding="utf-8",
        )
    else:
        coco_image_dir = None
        temporary_annotation_path = None
        temporary_annotation_file = None

    coco_images: list[dict[str, Any]] = []
    total_labels = 0
    empty_images = 0
    invalid_boxes = 0
    skipped_objects = 0
    annotation_id = 1
    class_counts = {
        class_name: 0
        for class_name in DETECTION_CLASSES
    }

    try:
        for image_id, record in enumerate(
            tqdm(
                records,
                desc=f"Converting {args.split}",
            ),
            start=1,
        ):
            image_name = record.get("name")

            if not image_name:
                raise ValueError(
                    f"Missing image name in record #{image_id}"
                )

            source_image_path = resolve_image_path(
                image_dir=image_dir,
                image_name=str(image_name),
            )
            image_width, image_height = resolve_image_size(
                record=record,
                image_path=source_image_path,
                args=args,
            )
            detection_objects, current_invalid_boxes, current_skipped_objects = (
                parse_detection_objects(
                    record=record,
                    image_width=image_width,
                    image_height=image_height,
                )
            )

            invalid_boxes += current_invalid_boxes
            skipped_objects += current_skipped_objects

            if not detection_objects:
                empty_images += 1

            image_attributes = get_record_attributes(
                record
            )

            for obj in detection_objects:
                class_counts[obj["class_name"]] += 1

            total_labels += len(detection_objects)

            if "yolo" in formats:
                assert yolo_image_dir is not None
                assert yolo_label_dir is not None
                assert metadata_file is not None

                prepare_image(
                    source_path=source_image_path,
                    destination_path=yolo_image_dir / source_image_path.name,
                    mode=args.image_mode,
                )
                write_yolo_label(
                    label_path=yolo_label_dir / f"{source_image_path.stem}.txt",
                    detection_objects=detection_objects,
                    image_width=image_width,
                    image_height=image_height,
                )
                metadata_file.write(
                    json.dumps(
                        {
                            "image": source_image_path.name,
                            "split": args.split,
                            "weather": image_attributes["weather"],
                            "scene": image_attributes["scene"],
                            "timeofday": image_attributes["timeofday"],
                            "num_objects": len(detection_objects),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if "coco" in formats:
                assert coco_image_dir is not None
                assert temporary_annotation_file is not None

                prepare_image(
                    source_path=source_image_path,
                    destination_path=coco_image_dir / source_image_path.name,
                    mode=args.image_mode,
                )
                coco_images.append(
                    {
                        "id": image_id,
                        "file_name": (
                            f"images/{args.split}/{source_image_path.name}"
                        ),
                        "width": image_width,
                        "height": image_height,
                        "attributes": image_attributes,
                    }
                )

                for obj in detection_objects:
                    left, top, width, height = obj["bbox"]
                    write_coco_annotation_line(
                        file=temporary_annotation_file,
                        annotation={
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": obj["coco_category_id"],
                            "bbox": [
                                left,
                                top,
                                width,
                                height,
                            ],
                            "area": width * height,
                            "iscrowd": 0,
                            "segmentation": [],
                            "attributes": obj["attributes"],
                        },
                    )
                    annotation_id += 1
    finally:
        if metadata_file is not None:
            metadata_file.close()

        if temporary_annotation_file is not None:
            temporary_annotation_file.close()

    stats = {
        "split": args.split,
        "formats": sorted(formats),
        "label_file": str(label_file),
        "image_dir": str(image_dir),
        "images": len(records),
        "detection_objects": total_labels,
        "empty_images": empty_images,
        "invalid_boxes": invalid_boxes,
        "skipped_non_detection_objects": skipped_objects,
        "class_counts": class_counts,
    }

    if "yolo" in formats:
        write_yolo_yaml(
            yolo_output_root
        )

        stats_path = (
            yolo_output_root
            / "metadata"
            / f"stats_{args.split}.json"
        )

        with stats_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                stats,
                file,
                ensure_ascii=False,
                indent=2,
            )

    if "coco" in formats:
        assert temporary_annotation_path is not None

        coco_output_path = (
            coco_output_root
            / "annotations"
            / f"instances_{args.split}.json"
        )
        build_coco_json(
            output_path=coco_output_path,
            temporary_annotation_path=temporary_annotation_path,
            images=coco_images,
        )

        stats_path = (
            coco_output_root
            / "annotations"
            / f"stats_{args.split}.json"
        )

        with stats_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                stats,
                file,
                ensure_ascii=False,
                indent=2,
            )

    return stats


def main() -> None:
    args = parse_args()
    stats = convert_dataset(
        args
    )

    print("\n========== Official BDD100K Conversion ==========")
    print(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
