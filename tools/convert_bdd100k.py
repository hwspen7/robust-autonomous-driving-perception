"""
将Dataset Ninja Supervisely格式BDD100K同时转换为YOLO和COCO格式。

核心设计：
1. 每个Supervisely JSON只读取和解析一次；
2. 将矩形目标统一转换为内部检测框记录；
3. 同一份记录分别写入YOLO和COCO；
4. 只转换10类rectangle检测目标；
5. lane和drivable area不进入目标检测数据；
6. 保留空标注图片作为负样本；
7. 保留天气、场景、昼夜、遮挡和截断属性；
8. COCO标注使用临时JSONL流式写入，避免全量标注占用大量内存。
"""

import argparse
import ast
import json
import shutil
import sys
from pathlib import Path
from typing import Any, TextIO

import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
        description="将BDD100K Supervisely标注转换为YOLO和COCO格式。"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "datasets/downloads/bdd100k/bdd100k-images-100k"
        ),
        help="Supervisely格式BDD100K根目录。",
    )
    parser.add_argument(
        "--yolo-output-root",
        type=Path,
        default=Path("datasets/yolo/bdd100k"),
        help="YOLO格式输出目录。",
    )
    parser.add_argument(
        "--coco-output-root",
        type=Path,
        default=Path("datasets/coco/bdd100k"),
        help="COCO格式输出目录。",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="数据划分，例如train、val或test。",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("yolo", "coco"),
        default=("yolo", "coco"),
        help="需要生成的格式，默认同时生成YOLO和COCO。",
    )
    parser.add_argument(
        "--image-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="图片使用软链接或直接复制。",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="最多转换多少张图片；默认转换全部。",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="转换前清理当前split已有的输出。",
    )
    return parser.parse_args()


def has_split_dirs(
    root: Path,
    split: str,
) -> bool:
    split_root = root / split

    return (
        (split_root / "ann").exists()
        and (split_root / "img").exists()
    )


def is_split_root(
    root: Path,
) -> bool:
    return (
        (root / "ann").exists()
        and (root / "img").exists()
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


def resolve_data_root(
    data_root: Path,
    split: str,
) -> Path:
    candidates: list[Path] = [
        data_root,
    ]

    if not data_root.is_absolute():
        candidates.append(
            PROJECT_ROOT / data_root
        )

    download_roots = [
        Path("datasets/downloads/bdd100k"),
        PROJECT_ROOT
        / "datasets"
        / "downloads"
        / "bdd100k",
        data_root.parent,
    ]

    for download_root in download_roots:
        candidates.extend(
            [
                download_root
                / "bdd100k:-images-100k",
                download_root
                / "bdd100k-images-100k",
            ]
        )

        if not download_root.exists():
            continue

        candidates.extend(
            sorted(
                path
                for path in download_root.iterdir()
                if path.is_dir()
            )
        )

    searched = dedupe_paths(candidates)

    for candidate in searched:
        if has_split_dirs(candidate, split):
            return candidate

        if candidate.name == split and is_split_root(candidate):
            return candidate.parent

    raise FileNotFoundError(
        f"找不到BDD100K Supervisely数据根目录，split={split}。\n"
        "已搜索：\n"
        + "\n".join(f"  - {path}" for path in searched)
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(f"JSON顶层结构不是字典：{path}")

    return data


def parse_tag_map(
    tags: list[dict[str, Any]],
) -> dict[str, str]:
    result: dict[str, str] = {}

    for tag in tags:
        name = tag.get("name")
        value = tag.get("value")

        if name is not None and value is not None:
            result[str(name)] = str(value)

    return result


def parse_object_attributes(
    tags: list[dict[str, Any]],
) -> dict[str, Any]:
    for tag in tags:
        if tag.get("name") != "attributes":
            continue

        value = tag.get("value", {})

        if isinstance(value, dict):
            return value

        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return {}

            if isinstance(parsed, dict):
                return parsed

    return {}


def resolve_image_path(
    image_dir: Path,
    annotation_path: Path,
) -> Path:
    # foo.jpg.json删除最后的.json后得到foo.jpg。
    image_name = annotation_path.name.removesuffix(".json")
    image_path = image_dir / image_name

    if image_path.exists():
        return image_path

    # 兼容标注文件名没有保留图片扩展名的情况。
    candidates = [
        path
        for path in image_dir.glob(f"{annotation_path.stem}.*")
        if path.is_file()
    ]

    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        f"找不到标注对应图片：{annotation_path}"
    )


def parse_bbox(
    obj: dict[str, Any],
    image_width: int,
    image_height: int,
) -> list[float] | None:
    # Supervisely矩形框：
    # exterior[0]是一个角点，exterior[1]是另一个角点。
    exterior = (
        obj.get("points", {})
        .get("exterior", [])
    )

    if len(exterior) < 2:
        return None

    try:
        x1, y1 = map(float, exterior[0])
        x2, y2 = map(float, exterior[1])
    except (TypeError, ValueError):
        return None

    left = max(0.0, min(x1, x2))
    top = max(0.0, min(y1, y2))
    right = min(float(image_width), max(x1, x2))
    bottom = min(float(image_height), max(y1, y2))

    box_width = right - left
    box_height = bottom - top

    if box_width <= 0 or box_height <= 0:
        return None

    # COCO格式：[left, top, width, height]。
    return [
        left,
        top,
        box_width,
        box_height,
    ]


def coco_bbox_to_yolo(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    left, top, box_width, box_height = bbox

    center_x = left + box_width / 2
    center_y = top + box_height / 2

    return (
        center_x / image_width,
        center_y / image_height,
        box_width / image_width,
        box_height / image_height,
    )


def parse_detection_objects(
    data: dict[str, Any],
    image_width: int,
    image_height: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """
    将一张图片中的Supervisely对象转换为统一检测目标。

    返回：
    - detection_objects：有效检测目标；
    - invalid_boxes：无效矩形框数量；
    - skipped_objects：非检测目标数量。
    """
    detection_objects: list[dict[str, Any]] = []
    invalid_boxes = 0
    skipped_objects = 0

    for obj in data.get("objects", []):
        if obj.get("geometryType") != "rectangle":
            skipped_objects += 1
            continue

        source_class = str(
            obj.get("classTitle", "")
        )

        canonical_class = SOURCE_TO_CANONICAL.get(
            source_class
        )

        if canonical_class is None:
            skipped_objects += 1
            continue

        bbox = parse_bbox(
            obj=obj,
            image_width=image_width,
            image_height=image_height,
        )

        if bbox is None:
            invalid_boxes += 1
            continue

        attributes = parse_object_attributes(
            obj.get("tags", [])
        )

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

    return (
        detection_objects,
        invalid_boxes,
        skipped_objects,
    )


def prepare_image(
    source_path: Path,
    destination_path: Path,
    mode: str,
) -> None:
    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        destination_path.exists()
        or destination_path.is_symlink()
    ):
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
    args: argparse.Namespace,
    formats: set[str],
) -> None:
    if not args.clean:
        return

    if "yolo" in formats:
        for path in (
            args.yolo_output_root
            / "images"
            / args.split,
            args.yolo_output_root
            / "labels"
            / args.split,
        ):
            if path.exists() or path.is_symlink():
                shutil.rmtree(path)

        for path in (
            args.yolo_output_root
            / "metadata"
            / f"{args.split}.jsonl",
            args.yolo_output_root
            / "metadata"
            / f"stats_{args.split}.json",
        ):
            path.unlink(missing_ok=True)

    if "coco" in formats:
        image_dir = (
            args.coco_output_root
            / "images"
            / args.split
        )

        if image_dir.exists() or image_dir.is_symlink():
            shutil.rmtree(image_dir)

        annotation_dir = (
            args.coco_output_root
            / "annotations"
        )

        for path in (
            annotation_dir
            / f"instances_{args.split}.json",
            annotation_dir
            / f"stats_{args.split}.json",
            annotation_dir
            / f".{args.split}_annotations.jsonl",
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
        center_x, center_y, width, height = (
            coco_bbox_to_yolo(
                bbox=obj["bbox"],
                image_width=image_width,
                image_height=image_height,
            )
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

    # 空文件表示该图片没有目标，是合法的YOLO负样本。
    label_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_yolo_yaml(
    output_root: Path,
) -> None:
    if (
        output_root
        / "images"
        / "val"
    ).exists():
        val_path = "images/val"
    elif (
        output_root
        / "images"
        / "test"
    ).exists():
        val_path = "images/test"
    else:
        # 只转换train进行测试时暂时指向train；
        # 转换val后脚本会自动更新。
        val_path = "images/train"

    yaml_data = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": val_path,
        "test": "images/test",
        "names": {
            class_id: class_name
            for class_id, class_name
            in enumerate(DETECTION_CLASSES)
        },
    }

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        output_root
        / "data.yaml"
    ).open(
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
    """
    将临时JSONL标注流式合并为正式COCO JSON。

    图片列表只有约7万项，可以保存在内存中；
    检测框超过百万项，使用临时文件避免占用过多内存。
    """
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
                "description": (
                    "BDD100K Dataset Ninja "
                    "Supervisely to COCO"
                ),
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
        missing_ok=True
    )


def convert_dataset(
    args: argparse.Namespace,
) -> dict[str, Any]:
    formats = set(args.formats)
    data_root = resolve_data_root(
        args.data_root,
        args.split,
    )

    source_split_root = (
        data_root
        / args.split
    )
    annotation_dir = (
        source_split_root
        / "ann"
    )
    image_dir = (
        source_split_root
        / "img"
    )

    if not annotation_dir.exists():
        raise FileNotFoundError(
            f"标注目录不存在：{annotation_dir}"
        )

    if not image_dir.exists():
        raise FileNotFoundError(
            f"图片目录不存在：{image_dir}"
        )

    annotation_files = sorted(
        annotation_dir.glob("*.json")
    )

    if args.max_images is not None:
        annotation_files = annotation_files[
            :args.max_images
        ]

    if not annotation_files:
        raise RuntimeError(
            f"没有找到标注文件：{annotation_dir}"
        )

    clean_split_outputs(
        args=args,
        formats=formats,
    )

    if "yolo" in formats:
        yolo_image_dir = (
            args.yolo_output_root
            / "images"
            / args.split
        )
        yolo_label_dir = (
            args.yolo_output_root
            / "labels"
            / args.split
        )
        yolo_metadata_dir = (
            args.yolo_output_root
            / "metadata"
        )

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

        metadata_path = (
            yolo_metadata_dir
            / f"{args.split}.jsonl"
        )
        metadata_file = metadata_path.open(
            "w",
            encoding="utf-8",
        )
    else:
        yolo_image_dir = None
        yolo_label_dir = None
        metadata_file = None

    if "coco" in formats:
        coco_image_dir = (
            args.coco_output_root
            / "images"
            / args.split
        )
        coco_annotation_dir = (
            args.coco_output_root
            / "annotations"
        )

        coco_image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        coco_annotation_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_annotation_path = (
            coco_annotation_dir
            / f".{args.split}_annotations.jsonl"
        )
        temporary_annotation_file = (
            temporary_annotation_path.open(
                "w",
                encoding="utf-8",
            )
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
        for image_id, annotation_path in enumerate(
            tqdm(
                annotation_files,
                desc=f"Converting {args.split}",
            ),
            start=1,
        ):
            data = load_json(
                annotation_path
            )

            source_image_path = resolve_image_path(
                image_dir=image_dir,
                annotation_path=annotation_path,
            )

            size = data.get("size", {})
            image_width = int(
                size.get("width", 0)
            )
            image_height = int(
                size.get("height", 0)
            )

            if image_width <= 0 or image_height <= 0:
                raise ValueError(
                    f"图片尺寸无效：{annotation_path}"
                )

            (
                detection_objects,
                current_invalid_boxes,
                current_skipped_objects,
            ) = parse_detection_objects(
                data=data,
                image_width=image_width,
                image_height=image_height,
            )

            invalid_boxes += current_invalid_boxes
            skipped_objects += current_skipped_objects

            if not detection_objects:
                empty_images += 1

            image_tags = parse_tag_map(
                data.get("tags", [])
            )

            for obj in detection_objects:
                class_counts[
                    obj["class_name"]
                ] += 1

            total_labels += len(
                detection_objects
            )

            if "yolo" in formats:
                assert yolo_image_dir is not None
                assert yolo_label_dir is not None
                assert metadata_file is not None

                prepare_image(
                    source_path=source_image_path,
                    destination_path=(
                        yolo_image_dir
                        / source_image_path.name
                    ),
                    mode=args.image_mode,
                )

                write_yolo_label(
                    label_path=(
                        yolo_label_dir
                        / f"{source_image_path.stem}.txt"
                    ),
                    detection_objects=detection_objects,
                    image_width=image_width,
                    image_height=image_height,
                )

                metadata_file.write(
                    json.dumps(
                        {
                            "image": source_image_path.name,
                            "split": args.split,
                            "weather": image_tags.get(
                                "weather",
                                "undefined",
                            ),
                            "scene": image_tags.get(
                                "scene",
                                "undefined",
                            ),
                            "timeofday": image_tags.get(
                                "timeofday",
                                "undefined",
                            ),
                            "num_objects": len(
                                detection_objects
                            ),
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
                    destination_path=(
                        coco_image_dir
                        / source_image_path.name
                    ),
                    mode=args.image_mode,
                )

                coco_images.append(
                    {
                        "id": image_id,
                        "file_name": (
                            f"images/{args.split}/"
                            f"{source_image_path.name}"
                        ),
                        "width": image_width,
                        "height": image_height,
                        "attributes": {
                            "weather": image_tags.get(
                                "weather",
                                "undefined",
                            ),
                            "scene": image_tags.get(
                                "scene",
                                "undefined",
                            ),
                            "timeofday": image_tags.get(
                                "timeofday",
                                "undefined",
                            ),
                        },
                    }
                )

                for obj in detection_objects:
                    left, top, width, height = (
                        obj["bbox"]
                    )

                    write_coco_annotation_line(
                        file=temporary_annotation_file,
                        annotation={
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": obj[
                                "coco_category_id"
                            ],
                            "bbox": [
                                left,
                                top,
                                width,
                                height,
                            ],
                            "area": width * height,
                            "iscrowd": 0,
                            "segmentation": [],
                            "attributes": obj[
                                "attributes"
                            ],
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
        "images": len(annotation_files),
        "detection_objects": total_labels,
        "empty_images": empty_images,
        "invalid_boxes": invalid_boxes,
        "skipped_non_detection_objects": (
            skipped_objects
        ),
        "class_counts": class_counts,
    }

    if "yolo" in formats:
        write_yolo_yaml(
            args.yolo_output_root
        )

        stats_path = (
            args.yolo_output_root
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
            args.coco_output_root
            / "annotations"
            / f"instances_{args.split}.json"
        )

        build_coco_json(
            output_path=coco_output_path,
            temporary_annotation_path=(
                temporary_annotation_path
            ),
            images=coco_images,
        )

        stats_path = (
            args.coco_output_root
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

    print(
        "\n========== BDD100K Conversion =========="
    )
    print(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        )
    )

    if "yolo" in args.formats:
        print(
            "YOLO output：",
            args.yolo_output_root.resolve(),
        )

    if "coco" in args.formats:
        print(
            "COCO output：",
            args.coco_output_root.resolve(),
        )


if __name__ == "__main__":
    main()
