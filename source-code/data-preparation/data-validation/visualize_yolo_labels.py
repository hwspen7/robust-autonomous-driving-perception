














import argparse
import random
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize BDD100K annotations in YOLO format."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/yolo/bdd100k"),
        help="Root directory of the YOLO-format dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split, for example train or val.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=20,
        help="Number of randomly selected images to visualize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/data_check/yolo"),
        help="Output directory for visualizations.",
    )
    return parser.parse_args()

def load_class_names(
        yaml_path: Path
) -> dict[int, str]:
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"YOLO configuration file not found"
        )

    with open(yaml_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    names = data.get("names")

    if isinstance(names, list):
        return {
            class_id: str(class_name) for class_id, class_name in enumerate(names)
        }

    if isinstance(names, dict):
        return {
            int(class_id): str(class_name) for class_id, class_name in names.items()
        }

    raise ValueError(
        f"names' format in data.yaml is invalid: {yaml_path}"
    )

def load_yolo_labels(
        label_path: Path,
) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []

    labels = []

    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()

        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"{label_path} does not have 5 parts"
            )

        class_id = int(parts[0])
        center_x, center_y, width, height = map(float, parts[1:])
        values = (
            center_x,
            center_y,
            width,
            height,
        )

        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(
                f"{label_path} does not have all values between 0 and 1"
            )
        if width <= 0 or height <= 0:
            raise ValueError(
                f"{label_path} does not have all positive values"
            )

        labels.append(
            (
                class_id,
                center_x,
                center_y,
                width,
                height,
            )
        )
    return labels

def yolo_to_xyxy(
        center_x: float,
        center_y: float,
        box_width: float,
        box_height: float,
        image_width: int,
        image_height: int,
) -> tuple[float, float, float, float]:


    center_x *= image_width
    center_y *= image_height
    box_width *= image_width
    box_height *= image_height

    x1 = center_x - box_width / 2
    y1 = center_y - box_height / 2
    x2 = center_x + box_width / 2
    y2 = center_y + box_height / 2

    return (
        max(0.0, x1),
        max(0.0, y1),
        min(float(image_width - 1), x2),
        min(float(image_height - 1), y2),
    )

def draw_label(
        draw: ImageDraw.ImageDraw,
        box: tuple[float, float, float, float],
        text: str,
        font: ImageFont.ImageFont,
) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle(
        (x1, y1, x2, y2),
        outline="red",
        width=3
    )

    text_bbox = draw.textbbox(
        (x1, y1),
        text,
        font=font,
    )

    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    text_y = max(0.0, y1 - text_height - 6)

    draw.rectangle(
        (
            x1,
            text_y,
            x1 + text_width + 8,
            text_y + text_height + 6,
        ),
        fill="black"
    )

    draw.text(
        (
            x1 + 4,
            text_y + 3
        ),
        text,
        fill="white",
        font=font,
    )

def main() -> None:
    args = parse_args()

    image_dir = args.dataset_root / "images" / args.split
    label_dir = args.dataset_root / "labels" / args.split
    yaml_path = args.dataset_root / "data.yaml"

    if not image_dir.exists():
        raise FileNotFoundError(
            f"The image catalog does not exist"
        )

    if not label_dir.exists():
        raise FileNotFoundError(
            f"The label catalog does not exist"
        )

    class_names = load_class_names(yaml_path)
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(
            f"The image catalog contains no image files"
        )

    rng = random.Random(args.seed)

    selected_paths = rng.sample(
        image_paths,
        min(args.num_images, len(image_paths)),
    )

    output_dir = args.output_dir / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    font = ImageFont.load_default()

    total_boxes = 0
    empty_images = 0
    for image_path in selected_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        labels = load_yolo_labels(label_path)
        total_boxes += len(labels)

        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")

        image_width, image_height = image.size
        draw = ImageDraw.Draw(image)

        if not labels:
            empty_images += 1
            draw.rectangle(
                (0, 0, 190, 28),
                fill="black",
            )
            draw.text(
                (6, 7),
                "No detection objects",
                fill="white",
                font=font,
            )

        for (class_id, center_x, center_y, width, height) in labels:
            if class_id not in class_names:
                raise ValueError(
                    f"Unknown class id: {class_id}"
                )

            box = yolo_to_xyxy(
                center_x,
                center_y,
                width,
                height,
                image_width,
                image_height,
            )

            class_name = class_names[class_id]
            label_text = (
                f"{class_id} "
                f"{class_name}"
            )
            draw_label(
                draw,
                box,
                label_text,
                font,
            )

        output_path = output_dir / image_path.name
        image.save(output_path, quality=95)

    print("\n========== YOLO Label Visualization ==========")
    print(f"Split：{args.split}")
    print(f"Images：{len(selected_paths)}")
    print(f"Boxes：{total_boxes}")
    print(f"Empty images：{empty_images}")
    print(f"Output：{output_dir.resolve()}")

if __name__ == '__main__':
    main()
