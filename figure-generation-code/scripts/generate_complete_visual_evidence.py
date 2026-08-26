from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
import generate_publication_figures as base


PROJECT = base.PROJECT
OUT = base.OUT
TABLES = base.TABLES

INK = base.INK
MUTED = base.MUTED
GRID = base.GRID
BG = base.BG
BLUE = base.BLUE
NAVY = base.NAVY
ORANGE = base.ORANGE
RED = base.RED
GREEN = base.GREEN
PURPLE = base.PURPLE
PALE_BLUE = base.PALE_BLUE
PALE_ORANGE = base.PALE_ORANGE
PALE_GREEN = base.PALE_GREEN
PALE_RED = base.PALE_RED

text = base.text
multiline = base.multiline
rect = base.rect
line = base.line
circle = base.circle
raster_panel = base.raster_panel
title_block = base.title_block
save_figure = base.save_figure

VAL_GT = PROJECT / "data-and-baselines/dataset/annotations/instances_val.json"
TRAIN_GT = PROJECT / "data-and-baselines/dataset/annotations/instances_train.json"
FAILURE_TABLE = PROJECT / "final-testing/failure-subsets/failure_object_table.csv"
CASE_SELECTION = PROJECT / "final-testing/qualitative-cases/case_selection.json"
YOLO_PREDICTIONS = PROJECT / "failure-diagnosis/predictions/yolo11m/predictions.json"
DFINE_PREDICTIONS = PROJECT / "failure-diagnosis/predictions/dfine-m/predictions.json"
YOLO_SELECTED_CACHE = Path("/tmp/publication-yolo-selected.json")
DFINE_SELECTED_CACHE = Path("/tmp/publication-dfine-selected.json")

CATEGORY_COLORS = {
    1: "#2D5478",
    2: "#625878",
    3: "#3E695B",
    4: "#9B5C32",
    5: "#7B4F37",
    6: "#565656",
    7: "#7A6A36",
    8: "#416C74",
    9: "#913F3F",
    10: "#5E6274",
}

DATASET_OVERVIEW_IDS = [7849, 1477, 9339, 3962, 2439, 4844]
SCALE_ANNOTATION_IDS = [289, 385, 163, 2, 1059, 16350]
FAILURE_CASES = [
    ("Small jointly missed", 6664, 367),
    ("D-FINE-supported YOLO failure", 16384, 896),
    ("YOLO low confidence", 15726, 863),
    ("YOLO localization failure", 37936, 2043),
    ("YOLO classification failure", 4927, 274),
    ("Small high risk", 4222, 236),
]


def load_coco(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in data["images"]}
    annotations = {int(row["id"]): row for row in data["annotations"]}
    by_image = defaultdict(list)
    for row in data["annotations"]:
        by_image[int(row["image_id"])].append(row)
    categories = {int(row["id"]): row["name"] for row in data["categories"]}
    return images, annotations, by_image, categories


VAL_IMAGES, VAL_ANNOTATIONS, VAL_BY_IMAGE, CATEGORIES = load_coco(VAL_GT)
TRAIN_IMAGES, TRAIN_ANNOTATIONS, TRAIN_BY_IMAGE, _ = load_coco(TRAIN_GT)


def image_path(record: dict) -> Path:
    return base.dataset_image(record["file_name"])


def pil_color(value: str):
    return value


def draw_box(draw: ImageDraw.ImageDraw, bbox, color, width=4):
    x, y, w, h = [float(v) for v in bbox]
    draw.rectangle((x, y, x + w, y + h), outline=pil_color(color), width=max(1, int(width)))


def draw_gt_scene(image_id: int, *, emphasize_annotation_id: int | None = None, all_boxes=True):
    image = Image.open(image_path(VAL_IMAGES[image_id])).convert("RGB")
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for annotation in VAL_BY_IMAGE[image_id]:
        if not all_boxes and int(annotation["id"]) != emphasize_annotation_id:
            continue
        color = CATEGORY_COLORS[int(annotation["category_id"])]
        width = 7 if int(annotation["id"]) == emphasize_annotation_id else 2
        draw_box(draw, annotation["bbox"], color, width)
    return output


def prediction_cache(path: Path):
    records = json.loads(path.read_text(encoding="utf-8"))
    by_image = defaultdict(list)
    for row in records:
        by_image[int(row["image_id"])].append(row)
    return by_image


def ensure_prediction_caches():
    missing = [p for p in (YOLO_SELECTED_CACHE, DFINE_SELECTED_CACHE) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Selected prediction caches are missing. Extract the six frozen image IDs before generating figures."
        )


ensure_prediction_caches()
YOLO_BY_IMAGE = prediction_cache(YOLO_SELECTED_CACHE)
DFINE_BY_IMAGE = prediction_cache(DFINE_SELECTED_CACHE)


def draw_prediction_scene(
    image_id: int,
    predictions: list[dict],
    threshold: float,
    *,
    target_annotation_id: int | None = None,
    max_boxes: int = 40,
):
    image = Image.open(image_path(VAL_IMAGES[image_id])).convert("RGB")
    output = image.copy()
    draw = ImageDraw.Draw(output)
    kept = sorted(
        [row for row in predictions if float(row["score"]) >= threshold],
        key=lambda row: float(row["score"]),
        reverse=True,
    )[:max_boxes]
    for row in kept:
        draw_box(draw, row["bbox"], CATEGORY_COLORS[int(row["category_id"])], 3)
    if target_annotation_id is not None:
        draw_box(draw, VAL_ANNOTATIONS[target_annotation_id]["bbox"], "#FFFFFF", 8)
        draw_box(draw, VAL_ANNOTATIONS[target_annotation_id]["bbox"], NAVY, 4)
    return output, len(kept)


def bbox_intersection(a, b):
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def target_context_geometry(image: Image.Image, bbox, *, width_fraction=0.38):
    crop, geometry = base.context_crop(image, bbox, width_fraction=width_fraction)
    return crop, geometry


def draw_target_crop(
    image_id: int,
    annotation_id: int,
    *,
    predictions: list[dict] | None = None,
    threshold: float = 0.0,
    width_fraction=0.38,
):
    image = Image.open(image_path(VAL_IMAGES[image_id])).convert("RGB")
    annotation = VAL_ANNOTATIONS[annotation_id]
    crop, geometry = target_context_geometry(image, annotation["bbox"], width_fraction=width_fraction)
    gx, gy, gw, gh = geometry
    output = crop.copy()
    draw = ImageDraw.Draw(output)
    if predictions is None:
        x, y, w, h = annotation["bbox"]
        draw_box(draw, (x - gx, y - gy, w, h), CATEGORY_COLORS[int(annotation["category_id"])], 6)
    else:
        crop_bbox = (gx, gy, gw, gh)
        kept = sorted(
            [
                row
                for row in predictions
                if float(row["score"]) >= threshold and bbox_intersection(row["bbox"], crop_bbox) > 0
            ],
            key=lambda row: float(row["score"]),
            reverse=True,
        )[:20]
        for row in kept:
            x, y, w, h = row["bbox"]
            draw_box(draw, (x - gx, y - gy, w, h), CATEGORY_COLORS[int(row["category_id"])], 4)
        x, y, w, h = annotation["bbox"]
        draw_box(draw, (x - gx, y - gy, w, h), "#FFFFFF", 8)
        draw_box(draw, (x - gx, y - gy, w, h), NAVY, 4)
    return output


def figure_dataset_annotation_overview():
    width, height = 2300, 1500
    body = title_block(
        "BDD100K Scene Diversity and Annotation Density",
        "Six deterministic official-validation scenes spanning illumination, weather, and road context",
        width,
    )
    panel_w, panel_h = 650, 366
    xs = [100, 825, 1550]
    ys = [220, 725]
    source_paths = [VAL_GT]
    for index, image_id in enumerate(DATASET_OVERVIEW_IDS):
        record = VAL_IMAGES[image_id]
        source_paths.append(image_path(record))
        rendered = draw_gt_scene(image_id)
        x, y = xs[index % 3], ys[index // 3]
        attributes = record["attributes"]
        annotations = VAL_BY_IMAGE[image_id]
        small = sum(float(row["area"]) < 32**2 for row in annotations)
        body.append(text(x, y - 24, f"({chr(97 + index)}) {attributes['timeofday'].title()} · {attributes['weather'].title()}", 21, 500))
        body.append(raster_panel(x, y, panel_w, panel_h, rendered, fit="cover"))
        body.append(rect(x, y, panel_w, panel_h, "none", INK, 1))
        body.append(text(x, y + panel_h + 28, f"{attributes['scene']} · {len(annotations)} objects · {small} small", 17, fill=MUTED))

    body.append(line(100, 1240, 2200, 1240, GRID, 1.2))
    body.append(text(100, 1285, "Class colors", 18, 500))
    for index, category_id in enumerate(sorted(CATEGORIES)):
        x = 100 + (index % 5) * 420
        y = 1330 + (index // 5) * 55
        body.append(rect(x, y, 22, 22, "none", CATEGORY_COLORS[category_id], 3))
        body.append(text(x + 36, y + 19, CATEGORIES[category_id], 17))
    return save_figure(
        "figure-13-dataset-annotation-overview",
        width,
        height,
        "BDD100K scene and annotation overview",
        "Representative raw validation scenes with ground-truth boxes across daytime, night, rain, snow, highway, residential, and city-street conditions.",
        body,
        source_paths,
    )


def figure_object_scale_gallery():
    width, height = 2300, 1260
    body = title_block(
        "What Small, Medium, and Large Objects Look Like in the Native Scene",
        "Target-centered views preserve the surrounding scene while making the annotated object inspectable",
        width,
    )
    panel_w, panel_h = 660, 285
    xs = [100, 820, 1540]
    ys = [220, 710]
    scale_labels = ["Small", "Small", "Medium", "Medium", "Large", "Large"]
    sources = [VAL_GT]
    for index, annotation_id in enumerate(SCALE_ANNOTATION_IDS):
        annotation = VAL_ANNOTATIONS[annotation_id]
        image_id = int(annotation["image_id"])
        record = VAL_IMAGES[image_id]
        sources.append(image_path(record))
        source = Image.open(image_path(record)).convert("RGB")
        context, geometry = base.context_crop(source, annotation["bbox"], width_fraction=0.46)
        gx, gy, _, _ = geometry
        target_bbox = annotation["bbox"]
        context = base.draw_bbox(
            context,
            (target_bbox[0] - gx, target_bbox[1] - gy, target_bbox[2], target_bbox[3]),
            color=CATEGORY_COLORS[int(annotation["category_id"])],
            width=6,
        )
        tx, ty, tw, th = [float(v) for v in target_bbox]
        pad = max(8, int(max(tw, th) * 0.35))
        target = source.crop((max(0, tx - pad), max(0, ty - pad), min(source.width, tx + tw + pad), min(source.height, ty + th + pad)))
        target = ImageOps.expand(target, border=4, fill=CATEGORY_COLORS[int(annotation["category_id"])])
        x, y = xs[index % 3], ys[index // 3]
        body.append(text(x, y - 28, f"({chr(97 + index)}) {scale_labels[index]} · {CATEGORIES[int(annotation['category_id'])]}", 21, 500))
        body.append(raster_panel(x, y, 480, panel_h, context, fit="cover"))
        body.append(rect(x, y, 480, panel_h, "none", INK, 1))
        body.append(raster_panel(x + 500, y, 160, panel_h, target, fit="contain"))
        body.append(rect(x + 500, y, 160, panel_h, "none", GRID, 1))
        short_side = min(float(tw), float(th))
        area_fraction = float(annotation["area"]) / (record["width"] * record["height"])
        body.append(text(x, y + panel_h + 30, f"box = {int(tw)} × {int(th)} px · short side = {short_side:.0f} px", 17, fill=MUTED))
        body.append(text(x, y + panel_h + 57, f"image-area fraction = {area_fraction:.4%}", 16, fill=MUTED))
    body.append(line(100, 1150, 2200, 1150, GRID, 1.2))
    body.append(text(1150, 1200, "Small-object difficulty is a native-scene property, not merely a low crop resolution.", 21, 500, anchor="middle"))
    return save_figure(
        "figure-14-native-object-scale-gallery",
        width,
        height,
        "Native object-scale gallery",
        "Six real objects illustrate the visual meaning of COCO small, medium, and large scales in the unmodified validation scene.",
        body,
        sources,
    )


def figure_baseline_detector_comparison():
    width, height = 2300, 1390
    body = title_block(
        "Baseline Detector Outputs on the Same Failure-Bearing Scenes",
        "White and navy rectangles identify the frozen target object in prediction panels",
        width,
    )
    selected = FAILURE_CASES[:3]
    columns = [("Ground truth", 100), ("YOLO11m", 835), ("D-FINE-M", 1570)]
    for title_value, x in columns:
        body.append(text(x, 190, title_value, 22, 500))
    panel_w, panel_h = 640, 360
    sources = [VAL_GT, YOLO_PREDICTIONS, DFINE_PREDICTIONS, FAILURE_TABLE]
    failure_df = pd.read_csv(FAILURE_TABLE).set_index("annotation_id")
    for row_index, (family, annotation_id, image_id) in enumerate(selected):
        y = 235 + row_index * 385
        gt = draw_gt_scene(image_id, emphasize_annotation_id=annotation_id, all_boxes=True)
        yolo, yolo_count = draw_prediction_scene(
            image_id, YOLO_BY_IMAGE[image_id], 0.2252714485, target_annotation_id=annotation_id
        )
        dfine, dfine_count = draw_prediction_scene(
            image_id, DFINE_BY_IMAGE[image_id], 0.4963708520, target_annotation_id=annotation_id
        )
        for image, x in zip((gt, yolo, dfine), (100, 835, 1570)):
            body.append(raster_panel(x, y, panel_w, panel_h, image, fit="cover"))
            body.append(rect(x, y, panel_w, panel_h, "none", INK, 1))
        info = failure_df.loc[annotation_id]
        body.append(text(100, y + panel_h + 25, f"{family} · {info.class_name} · {info.scale}", 17, 500))
        body.append(text(835, y + panel_h + 25, f"status: {info.yolo_status} · displayed boxes: {yolo_count}", 16, fill=MUTED))
        body.append(text(1570, y + panel_h + 25, f"status: {info.dfine_status} · displayed boxes: {dfine_count}", 16, fill=MUTED))
        sources.append(image_path(VAL_IMAGES[image_id]))
    return save_figure(
        "figure-15-baseline-detector-comparison",
        width,
        height,
        "Baseline detector comparison",
        "Ground truth, YOLO11m, and D-FINE-M outputs on three frozen scenes representing jointly missed, teacher-supported, and low-confidence failures.",
        body,
        sources,
    )


def figure_failure_taxonomy_cases():
    width, height = 2400, 1620
    body = title_block(
        "Six Object-Level Failure Types in Real Validation Images",
        "Each case is shown as ground truth, YOLO11m output, and D-FINE-M output around the same target",
        width,
    )
    failure_df = pd.read_csv(FAILURE_TABLE).set_index("annotation_id")
    panel_w, panel_h = 320, 180
    block_w = 1080
    sources = [VAL_GT, YOLO_PREDICTIONS, DFINE_PREDICTIONS, FAILURE_TABLE]
    for index, (family, annotation_id, image_id) in enumerate(FAILURE_CASES):
        col, row = index % 2, index // 2
        ox, oy = 90 + col * 1160, 210 + row * 455
        info = failure_df.loc[annotation_id]
        body.append(text(ox, oy, f"({chr(97 + index)}) {family}", 21, 500))
        body.append(text(ox + block_w, oy, f"risk {float(info.risk_score):.3f}", 17, anchor="end", fill=MUTED))
        panels = [
            ("Ground truth", draw_target_crop(image_id, annotation_id)),
            ("YOLO11m", draw_target_crop(image_id, annotation_id, predictions=YOLO_BY_IMAGE[image_id], threshold=0.2252714485)),
            ("D-FINE-M", draw_target_crop(image_id, annotation_id, predictions=DFINE_BY_IMAGE[image_id], threshold=0.4963708520)),
        ]
        for panel_index, (label, image) in enumerate(panels):
            x = ox + panel_index * 355
            body.append(text(x, oy + 32, label, 16, 500))
            body.append(raster_panel(x, oy + 48, panel_w, panel_h, image, fit="cover"))
            body.append(rect(x, oy + 48, panel_w, panel_h, "none", INK, 1))
        body.append(text(ox, oy + 258, f"target: {info.class_name} · {info.scale} · {info.timeofday} · {info.weather}", 16, fill=MUTED))
        body.append(text(ox, oy + 288, f"YOLO: {info.yolo_status}   D-FINE: {info.dfine_status}", 16, fill=MUTED))
        body.append(line(ox, oy + 315, ox + block_w, oy + 315, GRID, 1))
        sources.append(image_path(VAL_IMAGES[image_id]))
    return save_figure(
        "figure-16-failure-taxonomy-cases",
        width,
        height,
        "Qualitative failure taxonomy",
        "A six-case visual taxonomy of jointly missed, D-FINE-supported, low-confidence, localization, classification, and high-risk small-object failures.",
        body,
        sources,
    )


def crop_geometry_from_case(case: dict, annotation: dict):
    details = case["scale_context"]
    tx, ty, tw, th = [float(v) for v in annotation["bbox"]]
    x1 = int(round(tx - float(details["left_context_margin_px"])))
    y1 = int(round(ty - float(details["top_context_margin_px"])))
    x2 = int(round(tx + tw + float(details["right_context_margin_px"])))
    y2 = int(round(ty + th + float(details["bottom_context_margin_px"])))
    return x1, y1, x2, y2


def draw_crop_case(case: dict, *, annotate_source=True, annotate_crop=True):
    annotation = TRAIN_ANNOTATIONS[int(case["annotation_id"])]
    image_id = int(case["image_id"])
    source = Image.open(image_path(TRAIN_IMAGES[image_id])).convert("RGB")
    x1, y1, x2, y2 = crop_geometry_from_case(case, annotation)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(source.width, x2), min(source.height, y2)
    crop_box = (x1, y1, x2, y2)

    source_rendered = source.copy()
    if annotate_source:
        draw = ImageDraw.Draw(source_rendered)
        draw.rectangle(crop_box, outline=ORANGE, width=6)
        draw_box(draw, annotation["bbox"], BLUE, 7)

    crop = source.crop(crop_box)
    if annotate_crop:
        draw = ImageDraw.Draw(crop)
        for row in TRAIN_BY_IMAGE[image_id]:
            if bbox_intersection(row["bbox"], (x1, y1, x2 - x1, y2 - y1)) <= 0:
                continue
            rx, ry, rw, rh = [float(v) for v in row["bbox"]]
            clipped_x1 = max(rx, x1) - x1
            clipped_y1 = max(ry, y1) - y1
            clipped_x2 = min(rx + rw, x2) - x1
            clipped_y2 = min(ry + rh, y2) - y1
            draw_box(
                draw,
                (clipped_x1, clipped_y1, clipped_x2 - clipped_x1, clipped_y2 - clipped_y1),
                CATEGORY_COLORS[int(row["category_id"])],
                3,
            )
        rx, ry, rw, rh = [float(v) for v in annotation["bbox"]]
        draw_box(draw, (rx - x1, ry - y1, rw, rh), BLUE, 7)
    return source, source_rendered, crop, crop_box


def figure_crop_materialization_pipeline():
    selection = json.loads(CASE_SELECTION.read_text(encoding="utf-8"))
    case = selection["families"]["crop_scale_condition_shift"][0]
    annotation = TRAIN_ANNOTATIONS[int(case["annotation_id"])]
    source, source_rendered, crop, crop_box = draw_crop_case(case)
    details = case["scale_context"]
    target_only = source.copy()
    draw_box(ImageDraw.Draw(target_only), annotation["bbox"], BLUE, 7)
    crop_boundary = source.copy()
    draw = ImageDraw.Draw(crop_boundary)
    draw.rectangle(crop_box, outline=ORANGE, width=7)
    draw_box(draw, annotation["bbox"], BLUE, 7)
    network_input = crop.resize((960, 540), Image.Resampling.LANCZOS)

    width, height = 2500, 1180
    body = title_block(
        "Formal Object-Centric View Materialization",
        "One frozen training example traced from the native scene to the resized network input",
        width,
    )
    panels = [
        ("1  Raw source", source, "unmodified 1280 × 720 scene"),
        ("2  Target annotation", target_only, "native target short side: 25 px"),
        ("3  Context window", crop_boundary, "608 × 342 retained window"),
        ("4  Transformed labels", crop, "7 supervised objects remain"),
        ("5  Network input", network_input, "target short side at input: 39.47 px"),
    ]
    panel_w, panel_h = 430, 242
    for index, (label, image, detail) in enumerate(panels):
        x = 70 + index * 485
        body.append(text(x, 210, label, 20, 500))
        body.append(raster_panel(x, 240, panel_w, panel_h, image, fit="cover"))
        body.append(rect(x, 240, panel_w, panel_h, "none", INK, 1))
        body.append(text(x, 513, detail, 16, fill=MUTED))
        if index < len(panels) - 1:
            body.append(line(x + 440, 360, x + 472, 360, INK, 1.6, marker="arrow"))

    body.append(line(70, 595, 2430, 595, GRID, 1.2))
    metrics = [
        ("Target amplification", f"{float(details['short_side_amplification']):.3f}×"),
        ("Scene area retained", f"{float(details['crop_area_over_source_area']):.2%}"),
        ("Source objects", str(int(float(details["source_object_count"])))),
        ("Crop objects", str(int(float(details["crop_object_count"])))),
        ("Neighbors retained", f"{float(details['neighbor_retention_fraction']):.1%}"),
        ("Objects fully removed", str(int(float(details["fully_removed_objects"])))),
        ("Boundary-clipped objects", str(int(float(details["boundary_clipped_objects"])))),
    ]
    for index, (label, value) in enumerate(metrics):
        x = 90 + (index % 4) * 590
        y = 675 + (index // 4) * 150
        body.append(text(x, y, label, 17, fill=MUTED))
        body.append(text(x, y + 45, value, 28, 500, fill=NAVY if index < 5 else ORANGE))
        body.append(line(x, y + 72, x + 500, y + 72, GRID, 1))
    body.append(text(1250, 1085, "The crop changes scale, context, object count, and label geometry simultaneously.", 22, 500, anchor="middle"))
    return save_figure(
        "figure-17-crop-materialization-pipeline",
        width,
        height,
        "Object-centric crop materialization pipeline",
        "A five-stage reconstruction of the formal crop pipeline, including coordinate transformation, context removal, and network resizing.",
        body,
        [TRAIN_GT, CASE_SELECTION, image_path(TRAIN_IMAGES[int(case["image_id"])])],
    )


def figure_crop_shift_case_gallery():
    selection = json.loads(CASE_SELECTION.read_text(encoding="utf-8"))
    scale_cases = selection["families"]["crop_scale_condition_shift"]
    context_cases = selection["families"]["crop_context_condition_shift"]
    cases = [("Scale shift", case) for case in scale_cases] + [("Context shift", case) for case in context_cases]
    width, height = 2500, 2100
    body = title_block(
        "Crop-Induced Scale and Context Shift Across Eight Frozen Cases",
        "Each pair compares the native scene with the materialized crop used by the risk-data arm",
        width,
    )
    panel_w, panel_h = 500, 281
    sources = [TRAIN_GT, CASE_SELECTION]
    for index, (family, case) in enumerate(cases):
        col, row = index % 2, index // 2
        ox, oy = 80 + col * 1220, 205 + row * 455
        _, source_rendered, crop, _ = draw_crop_case(case)
        details = case["scale_context"]
        body.append(text(ox, oy, f"({chr(97 + index)}) {family} · {case['class_name']}", 20, 500))
        body.append(text(ox, oy + 31, "Native scene", 16, fill=MUTED))
        body.append(text(ox + 560, oy + 31, "Object-centric crop", 16, fill=MUTED))
        body.append(raster_panel(ox, oy + 48, panel_w, panel_h, source_rendered, fit="cover"))
        body.append(rect(ox, oy + 48, panel_w, panel_h, "none", INK, 1))
        body.append(raster_panel(ox + 560, oy + 48, panel_w, panel_h, crop, fit="cover"))
        body.append(rect(ox + 560, oy + 48, panel_w, panel_h, "none", INK, 1))
        body.append(text(ox, oy + 360, f"scale ×{float(details['short_side_amplification']):.2f}", 16, 500, fill=NAVY))
        body.append(text(ox + 180, oy + 360, f"neighbors retained {float(details['neighbor_retention_fraction']):.1%}", 16, fill=MUTED))
        body.append(text(ox + 490, oy + 360, f"removed {int(float(details['fully_removed_objects']))}", 16, fill=MUTED))
        body.append(text(ox + 650, oy + 360, f"clipped {int(float(details['boundary_clipped_objects']))}", 16, fill=MUTED))
        body.append(line(ox, oy + 390, ox + 1060, oy + 390, GRID, 1))
        sources.append(image_path(TRAIN_IMAGES[int(case["image_id"])]))
    return save_figure(
        "figure-18-crop-shift-case-gallery",
        width,
        height,
        "Multi-case crop scale and context shift",
        "Eight frozen examples demonstrate that target enlargement, neighbor removal, and boundary clipping are systematic rather than isolated.",
        body,
        sources,
    )


def polyline_chart(body, x, y, w, h, xs, series, y_min, y_max, x_label, y_label):
    for tick in np.linspace(y_min, y_max, 5):
        yy = y + h - (tick - y_min) / (y_max - y_min) * h
        body.append(line(x, yy, x + w, yy, GRID, 1))
        body.append(text(x - 14, yy + 5, f"{tick:.2f}", 15, anchor="end", fill=MUTED))
    body.append(line(x, y + h, x + w, y + h, INK, 1.2))
    body.append(line(x, y, x, y + h, INK, 1.2))
    x_min, x_max = min(xs), max(xs)
    for label, values, color in series:
        points = []
        for xv, value in zip(xs, values):
            px = x + (xv - x_min) / (x_max - x_min) * w
            py = y + h - (value - y_min) / (y_max - y_min) * h
            points.append((px, py))
        body.append(base.polyline(points, color, 2.5))
        body.append(text(x + w - 5, points[-1][1] - 8, label, 16, 500, anchor="end", fill=color))
    body.append(text(x + w / 2, y + h + 50, x_label, 17, 500, anchor="middle"))
    body.append(text(x - 65, y + h / 2, y_label, 17, 500, anchor="middle", rotate=-90))


def supplement_baseline_training_curves():
    yolo_source = PROJECT / "data-and-baselines/yolo11m/results.csv"
    dfine_source = PROJECT / "data-and-baselines/dfine-m/metrics/training_metrics.csv"
    yolo = pd.read_csv(yolo_source)
    dfine = pd.read_csv(dfine_source)
    width, height = 2200, 1200
    body = title_block(
        "Baseline Optimization and Validation Curves",
        "YOLO11m and D-FINE-M training evidence retained from their original baseline runs",
        width,
    )
    body.append(text(110, 190, "A  YOLO11m validation accuracy", 22, 500))
    polyline_chart(
        body, 150, 245, 820, 360, yolo.epoch.tolist(),
        [("mAP50", yolo["metrics/mAP50(B)"].tolist(), BLUE), ("mAP50–95", yolo["metrics/mAP50-95(B)"].tolist(), ORANGE)],
        0.20, 0.70, "epoch", "metric",
    )
    body.append(text(1180, 190, "B  YOLO11m training loss", 22, 500))
    yolo_total = yolo["train/box_loss"] + yolo["train/cls_loss"] + yolo["train/dfl_loss"]
    polyline_chart(body, 1220, 245, 820, 360, yolo.epoch.tolist(), [("total", yolo_total.tolist(), NAVY)], 2.0, 4.0, "epoch", "loss")
    body.append(text(110, 710, "C  D-FINE-M validation accuracy", 22, 500))
    polyline_chart(body, 150, 760, 820, 300, dfine.epoch.tolist(), [("AP50", dfine.AP50.tolist(), BLUE), ("AP", dfine.AP.tolist(), ORANGE)], 0.25, 0.70, "epoch", "metric")
    body.append(text(1180, 710, "D  D-FINE-M training loss", 22, 500))
    polyline_chart(body, 1220, 760, 820, 300, dfine.epoch.tolist(), [("loss", dfine.loss.tolist(), NAVY)], float(dfine.loss.min()) * 0.95, float(dfine.loss.max()) * 1.02, "epoch", "loss")
    return save_figure(
        "supplement-s5-baseline-training-curves",
        width,
        height,
        "Baseline training curves",
        "Optimization and validation trajectories for the original YOLO11m and D-FINE-M baselines.",
        body,
        [yolo_source, dfine_source],
    )


def supplement_per_class_baseline_metrics():
    source = PROJECT / "failure-diagnosis/unified-evaluation/per_class_metrics.csv"
    df = pd.read_csv(source)
    classes = list(dict.fromkeys(df.class_name.tolist()))
    width, height = 2200, 1260
    body = title_block(
        "Per-Class Baseline Accuracy on Official Validation",
        "Canonical COCO AP for YOLO11m and D-FINE-M under the unified evaluation protocol",
        width,
    )
    x0, y0, chart_w = 520, 220, 1450
    max_ap = 0.70
    for index, class_name in enumerate(classes):
        y = y0 + index * 92
        body.append(text(x0 - 30, y + 30, class_name, 18, 500, anchor="end"))
        yolo_ap = float(df[(df.model == "YOLO11m") & (df.class_name == class_name)].iloc[0].AP)
        dfine_ap = float(df[(df.model == "D-FINE-M") & (df.class_name == class_name)].iloc[0].AP)
        body.append(rect(x0, y, chart_w * yolo_ap / max_ap, 25, BLUE, "none"))
        body.append(rect(x0, y + 34, chart_w * dfine_ap / max_ap, 25, ORANGE, "none"))
        body.append(text(x0 + chart_w + 20, y + 20, f"{yolo_ap:.3f}", 16, fill=BLUE))
        body.append(text(x0 + chart_w + 20, y + 54, f"{dfine_ap:.3f}", 16, fill=ORANGE))
        body.append(line(x0, y + 73, x0 + chart_w, y + 73, GRID, 0.8))
    body.append(rect(520, 1165, 22, 22, BLUE))
    body.append(text(555, 1183, "YOLO11m", 17))
    body.append(rect(720, 1165, 22, 22, ORANGE))
    body.append(text(755, 1183, "D-FINE-M", 17))
    return save_figure(
        "supplement-s6-per-class-baseline-ap",
        width,
        height,
        "Per-class baseline AP",
        "Canonical per-class AP comparison between YOLO11m and D-FINE-M.",
        body,
        [source],
    )


def supplement_yolo11seg_examples():
    root = PROJECT / "data-and-baselines/yolo11-seg/visualizations"
    names = [
        "0000001_b1c66a42-6f7d68ca.jpg",
        "0000005_b1ca2e5d-84cf9134.jpg",
        "0000021_b1d0a191-5490450b.jpg",
        "0000033_b1d4b62c-60aab822.jpg",
        "0000049_b1d9e136-6c94ea3f.jpg",
        "0000077_b1f022d3-3774de8b.jpg",
    ]
    paths = [root / name for name in names]
    width, height = 2200, 1250
    body = title_block(
        "YOLO11-Seg Auxiliary Qualitative Evidence",
        "Segmentation outputs are retained as supporting perception evidence and are not part of the detector intervention comparison",
        width,
    )
    xs = [100, 820, 1540]
    ys = [220, 710]
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        x, y = xs[index % 3], ys[index // 3]
        body.append(text(x, y - 24, f"({chr(97 + index)}) Auxiliary segmentation example", 19, 500))
        body.append(raster_panel(x, y, 650, 366, image, fit="cover"))
        body.append(rect(x, y, 650, 366, "none", INK, 1))
    body.append(text(1100, 1180, "This evidence is reported separately to avoid conflating instance segmentation with bounding-box detection.", 20, 500, anchor="middle"))
    return save_figure(
        "supplement-s7-yolo11seg-auxiliary-examples",
        width,
        height,
        "YOLO11-Seg auxiliary examples",
        "Six retained YOLO11-Seg visualizations used only as auxiliary perception evidence.",
        body,
        paths,
    )


QUALITATIVE_FAMILIES = [
    ("P2 Repairs of Small-Object Failures", "p2_repairs_small_failure"),
    ("Failures Remaining After P2", "p2_remaining_failure"),
    ("Short-Term Recovery Under Risk Training", "risk_short_term_recovery"),
    ("Risk-Training Non-Transfer in the Native Scene", "risk_non_transfer_in_original_scene"),
    ("Crop-Induced Scale-Condition Shift", "crop_scale_condition_shift"),
    ("Crop-Induced Context-Condition Shift", "crop_context_condition_shift"),
    ("Blur or Noise Cases Repaired by P2", "blur_or_noise_success"),
    ("Blur or Noise Failures Remaining Under P2", "blur_or_noise_failure"),
]


def supplement_family_sheet(title_value: str, stem: str, number: int):
    source = PROJECT / "final-testing/qualitative-cases/figures" / f"{stem}.png"
    image = Image.open(source).convert("RGB")
    width, height = 2500, 1500
    body = title_block(title_value, "Four deterministically selected official-validation cases", width)
    body.append(raster_panel(70, 175, 2360, 1220, image, fit="contain"))
    body.append(rect(70, 175, 2360, 1220, "none", GRID, 1))
    return save_figure(
        f"supplement-s{number:02d}-{stem.replace('_', '-')}",
        width,
        height,
        title_value,
        f"Four frozen qualitative cases for the {stem.replace('_', ' ')} family.",
        body,
        [source, CASE_SELECTION],
    )


def write_captions(records, main_count):
    lines = [
        "# Complete Publication Visual Evidence Suite",
        "",
        "All figures are derived from frozen project evidence. No training or inference was performed.",
        "",
    ]
    for index, record in enumerate(records, 1):
        if index <= main_count:
            label = f"Figure {index}"
        else:
            label = f"Supplementary Figure S{index - main_count}"
        lines.extend([f"## {label}. {record['title']}", "", record["description"], "", f"SVG: `{Path(record['svg']).name}`", "", "Data sources:", ""])
        lines.extend(f"- `{source}`" for source in record["sources"])
        lines.append("")
    (OUT / "FIGURE_CAPTIONS_AND_PROVENANCE.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(records, main_count):
    files = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "figure-manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(OUT)),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    payload = {
        "status": "complete",
        "figure_count": len(records),
        "main_figures": main_count,
        "supplementary_figures": len(records) - main_count,
        "new_training": False,
        "new_inference": False,
        "project_source_read_only": True,
        "figures": records,
        "files": files,
    }
    (OUT / "figure-manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    current = json.loads((OUT / "figure-manifest.json").read_text(encoding="utf-8"))["figures"]
    current_by_stem = {record["stem"]: record for record in current}
    original_main_stems = [
        "figure-01-research-pipeline",
        "figure-02-failure-structure",
        "figure-03-p2-architecture",
        "figure-04-method-gate",
        "figure-05-bootstrap-forest",
        "figure-06-training-trajectory",
        "figure-07-intervention-semantics",
        "figure-08-corruption-heatmap",
        "figure-09-efficiency-tradeoff",
        "figure-10-object-crop-transformation",
        "figure-11-controlled-corruption-examples",
        "figure-12-qualitative-repair-transfer",
    ]
    original_supplement_stems = [
        "supplement-s1-threshold-sensitivity",
        "supplement-s2-failure-overlap",
        "supplement-s3-matching-taxonomy-protocol",
        "supplement-s4-equal-budget-supervision",
    ]
    original_main = [current_by_stem[stem] for stem in original_main_stems]
    original_supplement = [current_by_stem[stem] for stem in original_supplement_stems]
    new_main = [
        figure_dataset_annotation_overview(),
        figure_object_scale_gallery(),
        figure_baseline_detector_comparison(),
        figure_failure_taxonomy_cases(),
        figure_crop_materialization_pipeline(),
        figure_crop_shift_case_gallery(),
    ]
    new_supplement = [
        supplement_baseline_training_curves(),
        supplement_per_class_baseline_metrics(),
        supplement_yolo11seg_examples(),
    ]
    family_supplement = [
        supplement_family_sheet(title_value, stem, 8 + index)
        for index, (title_value, stem) in enumerate(QUALITATIVE_FAMILIES)
    ]
    records = original_main + new_main + original_supplement + new_supplement + family_supplement
    main_count = len(original_main) + len(new_main)
    write_captions(records, main_count)
    write_manifest(records, main_count)
    print(json.dumps({"status": "complete", "figures": len(records), "main": main_count, "supplementary": len(records) - main_count, "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
