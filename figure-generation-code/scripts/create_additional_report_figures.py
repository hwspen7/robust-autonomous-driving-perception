from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps


PROJECT = Path(
    os.environ.get("DBR_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
OUTPUT = Path(
    os.environ.get(
        "DBR_ADDITIONAL_FIGURE_OUTPUT",
        PROJECT / "report-figures" / "generated" / "additional",
    )
).expanduser().resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)

FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")

WHITE = "#FFFFFF"
NAVY = "#173A5E"
BLUE = "#2C6E9F"
RUST = "#A0523D"
GREEN = "#3A7D67"
GOLD = "#B5893F"
INK = "#26313A"
MUTED = "#68757F"
GRID = "#D9E0E5"
LIGHT = "#EFF3F6"
RED = "#9B3B3B"


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size=size)


def text_width(draw: ImageDraw.ImageDraw, value: str, fnt: ImageFont.FreeTypeFont) -> float:
    box = draw.textbbox((0, 0), value, font=fnt)
    return box[2] - box[0]


def centered(draw: ImageDraw.ImageDraw, y: int, value: str, fnt: ImageFont.FreeTypeFont, fill: str, width: int) -> None:
    draw.text(((width - text_width(draw, value, fnt)) / 2, y), value, font=fnt, fill=fill)


def multiline_centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    value: str,
    fnt: ImageFont.FreeTypeFont,
    fill: str,
    spacing: int = 8,
) -> None:
    x1, y1, x2, y2 = box
    lines = value.split("\n")
    line_height = fnt.size + spacing
    total = len(lines) * line_height - spacing
    y = y1 + (y2 - y1 - total) / 2
    for line in lines:
        x = x1 + (x2 - x1 - text_width(draw, line, fnt)) / 2
        draw.text((x, y), line, font=fnt, fill=fill)
        y += line_height


def draw_panel_title(draw: ImageDraw.ImageDraw, x: int, y: int, value: str, color: str = NAVY) -> None:
    draw.text((x, y), value, font=font(42), fill=color, stroke_width=1, stroke_fill=color)


def horizontal_bar_panel(
    draw: ImageDraw.ImageDraw,
    panel: tuple[int, int, int, int],
    labels: list[str],
    values: list[float],
    colors: list[str],
    x_min: float,
    x_max: float,
    axis_label: str,
) -> None:
    x1, y1, x2, y2 = panel
    label_width = int((x2 - x1) * 0.42)
    plot_left = x1 + label_width
    plot_right = x2 - 80
    plot_top = y1 + 30
    plot_bottom = y2 - 90
    plot_width = plot_right - plot_left
    zero_x = plot_left + (0 - x_min) / (x_max - x_min) * plot_width

    for tick in range(int(x_min), int(x_max) + 1):
        tx = plot_left + (tick - x_min) / (x_max - x_min) * plot_width
        draw.line((tx, plot_top, tx, plot_bottom), fill=GRID, width=2)
        tick_text = f"{tick:+d}"
        draw.text((tx - text_width(draw, tick_text, font(24)) / 2, plot_bottom + 12), tick_text, font=font(24), fill=MUTED)
    draw.line((zero_x, plot_top, zero_x, plot_bottom), fill=INK, width=4)

    row_height = (plot_bottom - plot_top) / len(labels)
    bar_height = row_height * 0.56
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        cy = plot_top + (index + 0.5) * row_height
        label_y = cy - font(27).size / 2
        draw.text((x1 + 4, label_y), label, font=font(27), fill=INK)
        vx = plot_left + (value - x_min) / (x_max - x_min) * plot_width
        left = min(zero_x, vx)
        right = max(zero_x, vx)
        if right - left < 4:
            right = left + 4
        draw.rounded_rectangle((left, cy - bar_height / 2, right, cy + bar_height / 2), radius=5, fill=color)
        value_text = f"{value:+.2f}"
        if value >= 0:
            value_x = min(vx + 14, plot_right - text_width(draw, value_text, font(25)))
        else:
            value_x = max(plot_left, vx - 14 - text_width(draw, value_text, font(25)))
        draw.text((value_x, cy - font(25).size / 2), value_text, font=font(25), fill=INK)

    axis_width = text_width(draw, axis_label, font(27))
    draw.text((plot_left + (plot_width - axis_width) / 2, y2 - 48), axis_label, font=font(27), fill=MUTED)


def figure_unified_effects() -> None:
    width, height = 4300, 2450
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    centered(draw, 85, "Do Intervention Gains Transfer Beyond the Targeted Failure Condition?", font(66), NAVY, width)
    centered(draw, 185, "Absolute changes relative to the matched control, reported in percentage points", font(34), MUTED, width)

    draw_panel_title(draw, 130, 300, "A  Architecture gains remain positive on clean and corrupted inputs")
    draw_panel_title(draw, 2070, 300, "B  Targeted Risk recovery does not become a general repair")

    architecture_labels = ["Clean AP", "Clean AP75", "Clean APs", "Clean ARsmall", "Corruption AP", "Corruption APs"]
    architecture_values = [0.9033, 1.0603, 1.1807, 5.8590, 0.8259, 1.1182]
    horizontal_bar_panel(
        draw,
        (130, 400, 1940, 1900),
        architecture_labels,
        architecture_values,
        [NAVY] * len(architecture_labels),
        -0.5,
        6.5,
        "P2-960 − Standard-960  Δ percentage points",
    )
    architecture_note = "P2 preserves the native scene, scale, and context\nIts strongest gains occur in small-object precision and recall and persist under corruption"
    multiline_centered(draw, (130, 1950, 1940, 2170), architecture_note, font(29), MUTED)

    risk_labels = [
        "Gate clean AP",
        "Gate clean APs",
        "Failure macro AR100",
        "Small jointly-missed Recall50",
        "E56 clean AP",
        "E56 clean APs",
        "E56 corruption AP",
        "E56 corruption APs",
        "E100 clean AP",
        "E100 clean APs",
    ]
    risk_values = [0.0534, 0.2162, 0.7357, 3.2048, 0.0685, -0.2229, -0.0251, -0.2074, -0.3207, -0.2048]
    risk_colors = [BLUE, BLUE, GREEN, GREEN, GOLD, GOLD, RUST, RUST, RED, RED]
    horizontal_bar_panel(
        draw,
        (2070, 400, 4170, 1900),
        risk_labels,
        risk_values,
        risk_colors,
        -0.75,
        3.75,
        "Risk intervention − Uniform  Δ percentage points",
    )
    risk_note = "Green marks short-term recovery on the targeted failure subsets\nBrown and red show transfer back to native scenes, corruptions, and long-horizon training"
    multiline_centered(draw, (2070, 1950, 4170, 2170), risk_note, font(29), MUTED)

    draw.line((300, 2260, 4000, 2260), fill=GRID, width=3)
    centered(draw, 2290, "Targeted recovery and generalized repair must be evaluated separately", font(34), NAVY, width)
    image.save(OUTPUT / "figure-unified-intervention-transfer.png", dpi=(300, 300))


def load_per_class() -> dict[str, dict[str, float]]:
    path = PROJECT / "final-testing/canonical-clean/canonical_clean_per_class_metrics.csv"
    models: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            models.setdefault(row["model_key"], {})[row["class_name"]] = float(row["AP"])
    return models


def figure_per_class_response() -> None:
    models = load_per_class()
    order = ["traffic light", "pedestrian", "traffic sign", "car", "motorcycle", "bus", "rider", "bicycle", "train", "truck"]
    p2 = [(models["p2_e20"][c] - models["standard_e20"][c]) * 100 for c in order]
    risk = [(models["risk_e56"][c] - models["uniform_e56"][c]) * 100 for c in order]

    width, height = 4000, 2450
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    centered(draw, 80, "Class-Level Responses Reveal Different Intervention Stability", font(72), NAVY, width)
    centered(draw, 180, "Canonical COCO AP changes for matched classes, reported in percentage points", font(34), MUTED, width)
    draw.text((240, 300), "P2 improves all ten classes, while Risk responses are class-dependent and mixed in sign", font=font(42), fill=NAVY, stroke_width=1, stroke_fill=NAVY)

    plot_left, plot_right = 850, 3740
    plot_top, plot_bottom = 470, 2070
    x_min, x_max = -0.75, 1.85
    plot_width = plot_right - plot_left
    zero_x = plot_left + (0 - x_min) / (x_max - x_min) * plot_width

    ticks = [-0.5, 0, 0.5, 1.0, 1.5]
    for tick in ticks:
        tx = plot_left + (tick - x_min) / (x_max - x_min) * plot_width
        draw.line((tx, plot_top, tx, plot_bottom), fill=GRID, width=2)
        label = f"{tick:+.1f}"
        draw.text((tx - text_width(draw, label, font(25)) / 2, plot_bottom + 20), label, font=font(25), fill=MUTED)
    draw.line((zero_x, plot_top, zero_x, plot_bottom), fill=INK, width=4)

    row_height = (plot_bottom - plot_top) / len(order)
    bar_height = row_height * 0.30
    for index, category in enumerate(order):
        cy = plot_top + (index + 0.5) * row_height
        label = category.title()
        draw.text((240, cy - font(31).size / 2), label, font=font(31), fill=INK)
        for value, color, offset in [(p2[index], NAVY, -bar_height * 0.65), (risk[index], RUST, bar_height * 0.65)]:
            y = cy + offset
            vx = plot_left + (value - x_min) / (x_max - x_min) * plot_width
            left = min(zero_x, vx)
            right = max(zero_x, vx)
            draw.rounded_rectangle((left, y - bar_height / 2, max(right, left + 4), y + bar_height / 2), radius=5, fill=color)
            value_text = f"{value:+.2f}"
            if value >= 0:
                value_x = min(vx + 12, plot_right - text_width(draw, value_text, font(23)))
            else:
                value_x = max(plot_left, vx - 12 - text_width(draw, value_text, font(23)))
            draw.text((value_x, y - font(23).size / 2), value_text, font=font(23), fill=INK)

    draw.rounded_rectangle((2450, 300, 3750, 405), radius=8, fill=LIGHT)
    draw.rectangle((2490, 333, 2560, 363), fill=NAVY)
    draw.text((2580, 325), "P2-960 − Standard-960", font=font(27), fill=INK)
    draw.rectangle((3150, 333, 3220, 363), fill=RUST)
    draw.text((3240, 325), "Risk-E56 − Uniform-E56", font=font(27), fill=INK)

    centered(draw, 2135, "Class AP change  Δ percentage points", font(29), MUTED, width)
    draw.text((240, 2290), "Class-level values are descriptive point estimates. Frozen paired bootstrap provides the formal statistical inference.", font=font(29), fill=MUTED)
    image.save(OUTPUT / "figure-class-level-intervention-response.png", dpi=(300, 300))


def figure_failure_overlap() -> None:
    source = PROJECT / "final-testing/failure-subsets/subset_overlap.csv"
    subsets = [
        "small_jointly_missed",
        "dfine_supported_yolo_failure",
        "yolo_low_confidence",
        "yolo_localization_failure",
        "yolo_classification_failure",
        "small_high_risk",
    ]
    row_labels = [
        "Small jointly missed",
        "D-FINE-supported failure",
        "Low-confidence failure",
        "Localization failure",
        "Classification failure",
        "Small high-risk objects",
    ]
    column_labels = [
        "Small jointly\nmissed",
        "D-FINE-supported\nfailure",
        "Low-confidence\nfailure",
        "Localization\nfailure",
        "Classification\nfailure",
        "Small high-risk\nobjects",
    ]
    values: dict[tuple[str, str], float] = {}
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values[(row["left_subset"], row["right_subset"])] = float(row["jaccard"])

    width, height = 3400, 2250
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    centered(draw, 65, "Overlap Between Frozen Failure Subsets", font(70), NAVY, width)
    centered(draw, 160, "Jaccard similarity identifies diagnostic categories that contain the same objects", font(32), MUTED, width)
    draw.line((120, 245, width - 120, 245), fill=GRID, width=3)

    x0, y0, cell = 1120, 500, 250
    for index, label in enumerate(column_labels):
        multiline_centered(
            draw,
            (x0 + index * cell + 8, 330, x0 + (index + 1) * cell - 8, 485),
            label,
            font(25),
            INK,
            spacing=4,
        )

    for i, row_label in enumerate(row_labels):
        label_box = (170, y0 + i * cell, x0 - 45, y0 + (i + 1) * cell)
        multiline_centered(draw, label_box, row_label, font(28), INK, spacing=4)
        for j in range(len(subsets)):
            value = values[(subsets[i], subsets[j])]
            start = (246, 248, 251)
            end = (98, 87, 120)
            rgb = tuple(round(a + (b - a) * value) for a, b in zip(start, end))
            color = "#" + "".join(f"{channel:02X}" for channel in rgb)
            left = x0 + j * cell
            top = y0 + i * cell
            draw.rounded_rectangle(
                (left + 5, top + 5, left + cell - 5, top + cell - 5),
                radius=8,
                fill=color,
                outline=WHITE,
                width=4,
            )
            value_text = f"{value:.2f}"
            value_font = font(30)
            value_color = WHITE if value > 0.55 else INK
            draw.text(
                (
                    left + (cell - text_width(draw, value_text, value_font)) / 2,
                    top + (cell - value_font.size) / 2 - 4,
                ),
                value_text,
                font=value_font,
                fill=value_color,
            )

    note_top = 2050
    draw.rounded_rectangle((340, note_top, width - 340, 2175), radius=10, fill=LIGHT)
    centered(
        draw,
        note_top + 38,
        "Small jointly missed and small high-risk objects overlap strongly, while D-FINE-supported failures remain distinct.",
        font(28),
        INK,
        width,
    )
    image.save(OUTPUT / "supplement-s2-failure-overlap.png", dpi=(300, 300))


def image_file_and_bbox() -> tuple[Path, tuple[float, float, float, float]]:
    annotations = PROJECT / "data-and-baselines/dataset/annotations/instances_val.json"
    with annotations.open(encoding="utf-8") as handle:
        data = json.load(handle)
    image_record = next(item for item in data["images"] if item["id"] == 3650)
    annotation = next(item for item in data["annotations"] if item["id"] == 68199)
    rel = image_record["file_name"].split("images/val/", 1)[-1]
    image_path = PROJECT / "local-archive/data-disk/autodrive/datasets/datasets/bdd100k_final/images/val" / rel
    x, y, w, h = map(float, annotation["bbox"])
    return image_path, (x, y, w, h)


def crop_model_panel(sheet: Image.Image, column: int) -> Image.Image:
    width, _ = sheet.size
    col_w = width // 5
    left = column * col_w
    right = width if column == 4 else (column + 1) * col_w
    return sheet.crop((left, 720, right, 970))


def figure_same_object_loop() -> None:
    sheet_path = PROJECT / "final-testing/qualitative-cases/figures/risk_non_transfer_in_original_scene.png"
    sheet = Image.open(sheet_path).convert("RGB")
    panels = [crop_model_panel(sheet, index) for index in range(5)]
    titles = [
        "Ground Truth",
        "Baseline YOLO\nMissed",
        "P2-E20\nDetected",
        "Uniform-E56\nDetected",
        "Risk-E56\nMissed",
    ]

    image_path, bbox = image_file_and_bbox()
    source = Image.open(image_path).convert("RGB")
    x, y, w, h = bbox
    target_cx = x + w / 2
    target_cy = y + h / 2
    crop_w, crop_h = 384, 216
    left = max(0, min(source.width - crop_w, int(target_cx - crop_w / 2)))
    top = max(0, min(source.height - crop_h, int(target_cy - crop_h / 2)))
    crop = source.crop((left, top, left + crop_w, top + crop_h))
    crop = ImageEnhance.Contrast(crop).enhance(1.04)
    crop_draw = ImageDraw.Draw(crop)
    bx, by = x - left, y - top
    crop_draw.rectangle((bx, by, bx + w, by + h), outline="#FFD34D", width=3)
    panels.append(crop)
    titles.append("Object-Centric Analytical View\nScale and Context Changed")

    width, height = 3600, 2200
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    centered(draw, 55, "One Object Across Diagnosis, Intervention, and Verification", font(66), NAVY, width)
    centered(draw, 145, "Official-validation object  |  Annotation 68199  |  Car  |  Night highway  |  Native box 32 × 25 px", font(31), MUTED, width)

    margin_x, gap_x = 100, 45
    cell_w = (width - 2 * margin_x - 2 * gap_x) // 3
    cell_h = 570
    title_h = 88
    start_y = 245
    border_colors = [GOLD, RED, NAVY, GREEN, RED, RUST]

    for index, (panel, title, color) in enumerate(zip(panels, titles, border_colors)):
        row, col = divmod(index, 3)
        cell_x = margin_x + col * (cell_w + gap_x)
        cell_y = start_y + row * (cell_h + title_h + 72)
        multiline_centered(draw, (cell_x, cell_y, cell_x + cell_w, cell_y + title_h), title, font(31), color, spacing=4)
        panel_copy = ImageOps.contain(
            panel.convert("RGB"),
            (cell_w - 28, cell_h - 24),
            method=Image.Resampling.LANCZOS,
        )
        px = cell_x + (cell_w - panel_copy.width) // 2
        py = cell_y + title_h + (cell_h - panel_copy.height) // 2
        image.paste(panel_copy, (px, py))
        draw.rectangle((px, py, px + panel_copy.width, py + panel_copy.height), outline=color, width=7)

    note_y = 1905
    centered(
        draw,
        note_y,
        "P2 recovers the object in the native night scene. Risk training does not retain that recovery under the same conditions.",
        font(29),
        INK,
        width,
    )
    centered(
        draw,
        note_y + 70,
        "The analytical crop preserves object identity but changes scale and context. At 960 input, the short side increases from about 25 px to 63 px.",
        font(25),
        MUTED,
        width,
    )
    image.save(OUTPUT / "figure-same-object-diagnose-intervene-verify.png", dpi=(300, 300))


def main() -> None:
    figure_unified_effects()
    figure_per_class_response()
    figure_failure_overlap()
    figure_same_object_loop()
    for path in sorted(OUTPUT.glob("*.png")):
        print(path.name, path.stat().st_size)


if __name__ == "__main__":
    main()
