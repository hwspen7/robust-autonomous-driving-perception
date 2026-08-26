from __future__ import annotations

import base64
import hashlib
import html
import importlib.util
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageOps


PROJECT = Path(
    os.environ.get("DBR_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
OUT = Path(
    os.environ.get(
        "DBR_FIGURE_OUTPUT",
        PROJECT / "report-figures" / "generated",
    )
).expanduser().resolve()
TABLES = PROJECT / "research-evidence/frozen-results/tables"

INK = "#222222"
MUTED = "#606060"
GRID = "#D0D0D0"
BG = "#FFFFFF"
BLUE = "#2D5478"
BLUE2 = "#7290AA"
NAVY = "#1E3A52"
ORANGE = "#9B5C32"
RED = "#913F3F"
GREEN = "#3E695B"
PURPLE = "#625878"
PALE_BLUE = "#F1F4F6"
PALE_ORANGE = "#F6F2EE"
PALE_GREEN = "#F0F4F2"
PALE_RED = "#F6F0F0"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(x, y, value, size=24, weight=400, anchor="start", fill=INK, rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (
        f'<text x="{x}" y="{y}" font-family="Times New Roman, Times, serif" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}"{transform}>{esc(value)}</text>'
    )


def multiline(x, y, lines, size=24, weight=400, anchor="start", fill=INK, gap=1.25):
    spans = []
    for i, value in enumerate(lines):
        dy = 0 if i == 0 else size * gap
        spans.append(f'<tspan x="{x}" dy="{dy}">{esc(value)}</tspan>')
    return (
        f'<text x="{x}" y="{y}" font-family="Times New Roman, Times, serif" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">' + "".join(spans) + "</text>"
    )


def rect(x, y, w, h, fill="none", stroke="none", sw=1, rx=0, opacity=1.0):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>'


def line(x1, y1, x2, y2, stroke=INK, sw=2, dash=None, marker=None, opacity=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"{d}{m}/>'


def circle(cx, cy, r, fill="none", stroke="none", sw=1):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def polyline(points, stroke=INK, sw=3, dash=None, opacity=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    pts = " ".join(f"{x},{y}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}" stroke-linejoin="round" stroke-linecap="round"{d}/>'


def svg_doc(width: int, height: int, title_value: str, desc: str, body: list[str]) -> str:
    defs = """
    <defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#222222"/>
      </marker>
    </defs>
    """
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<title>{esc(title_value)}</title><desc>{esc(desc)}</desc>{defs}{rect(0, 0, width, height, BG)}'
        + "".join(body) + "</svg>\n"
    )


def save_figure(stem: str, width: int, height: int, title_value: str, desc: str, body: list[str], sources: list[Path]):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{stem}.svg"
    path.write_text(svg_doc(width, height, title_value, desc, body), encoding="utf-8")
    return {
        "stem": stem,
        "title": title_value,
        "description": desc,
        "svg": str(path),
        "sources": [str(p) for p in sources],
    }


def title_block(title_value: str, subtitle: str, width: int) -> list[str]:
    return [
        text(80, 66, title_value, 31, 500),
        text(80, 101, subtitle, 18, fill=MUTED),
        line(80, 126, width - 80, 126, GRID, 1.4),
    ]


def blend(a: str, b: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    av = tuple(int(a[i : i + 2], 16) for i in (1, 3, 5))
    bv = tuple(int(b[i : i + 2], 16) for i in (1, 3, 5))
    cv = tuple(round(x + (y - x) * t) for x, y in zip(av, bv))
    return "#" + "".join(f"{v:02X}" for v in cv)


def raster_data_url(image: Image.Image, *, quality: int = 91) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def raster_panel(x, y, w, h, image: Image.Image, *, fit: str = "contain") -> str:
    target = (max(1, int(w)), max(1, int(h)))
    if fit == "cover":
        rendered = ImageOps.fit(image.convert("RGB"), target, method=Image.Resampling.LANCZOS)
    else:
        rendered = ImageOps.contain(image.convert("RGB"), target, method=Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", target, "white")
        canvas.paste(rendered, ((target[0] - rendered.width) // 2, (target[1] - rendered.height) // 2))
        rendered = canvas
    return (
        f'<image x="{x}" y="{y}" width="{w}" height="{h}" '
        f'href="{raster_data_url(rendered)}" preserveAspectRatio="none"/>'
    )


def draw_bbox(image: Image.Image, bbox, *, color="#234A6F", width=5) -> Image.Image:
    output = image.convert("RGB").copy()
    x, y, w, h = [float(v) for v in bbox]
    draw = ImageDraw.Draw(output)
    draw.rectangle((x, y, x + w, y + h), outline=color, width=width)
    return output


def load_coco(annotation_file: Path):
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in data["images"]}
    annotations = {int(row["id"]): row for row in data["annotations"]}
    return images, annotations


def dataset_image(file_name: str) -> Path:
    return (
        PROJECT
        / "local-archive/data-disk/autodrive/datasets/datasets/bdd100k_final"
        / file_name
    )


def context_crop(image: Image.Image, bbox, *, width_fraction=0.48, aspect=16 / 9):
    x, y, w, h = [float(v) for v in bbox]
    source_w, source_h = image.size
    crop_w = min(source_w, max(420, int(source_w * width_fraction)))
    crop_h = min(source_h, int(crop_w / aspect))
    cx, cy = x + w / 2, y + h / 2
    x1 = int(round(cx - crop_w / 2))
    y1 = int(round(cy - crop_h / 2))
    x1 = max(0, min(source_w - crop_w, x1))
    y1 = max(0, min(source_h - crop_h, y1))
    return image.crop((x1, y1, x1 + crop_w, y1 + crop_h)), (x1, y1, crop_w, crop_h)


def axis_horizontal(body, x0, y0, w, vmin, vmax, ticks, label=None, fmt_fn=lambda x: f"{x:.2f}"):
    body.append(line(x0, y0, x0 + w, y0, INK, 1.5))
    for v in ticks:
        x = x0 + (v - vmin) / (vmax - vmin) * w
        body.append(line(x, y0, x, y0 + 8, INK, 1))
        body.append(text(x, y0 + 30, fmt_fn(v), 16, anchor="middle", fill=MUTED))
    if label:
        body.append(text(x0 + w / 2, y0 + 62, label, 18, 500, anchor="middle"))


def axis_vertical(body, x0, y0, h, vmin, vmax, ticks, label=None, fmt_fn=lambda x: f"{x:.2f}"):
    body.append(line(x0, y0 - h, x0, y0, INK, 1.5))
    for v in ticks:
        y = y0 - (v - vmin) / (vmax - vmin) * h
        body.append(line(x0 - 7, y, x0, y, INK, 1))
        body.append(text(x0 - 14, y + 6, fmt_fn(v), 16, anchor="end", fill=MUTED))
    if label:
        body.append(text(x0 - 72, y0 - h / 2, label, 18, 500, anchor="middle", rotate=-90))


def figure_research_pipeline():
    width, height = 1900, 1040
    body = title_block(
        "Diagnose Before Repair: Experimental Logic",
        "Object-level failure diagnosis, matched interventions, and multi-axis verification",
        width,
    )
    x = [90, 500, 930, 1420]
    w = [330, 340, 380, 390]
    y, h = 210, 670
    fills = [PALE_BLUE, "#EEF1FA", PALE_ORANGE, PALE_GREEN]
    headers = ["1  DIAGNOSE", "2  REPRESENT", "3  INTERVENE", "4  VERIFY"]
    for xi, wi, fill, header in zip(x, w, fills, headers):
        body.append(rect(xi, y, wi, h, fill, GRID, 2, 12))
        body.append(text(xi + 28, y + 52, header, 24, 500, fill=NAVY))
    body += [
        multiline(120, 315, ["YOLO11m", "D-FINE-M"], 27, 500),
        text(120, 405, "Paired object matching", 21),
        text(120, 452, "185,523 validation objects", 21),
        text(120, 520, "Six frozen failure subsets", 21),
        multiline(120, 595, ["Missed", "Low confidence", "Localization", "Classification"], 20, fill=MUTED),
        multiline(530, 315, ["Object risk", "Failure type", "Scale", "Teacher agreement"], 24, 500),
        text(530, 485, "Failure is a condition", 22, 500, fill=PURPLE),
        multiline(530, 535, ["not merely a high", "risk score"], 20, fill=MUTED),
    ]
    body.append(rect(965, 300, 310, 205, PALE_BLUE, BLUE2, 2, 10))
    body.append(text(1120, 348, "Architecture", 24, 500, anchor="middle", fill=NAVY))
    body.append(text(1120, 392, "Projected P2 head", 22, 500, anchor="middle"))
    body.append(multiline(1120, 438, ["Preserves scene", "and native scale"], 19, anchor="middle", fill=MUTED))
    body.append(rect(965, 575, 310, 205, PALE_ORANGE, ORANGE, 2, 10))
    body.append(text(1120, 623, "Risk data", 24, 500, anchor="middle", fill="#9A531E"))
    body.append(text(1120, 667, "Object-centric crops", 22, 500, anchor="middle"))
    body.append(multiline(1120, 713, ["Changes scale", "and context"], 19, anchor="middle", fill=MUTED))
    body += [
        multiline(1455, 300, ["Canonical clean", "Failure recovery", "Corruption", "Training trajectory", "Efficiency", "Qualitative cases"], 23, 500),
        text(1455, 610, "Decision principle", 22, 500, fill=GREEN),
        multiline(1455, 655, ["A repair is valid only", "when the original failure", "condition is preserved"], 21, fill=MUTED),
    ]
    for a, b in [(420, 500), (840, 930), (1310, 1420)]:
        body.append(line(a, 545, b - 18, 545, INK, 3, marker="arrow"))
    body.append(text(950, 938, "Diagnose  →  Match the intervention to the failure mechanism  →  Verify transfer", 26, 500, anchor="middle"))
    return save_figure(
        "figure-01-research-pipeline", width, height,
        "Diagnose Before Repair experimental logic",
        "The full research pipeline from paired detector diagnosis to architecture and risk-data interventions and verification.",
        body, [PROJECT / "final-testing/frozen-protocol/final-testing-protocol.json"]
    )


def figure_failure_structure():
    source = PROJECT / "final-testing/failure-subsets/failure_object_table.csv"
    df = pd.read_csv(source)
    subsets = [
        ("Small jointly missed", "in_small_jointly_missed"),
        ("D-FINE supported YOLO failure", "in_dfine_supported_yolo_failure"),
        ("YOLO low confidence", "in_yolo_low_confidence"),
        ("YOLO localization failure", "in_yolo_localization_failure"),
        ("YOLO classification failure", "in_yolo_classification_failure"),
        ("Small high risk", "in_small_high_risk"),
    ]
    scales = ["small", "medium", "large"]
    classes = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"]
    scale_values, class_values, counts = [], [], []
    for _, flag in subsets:
        sub = df[df[flag] == 1]
        counts.append(len(sub))
        scale_values.append([(sub["scale"] == s).mean() for s in scales])
        class_values.append([(sub["class_name"] == c).mean() for c in classes])
    width, height = 2200, 1200
    body = title_block("Where the Failures Concentrate", "Row-normalized composition of six frozen official-validation failure subsets", width)
    row_h, y0 = 118, 260
    body.append(text(520, 185, "Scale composition", 24, 500, anchor="middle"))
    body.append(text(1510, 185, "Class composition", 24, 500, anchor="middle"))
    scale_x, scale_cw = 390, 170
    class_x, class_cw = 890, 112
    for j, value in enumerate(scales):
        body.append(text(scale_x + j * scale_cw + scale_cw / 2, 225, value.title(), 18, 500, anchor="middle"))
    class_labels = ["Ped", "Rider", "Car", "Truck", "Bus", "Train", "Moto", "Bike", "T-light", "T-sign"]
    for j, value in enumerate(class_labels):
        body.append(text(class_x + j * class_cw + class_cw / 2, 230, value, 16, 500, anchor="middle", rotate=-35))
    max_class = max(max(row) for row in class_values)
    for i, ((label, _), count, sv, cv) in enumerate(zip(subsets, counts, scale_values, class_values)):
        y = y0 + i * row_h
        body.append(text(350, y + 43, label, 19, 500, anchor="end"))
        body.append(text(350, y + 72, f"n = {count:,}", 16, anchor="end", fill=MUTED))
        for j, value in enumerate(sv):
            color = blend("#F3F7FB", NAVY, value)
            body.append(rect(scale_x + j * scale_cw, y, scale_cw - 8, 88, color, BG, 2, 4))
            body.append(text(scale_x + j * scale_cw + (scale_cw - 8) / 2, y + 54, f"{value*100:.1f}%", 19, 500, anchor="middle", fill=BG if value > 0.48 else INK))
        for j, value in enumerate(cv):
            color = blend("#F6F8FB", PURPLE, value / max_class)
            body.append(rect(class_x + j * class_cw, y, class_cw - 5, 88, color, BG, 2, 3))
            if value >= 0.025:
                body.append(text(class_x + j * class_cw + (class_cw - 5) / 2, y + 53, f"{value*100:.0f}%", 16, 500, anchor="middle", fill=BG if value / max_class > 0.55 else INK))
    body.append(text(390, 1035, "Darker cells indicate a larger share within the same failure subset", 18, fill=MUTED))
    body.append(rect(390, 1070, 1520, 70, PALE_BLUE, rx=8))
    body.append(text(1150, 1114, "Small targets dominate jointly missed and high-risk subsets, while detector-specific failures retain distinct class structure.", 23, 500, anchor="middle"))
    return save_figure(
        "figure-02-failure-structure", width, height,
        "Failure structure across scale and class",
        "Heatmaps show the scale and class composition of the six official-validation failure subsets.",
        body, [source, PROJECT / "final-testing/failure-subsets/subset_summary.csv"]
    )


def feature_pyramid(body, x, y, p2=False):
    levels = [("P5", "30×30", 155), ("P4", "60×60", 205), ("P3", "120×120", 270)]
    if p2:
        levels.append(("P2", "240×240", 345))
    for i, (name, res, size) in enumerate(levels):
        yy = y + (len(levels) - i - 1) * 112
        fill = PALE_ORANGE if name == "P2" else PALE_BLUE
        stroke = ORANGE if name == "P2" else BLUE2
        body.append(rect(x + (345-size)/2, yy, size, 74, fill, stroke, 2, 6))
        body.append(text(x + 172, yy + 31, name, 22, 500, anchor="middle"))
        body.append(text(x + 172, yy + 57, res, 17, anchor="middle", fill=MUTED))
    body.append(text(x + 172, y + 515, "960 × 960 input", 20, 500, anchor="middle"))


def figure_p2_architecture():
    metrics = pd.read_csv(TABLES / "table_02_architecture_clean_efficiency.csv")
    std = metrics[metrics.model == "standard_e20"].iloc[0]
    p2 = metrics[metrics.model == "p2_e20"].iloc[0]
    width, height = 2100, 1120
    body = title_block(
        "Projected P2 Intervention Preserves the Original Failure Condition",
        "A stride-4 detection path is added while the full 960-pixel scene and native object scale remain unchanged", width)
    body.append(text(330, 205, "Standard YOLO11m", 27, 500, anchor="middle"))
    body.append(text(800, 205, "Projected P2 YOLO11m", 27, 500, anchor="middle"))
    feature_pyramid(body, 158, 260, False)
    feature_pyramid(body, 628, 260, True)
    body.append(line(505, 500, 610, 500, INK, 3, marker="arrow"))
    body.append(text(558, 458, "add P2", 19, 500, anchor="middle", fill=ORANGE))
    body.append(rect(1030, 205, 990, 760, "#FAFBFD", GRID, 2, 10))
    body.append(text(1080, 260, "Canonical effect on official validation", 25, 500))
    gains = [
        ("AP", p2.AP - std.AP, 0.015, BLUE),
        ("AP75", p2.AP75 - std.AP75, 0.015, BLUE),
        ("AP_small", p2.AP_small - std.AP_small, 0.015, ORANGE),
        ("AR_small", p2.AR_small - std.AR_small, 0.070, GREEN),
    ]
    bx, bw = 1290, 590
    for i, (label, value, vmax, color) in enumerate(gains):
        yy = 330 + i * 105
        body.append(text(1250, yy + 28, label, 21, 500, anchor="end"))
        body.append(rect(bx, yy, bw, 46, "#EDF1F5", rx=4))
        body.append(rect(bx, yy, bw * value / vmax, 46, color, rx=4))
        body.append(text(bx + bw + 28, yy + 31, f"+{value:.4f}", 21, 500, fill=color))
    body.append(line(1080, 760, 1960, 760, GRID, 2))
    body.append(text(1080, 815, "Measured cost relative to Standard", 23, 500))
    costs = [
        ("Parameters", p2.parameters / std.parameters - 1),
        ("GFLOPs", p2.GFLOPs / std.GFLOPs - 1),
        ("Median latency", p2.median_latency_ms / std.median_latency_ms - 1),
        ("Training memory", p2.peak_training_MiB / std.peak_training_MiB - 1),
    ]
    for i, (label, value) in enumerate(costs):
        xx = 1110 + i * 225
        body.append(text(xx + 90, 865, f"+{value*100:.1f}%", 27, 500, anchor="middle", fill=RED if value > 0.5 else ORANGE))
        body.append(text(xx + 90, 902, label, 17, anchor="middle", fill=MUTED))
    body.append(rect(140, 985, 1820, 72, PALE_GREEN, rx=8))
    body.append(text(1050, 1031, "The architectural repair improves small-object recall without rescaling or removing the surrounding scene.", 24, 500, anchor="middle"))
    return save_figure(
        "figure-03-p2-architecture", width, height,
        "Projected P2 architecture and measured trade-off",
        "Comparison between the standard stride 8, 16, 32 detection pyramid and the projected stride 4, 8, 16, 32 pyramid with canonical gains and costs.",
        body, [TABLES / "table_02_architecture_clean_efficiency.csv", PROJECT / "architecture-intervention/inputs/projected-p2-definition/yolo11m_p2_projected_bdd10.yaml"]
    )


def mini_delta_bars(body, x, y, w, h, labels, values, colors, title_value, axis_label):
    body.append(text(x, y - 28, title_value, 22, 500))
    vmin = min(min(values), 0)
    vmax = max(max(values), 0)
    pad = max((vmax - vmin) * 0.22, 0.001)
    vmin -= pad
    vmax += pad
    zero = x + (0 - vmin) / (vmax - vmin) * w
    body.append(line(zero, y, zero, y + h, GRID, 2))
    rh = h / len(labels)
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        cy = y + i * rh + rh * 0.5
        vx = x + (value - vmin) / (vmax - vmin) * w
        left = min(zero, vx)
        body.append(rect(left, cy - 16, abs(vx - zero), 32, color, rx=3))
        body.append(text(x - 16, cy + 6, label, 17, 500, anchor="end"))
        body.append(text(vx + (12 if value >= 0 else -12), cy + 6, f"{value:+.4f}", 16, 500, anchor="start" if value >= 0 else "end", fill=color))
    axis_horizontal(body, x, y + h + 8, w, vmin, vmax, [vmin + (vmax-vmin)*k/2 for k in range(3)], axis_label, lambda v: f"{v:+.3f}")


def figure_method_gate():
    clean = pd.read_csv(TABLES / "table_03_method_clean.csv").set_index("model_key")
    failure = pd.read_csv(TABLES / "table_05_failure_recovery.csv").set_index("model")
    corr_path = PROJECT / "risk-data-intervention/results/corruption-evaluation/corruption_metrics.json"
    corr = json.loads(corr_path.read_text())["summary"]
    models = ["gate_scalar", "gate_typed", "gate_typed_v2"]
    labels = ["Scalar risk", "Typed CAFR", "Typed CAFR V2"]
    colors = [BLUE, PURPLE, ORANGE]
    clean_ap = [clean.loc[m, "AP_small"] - clean.loc["gate_uniform", "AP_small"] for m in models]
    fail_macro = [failure.loc[m, "failure_macro_AR100"] - failure.loc["gate_uniform", "failure_macro_AR100"] for m in models]
    fail_small = [failure.loc[m, "small_jointly_missed_recall50"] - failure.loc["gate_uniform", "small_jointly_missed_recall50"] for m in models]
    width, height = 2200, 1120
    body = title_block(
        "Risk-Guided Crops Improve Selected Failure Metrics but Do Not Establish Transfer",
        "Equal-budget method gate on the same P2 initialization", width)
    mini_delta_bars(body, 270, 255, 460, 430, labels, clean_ap, colors, "A  Clean AP_small vs Uniform", "difference")
    body.append(text(1030, 225, "B  Failure recovery vs Uniform", 22, 500))
    vals = [fail_macro, fail_small]
    met_names = ["Failure macro AR100", "Small jointly missed recall@0.50"]
    x0, y0, plot_w = 1030, 280, 500
    maxv = max(max(v) for v in vals) * 1.18
    for j, (metric_name, metric_vals) in enumerate(zip(met_names, vals)):
        yy = y0 + j * 200
        body.append(text(x0, yy - 18, metric_name, 18, 500))
        for i, (label, value, color) in enumerate(zip(labels, metric_vals, colors)):
            by = yy + i * 47
            body.append(rect(x0, by, plot_w, 28, "#EDF1F5", rx=3))
            body.append(rect(x0, by, plot_w * value / maxv, 28, color, rx=3))
            body.append(text(x0 + plot_w + 15, by + 21, f"{value:+.4f}", 16, 500, fill=color))
    body.append(text(1710, 225, "C  Corruption AP_small", 22, 500))
    corr_labels = ["Uniform", "Scalar risk", "Typed CAFR"]
    corr_models = ["uniform", "scalar_risk", "typed_cafr"]
    corr_vals = [corr[m]["mean_corruption_AP_small"] for m in corr_models]
    cx, cy, cw, ch = 1710, 300, 360, 390
    cmin, cmax = 0.134, 0.140
    axis_vertical(body, cx, cy + ch, ch, cmin, cmax, [0.134, 0.136, 0.138, 0.140], "AP_small", lambda v: f"{v:.3f}")
    barw = 82
    for i, (label, value, color) in enumerate(zip(corr_labels, corr_vals, [MUTED, BLUE, PURPLE])):
        xx = cx + 55 + i * 105
        bh = (value - cmin) / (cmax - cmin) * ch
        body.append(rect(xx, cy + ch - bh, barw, bh, color, rx=3))
        body.append(text(xx + barw/2, cy + ch - bh - 12, f"{value:.4f}", 16, 500, anchor="middle"))
        body.append(text(xx + barw/2, cy + ch + 35, label, 15, 500, anchor="middle", rotate=-25))
    body.append(rect(160, 835, 1880, 155, "#FAFBFD", GRID, 2, 8))
    body.append(text(210, 885, "Interpretation", 22, 500, fill=NAVY))
    body.append(multiline(210, 925, [
        "Failure-focused sampling raises selected recovery measures at the short gate.",
        "The same intervention does not produce a comparable clean small-object advantage, and Typed CAFR adds no stable recovery over Scalar risk."
    ], 20, fill=MUTED))
    return save_figure(
        "figure-04-method-gate", width, height,
        "Method-gate comparison of clean, failure, and corruption outcomes",
        "Three panels compare clean small-object performance, failure recovery, and corruption performance for equal-budget Uniform, Scalar Risk, and Typed CAFR arms.",
        body, [TABLES / "table_03_method_clean.csv", TABLES / "table_05_failure_recovery.csv", corr_path]
    )


def figure_bootstrap_forest():
    source = TABLES / "table_07_bootstrap_primary.csv"
    df = pd.read_csv(source)
    wanted = [
        ("p2_e20_minus_standard_e20", "AP_small", "P2 − Standard  |  AP_small"),
        ("p2_e20_minus_standard_e20", "AR_small", "P2 − Standard  |  AR_small"),
        ("gate_scalar_minus_gate_uniform", "failure_macro_AR100", "Scalar − Uniform  |  failure macro AR100"),
        ("gate_scalar_minus_gate_uniform", "small_jointly_missed_recall50", "Scalar − Uniform  |  small-joint recall@0.50"),
        ("gate_typed_minus_gate_uniform", "failure_macro_AR100", "Typed − Uniform  |  failure macro AR100"),
        ("gate_typed_minus_gate_uniform", "small_jointly_missed_recall50", "Typed − Uniform  |  small-joint recall@0.50"),
        ("gate_typed_minus_gate_scalar", "failure_macro_AR100", "Typed − Scalar  |  failure macro AR100"),
        ("rebu_risk_e56_minus_uniform_e56", "AP", "Risk E56 − Uniform E56  |  AP"),
        ("rebu_risk_e56_minus_uniform_e56", "AP_small", "Risk E56 − Uniform E56  |  AP_small"),
        ("risk_e100_minus_uniform_e100", "AP", "Risk E100 − Uniform E100  |  AP"),
        ("risk_e100_minus_uniform_e100", "AP_small", "Risk E100 − Uniform E100  |  AP_small"),
    ]
    rows = []
    for comp, metric, label in wanted:
        rows.append((label, df[(df.comparison == comp) & (df.metric == metric)].iloc[0]))
    width, height = 2100, 1220
    body = title_block(
        "Paired Bootstrap Confidence Intervals",
        "Official-validation image-level resampling, 2,000 paired replicates, seed 20260816", width)
    x0, y0, w, row_h = 1030, 220, 880, 78
    vmin, vmax = -0.015, 0.075
    zero = x0 + (0 - vmin) / (vmax - vmin) * w
    body.append(line(zero, y0 - 30, zero, y0 + row_h * len(rows), INK, 2, dash="7 6"))
    for i, (label, row) in enumerate(rows):
        cy = y0 + i * row_h
        if i in [2, 6, 7, 9]:
            body.append(line(100, cy - 38, 1960, cy - 38, GRID, 1.5))
        lo, hi, point = float(row.ci95_lower), float(row.ci95_upper), float(row.point_estimate)
        lo_x = x0 + (lo - vmin) / (vmax - vmin) * w
        hi_x = x0 + (hi - vmin) / (vmax - vmin) * w
        px = x0 + (point - vmin) / (vmax - vmin) * w
        stable_pos = bool(row.stable_positive)
        stable_neg = bool(row.stable_negative)
        color = GREEN if stable_pos else RED if stable_neg else MUTED
        body.append(text(970, cy + 7, label, 18, 500, anchor="end"))
        body.append(line(lo_x, cy, hi_x, cy, color, 5))
        body.append(line(lo_x, cy - 10, lo_x, cy + 10, color, 3))
        body.append(line(hi_x, cy - 10, hi_x, cy + 10, color, 3))
        body.append(circle(px, cy, 9, color, BG, 2))
        body.append(text(1940, cy + 7, f"{point:+.4f}  [{lo:+.4f}, {hi:+.4f}]", 17, anchor="end", fill=color))
    axis_horizontal(body, x0, y0 + row_h * len(rows) + 5, w, vmin, vmax, [-0.01, 0.00, 0.02, 0.04, 0.06], "paired difference", lambda v: f"{v:+.2f}")
    body.append(text(110, 1150, "Green intervals exclude zero in the positive direction. Red intervals exclude zero in the negative direction. Gray intervals cross zero.", 18, fill=MUTED))
    return save_figure(
        "figure-05-bootstrap-forest", width, height,
        "Paired bootstrap confidence intervals",
        "Forest plot of paired image-level bootstrap estimates for the primary architecture, targeted recovery, and long-horizon comparisons.",
        body, [source, PROJECT / "final-testing/paired-bootstrap/metadata.json"]
    )


def plot_line(body, xs, ys, xmap, ymap, color, sw=4, dash=None, markers=True):
    points = [(xmap(x), ymap(y)) for x, y in zip(xs, ys)]
    body.append(polyline(points, color, sw, dash=dash))
    if markers:
        for x, y in points:
            body.append(circle(x, y, 5.5, color, BG, 1.5))


def figure_trajectory():
    source = PROJECT / "final-testing/training-dynamics/matched_epoch_attribution.csv"
    df = pd.read_csv(source)
    width, height = 2200, 1100
    body = title_block(
        "Long-Horizon Training Separates Shared Schedule Degradation from Risk-Specific Effects",
        "Matched Uniform and Risk checkpoints on the five-epoch recoverable grid", width)
    x1, y1, w1, h1 = 180, 250, 820, 620
    body.append(text(x1, 205, "A  Canonical AP trajectory", 23, 500))
    xmin, xmax = 40, 100
    ymin, ymax = 0.328, 0.355
    xmap = lambda v: x1 + (v - xmin) / (xmax - xmin) * w1
    ymap = lambda v: y1 + h1 - (v - ymin) / (ymax - ymin) * h1
    for v in [0.33, 0.335, 0.34, 0.345, 0.35, 0.355]:
        yy = ymap(v)
        body.append(line(x1, yy, x1 + w1, yy, GRID, 1))
        body.append(text(x1 - 15, yy + 6, f"{v:.3f}", 16, anchor="end", fill=MUTED))
    for epoch in [40, 50, 60, 70, 80, 90, 100]:
        body.append(text(xmap(epoch), y1 + h1 + 32, str(epoch), 16, anchor="middle", fill=MUTED))
    body.append(line(x1, y1 + h1, x1 + w1, y1 + h1, INK, 1.5))
    body.append(line(x1, y1, x1, y1 + h1, INK, 1.5))
    epochs = df.global_epoch.to_numpy()
    plot_line(body, epochs, df.uniform_AP.to_numpy(), xmap, ymap, BLUE)
    plot_line(body, epochs, df.risk_AP.to_numpy(), xmap, ymap, ORANGE)
    common = float(df.common_AP.iloc[0])
    body.append(line(x1, ymap(common), x1+w1, ymap(common), GREEN, 3, dash="10 7"))
    body.append(text(x1 + w1 - 8, ymap(common)-12, "Common40", 17, 500, anchor="end", fill=GREEN))
    body.append(text(x1 + w1/2, y1+h1+68, "global epoch", 18, 500, anchor="middle"))
    body.append(text(x1-68, y1+h1/2, "AP", 18, 500, anchor="middle", rotate=-90))
    body.append(rect(220, 905, 24, 5, BLUE))
    body.append(text(255, 912, "Uniform", 17, 500))
    body.append(rect(370, 905, 24, 5, ORANGE))
    body.append(text(405, 912, "Risk", 17, 500))
    x2, y2, w2, h2 = 1190, 250, 820, 620
    body.append(text(x2, 205, "B  Risk − Uniform", 23, 500))
    dmin, dmax = -0.013, 0.003
    xmap2 = lambda v: x2 + (v - xmin) / (xmax - xmin) * w2
    ymap2 = lambda v: y2 + h2 - (v - dmin) / (dmax - dmin) * h2
    body.append(rect(x2, ymap2(0), w2, y2+h2-ymap2(0), PALE_RED, opacity=0.65))
    for v in [-0.012, -0.008, -0.004, 0.0]:
        yy = ymap2(v)
        body.append(line(x2, yy, x2+w2, yy, GRID if v else INK, 2 if v == 0 else 1, dash="7 6" if v == 0 else None))
        body.append(text(x2-15, yy+6, f"{v:+.3f}", 16, anchor="end", fill=MUTED))
    for epoch in [40, 50, 60, 70, 80, 90, 100]:
        body.append(text(xmap2(epoch), y2+h2+32, str(epoch), 16, anchor="middle", fill=MUTED))
    body.append(line(x2, y2+h2, x2+w2, y2+h2, INK, 1.5))
    body.append(line(x2, y2, x2, y2+h2, INK, 1.5))
    plot_line(body, epochs, df.risk_specific_residual_AP.to_numpy(), xmap2, ymap2, BLUE)
    plot_line(body, epochs, df.risk_specific_residual_AP75.to_numpy(), xmap2, ymap2, PURPLE)
    plot_line(body, epochs, df.risk_specific_residual_AP_small.to_numpy(), xmap2, ymap2, ORANGE)
    for epoch, label in [(46, "max ΔAP"), (56, "risk peak"), (71, "sustained decline")]:
        xx = xmap2(epoch)
        body.append(line(xx, y2, xx, y2+h2, MUTED, 1.2, dash="4 6", opacity=0.7))
        body.append(text(xx, y2-12, label, 15, 500, anchor="middle", fill=MUTED))
    body.append(text(x2+w2/2, y2+h2+68, "global epoch", 18, 500, anchor="middle"))
    body.append(text(x2-78, y2+h2/2, "metric difference", 18, 500, anchor="middle", rotate=-90))
    for i, (label, color) in enumerate([("AP", BLUE), ("AP75", PURPLE), ("AP_small", ORANGE)]):
        xx = 1280 + i*190
        body.append(rect(xx, 905, 24, 5, color))
        body.append(text(xx+35, 912, label, 17, 500))
    body.append(rect(250, 980, 1700, 64, PALE_RED, rx=8))
    body.append(text(1100, 1022, "AP_small is negative at every matched checkpoint. The short AP gain at E46 does not persist.", 23, 500, anchor="middle"))
    return save_figure(
        "figure-06-training-trajectory", width, height,
        "Matched long-horizon training trajectory",
        "Absolute AP and Risk-minus-Uniform metric differences across the saved five-epoch checkpoint grid.",
        body, [source, TABLES / "table_09_training_dynamics.csv"]
    )


def figure_intervention_semantics():
    exposure_source = TABLES / "table_10_exposure_audit.csv"
    scale_source = TABLES / "table_12_scale_transition_matrix.csv"
    context_source = TABLES / "table_11_crop_scale_context.csv"
    exposure = pd.read_csv(exposure_source).set_index("arm")
    trans = pd.read_csv(scale_source)
    context = pd.read_csv(context_source).iloc[0]
    width, height = 2250, 1180
    body = title_block(
        "Object-Centric Cropping Alters Exposure, Scale, and Context",
        "The risk arm is equal in view count but not equivalent in supervised scene content", width)
    body.append(text(120, 205, "A  Risk arm relative to Uniform", 23, 500))
    labels = ["Training views", "Unique source images", "Supervised objects", "Target exposures"]
    ratios = [
        exposure.loc["rebu_risk", "training_views"] / exposure.loc["uniform", "training_views"],
        exposure.loc["rebu_risk", "unique_source_images"] / exposure.loc["uniform", "unique_source_images"],
        exposure.loc["rebu_risk", "total_supervised_objects"] / exposure.loc["uniform", "total_supervised_objects"],
        exposure.loc["rebu_risk", "target_objects"] / exposure.loc["uniform", "target_objects"],
    ]
    x0, y0, bw = 350, 270, 570
    for i, (label, value) in enumerate(zip(labels, ratios)):
        yy = y0 + i*105
        body.append(text(x0-20, yy+27, label, 18, 500, anchor="end"))
        body.append(rect(x0, yy, bw, 46, "#EDF1F5", rx=4))
        body.append(line(x0+bw/1.8, yy-8, x0+bw/1.8, yy+54, INK, 1.5, dash="5 4"))
        body.append(rect(x0, yy, min(value/1.8, 1)*bw, 46, ORANGE if value>1 else BLUE, rx=4))
        body.append(text(x0+bw+20, yy+31, f"{value:.2f}×", 20, 500, fill=ORANGE if value>1 else BLUE))
    body.append(text(x0+bw/1.8, y0+4*105+28, "1.0×", 16, 500, anchor="middle", fill=MUTED))
    body.append(text(1040, 205, "B  Original scale → crop scale", 23, 500))
    scales = ["small", "medium", "large"]
    matrix = np.zeros((3,3))
    for _, row in trans.iterrows():
        matrix[scales.index(row.source_scale), scales.index(row.target_scale)] = row.fraction_within_source_scale
    mx, my, cell = 1110, 300, 155
    for j, label in enumerate(scales):
        body.append(text(mx+j*cell+cell/2, my-24, label.title(), 18, 500, anchor="middle"))
    for i, label in enumerate(scales):
        body.append(text(mx-22, my+i*cell+cell/2+6, label.title(), 18, 500, anchor="end"))
        for j in range(3):
            value = matrix[i,j]
            color = blend("#F5F7FA", ORANGE, value)
            body.append(rect(mx+j*cell, my+i*cell, cell-6, cell-6, color, BG, 2, 4))
            body.append(text(mx+j*cell+(cell-6)/2, my+i*cell+(cell-6)/2+7, f"{value*100:.1f}%", 21, 500, anchor="middle", fill=BG if value>0.55 else INK))
    body.append(text(mx+cell*1.5, my+cell*3+40, "crop-coordinate scale", 18, 500, anchor="middle"))
    body.append(text(mx-120, my+cell*1.5, "original scale", 18, 500, anchor="middle", rotate=-90))
    body.append(text(1700, 205, "C  Context and geometry shift", 23, 500))
    stats = [
        ("Small targets that cease", "to be small", context.original_small_ceased_fraction*100, "%"),
        ("Neighbors retained", "inside crop", context.weighted_neighbor_retention*100, "%"),
        ("Mean short-side", "amplification", context.mean_short_side_amplification, "×"),
        ("Fully removed", "context objects", int(context.fully_removed_objects), "objects"),
        ("Boundary-clipped", "context objects", int(context.boundary_clipped_objects), "objects"),
    ]
    for i, (a, b, value, unit) in enumerate(stats):
        yy = 285+i*135
        body.append(rect(1700, yy, 430, 105, "#FAFBFD", GRID, 1.5, 7))
        if unit == "%": display = f"{value:.1f}%"
        elif unit == "×": display = f"{value:.3f}×"
        else: display = f"{value:,}"
        body.append(text(1740, yy+45, display, 29, 500, fill=ORANGE if i in [0,2,3,4] else BLUE))
        body.append(multiline(1900, yy+35, [a,b], 17, 500, fill=MUTED))
    body.append(rect(180, 1010, 1890, 78, PALE_ORANGE, rx=8))
    body.append(text(1125, 1059, "More target exposure is achieved by changing the visual condition that originally produced the failure.", 24, 500, anchor="middle"))
    return save_figure(
        "figure-07-intervention-semantics", width, height,
        "Exposure, scale, and context shift induced by object-centric crops",
        "Composite audit of source exposure, scale transition, target amplification, neighbor retention, and context removal.",
        body, [exposure_source, scale_source, context_source]
    )


def figure_corruption_heatmap():
    source = PROJECT / "final-testing/corruptions/corruption_metrics.csv"
    df = pd.read_csv(source)
    models = ["standard_e20", "p2_e20", "common40", "uniform_e56", "rebu_risk_e56"]
    model_labels = ["Standard E20", "P2 E20", "Common40", "Uniform E56", "Risk E56"]
    corruptions = ["low_light", "fog", "rain", "blur", "noise"]
    columns = [(c,s) for c in corruptions for s in [1,2,3]]
    values = np.array([[float(df[(df.model==m)&(df.corruption==c)&(df.severity==s)].iloc[0].AP_small_retention) for c,s in columns] for m in models])
    width, height = 2300, 1050
    body = title_block(
        "Small-Object Robustness Across Corruption Type and Severity",
        "AP_small retention relative to each model's own canonical clean result", width)
    x0, y0, cw, rh = 440, 290, 111, 116
    vmin, vmax = values.min(), 1.0
    for group, corruption in enumerate(corruptions):
        start = x0+group*3*cw
        body.append(text(start+1.5*cw, 205, corruption.replace("_"," ").title(), 21, 500, anchor="middle"))
        body.append(line(start, 230, start+3*cw-8, 230, GRID, 3))
        for severity in [1,2,3]:
            body.append(text(start+(severity-0.5)*cw, 267, f"S{severity}", 17, 500, anchor="middle", fill=MUTED))
    for i, label in enumerate(model_labels):
        yy=y0+i*rh
        body.append(text(x0-25, yy+53, label, 20, 500, anchor="end"))
        for j,value in enumerate(values[i]):
            t=(value-vmin)/(vmax-vmin)
            color=blend("#F5E1E1", GREEN, t)
            body.append(rect(x0+j*cw, yy, cw-7, 88, color, BG, 2, 3))
            body.append(text(x0+j*cw+(cw-7)/2, yy+54, f"{value*100:.1f}%", 16, 500, anchor="middle", fill=BG if t>0.62 else INK))
    lx, ly = 760, 895
    body.append(text(lx-25, ly+24, f"{vmin*100:.0f}%", 16, 500, anchor="end", fill=MUTED))
    for k in range(20):
        body.append(rect(lx+k*42,ly,43,30,blend("#F5E1E1",GREEN,k/19)))
    body.append(text(lx+20*42+20,ly+24,"100%",16,500,fill=MUTED))
    body.append(text(1150, 975, "Blur and noise are the dominant stressors. P2 retains its small-object advantage without introducing a corruption collapse.", 22, 500, anchor="middle"))
    return save_figure(
        "figure-08-corruption-heatmap", width, height,
        "Small-object corruption robustness heatmap",
        "Heatmap of AP-small retention for five representative checkpoints under five corruption families and three severity levels.",
        body, [source, TABLES / "table_13_corruption_summary.csv"]
    )


def figure_efficiency_tradeoff():
    source = TABLES / "table_14_efficiency.csv"
    perf_source = TABLES / "table_02_architecture_clean_efficiency.csv"
    eff = pd.read_csv(source).set_index("model")
    perf = pd.read_csv(perf_source).set_index("model")
    std, p2 = eff.loc["standard_e20"], eff.loc["p2_e20"]
    width, height = 2050, 1050
    body = title_block(
        "Accuracy–Efficiency Trade-off of the P2 Intervention",
        "RTX 4090, PyTorch FP16, batch 1, 960-pixel input, 200 measured iterations per round", width)
    body.append(text(120, 210, "A  Relative deployment and training cost", 23, 500))
    labels=["Parameters","GFLOPs","Median latency","Inference memory","Training memory"]
    vals=[p2.parameters/std.parameters-1,p2.GFLOPs/std.GFLOPs-1,p2.median_forward_latency_ms/std.median_forward_latency_ms-1,p2.peak_inference_memory_MiB/std.peak_inference_memory_MiB-1,p2.peak_training_memory_MiB/std.peak_training_memory_MiB-1]
    x0,y0,w=390,280,590
    vmax=0.85
    for i,(label,value) in enumerate(zip(labels,vals)):
        yy=y0+i*112
        body.append(text(x0-20,yy+29,label,19,500,anchor="end"))
        body.append(rect(x0,yy,w,48,"#EDF1F5",rx=4))
        body.append(rect(x0,yy,w*value/vmax,48,ORANGE if value<0.5 else RED,rx=4))
        body.append(text(x0+w+20,yy+32,f"+{value*100:.1f}%",20,500,fill=ORANGE if value<0.5 else RED))
    body.append(text(1160, 210, "B  Small-object benefit versus latency", 23, 500))
    sx,sy,sw,sh=1190,300,680,500
    xmin,xmax=13.8,17.6
    ymin,ymax=0.140,0.160
    xmap=lambda v:sx+(v-xmin)/(xmax-xmin)*sw
    ymap=lambda v:sy+sh-(v-ymin)/(ymax-ymin)*sh
    for value in [14,15,16,17]:
        xx=xmap(value)
        body.append(line(xx,sy,xx,sy+sh,GRID,1))
        body.append(text(xx,sy+sh+30,str(value),16,anchor="middle",fill=MUTED))
    for value in [0.145,0.150,0.155,0.160]:
        yy=ymap(value)
        body.append(line(sx,yy,sx+sw,yy,GRID,1))
        body.append(text(sx-15,yy+6,f"{value:.3f}",16,anchor="end",fill=MUTED))
    body.append(line(sx,sy+sh,sx+sw,sy+sh,INK,1.5))
    body.append(line(sx,sy,sx,sy+sh,INK,1.5))
    points=[
        ("Standard",float(std.median_forward_latency_ms),float(perf.loc["standard_e20","AP_small"]),BLUE,13),
        ("P2",float(p2.median_forward_latency_ms),float(perf.loc["p2_e20","AP_small"]),ORANGE,17)
    ]
    body.append(line(xmap(points[0][1]),ymap(points[0][2]),xmap(points[1][1]),ymap(points[1][2]),MUTED,2,dash="7 5"))
    for label,xv,yv,color,radius in points:
        body.append(circle(xmap(xv),ymap(yv),radius,color,BG,3))
        body.append(text(xmap(xv)+(18 if label=="Standard" else -18),ymap(yv)-20,f"{label}: {yv:.4f}",19,500,anchor="start" if label=="Standard" else "end",fill=color))
    body.append(text(sx+sw/2,sy+sh+68,"median forward latency (ms)",18,500,anchor="middle"))
    body.append(text(sx-72,sy+sh/2,"AP_small",18,500,anchor="middle",rotate=-90))
    d_ap=float(perf.loc["p2_e20","AP_small"]-perf.loc["standard_e20","AP_small"])
    d_lat=float(p2.median_forward_latency_ms/std.median_forward_latency_ms-1)
    body.append(rect(1120,875,830,90,PALE_BLUE,rx=8))
    body.append(text(1535,930,f"+{d_ap:.4f} AP_small for +{d_lat*100:.1f}% median latency",24,500,anchor="middle"))
    body.append(text(120, 980, "Training memory is the largest measured cost. Latency remains within the frozen 30% guardrail on the stated hardware.", 20, fill=MUTED))
    return save_figure(
        "figure-09-efficiency-tradeoff", width, height,
        "Accuracy-efficiency trade-off of Projected P2",
        "Relative cost bars and the measured small-object accuracy versus latency trade-off for Standard and P2 models.",
        body, [source, perf_source]
    )


def figure_threshold_sensitivity():
    source = PROJECT / "final-testing/threshold-sensitivity/subset_stability.csv"
    df = pd.read_csv(source)
    subsets = ["small_jointly_missed","dfine_supported_yolo_failure","yolo_low_confidence","yolo_localization_failure","yolo_classification_failure","small_high_risk"]
    labels = ["Small jointly missed","D-FINE-supported YOLO failure","YOLO low confidence","YOLO localization failure","YOLO classification failure","Small high risk"]
    points=["low","central","high"]
    width,height=2100,1420
    body=title_block("Threshold Sensitivity of Frozen Failure Subsets","Jaccard overlap against the central YOLO and D-FINE operating point",width)
    for idx,(subset,label) in enumerate(zip(subsets,labels)):
        col,row=idx%3,idx//3
        ox,oy=120+col*660,220+row*570
        body.append(text(ox,oy,label,21,500))
        cell=118
        mx,my=ox+170,oy+85
        for j,point in enumerate(points):
            body.append(text(mx+j*cell+cell/2,my-18,point.title(),16,500,anchor="middle"))
        for i,yolo_point in enumerate(points):
            body.append(text(mx-18,my+i*cell+cell/2+5,yolo_point.title(),16,500,anchor="end"))
            for j,dfine_point in enumerate(points):
                value=float(df[(df.subset==subset)&(df.yolo_point==yolo_point)&(df.dfine_point==dfine_point)].iloc[0].jaccard_against_central)
                color=blend("#F4F7FA",BLUE,value)
                body.append(rect(mx+j*cell,my+i*cell,cell-5,cell-5,color,BG,2,3))
                body.append(text(mx+j*cell+(cell-5)/2,my+i*cell+(cell-5)/2+6,f"{value:.2f}",18,500,anchor="middle",fill=BG if value>0.58 else INK))
        body.append(text(mx+1.5*cell,my+3*cell+28,"D-FINE threshold",16,500,anchor="middle",fill=MUTED))
        body.append(text(mx-100,my+1.5*cell,"YOLO threshold",16,500,anchor="middle",rotate=-90,fill=MUTED))
    body.append(text(1050,1368,"The central cell is 1.00 by construction. Off-center cells quantify how membership changes under nearby operating points.",19,anchor="middle",fill=MUTED))
    return save_figure(
        "supplement-s1-threshold-sensitivity", width,height,
        "Threshold sensitivity of failure subset membership",
        "Six 3 by 3 heatmaps quantify Jaccard overlap with the central failure subset under neighboring YOLO and D-FINE thresholds.",
        body,[source]
    )


def figure_subset_overlap():
    source=PROJECT/"final-testing/failure-subsets/subset_overlap.csv"
    df=pd.read_csv(source)
    subsets=["small_jointly_missed","dfine_supported_yolo_failure","yolo_low_confidence","yolo_localization_failure","yolo_classification_failure","small_high_risk"]
    labels=["Small missed","D-FINE supports","Low confidence","Localization","Classification","Small high risk"]
    width,height=1650,1200
    body=title_block("Overlap Between Failure Subsets","Jaccard similarity reveals which diagnostic categories share the same objects",width)
    x0,y0,cell=560,300,118
    for j,label in enumerate(labels):
        body.append(text(x0+j*cell+cell/2,y0-30,label,16,500,anchor="end",rotate=-40))
    for i,label in enumerate(labels):
        body.append(text(x0-20,y0+i*cell+cell/2+5,label,18,500,anchor="end"))
        for j in range(len(labels)):
            row=df[(df.left_subset==subsets[i])&(df.right_subset==subsets[j])].iloc[0]
            value=float(row.jaccard)
            color=blend("#F6F8FB",PURPLE,value)
            body.append(rect(x0+j*cell,y0+i*cell,cell-5,cell-5,color,BG,2,3))
            body.append(text(x0+j*cell+(cell-5)/2,y0+i*cell+(cell-5)/2+6,f"{value:.2f}",18,500,anchor="middle",fill=BG if value>0.55 else INK))
    body.append(rect(240,1035,1170,72,PALE_BLUE,rx=8))
    body.append(text(825,1080,"Small jointly missed and small high-risk objects strongly overlap, while D-FINE-supported failures form a distinct subset.",21,500,anchor="middle"))
    return save_figure(
        "supplement-s2-failure-overlap", width,height,
        "Overlap between official-validation failure subsets",
        "Jaccard similarity matrix for the six frozen failure subsets.",
        body,[source]
    )


def figure_object_crop_transformation():
    annotation_file = PROJECT / "data-and-baselines/dataset/annotations/instances_train.json"
    selection_file = PROJECT / "final-testing/qualitative-cases/case_selection.json"
    images, annotations = load_coco(annotation_file)
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    case = selection["families"]["crop_scale_condition_shift"][0]
    details = case["scale_context"]
    annotation = annotations[int(case["annotation_id"])]
    image_record = images[int(case["image_id"])]
    source_path = dataset_image(image_record["file_name"])
    source = Image.open(source_path).convert("RGB")

    tx, ty, tw, th = [float(v) for v in annotation["bbox"]]
    crop_x1 = int(round(tx - float(details["left_context_margin_px"])))
    crop_y1 = int(round(ty - float(details["top_context_margin_px"])))
    crop_x2 = int(round(tx + tw + float(details["right_context_margin_px"])))
    crop_y2 = int(round(ty + th + float(details["bottom_context_margin_px"])))
    crop_box = (
        max(0, crop_x1),
        max(0, crop_y1),
        min(source.width, crop_x2),
        min(source.height, crop_y2),
    )

    dimmed = ImageEnhance.Brightness(source).enhance(0.42)
    dimmed.paste(source.crop(crop_box), (crop_box[0], crop_box[1]))
    scene_draw = ImageDraw.Draw(dimmed)
    scene_draw.rectangle(crop_box, outline=ORANGE, width=6)
    scene_draw.rectangle((tx, ty, tx + tw, ty + th), outline=BLUE, width=6)

    crop = source.crop(crop_box)
    crop_target = (
        tx - crop_box[0],
        ty - crop_box[1],
        tw,
        th,
    )
    crop = draw_bbox(crop, crop_target, color=BLUE, width=5)

    width, height = 2200, 1080
    body = title_block(
        "Object-Centric Cropping Changes the Supervised Condition",
        "A frozen training example reconstructed from the formal risk arm",
        width,
    )
    body += [
        text(100, 177, "(a) Native scene and crop boundary", 22, 500),
        raster_panel(100, 205, 820, 461, dimmed, fit="cover"),
        rect(100, 205, 820, 461, "none", INK, 1),
        text(100, 700, "Blue: target object", 18, fill=MUTED),
        text(315, 700, "Brown: retained crop", 18, fill=MUTED),
        text(1015, 177, "(b) Materialized object-centric view", 22, 500),
        raster_panel(1015, 205, 820, 461, crop, fit="cover"),
        rect(1015, 205, 820, 461, "none", INK, 1),
        text(1015, 700, "The crop is later resized to the same 960-pixel network input.", 18, fill=MUTED),
        line(950, 380, 985, 380, INK, 2, marker="arrow"),
        text(1880, 177, "(c) What changed", 22, 500),
    ]
    metrics = [
        ("Short side at network input", "18.75 px  →  39.47 px"),
        ("Short-side amplification", "2.105×"),
        ("Source objects retained", "26  →  7"),
        ("Neighbors retained", "6 of 25  (24.0%)"),
        ("Fully removed objects", "15"),
        ("Boundary-clipped objects", "5"),
        ("Scene area retained", "22.56%"),
    ]
    x_label, x_value, y0 = 1880, 2130, 240
    for index, (label, value) in enumerate(metrics):
        y = y0 + index * 72
        body.append(text(x_label, y, label, 17, fill=MUTED))
        body.append(text(x_value, y + 27, value, 20, 500, anchor="end", fill=INK))
        body.append(line(x_label, y + 42, x_value, y + 42, GRID, 1))
    body += [
        line(100, 790, 2100, 790, GRID, 1.2),
        text(100, 835, "The sample budget is unchanged, but the visual evidence is not equivalent.", 24, 500),
        multiline(
            100,
            882,
            [
                "The target becomes larger and easier to resolve, while most surrounding objects and scene relations disappear.",
                "This is the concrete mechanism behind the scale and context shift measured across all 19,500 crops.",
            ],
            20,
            fill=MUTED,
        ),
    ]
    return save_figure(
        "figure-10-object-crop-transformation",
        width,
        height,
        "Object-centric crop transformation",
        "A real formal-training crop showing simultaneous target enlargement and context removal.",
        body,
        [annotation_file, selection_file, source_path],
    )


def figure_controlled_corruption_examples():
    annotation_file = PROJECT / "data-and-baselines/dataset/annotations/instances_val.json"
    corruption_source = PROJECT / "source-code/utilities/shared-components/controlled_corruptions.py"
    images, annotations = load_coco(annotation_file)
    annotation = annotations[65381]
    image_record = images[int(annotation["image_id"])]
    source_path = dataset_image(image_record["file_name"])
    source = Image.open(source_path).convert("RGB")

    spec = importlib.util.spec_from_file_location("frozen_controlled_corruptions", corruption_source)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    _, crop_geometry = context_crop(source, annotation["bbox"], width_fraction=0.55)
    cx, cy, cw, ch = crop_geometry
    variants = [("Clean", "Unmodified input", source)]
    labels = [
        ("Low light S2", "gamma 1.60, gain 0.78", "low_light"),
        ("Fog S2", "transmission 0.68", "fog"),
        ("Rain S2", "density 0.00035", "rain"),
        ("Blur S2", "Gaussian radius 2.0", "blur"),
        ("Noise S2", "Gaussian sigma 10.0", "noise"),
    ]
    for label, parameter, corruption in labels:
        transformed = module.apply_corruption(
            source,
            image_key=image_record["file_name"],
            corruption=corruption,
            severity=2,
        )
        variants.append((label, parameter, transformed))

    relative_bbox = (
        annotation["bbox"][0] - cx,
        annotation["bbox"][1] - cy,
        annotation["bbox"][2],
        annotation["bbox"][3],
    )
    rendered = []
    for label, parameter, image in variants:
        cropped = image.crop((cx, cy, cx + cw, cy + ch))
        cropped = draw_bbox(cropped, relative_bbox, color=BLUE, width=5)
        rendered.append((label, parameter, cropped))

    width, height = 2200, 1260
    body = title_block(
        "Controlled Corruption Protocol at Severity 2",
        "The same validation scene is transformed without changing image geometry or annotations",
        width,
    )
    panel_w, panel_h = 620, 349
    xs = [110, 790, 1470]
    ys = [210, 690]
    for index, (label, parameter, image) in enumerate(rendered):
        col, row = index % 3, index // 3
        x, y = xs[col], ys[row]
        body.append(text(x, y - 22, f"({chr(97 + index)}) {label}", 22, 500))
        body.append(raster_panel(x, y, panel_w, panel_h, image, fit="cover"))
        body.append(rect(x, y, panel_w, panel_h, "none", INK, 1))
        body.append(text(x, y + panel_h + 30, parameter, 17, fill=MUTED))
    body += [
        line(110, 1130, 2090, 1130, GRID, 1.2),
        text(
            110,
            1175,
            "All five corruptions use the frozen deterministic implementation and preserve the original bounding-box geometry.",
            21,
            500,
        ),
    ]
    return save_figure(
        "figure-11-controlled-corruption-examples",
        width,
        height,
        "Controlled corruption examples",
        "One real validation scene under the five frozen corruption families at severity two.",
        body,
        [annotation_file, corruption_source, source_path],
    )


def figure_qualitative_repair_transfer():
    qualitative_root = PROJECT / "final-testing/qualitative-cases/figures"
    p2_path = qualitative_root / "p2_repairs_small_failure.png"
    risk_path = qualitative_root / "risk_non_transfer_in_original_scene.png"
    p2_sheet = Image.open(p2_path).convert("RGB")
    risk_sheet = Image.open(risk_path).convert("RGB")
    p2_row = p2_sheet.crop((0, 46, p2_sheet.width, 310))
    risk_row = risk_sheet.crop((0, 46, risk_sheet.width, 310))

    width, height = 2400, 900
    body = title_block(
        "Qualitative Evidence for Repair and Non-Transfer",
        "Exact model outputs from deterministic official-validation case selection",
        width,
    )
    body += [
        text(90, 178, "(a) Architecture intervention repairs a baseline small-object miss", 22, 500),
        raster_panel(90, 205, 2220, 244, p2_row, fit="contain"),
        rect(90, 205, 2220, 244, "none", INK, 1),
        text(90, 500, "(b) Risk-guided crop training does not transfer for the same failure condition", 22, 500),
        raster_panel(90, 527, 2220, 244, risk_row, fit="contain"),
        rect(90, 527, 2220, 244, "none", INK, 1),
        line(90, 817, 2310, 817, GRID, 1.2),
        text(
            90,
            855,
            "The images support the quantitative result: preserving the native scene can repair a miss, while extra crop exposure alone does not guarantee transfer.",
            19,
            fill=MUTED,
        ),
    ]
    return save_figure(
        "figure-12-qualitative-repair-transfer",
        width,
        height,
        "Qualitative repair and transfer comparison",
        "Exact qualitative rows contrasting a P2 repair with a risk-data non-transfer case.",
        body,
        [p2_path, risk_path, PROJECT / "final-testing/qualitative-cases/case_selection.json"],
    )


def figure_matching_taxonomy_protocol():
    operating_points = (
        PROJECT
        / "risk-data-intervention/risk-labels/operating-points/operating_points.json"
    )
    failure_metadata = PROJECT / "final-testing/failure-subsets/metadata.json"
    matcher_source = (
        PROJECT
        / "source-code/data-preparation/detection-matching/build_detection_matches.py"
    )
    thresholds = json.loads(operating_points.read_text(encoding="utf-8"))["models"]
    metadata = json.loads(failure_metadata.read_text(encoding="utf-8"))

    yolo_threshold = float(thresholds["YOLO11m"]["operating_threshold"])
    dfine_threshold = float(thresholds["D-FINE-M"]["operating_threshold"])
    subset_counts = metadata["subset_counts"]

    width, height = 2300, 1240
    body = title_block(
        "From Detector Outputs to Object-Level Failure Types",
        "Frozen operating points, one-to-one matching, and official-validation subset construction",
        width,
    )

    column_x = [100, 830, 1560]
    column_w = [570, 570, 640]
    column_titles = [
        "A  Evidence inputs",
        "B  Object-level matching",
        "C  Failure assignment",
    ]
    for x, w, title_value in zip(column_x, column_w, column_titles):
        body.append(text(x, 185, title_value, 23, 500))
        body.append(rect(x, 215, w, 540, "#FAFAFA", GRID, 1.2))

    input_rows = [
        ("Ground truth", "185,523 objects in 10,000 images", INK),
        ("YOLO11m", f"score threshold = {yolo_threshold:.3f}", BLUE),
        ("D-FINE-M", f"score threshold = {dfine_threshold:.3f}", ORANGE),
    ]
    for index, (label, detail, color) in enumerate(input_rows):
        y = 285 + index * 135
        body.append(rect(145, y, 480, 94, BG, color, 2))
        body.append(text(175, y + 35, label, 21, 500, fill=color))
        body.append(text(175, y + 68, detail, 17, fill=MUTED))
    body.append(text(385, 710, "Predictions retain class, score, and box geometry", 17, anchor="middle", fill=MUTED))

    matching_steps = [
        ("1", "Apply calibrated score threshold"),
        ("2", "Match within image and category"),
        ("3", "Greedy one-to-one assignment"),
        ("4", "Accept detection at IoU ≥ 0.50"),
    ]
    for index, (number, label) in enumerate(matching_steps):
        y = 270 + index * 103
        body.append(circle(900, y + 25, 22, NAVY))
        body.append(text(900, y + 32, number, 17, 500, anchor="middle", fill=BG))
        body.append(text(945, y + 31, label, 19, 500))
        if index < len(matching_steps) - 1:
            body.append(line(900, y + 49, 900, y + 80, GRID, 1.5, marker="arrow"))
    body.append(rect(875, 670, 480, 58, PALE_ORANGE, ORANGE, 1.2))
    body.append(text(1115, 706, "Localization band: 0.10 ≤ IoU < 0.50", 18, 500, anchor="middle"))

    status_rows = [
        ("Detected", "same class, score passes, IoU ≥ 0.50", GREEN),
        ("Low confidence", "same class and IoU ≥ 0.50, score fails", BLUE),
        ("Localization", "same class and 0.10 ≤ IoU < 0.50", ORANGE),
        ("Classification", "wrong class and IoU ≥ 0.50", PURPLE),
        ("Missed", "no qualifying prediction", RED),
    ]
    for index, (label, detail, color) in enumerate(status_rows):
        y = 255 + index * 92
        body.append(rect(1600, y, 560, 70, BG, color, 1.8))
        body.append(text(1625, y + 29, label, 19, 500, fill=color))
        body.append(text(1625, y + 54, detail, 15, fill=MUTED))

    body.append(line(100, 815, 2200, 815, GRID, 1.2))
    body.append(text(100, 860, "D  Frozen failure subsets on official validation", 23, 500))
    subsets = [
        ("Small jointly missed", subset_counts["small_jointly_missed"]["objects"]),
        ("D-FINE-supported YOLO failure", subset_counts["dfine_supported_yolo_failure"]["objects"]),
        ("YOLO low confidence", subset_counts["yolo_low_confidence"]["objects"]),
        ("YOLO localization failure", subset_counts["yolo_localization_failure"]["objects"]),
        ("YOLO classification failure", subset_counts["yolo_classification_failure"]["objects"]),
        ("Small high risk", subset_counts["small_high_risk"]["objects"]),
    ]
    for index, (label, count) in enumerate(subsets):
        x = 100 + (index % 3) * 720
        y = 900 + (index // 3) * 120
        body.append(rect(x, y, 650, 86, "#FAFAFA", GRID, 1.1))
        body.append(text(x + 24, y + 34, label, 18, 500))
        body.append(text(x + 626, y + 35, f"{count:,}", 22, 500, anchor="end", fill=NAVY))
        body.append(text(x + 626, y + 64, "objects", 15, anchor="end", fill=MUTED))

    return save_figure(
        "supplement-s3-matching-taxonomy-protocol",
        width,
        height,
        "Object-level matching and failure taxonomy protocol",
        "The frozen procedure that converts paired detector outputs into typed object-level failures and six official-validation subsets.",
        body,
        [operating_points, failure_metadata, matcher_source],
    )


def figure_equal_budget_supervision():
    source = TABLES / "table_10_exposure_audit.csv"
    df = pd.read_csv(source).set_index("arm")
    uniform = df.loc["uniform"]
    risk = df.loc["rebu_risk"]

    width, height = 2250, 1160
    body = title_block(
        "Equal View Budget Does Not Imply Equal Supervision",
        "Formal Uniform and Risk arms both contain 70,000 training views but expose different visual evidence",
        width,
    )

    body.append(text(100, 190, "A  Training-view composition", 23, 500))
    x0, bar_w, bar_h = 300, 720, 90
    scale = bar_w / 70000
    bars = [
        ("Uniform", int(uniform.original_views), int(uniform.object_centric_views), 300),
        ("Risk", int(risk.original_views), int(risk.object_centric_views), 480),
    ]
    for label, originals, crops, y in bars:
        body.append(text(x0 - 30, y + 55, label, 22, 500, anchor="end"))
        body.append(rect(x0, y, originals * scale, bar_h, BLUE, "none"))
        if crops:
            body.append(rect(x0 + originals * scale, y, crops * scale, bar_h, ORANGE, "none"))
        body.append(rect(x0, y, bar_w, bar_h, "none", INK, 1.2))
        body.append(text(x0 + originals * scale / 2, y + 55, f"{originals:,} original", 19, 500, anchor="middle", fill=BG))
        if crops:
            body.append(text(x0 + originals * scale + crops * scale / 2, y + 55, f"{crops:,} crops", 18, 500, anchor="middle", fill=BG))
        body.append(text(x0 + bar_w + 20, y + 55, "70,000 views", 19, 500))
    body.append(rect(300, 625, 18, 18, BLUE))
    body.append(text(330, 641, "Original full-scene view", 17, fill=MUTED))
    body.append(rect(565, 625, 18, 18, ORANGE))
    body.append(text(595, 641, "Object-centric crop", 17, fill=MUTED))

    body.append(text(1220, 190, "B  Resulting supervision", 23, 500))
    comparisons = [
        ("Unique source images", int(uniform.unique_source_images), int(risk.unique_source_images), "images"),
        ("Supervised objects", int(uniform.total_supervised_objects), int(risk.total_supervised_objects), "objects"),
        ("Target-object exposures", int(uniform.target_objects), int(risk.target_objects), "exposures"),
    ]
    for index, (label, uniform_value, risk_value, unit) in enumerate(comparisons):
        y = 265 + index * 210
        maximum = max(uniform_value, risk_value)
        chart_w = 690
        body.append(text(1220, y, label, 20, 500))
        body.append(text(1220, y + 48, "Uniform", 17, fill=MUTED))
        body.append(rect(1330, y + 23, chart_w * uniform_value / maximum, 34, BLUE, "none"))
        body.append(text(2050, y + 50, f"{uniform_value:,}", 18, 500, anchor="end"))
        body.append(text(1220, y + 101, "Risk", 17, fill=MUTED))
        body.append(rect(1330, y + 76, chart_w * risk_value / maximum, 34, ORANGE, "none"))
        body.append(text(2050, y + 103, f"{risk_value:,}", 18, 500, anchor="end"))
        delta = risk_value / uniform_value - 1
        body.append(text(2075, y + 77, f"{delta:+.1%}", 18, 500, fill=GREEN if delta > 0 else RED))
        body.append(text(1220, y + 136, unit, 15, fill=MUTED))

    body.append(line(100, 930, 2150, 930, GRID, 1.2))
    body.append(text(100, 980, "Controlled quantity", 18, 500, fill=NAVY))
    body.append(text(310, 980, "70,000 network inputs per arm", 19))
    body.append(text(100, 1035, "Changed quantities", 18, 500, fill=ORANGE))
    body.append(text(310, 1035, "source diversity, object count, target scale, surrounding context, and exposure frequency", 19))
    body.append(text(100, 1100, "The comparison isolates an intervention package, not a single scalar increase in hard-example frequency.", 20, 500))

    return save_figure(
        "supplement-s4-equal-budget-supervision",
        width,
        height,
        "Equal-budget supervision comparison",
        "Matched view counts conceal differences in source diversity, object supervision, and target exposure between the two formal training arms.",
        body,
        [source],
    )


def write_captions(records):
    captions = [
        "# Publication Figure Suite", "",
        "All figures are derived from frozen project evidence. No training or inference was performed.", "",
    ]
    for idx, record in enumerate(records, 1):
        label = f"Figure {idx}" if idx <= 12 else f"Supplementary Figure S{idx-12}"
        captions += [
            f"## {label}. {record['title']}", "", record["description"], "",
            f"SVG: `{Path(record['svg']).name}`", "", "Data sources:", "",
        ]
        for source in record["sources"]:
            captions.append(f"- `{source}`")
        captions.append("")
    (OUT / "FIGURE_CAPTIONS_AND_PROVENANCE.md").write_text("\n".join(captions), encoding="utf-8")


def write_manifest(records):
    files=[]
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "figure-manifest.json":
            files.append({
                "path": str(path.relative_to(OUT)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    manifest={
        "status":"complete", "figure_count":len(records), "main_figures":12, "supplementary_figures":4,
        "new_training":False, "new_inference":False, "project_source_read_only":True,
        "figures":records, "files":files,
    }
    (OUT/"figure-manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records=[
        figure_research_pipeline(), figure_failure_structure(), figure_p2_architecture(),
        figure_method_gate(), figure_bootstrap_forest(), figure_trajectory(),
        figure_intervention_semantics(), figure_corruption_heatmap(), figure_efficiency_tradeoff(),
        figure_object_crop_transformation(), figure_controlled_corruption_examples(),
        figure_qualitative_repair_transfer(),
        figure_threshold_sensitivity(), figure_subset_overlap(),
        figure_matching_taxonomy_protocol(), figure_equal_budget_supervision(),
    ]
    write_captions(records)
    write_manifest(records)
    print(json.dumps({"status":"complete","output":str(OUT),"figures":len(records)},ensure_ascii=False))


if __name__ == "__main__":
    main()
