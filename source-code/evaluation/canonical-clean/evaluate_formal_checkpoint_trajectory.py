from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


REPO = Path(
    "/root/rivermind-data/autodrive/code/"
    "robust-autonomous-driving-perception"
)

RESULT_ROOT = Path(
    "/root/rivermind-data/autodrive/results/evaluation"
)

FORMAL_ROOT = RESULT_ROOT / "formal_training"

OUTPUT_ROOT = FORMAL_ROOT / "trajectory_evaluation_v1"
METRICS_DIR = OUTPUT_ROOT / "metrics"
WORK_DIR = OUTPUT_ROOT / "temporary_predictions"

DATASET_ROOT = Path(
    "/root/rivermind-data/autodrive/datasets/datasets/"
    "bdd100k_final"
)

VAL_GT = (
    DATASET_ROOT
    / "coco/annotations/instances_val.json"
)

EXPORTER = REPO / "scripts/export_yolo11m_predictions.py"

FORMAL_MANIFEST = (
    FORMAL_ROOT
    / "training_v2/formal_branches_manifest.json"
)

COMMON40 = (
    FORMAL_ROOT
    / "training_v1/common_uniform_e1_40/"
    "weights/common40_ema.pt"
)

UNIFORM_DIR = (
    FORMAL_ROOT
    / "training_v2/uniform_e41_100/weights"
)

REBU_DIR = (
    FORMAL_ROOT
    / "training_v2/rebu_risk_e41_100/weights"
)

EXPECTED_FIXED_HASHES = {
    str(VAL_GT):
        "7be8a02e147743123a4a62b2c92ed2e4"
        "a5712d261c8868223e6325bad6e346d6",

    str(EXPORTER):
        "974efcdb71dbbe4fff980cae27ebb7cf8"
        "26a630bb9d5c2afd654ed3b36f23166",

    str(FORMAL_MANIFEST):
        "64dd9fdfc15db3fae8881a63f8625489"
        "9187f6f6570b6bfa89129799bb589cd0",

    str(COMMON40):
        "c05ed70a608d026fef6178f40a4dee62"
        "e96323731700b42e426c311dd78b27f3",
}

UNIFORM_HASHES = {
    41: "c7796cc7e0e0d9f03968e2fe33c40426723e089ec1c3b453dd5f9fbf2b11914e",
    46: "84e4572cadabf4717363d9fabab2e715c8733fe71f581188a04cfe50cca3712f",
    51: "948228b888eac4c0e92f348e28ef054cf2e7b6544076d299ea7ca44ebf714a30",
    56: "84aa86771d185c2c72fa277c609c1222d7a417a8197523c4a5ced2a6a91ebb1b",
    61: "0d89feceeff48acae4ef2d01f9ed9b75598ba5b57f7252e41dcfa673c2000e71",
    66: "45de3cb7d78221347941b03b2af5b05ed758a580732c369ee30578321989c416",
    71: "aff8681ca8bf72ceb43ff2d5d5dc07dc62aef93e6d6a2ec0f0b6e180db33a040",
    76: "421ec7084e4021189e7217c6b30eb093308ec04f7b80047e366db3e8f30479da",
    81: "dcbcd19063039372dd4087ca29a311cc256fd3d1f0be026ac03967eabd62f0b0",
    86: "69a9a07ad1dd8948376166f96d09bcd852b80aa5b748217343e264cf597d3116",
    91: "488e9f682857c5608f96305c5bea7fa900aa9bbd28357ff17564e01fae1d4f95",
    96: "55a588a24c803448bb1b82bac50f7a542c3a8965aaf454133808cedf8d94ffce",
    100: "2af91824191b0d48d40e35856abafd388776815aaf0a629ace4e734a9be01cf7",
}

REBU_HASHES = {
    41: "e2e2f112a2afd192d5025a4260837ec911aad510106d3ed3d31d4e54198bcd15",
    46: "e3320e3eac16911177f485416390b1a8cee2c11ce7ccddfa529a65e76f4d8716",
    51: "bae318237b4e1849da492b84c38b503c752189504e07ea07ba99e7d701402421",
    56: "879b179c219914a83775cacb7074f1d7bd9836971fae24592a6a16d6e87fe102",
    61: "45d80fb25404c8665d3e5b2ccdb726391333df248b067c24b890a33f13044883",
    66: "50db193e0dbf874231234ff86590e343381ccd86f0c5fe4758dfff821018c532",
    71: "6e272abf3d823a07de7ac4890616430af2e794ca43096cb71cc063a55930a0a7",
    76: "298957c95a33052b8b1cd0bde030268d49384385eb30f61db467c01b9aba9d30",
    81: "c4c546d51c538f1a84062f8540e51f2e27f659f15261cd986dce16ab3afcccf2",
    86: "5f5d0f946fbe67fd4b671b9364d0e86fad20807cd25a51bc1589c89cd1d171d7",
    91: "6f80c4263ce6f22ad83e868997dae963eea8837252618436513a3cdd6f3907c1",
    96: "4247e393ef0854275442296c261a094a45cac5e016ca7d2d8856a49e116afcfe",
    100: "d85a864b7f84d0f408a68697cf9f4cbbfb73a9b821bb8ca392ba49c2956c447a",
}

CLASS_NAMES = {
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

EPOCHS = [
    41, 46, 51, 56, 61, 66, 71,
    76, 81, 86, 91, 96, 100,
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every recoverable Common40/Uniform/Rebu "
            "checkpoint without training."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run inference and COCO evaluation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted trajectory evaluation.",
    )
    parser.add_argument(
        "--keep-predictions",
        action="store_true",
        help="Retain large prediction JSON files.",
    )
    return parser.parse_args()


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_suffix(
        path.suffix + ".incomplete"
    )
    temporary.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ) + "\n"
    )
    os.replace(
        temporary,
        path,
    )


def checkpoint_path(arm: str, epoch: int):
    if arm == "uniform":
        directory = UNIFORM_DIR
        final_name = "uniform100_ema.pt"
    elif arm == "rebu_risk":
        directory = REBU_DIR
        final_name = "rebu_yolo100_ema.pt"
    else:
        raise ValueError(arm)

    if epoch == 100:
        return directory / final_name

    stored_epoch = epoch - 41
    return directory / f"epoch{stored_epoch}.pt"


def build_candidates():
    candidates = [{
        "key": "common_e40",
        "arm": "common",
        "global_epoch": 40,
        "stored_epoch": -1,
        "path": str(COMMON40),
        "sha256": EXPECTED_FIXED_HASHES[str(COMMON40)],
    }]

    for epoch in EPOCHS:
        candidates.append({
            "key": f"uniform_e{epoch}",
            "arm": "uniform",
            "global_epoch": epoch,
            "stored_epoch": (
                -1 if epoch == 100
                else epoch - 41
            ),
            "path": str(
                checkpoint_path(
                    "uniform",
                    epoch,
                )
            ),
            "sha256": UNIFORM_HASHES[epoch],
        })

        candidates.append({
            "key": f"rebu_risk_e{epoch}",
            "arm": "rebu_risk",
            "global_epoch": epoch,
            "stored_epoch": (
                -1 if epoch == 100
                else epoch - 41
            ),
            "path": str(
                checkpoint_path(
                    "rebu_risk",
                    epoch,
                )
            ),
            "sha256": REBU_HASHES[epoch],
        })

    return candidates


def normalize_names(names):
    if isinstance(names, list):
        return {
            index: name
            for index, name in enumerate(names)
        }

    return {
        int(key): value
        for key, value in names.items()
    }


def inspect_checkpoint(candidate):
    path = Path(candidate["path"])

    if not path.is_file():
        raise FileNotFoundError(path)

    actual_hash = sha256(path)
    expected_hash = candidate["sha256"]

    print(path)
    print(" expected:", expected_hash)
    print(" actual  :", actual_hash)

    if actual_hash != expected_hash:
        raise AssertionError(path)

    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    stored_epoch = checkpoint.get("epoch")

    if candidate["global_epoch"] < 100:
        if candidate["arm"] != "common":
            if stored_epoch != candidate["stored_epoch"]:
                raise AssertionError(
                    (
                        candidate["key"],
                        stored_epoch,
                        candidate["stored_epoch"],
                    )
                )

    model = (
        checkpoint.get("ema")
        or checkpoint.get("model")
    )

    if model is None:
        raise AssertionError(
            f"No EMA/model object in {path}"
        )

    strides = [
        float(value)
        for value in model.stride.detach().cpu().tolist()
    ]
    if strides != [4.0, 8.0, 16.0, 32.0]:
        raise AssertionError(
            (candidate["key"], strides)
        )

    names = normalize_names(model.names)
    expected_names = {
        index - 1: name
        for index, name in CLASS_NAMES.items()
    }

    if names != expected_names:
        raise AssertionError(
            (candidate["key"], names)
        )

    del model
    del checkpoint
    gc.collect()


def verify_fixed_inputs():
    print("=== FIXED INPUT HASH CHECK ===")

    for raw_path, expected in (
        EXPECTED_FIXED_HASHES.items()
    ):
        path = Path(raw_path)

        if not path.is_file():
            raise FileNotFoundError(path)

        actual = sha256(path)

        print(path)
        print(" expected:", expected)
        print(" actual  :", actual)

        if actual != expected:
            raise AssertionError(path)


def validate_ground_truth():
    data = json.loads(
        VAL_GT.read_text()
    )

    images = data["images"]
    annotations = data["annotations"]
    categories = {
        int(row["id"]): row["name"]
        for row in data["categories"]
    }

    if len(images) != 10000:
        raise AssertionError(len(images))

    if len(annotations) != 185523:
        raise AssertionError(len(annotations))

    if categories != CLASS_NAMES:
        raise AssertionError(categories)

    print("GT images:", len(images))
    print("GT annotations:", len(annotations))
    print("GT categories:", categories)


def protocol_data(candidates):
    return {
        "version":
            "formal_checkpoint_trajectory_protocol_v1",
        "status":
            "frozen_before_evaluation",
        "purpose":
            "Recover the best saved checkpoint and diagnose "
            "late-training degradation without retraining.",
        "training_allowed": False,
        "official_validation": {
            "path": str(VAL_GT),
            "sha256": EXPECTED_FIXED_HASHES[
                str(VAL_GT)
            ],
            "images": 10000,
            "annotations": 185523,
        },
        "exporter": {
            "path": str(EXPORTER),
            "sha256": EXPECTED_FIXED_HASHES[
                str(EXPORTER)
            ],
        },
        "evaluation": {
            "image_size": 960,
            "batch": 1,
            "confidence_floor": 0.001,
            "nms_iou": 0.7,
            "export_max_det": 300,
            "coco_max_dets": [1, 10, 100],
            "primary_metric": "AP",
            "tie_tolerance": 0.0005,
            "tie_break": (
                "earliest epoch among checkpoints "
                "within 0.0005 of maximum AP"
            ),
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def prepare_output(protocol, resume):
    protocol_path = (
        OUTPUT_ROOT
        / "trajectory_protocol.json"
    )

    if OUTPUT_ROOT.exists():
        if not resume:
            raise FileExistsError(
                "Output exists. Use --resume only after "
                f"inspection: {OUTPUT_ROOT}"
            )

        if not protocol_path.is_file():
            raise AssertionError(
                "Existing output has no frozen protocol."
            )

        existing = json.loads(
            protocol_path.read_text()
        )

        if existing != protocol:
            raise AssertionError(
                "Frozen protocol mismatch."
            )
    else:
        OUTPUT_ROOT.mkdir(
            parents=True,
            exist_ok=False,
        )
        METRICS_DIR.mkdir()
        WORK_DIR.mkdir()

        atomic_json(
            protocol_path,
            protocol,
        )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return protocol_path


def run_export(candidate):
    key = candidate["key"]
    checkpoint = Path(candidate["path"])
    candidate_work = WORK_DIR / key
    predictions_path = (
        candidate_work
        / "predictions.json"
    )

    if predictions_path.is_file():
        print(
            "Using existing completed predictions:",
            predictions_path,
        )
        return predictions_path

    if candidate_work.exists():
        raise RuntimeError(
            "Incomplete prediction directory requires "
            f"manual inspection: {candidate_work}"
        )

    command = [
        sys.executable,
        "-u",
        str(EXPORTER),
        "--model",
        str(checkpoint),
        "--dataset-root",
        str(DATASET_ROOT),
        "--annotation",
        str(VAL_GT),
        "--output-dir",
        str(candidate_work),
        "--device",
        "0",
        "--imgsz",
        "960",
        "--batch",
        "1",
        "--conf",
        "0.001",
        "--iou",
        "0.7",
        "--max-det",
        "300",
        "--limit",
        "0",
        "--save-vis",
        "0",
    ]

    print()
    print("=" * 100)
    print("EXPORTING:", key)
    print("CHECKPOINT:", checkpoint)
    print("=" * 100)

    subprocess.run(
        command,
        cwd=REPO,
        check=True,
    )

    if not predictions_path.is_file():
        raise FileNotFoundError(
            predictions_path
        )

    return predictions_path


def mean_valid(values):
    values = np.asarray(values)
    values = values[values > -1]

    if values.size == 0:
        return None

    return float(values.mean())


def evaluate_predictions(
    ground_truth,
    candidate,
    predictions_path,
):
    print()
    print("=" * 100)
    print("COCO EVALUATION:", candidate["key"])
    print("=" * 100)

    prediction_sha = sha256(
        predictions_path
    )

    with predictions_path.open("r") as handle:
        predictions = json.load(handle)

    if not isinstance(predictions, list):
        raise TypeError(
            "Predictions must be a JSON list."
        )

    if len(predictions) == 0:
        raise AssertionError(
            "Prediction file is empty."
        )

    prediction_image_ids = {
        int(row["image_id"])
        for row in predictions
    }

    unknown_image_ids = (
        prediction_image_ids
        - set(ground_truth.getImgIds())
    )
    if unknown_image_ids:
        raise AssertionError(
            sorted(unknown_image_ids)[:20]
        )

    detection_results = ground_truth.loadRes(
        predictions
    )

    evaluator = COCOeval(
        ground_truth,
        detection_results,
        "bbox",
    )
    evaluator.params.imgIds = sorted(
        ground_truth.getImgIds()
    )
    evaluator.params.catIds = sorted(
        ground_truth.getCatIds()
    )
    evaluator.params.maxDets = [
        1,
        10,
        100,
    ]

    started = time.time()

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    elapsed_minutes = (
        time.time() - started
    ) / 60.0

    stats = evaluator.stats.tolist()

    result = {
        "key": candidate["key"],
        "arm": candidate["arm"],
        "global_epoch":
            candidate["global_epoch"],
        "checkpoint":
            candidate["path"],
        "checkpoint_sha256":
            candidate["sha256"],
        "prediction_sha256":
            prediction_sha,
        "predictions":
            len(predictions),
        "images_with_predictions":
            len(prediction_image_ids),
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR1": float(stats[6]),
        "AR10": float(stats[7]),
        "AR100": float(stats[8]),
        "AR_small": float(stats[9]),
        "AR_medium": float(stats[10]),
        "AR_large": float(stats[11]),
        "elapsed_minutes":
            elapsed_minutes,
        "per_class": {},
    }

    precision = evaluator.eval["precision"]
    category_ids = evaluator.params.catIds

    for category_index, category_id in enumerate(
        category_ids
    ):
        all_iou = precision[
            :,
            :,
            category_index,
            0,
            2,
        ]
        iou_50 = precision[
            0,
            :,
            category_index,
            0,
            2,
        ]
        iou_75 = precision[
            5,
            :,
            category_index,
            0,
            2,
        ]

        result["per_class"][
            str(category_id)
        ] = {
            "class_name":
                CLASS_NAMES[category_id],
            "AP":
                mean_valid(all_iou),
            "AP50":
                mean_valid(iou_50),
            "AP75":
                mean_valid(iou_75),
        }

    del predictions
    del detection_results
    del evaluator
    gc.collect()

    return result


def load_results():
    results = []

    for path in sorted(
        METRICS_DIR.glob("*.json")
    ):
        results.append(
            json.loads(path.read_text())
        )

    return results


def select_best(rows):
    if not rows:
        return None

    maximum_ap = max(
        row["AP"]
        for row in rows
    )

    tied = [
        row
        for row in rows
        if maximum_ap - row["AP"] <= 0.0005
    ]

    selected = min(
        tied,
        key=lambda row: row["global_epoch"],
    )

    return {
        "key": selected["key"],
        "arm": selected["arm"],
        "global_epoch":
            selected["global_epoch"],
        "checkpoint":
            selected["checkpoint"],
        "checkpoint_sha256":
            selected["checkpoint_sha256"],
        "AP": selected["AP"],
        "AP50": selected["AP50"],
        "AP75": selected["AP75"],
        "AP_small": selected["AP_small"],
        "AP_medium": selected["AP_medium"],
        "AP_large": selected["AP_large"],
        "AR100": selected["AR100"],
        "maximum_observed_AP":
            maximum_ap,
        "tie_tolerance": 0.0005,
    }


def sustained_decline(rows):
    if len(rows) < 2:
        return None

    ordered = sorted(
        rows,
        key=lambda row: row["global_epoch"],
    )
    peak = max(
        ordered,
        key=lambda row: row["AP"],
    )
    threshold = peak["AP"] - 0.002

    later = [
        row
        for row in ordered
        if row["global_epoch"]
        > peak["global_epoch"]
    ]

    for index, row in enumerate(later):
        remaining = later[index:]

        if (
            row["AP"] <= threshold
            and all(
                candidate["AP"] <= threshold
                for candidate in remaining
            )
        ):
            return {
                "peak_epoch":
                    peak["global_epoch"],
                "peak_AP":
                    peak["AP"],
                "first_sustained_decline_epoch":
                    row["global_epoch"],
                "decline_threshold":
                    0.002,
            }

    return {
        "peak_epoch":
            peak["global_epoch"],
        "peak_AP":
            peak["AP"],
        "first_sustained_decline_epoch":
            None,
        "decline_threshold":
            0.002,
    }


def write_csv_outputs(results):
    ordered = sorted(
        results,
        key=lambda row: (
            row["global_epoch"],
            row["arm"],
        ),
    )

    metric_fields = [
        "key",
        "arm",
        "global_epoch",
        "checkpoint",
        "checkpoint_sha256",
        "prediction_sha256",
        "predictions",
        "images_with_predictions",
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large",
        "elapsed_minutes",
    ]

    metric_path = (
        OUTPUT_ROOT
        / "trajectory_metrics.csv"
    )
    metric_temp = metric_path.with_suffix(
        ".csv.incomplete"
    )

    with metric_temp.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=metric_fields,
        )
        writer.writeheader()

        for row in ordered:
            writer.writerow({
                field: row.get(field)
                for field in metric_fields
            })

    os.replace(
        metric_temp,
        metric_path,
    )

    per_class_path = (
        OUTPUT_ROOT
        / "trajectory_per_class_metrics.csv"
    )
    per_class_temp = per_class_path.with_suffix(
        ".csv.incomplete"
    )

    per_class_fields = [
        "key",
        "arm",
        "global_epoch",
        "category_id",
        "class_name",
        "AP",
        "AP50",
        "AP75",
    ]

    with per_class_temp.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=per_class_fields,
        )
        writer.writeheader()

        for row in ordered:
            for category_id, values in (
                row["per_class"].items()
            ):
                writer.writerow({
                    "key": row["key"],
                    "arm": row["arm"],
                    "global_epoch":
                        row["global_epoch"],
                    "category_id":
                        int(category_id),
                    "class_name":
                        values["class_name"],
                    "AP": values["AP"],
                    "AP50": values["AP50"],
                    "AP75": values["AP75"],
                })

    os.replace(
        per_class_temp,
        per_class_path,
    )

    uniform = {
        row["global_epoch"]: row
        for row in results
        if row["arm"] == "uniform"
    }
    rebu = {
        row["global_epoch"]: row
        for row in results
        if row["arm"] == "rebu_risk"
    }

    matched = []

    delta_fields = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]

    for epoch in sorted(
        set(uniform) & set(rebu)
    ):
        row = {
            "global_epoch": epoch,
            "uniform_key":
                uniform[epoch]["key"],
            "rebu_key":
                rebu[epoch]["key"],
        }

        for field in delta_fields:
            row[f"uniform_{field}"] = (
                uniform[epoch][field]
            )
            row[f"rebu_{field}"] = (
                rebu[epoch][field]
            )
            row[f"delta_{field}"] = (
                rebu[epoch][field]
                - uniform[epoch][field]
            )

        matched.append(row)

    matched_path = (
        OUTPUT_ROOT
        / "matched_rebu_minus_uniform.csv"
    )
    matched_temp = matched_path.with_suffix(
        ".csv.incomplete"
    )

    if matched:
        with matched_temp.open(
            "w",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(
                    matched[0].keys()
                ),
            )
            writer.writeheader()
            writer.writerows(matched)

        os.replace(
            matched_temp,
            matched_path,
        )

    return matched


def update_summary(results, expected_count):
    matched = write_csv_outputs(results)

    common_rows = [
        row for row in results
        if row["arm"] == "common"
    ]
    uniform_rows = [
        row for row in results
        if row["arm"] == "uniform"
    ]
    rebu_rows = [
        row for row in results
        if row["arm"] == "rebu_risk"
    ]

    best_overall = select_best(results)
    best_uniform = select_best(uniform_rows)
    best_rebu = select_best(rebu_rows)

    best_rebu_delta_ap = None
    if matched:
        row = max(
            matched,
            key=lambda item: item["delta_AP"],
        )
        best_rebu_delta_ap = {
            key: value
            for key, value in row.items()
            if (
                key == "global_epoch"
                or key.startswith("delta_")
            )
        }

    best_rebu_delta_small = None
    if matched:
        row = max(
            matched,
            key=lambda item:
                item["delta_AP_small"],
        )
        best_rebu_delta_small = {
            key: value
            for key, value in row.items()
            if (
                key == "global_epoch"
                or key.startswith("delta_")
            )
        }

    summary = {
        "version":
            "formal_checkpoint_trajectory_summary_v1",
        "status": (
            "complete"
            if len(results) == expected_count
            else "running"
        ),
        "completed_candidates":
            len(results),
        "expected_candidates":
            expected_count,
        "best_common":
            select_best(common_rows),
        "best_uniform":
            best_uniform,
        "best_rebu":
            best_rebu,
        "best_overall":
            best_overall,
        "best_rebu_minus_uniform_by_AP":
            best_rebu_delta_ap,
        "best_rebu_minus_uniform_by_AP_small":
            best_rebu_delta_small,
        "uniform_decline":
            sustained_decline(uniform_rows),
        "rebu_decline":
            sustained_decline(rebu_rows),
        "interpretation_limit": (
            "This selects the best recoverable saved "
            "checkpoint on a five-epoch grid. It does not "
            "recover unsaved epochs."
        ),
    }

    atomic_json(
        OUTPUT_ROOT
        / "trajectory_summary.json",
        summary,
    )

    return summary


def build_manifest():
    artifacts = []

    for path in sorted(
        OUTPUT_ROOT.rglob("*")
    ):
        if not path.is_file():
            continue

        if path.name == "artifact_manifest.json":
            continue

        if path.name == "predictions.json":
            continue

        artifacts.append({
            "path": str(
                path.relative_to(OUTPUT_ROOT)
            ),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })

    manifest = {
        "version":
            "formal_checkpoint_trajectory_artifacts_v1",
        "status": "complete",
        "artifacts": artifacts,
    }

    path = (
        OUTPUT_ROOT
        / "artifact_manifest.json"
    )
    atomic_json(
        path,
        manifest,
    )

    return path, sha256(path)


def main():
    args = parse_args()
    candidates = build_candidates()

    verify_fixed_inputs()
    validate_ground_truth()

    print()
    print("=== CHECKPOINT PREFLIGHT ===")

    for candidate in candidates:
        inspect_checkpoint(candidate)

    free_gib = (
        shutil.disk_usage(
            RESULT_ROOT
        ).free
        / 1024 ** 3
    )

    print()
    print("candidate_count:", len(candidates))
    print("free_GiB:", round(free_gib, 3))

    if free_gib < 3.0:
        raise RuntimeError(
            "At least 3 GiB free storage is required."
        )

    protocol = protocol_data(candidates)

    print()
    print("PASS: all trajectory inputs are valid")

    if not args.execute:
        print("NOTHING EVALUATED OR MODIFIED")
        print(
            "Run with --execute to start "
            "evaluation."
        )
        return

    protocol_path = prepare_output(
        protocol,
        args.resume,
    )

    ground_truth = COCO(
        str(VAL_GT)
    )

    existing_results = {
        row["key"]: row
        for row in load_results()
    }

    for candidate in candidates:
        key = candidate["key"]
        metric_path = (
            METRICS_DIR
            / f"{key}.json"
        )

        if key in existing_results:
            existing = existing_results[key]

            if (
                existing["checkpoint_sha256"]
                != candidate["sha256"]
            ):
                raise AssertionError(
                    f"Resume hash mismatch: {key}"
                )

            print("SKIP COMPLETED:", key)
            continue

        predictions_path = run_export(
            candidate
        )

        result = evaluate_predictions(
            ground_truth,
            candidate,
            predictions_path,
        )

        atomic_json(
            metric_path,
            result,
        )

        if not args.keep_predictions:
            predictions_path.unlink()
            print(
                "Removed temporary predictions:",
                predictions_path,
            )

        existing_results[key] = result

        summary = update_summary(
            list(existing_results.values()),
            len(candidates),
        )

        print()
        print(
            f"PROGRESS: {len(existing_results)}/"
            f"{len(candidates)}"
        )
        print(
            key,
            "AP=",
            round(result["AP"], 6),
            "APs=",
            round(result["AP_small"], 6),
            "AP75=",
            round(result["AP75"], 6),
        )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = load_results()

    if len(results) != len(candidates):
        raise AssertionError(
            (
                len(results),
                len(candidates),
            )
        )

    final_summary = update_summary(
        results,
        len(candidates),
    )

    manifest_path, manifest_sha = (
        build_manifest()
    )

    print()
    print("=" * 100)
    print(
        "FORMAL CHECKPOINT TRAJECTORY "
        "EVALUATION COMPLETE"
    )
    print("=" * 100)
    print(
        "best_overall:",
        json.dumps(
            final_summary["best_overall"],
            ensure_ascii=False,
            indent=2,
        ),
    )
    print(
        "best_uniform:",
        json.dumps(
            final_summary["best_uniform"],
            ensure_ascii=False,
            indent=2,
        ),
    )
    print(
        "best_rebu:",
        json.dumps(
            final_summary["best_rebu"],
            ensure_ascii=False,
            indent=2,
        ),
    )
    print(
        "uniform_decline:",
        json.dumps(
            final_summary["uniform_decline"],
            ensure_ascii=False,
            indent=2,
        ),
    )
    print(
        "rebu_decline:",
        json.dumps(
            final_summary["rebu_decline"],
            ensure_ascii=False,
            indent=2,
        ),
    )
    print("protocol:", protocol_path)
    print("output:", OUTPUT_ROOT)
    print("manifest:", manifest_path)
    print("manifest_sha256:", manifest_sha)
    print("NOTHING TRAINED")
    print("=" * 100)


if __name__ == "__main__":
    main()