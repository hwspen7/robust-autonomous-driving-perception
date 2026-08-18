from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    wait,
)
from pathlib import Path

import numpy as np

import run_final_representative_corruptions_v1 as v1


V1_SCRIPT = Path(v1.__file__).resolve()
EXPECTED_V1_SHA256 = (
    "62b434c7781769d4382b19f134413df245758a2e2daced0e09134506542ed5c7"
)
EXPECTED_ORIGINAL_EXPORTER_SHA256 = (
    "ce8f84f3ac4e687cdf9127fa2a538e81606005b8329c763e6fe66df579a3f44e"
)
EXPECTED_FAST_EXPORTER_SHA256 = (
    "baeaafbe1e2a1520eb0fd18364603363c1369b8cd8021567f51275aa94682969"
)

RUNTIME_ROOT = (
    v1.TEMP_ROOT / "_executor_v2"
)
FAST_EXPORTER = (
    RUNTIME_ROOT
    / "export_yolo11m_controlled_corruptions_v2.py"
)
RUNTIME_CORRUPTION_SOURCE = (
    RUNTIME_ROOT / "controlled_corruptions.py"
)
EQUIVALENCE_ROOT = (
    RUNTIME_ROOT / "equivalence"
)
EQUIVALENCE_RECORD = (
    v1.WORK / "executor_v2_equivalence.json"
)


def parse_args():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )
    group.add_argument(
        "--preflight-only",
        action="store_true",
    )
    group.add_argument(
        "--execute",
        action="store_true",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--corruption-workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--evaluation-workers",
        type=int,
        default=2,
    )

    return parser.parse_args()


def sha256_bytes(payload: bytes):
    return hashlib.sha256(payload).hexdigest()


def replace_once(
    source: str,
    old: str,
    new: str,
):
    count = source.count(old)

    if count != 1:
        raise AssertionError({
            "replacement_count": count,
            "old_prefix": old[:120],
        })

    return source.replace(old, new)


def derive_fast_exporter_source():
    source = v1.EXPORTER.read_text(
        encoding="utf-8"
    )

    actual_original = sha256_bytes(
        source.encode("utf-8")
    )

    if (
        actual_original
        != EXPECTED_ORIGINAL_EXPORTER_SHA256
    ):
        raise AssertionError({
            "original_exporter_sha256":
                actual_original,
        })

    source = replace_once(
        source,
        "import time\n"
        "from collections import Counter\n",
        "import time\n"
        "from concurrent.futures import "
        "ThreadPoolExecutor\n"
        "from collections import Counter\n",
    )

    source = replace_once(
        source,
        '''    parser.add_argument(
        "--conf",
        type=float,
''',
        '''    parser.add_argument(
        "--corruption-workers",
        type=int,
        default=8,
        help=(
            "Parallel CPU workers for deterministic corruption "
            "generation. Output order remains fixed."
        ),
    )

    parser.add_argument(
        "--conf",
        type=float,
''',
    )

    source = replace_once(
        source,
        "\n\ndef main() -> None:\n",
        '''

def iter_prefetched_corrupted_batches(
    *,
    items: list[dict[str, Any]],
    batch_size: int,
    workers: int,
    dataset_root: Path,
    corruption: str,
    severity: int,
):
    """Parallel deterministic corruption with one-batch prefetch.

    Results are consumed in submission order. Image order, random
    seeds, corrupted pixels, and prediction order remain fixed.
    """
    batches = iter(iter_batches(items, batch_size))

    def submit_batch(pool, batch_infos):
        return [
            pool.submit(
                load_and_corrupt_image,
                dataset_root=dataset_root,
                image_info=image_info,
                corruption=corruption,
                severity=severity,
            )
            for image_info in batch_infos
        ]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            current_infos = next(batches)
        except StopIteration:
            return

        current_futures = submit_batch(
            pool,
            current_infos,
        )

        while True:
            current_outputs = [
                future.result()
                for future in current_futures
            ]

            try:
                next_infos = next(batches)
            except StopIteration:
                next_infos = None
                next_futures = None
            else:
                next_futures = submit_batch(
                    pool,
                    next_infos,
                )

            yield current_infos, current_outputs

            if next_infos is None:
                break

            current_infos = next_infos
            current_futures = next_futures


def main() -> None:
''',
    )

    source = replace_once(
        source,
        '''    if args.max_det <= 0:
        raise ValueError(
            "--max-det must be positive."
        )
''',
        '''    if args.corruption_workers <= 0:
        raise ValueError(
            "--corruption-workers must be positive."
        )

    if args.max_det <= 0:
        raise ValueError(
            "--max-det must be positive."
        )
''',
    )

    source = replace_once(
        source,
        '''    print(f"batch       : {args.batch}")
    print(f"conf        : {args.conf}")
''',
        '''    print(f"batch       : {args.batch}")
    print(f"CPU workers : {args.corruption_workers}")
    print(f"prefetch    : one batch")
    print(f"JSON writes : one compact payload per batch")
    print(f"conf        : {args.conf}")
''',
    )

    source = replace_once(
        source,
        '''            for batch_infos in iter_batches(
                images,
                args.batch,
            ):
''',
        '''            for batch_infos, batch_outputs in (
                iter_prefetched_corrupted_batches(
                    items=images,
                    batch_size=args.batch,
                    workers=args.corruption_workers,
                    dataset_root=dataset_root,
                    corruption=args.corruption,
                    severity=args.severity,
                )
            ):
''',
    )

    source = replace_once(
        source,
        '''                corruption_seeds: list[int] = []

                # --------------------------------------------
''',
        '''                corruption_seeds: list[int] = []

                batch_predictions: list[dict[str, Any]] = []

                # --------------------------------------------
''',
    )

    source = replace_once(
        source,
        '''                for image_info in batch_infos:
                    corrupted_image, seed = (
                        load_and_corrupt_image(
                            dataset_root=dataset_root,
                            image_info=image_info,
                            corruption=args.corruption,
                            severity=args.severity,
                        )
                    )

                    corrupted_images.append(
                        corrupted_image
                    )

                    corruption_seeds.append(
                        seed
                    )
''',
        '''                for corrupted_image, seed in batch_outputs:
                    corrupted_images.append(
                        corrupted_image
                    )

                    corruption_seeds.append(
                        seed
                    )
''',
    )

    source = replace_once(
        source,
        '''                            if not first_prediction:
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

                            total_predictions += 1
''',
        '''                            batch_predictions.append(
                                prediction
                            )

                            total_predictions += 1
''',
    )

    source = replace_once(
        source,
        '''                # \u663e\u5f0f\u91ca\u653e\u5f53\u524d batch \u7684 PIL \u5f15\u7528。
                corrupted_images.clear()
''',
        '''                if batch_predictions:
                    serialized = json.dumps(
                        batch_predictions,
                        separators=(",", ":"),
                        allow_nan=False,
                    )[1:-1]

                    if not first_prediction:
                        prediction_file.write(",")

                    prediction_file.write(serialized)
                    first_prediction = False

                # \u663e\u5f0f\u91ca\u653e\u5f53\u524d batch \u7684 PIL \u5f15\u7528。
                corrupted_images.clear()
                batch_predictions.clear()
''',
    )

    source = replace_once(
        source,
        '''            "batch": args.batch,
            "conf": args.conf,
''',
        '''            "batch": args.batch,
            "corruption_workers": args.corruption_workers,
            "prefetch_batches": 1,
            "parallel_corruption_preserves_submission_order": True,
            "prediction_json_write_mode": "one_compact_payload_per_batch",
            "conf": args.conf,
''',
    )

    payload = source.encode("utf-8")
    actual = sha256_bytes(payload)

    if actual != EXPECTED_FAST_EXPORTER_SHA256:
        raise AssertionError({
            "derived_exporter_sha256": actual,
            "expected":
                EXPECTED_FAST_EXPORTER_SHA256,
        })

    return source


def materialize_runtime_exporter():
    source = derive_fast_exporter_source()

    RUNTIME_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = FAST_EXPORTER.with_suffix(
        ".py.tmp"
    )
    temporary.write_text(
        source,
        encoding="utf-8",
    )
    os.replace(temporary, FAST_EXPORTER)

    if (
        sha256_bytes(FAST_EXPORTER.read_bytes())
        != EXPECTED_FAST_EXPORTER_SHA256
    ):
        raise AssertionError(FAST_EXPORTER)

    if RUNTIME_CORRUPTION_SOURCE.exists():
        if (
            RUNTIME_CORRUPTION_SOURCE.resolve()
            != v1.CORRUPTION_SOURCE.resolve()
        ):
            raise AssertionError(
                RUNTIME_CORRUPTION_SOURCE
            )
    else:
        RUNTIME_CORRUPTION_SOURCE.symlink_to(
            v1.CORRUPTION_SOURCE
        )


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(path)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def run_exporter(
    *,
    exporter: Path,
    checkpoint: Path,
    output_dir: Path,
    corruption: str,
    severity: int,
    batch: int,
    limit: int,
    corruption_workers: int | None,
):
    command = [
        sys.executable,
        "-u",
        str(exporter),
        "--model",
        str(checkpoint),
        "--dataset-root",
        str(v1.DATASET),
        "--annotation",
        str(v1.GT),
        "--output-dir",
        str(output_dir),
        "--corruption",
        corruption,
        "--severity",
        str(severity),
        "--device",
        "0",
        "--imgsz",
        "960",
        "--batch",
        str(batch),
        "--conf",
        "0.001",
        "--iou",
        "0.7",
        "--max-det",
        "300",
        "--limit",
        str(limit),
        "--save-vis",
        "0",
        "--overwrite",
    ]

    if corruption_workers is not None:
        command.extend([
            "--corruption-workers",
            str(corruption_workers),
        ])

    started = time.time()

    subprocess.run(
        command,
        cwd=str(v1.PROJECT),
        check=True,
    )

    return time.time() - started


def compare_predictions(
    old_path: Path,
    new_path: Path,
):
    old = json.loads(old_path.read_text())
    new = json.loads(new_path.read_text())

    if len(old) != len(new):
        raise AssertionError(
            (len(old), len(new))
        )

    maximum_bbox_difference = 0.0
    maximum_score_difference = 0.0

    for index, (left, right) in enumerate(
        zip(old, new, strict=True)
    ):
        if (
            int(left["image_id"])
            != int(right["image_id"])
            or int(left["category_id"])
            != int(right["category_id"])
        ):
            raise AssertionError(
                (index, left, right)
            )

        bbox_difference = max(
            abs(float(a) - float(b))
            for a, b in zip(
                left["bbox"],
                right["bbox"],
                strict=True,
            )
        )

        score_difference = abs(
            float(left["score"])
            - float(right["score"])
        )

        maximum_bbox_difference = max(
            maximum_bbox_difference,
            bbox_difference,
        )
        maximum_score_difference = max(
            maximum_score_difference,
            score_difference,
        )

    if maximum_bbox_difference > 1e-6:
        raise AssertionError(
            maximum_bbox_difference
        )

    if maximum_score_difference > 1e-7:
        raise AssertionError(
            maximum_score_difference
        )

    return {
        "predictions": len(old),
        "old_prediction_sha256":
            v1.sha256_file(old_path),
        "new_prediction_sha256":
            v1.sha256_file(new_path),
        "maximum_bbox_difference":
            maximum_bbox_difference,
        "maximum_score_difference":
            maximum_score_difference,
    }


def run_equivalence_gate(
    protocol: dict,
    batch: int,
    corruption_workers: int,
):
    if EQUIVALENCE_RECORD.is_file():
        record = v1.load_json(
            EQUIVALENCE_RECORD
        )

        if (
            record.get("status") == "pass"
            and record.get(
                "fast_exporter_sha256"
            ) == EXPECTED_FAST_EXPORTER_SHA256
            and int(record.get("batch")) == batch
            and int(
                record.get(
                    "corruption_workers"
                )
            ) == corruption_workers
        ):
            print(
                "PASS: existing V2 equivalence "
                "record reused"
            )
            return record

        raise AssertionError(
            EQUIVALENCE_RECORD
        )

    original = load_module(
        v1.EXPORTER,
        "original_corruption_exporter",
    )
    fast = load_module(
        FAST_EXPORTER,
        "fast_corruption_exporter",
    )

    gt = json.loads(v1.GT.read_text())
    images = sorted(
        gt["images"],
        key=lambda row: int(row["id"]),
    )[:8]

    pixel_cases = 0

    for corruption in protocol[
        "corruption_evaluation"
    ]["corruptions"]:
        for severity in protocol[
            "corruption_evaluation"
        ]["severities"]:
            sequential = [
                original.load_and_corrupt_image(
                    dataset_root=v1.DATASET,
                    image_info=image,
                    corruption=corruption,
                    severity=severity,
                )
                for image in images
            ]

            parallel_batches = list(
                fast.iter_prefetched_corrupted_batches(
                    items=images,
                    batch_size=batch,
                    workers=corruption_workers,
                    dataset_root=v1.DATASET,
                    corruption=corruption,
                    severity=severity,
                )
            )

            parallel = [
                item
                for _, outputs in parallel_batches
                for item in outputs
            ]

            if len(sequential) != len(parallel):
                raise AssertionError(
                    (len(sequential), len(parallel))
                )

            for (
                sequential_image,
                sequential_seed,
            ), (
                parallel_image,
                parallel_seed,
            ) in zip(
                sequential,
                parallel,
                strict=True,
            ):
                if sequential_seed != parallel_seed:
                    raise AssertionError(
                        (
                            sequential_seed,
                            parallel_seed,
                        )
                    )

                if (
                    sequential_image.size
                    != parallel_image.size
                    or sequential_image.mode
                    != parallel_image.mode
                    or sequential_image.tobytes()
                    != parallel_image.tobytes()
                ):
                    raise AssertionError(
                        (
                            corruption,
                            severity,
                            pixel_cases,
                        )
                    )

                sequential_image.close()
                parallel_image.close()
                pixel_cases += 1

    if EQUIVALENCE_ROOT.exists():
        shutil.rmtree(EQUIVALENCE_ROOT)

    old_output = EQUIVALENCE_ROOT / "old"
    new_output = EQUIVALENCE_ROOT / "new"

    checkpoint = v1.MODEL_SPECS[
        "standard_e20"
    ]["checkpoint"]

    old_seconds = run_exporter(
        exporter=v1.EXPORTER,
        checkpoint=checkpoint,
        output_dir=old_output,
        corruption="blur",
        severity=1,
        batch=batch,
        limit=64,
        corruption_workers=None,
    )
    new_seconds = run_exporter(
        exporter=FAST_EXPORTER,
        checkpoint=checkpoint,
        output_dir=new_output,
        corruption="blur",
        severity=1,
        batch=batch,
        limit=64,
        corruption_workers=corruption_workers,
    )

    prediction_equivalence = compare_predictions(
        old_output / "predictions.json",
        new_output / "predictions.json",
    )

    old_summary = json.loads(
        (
            old_output / "image_summary.json"
        ).read_text()
    )
    new_summary = json.loads(
        (
            new_output / "image_summary.json"
        ).read_text()
    )

    if old_summary != new_summary:
        raise AssertionError(
            "Image-summary equivalence failed."
        )

    record = {
        "version":
            "corruption_executor_v2_equivalence",
        "status": "pass",
        "created_utc": v1.now_utc(),
        "original_exporter_sha256":
            EXPECTED_ORIGINAL_EXPORTER_SHA256,
        "fast_exporter_sha256":
            EXPECTED_FAST_EXPORTER_SHA256,
        "pixel_equivalence_cases":
            pixel_cases,
        "prediction_equivalence":
            prediction_equivalence,
        "image_summary_exact_match":
            True,
        "batch": batch,
        "corruption_workers":
            corruption_workers,
        "old_64_image_seconds":
            old_seconds,
        "new_64_image_seconds":
            new_seconds,
    }

    v1.write_json_atomic(
        EQUIVALENCE_RECORD,
        record,
    )

    shutil.rmtree(EQUIVALENCE_ROOT)

    print(
        "PASS: V2 pixel and prediction "
        "equivalence established"
    )
    print(
        "pixel_cases:",
        pixel_cases,
    )
    print(
        "old_64_image_seconds:",
        round(old_seconds, 3),
    )
    print(
        "new_64_image_seconds:",
        round(new_seconds, 3),
    )

    return record


def evaluation_worker(payload: dict):
    benchmark = v1.load_benchmark_module()
    benchmark.GT_PATH = v1.GT
    benchmark.CLEAN = {
        payload["model"]:
            payload["clean_metrics"]
    }

    return benchmark.evaluate(
        model=payload["model"],
        variant=payload["variant"],
        corruption=payload["corruption"],
        severity=payload["severity"],
        prediction_path=Path(
            payload["prediction_path"]
        ),
        gt_ids=payload["gt_ids"],
        gt_count_by_class=payload[
            "gt_counts"
        ],
        inference_seconds=payload[
            "inference_seconds"
        ],
    )


def preflight(
    batch: int,
    corruption_workers: int,
    evaluation_workers: int,
):
    if corruption_workers <= 0:
        raise ValueError(
            corruption_workers
        )

    if evaluation_workers <= 0:
        raise ValueError(
            evaluation_workers
        )

    actual_v1 = v1.sha256_file(V1_SCRIPT)

    print("=== V2 EXECUTOR PREFLIGHT ===")
    print("V1 script:", V1_SCRIPT)
    print(" expected:", EXPECTED_V1_SHA256)
    print(" actual  :", actual_v1)

    if actual_v1 != EXPECTED_V1_SHA256:
        raise AssertionError(V1_SCRIPT)

    protocol, clean = v1.preflight(batch)

    fast_source = derive_fast_exporter_source()
    fast_hash = sha256_bytes(
        fast_source.encode("utf-8")
    )

    if fast_hash != EXPECTED_FAST_EXPORTER_SHA256:
        raise AssertionError(fast_hash)

    print("derived_fast_exporter_sha256:", fast_hash)
    print("corruption_workers:", corruption_workers)
    print("evaluation_workers:", evaluation_workers)
    print(
        "PASS: optimized executor is derivable "
        "from the frozen exporter"
    )
    print(
        "NOTHING CREATED, MODIFIED, "
        "INFERRED OR TRAINED"
    )

    return protocol, clean


def execute(
    protocol: dict,
    clean_metrics: dict,
    batch: int,
    corruption_workers: int,
    evaluation_workers: int,
):
    v1.WORK.mkdir(
        parents=True,
        exist_ok=True,
    )

    materialize_runtime_exporter()

    run_equivalence_gate(
        protocol=protocol,
        batch=batch,
        corruption_workers=
            corruption_workers,
    )

    benchmark = v1.load_benchmark_module()
    benchmark.GT_PATH = v1.GT
    benchmark.CLEAN = clean_metrics
    benchmark.VARIANTS = [
        (corruption, severity)
        for corruption in protocol[
            "corruption_evaluation"
        ]["corruptions"]
        for severity in protocol[
            "corruption_evaluation"
        ]["severities"]
    ]

    (
        gt_ids,
        gt_id_set,
        gt_counts,
    ) = benchmark.validate_gt()

    (v1.WORK / "variant_records").mkdir(
        parents=True,
        exist_ok=True,
    )
    v1.TEMP_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_json = (
        v1.WORK / "corruption_metrics.json"
    )
    class_json = (
        v1.WORK
        / "corruption_per_class_metrics.json"
    )
    progress_path = (
        v1.WORK / "progress.json"
    )

    metrics = benchmark.load_existing(
        metric_json
    )
    per_class = benchmark.load_existing(
        class_json
    )

    if progress_path.is_file():
        progress = v1.load_json(
            progress_path
        )
    else:
        progress = {
            "version":
                "representative_corruption_progress_v2",
            "status": "running",
            "protocol_sha256":
                v1.EXPECTED_SHA256[v1.PROTOCOL],
            "batch": batch,
            "completed": {},
            "created_utc": v1.now_utc(),
        }

    if int(progress["batch"]) != batch:
        raise AssertionError({
            "existing_batch": progress["batch"],
            "requested_batch": batch,
        })

    progress["executor_version"] = (
        "parallel_corruption_pipelined_eval_v2"
    )
    progress["corruption_workers"] = (
        corruption_workers
    )
    progress["evaluation_workers"] = (
        evaluation_workers
    )
    progress["updated_utc"] = v1.now_utc()

    benchmark.save_outputs(
        v1.WORK,
        metrics,
        per_class,
        progress,
    )

    pending = {}
    total = 75
    run_index = 0

    def persist_future(future):
        nonlocal metrics, per_class

        context = pending.pop(future)
        metric_row, class_rows = (
            future.result()
        )

        model_key = context["model"]
        variant = context["variant"]

        metrics = benchmark.replace_rows(
            metrics,
            model_key,
            variant,
            [metric_row],
        )
        per_class = benchmark.replace_rows(
            per_class,
            model_key,
            variant,
            class_rows,
        )

        record = {
            "version":
                "representative_corruption_condition_record_v2",
            "status": "complete",
            "model": model_key,
            "display_name":
                v1.MODEL_SPECS[
                    model_key
                ]["display_name"],
            "variant": variant,
            "metric": metric_row,
            "exporter_metadata":
                context["exporter_metadata"],
            "temporary_prediction_deleted":
                True,
            "executor_version":
                context["executor_version"],
            "saved_utc": v1.now_utc(),
        }

        v1.write_json_atomic(
            v1.WORK
            / "variant_records"
            / f"{model_key}_{variant}.json",
            record,
        )

        progress["completed"][
            f"{model_key}::{variant}"
        ] = {
            "status": "done",
            "AP": metric_row["AP"],
            "AP_small":
                metric_row["AP_small"],
            "prediction_sha256":
                metric_row[
                    "prediction_sha256"
                ],
            "updated_utc": v1.now_utc(),
        }
        progress["updated_utc"] = (
            v1.now_utc()
        )

        benchmark.save_outputs(
            v1.WORK,
            metrics,
            per_class,
            progress,
        )

        v1.safe_remove_temp(
            context["temp_dir"]
        )

        print(
            f"[EVALUATED] {model_key} "
            f"{variant} "
            f"AP={metric_row['AP']:.6f} "
            f"APs={metric_row['AP_small']:.6f}",
            flush=True,
        )

    def collect_completed(
        block: bool,
    ):
        if not pending:
            return

        if block:
            done, _ = wait(
                list(pending),
                return_when=FIRST_COMPLETED,
            )
        else:
            done = {
                future
                for future in pending
                if future.done()
            }

        for future in list(done):
            persist_future(future)

    with ProcessPoolExecutor(
        max_workers=evaluation_workers
    ) as evaluation_pool:
        for model_key, spec in (
            v1.MODEL_SPECS.items()
        ):
            for (
                corruption,
                severity,
            ) in benchmark.VARIANTS:
                run_index += 1
                variant = benchmark.vname(
                    corruption,
                    severity,
                )

                collect_completed(
                    block=False
                )

                if benchmark.has_metric(
                    metrics,
                    model_key,
                    variant,
                ):
                    temp_dir = (
                        v1.TEMP_ROOT
                        / f"{model_key}_{variant}"
                    )

                    if temp_dir.exists():
                        v1.safe_remove_temp(
                            temp_dir
                        )

                    print(
                        f"[SKIP {run_index}/{total}] "
                        f"{model_key} {variant}"
                    )
                    continue

                while (
                    len(pending)
                    >= evaluation_workers
                ):
                    collect_completed(
                        block=True
                    )

                temp_dir = (
                    v1.TEMP_ROOT
                    / f"{model_key}_{variant}"
                )
                request_path = (
                    temp_dir / "request.json"
                )

                base_request = {
                    "model": model_key,
                    "checkpoint_sha256":
                        spec["sha256"],
                    "corruption": corruption,
                    "severity": severity,
                    "image_size": 960,
                    "batch": batch,
                    "confidence_floor": 0.001,
                    "nms_iou": 0.7,
                    "max_det": 300,
                }

                fast_request = {
                    **base_request,
                    "executor_version":
                        "parallel_corruption_v2",
                    "exporter_sha256":
                        EXPECTED_FAST_EXPORTER_SHA256,
                    "corruption_workers":
                        corruption_workers,
                }

                reusable = False
                request = None

                if request_path.is_file():
                    request = v1.load_json(
                        request_path
                    )

                    reusable = (
                        request
                        in (
                            base_request,
                            fast_request,
                        )
                        and (
                            temp_dir
                            / "predictions.json"
                        ).is_file()
                        and (
                            temp_dir
                            / "image_summary.json"
                        ).is_file()
                        and (
                            temp_dir
                            / "metadata.json"
                        ).is_file()
                    )

                if reusable:
                    print(
                        f"[RESUME TEMP "
                        f"{run_index}/{total}] "
                        f"{model_key} {variant}"
                    )

                    exporter_metadata = (
                        v1.load_json(
                            temp_dir
                            / "metadata.json"
                        )
                    )
                    inference_seconds = float(
                        exporter_metadata[
                            "elapsed_seconds"
                        ]
                    )
                    executor_version = (
                        "sequential_v1_reused"
                        if request == base_request
                        else "parallel_v2_reused"
                    )
                else:
                    if temp_dir.exists():
                        v1.safe_remove_temp(
                            temp_dir
                        )

                    temp_dir.mkdir(
                        parents=True
                    )

                    v1.write_json_atomic(
                        request_path,
                        fast_request,
                    )

                    progress["completed"][
                        f"{model_key}::{variant}"
                    ] = {
                        "status":
                            "inference_started_v2",
                        "updated_utc":
                            v1.now_utc(),
                    }
                    progress["updated_utc"] = (
                        v1.now_utc()
                    )

                    benchmark.save_outputs(
                        v1.WORK,
                        metrics,
                        per_class,
                        progress,
                    )

                    print()
                    print("=" * 100)
                    print(
                        f"FAST INFERENCE "
                        f"{run_index}/{total}: "
                        f"{model_key} {variant}"
                    )
                    print("=" * 100)

                    inference_seconds = (
                        run_exporter(
                            exporter=FAST_EXPORTER,
                            checkpoint=
                                spec["checkpoint"],
                            output_dir=temp_dir,
                            corruption=corruption,
                            severity=severity,
                            batch=batch,
                            limit=0,
                            corruption_workers=
                                corruption_workers,
                        )
                    )

                    exporter_metadata = (
                        v1.load_json(
                            temp_dir
                            / "metadata.json"
                        )
                    )
                    executor_version = (
                        "parallel_corruption_v2"
                    )

                (
                    prediction_path,
                    _,
                    _,
                    exporter_metadata,
                ) = v1.validate_export(
                    temp_dir=temp_dir,
                    model_key=model_key,
                    corruption=corruption,
                    severity=severity,
                    batch=batch,
                    benchmark=benchmark,
                    gt_id_set=gt_id_set,
                )

                payload = {
                    "model": model_key,
                    "variant": variant,
                    "corruption": corruption,
                    "severity": severity,
                    "prediction_path":
                        str(prediction_path),
                    "gt_ids": gt_ids,
                    "gt_counts": gt_counts,
                    "inference_seconds":
                        inference_seconds,
                    "clean_metrics":
                        clean_metrics[model_key],
                }

                future = (
                    evaluation_pool.submit(
                        evaluation_worker,
                        payload,
                    )
                )

                pending[future] = {
                    "model": model_key,
                    "variant": variant,
                    "temp_dir": temp_dir,
                    "exporter_metadata":
                        exporter_metadata,
                    "executor_version":
                        executor_version,
                }

                print(
                    f"[QUEUED COCOEVAL] "
                    f"{model_key} {variant} "
                    f"pending={len(pending)}",
                    flush=True,
                )

        while pending:
            collect_completed(
                block=True
            )

    if len(metrics) != 75:
        raise AssertionError(len(metrics))

    if len(per_class) != 750:
        raise AssertionError(len(per_class))

    remaining_predictions = list(
        v1.TEMP_ROOT.rglob(
            "predictions.json"
        )
    )

    if remaining_predictions:
        raise AssertionError(
            remaining_predictions
        )

    summary = v1.save_derived_outputs(
        benchmark,
        metrics,
        clean_metrics,
    )

    progress["status"] = "complete"
    progress["completed_conditions"] = 75
    progress["updated_utc"] = v1.now_utc()

    benchmark.save_outputs(
        v1.WORK,
        metrics,
        per_class,
        progress,
    )

    metadata = {
        "version":
            "final_representative_corruptions_executor_v2",
        "status": "complete",
        "created_utc": v1.now_utc(),
        "script":
            str(Path(__file__).resolve()),
        "script_sha256":
            v1.sha256_file(
                Path(__file__).resolve()
            ),
        "parent_v1_script_sha256":
            EXPECTED_V1_SHA256,
        "original_exporter_sha256":
            EXPECTED_ORIGINAL_EXPORTER_SHA256,
        "derived_fast_exporter_sha256":
            EXPECTED_FAST_EXPORTER_SHA256,
        "equivalence_record":
            str(EQUIVALENCE_RECORD),
        "configuration": {
            "conditions": 75,
            "image_size": 960,
            "batch": batch,
            "corruption_workers":
                corruption_workers,
            "evaluation_workers":
                evaluation_workers,
            "corruption_prefetch_batches": 1,
            "prediction_json_write_mode":
                "one_compact_payload_per_batch",
            "confidence_floor": 0.001,
            "nms_iou": 0.7,
            "native_max_det": 300,
            "coco_max_det": 100,
        },
        "execution": {
            "parallel_cpu_corruption": True,
            "gpu_cpu_prefetch_overlap": True,
            "gpu_inference_cocoeval_pipeline":
                True,
            "new_training": False,
        },
    }

    v1.write_json_atomic(
        v1.WORK / "benchmark_metadata.json",
        metadata,
    )

    if RUNTIME_ROOT.exists():
        shutil.rmtree(RUNTIME_ROOT)

    manifest_path, manifest_sha = (
        v1.build_manifest(v1.WORK)
    )

    os.replace(
        v1.WORK,
        v1.OUTPUT,
    )

    if v1.TEMP_ROOT.exists():
        try:
            v1.TEMP_ROOT.rmdir()
        except OSError:
            pass

    print()
    print("=" * 100)
    print(
        "FAST REPRESENTATIVE CORRUPTION "
        "EVALUATION COMPLETE"
    )
    print("=" * 100)

    for model_key, row in (
        summary["models"].items()
    ):
        print(
            model_key,
            "mean_AP=",
            round(
                row[
                    "mean_corruption_AP"
                ],
                6,
            ),
            "mean_APs=",
            round(
                row[
                    "mean_corruption_AP_small"
                ],
                6,
            ),
        )

    print("output:", v1.OUTPUT)
    print(
        "manifest:",
        v1.OUTPUT / manifest_path.name,
    )
    print("manifest_sha256:", manifest_sha)
    print("temporary_predictions_remaining: 0")
    print("NOTHING TRAINED")
    print("NEXT: frozen efficiency evaluation")
    print("=" * 100)


def main():
    args = parse_args()

    protocol, clean = preflight(
        batch=args.batch,
        corruption_workers=
            args.corruption_workers,
        evaluation_workers=
            args.evaluation_workers,
    )

    if args.preflight_only:
        return

    execute(
        protocol=protocol,
        clean_metrics=clean,
        batch=args.batch,
        corruption_workers=
            args.corruption_workers,
        evaluation_workers=
            args.evaluation_workers,
    )


if __name__ == "__main__":
    main()