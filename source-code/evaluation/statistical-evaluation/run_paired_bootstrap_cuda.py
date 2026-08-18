from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch


V1_PATH = Path(__file__).with_name(
    "run_final_paired_bootstrap_v1.py"
)
EXPECTED_V1_SHA256 = (
    "9ff688359408b0d4a41424d75d05e74643e1b7076e70199c01bfe71cfc6a5008"
)

BENCHMARK_REPLICATES = 100
EQUIVALENCE_REPLICATES = 2
GPU_BATCH_SIZE = 32
MAX_PROJECTED_GPU_HOURS = 6.0
EQUIVALENCE_TOLERANCE = 5e-10


def raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_v1():
    if not V1_PATH.is_file():
        raise FileNotFoundError(V1_PATH)

    actual = raw_sha256(V1_PATH)
    print("v1_expected_sha256:", EXPECTED_V1_SHA256)
    print("v1_actual_sha256  :", actual)

    if actual != EXPECTED_V1_SHA256:
        raise AssertionError(
            "V1 source changed; refusing an unverified compatibility run."
        )

    module_name = "run_final_paired_bootstrap_v1_frozen"

    spec = importlib.util.spec_from_file_location(
        module_name,
        V1_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(V1_PATH)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


v1 = load_v1()

ROOT = v1.ROOT
OUTPUT = v1.OUTPUT
WORK = ROOT / "paired_bootstrap_v1.work-v2"
MODEL_DIR = WORK / "model_distributions"

REPLICATES = v1.REPLICATES
EXPECTED_IMAGES = v1.EXPECTED_IMAGES
CLEAN_METRICS = v1.CLEAN_METRICS
FAILURE_METRICS = v1.FAILURE_METRICS
MODELS = tuple(v1.PREDICTIONS.keys())
GATE_MODELS = set(v1.GATE_MODELS)


def parse_args():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--preflight-only",
        action="store_true",
    )
    group.add_argument(
        "--execute",
        action="store_true",
    )

    return parser.parse_args()


def generate_weights() -> np.ndarray:
    print("\n=== GENERATING FROZEN PAIRED IMAGE WEIGHTS ===")

    rng = np.random.default_rng(v1.SEED)

    weights = np.empty(
        (REPLICATES, EXPECTED_IMAGES),
        dtype=np.uint16,
    )

    for replicate in range(REPLICATES):
        sampled = rng.integers(
            0,
            EXPECTED_IMAGES,
            size=EXPECTED_IMAGES,
            dtype=np.int32,
        )

        weights[replicate] = np.bincount(
            sampled,
            minlength=EXPECTED_IMAGES,
        ).astype(np.uint16)

        if (replicate + 1) % 200 == 0:
            print(
                f"weights={replicate + 1}/{REPLICATES}",
                flush=True,
            )

    if not np.all(
        weights.sum(axis=1, dtype=np.int64)
        == EXPECTED_IMAGES
    ):
        raise AssertionError("Invalid bootstrap multiplicities.")

    return weights


def weights_sha256(weights: np.ndarray) -> str:
    return hashlib.sha256(weights.tobytes()).hexdigest()


def build_all_caches(
    canonical_rows: dict[str, dict],
    failure_ids: dict[str, set[int]],
):
    print("\n=== BUILDING FROZEN COCO EVALUATION CACHES ===")

    gt_data = json.loads(v1.GT.read_text())

    image_ids = sorted(
        int(image["id"])
        for image in gt_data["images"]
    )

    if len(image_ids) != EXPECTED_IMAGES:
        raise AssertionError(len(image_ids))

    subset_datasets = {}

    for subset, ids in failure_ids.items():
        annotations = [
            annotation
            for annotation in gt_data["annotations"]
            if int(annotation["id"]) in ids
        ]

        subset_datasets[subset] = {
            "images": gt_data["images"],
            "categories": gt_data["categories"],
            "annotations": annotations,
            "info": gt_data.get("info", {}),
            "licenses": gt_data.get("licenses", []),
        }

    clean_caches = {}
    failure_caches = {}

    original_weights = np.ones(
        EXPECTED_IMAGES,
        dtype=np.uint16,
    )

    for model, prediction_path in v1.PREDICTIONS.items():
        print("\n" + "=" * 100)
        print("CACHE MODEL:", model)
        print("=" * 100)

        predictions = json.loads(
            prediction_path.read_text()
        )

        clean_cache = v1.build_evaluation_cache(
            gt_data,
            predictions,
            (
                "all",
                "small",
                "medium",
                "large",
            ),
        )

        reconstructed = v1.clean_metrics_from_cache(
            clean_cache,
            original_weights,
        )

        for metric_index, metric in enumerate(CLEAN_METRICS):
            expected = float(canonical_rows[model][metric])
            actual = float(reconstructed[metric_index])

            if not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=2e-10,
            ):
                raise AssertionError(
                    (
                        "canonical_reconstruction",
                        model,
                        metric,
                        expected,
                        actual,
                    )
                )

        clean_caches[model] = clean_cache

        print("PASS canonical reconstruction:", model)

        if model in GATE_MODELS:
            failure_caches[model] = {}

            for subset in v1.FAILURE_SUBSETS:
                print(
                    "failure cache:",
                    model,
                    subset,
                )

                failure_caches[model][subset] = (
                    v1.build_evaluation_cache(
                        subset_datasets[subset],
                        predictions,
                        ("all",),
                    )
                )

        del predictions
        gc.collect()

    return clean_caches, failure_caches


def scalar_reference(
    clean_caches: dict,
    failure_caches: dict,
    weights: np.ndarray,
) -> dict:
    print("\n=== BUILDING V1 SCALAR REFERENCE ===")

    v1.GLOBAL_CLEAN_CACHES = clean_caches
    v1.GLOBAL_FAILURE_CACHES = failure_caches
    v1.GLOBAL_WEIGHTS = weights

    tasks = [
        (model, replicate)
        for model in MODELS
        for replicate in range(EQUIVALENCE_REPLICATES)
    ]

    references = {}

    context = mp.get_context("fork")

    with context.Pool(
        processes=min(32, len(tasks)),
        maxtasksperchild=1,
    ) as pool:
        for model, replicate, clean, failure in (
            pool.imap_unordered(
                v1.bootstrap_worker,
                tasks,
                chunksize=1,
            )
        ):
            references[(model, replicate)] = {
                "clean": clean,
                "failure": failure,
            }

    if len(references) != len(tasks):
        raise AssertionError(
            (len(references), len(tasks))
        )

    print(
        "PASS: V1 scalar references:",
        len(references),
    )

    return references


def to_gpu_weights(
    weights: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    host = np.asarray(
        weights,
        dtype=np.int64,
        order="C",
    )

    return torch.from_numpy(host).to(
        device=device,
        dtype=torch.int64,
    )


@torch.inference_mode()
def fast_accumulate_area(
    cache: dict,
    area_label: str,
    weights: torch.Tensor,
    batch_size: int,
) -> dict[str, np.ndarray]:
    device = weights.device
    replicate_count = int(weights.shape[0])

    recall_thresholds = torch.from_numpy(
        np.ascontiguousarray(
            cache["recall_thresholds"],
            dtype=np.float64,
        )
    ).to(
        device=device,
        dtype=torch.float64,
    )

    iou_thresholds = np.asarray(
        cache["iou_thresholds"],
        dtype=np.float64,
    )

    threshold_count = len(iou_thresholds)
    recall_count = len(cache["recall_thresholds"])

    iou75_index = int(np.argmin(
        np.abs(iou_thresholds - 0.75)
    ))

    precision_all_sum = torch.zeros(
        replicate_count,
        dtype=torch.float64,
        device=device,
    )
    precision50_sum = torch.zeros_like(
        precision_all_sum
    )
    precision75_sum = torch.zeros_like(
        precision_all_sum
    )
    recall_sum = torch.zeros_like(
        precision_all_sum
    )
    category_count = torch.zeros_like(
        precision_all_sum
    )
    micro_tp50 = torch.zeros_like(
        precision_all_sum
    )
    micro_gt = torch.zeros_like(
        precision_all_sum
    )

    epsilon = float(np.spacing(1))

    for category_index, cell in enumerate(
        cache["areas"][area_label]
    ):
        gt_counts = torch.from_numpy(
            np.ascontiguousarray(
                cell["gt_counts"],
                dtype=np.int64,
            )
        ).to(
            device=device,
            dtype=torch.int64,
        )

        image_indices = torch.from_numpy(
            np.ascontiguousarray(
                cell["image_indices"],
                dtype=np.int64,
            )
        ).to(
            device=device,
            dtype=torch.int64,
        )

        detection_count = int(image_indices.numel())

        if detection_count:
            matches = torch.from_numpy(
                np.ascontiguousarray(
                    cell["matches"],
                    dtype=bool,
                )
            ).to(
                device=device,
                dtype=torch.bool,
            )

            ignores = torch.from_numpy(
                np.ascontiguousarray(
                    cell["ignores"],
                    dtype=bool,
                )
            ).to(
                device=device,
                dtype=torch.bool,
            )

            active = ~ignores

            tp_masks = (
                matches & active
            ).to(dtype=torch.int64)

            fp_masks = (
                (~matches) & active
            ).to(dtype=torch.int64)

            del matches
            del ignores
            del active
        else:
            tp_masks = None
            fp_masks = None

        for start in range(
            0,
            replicate_count,
            batch_size,
        ):
            stop = min(
                start + batch_size,
                replicate_count,
            )

            batch_weights = weights[start:stop]

            gt_total = (
                batch_weights
                * gt_counts.unsqueeze(0)
            ).sum(dim=1)

            valid_category = gt_total > 0
            valid_float = valid_category.to(
                dtype=torch.float64
            )

            category_count[start:stop].add_(
                valid_float
            )

            micro_gt[start:stop].add_(
                gt_total.to(torch.float64)
                * valid_float
            )

            if detection_count == 0:
                continue

            detection_weights = (
                batch_weights.index_select(
                    1,
                    image_indices,
                )
            )

            safe_gt = torch.where(
                valid_category,
                gt_total,
                torch.ones_like(gt_total),
            ).to(torch.float64)

            for threshold_index in range(
                threshold_count
            ):
                tp_cumulative = torch.cumsum(
                    detection_weights
                    * tp_masks[
                        threshold_index
                    ].unsqueeze(0),
                    dim=1,
                    dtype=torch.int64,
                )

                fp_cumulative = torch.cumsum(
                    detection_weights
                    * fp_masks[
                        threshold_index
                    ].unsqueeze(0),
                    dim=1,
                    dtype=torch.int64,
                )

                if threshold_index == 0:
                    micro_tp50[start:stop].add_(
                        tp_cumulative[:, -1].to(
                            torch.float64
                        )
                        * valid_float
                    )

                tp_float = tp_cumulative.to(
                    torch.float64
                )

                recall = (
                    tp_float
                    / safe_gt.unsqueeze(1)
                )

                precision = (
                    tp_float
                    / (
                        (
                            tp_cumulative
                            + fp_cumulative
                        ).to(torch.float64)
                        + epsilon
                    )
                )

                precision = torch.flip(
                    torch.cummax(
                        torch.flip(
                            precision,
                            dims=(1,),
                        ),
                        dim=1,
                    ).values,
                    dims=(1,),
                )

                targets = (
                    recall_thresholds
                    .unsqueeze(0)
                    .expand(
                        stop - start,
                        -1,
                    )
                    .contiguous()
                )

                indices = torch.searchsorted(
                    recall.contiguous(),
                    targets,
                    right=False,
                )

                valid_index = (
                    indices < detection_count
                )

                safe_indices = indices.clamp(
                    min=0,
                    max=detection_count - 1,
                )

                sampled_precision = torch.gather(
                    precision,
                    dim=1,
                    index=safe_indices,
                )

                sampled_precision = torch.where(
                    valid_index,
                    sampled_precision,
                    torch.zeros_like(
                        sampled_precision
                    ),
                )

                sampled_sum = (
                    sampled_precision.sum(dim=1)
                    * valid_float
                )

                precision_all_sum[
                    start:stop
                ].add_(sampled_sum)

                if threshold_index == 0:
                    precision50_sum[
                        start:stop
                    ].add_(sampled_sum)

                if threshold_index == iou75_index:
                    precision75_sum[
                        start:stop
                    ].add_(sampled_sum)

                recall_sum[start:stop].add_(
                    recall[:, -1]
                    * valid_float
                )

                del tp_cumulative
                del fp_cumulative
                del tp_float
                del recall
                del precision
                del indices
                del safe_indices
                del sampled_precision

            del detection_weights
            del safe_gt

        del gt_counts
        del image_indices
        del tp_masks
        del fp_masks

        if (category_index + 1) % 5 == 0:
            print(
                f"  {area_label}: "
                f"categories={category_index + 1}",
                flush=True,
            )

    nan = torch.full_like(
        category_count,
        float("nan"),
    )

    ap_denominator = (
        category_count
        * threshold_count
        * recall_count
    )
    single_iou_denominator = (
        category_count
        * recall_count
    )
    recall_denominator = (
        category_count
        * threshold_count
    )

    ap = torch.where(
        ap_denominator > 0,
        precision_all_sum / ap_denominator,
        nan,
    )
    ap50 = torch.where(
        single_iou_denominator > 0,
        precision50_sum / single_iou_denominator,
        nan,
    )
    ap75 = torch.where(
        single_iou_denominator > 0,
        precision75_sum / single_iou_denominator,
        nan,
    )
    ar100 = torch.where(
        recall_denominator > 0,
        recall_sum / recall_denominator,
        nan,
    )
    recall50_micro = torch.where(
        micro_gt > 0,
        micro_tp50 / micro_gt,
        nan,
    )

    return {
        "AP": ap.cpu().numpy(),
        "AP50": ap50.cpu().numpy(),
        "AP75": ap75.cpu().numpy(),
        "AR100": ar100.cpu().numpy(),
        "recall50_micro":
            recall50_micro.cpu().numpy(),
    }


def fast_clean_metrics(
    cache: dict,
    weights: torch.Tensor,
) -> np.ndarray:
    all_metrics = fast_accumulate_area(
        cache,
        "all",
        weights,
        GPU_BATCH_SIZE,
    )
    small_metrics = fast_accumulate_area(
        cache,
        "small",
        weights,
        GPU_BATCH_SIZE,
    )
    medium_metrics = fast_accumulate_area(
        cache,
        "medium",
        weights,
        GPU_BATCH_SIZE,
    )
    large_metrics = fast_accumulate_area(
        cache,
        "large",
        weights,
        GPU_BATCH_SIZE,
    )

    return np.column_stack([
        all_metrics["AP"],
        all_metrics["AP50"],
        all_metrics["AP75"],
        small_metrics["AP"],
        medium_metrics["AP"],
        large_metrics["AP"],
        all_metrics["AR100"],
        small_metrics["AR100"],
    ]).astype(np.float64, copy=False)


def fast_failure_metrics(
    caches: dict[str, dict],
    weights: torch.Tensor,
) -> np.ndarray:
    results = {
        subset: fast_accumulate_area(
            cache,
            "all",
            weights,
            GPU_BATCH_SIZE,
        )
        for subset, cache in caches.items()
    }

    failure_macro = np.mean(
        np.stack([
            results[subset]["AR100"]
            for subset in v1.FAILURE_SUBSETS
        ]),
        axis=0,
    )

    small_joint_recall = results[
        "small_jointly_missed"
    ]["recall50_micro"]

    return np.column_stack([
        failure_macro,
        small_joint_recall,
    ]).astype(np.float64, copy=False)


def fast_model_metrics(
    model: str,
    clean_caches: dict,
    failure_caches: dict,
    weights: torch.Tensor,
):
    print("\nFAST MODEL:", model, flush=True)

    clean = fast_clean_metrics(
        clean_caches[model],
        weights,
    )

    failure = None

    if model in GATE_MODELS:
        failure = fast_failure_metrics(
            failure_caches[model],
            weights,
        )

    torch.cuda.empty_cache()

    return clean, failure


def verify_equivalence(
    references: dict,
    clean_results: dict[str, np.ndarray],
    failure_results: dict[str, np.ndarray],
) -> dict:
    print("\n=== V1/V2 NUMERICAL EQUIVALENCE ===")

    maximum_clean_error = 0.0
    maximum_failure_error = 0.0

    for model in MODELS:
        for replicate in range(
            EQUIVALENCE_REPLICATES
        ):
            expected_clean = references[
                (model, replicate)
            ]["clean"]

            actual_clean = clean_results[
                model
            ][replicate]

            error = float(np.nanmax(np.abs(
                expected_clean - actual_clean
            )))
            maximum_clean_error = max(
                maximum_clean_error,
                error,
            )

            np.testing.assert_allclose(
                actual_clean,
                expected_clean,
                rtol=0.0,
                atol=EQUIVALENCE_TOLERANCE,
                equal_nan=True,
            )

            expected_failure = references[
                (model, replicate)
            ]["failure"]

            if expected_failure is not None:
                actual_failure = failure_results[
                    model
                ][replicate]

                error = float(np.nanmax(np.abs(
                    expected_failure
                    - actual_failure
                )))
                maximum_failure_error = max(
                    maximum_failure_error,
                    error,
                )

                np.testing.assert_allclose(
                    actual_failure,
                    expected_failure,
                    rtol=0.0,
                    atol=EQUIVALENCE_TOLERANCE,
                    equal_nan=True,
                )

    result = {
        "status": "pass",
        "reference_executor": "v1_scalar_numpy",
        "candidate_executor": "v2_batched_cuda",
        "replicates_checked":
            EQUIVALENCE_REPLICATES,
        "models_checked": len(MODELS),
        "absolute_tolerance":
            EQUIVALENCE_TOLERANCE,
        "maximum_clean_error":
            maximum_clean_error,
        "maximum_failure_error":
            maximum_failure_error,
    }

    print(json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ))

    return result


def checkpoint_path(model: str) -> Path:
    return MODEL_DIR / f"{model}.npz"


def save_model_checkpoint(
    model: str,
    clean: np.ndarray,
    failure: np.ndarray | None,
) -> None:
    path = checkpoint_path(model)
    temporary = path.with_suffix(".npz.incomplete")

    payload = {"clean": clean}

    if failure is not None:
        payload["failure"] = failure

    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **payload,
        )

    os.replace(temporary, path)


def load_model_checkpoint(model: str):
    path = checkpoint_path(model)

    if not path.is_file():
        return None

    with np.load(path) as data:
        clean = np.asarray(
            data["clean"],
            dtype=np.float64,
        )

        failure = (
            np.asarray(
                data["failure"],
                dtype=np.float64,
            )
            if "failure" in data.files
            else None
        )

    if clean.shape != (
        REPLICATES,
        len(CLEAN_METRICS),
    ):
        raise AssertionError(
            (model, clean.shape)
        )

    if model in GATE_MODELS:
        if failure is None:
            raise AssertionError(model)

        if failure.shape != (
            REPLICATES,
            len(FAILURE_METRICS),
        ):
            raise AssertionError(
                (model, failure.shape)
            )

    return clean, failure


def build_result_rows(
    canonical_rows: dict[str, dict],
    clean_distributions: dict[str, np.ndarray],
    failure_distributions: dict[str, np.ndarray],
    failure_points: dict[str, np.ndarray],
) -> list[dict]:
    rows = []

    for comparison, (
        positive_model,
        reference_model,
    ) in v1.COMPARISONS.items():
        distributions = {
            metric: (
                clean_distributions[
                    positive_model
                ][:, metric_index]
                - clean_distributions[
                    reference_model
                ][:, metric_index]
            )
            for metric_index, metric in enumerate(
                CLEAN_METRICS
            )
        }

        if (
            positive_model in GATE_MODELS
            and reference_model in GATE_MODELS
        ):
            for metric_index, metric in enumerate(
                FAILURE_METRICS
            ):
                distributions[metric] = (
                    failure_distributions[
                        positive_model
                    ][:, metric_index]
                    - failure_distributions[
                        reference_model
                    ][:, metric_index]
                )

        for metric, distribution in distributions.items():
            if metric in CLEAN_METRICS:
                point_estimate = (
                    float(
                        canonical_rows[
                            positive_model
                        ][metric]
                    )
                    - float(
                        canonical_rows[
                            reference_model
                        ][metric]
                    )
                )
            else:
                metric_index = (
                    FAILURE_METRICS.index(metric)
                )

                point_estimate = float(
                    failure_points[
                        positive_model
                    ][metric_index]
                    - failure_points[
                        reference_model
                    ][metric_index]
                )

            lower, upper = np.quantile(
                distribution,
                [0.025, 0.975],
            )

            rows.append({
                "comparison": comparison,
                "positive_model": positive_model,
                "reference_model": reference_model,
                "metric": metric,
                "evidence_role":
                    v1.evidence_role(
                        comparison,
                        metric,
                    ),
                "point_estimate": point_estimate,
                "bootstrap_mean":
                    float(np.mean(distribution)),
                "bootstrap_median":
                    float(np.median(distribution)),
                "bootstrap_standard_error":
                    float(
                        np.std(
                            distribution,
                            ddof=1,
                        )
                    ),
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "ci_crosses_zero":
                    bool(lower <= 0.0 <= upper),
                "stable_positive":
                    bool(lower > 0.0),
                "stable_negative":
                    bool(upper < 0.0),
                "p_value":
                    v1.empirical_two_sided_p(
                        distribution
                    ),
                "replicates": REPLICATES,
                "seed": v1.SEED,
            })

    v1.apply_bh(rows)

    return rows


def prepare_work(
    weights: np.ndarray,
) -> dict:
    WORK.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    state = {
        "version": "paired_bootstrap_executor_v2",
        "protocol_sha256":
            raw_sha256(v1.PROTOCOL),
        "v1_script_sha256":
            raw_sha256(V1_PATH),
        "v2_script_sha256":
            raw_sha256(Path(__file__).resolve()),
        "weights_sha256":
            weights_sha256(weights),
        "replicates": REPLICATES,
        "seed": v1.SEED,
        "expected_images": EXPECTED_IMAGES,
        "gpu_batch_size": GPU_BATCH_SIZE,
    }

    state_path = WORK / "run_state.json"

    if state_path.is_file():
        existing = json.loads(
            state_path.read_text()
        )

        if existing != state:
            raise AssertionError(
                "Existing V2 work directory belongs "
                "to a different executor or input state."
            )
    else:
        v1.write_json(state_path, state)

    weights_path = WORK / "paired_image_weights.npy"

    if weights_path.is_file():
        existing_weights = np.load(
            weights_path,
            mmap_mode="r",
        )

        if (
            existing_weights.shape != weights.shape
            or existing_weights.dtype
            != weights.dtype
            or hashlib.sha256(
                np.ascontiguousarray(
                    existing_weights
                ).tobytes()
            ).hexdigest()
            != state["weights_sha256"]
        ):
            raise AssertionError(weights_path)
    else:
        np.save(weights_path, weights)

    return state


def finalize(
    started: float,
    canonical_rows: dict[str, dict],
    clean_distributions: dict[str, np.ndarray],
    failure_distributions: dict[str, np.ndarray],
    failure_points: dict[str, np.ndarray],
    state: dict,
    equivalence: dict,
    benchmark: dict,
):
    result_rows = build_result_rows(
        canonical_rows,
        clean_distributions,
        failure_distributions,
        failure_points,
    )

    v1.write_csv(
        WORK / "paired_bootstrap_intervals.csv",
        result_rows,
    )

    payload = {}

    for model, values in clean_distributions.items():
        payload[f"clean__{model}"] = values

    for model, values in failure_distributions.items():
        payload[f"failure__{model}"] = values

    np.savez_compressed(
        WORK / "bootstrap_model_metrics.npz",
        **payload,
    )

    stable_primary = [
        row
        for row in result_rows
        if row["evidence_role"].startswith(
            "primary"
        )
        and not row["ci_crosses_zero"]
    ]

    summary = {
        "version": "paired_bootstrap_v1",
        "executor_version":
            "batched_cuda_v2",
        "status": "complete",
        "protocol_sha256":
            raw_sha256(v1.PROTOCOL),
        "replicates": REPLICATES,
        "seed": v1.SEED,
        "confidence_level": 0.95,
        "interval": "percentile",
        "resampling_unit":
            "official_val_image",
        "paired_resampling": True,
        "weights_sha256":
            state["weights_sha256"],
        "comparisons":
            list(v1.COMPARISONS),
        "clean_metrics":
            list(CLEAN_METRICS),
        "targeted_recovery_metrics":
            list(FAILURE_METRICS),
        "equivalence_validation":
            equivalence,
        "benchmark": benchmark,
        "stable_primary_results":
            stable_primary,
        "interpretation_rule": (
            "Intervals crossing zero are reported "
            "as no stable detected difference. "
            "This does not establish equivalence "
            "or proof of no effect."
        ),
        "scope_warning": (
            "Image bootstrap quantifies "
            "validation-image sampling uncertainty "
            "and does not replace training-seed "
            "replication."
        ),
        "elapsed_minutes":
            (time.time() - started) / 60.0,
    }

    v1.write_json(
        WORK / "summary.json",
        summary,
    )

    metadata = {
        "version":
            "paired_bootstrap_executor_v2",
        "status": "complete",
        "created_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "script":
            str(Path(__file__).resolve()),
        "script_sha256":
            raw_sha256(Path(__file__).resolve()),
        "v1_reference_script":
            str(V1_PATH.resolve()),
        "v1_reference_sha256":
            raw_sha256(V1_PATH),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_device":
            torch.cuda.get_device_name(0),
        "gpu_batch_size": GPU_BATCH_SIZE,
        "new_training": False,
        "new_gpu_inference": False,
        "method": (
            "The frozen image-cluster bootstrap "
            "multiplicity weights and complete COCO "
            "precision-recall re-accumulation are "
            "evaluated with numerically verified "
            "batched CUDA operations."
        ),
    }

    v1.write_json(
        WORK / "metadata.json",
        metadata,
    )

    manifest_path, manifest_sha256 = (
        v1.build_manifest(WORK)
    )

    os.replace(WORK, OUTPUT)

    print("\n" + "=" * 100)
    print("PAIRED BOOTSTRAP COMPLETE")
    print("=" * 100)

    for row in result_rows:
        if row["evidence_role"] != "secondary":
            print(
                row["comparison"],
                row["metric"],
                "delta=",
                round(row["point_estimate"], 6),
                "CI95=(",
                round(row["ci95_lower"], 6),
                ",",
                round(row["ci95_upper"], 6),
                ")",
                "stable_positive=",
                row["stable_positive"],
                "stable_negative=",
                row["stable_negative"],
            )

    print("output:", OUTPUT)
    print(
        "manifest:",
        OUTPUT / manifest_path.name,
    )
    print(
        "manifest_sha256:",
        manifest_sha256,
    )
    print("NOTHING TRAINED OR GPU-INFERRED")
    print("NEXT: exposure-budget audit")


def execute(
    canonical_rows: dict[str, dict],
    failure_ids: dict[str, set[int]],
):
    started = time.time()

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if torch.cuda.device_count() != 1:
        print(
            "WARNING: executor uses CUDA device 0 only."
        )

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    device = torch.device("cuda:0")

    print("gpu:", torch.cuda.get_device_name(0))
    print(
        "gpu_memory_GiB:",
        round(
            torch.cuda.get_device_properties(
                0
            ).total_memory / 1024**3,
            3,
        ),
    )

    weights = generate_weights()
    state = prepare_work(weights)

    clean_caches, failure_caches = (
        build_all_caches(
            canonical_rows,
            failure_ids,
        )
    )

    references = scalar_reference(
        clean_caches,
        failure_caches,
        weights,
    )

    print("\n=== CUDA BENCHMARK: 100 REPLICATES ===")

    benchmark_weights = to_gpu_weights(
        weights[:BENCHMARK_REPLICATES],
        device,
    )

    benchmark_clean = {}
    benchmark_failure = {}

    torch.cuda.synchronize()
    benchmark_started = time.time()

    for model in MODELS:
        clean, failure = fast_model_metrics(
            model,
            clean_caches,
            failure_caches,
            benchmark_weights,
        )

        benchmark_clean[model] = clean

        if failure is not None:
            benchmark_failure[model] = failure

    torch.cuda.synchronize()

    benchmark_seconds = (
        time.time() - benchmark_started
    )

    projected_gpu_hours = (
        benchmark_seconds
        * REPLICATES
        / BENCHMARK_REPLICATES
        / 3600.0
    )

    equivalence = verify_equivalence(
        references,
        benchmark_clean,
        benchmark_failure,
    )

    benchmark = {
        "replicates": BENCHMARK_REPLICATES,
        "elapsed_seconds":
            benchmark_seconds,
        "seconds_per_replicate_all_models":
            benchmark_seconds
            / BENCHMARK_REPLICATES,
        "projected_full_gpu_hours":
            projected_gpu_hours,
        "maximum_allowed_projected_hours":
            MAX_PROJECTED_GPU_HOURS,
        "passed":
            projected_gpu_hours
            <= MAX_PROJECTED_GPU_HOURS,
    }

    v1.write_json(
        WORK / "equivalence_validation.json",
        equivalence,
    )
    v1.write_json(
        WORK / "performance_benchmark.json",
        benchmark,
    )

    print("\n=== PERFORMANCE BENCHMARK ===")
    print(json.dumps(
        benchmark,
        indent=2,
        ensure_ascii=False,
    ))

    if projected_gpu_hours > (
        MAX_PROJECTED_GPU_HOURS
    ):
        raise RuntimeError(
            "V2 benchmark exceeded the frozen "
            "six-hour execution ceiling. "
            "Full bootstrap was not started."
        )

    del benchmark_weights
    torch.cuda.empty_cache()

    remaining_weights = to_gpu_weights(
        weights[BENCHMARK_REPLICATES:],
        device,
    )

    clean_distributions = {}
    failure_distributions = {}

    print("\n=== FULL VECTORIZED BOOTSTRAP ===")

    for model_index, model in enumerate(
        MODELS,
        start=1,
    ):
        existing = load_model_checkpoint(model)

        if existing is not None:
            clean, failure = existing
            print(
                f"RESUME {model_index}/{len(MODELS)}:",
                model,
            )
        else:
            print(
                f"MODEL {model_index}/{len(MODELS)}:",
                model,
            )

            remaining_clean, remaining_failure = (
                fast_model_metrics(
                    model,
                    clean_caches,
                    failure_caches,
                    remaining_weights,
                )
            )

            clean = np.concatenate(
                [
                    benchmark_clean[model],
                    remaining_clean,
                ],
                axis=0,
            )

            failure = None

            if model in GATE_MODELS:
                failure = np.concatenate(
                    [
                        benchmark_failure[model],
                        remaining_failure,
                    ],
                    axis=0,
                )

            save_model_checkpoint(
                model,
                clean,
                failure,
            )

        clean_distributions[model] = clean

        if failure is not None:
            failure_distributions[model] = (
                failure
            )

        print(
            f"PASS model={model} "
            f"replicates={len(clean)}",
            flush=True,
        )

    del remaining_weights
    torch.cuda.empty_cache()

    print("\n=== ORIGINAL-SAMPLE FAILURE POINTS ===")

    original_weights = torch.ones(
        (1, EXPECTED_IMAGES),
        dtype=torch.int64,
        device=device,
    )

    failure_points = {}

    for model in sorted(GATE_MODELS):
        failure_points[model] = (
            fast_failure_metrics(
                failure_caches[model],
                original_weights,
            )[0]
        )

    del original_weights
    torch.cuda.empty_cache()

    finalize(
        started,
        canonical_rows,
        clean_distributions,
        failure_distributions,
        failure_points,
        state,
        equivalence,
        benchmark,
    )


def main():
    args = parse_args()

    protocol, canonical_rows, failure_ids = (
        v1.preflight()
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    print("\n=== V2 EXECUTOR PREFLIGHT ===")
    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0))
    print("replicates:", REPLICATES)
    print(
        "benchmark_replicates:",
        BENCHMARK_REPLICATES,
    )
    print(
        "equivalence_replicates:",
        EQUIVALENCE_REPLICATES,
    )
    print("gpu_batch_size:", GPU_BATCH_SIZE)
    print(
        "projected_time_ceiling_hours:",
        MAX_PROJECTED_GPU_HOURS,
    )

    if args.preflight_only:
        print(
            "\nPASS: V2 preflight only\n"
            "NOTHING CREATED OR RESAMPLED"
        )
        return

    execute(
        canonical_rows,
        failure_ids,
    )


if __name__ == "__main__":
    main()