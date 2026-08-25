from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


PROJECT = Path(
    "/root/rivermind-data/autodrive/code/"
    "robust-autonomous-driving-perception"
)
RESULTS = Path(
    "/root/rivermind-data/autodrive/results/evaluation"
)
FINAL = RESULTS / "final_analysis_v1"

PROTOCOL = (
    FINAL / "frozen_protocol/final_testing_protocol_v1.json"
)
CORRUPTION_MANIFEST = (
    FINAL / "representative_corruptions_v1/artifact_manifest.json"
)
SMOKE_SUMMARY = (
    RESULTS / "architecture_gate/training_smoke_v1/"
    "smoke_summary.json"
)

OUTPUT = FINAL / "efficiency_evaluation_v1"
WORK = FINAL / (
    "efficiency_evaluation_v1.incomplete-"
    "693c39375cc1"
)

MODELS = {
    "standard_e20": {
        "path": (
            RESULTS / "architecture_gate/formal_gate_v4/"
            "standard/weights/resume_epoch20.pt"
        ),
        "sha256": (
            "dc1e3ef68e233a2fad4911955bbe178f0104cd9000dcbdc772"
            "dac8d0a58438b5"
        ),
        "smoke_key": "standard",
        "expected_strides": [8.0, 16.0, 32.0],
    },
    "p2_e20": {
        "path": (
            RESULTS / "architecture_gate/formal_gate_v4/"
            "p2_projected/weights/resume_epoch20.pt"
        ),
        "sha256": (
            "642a6b4009f085649b6da3f1647821d3e760e6724b019e08d"
            "022080c8050bc29"
        ),
        "smoke_key": "p2",
        "expected_strides": [4.0, 8.0, 16.0, 32.0],
    },
}

EXPECTED = {
    PROTOCOL: (
        "693c39375cc1eefc4f3ff9f957a8355fb947552a19dfef228"
        "cae6cfc21e3dcfe"
    ),
    CORRUPTION_MANIFEST: (
        "b2be2be57d06b103317ef7c80934d010c9171cd2d21ca853"
        "6e44cf21a25b85b3"
    ),
    SMOKE_SUMMARY: (
        "471ebe97a112b5eacf4b949b644c1717b9978996e52bcde186"
        "d2afe696e50b7d"
    ),
}

IMAGE_SIZE = 960
BATCH = 1
WARMUP = 50
MEASUREMENTS = 200
DEVICE = 0

EXPECTED_CLASSES = {
    0: "pedestrian",
    1: "rider",
    2: "car",
    3: "truck",
    4: "bus",
    5: "train",
    6: "motorcycle",
    7: "bicycle",
    8: "traffic light",
    9: "traffic sign",
}


def parse_args():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight-only", action="store_true")
    group.add_argument("--execute", action="store_true")

    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def write_json(path: Path, data) -> None:
    payload = (
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def run_command(command: list[str]) -> str:
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as error:
        return f"{type(error).__name__}: {error}"


def gpu_snapshot() -> str:
    return run_command([
        "nvidia-smi",
        "--query-gpu="
        "name,driver_version,memory.total,"
        "temperature.gpu,pstate,clocks.sm,power.draw",
        "--format=csv,noheader,nounits",
    ])


def other_gpu_processes() -> list[str]:
    output = run_command([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])

    if not output or output.startswith(
        ("CalledProcessError", "FileNotFoundError")
    ):
        return []

    current_pid = str(os.getpid())
    result = []

    for line in output.splitlines():
        pid = line.split(",", 1)[0].strip()
        if pid and pid != current_pid:
            result.append(line.strip())

    return result


def verify_file(path: Path, expected: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)

    actual = sha256(path)

    print(path)
    print(" expected:", expected)
    print(" actual  :", actual)

    if actual != expected:
        raise AssertionError(path)

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual,
    }


def normalize_names(names) -> dict[int, str]:
    return {
        int(key): str(value)
        for key, value in dict(names).items()
    }


def preflight():
    print("=== FINAL EFFICIENCY PREFLIGHT ===")

    records = {}

    for path, expected in EXPECTED.items():
        records[str(path)] = verify_file(path, expected)

    for spec in MODELS.values():
        records[str(spec["path"])] = verify_file(
            spec["path"],
            spec["sha256"],
        )

    protocol = json.loads(PROTOCOL.read_text())
    rules = protocol["efficiency_evaluation"]

    assert rules["architecture_pair"] == [
        "standard_e20",
        "p2_e20",
    ]
    assert rules["image_size"] == IMAGE_SIZE
    assert rules["batch"] == BATCH
    assert rules["precision"] == "FP16"
    assert rules["hardware"] == "NVIDIA GeForce RTX 4090"
    assert rules["warmup_iterations"] == WARMUP
    assert rules["measurement_iterations"] == MEASUREMENTS
    assert rules["round_orders"] == [
        ["standard", "p2"],
        ["p2", "standard"],
    ]
    assert rules["cuda_synchronize_each_measurement"] is True

    smoke = json.loads(SMOKE_SUMMARY.read_text())
    assert smoke["status"] == "passed"

    assert smoke["standard"]["name"] == "standard_batch8"
    assert smoke["p2"]["name"] == "p2_projected_batch8"

    for key in ("standard", "p2"):
        assert smoke[key]["batch"] == 8
        assert smoke[key]["image_size"] == 960
        assert smoke[key]["fraction"] == 0.002
        assert smoke[key]["peak_allocated_GiB"] > 0
        assert smoke[key]["peak_reserved_GiB"] > 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    gpu_name = torch.cuda.get_device_name(DEVICE)
    if gpu_name != "NVIDIA GeForce RTX 4090":
        raise AssertionError(gpu_name)

    if torch.__version__ != "2.2.2":
        raise AssertionError(torch.__version__)

    if ultralytics.__version__ != "8.4.115":
        raise AssertionError(ultralytics.__version__)

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    if WORK.exists():
        raise FileExistsError(WORK)

    other_processes = other_gpu_processes()
    if other_processes:
        raise RuntimeError({
            "other_gpu_processes": other_processes,
        })

    print("GPU:", gpu_name)
    print(
        "GPU memory GiB:",
        torch.cuda.get_device_properties(
            DEVICE
        ).total_memory / 1024 ** 3,
    )
    print("torch:", torch.__version__)
    print("ultralytics:", ultralytics.__version__)
    print("warmup:", WARMUP)
    print("measurements_per_round:", MEASUREMENTS)
    print("rounds:", 2)
    print("PASS: all frozen efficiency inputs are valid")
    print("NOTHING CREATED, MODIFIED, INFERRED OR TRAINED")

    return protocol, smoke, records


def clean_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(DEVICE)
    time.sleep(0.5)


def static_model_metrics(key: str, spec: dict) -> dict:
    print(f"\nSTATIC MODEL AUDIT: {key}")

    wrapper = YOLO(str(spec["path"]))
    model = wrapper.model.cpu().float().eval()

    names = normalize_names(model.names)
    strides = [
        float(value)
        for value in model.stride.detach().cpu().tolist()
    ]

    if names != EXPECTED_CLASSES:
        raise AssertionError({
            "model": key,
            "names": names,
        })

    if strides != spec["expected_strides"]:
        raise AssertionError({
            "model": key,
            "strides": strides,
        })

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    gflops = float(
        get_flops(model, imgsz=IMAGE_SIZE)
    )

    if parameters <= 0 or gflops <= 0:
        raise AssertionError({
            "parameters": parameters,
            "GFLOPs": gflops,
        })



    buffer = io.BytesIO()
    model.half()
    torch.save(
        {
            "model": model,
            "precision": "FP16",
        },
        buffer,
    )
    model_file_size_mib = (
        buffer.getbuffer().nbytes / 1024 ** 2
    )
    buffer.close()

    checkpoint_file_size_mib = (
        spec["path"].stat().st_size / 1024 ** 2
    )

    result = {
        "model": key,
        "checkpoint": str(spec["path"]),
        "checkpoint_sha256": spec["sha256"],
        "parameters": int(parameters),
        "trainable_parameters_at_load": int(
            trainable_parameters
        ),
        "GFLOPs": gflops,
        "model_file_size_MiB": model_file_size_mib,
        "model_file_size_definition": (
            "In-memory torch serialization containing only the "
            "FP16 model; optimizer and resume state excluded."
        ),
        "source_checkpoint_file_size_MiB":
            checkpoint_file_size_mib,
        "strides": strides,
        "class_names": names,
    }

    print("parameters:", parameters)
    print("GFLOPs:", round(gflops, 4))
    print(
        "model_file_size_MiB:",
        round(model_file_size_mib, 4),
    )
    print("strides:", strides)

    del model
    del wrapper
    clean_cuda()

    return result


def measure_forward(
    *,
    model_key: str,
    spec: dict,
    round_index: int,
    position: int,
) -> tuple[dict, np.ndarray]:
    clean_cuda()

    print(
        f"\nROUND {round_index} POSITION {position}: "
        f"{model_key}"
    )

    wrapper = YOLO(str(spec["path"]))
    model = (
        wrapper.model
        .to(f"cuda:{DEVICE}")
        .eval()
        .half()
    )

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    sample = torch.zeros(
        (BATCH, 3, IMAGE_SIZE, IMAGE_SIZE),
        device=f"cuda:{DEVICE}",
        dtype=torch.float16,
    )

    torch.cuda.reset_peak_memory_stats(DEVICE)

    with torch.inference_mode():
        for _ in range(WARMUP):
            output = model(sample)
            del output

        torch.cuda.synchronize(DEVICE)

        latency_ms = np.empty(
            MEASUREMENTS,
            dtype=np.float64,
        )

        for index in range(MEASUREMENTS):
            torch.cuda.synchronize(DEVICE)
            started = time.perf_counter()

            output = model(sample)

            torch.cuda.synchronize(DEVICE)
            latency_ms[index] = (
                time.perf_counter() - started
            ) * 1000.0

            del output

    peak_memory_mib = (
        torch.cuda.max_memory_allocated(DEVICE)
        / 1024 ** 2
    )

    result = {
        "model": model_key,
        "round": round_index,
        "position": position,
        "warmup_iterations": WARMUP,
        "measurement_iterations": MEASUREMENTS,
        "batch": BATCH,
        "image_size": IMAGE_SIZE,
        "precision": "FP16",
        "input": "synthetic_zero_tensor",
        "median_forward_latency_ms":
            float(np.median(latency_ms)),
        "p90_forward_latency_ms":
            float(np.percentile(latency_ms, 90)),
        "mean_forward_latency_ms":
            float(np.mean(latency_ms)),
        "std_forward_latency_ms":
            float(np.std(latency_ms)),
        "minimum_forward_latency_ms":
            float(np.min(latency_ms)),
        "maximum_forward_latency_ms":
            float(np.max(latency_ms)),
        "peak_inference_memory_MiB":
            float(peak_memory_mib),
    }

    print(json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ))

    del sample
    del model
    del wrapper
    clean_cuda()

    return result, latency_ms


def summarize_model(
    *,
    model_key: str,
    static: dict,
    round_records: list[dict],
    arrays: list[np.ndarray],
    smoke: dict,
    smoke_key: str,
) -> dict:
    combined = np.concatenate(arrays)

    smoke_row = smoke[smoke_key]

    return {
        **static,
        "latency_measurements": int(combined.size),
        "median_forward_latency_ms":
            float(np.median(combined)),
        "p90_forward_latency_ms":
            float(np.percentile(combined, 90)),
        "mean_forward_latency_ms":
            float(np.mean(combined)),
        "std_forward_latency_ms":
            float(np.std(combined)),
        "peak_inference_memory_MiB":
            float(max(
                row["peak_inference_memory_MiB"]
                for row in round_records
            )),
        "peak_training_memory_MiB": (
            float(smoke_row["peak_allocated_GiB"])
            * 1024.0
        ),
        "peak_training_reserved_memory_MiB": (
            float(smoke_row["peak_reserved_GiB"])
            * 1024.0
        ),
        "training_memory_evidence": {
            "source": str(SMOKE_SUMMARY),
            "source_sha256": EXPECTED[
                SMOKE_SUMMARY
            ],
            "measurement_type":
                "existing_architecture_training smoke",
            "physical_batch": smoke_row["batch"],
            "image_size": smoke_row["image_size"],
            "dataset_fraction": smoke_row["fraction"],
            "note": (
                "Training memory is existing batch-8 evidence. "
                "No new training was performed in Stage 9."
            ),
        },
        "round_records": round_records,
    }


def delta_record(
    standard: dict,
    p2: dict,
) -> dict:
    fields = [
        "parameters",
        "GFLOPs",
        "model_file_size_MiB",
        "median_forward_latency_ms",
        "p90_forward_latency_ms",
        "mean_forward_latency_ms",
        "peak_inference_memory_MiB",
        "peak_training_memory_MiB",
    ]

    output = {}

    for field in fields:
        standard_value = float(standard[field])
        p2_value = float(p2[field])

        output[field] = {
            "standard": standard_value,
            "p2": p2_value,
            "p2_minus_standard":
                p2_value - standard_value,
            "increase_fraction": (
                p2_value / standard_value - 1.0
            ),
        }

    return output


def build_manifest(directory: Path) -> dict:
    artifacts = []

    for path in sorted(directory.iterdir()):
        if (
            not path.is_file()
            or path.name == "artifact_manifest.json"
        ):
            continue

        artifacts.append({
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })

    return {
        "version": "final_efficiency_manifest_v1",
        "created_utc":
            datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }


def execute(
    protocol: dict,
    smoke: dict,
    input_records: dict,
):
    WORK.mkdir(parents=True, exist_ok=False)

    started = time.time()
    before_gpu = gpu_snapshot()

    torch.manual_seed(20260816)
    np.random.seed(20260816)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    static = {
        key: static_model_metrics(key, spec)
        for key, spec in MODELS.items()
    }

    round_orders = [
        ["standard_e20", "p2_e20"],
        ["p2_e20", "standard_e20"],
    ]

    round_records = {
        key: []
        for key in MODELS
    }
    measurement_arrays = {
        key: []
        for key in MODELS
    }
    npz_payload = {}

    for round_index, order in enumerate(
        round_orders,
        start=1,
    ):
        for position, model_key in enumerate(
            order,
            start=1,
        ):
            record, values = measure_forward(
                model_key=model_key,
                spec=MODELS[model_key],
                round_index=round_index,
                position=position,
            )

            round_records[model_key].append(record)
            measurement_arrays[model_key].append(values)

            npz_payload[
                f"{model_key}_round{round_index}"
            ] = values

    models = {}

    for key, spec in MODELS.items():
        models[key] = summarize_model(
            model_key=key,
            static=static[key],
            round_records=round_records[key],
            arrays=measurement_arrays[key],
            smoke=smoke,
            smoke_key=spec["smoke_key"],
        )

    deltas = delta_record(
        models["standard_e20"],
        models["p2_e20"],
    )

    latency_increase = deltas[
        "median_forward_latency_ms"
    ]["increase_fraction"]

    latency_rule_max = 0.30
    latency_guardrail_passed = (
        latency_increase <= latency_rule_max
    )

    summary = {
        "version": "final_efficiency_evaluation_v1",
        "status": "complete",
        "scope": (
            "RTX4090 PyTorch FP16 forward-pass evidence; "
            "not edge-device end-to-end latency."
        ),
        "models": models,
        "p2_minus_standard": deltas,
        "latency_guardrail": {
            "metric":
                "median_forward_latency_ms",
            "maximum_increase_fraction":
                latency_rule_max,
            "observed_increase_fraction":
                latency_increase,
            "passed": latency_guardrail_passed,
        },
        "elapsed_minutes":
            (time.time() - started) / 60.0,
    }

    write_json(
        WORK / "efficiency_summary.json",
        summary,
    )

    with (
        WORK / "efficiency_table.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = [
            "model",
            "parameters",
            "GFLOPs",
            "model_file_size_MiB",
            "median_forward_latency_ms",
            "p90_forward_latency_ms",
            "mean_forward_latency_ms",
            "peak_inference_memory_MiB",
            "peak_training_memory_MiB",
            "training_memory_batch",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for key, row in models.items():
            writer.writerow({
                "model": key,
                "parameters": row["parameters"],
                "GFLOPs": row["GFLOPs"],
                "model_file_size_MiB":
                    row["model_file_size_MiB"],
                "median_forward_latency_ms":
                    row["median_forward_latency_ms"],
                "p90_forward_latency_ms":
                    row["p90_forward_latency_ms"],
                "mean_forward_latency_ms":
                    row["mean_forward_latency_ms"],
                "peak_inference_memory_MiB":
                    row["peak_inference_memory_MiB"],
                "peak_training_memory_MiB":
                    row["peak_training_memory_MiB"],
                "training_memory_batch":
                    row["training_memory_evidence"][
                        "physical_batch"
                    ],
            })

    np.savez_compressed(
        WORK / "latency_measurements.npz",
        **npz_payload,
    )

    metadata = {
        "version": "final_efficiency_metadata_v1",
        "status": "complete",
        "created_utc":
            datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256":
            sha256(Path(__file__).resolve()),
        "protocol": str(PROTOCOL),
        "protocol_sha256": EXPECTED[PROTOCOL],
        "input_records": input_records,
        "torch_version": torch.__version__,
        "ultralytics_version":
            ultralytics.__version__,
        "gpu_before": before_gpu,
        "gpu_after": gpu_snapshot(),
        "seed": 20260816,
        "new_training_performed": False,
        "new_dataset_inference_performed": False,
    }

    write_json(
        WORK / "metadata.json",
        metadata,
    )

    manifest = build_manifest(WORK)
    write_json(
        WORK / "artifact_manifest.json",
        manifest,
    )
    manifest_sha = sha256(
        WORK / "artifact_manifest.json"
    )

    os.replace(WORK, OUTPUT)

    print("\n" + "=" * 100)
    print("FINAL EFFICIENCY EVALUATION COMPLETE")
    print("=" * 100)

    for key, row in models.items():
        print(
            key,
            "parameters=",
            row["parameters"],
            "GFLOPs=",
            round(row["GFLOPs"], 4),
            "median_ms=",
            round(
                row["median_forward_latency_ms"],
                4,
            ),
            "p90_ms=",
            round(
                row["p90_forward_latency_ms"],
                4,
            ),
            "peak_inference_MiB=",
            round(
                row["peak_inference_memory_MiB"],
                3,
            ),
            "peak_training_MiB=",
            round(
                row["peak_training_memory_MiB"],
                3,
            ),
        )

    print(
        "p2_latency_increase_fraction:",
        latency_increase,
    )
    print(
        "latency_guardrail_passed:",
        latency_guardrail_passed,
    )
    print("output:", OUTPUT)
    print(
        "manifest:",
        OUTPUT / "artifact_manifest.json",
    )
    print("manifest_sha256:", manifest_sha)
    print("NOTHING TRAINED")
    print("NEXT: deterministic qualitative cases")
    print("=" * 100)


def main():
    args = parse_args()

    protocol, smoke, records = preflight()

    if args.preflight_only:
        return

    execute(
        protocol=protocol,
        smoke=smoke,
        input_records=records,
    )


if __name__ == "__main__":
    main()