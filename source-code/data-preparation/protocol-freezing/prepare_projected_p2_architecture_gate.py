from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


SEED = 20260809

ROOT = Path("/root/rivermind-data/autodrive")
REPO = ROOT / "code/robust-autonomous-driving-perception"
RESULTS = ROOT / "results"

OFFICIAL_WEIGHT = REPO / "yolo11m.pt"

V1_DIR = (
    RESULTS
    / "evaluation/architecture_gate/prepared_inputs_v1"
)
V1_MANIFEST = V1_DIR / "artifact_manifest.json"
V1_P2_CONFIG = V1_DIR / "yolo11m_p2_gate.yaml"

OUTPUT_DIR = (
    RESULTS
    / "evaluation/architecture_gate/prepared_inputs_v2"
)

EXPECTED = {
    OFFICIAL_WEIGHT:
        "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95",
    V1_MANIFEST:
        "3d4ad6089b1e44d95fe05244bfb8c6527425c4847692b3e70c86fe2044e0e43e",
    V1_P2_CONFIG:
        "29a859e1e6f7b88aa9739e6dd9981206e9bca942c82bfc226294ef76afcdd243",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def write_json(
    path: Path,
    value,
) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def verify_v1_manifest() -> None:
    manifest = json.loads(
        V1_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    for name, record in manifest["files"].items():
        path = V1_DIR / name

        if not path.is_file():
            raise FileNotFoundError(path)

        actual = sha256(path)

        if actual != record["sha256"]:
            raise AssertionError(
                f"V1 artifact mismatch:\n"
                f"path={path}\n"
                f"expected={record['sha256']}\n"
                f"actual={actual}"
            )

    print("PASS: prepared_inputs_v1 manifest")


def source_for_target(
    target_name: str,
    target_tensor: torch.Tensor,
    source_state: dict[str, torch.Tensor],
    source_detect_index: int,
    target_detect_index: int,
) -> tuple[str | None, str]:
    if target_name in source_state:
        source_tensor = source_state[
            target_name
        ]

        if (
            source_tensor.shape
            == target_tensor.shape
        ):
            return target_name, "exact"

    match = re.match(
        rf"model\.{target_detect_index}\."
        rf"(cv2|cv3)\.(\d+)\.(.+)",
        target_name,
    )

    if match:
        tower = match.group(1)
        target_branch = int(match.group(2))
        suffix = match.group(3)


        if target_branch >= 1:
            source_name = (
                f"model.{source_detect_index}."
                f"{tower}.{target_branch - 1}."
                f"{suffix}"
            )

            source_tensor = source_state.get(
                source_name
            )

            if (
                source_tensor is not None
                and source_tensor.shape
                == target_tensor.shape
            ):
                return (
                    source_name,
                    f"{tower}_shift",
                )

    dfl_prefix = (
        f"model.{target_detect_index}.dfl."
    )

    if target_name.startswith(dfl_prefix):
        suffix = target_name[
            len(dfl_prefix):
        ]
        source_name = (
            f"model.{source_detect_index}."
            f"dfl.{suffix}"
        )

        source_tensor = source_state.get(
            source_name
        )

        if (
            source_tensor is not None
            and source_tensor.shape
            == target_tensor.shape
        ):
            return source_name, "dfl_shift"

    return None, "random"


def main() -> None:
    print("=== P2-V2 FROZEN INPUT CHECK ===")

    for path, expected in EXPECTED.items():
        if not path.is_file():
            raise FileNotFoundError(path)

        actual = sha256(path)

        print(path)
        print(" expected:", expected)
        print(" actual  :", actual)

        if actual != expected:
            raise AssertionError(
                f"SHA256 mismatch: {path}"
            )

    verify_v1_manifest()

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {OUTPUT_DIR}"
        )

    OUTPUT_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.manual_seed(SEED)

    source_wrapper = YOLO(
        str(OFFICIAL_WEIGHT),
        task="detect",
    )
    source_model = (
        source_wrapper.model.float()
    )
    source_state = source_model.state_dict()
    source_parameters = dict(
        source_model.named_parameters()
    )

    source_detect_index = (
        len(source_model.model) - 1
    )

    if source_detect_index != 23:
        raise AssertionError(
            source_detect_index
        )

    source_names = source_model.names

    if isinstance(source_names, list):
        source_names = {
            index: name
            for index, name
            in enumerate(source_names)
        }
    else:
        source_names = {
            int(index): name
            for index, name
            in source_names.items()
        }

    base_config = yaml.safe_load(
        V1_P2_CONFIG.read_text(
            encoding="utf-8"
        )
    )




    projected_bdd_config = copy.deepcopy(
        base_config
    )
    projected_bdd_config["nc"] = 10
    projected_bdd_config["head"] = (
        copy.deepcopy(
            base_config["head"][:-1]
        )
        + [
            [-1, 1, "Conv", [256, 1, 1]],
            [[26, 16, 19, 22], 1, "Detect", ["nc"]],
        ]
    )

    projected_coco_config = copy.deepcopy(
        projected_bdd_config
    )
    projected_coco_config["nc"] = 80

    with tempfile.TemporaryDirectory(
        prefix="prepared_inputs_v2.incomplete-",
        dir=OUTPUT_DIR.parent,
    ) as temporary_name:
        temporary_dir = Path(temporary_name)

        bdd_config_path = (
            temporary_dir
            / "yolo11m_p2_projected_bdd10.yaml"
        )
        coco_config_path = (
            temporary_dir
            / "yolo11m_p2_projected_coco80.yaml"
        )
        init_path = (
            temporary_dir
            / "yolo11m_p2_projected_coco_init.pt"
        )

        bdd_config_path.write_text(
            yaml.safe_dump(
                projected_bdd_config,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        coco_config_path.write_text(
            yaml.safe_dump(
                projected_coco_config,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

        torch.manual_seed(SEED)

        target_model = DetectionModel(
            cfg=projected_coco_config,
            ch=3,
            nc=80,
            verbose=False,
        ).float()

        target_model.names = copy.deepcopy(
            source_names
        )

        target_detect_index = (
            len(target_model.model) - 1
        )

        if target_detect_index != 27:
            raise AssertionError(
                target_detect_index
            )

        source_strides = [
            int(value)
            for value
            in source_model.stride.tolist()
        ]
        target_strides = [
            int(value)
            for value
            in target_model.stride.tolist()
        ]

        if source_strides != [8, 16, 32]:
            raise AssertionError(
                source_strides
            )

        if target_strides != [
            4,
            8,
            16,
            32,
        ]:
            raise AssertionError(
                target_strides
            )

        target_state = target_model.state_dict()
        mapping = []
        state_counts = Counter()
        state_numel = Counter()

        for target_name, target_tensor in (
            target_state.items()
        ):
            source_name, mapping_type = (
                source_for_target(
                    target_name,
                    target_tensor,
                    source_state,
                    source_detect_index,
                    target_detect_index,
                )
            )

            state_counts[mapping_type] += 1
            state_numel[mapping_type] += (
                target_tensor.numel()
            )

            if source_name is not None:
                source_tensor = source_state[
                    source_name
                ]

                target_state[target_name] = (
                    source_tensor.detach()
                    .clone()
                    .to(
                        dtype=target_tensor.dtype,
                        device=target_tensor.device,
                    )
                )

            mapping.append(
                {
                    "target": target_name,
                    "source": source_name,
                    "type": mapping_type,
                    "shape": list(
                        target_tensor.shape
                    ),
                    "numel": target_tensor.numel(),
                }
            )

        target_model.load_state_dict(
            target_state,
            strict=True,
        )


        target_parameters = dict(
            target_model.named_parameters()
        )
        parameter_counts = Counter()
        parameter_numel = Counter()
        random_parameter_groups = Counter()
        random_parameter_names = []

        for target_name, target_parameter in (
            target_parameters.items()
        ):
            source_name, mapping_type = (
                source_for_target(
                    target_name,
                    target_parameter,
                    source_parameters,
                    source_detect_index,
                    target_detect_index,
                )
            )

            parameter_counts[
                mapping_type
            ] += 1
            parameter_numel[
                mapping_type
            ] += target_parameter.numel()

            if mapping_type == "random":
                random_parameter_names.append(
                    target_name
                )

                parts = target_name.split(".")

                if (
                    len(parts) >= 4
                    and parts[1]
                    == str(target_detect_index)
                    and parts[2]
                    in {"cv2", "cv3"}
                ):
                    group = ".".join(
                        parts[:4]
                    )
                else:
                    group = ".".join(
                        parts[:3]
                    )

                random_parameter_groups[
                    group
                ] += target_parameter.numel()

        allowed_random_prefixes = (
            "model.25.",
            "model.26.",
            "model.27.cv2.0.",
            "model.27.cv3.0.",
        )

        unexpected_random = [
            name
            for name in random_parameter_names
            if not name.startswith(
                allowed_random_prefixes
            )
        ]

        if unexpected_random:
            raise AssertionError(
                {
                    "unexpected_random":
                        unexpected_random
                }
            )

        total_parameters = sum(
            parameter.numel()
            for parameter
            in target_parameters.values()
        )
        source_total_parameters = sum(
            parameter.numel()
            for parameter
            in source_parameters.values()
        )

        loaded_parameters = (
            parameter_numel["exact"]
            + parameter_numel["cv2_shift"]
            + parameter_numel["cv3_shift"]
            + parameter_numel["dfl_shift"]
        )
        loaded_fraction = (
            loaded_parameters
            / total_parameters
        )

        if loaded_fraction < 0.97:
            raise AssertionError(
                loaded_fraction
            )

        if parameter_numel[
            "cv2_shift"
        ] != 861_120:
            raise AssertionError(
                parameter_numel["cv2_shift"]
            )

        if parameter_numel[
            "cv3_shift"
        ] != 611_568:
            raise AssertionError(
                parameter_numel["cv3_shift"]
            )



        source_wrapper.model = target_model
        source_wrapper.save(
            str(init_path)
        )

        if not init_path.is_file():
            raise FileNotFoundError(
                init_path
            )

        reloaded = YOLO(
            str(init_path),
            task="detect",
        )
        reloaded_model = reloaded.model

        reloaded_strides = [
            int(value)
            for value
            in reloaded_model.stride.tolist()
        ]

        if reloaded_strides != [
            4,
            8,
            16,
            32,
        ]:
            raise AssertionError(
                reloaded_strides
            )

        reloaded_names = (
            reloaded_model.names
        )

        if isinstance(reloaded_names, list):
            reloaded_names = {
                index: name
                for index, name
                in enumerate(reloaded_names)
            }
        else:
            reloaded_names = {
                int(index): name
                for index, name
                in reloaded_names.items()
            }

        if reloaded_names != source_names:
            raise AssertionError(
                "Reloaded COCO class names differ"
            )

        mapping_record = {
            "source_detect_index":
                source_detect_index,
            "target_detect_index":
                target_detect_index,
            "source_strides":
                source_strides,
            "target_strides":
                target_strides,
            "state_counts":
                dict(state_counts),
            "state_numel":
                dict(state_numel),
            "parameter_counts":
                dict(parameter_counts),
            "parameter_numel":
                dict(parameter_numel),
            "source_parameters":
                source_total_parameters,
            "target_parameters":
                total_parameters,
            "parameter_increase_fraction": (
                total_parameters
                / source_total_parameters
                - 1
            ),
            "loaded_parameters":
                loaded_parameters,
            "loaded_parameter_fraction":
                loaded_fraction,
            "random_parameter_groups":
                dict(
                    random_parameter_groups
                ),
            "mapping": mapping,
        }

        write_json(
            temporary_dir
            / "weight_mapping.json",
            mapping_record,
        )

        protocol = {
            "version":
                "architecture_gate_protocol_v2",
            "status":
                "p2_eligible_not_yet_adopted",
            "seed": SEED,
            "data_inputs": {
                "directory": str(V1_DIR),
                "manifest_sha256":
                    sha256(V1_MANIFEST),
                "train_core_images": 65_000,
                "train_dev_images": 5_000,
            },
            "standard_initialization": {
                "path":
                    str(OFFICIAL_WEIGHT),
                "sha256":
                    sha256(OFFICIAL_WEIGHT),
            },
            "p2_initialization": {
                "file": init_path.name,
                "sha256": sha256(init_path),
                "official_coco_names":
                    source_names,
                "loaded_parameter_fraction":
                    loaded_fraction,
            },
            "architecture": {
                "name":
                    "YOLO11m-P2-Projected-960",
                "strides":
                    target_strides,
                "parameter_increase_fraction": (
                    total_parameters
                    / source_total_parameters
                    - 1
                ),
                "gflops_increase_fraction":
                    0.3871859848104282,
            },
            "latency_pre_gate": {
                "device":
                    "NVIDIA GeForce RTX 4090",
                "precision": "FP16",
                "batch": 1,
                "image_size": 960,
                "standard_median_ms":
                    9.177048206329346,
                "p2_median_ms":
                    10.217368125915527,
                "median_increase_fraction":
                    0.11336106078953367,
                "standard_p90_ms":
                    9.951280117034912,
                "p2_p90_ms":
                    11.012976169586182,
                "p90_increase_fraction":
                    0.10668939473765038,
                "threshold": 0.30,
                "passed": True,
            },
            "training_gate": {
                "epochs": 20,
                "image_size": 960,
                "primary_delta_AP_small_min":
                    0.010,
                "delta_AP_min": -0.002,
                "delta_AP_medium_min": -0.003,
                "delta_AP_large_min": -0.005,
                "delta_AP75_min": -0.003,
            },
            "historical_bdd_best_used":
                False,
        }

        write_json(
            temporary_dir
            / "gate_protocol_v2.json",
            protocol,
        )

        metadata = {
            "created_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),
            "builder_script":
                str(Path(__file__).resolve()),
            "builder_script_sha256":
                sha256(
                    Path(__file__).resolve()
                ),
            "python_torch_version":
                torch.__version__,
            "ultralytics_version":
                ultralytics.__version__,
            "supersedes_p2_config": {
                "path": str(
                    V1_P2_CONFIG
                ),
                "sha256": sha256(
                    V1_P2_CONFIG
                ),
                "reason": (
                    "The original P2 configuration "
                    "reduced Detect classification-"
                    "tower width and prevented full "
                    "P3/P4/P5 classification-head "
                    "transfer."
                ),
            },
        }

        write_json(
            temporary_dir / "metadata.json",
            metadata,
        )

        artifact_files = sorted(
            path
            for path in temporary_dir.iterdir()
            if path.is_file()
        )

        manifest = {
            "version":
                "prepared_inputs_v2",
            "files": {
                path.name: {
                    "bytes":
                        path.stat().st_size,
                    "sha256":
                        sha256(path),
                }
                for path in artifact_files
            },
        }

        write_json(
            temporary_dir
            / "artifact_manifest.json",
            manifest,
        )

        os.replace(
            temporary_dir,
            OUTPUT_DIR,
        )

    print("\n=== P2-V2 INITIALIZATION COMPLETE ===")
    print("output:", OUTPUT_DIR)
    print(
        "p2_init:",
        OUTPUT_DIR
        / "yolo11m_p2_projected_coco_init.pt",
    )
    print(
        "p2_init_sha256:",
        sha256(
            OUTPUT_DIR
            / "yolo11m_p2_projected_coco_init.pt"
        ),
    )
    print(
        "loaded_parameter_fraction:",
        loaded_fraction,
    )
    print(
        "parameter_increase_fraction:",
        total_parameters
        / source_total_parameters
        - 1,
    )
    print(
        "manifest_sha256:",
        sha256(
            OUTPUT_DIR
            / "artifact_manifest.json"
        ),
    )
    print(
        "PASS: projected P2 COCO initialization "
        "is frozen and reloadable"
    )


if __name__ == "__main__":
    main()