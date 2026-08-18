from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os

ROOT = Path(
    "/root/rivermind-data/autodrive/results/evaluation"
)
METHOD = ROOT / "method_gate"

V1_PROTOCOL = (
    METHOD / "frozen_manifests/method_gate_protocol_v1.json"
)
SAMPLING_ROOT = METHOD / "sampling_audit_v3"
SAMPLING_MANIFEST = SAMPLING_ROOT / "artifact_manifest.json"
TRAINING_MANIFEST = (
    METHOD / "training_v1/training_artifact_manifest.json"
)
CLEAN_ROOT = METHOD / "evaluation_v2_clean"
CLEAN_MANIFEST = CLEAN_ROOT / "clean_artifact_manifest.json"
CLEAN_RESULT = CLEAN_ROOT / "clean_comparison.json"
FAILURE_ROOT = METHOD / "evaluation_v3_failures"
FAILURE_MANIFEST = FAILURE_ROOT / "artifact_manifest.json"
FAILURE_RESULT = FAILURE_ROOT / "failure_evaluation.json"
CORRUPTION_ROOT = METHOD / "evaluation_v4_corruptions"
CORRUPTION_MANIFEST = CORRUPTION_ROOT / "artifact_manifest.json"
FINAL_DECISION = CORRUPTION_ROOT / "final_gate_decision.json"

ARCHITECTURE_FREEZE = (
    ROOT / "architecture_gate/frozen_manifests/"
    "final_architecture_selection_v1.json"
)
INITIALIZATION = (
    ROOT / "architecture_gate/formal_gate_v4/"
    "p2_projected/weights/resume_epoch20.pt"
)

OUTPUT = (
    METHOD / "frozen_manifests/cafr_v2_gate_v1"
)

EXPECTED = {
    V1_PROTOCOL:
        "6be7ae25550360e38195bb785a064bd19cb9c4a65b4625b30b5698861cc1b00a",
    SAMPLING_MANIFEST:
        "7c96b15970798025e2482613c6fb0304d972e480a7166c5f2e8817b1e7ec93a9",
    TRAINING_MANIFEST:
        "c40007992f0aa49e8062734201fcd2933d5ffb7c4ca6dca3e8866757c8e74e57",
    CLEAN_MANIFEST:
        "29dc8882d383c37c6e9eb5a8cf826b9d562a3c48e9d0ef8d5a23be5b78350f97",
    FAILURE_MANIFEST:
        "2e57f0e7cb31731362511ca8d29fa9fa52328336241aea696a322eee9f5c2457",
    CORRUPTION_MANIFEST:
        "4acd7c2bebb826dfcfe7f59ee2cc753644e63419c147dfc8ea1c84d278381ce8",
    ARCHITECTURE_FREEZE:
        "b3a20fc148a508334268aa4d43bc251219244dffba134173dee3ac1648de29f0",
    INITIALIZATION:
        "642a6b4009f085649b6da3f1647821d3e760e6724b019e08d022080c8050bc29",
}

V1_QUOTAS = {
    "small_joint_not_detected": 7800,
    "dfine_supported_yolo_failure": 3900,
    "yolo_geometric_failure": 1950,
    "yolo_low_confidence": 1950,
    "yolo_classification_failure": 1365,
    "other_joint_not_detected": 975,
    "small_high_or_critical": 975,
    "other_high_or_critical": 585,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)

    actual = sha256(path)
    print(path)
    print(" expected:", expected)
    print(" actual  :", actual)

    if actual != expected:
        raise AssertionError(path)


def verify_manifest(
    root: Path,
    manifest_path: Path,
    ignored_files: set[str] | None = None,
) -> dict:
    manifest = json.loads(manifest_path.read_text())
    ignored_files = ignored_files or set()

    for relative, record in manifest["files"].items():
        if relative in ignored_files:
            print(
                "SKIP mutable post-manifest state:",
                root / relative,
            )
            continue

        path = root / relative
        verify(path, record["sha256"])
        assert path.stat().st_size == int(record["bytes"])

    return manifest


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def load_and_validate():
    print("=== CAFR V2 FROZEN-INPUT PREFLIGHT ===")

    for path, expected in EXPECTED.items():
        verify(path, expected)

    sampling_manifest = verify_manifest(
        SAMPLING_ROOT,
        SAMPLING_MANIFEST,
    )
    verify_manifest(
        METHOD / "training_v1",
        TRAINING_MANIFEST,
        ignored_files={"controller_state.json"},
    )

    controller_state_path = (
        METHOD / "training_v1/controller_state.json"
    )
    controller_state = json.loads(
        controller_state_path.read_text()
    )

    assert controller_state["status"] == (
        "training_complete_pending_evaluation"
    )
    assert controller_state["current_arm"] is None
    assert controller_state["completed_arms"] == [
        "uniform",
        "scalar_risk",
        "typed_cafr",
    ]

    for arm in (
        "uniform",
        "scalar_risk",
        "typed_cafr",
    ):
        assert (
            controller_state["arm_states"][arm]["status"]
            == "complete"
        )
        assert (
            controller_state["arm_states"][arm]["epochs"]
            == 10
        )

    print(
        "PASS: mutable controller state independently "
        "confirms all three arms complete"
    )
    print(
        "controller_state_current_sha256:",
        sha256(controller_state_path),
    )
    verify_manifest(
        CLEAN_ROOT,
        CLEAN_MANIFEST,
    )
    verify_manifest(
        FAILURE_ROOT,
        FAILURE_MANIFEST,
    )
    verify_manifest(
        CORRUPTION_ROOT,
        CORRUPTION_MANIFEST,
    )

    v1 = json.loads(V1_PROTOCOL.read_text())
    clean = json.loads(CLEAN_RESULT.read_text())
    failure = json.loads(FAILURE_RESULT.read_text())
    decision = json.loads(FINAL_DECISION.read_text())

    assert v1["version"] == "method_gate_protocol_v1"
    assert v1["research_role"][
        "primary_theme"
    ] == "object-level failure diagnosis and targeted repair"

    assert v1["arms"]["typed_cafr"]["views"] == 65000
    assert (
        v1["arms"]["typed_cafr"]["object_centric_views"]
        == 19500
    )
    assert sum(V1_QUOTAS.values()) == 19500

    typed_record = sampling_manifest["files"][
        "typed_cafr_selection.csv"
    ]
    typed_path = (
        SAMPLING_ROOT / "typed_cafr_selection.csv"
    )
    assert sha256(typed_path) == typed_record["sha256"]

    delta_small = clean["comparisons"][
        "typed_cafr_minus_uniform"
    ]["delta_AP_small"]

    failure_delta = failure["comparisons"][
        "typed_cafr_minus_uniform"
    ]["delta_failure_macro_AR100"]

    small_joint_delta = failure["comparisons"][
        "typed_cafr_minus_uniform"
    ]["delta_small_joint_not_detected_recall50"]

    typed_scalar_failure = failure["comparisons"][
        "typed_cafr_minus_scalar_risk"
    ]["delta_failure_macro_AR100"]

    typed_scalar_low_conf = failure["comparisons"][
        "typed_cafr_minus_scalar_risk"
    ]["delta_yolo_low_confidence_AR100"]

    corruption = decision["corruption_guardrail"][
        "delta_mean_corruption_AP_small"
    ]

    assert delta_small < 0.002
    assert failure_delta >= 0.005
    assert small_joint_delta >= 0.005
    assert typed_scalar_failure >= 0.002
    assert typed_scalar_low_conf < 0.0
    assert decision["gate_result"] == "FAIL"
    assert decision["advance_typed_cafr_v1"] is False
    assert decision["corruption_guardrail"]["passed"] is True

    evidence = {
        "clean_delta_AP_small_typed_minus_uniform":
            delta_small,
        "failure_macro_AR100_typed_minus_uniform":
            failure_delta,
        "small_joint_recall50_typed_minus_uniform":
            small_joint_delta,
        "failure_macro_AR100_typed_minus_scalar":
            typed_scalar_failure,
        "low_confidence_AR100_typed_minus_scalar":
            typed_scalar_low_conf,
        "corruption_AP_small_typed_minus_uniform":
            corruption,
    }

    return v1, sampling_manifest, evidence


def build_protocol(v1, sampling_manifest, evidence):
    materialization = copy.deepcopy(
        v1["materialization"]
    )

    old_policies = copy.deepcopy(
        materialization["typed_crop_policies"]
    )
    new_policies = copy.deepcopy(old_policies)


    new_policies["small_joint_not_detected"] = [
        192, 560, 7.0, 14.0
    ]


    new_policies["yolo_low_confidence"] = [
        288, 680, 10.0, 18.0
    ]



    new_policies["small_high_or_critical"] = [
        128, 360, 7.0, 12.0
    ]

    materialization["typed_crop_policies"] = new_policies

    typed_record = sampling_manifest["files"][
        "typed_cafr_selection.csv"
    ]
    common_record = sampling_manifest["files"][
        "common_base_image_ids.json"
    ]

    protocol = {
        "version": "cafr_v2_gate_protocol_v1",
        "status": "frozen_before_v2_materialization_and_training",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "research_role": {
            "primary_theme":
                "object-level failure diagnosis and targeted repair",
            "method": "CAFR-V2-context-balanced",
            "purpose":
                "Single corrective gate after CAFR V1 missed only "
                "the frozen clean AP_small progress threshold.",
            "formal_final_model": False,
            "gate_checkpoint_initializes_formal_training": False,
        },
        "v1_diagnosis": {
            "gate_result": "FAIL",
            "failed_rule":
                "delta_AP_small >= 0.002",
            "failure_and_corruption_direction_supported": True,
            "evidence": evidence,
        },
        "control_preservation": {
            "reuse_exact_v1_typed_target_selection": True,
            "reuse_exact_v1_common_original_images": True,
            "typed_quotas_changed": False,
            "reason_quotas_not_changed":
                "Changing quotas would invalidate the existing "
                "class-scale-risk matched Scalar control and require "
                "another Scalar training run.",
            "typed_quotas": V1_QUOTAS,
            "total_views": 65000,
            "common_original_views": 45500,
            "object_centric_views": 19500,
            "only_training_variable_changed":
                "crop geometry for three diagnosed failure groups",
        },
        "sampling": {
            "typed_selection":
                str(
                    SAMPLING_ROOT /
                    "typed_cafr_selection.csv"
                ),
            "typed_selection_sha256":
                typed_record["sha256"],
            "typed_selection_bytes":
                typed_record["bytes"],
            "common_base":
                str(
                    SAMPLING_ROOT /
                    "common_base_image_ids.json"
                ),
            "common_base_sha256":
                common_record["sha256"],
            "common_base_bytes":
                common_record["bytes"],
        },
        "crop_policy_amendment": {
            "old": {
                key: old_policies[key]
                for key in (
                    "small_joint_not_detected",
                    "yolo_low_confidence",
                    "small_high_or_critical",
                )
            },
            "new": {
                key: new_policies[key]
                for key in (
                    "small_joint_not_detected",
                    "yolo_low_confidence",
                    "small_high_or_critical",
                )
            },
            "unchanged_groups": sorted(
                set(old_policies)
                - {
                    "small_joint_not_detected",
                    "yolo_low_confidence",
                    "small_high_or_critical",
                }
            ),
        },
        "materialization": materialization,
        "initialization": copy.deepcopy(
            v1["initialization"]
        ),
        "training": copy.deepcopy(
            v1["training"]
        ),
        "evaluation": copy.deepcopy(
            v1["evaluation"]
        ),
        "advance_only_if": copy.deepcopy(
            v1["advance_typed_cafr_only_if"]
        ),
        "execution_policy": {
            "train_arms": ["typed_cafr_v2"],
            "gate_epochs": 10,
            "uniform_retrained": False,
            "scalar_retrained": False,
            "comparators":
                "existing frozen Uniform and matched Scalar V1",
            "maximum_corrective_iterations": 1,
            "further_v3_allowed": False,
            "if_v2_fails":
                "Do not tune again on frozen train-dev; stop before "
                "formal long training and report the bounded result.",
        },
        "dependencies": {
            str(path): expected
            for path, expected in EXPECTED.items()
        },
    }

    assert protocol["training"]["gate_finetune_epochs"] == 10
    assert protocol["training"]["seed"] == 20260809
    assert protocol["training"]["deterministic"] is True
    assert protocol["training"]["optimizer"] == "SGD"
    assert protocol["training"]["resume_old_optimizer"] is False

    return protocol


def preflight():
    v1, sampling_manifest, evidence = (
        load_and_validate()
    )

    protocol = build_protocol(
        v1,
        sampling_manifest,
        evidence,
    )

    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {OUTPUT}"
        )

    incomplete = list(
        OUTPUT.parent.glob(
            f"{OUTPUT.name}.incomplete-*"
        )
    )
    if incomplete:
        raise FileExistsError(incomplete)

    print("\n=== CAFR V2 AMENDMENT ===")
    print(json.dumps(
        protocol["crop_policy_amendment"],
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ))
    print("PASS: V2 retains exact V1 sampling and matched controls")
    print("PASS: only three crop-geometry policies are amended")
    print("NOTHING WRITTEN OR TRAINED")

    return protocol


def execute():
    protocol = preflight()

    staging = OUTPUT.with_name(
        f"{OUTPUT.name}.incomplete-{os.getpid()}"
    )
    staging.mkdir(parents=False, exist_ok=False)

    protocol_path = staging / "cafr_v2_gate_protocol.json"
    write_json(protocol_path, protocol)

    manifest = {
        "version": "cafr_v2_gate_freeze_v1",
        "status": "frozen",
        "files": {
            protocol_path.name: {
                "bytes": protocol_path.stat().st_size,
                "sha256": sha256(protocol_path),
            }
        },
    }

    manifest_path = staging / "artifact_manifest.json"
    write_json(manifest_path, manifest)

    os.replace(staging, OUTPUT)

    print("\n" + "=" * 96)
    print("CAFR V2 GATE PROTOCOL FROZEN")
    print("=" * 96)
    print(
        "protocol:",
        OUTPUT / "cafr_v2_gate_protocol.json",
    )
    print(
        "protocol_sha256:",
        sha256(
            OUTPUT / "cafr_v2_gate_protocol.json"
        ),
    )
    print(
        "manifest:",
        OUTPUT / "artifact_manifest.json",
    )
    print(
        "manifest_sha256:",
        sha256(
            OUTPUT / "artifact_manifest.json"
        ),
    )
    print("PASS: CAFR V2 is immutable before materialization")
    print("NOTHING MATERIALIZED OR TRAINED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.execute:
        execute()
    else:
        preflight()


if __name__ == "__main__":
    main()