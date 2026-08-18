from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

ORIGINAL_CONTROLLER = (
    PROJECT_ROOT
    / "source-code"
    / "training"
    / "architecture-gate"
    / "run_architecture_gate_original_exporter.py"
)
FIXED_CONTROLLER = (
    PROJECT_ROOT
    / "source-code"
    / "training"
    / "architecture-gate"
    / "run_architecture_gate_streaming.py"
)

EXPECTED_ORIGINAL_SHA256 = (
    "79135bb1245ef26e23eaecfb80513f391"
    "6dde876ae12826ec9428da2efffcb41"
)
EXPECTED_FIXED_SHA256 = (
    "fd48e9575a9ef8606e2c20c882ec2dff"
    "6552e10d19016f43da8dbfd78547345b"
)
EXPECTED_STANDARD_EPOCH10_SHA256 = (
    "7dfc9291bf5a866cf4d8cc83f0f4dc7d"
    "cf9838ec72bf276937f9041f78e8a106"
)

FORMAL_ROOT = Path(
    "/root/rivermind-data/autodrive/results/evaluation/"
    "architecture_gate/formal_gate_v4"
)
EVALUATIONS_DIR = FORMAL_ROOT / "evaluations"

STATE_PATH = FORMAL_ROOT / "controller_state.json"
RUN_METADATA_PATH = FORMAL_ROOT / "run_metadata.json"

STANDARD_EPOCH10_CHECKPOINT = (
    FORMAL_ROOT
    / "standard/weights/resume_epoch10.pt"
)
STANDARD_EPOCH10_RECORD = (
    FORMAL_ROOT
    / "standard/epoch10_checkpoint.json"
)

FAILED_STAGING_ORIGINAL = (
    EVALUATIONS_DIR
    / "standard_epoch10.incomplete-1553"
)
FAILED_STAGING_ARCHIVE = (
    EVALUATIONS_DIR
    / "standard_epoch10.failed_oom_v1"
)

RECOVERY_RECORD = (
    FORMAL_ROOT
    / "recovery_after_standard_eval_oom_v1.json"
)


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def load_fixed_controller():
    if not ORIGINAL_CONTROLLER.is_file():
        raise FileNotFoundError(ORIGINAL_CONTROLLER)
    if not FIXED_CONTROLLER.is_file():
        raise FileNotFoundError(FIXED_CONTROLLER)

    original_hash = sha256(ORIGINAL_CONTROLLER)
    fixed_hash = sha256(FIXED_CONTROLLER)

    if original_hash != EXPECTED_ORIGINAL_SHA256:
        raise AssertionError(
            f"Original controller SHA256 mismatch: "
            f"{original_hash}"
        )
    if fixed_hash != EXPECTED_FIXED_SHA256:
        raise AssertionError(
            f"Fixed controller SHA256 mismatch: "
            f"{fixed_hash}"
        )

    specification = importlib.util.spec_from_file_location(
        "formal_gate_v4_fixed_controller",
        FIXED_CONTROLLER,
    )
    if specification is None:
        raise RuntimeError("Unable to create module specification")
    if specification.loader is None:
        raise RuntimeError("Controller module loader is unavailable")

    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def validate_recovery_state(gate) -> None:
    if gate.FORMAL_ROOT.resolve() != FORMAL_ROOT.resolve():
        raise AssertionError(
            f"FORMAL_ROOT mismatch: {gate.FORMAL_ROOT}"
        )

    if not FORMAL_ROOT.is_dir():
        raise FileNotFoundError(FORMAL_ROOT)
    if not EVALUATIONS_DIR.is_dir():
        raise FileNotFoundError(EVALUATIONS_DIR)
    if not STATE_PATH.is_file():
        raise FileNotFoundError(STATE_PATH)
    if not RUN_METADATA_PATH.is_file():
        raise FileNotFoundError(RUN_METADATA_PATH)

    state = load_json(STATE_PATH)
    expected_state = {
        "stage": "standard_evaluation_10",
        "status": "running",
    }
    if state != expected_state:
        raise AssertionError(
            f"Unexpected controller state: {state}"
        )

    run_metadata = load_json(RUN_METADATA_PATH)
    if (
        run_metadata.get("controller_script_sha256")
        != EXPECTED_ORIGINAL_SHA256
    ):
        raise AssertionError(
            "Original run metadata controller hash mismatch"
        )

    if not STANDARD_EPOCH10_CHECKPOINT.is_file():
        raise FileNotFoundError(
            STANDARD_EPOCH10_CHECKPOINT
        )
    if not STANDARD_EPOCH10_RECORD.is_file():
        raise FileNotFoundError(
            STANDARD_EPOCH10_RECORD
        )

    checkpoint_hash = sha256(
        STANDARD_EPOCH10_CHECKPOINT
    )
    if checkpoint_hash != EXPECTED_STANDARD_EPOCH10_SHA256:
        raise AssertionError(
            f"Standard epoch-10 checkpoint hash mismatch: "
            f"{checkpoint_hash}"
        )

    gate.validate_resume_checkpoint(
        STANDARD_EPOCH10_CHECKPOINT,
        10,
        gate.DATA_YAML,
    )

    formal_standard_evaluation = (
        EVALUATIONS_DIR / "standard_epoch10"
    )
    if formal_standard_evaluation.exists():
        raise FileExistsError(
            formal_standard_evaluation
        )

    if (FORMAL_ROOT / "p2_projected").exists():
        raise FileExistsError(
            FORMAL_ROOT / "p2_projected"
        )
    if (FORMAL_ROOT / "epoch10_decision.json").exists():
        raise FileExistsError(
            FORMAL_ROOT / "epoch10_decision.json"
        )
    if (FORMAL_ROOT / "final_decision.json").exists():
        raise FileExistsError(
            FORMAL_ROOT / "final_decision.json"
        )

    source_exists = FAILED_STAGING_ORIGINAL.exists()
    archive_exists = FAILED_STAGING_ARCHIVE.exists()

    if source_exists == archive_exists:
        raise AssertionError(
            "Exactly one failed-staging location must exist: "
            f"source={source_exists}, archive={archive_exists}"
        )

    failed_directory = (
        FAILED_STAGING_ORIGINAL
        if source_exists
        else FAILED_STAGING_ARCHIVE
    )
    if not failed_directory.is_dir():
        raise NotADirectoryError(failed_directory)
    if any(failed_directory.iterdir()):
        raise AssertionError(
            f"Failed staging directory is not empty: "
            f"{failed_directory}"
        )

    print("controller_state:", state)
    print(
        "standard_epoch10_sha256:",
        checkpoint_hash,
    )
    print(
        "failed_staging:",
        failed_directory,
    )
    print("PASS: exact recovery state verified")


def archive_failed_staging() -> None:
    if FAILED_STAGING_ORIGINAL.exists():
        if FAILED_STAGING_ARCHIVE.exists():
            raise FileExistsError(FAILED_STAGING_ARCHIVE)
        if any(FAILED_STAGING_ORIGINAL.iterdir()):
            raise AssertionError(
                "Refusing to archive non-empty failed staging"
            )
        FAILED_STAGING_ORIGINAL.rename(
            FAILED_STAGING_ARCHIVE
        )
    elif not FAILED_STAGING_ARCHIVE.exists():
        raise FileNotFoundError(
            FAILED_STAGING_ORIGINAL
        )

    if not FAILED_STAGING_ARCHIVE.is_dir():
        raise NotADirectoryError(
            FAILED_STAGING_ARCHIVE
        )
    if any(FAILED_STAGING_ARCHIVE.iterdir()):
        raise AssertionError(
            "Archived staging directory must remain empty"
        )

    print(
        "archived_failed_staging:",
        FAILED_STAGING_ARCHIVE,
    )


def write_recovery_record(gate) -> None:
    if RECOVERY_RECORD.exists():
        existing = load_json(RECOVERY_RECORD)
        if existing.get("status") != "applied":
            raise AssertionError(existing)
        if (
            existing.get("fixed_controller_sha256")
            != EXPECTED_FIXED_SHA256
        ):
            raise AssertionError(existing)
        if (
            existing.get(
                "standard_epoch10_checkpoint_sha256"
            )
            != EXPECTED_STANDARD_EPOCH10_SHA256
        ):
            raise AssertionError(existing)
        print(
            "PASS: existing recovery record verified:",
            RECOVERY_RECORD,
        )
        return

    record = {
        "version":
            "formal_gate_v4_standard_eval_oom_recovery_v1",
        "status": "applied",
        "applied_at":
            datetime.now().astimezone().isoformat(),
        "failure_stage": "standard_evaluation_10",
        "failure_type":
            "ultralytics_source_list_giant_warmup_batch_oom",
        "failure_occurred_before_first_evaluation_result":
            True,
        "training_repeated": False,
        "protocol_changed": False,
        "training_arguments_changed": False,
        "architecture_rules_changed": False,
        "evaluation_metrics_changed": False,
        "evaluation_input_transport_change": {
            "before": "source=image_paths",
            "after": "source=str(dev_list)",
            "purpose":
                "Stream the frozen 5k TXT list at batch=1",
        },
        "original_controller":
            str(ORIGINAL_CONTROLLER),
        "original_controller_sha256":
            EXPECTED_ORIGINAL_SHA256,
        "fixed_controller":
            str(FIXED_CONTROLLER),
        "fixed_controller_sha256":
            EXPECTED_FIXED_SHA256,
        "recovery_controller":
            str(Path(__file__).resolve()),
        "recovery_controller_sha256":
            sha256(Path(__file__).resolve()),
        "standard_epoch10_checkpoint":
            str(STANDARD_EPOCH10_CHECKPOINT),
        "standard_epoch10_checkpoint_sha256":
            EXPECTED_STANDARD_EPOCH10_SHA256,
        "failed_staging_original":
            str(FAILED_STAGING_ORIGINAL),
        "failed_staging_archive":
            str(FAILED_STAGING_ARCHIVE),
        "failed_staging_was_empty": True,
        "resume_stage":
            "standard_evaluation_10",
    }

    gate.atomic_write_json(
        RECOVERY_RECORD,
        record,
    )
    print("recovery_record:", RECOVERY_RECORD)
    print(
        "recovery_record_sha256:",
        sha256(RECOVERY_RECORD),
    )


def continue_gate(gate, protocol: dict) -> None:
    standard_epoch10_checkpoint = {
        "path": str(STANDARD_EPOCH10_CHECKPOINT),
        "sha256":
            EXPECTED_STANDARD_EPOCH10_SHA256,
    }

    print(
        "\n=== RECOVERING STANDARD EPOCH-10 "
        "EVALUATION ==="
    )
    standard_metrics_10 = gate.evaluate_checkpoint(
        protocol,
        "standard",
        10,
        STANDARD_EPOCH10_CHECKPOINT,
    )

    gate.atomic_write_json(
        STATE_PATH,
        {
            "status": "running",
            "stage": "p2_epoch_10",
        },
    )
    p2_epoch10_checkpoint = gate.train_to_epoch_10(
        protocol,
        "p2_projected",
    )

    gate.atomic_write_json(
        STATE_PATH,
        {
            "status": "running",
            "stage": "p2_evaluation_10",
        },
    )
    p2_metrics_10 = gate.evaluate_checkpoint(
        protocol,
        "p2_projected",
        10,
        Path(p2_epoch10_checkpoint["path"]),
    )

    epoch10_decision = gate.build_epoch_10_decision(
        protocol,
        standard_metrics_10,
        p2_metrics_10,
    )
    gate.atomic_write_json(
        FORMAL_ROOT / "epoch10_decision.json",
        epoch10_decision,
    )

    print("\n=== EPOCH-10 DECISION ===")
    print(json.dumps(
        epoch10_decision,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ))

    if epoch10_decision["early_reject_p2"]:
        final_decision = {
            "version": "architecture_gate_final_v4",
            "decision_epoch": 10,
            "reason": "p2_early_rejection_rule_passed",
            "selected_architecture": "standard",
            "selected_architecture_name":
                protocol["architectures"]
                ["standard"]["name"],
            "epoch10_decision": epoch10_decision,
            "gate_checkpoint_is_formal_final_model":
                False,
            "extension_to_epoch_20_executed":
                False,
            "extension_to_epoch_30_allowed":
                False,
        }
        gate.atomic_write_json(
            FORMAL_ROOT / "final_decision.json",
            final_decision,
        )
        gate.atomic_write_json(
            STATE_PATH,
            {
                "status": "complete",
                "stage": "p2_rejected_at_epoch_10",
                "selected_architecture": "standard",
            },
        )

        manifest = gate.build_final_manifest()
        print("\n=== FORMAL GATE COMPLETE ===")
        print("selected_architecture: standard")
        print("decision_epoch: 10")
        print("manifest:", manifest)
        print(
            "manifest_sha256:",
            gate.sha256(manifest),
        )
        return

    gate.atomic_write_json(
        STATE_PATH,
        {
            "status": "running",
            "stage": "standard_resume_to_epoch_20",
        },
    )
    standard_epoch20_checkpoint = (
        gate.resume_to_epoch_20(
            protocol,
            "standard",
            Path(
                standard_epoch10_checkpoint["path"]
            ),
        )
    )

    gate.atomic_write_json(
        STATE_PATH,
        {
            "status": "running",
            "stage": "standard_evaluation_20",
        },
    )
    standard_metrics_20 = gate.evaluate_checkpoint(
        protocol,
        "standard",
        20,
        Path(standard_epoch20_checkpoint["path"]),
    )

    gate.atomic_write_json(
        STATE_PATH,
        {
            "status": "running",
            "stage": "p2_resume_to_epoch_20",
        },
    )
    p2_epoch20_checkpoint = gate.resume_to_epoch_20(
        protocol,
        "p2_projected",
        Path(p2_epoch10_checkpoint["path"]),
    )

    gate.atomic_write_json(
        STATE_PATH,
        {
            "status": "running",
            "stage": "p2_evaluation_20",
        },
    )
    p2_metrics_20 = gate.evaluate_checkpoint(
        protocol,
        "p2_projected",
        20,
        Path(p2_epoch20_checkpoint["path"]),
    )

    final_decision = gate.build_epoch_20_decision(
        protocol,
        standard_metrics_20,
        p2_metrics_20,
    )
    gate.atomic_write_json(
        FORMAL_ROOT / "final_decision.json",
        final_decision,
    )
    gate.atomic_write_json(
        STATE_PATH,
        {
            "status": "complete",
            "stage": "epoch_20_decision_complete",
            "selected_architecture":
                final_decision[
                    "selected_architecture"
                ],
        },
    )

    manifest = gate.build_final_manifest()

    print("\n=== FORMAL GATE COMPLETE ===")
    print(
        "selected_architecture:",
        final_decision["selected_architecture"],
    )
    print("decision_epoch: 20")
    print("manifest:", manifest)
    print(
        "manifest_sha256:",
        gate.sha256(manifest),
    )
    print(json.dumps(
        final_decision,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ))


def preflight():
    print("=== RECOVERY CONTROLLER PREFLIGHT ===")
    gate = load_fixed_controller()
    protocol = gate.preflight()
    validate_recovery_state(gate)

    print("original_controller_sha256:",
          sha256(ORIGINAL_CONTROLLER))
    print("fixed_controller_sha256:",
          sha256(FIXED_CONTROLLER))
    print("recovery_controller_sha256:",
          sha256(Path(__file__).resolve()))
    print("PASS: recovery preflight complete")
    return gate, protocol


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resume formal gate V4 after the Standard "
            "epoch-10 evaluation input-list OOM."
        )
    )
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
    args = parser.parse_args()

    gate, protocol = preflight()

    if args.preflight_only:
        print("NOTHING TRAINED")
        return

    archive_failed_staging()
    write_recovery_record(gate)
    continue_gate(gate, protocol)


if __name__ == "__main__":
    main()