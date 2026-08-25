from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(
    "/root/rivermind-data/autodrive/code/"
    "robust-autonomous-driving-perception"
)
RESULTS = Path(
    "/root/rivermind-data/autodrive/results/evaluation"
)
DATASET = Path(
    "/root/rivermind-data/autodrive/datasets/datasets/"
    "bdd100k_final"
)

OUTPUT_DIR = (
    RESULTS
    / "final_analysis_v1"
    / "frozen_protocol"
)

EXPECTED = {
    "official_val_gt": (
        DATASET / "coco/annotations/instances_val.json",
        "7be8a02e147743123a4a62b2c92ed2e4a5712d261c8868223e6325bad6e346d6",
    ),
    "train_gt": (
        DATASET / "coco/annotations/instances_train.json",
        "a8e9c188e58a328901af6e3a93b747d517da354e2dab3292ff26d7c0fbd32cd2",
    ),
    "baseline_yolo_predictions": (
        RESULTS
        / "unified_detection/canonical_predictions/YOLO11m.json",
        "14ff8491f7849077cb7ead3ce7d9e618d2f92ee04bef0a8c68efdb1b836a54b0",
    ),
    "baseline_dfine_predictions": (
        RESULTS
        / "unified_detection/canonical_predictions/D-FINE-M.json",
        "10b4159d0f52e412714ac9ae076ac574ebbb74321ddcebab03e2bdbad55e3ebd",
    ),
    "architecture_decision": (
        RESULTS
        / "architecture_gate/borderline_adjudication_v5_1/"
        "formal_evaluation/final_decision.json",
        "e22114f58da2514ed81c9c150013cd7f6a410aead701540b6c7870f9a188e357",
    ),
    "architecture_selection_freeze": (
        RESULTS
        / "architecture_gate/frozen_manifests/"
        "final_architecture_selection_v1.json",
        "b3a20fc148a508334268aa4d43bc251219244dffba134173dee3ac1648de29f0",
    ),
    "object_risk_freeze": (
        RESULTS
        / "train_risk_labels/frozen_manifests/"
        "object_risk_v1_freeze.json",
        "b5e1934ee9108e29548288e6f98cc611bb424c2f6d5b0ab99329b490699ed9d6",
    ),
    "object_risk_manifest": (
        RESULTS
        / "train_risk_labels/object_risk_v1/"
        "artifact_manifest.json",
        "f70d77c963f114f1cf05feebf72a0cbe93de43bb1c23df3b5a524329f23031c1",
    ),
    "object_risk_metadata": (
        RESULTS
        / "train_risk_labels/object_risk_v1/"
        "metadata.json",
        "95589945570a9eca8ce23503aed11c3f840429e4cd9faea74bddbe4cfd220416",
    ),
    "operating_points": (
        RESULTS
        / "train_risk_labels/operating_point_calibration/"
        "operating_points.json",
        "f13868c935ea16091fd0bae592bb9817d77bd58d2f45597dc7d2add8e2fa44ae",
    ),
    "yolo_calibration": (
        RESULTS
        / "train_risk_labels/operating_point_calibration/"
        "YOLO11m_calibration.json",
        "cb77222651c4de92996d57a28bfaee89065c9d1e7023dc30ed430780540d9c97",
    ),
    "dfine_calibration": (
        RESULTS
        / "train_risk_labels/operating_point_calibration/"
        "D-FINE-M_calibration.json",
        "e933d2ed534a0d0f1f6418edd1fe0b5f8c3cfafcae318215fc76a4a315a440c0",
    ),
    "yolo_threshold_sensitivity": (
        RESULTS
        / "train_risk_labels/operating_point_calibration/"
        "YOLO11m_threshold_sensitivity.csv",
        "e8a77870d1f27d889bb6c465c5259256452273a4f7824c0a74e319c56fd5df22",
    ),
    "dfine_threshold_sensitivity": (
        RESULTS
        / "train_risk_labels/operating_point_calibration/"
        "D-FINE-M_threshold_sensitivity.csv",
        "d47c36fd350b4c6b5318787403f39da07a056712db86fe75d08e8638211c997d",
    ),
    "cafr_sampling_protocol": (
        RESULTS
        / "method_gate/frozen_manifests/"
        "cafr_sampling_protocol_v1.json",
        "3812c5692d63629602ebb0393d6369c49f5d84cb5b467bde47a457741a65967d",
    ),
    "method_gate_protocol": (
        RESULTS
        / "method_gate/frozen_manifests/"
        "method_gate_protocol_v1.json",
        "6be7ae25550360e38195bb785a064bd19cb9c4a65b4625b30b5698861cc1b00a",
    ),
    "method_training_manifest": (
        RESULTS
        / "method_gate/training_v1/"
        "training_artifact_manifest.json",
        "c40007992f0aa49e8062734201fcd2933d5ffb7c4ca6dca3e8866757c8e74e57",
    ),
    "cafr_v2_training_manifest": (
        RESULTS
        / "method_gate/cafr_v2_training_v1/"
        "training_artifact_manifest.json",
        "888c6429f6108dfa8e78577922607b50c3c904a76412a6b3a312a57c36dbcd6d",
    ),
    "method_clean_manifest": (
        RESULTS
        / "method_gate/evaluation_v2_clean/"
        "clean_artifact_manifest.json",
        "29dc8882d383c37c6e9eb5a8cf826b9d562a3c48e9d0ef8d5a23be5b78350f97",
    ),
    "method_failure_manifest": (
        RESULTS
        / "method_gate/evaluation_v3_failures/"
        "artifact_manifest.json",
        "2e57f0e7cb31731362511ca8d29fa9fa52328336241aea696a322eee9f5c2457",
    ),
    "method_corruption_manifest": (
        RESULTS
        / "method_gate/evaluation_v4_corruptions/"
        "artifact_manifest.json",
        "4acd7c2bebb826dfcfe7f59ee2cc753644e63419c147dfc8ea1c84d278381ce8",
    ),
    "cafr_v2_clean_manifest": (
        RESULTS
        / "method_gate/evaluation_v5_cafr_v2_clean/"
        "clean_artifact_manifest.json",
        "c55efe0d1faa28a2dbd7e8d1bfd2fe8c3e9bf9e3cc38c5b63e4501982a571432",
    ),
    "common40_manifest": (
        RESULTS
        / "formal_training/training_v1/"
        "common_uniform_e1_40/artifact_manifest.json",
        "5fff0d57aa336cf36074cf8da3aff1b3c0d58a169f5bc924d5ebaf978a6f77b7",
    ),
    "formal_branches_manifest": (
        RESULTS
        / "formal_training/training_v2/"
        "formal_branches_manifest.json",
        "64dd9fdfc15db3fae8881a63f86254899187f6f6570b6bfa89129799bb589cd0",
    ),
    "trajectory_protocol": (
        RESULTS
        / "formal_training/trajectory_evaluation_v1/"
        "trajectory_protocol.json",
        "7dc023019c6c6945f837a1608847ec67f80efa4e6b505b0d11a48d2963e68afb",
    ),
    "trajectory_manifest": (
        RESULTS
        / "formal_training/trajectory_evaluation_v1/"
        "artifact_manifest.json",
        "922f27c7b573e500f449809591af7e7886449b62e9f662ba50609ac1db695485",
    ),
    "formal_input_manifest": (
        RESULTS
        / "formal_training/prepared_inputs_v1/"
        "artifact_manifest.json",
        "ccf01cc4092237665dc4d73a587bd2a4f9a454cf7fceb0b4f17712ab8c6854aa",
    ),
    "formal_method_selection": (
        RESULTS
        / "formal_training/prepared_inputs_v1/"
        "formal_method_selection.json",
        "ab2607964e4000453d699a23c7a4868bd71802ff1d132455b34a6df001b8dfbe",
    ),
    "formal_training_protocol": (
        RESULTS
        / "formal_training/prepared_inputs_v1/"
        "formal_training_protocol.json",
        "c40ad7567ebddd44d96390aa5e3c0340a19cbe6f84f83c944d0442ae69205a14",
    ),
    "uniform_yaml": (
        RESULTS
        / "formal_training/prepared_inputs_v1/"
        "uniform_full70k.yaml",
        "9cf4e0be79b3b78e48c2bb149dad1e0daf1f7f443e1ccf197b91ee9f1f04dcee",
    ),
    "uniform_train_list": (
        RESULTS
        / "formal_training/prepared_inputs_v1/"
        "uniform_full70k_train.txt",
        "d86707c944b4802105c11438621847e50b0af8512ee8ff9dce19a290adc4542c",
    ),
    "risk_yaml": (
        RESULTS
        / "formal_training/prepared_inputs_v1/"
        "rebu_risk_full70k.yaml",
        "237e2ac21ab0b7a5115bb0cd08c6722f22e2e4122756f3e96ea205e371f7b286",
    ),
    "risk_train_list": (
        RESULTS
        / "formal_training/prepared_inputs_v1/"
        "rebu_risk_full70k_train.txt",
        "7a4c8f01a1ecc1ca2c4d6b4c6cd5a0afd341c2a091e8102e08995bd0351dfb92",
    ),
    "standard_e20": (
        RESULTS
        / "architecture_gate/formal_gate_v4/"
        "standard/weights/resume_epoch20.pt",
        "dc1e3ef68e233a2fad4911955bbe178f0104cd9000dcbdc772dac8d0a58438b5",
    ),
    "p2_e20": (
        RESULTS
        / "architecture_gate/formal_gate_v4/"
        "p2_projected/weights/resume_epoch20.pt",
        "642a6b4009f085649b6da3f1647821d3e760e6724b019e08d022080c8050bc29",
    ),
    "common40": (
        RESULTS
        / "formal_training/training_v1/"
        "common_uniform_e1_40/weights/common40_ema.pt",
        "c05ed70a608d026fef6178f40a4dee62e96323731700b42e426c311dd78b27f3",
    ),
    "gate_uniform": (
        RESULTS / "method_gate/training_v1/uniform/weights/last.pt",
        "45f397e81cf6aa3e8fbb08a8fef79b8e95975f85f5b360fa9f4bf2a4f6757210",
    ),
    "gate_scalar": (
        RESULTS / "method_gate/training_v1/scalar_risk/weights/last.pt",
        "3093b249d8052924d8324c826731a77b82ab98327a41ff68ab3a2ad615f74b9d",
    ),
    "gate_typed": (
        RESULTS / "method_gate/training_v1/typed_cafr/weights/last.pt",
        "217b6d453d399c345a7e1be5595783044f2555a65ae95f047d6a7278d105c4ee",
    ),
    "gate_typed_v2": (
        RESULTS
        / "method_gate/cafr_v2_training_v1/"
        "typed_cafr/weights/last.pt",
        "eaa3dd7eee88f7d0c74cfbe45f6237c82ecb4eedacb9615a260ce351bc2da7dd",
    ),
    "uniform_e100": (
        RESULTS
        / "formal_training/training_v2/"
        "uniform_e41_100/weights/uniform100_ema.pt",
        "2af91824191b0d48d40e35856abafd388776815aaf0a629ace4e734a9be01cf7",
    ),
    "risk_e100": (
        RESULTS
        / "formal_training/training_v2/"
        "rebu_risk_e41_100/weights/rebu_yolo100_ema.pt",
        "d85a864b7f84d0f408a68697cf9f4cbbfb73a9b821bb8ca392ba49c2956c447a",
    ),
    "prediction_exporter": (
        PROJECT / "scripts/export_yolo11m_predictions.py",
        "974efcdb71dbbe4fff980cae27ebb7cf826a630bb9d5c2afd654ed3b36f23166",
    ),
    "canonical_evaluator_source": (
        PROJECT / "scripts/run_yolo_architecture_gate_v4_fixed.py",
        "fd48e9575a9ef8606e2c20c882ec2dff6552e10d19016f43da8dbfd78547345b",
    ),
    "corruption_source": (
        PROJECT / "scripts/controlled_corruptions.py",
        "cb451b3ab1c5ba134bd5251298293f38f645d279d30614c7dbba7ecac550273a",
    ),
    "corruption_benchmark_source": (
        PROJECT / "scripts/run_controlled_corruption_benchmark.py",
        "caaf850b01b785365185c847af7942144d29d49061cc457f3896ce417949ad84",
    ),
    "match_builder_source": (
        PROJECT / "scripts/build_train_detection_matches.py",
        "3a3b5aa6efd5c956b429769fcdf45f0def85ec174c8164e5f3c217d62e6337c2",
    ),
    "risk_builder_source": (
        PROJECT / "scripts/build_train_risk_table.py",
        "e2f8f029da6cb7316f7be9ae49d73f55b11fade9cac925127f5530bba83aa44c",
    ),
}

TRAJECTORY_HASHES = {
    "uniform": {
        "epoch0.pt": "c7796cc7e0e0d9f03968e2fe33c40426723e089ec1c3b453dd5f9fbf2b11914e",
        "epoch5.pt": "84e4572cadabf4717363d9fabab2e715c8733fe71f581188a04cfe50cca3712f",
        "epoch10.pt": "948228b888eac4c0e92f348e28ef054cf2e7b6544076d299ea7ca44ebf714a30",
        "epoch15.pt": "84aa86771d185c2c72fa277c609c1222d7a417a8197523c4a5ced2a6a91ebb1b",
        "epoch20.pt": "0d89feceeff48acae4ef2d01f9ed9b75598ba5b57f7252e41dcfa673c2000e71",
        "epoch25.pt": "45de3cb7d78221347941b03b2af5b05ed758a580732c369ee30578321989c416",
        "epoch30.pt": "aff8681ca8bf72ceb43ff2d5d5dc07dc62aef93e6d6a2ec0f0b6e180db33a040",
        "epoch35.pt": "421ec7084e4021189e7217c6b30eb093308ec04f7b80047e366db3e8f30479da",
        "epoch40.pt": "dcbcd19063039372dd4087ca29a311cc256fd3d1f0be026ac03967eabd62f0b0",
        "epoch45.pt": "69a9a07ad1dd8948376166f96d09bcd852b80aa5b748217343e264cf597d3116",
        "epoch50.pt": "488e9f682857c5608f96305c5bea7fa900aa9bbd28357ff17564e01fae1d4f95",
        "epoch55.pt": "55a588a24c803448bb1b82bac50f7a542c3a8965aaf454133808cedf8d94ffce",
    },
    "rebu_risk": {
        "epoch0.pt": "e2e2f112a2afd192d5025a4260837ec911aad510106d3ed3d31d4e54198bcd15",
        "epoch5.pt": "e3320e3eac16911177f485416390b1a8cee2c11ce7ccddfa529a65e76f4d8716",
        "epoch10.pt": "bae318237b4e1849da492b84c38b503c752189504e07ea07ba99e7d701402421",
        "epoch15.pt": "879b179c219914a83775cacb7074f1d7bd9836971fae24592a6a16d6e87fe102",
        "epoch20.pt": "45d80fb25404c8665d3e5b2ccdb726391333df248b067c24b890a33f13044883",
        "epoch25.pt": "50db193e0dbf874231234ff86590e343381ccd86f0c5fe4758dfff821018c532",
        "epoch30.pt": "6e272abf3d823a07de7ac4890616430af2e794ca43096cb71cc063a55930a0a7",
        "epoch35.pt": "298957c95a33052b8b1cd0bde030268d49384385eb30f61db467c01b9aba9d30",
        "epoch40.pt": "c4c546d51c538f1a84062f8540e51f2e27f659f15261cd986dce16ab3afcccf2",
        "epoch45.pt": "5f5d0f946fbe67fd4b671b9364d0e86fad20807cd25a51bc1589c89cd1d171d7",
        "epoch50.pt": "6f80c4263ce6f22ad83e868997dae963eea8837252618436513a3cdd6f3907c1",
        "epoch55.pt": "4247e393ef0854275442296c261a094a45cac5e016ca7d2d8856a49e116afcfe",
    },
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


def write_json(path: Path, data: dict) -> None:
    payload = (
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )
    path.write_text(payload, encoding="utf-8")


def file_record(
    label: str,
    path: Path,
    expected_sha256: str,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)

    actual = sha256(path)

    if actual != expected_sha256:
        raise AssertionError(
            f"{label} SHA256 mismatch\n"
            f"path: {path}\n"
            f"expected: {expected_sha256}\n"
            f"actual:   {actual}"
        )

    return {
        "label": label,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual,
    }


def validate_coco(
    path: Path,
    expected_images: int,
    expected_annotations: int,
) -> dict:
    data = json.loads(path.read_text())

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    if len(images) != expected_images:
        raise AssertionError(
            (path, len(images), expected_images)
        )

    if len(annotations) != expected_annotations:
        raise AssertionError(
            (path, len(annotations), expected_annotations)
        )

    expected_categories = {
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

    actual_categories = {
        int(item["id"]): item["name"]
        for item in categories
    }

    if actual_categories != expected_categories:
        raise AssertionError(
            (path, actual_categories)
        )

    return {
        "images": len(images),
        "annotations": len(annotations),
        "categories": actual_categories,
    }


def verify_inputs() -> tuple[list[dict], list[dict], dict]:
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite frozen output: {OUTPUT_DIR}"
        )

    records = []

    print("=== VERIFYING FIXED INPUTS ===")

    for label, (path, expected_hash) in EXPECTED.items():
        record = file_record(
            label,
            path,
            expected_hash,
        )
        records.append(record)

        print(
            f"PASS {label:32s} "
            f"{record['sha256']}"
        )

    print("\n=== VERIFYING COCO DATASETS ===")

    val_summary = validate_coco(
        EXPECTED["official_val_gt"][0],
        expected_images=10_000,
        expected_annotations=185_523,
    )

    train_summary = validate_coco(
        EXPECTED["train_gt"][0],
        expected_images=70_000,
        expected_annotations=1_286_852,
    )

    print("official_val:", val_summary)
    print("train:", train_summary)

    print("\n=== VERIFYING 27 FORMAL MODEL NODES ===")

    model_nodes = [
        {
            "key": "common40",
            "arm": "common",
            "global_epoch": 40,
            "path": str(EXPECTED["common40"][0]),
            "sha256": EXPECTED["common40"][1],
            "checkpoint_role": "released_reference_model",
        }
    ]

    for arm, expected_files in TRAJECTORY_HASHES.items():
        root = (
            RESULTS
            / "formal_training/training_v2"
            / f"{arm}_e41_100"
            / "weights"
        )

        actual_names = {
            path.name
            for path in root.glob("epoch*.pt")
        }

        if actual_names != set(expected_files):
            raise AssertionError(
                (
                    arm,
                    sorted(actual_names),
                    sorted(expected_files),
                )
            )

        for filename, expected_hash in expected_files.items():
            path = root / filename
            actual_hash = sha256(path)

            if actual_hash != expected_hash:
                raise AssertionError(
                    (
                        arm,
                        filename,
                        expected_hash,
                        actual_hash,
                    )
                )

            local_epoch = int(
                Path(filename).stem.replace("epoch", "")
            )
            global_epoch = 41 + local_epoch

            model_nodes.append(
                {
                    "key": f"{arm}_e{global_epoch}",
                    "arm": arm,
                    "global_epoch": global_epoch,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": actual_hash,
                    "checkpoint_role": "saved_trajectory_node",
                }
            )

            print(
                f"PASS {arm:10s} "
                f"global_epoch={global_epoch:3d} "
                f"{actual_hash}"
            )

    for key, arm in (
        ("uniform_e100", "uniform"),
        ("risk_e100", "rebu_risk"),
    ):
        path, expected_hash = EXPECTED[key]
        model_nodes.append(
            {
                "key": key,
                "arm": arm,
                "global_epoch": 100,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": expected_hash,
                "checkpoint_role": "fixed_terminal_model",
            }
        )

    if len(model_nodes) != 27:
        raise AssertionError(
            f"Expected 27 formal nodes, found "
            f"{len(model_nodes)}"
        )

    dataset_summary = {
        "official_validation": val_summary,
        "training": train_summary,
    }

    return records, model_nodes, dataset_summary


def build_protocol(
    inventory_sha256: str,
    freeze_script_sha256: str,
    created_utc: str,
) -> dict:
    return {
        "version": "final_testing_protocol_v1",
        "status": "frozen_before_final_testing",
        "created_utc": created_utc,
        "study": {
            "title": (
                "Diagnose Before Repair: Matching Architectural "
                "and Data Interventions to Object-Level Failures "
                "in Autonomous-Driving Detection"
            ),
            "central_claim": (
                "Correct identification of object-level failures "
                "does not automatically imply generalized failure "
                "repair. Intervention effectiveness must be tested "
                "for failure alignment, condition preservation, "
                "clean generalization, corruption robustness, and "
                "long-horizon persistence."
            ),
        },
        "scope": {
            "new_training_allowed": False,
            "checkpoint_modification_allowed": False,
            "prior_artifact_modification_allowed": False,
            "official_val_is_untouched_test_set": False,
            "protocol_is_training_preregistration": False,
            "protocol_role": (
                "Freeze all remaining tests, metrics, comparisons, "
                "statistics, and interpretation rules before final "
                "analysis."
            ),
            "released_model": "common40",
            "released_model_name": (
                "P2-960 Reference Detector (Common40)"
            ),
            "d_fine_role": (
                "Independent diagnostic reference only; "
                "not part of final inference."
            ),
            "yolo11seg_in_core_study": False,
        },
        "provenance": {
            "input_inventory_sha256": inventory_sha256,
            "freeze_script_sha256": freeze_script_sha256,
        },
        "canonical_evaluation": {
            "dataset": "BDD100K official validation",
            "images": 10_000,
            "annotations": 185_523,
            "category_ids": list(range(1, 11)),
            "task": "bbox",
            "image_size": 960,
            "prediction_confidence_floor": 0.001,
            "native_nms_iou": 0.7,
            "native_max_detections": 300,
            "coco_max_detections": 100,
            "iou_range": "0.50:0.95",
            "metrics": [
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
            ],
            "ultralytics_metrics_in_formal_tables": False,
            "all_formal_tables_use_canonical_coco": True,
        },
        "clean_model_groups": {
            "baseline": [
                "YOLO11m",
                "D-FINE-M",
            ],
            "architecture": [
                "standard_e20",
                "p2_e20",
            ],
            "released_reference": [
                "common40",
            ],
            "method_gate": [
                "gate_uniform",
                "gate_scalar",
                "gate_typed",
                "gate_typed_v2",
            ],
            "long_horizon_representatives": [
                "uniform_e56",
                "rebu_risk_e56",
                "uniform_e100",
                "risk_e100",
            ],
        },
        "failure_subsets": {
            "definition_models": [
                "pre_intervention_YOLO11m",
                "pre_intervention_D-FINE-M",
            ],
            "definition_split": "official_validation",
            "post_intervention_models_may_define_subsets": False,
            "operating_thresholds": {
                "YOLO11m": 0.22527144849300385,
                "D-FINE-M": 0.4963708519935608,
            },
            "correct_detection_iou": 0.50,
            "localization_candidate_iou_min": 0.10,
            "coco_small_area_max_exclusive": 1024,
            "subsets": [
                "small_jointly_missed",
                "dfine_supported_yolo_failure",
                "yolo_low_confidence",
                "yolo_localization_failure",
                "yolo_classification_failure",
                "small_high_risk",
            ],
            "required_subset_artifacts": [
                "image_ids",
                "annotation_ids",
                "object_count",
                "image_count",
                "class_distribution",
                "scale_distribution",
                "definition_metadata",
                "sha256_manifest",
            ],
            "interpretation": (
                "Conditional recovery on frozen baseline-failure "
                "sets, not an unbiased estimate for every unseen "
                "failure."
            ),
        },
        "threshold_sensitivity": {
            "enabled": True,
            "central_threshold": "maximum_macro_f1",
            "plateau_rule": (
                "macro_f1 >= 0.95 * maximum_macro_f1"
            ),
            "evaluated_points": [
                "lowest_threshold_on_plateau",
                "central_threshold",
                "highest_threshold_on_plateau",
            ],
            "reported_outputs": [
                "subset_counts",
                "subset_jaccard_against_central",
                "class_composition",
                "scale_composition",
                "primary_effect_direction_stability",
            ],
        },
        "primary_evidence": {
            "architecture_intervention": {
                "comparison": "p2_e20_minus_standard_e20",
                "primary_metrics": [
                    "AP_small",
                    "AR_small",
                ],
                "guardrails": [
                    "AP",
                    "AP75",
                    "AP_medium",
                    "AP_large",
                ],
            },
            "targeted_recovery": {
                "comparisons": [
                    "gate_scalar_minus_gate_uniform",
                    "gate_typed_minus_gate_uniform",
                    "gate_typed_minus_gate_scalar",
                    "gate_typed_v2_minus_gate_uniform",
                ],
                "primary_metrics": [
                    "failure_macro_AR100",
                    "small_jointly_missed_recall50",
                ],
            },
            "clean_generalization": {
                "metrics": [
                    "AP",
                    "AP_small",
                    "AP75",
                ],
            },
            "robust_generalization": {
                "metrics": [
                    "mean_corruption_AP",
                    "mean_corruption_AP_small",
                ],
            },
            "long_horizon_persistence": {
                "metrics": [
                    "matched_epoch_delta_AP",
                    "matched_epoch_delta_AP_small",
                    "matched_epoch_delta_AP75",
                    "trajectory_mean",
                    "trajectory_AUC",
                    "positive_delta_checkpoint_fraction",
                ],
            },
        },
        "paired_bootstrap": {
            "enabled": True,
            "resampling_unit": "official_val_image",
            "paired_images_for_both_models": True,
            "replicates": 2000,
            "seed": 20260816,
            "confidence_level": 0.95,
            "interval": "percentile",
            "method": (
                "Image-cluster resampling with complete COCO "
                "precision-recall re-accumulation."
            ),
            "naive_mean_per_image_AP_forbidden": True,
            "primary_comparisons": [
                "p2_e20_minus_standard_e20",
                "gate_scalar_minus_gate_uniform",
                "gate_typed_minus_gate_uniform",
                "gate_typed_minus_gate_scalar",
                "rebu_risk_e56_minus_uniform_e56",
                "risk_e100_minus_uniform_e100",
                "common40_minus_baseline_YOLO11m",
            ],
            "multiple_comparison_policy": {
                "primary_metrics": (
                    "Report pre-specified paired 95% intervals."
                ),
                "secondary_families": (
                    "Apply Benjamini-Hochberg FDR at q=0.05 "
                    "when inferential p-values are reported."
                ),
            },
            "scope_warning": (
                "Image bootstrap estimates validation-image "
                "sampling uncertainty and does not replace "
                "training-seed replication."
            ),
        },
        "trajectory_analysis": {
            "formal_model_nodes": 27,
            "common_epoch": 40,
            "matched_branch_epochs": [
                41, 46, 51, 56, 61, 66, 71,
                76, 81, 86, 91, 96, 100,
            ],
            "fixed_terminal_epoch": 100,
            "report_fixed_e100": True,
            "report_all_matched_epochs": True,
            "report_grid_mean": True,
            "report_trajectory_auc": True,
            "observed_peak_is_descriptive_only": True,
            "cross_epoch_method_comparison_forbidden": True,
            "common_post_convergence_effect": (
                "Uniform(epoch) - Common40"
            ),
            "risk_marginal_effect": (
                "Risk(epoch) - Uniform(epoch)"
            ),
            "training_dynamics_outputs": [
                "canonical_metric_trajectory.csv",
                "matched_epoch_deltas.csv",
                "training_loss_and_lr.csv",
                "common_effect_summary.json",
                "risk_marginal_effect_summary.json",
                "trajectory_figure",
            ],
            "overfitting_term_rule": (
                "Use 'overfitting' only when declining canonical "
                "validation performance is accompanied by continued "
                "improvement in training objective. Otherwise use "
                "'post-convergence degradation'."
            ),
        },
        "exposure_audit": {
            "required": True,
            "metrics": [
                "training_views",
                "original_views",
                "object_centric_views",
                "unique_source_images",
                "source_image_exposure_counts",
                "total_supervised_objects",
                "target_objects",
                "non_target_objects",
                "objects_per_view",
                "class_exposure",
                "scale_exposure",
                "input_pixel_budget",
                "supervision_density",
            ],
            "equal_budget_wording": (
                "Equal in training views and optimization steps; "
                "not necessarily equal in objects, pixels, context, "
                "or supervision content."
            ),
        },
        "scale_shift_audit": {
            "required": True,
            "coordinate_spaces": [
                "original_image",
                "crop_before_resize",
                "network_input_after_960_letterbox",
            ],
            "coco_area_thresholds": {
                "small": "[0, 1024)",
                "medium": "[1024, 9216)",
                "large": "[9216, infinity)",
            },
            "metrics": [
                "bbox_short_side",
                "bbox_area",
                "normalized_bbox_area",
                "short_side_amplification",
                "area_amplification",
                "scale_transition_matrix",
            ],
            "primary_question": (
                "How many original small failure objects cease "
                "to be small after object-centric transformation?"
            ),
        },
        "context_shift_audit": {
            "required": True,
            "neighbor_definition": (
                "Every non-target GT object in the same source image."
            ),
            "neighbor_retained_rule": (
                "At least 50% of the original neighbor bbox area "
                "remains inside the crop before resize."
            ),
            "metrics": [
                "crop_area_over_source_area",
                "target_area_over_crop_area",
                "source_neighbor_count",
                "retained_neighbor_count",
                "neighbor_retention_fraction",
                "fully_removed_objects",
                "boundary_clipped_objects",
                "objects_per_view_change",
                "context_margin_around_target",
            ],
            "causal_language_allowed": False,
            "interpretation": (
                "Quantifies association between crop-induced "
                "condition shift and non-transfer; does not prove "
                "strict causality."
            ),
        },
        "corruption_evaluation": {
            "models": [
                "standard_e20",
                "p2_e20",
                "common40",
                "uniform_e56",
                "rebu_risk_e56",
            ],
            "corruptions": [
                "blur",
                "noise",
                "fog",
                "rain",
                "low_light",
            ],
            "severities": [1, 2, 3],
            "total_model_condition_runs": 75,
            "canonical_metrics": True,
            "on_the_fly_or_ephemeral_images": True,
            "existing_project_assets_may_be_deleted": False,
            "temporary_predictions_policy": (
                "Retain only until metric, count, and SHA256 "
                "validation succeeds; then remove only newly "
                "created temporary prediction files."
            ),
        },
        "efficiency_evaluation": {
            "architecture_pair": [
                "standard_e20",
                "p2_e20",
            ],
            "image_size": 960,
            "batch": 1,
            "precision": "FP16",
            "hardware": "NVIDIA GeForce RTX 4090",
            "warmup_iterations": 50,
            "measurement_iterations": 200,
            "round_orders": [
                ["standard", "p2"],
                ["p2", "standard"],
            ],
            "cuda_synchronize_each_measurement": True,
            "metrics": [
                "parameters",
                "GFLOPs",
                "model_file_size",
                "median_forward_latency_ms",
                "p90_forward_latency_ms",
                "mean_forward_latency_ms",
                "peak_inference_memory_MiB",
                "peak_training_memory_MiB",
            ],
            "d_fine_receives_p2_intervention": False,
            "deployment_scope": (
                "RTX4090 PyTorch evidence; not an edge-device "
                "end-to-end latency claim."
            ),
        },
        "per_class_reliability": {
            "report_all_ten_classes": True,
            "required_fields": [
                "num_gt",
                "num_images",
                "AP",
                "AR100",
                "paired_interval_when_applicable",
                "scale_composition",
                "evidence_reliability",
            ],
            "rare_train_class": {
                "category_id": 6,
                "class_name": "train",
                "strong_standalone_claim_allowed": False,
            },
        },
        "qualitative_cases": {
            "selection_must_be_deterministic": True,
            "manual_cherry_picking_allowed": False,
            "seed": 20260816,
            "case_families": [
                "p2_repairs_small_failure",
                "p2_remaining_failure",
                "risk_short_term_recovery",
                "risk_non_transfer_in_original_scene",
                "crop_scale_condition_shift",
                "crop_context_condition_shift",
                "blur_or_noise_success",
                "blur_or_noise_failure",
            ],
            "equal_cases_per_family": True,
            "required_overlays": [
                "ground_truth",
                "baseline_yolo",
                "p2",
                "uniform",
                "risk",
                "failure_type",
                "scale_and_context_statistics",
            ],
        },
        "interpretation_rules": {
            "risk_method_universally_ineffective_claim_allowed": False,
            "preferred_risk_wording": (
                "Under the evaluated fixed-ratio object-centric "
                "training protocol, risk-guided exposure produced "
                "targeted recovery but did not establish stable "
                "generalized repair."
            ),
            "confidence_interval_crosses_zero": (
                "Report no stable detected difference; do not "
                "claim equivalence or proof of no effect."
            ),
            "p2_claim_requires": (
                "Positive small-object evidence together with "
                "reported clean guardrails and efficiency cost."
            ),
            "generalized_repair_requires": [
                "targeted_recovery",
                "clean_generalization",
                "robust_generalization",
                "long_horizon_persistence",
            ],
        },
        "reporting_outputs": [
            "canonical_clean_table",
            "failure_subset_table",
            "threshold_sensitivity_table",
            "paired_bootstrap_table",
            "trajectory_table_and_figure",
            "exposure_budget_table",
            "scale_transition_matrix",
            "context_retention_table",
            "corruption_table_and_figure",
            "efficiency_table",
            "per_class_table",
            "deterministic_qualitative_figure",
            "final_artifact_manifest",
            "paper_ready_markdown_results",
        ],
        "analysis_order": [
            "canonical_clean",
            "official_val_failure_subsets",
            "threshold_sensitivity",
            "training_dynamics_attribution",
            "paired_bootstrap",
            "exposure_audit",
            "scale_context_audit",
            "corruption_evaluation",
            "efficiency_evaluation",
            "qualitative_cases",
            "paper_tables_and_markdown",
            "final_manifest",
        ],
    }


def freeze(
    records: list[dict],
    model_nodes: list[dict],
    dataset_summary: dict,
) -> None:
    OUTPUT_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging = OUTPUT_DIR.parent / (
        OUTPUT_DIR.name
        + f".incomplete-{os.getpid()}"
    )

    if staging.exists():
        raise FileExistsError(staging)

    staging.mkdir()

    try:
        created_utc = datetime.now(
            timezone.utc
        ).isoformat()

        script_path = Path(__file__).resolve()
        script_hash = sha256(script_path)

        inventory = {
            "version": "final_testing_input_inventory_v1",
            "status": "frozen",
            "created_utc": created_utc,
            "fixed_inputs": records,
            "formal_model_nodes": model_nodes,
            "formal_model_node_count": len(model_nodes),
            "dataset_summary": dataset_summary,
            "freeze_script": {
                "path": str(script_path),
                "bytes": script_path.stat().st_size,
                "sha256": script_hash,
            },
        }

        inventory_path = (
            staging / "input_inventory.json"
        )
        write_json(
            inventory_path,
            inventory,
        )

        inventory_hash = sha256(
            inventory_path
        )

        protocol = build_protocol(
            inventory_sha256=inventory_hash,
            freeze_script_sha256=script_hash,
            created_utc=created_utc,
        )

        protocol_path = (
            staging
            / "final_testing_protocol_v1.json"
        )
        write_json(
            protocol_path,
            protocol,
        )

        manifest = {
            "version": (
                "final_testing_protocol_artifact_manifest_v1"
            ),
            "status": "frozen",
            "created_utc": created_utc,
            "files": {
                "input_inventory.json": {
                    "bytes": inventory_path.stat().st_size,
                    "sha256": sha256(inventory_path),
                },
                "final_testing_protocol_v1.json": {
                    "bytes": protocol_path.stat().st_size,
                    "sha256": sha256(protocol_path),
                },
            },
        }

        manifest_path = (
            staging / "artifact_manifest.json"
        )
        write_json(
            manifest_path,
            manifest,
        )

        os.replace(
            staging,
            OUTPUT_DIR,
        )

    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    final_protocol = (
        OUTPUT_DIR
        / "final_testing_protocol_v1.json"
    )
    final_inventory = (
        OUTPUT_DIR
        / "input_inventory.json"
    )
    final_manifest = (
        OUTPUT_DIR
        / "artifact_manifest.json"
    )

    print("\n" + "=" * 100)
    print("FINAL TESTING PROTOCOL FROZEN")
    print("=" * 100)
    print("protocol:", final_protocol)
    print("protocol_sha256:", sha256(final_protocol))
    print("inventory:", final_inventory)
    print("inventory_sha256:", sha256(final_inventory))
    print("manifest:", final_manifest)
    print("manifest_sha256:", sha256(final_manifest))
    print("formal_model_nodes:", len(model_nodes))
    print("new_training_allowed: False")
    print(
        "PASS: final tests, statistics, comparisons "
        "and interpretation rules are immutable"
    )
    print("NOTHING TRAINED OR EVALUATED")
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the no-new-training final testing "
            "and evidence consolidation protocol."
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
        "--freeze",
        action="store_true",
    )

    args = parser.parse_args()

    records, model_nodes, dataset_summary = (
        verify_inputs()
    )

    if args.preflight_only:
        print("\n" + "=" * 100)
        print("FINAL TESTING PROTOCOL PREFLIGHT COMPLETE")
        print("=" * 100)
        print("fixed_inputs:", len(records))
        print("formal_model_nodes:", len(model_nodes))
        print("output_exists:", OUTPUT_DIR.exists())
        print("PASS: all frozen inputs and hashes are valid")
        print("NOTHING CREATED, MODIFIED, EVALUATED OR TRAINED")
        print("=" * 100)
        return

    freeze(
        records,
        model_nodes,
        dataset_summary,
    )


if __name__ == "__main__":
    main()
