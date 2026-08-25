# Diagnose Before Repair

This repository contains the public code, frozen evidence, and reproducibility metadata for a study of object-level failure diagnosis and intervention matching in autonomous-driving object detection.

The study asks a specific question. After a detector failure has been identified, does the proposed intervention preserve and repair the condition that caused the failure? Experiments on BDD100K compare a projected P2 architecture intervention with object-centric risk-guided data interventions. Evaluation covers canonical clean performance, predefined failure subsets, controlled corruptions, training trajectories, paired bootstrap intervals, exposure accounting, crop-induced scale and context shifts, and efficiency.

## Main findings

- Failure diagnosis is informative, but diagnosis alone does not guarantee a transferable repair.
- The projected P2 intervention improves small-object detection while preserving the original scene and context.
- Risk-guided object-centric crops can improve selected failure subsets over short horizons.
- Crop-based gains do not consistently transfer to clean-scene AP, corruption robustness, or long-horizon training.
- The crop intervention changes the learning condition. In the frozen audit, 67.79 percent of originally small targets cease to be small, the mean target short-side amplification is 2.64 times, and only 57.88 percent of neighboring objects are retained.

These results support a diagnose, intervene, and verify workflow in which targeted recovery, clean generalization, robustness, persistence, and intervention cost are evaluated separately.

## Repository structure

| Directory | Purpose |
|---|---|
| `data-and-baselines/` | Dataset metadata and YOLO11m, YOLO11-seg, and D-FINE-M baseline evidence |
| `failure-diagnosis/` | Calibrated detector matching and object-level failure taxonomy |
| `architecture-intervention/` | Projected P2 definition, protocols, and evaluation evidence |
| `risk-data-intervention/` | Risk labels, sampling, object-centric materialization, and method-gate evidence |
| `formal-training/` | Common, uniform, and risk-branch training records and checkpoint trajectories |
| `final-testing/` | Canonical clean, failure, bootstrap, corruption, efficiency, and qualitative tests |
| `research-evidence/` | Frozen paper tables, figures, and claim-to-evidence records |
| `source-code/` | Categorized project implementation and the required D-FINE source snapshot |
| `environment-and-reproducibility/` | Environment snapshots, dependency records, and runtime layout contract |
| `artifacts/` | External artifact inventory, checksums, and archive dependency declarations |

## Environments

The project uses two pinned Python environments because the YOLO and D-FINE dependency stacks are not installed together.

- `requirements.txt` contains the main YOLO, analysis, evaluation, and statistical stack.
- `requirements-dfine.txt` contains the D-FINE inference and teacher-evidence stack.

Formal GPU execution was performed on Linux with an NVIDIA RTX 4090. Large datasets, model checkpoints, and prediction payloads are declared external artifacts and are intentionally excluded from Git.

## Source preflight

From the repository root, verify the frozen source inventory and runtime contract with:

```bash
python3 source-code/utilities/runtime-layout/prepare_runtime_layout.py \
  --preflight-only \
  --project-root "$PWD"
```

The command checks the release source hashes and reports the expected compatibility paths without creating links or changing data.

## Reproduction workflow

1. Create separate Main and D-FINE environments from the pinned requirement files.
2. Obtain BDD100K through its official distribution and prepare the expected dataset layout.
3. Resolve external artifacts using `artifacts/manifest.json`, `artifacts/checksums.sha256`, and `artifacts/archive-dependencies.json`.
4. Run the source preflight before any GPU task.
5. Follow the frozen protocols in architecture intervention, risk-data intervention, formal training, and final testing.
6. Validate outputs against their artifact manifests before interpreting results.

The repository preserves completed experimental evidence. Reproducing every numerical result requires the external dataset, model checkpoints, and large prediction files identified by the artifact records.

## Integrity policy

Scientific CSV and TSV files are stored byte-for-byte because their SHA256 values are part of the frozen evidence chain. Git line-ending conversion is disabled for these files. Source release hashes and original experiment hashes are recorded separately where source portability changes were required.

## Citation and license

Citation metadata is provided in `CITATION.cff`. Repository-level licensing is described in `LICENSE`. Third-party components retain their original licenses and notices.
