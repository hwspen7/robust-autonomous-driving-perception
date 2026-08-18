# Diagnose Before Repair

## Matching Architectural and Data Interventions to Object-Level Failures in Autonomous-Driving Detection

This paper-facing result record is generated only from the frozen Stage 1-10 evidence. It performs no new training, inference, threshold tuning, model selection, or qualitative replacement.

## Experimental scope

- BDD100K official validation: 10,000 images and 185,523 objects.
- YOLO11m at 960 pixels is the final detector family; D-FINE-M is diagnosis-only.
- Statistical uncertainty uses 2,000 paired official-validation image bootstrap replicates.
- Training evidence uses one fixed seed.
- Official validation participated in architecture adjudication and is not an untouched test set.

## Architecture intervention

| Model | AP | AP50 | AP75 | AP small | AR small |
| --- | --- | --- | --- | --- | --- |
| Standard-960-E20 | 0.3264 | 0.5675 | 0.3155 | 0.1449 | 0.2434 |
| P2-960-E20 | 0.3355 | 0.5783 | 0.3261 | 0.1568 | 0.3020 |
| P2 minus Standard | 0.0090 | 0.0108 | 0.0106 | 0.0118 | 0.0586 |

P2 minus Standard changes canonical AP by +0.0090, AP small by +0.0118, and AR small by +0.0586.

## Frozen failure subsets

| Subset | Objects | Images | Mean risk |
| --- | --- | --- | --- |
| small_jointly_missed | 34,698 | 8,184 | 0.8663 |
| dfine_supported_yolo_failure | 9,000 | 5,416 | 0.7589 |
| yolo_low_confidence | 26,804 | 8,262 | 0.8260 |
| yolo_localization_failure | 11,255 | 5,294 | 0.8260 |
| yolo_classification_failure | 1,797 | 1,441 | 0.8969 |
| small_high_risk | 37,331 | 8,497 | 0.8727 |

Teacher operating points were independently calibrated. Threshold sensitivity tests directional stability but does not turn diagnostic association into causal labels.

## Data-intervention results

| Model | AP | AP75 | AP small | AR100 |
| --- | --- | --- | --- | --- |
| Uniform-Gate-E10 | 0.3407 | 0.3304 | 0.1602 | 0.4818 |
| Scalar-Risk-Gate-E10 | 0.3412 | 0.3317 | 0.1624 | 0.4872 |
| Typed-CAFR-Gate-E10 | 0.3408 | 0.3301 | 0.1617 | 0.4872 |
| Typed-CAFR-V2-Gate-E10 | 0.3430 | 0.3330 | 0.1621 | 0.4880 |
| P2-960-Common40 | 0.3505 | 0.3407 | 0.1690 | 0.4846 |
| Uniform-E56 | 0.3454 | 0.3348 | 0.1692 | 0.4875 |
| Risk-E56 | 0.3461 | 0.3343 | 0.1670 | 0.4879 |
| Uniform-E100 | 0.3340 | 0.3191 | 0.1626 | 0.4749 |
| Risk-E100 | 0.3308 | 0.3154 | 0.1606 | 0.4689 |

Risk-E56 minus Uniform-E56 is +0.0007 AP and -0.0022 AP small. Risk-E100 minus Uniform-E100 is -0.0032 AP and -0.0020 AP small.

Across saved E41-E100 checkpoints, the positive risk-minus-uniform AP-small checkpoint fraction is 0.0%; normalized AP-small delta AUC is -0.0018. A persistent small-object advantage was not established.

## Training dynamics

At E100, the AP shared-schedule effect is -0.0165 and the risk-specific residual is -0.0032. These are descriptive decompositions, not randomized causal effects.

## Exposure, scale, and context

Both formal arms use 70,000 views. The risk arm contains 19,500 object-centric views and 1,195,604 supervised objects, versus 1,286,852 for uniform training. The arms are equal in views and optimization steps, not supervision content.

Among 19,500 crop targets, 67.8% of originally small targets cease to be small in crop coordinates. Mean short-side amplification is 2.636 times; weighted neighbor retention is 0.579. 159,728 neighbors are fully removed and 38,473 are boundary-clipped.

These measurements establish condition shift and an empirical association with non-transfer, not strict causal identification.

## Controlled corruptions

| Model | Mean corruption AP | Mean corruption AP small |
| --- | --- | --- |
| standard_e20 | 0.3017 | 0.1270 |
| p2_e20 | 0.3099 | 0.1382 |
| common40 | 0.3203 | 0.1469 |
| uniform_e56 | 0.3137 | 0.1483 |
| rebu_risk_e56 | 0.3135 | 0.1462 |

P2 minus Standard changes mean corruption AP by +0.0083 and mean corruption AP small by +0.0112. Risk-E56 minus Uniform-E56 changes the same metrics by -0.0003 and -0.0021.

## Efficiency

| Model | Parameters | GFLOPs | Median ms | P90 ms | Peak inference MiB | Peak training MiB |
| --- | --- | --- | --- | --- | --- | --- |
| standard_e20 | 20,060,718 | 154.12 | 14.443 | 15.412 | 289.1 | 8692.9 |
| p2_e20 | 20,559,224 | 211.98 | 16.984 | 18.057 | 388.7 | 15536.9 |

P2 increases RTX4090 PyTorch FP16 median forward latency by 17.6% and GFLOPs by 37.5%. This is not an edge-device end-to-end latency claim.

## Paired-bootstrap primary evidence

| Comparison | Metric | Delta | 95% CI | Detected direction |
| --- | --- | --- | --- | --- |
| p2_e20_minus_standard_e20 | AP_small | 0.0118 | [0.0093, 0.0160] | positive |
| p2_e20_minus_standard_e20 | AR_small | 0.0586 | [0.0537, 0.0698] | positive |
| gate_typed_minus_gate_uniform | failure_macro_AR100 | 0.0117 | [0.0066, 0.0177] | positive |
| gate_typed_minus_gate_uniform | small_jointly_missed_recall50 | 0.0317 | [0.0286, 0.0348] | positive |
| gate_typed_minus_gate_scalar | failure_macro_AR100 | 0.0043 | [-0.0012, 0.0086] | not detected |
| gate_typed_minus_gate_scalar | small_jointly_missed_recall50 | -0.0003 | [-0.0024, 0.0018] | not detected |
| rebu_risk_e56_minus_uniform_e56 | AP | 0.0007 | [-0.0071, 0.0027] | not detected |
| rebu_risk_e56_minus_uniform_e56 | AP75 | -0.0005 | [-0.0092, 0.0033] | not detected |
| rebu_risk_e56_minus_uniform_e56 | AP_small | -0.0022 | [-0.0048, -0.0005] | negative |
| risk_e100_minus_uniform_e100 | AP | -0.0032 | [-0.0061, -0.0010] | negative |
| risk_e100_minus_uniform_e100 | AP75 | -0.0038 | [-0.0089, 0.0007] | not detected |
| risk_e100_minus_uniform_e100 | AP_small | -0.0020 | [-0.0053, 0.0008] | not detected |

Intervals crossing zero are no stable detected difference, not proof of equivalence. Image bootstrap does not replace seed replication.

## Deterministic qualitative evidence

The frozen protocol produced 32 cases across 8 families, 4 each, using deterministic SHA ranking without manual replacement.

## Main conclusion

Correctly diagnosing an object-level failure does not guarantee that increasing exposure to a transformed copy will repair the original failure condition. In this study, the scene-preserving P2 intervention improves small-object and representative-corruption metrics with measurable efficiency cost. Object-centric risk cropping changes scale, context, and supervision composition and does not establish a persistent small-object advantage. Failure-guided training should therefore verify that its intervention preserves the condition that produced the failure.

## Required limitations

- One fixed training seed.
- Official validation participated in architecture adjudication.
- Formal E41-E100 optimization used a restarted schedule.
- Five-epoch checkpoint spacing cannot recover unsaved optima.
- Risk crops do not preserve identical supervision content.
- Mechanism evidence is associative, not strict causal identification.
- Controlled corruptions are synthetic.
- Efficiency evidence is RTX4090 PyTorch FP16 specific.

## Reproducibility

- Frozen final-testing protocol SHA256: 693c39375cc1eefc4f3ff9f957a8355fb947552a19dfef228cae6cfc21e3dcfe
- Paired bootstrap replicates: 2000
- Qualitative manifest SHA256: a1b13f5e699481c946fb03de62f657ee0b56c3fe3efd83122676fcb7759cf2d3
- This bundle performs no new training or inference.
