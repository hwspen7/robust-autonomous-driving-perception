# Source Code

This directory contains active implementation code only.

- `analysis`: failure taxonomy and intervention-semantics analysis.
- `data-preparation`: matching, risk labeling, protocol freezing,
  validation, conversion, and view materialization.
- `evaluation`: canonical, failure, corruption, efficiency,
  statistical, qualitative, and prediction-export evaluation.
- `training`: architecture gates, method gates, formal training,
  and training controls.
- `utilities`: diagnostics, shared components, shared tools, and
  portable runtime support.
- `third-party/D-FINE`: required upstream D-FINE implementation.

Dependencies, reports, datasets, model weights, predictions, and
experiment results do not belong in this directory.

## Naming and frozen provenance

Active project-owned executors use descriptive, version-free file names.
Historical executor names and original SHA256 values remain recorded in
`project-management/inventories/source-code-classification.tsv`. The Linux
runtime compatibility layer recreates those historical names as external
symbolic links without duplicating or modifying the active source files.
