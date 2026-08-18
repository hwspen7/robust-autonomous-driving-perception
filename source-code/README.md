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
Historical executor names and source provenance remain recorded in
`project-management/inventories/source-code-classification.tsv`. In that
inventory, `sha256` records the original server/frozen source bytes, while
`release_sha256` records the corresponding public release bytes. The Linux
runtime compatibility layer recreates historical executor names as external
symbolic links to the verified public release files without duplicating them.
