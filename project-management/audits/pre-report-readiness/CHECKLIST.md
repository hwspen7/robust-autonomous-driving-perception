# Project Readiness Checklist

## GitHub repository boundary

- [x] one active source-code tree
- [x] root requirements profiles are pinned and public
- [x] Docker is not required by the project reproduction guide
- [x] manuscripts, local archive, obsolete assets, and large artifacts are excluded
- [x] no public file exceeds 100 MiB
- [x] no public symlink exists
- [x] no private Mac path or server login appears in public text
- [x] GitHub Actions static audit is present

## Scientific integrity

- [x] 68 frozen experiment scripts are hash-valid
- [x] all project-owned Python files compile in memory
- [x] all 114 active artifact files are SHA256-valid
- [x] all 581 entries from 45 scientific manifests resolve and are SHA256-valid
- [x] 398 direct entries, 21 active-shared dependencies, and 162 archive dependencies are preserved
- [x] exactly 100 YOLO11-Seg visualizations are retained
- [x] no training or inference was performed during this audit

## Reproducibility boundary

- [x] repository-level source and evidence reproduction is prepared
- [x] BDD100K is documented as an external licensed dataset
- [x] excluded large artifacts are documented by path, size, and SHA256
- [ ] clean external Linux dependency installation has been executed
- [ ] dataset reconstruction has been tested from a fresh checkout
- [ ] end-to-end GPU reproduction has been executed from a fresh checkout

The three unchecked runtime tests are non-blocking for the initial GitHub publication, but they must remain disclosed. The repository must not claim a completed independent end-to-end reproduction until those tests are actually run.

## Decision

**READY_FOR_GITHUB_WITH_DECLARED_EXTERNAL_ASSETS**

Generated: `2026-08-18T10:25:10.125059+00:00`
