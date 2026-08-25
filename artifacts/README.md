# Artifact Manifest

This directory stores metadata only. Large artifacts remain outside
Git.

- `manifest.json` records relative paths, roles, sizes, and SHA256.
- `checksums.sha256` supports direct integrity verification.

Paths are relative to the repository root.

- `archive-dependencies.json` resolves non-direct frozen manifest entries through exact SHA256 mappings.
