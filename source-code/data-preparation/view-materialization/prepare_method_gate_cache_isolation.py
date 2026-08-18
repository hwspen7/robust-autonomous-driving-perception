from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from ultralytics.data.utils import img2label_paths

ROOT = Path(
    "/root/rivermind-data/autodrive/results/evaluation"
)
V1 = ROOT / "method_gate/materialization_v1"
V2 = (
    ROOT
    / "method_gate/materialization_v2_cache_isolated"
)
SHM = Path(
    "/dev/shm/rebu_yolo/method_gate_v1"
)
ANCHORS = SHM / "cache_anchors"

PROTOCOL = (
    ROOT
    / "method_gate/frozen_manifests/"
    "method_gate_protocol_v1.json"
)
V1_MANIFEST = V1 / "artifact_manifest.json"
ORIGINAL_DEV = (
    ROOT
    / "architecture_gate/prepared_inputs_v1/"
    "train_dev.txt"
)

EXPECTED = {
    PROTOCOL: (
        "6be7ae25550360e38195bb785a064bd19cb9c4a65b4625b30b5698861cc1b00a"
    ),
    V1_MANIFEST: (
        "74f26c5cc14a2f5bea497bea921966d5c12cbf81a43ac19c39fe71d226e18c06"
    ),
    ORIGINAL_DEV: (
        "d4e1c82556eb6adb30fa2d29f6d86e838922d70ad701f7403058b49184c0fcf2"
    ),
}

NAMES = {
    0: "pedestrian",
    1: "rider",
    2: "car",
    3: "truck",
    4: "bus",
    5: "train",
    6: "motorcycle",
    7: "bicycle",
    8: "traffic light",
    9: "traffic sign",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_paths(path: Path) -> list[str]:
    values = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if len(values) != len(set(values)):
        raise AssertionError(
            f"Duplicate paths: {path}"
        )
    return values


def predicted_cache(image_path: str) -> Path:
    label_path = img2label_paths(
        [image_path]
    )[0]
    return (
        Path(label_path)
        .parent
        .with_suffix(".cache")
    )


def yaml_text(
    train_list: Path,
    val_list: Path,
) -> str:
    lines = [
        "path: /root/rivermind-data/autodrive/"
        "datasets/datasets/bdd100k_final",
        f"train: {train_list}",
        f"val: {val_list}",
        "",
        "names:",
    ]

    for class_id, name in NAMES.items():
        lines.append(
            f"  {class_id}: {name}"
        )

    return "\n".join(lines) + "\n"


def make_anchor(
    root: Path,
    name: str,
    source_image: Path,
) -> tuple[Path, Path]:
    source_label = Path(
        img2label_paths(
            [str(source_image)]
        )[0]
    )

    if not source_image.is_file():
        raise FileNotFoundError(source_image)
    if not source_label.is_file():
        raise FileNotFoundError(source_label)

    image_dir = root / name / "images"
    label_dir = root / name / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    image_link = (
        image_dir
        / f"cache_anchor{source_image.suffix.lower()}"
    )
    label_link = (
        label_dir / "cache_anchor.txt"
    )

    os.symlink(
        source_image,
        image_link,
    )
    os.symlink(
        source_label,
        label_link,
    )

    if image_link.resolve() != source_image.resolve():
        raise AssertionError(image_link)
    if label_link.resolve() != source_label.resolve():
        raise AssertionError(label_link)

    return image_link, label_link


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
    )
    args = parser.parse_args()

    print("=== CACHE-ISOLATION PREFLIGHT ===")

    for path, expected in EXPECTED.items():
        if not path.is_file():
            raise FileNotFoundError(path)

        actual = sha256_file(path)
        print(path)
        print(" expected:", expected)
        print(" actual  :", actual)

        if actual != expected:
            raise AssertionError(path)

    v1_manifest = json.loads(
        V1_MANIFEST.read_text()
    )

    for name, record in (
        v1_manifest["files"].items()
    ):
        path = V1 / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != record["sha256"]:
            raise AssertionError(path)

    for arm in (
        "scalar_risk",
        "typed_cafr",
    ):
        image_count = len(list(
            (SHM / arm / "images").glob("*.jpg")
        ))
        label_count = len(list(
            (SHM / arm / "labels").glob("*.txt")
        ))

        if image_count != 19_500:
            raise AssertionError(
                (arm, image_count)
            )
        if label_count != 19_500:
            raise AssertionError(
                (arm, label_count)
            )

    original_lists = {
        arm: read_paths(
            V1 / f"{arm}_train.txt"
        )
        for arm in (
            "uniform",
            "scalar_risk",
            "typed_cafr",
        )
    }
    original_dev = read_paths(
        ORIGINAL_DEV
    )

    for arm, paths in original_lists.items():
        if len(paths) != 65_000:
            raise AssertionError(
                (arm, len(paths))
            )

    if len(original_dev) != 5_000:
        raise AssertionError(
            len(original_dev)
        )

    print()
    print("existing_cache_routes:")
    for arm, paths in original_lists.items():
        print(
            arm,
            predicted_cache(paths[0]),
        )

    if V2.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {V2}"
        )
    if ANCHORS.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {ANCHORS}"
        )

    if not args.execute:
        print()
        print("PASS: cache conflict reproduced")
        print("NOTHING CREATED")
        return

    token = str(os.getpid())
    v2_incomplete = Path(
        str(V2) + f".incomplete-{token}"
    )
    anchor_incomplete = Path(
        str(ANCHORS) + f".incomplete-{token}"
    )

    if v2_incomplete.exists():
        raise FileExistsError(v2_incomplete)
    if anchor_incomplete.exists():
        raise FileExistsError(
            anchor_incomplete
        )

    v2_incomplete.mkdir(parents=True)

    uniform_source = Path(
        original_lists["uniform"][0]
    )
    dev_source = Path(
        original_dev[0]
    )

    uniform_anchor_temp, _ = make_anchor(
        anchor_incomplete,
        "uniform",
        uniform_source,
    )
    dev_anchor_temp, _ = make_anchor(
        anchor_incomplete,
        "dev",
        dev_source,
    )

    uniform_anchor_final = (
        ANCHORS
        / "uniform"
        / "images"
        / uniform_anchor_temp.name
    )
    dev_anchor_final = (
        ANCHORS
        / "dev"
        / "images"
        / dev_anchor_temp.name
    )


    uniform_new = [
        str(uniform_anchor_final)
    ] + original_lists["uniform"][1:]

    dev_new = [
        str(dev_anchor_final)
    ] + original_dev[1:]


    reordered = {}

    for arm in (
        "scalar_risk",
        "typed_cafr",
    ):
        paths = original_lists[arm]
        crop_prefix = str(
            SHM / arm / "images"
        ) + "/"

        crop_paths = [
            path
            for path in paths
            if path.startswith(crop_prefix)
        ]
        original_paths = [
            path
            for path in paths
            if not path.startswith(crop_prefix)
        ]

        if len(crop_paths) != 19_500:
            raise AssertionError(
                (arm, len(crop_paths))
            )
        if len(original_paths) != 45_500:
            raise AssertionError(
                (arm, len(original_paths))
            )

        new_paths = crop_paths + original_paths

        if set(new_paths) != set(paths):
            raise AssertionError(
                f"Sample set changed: {arm}"
            )

        reordered[arm] = new_paths

    lists = {
        "uniform": uniform_new,
        "scalar_risk": reordered[
            "scalar_risk"
        ],
        "typed_cafr": reordered[
            "typed_cafr"
        ],
    }

    final_dev_list = V2 / "dev_cache_safe.txt"

    for arm, paths in lists.items():
        if len(paths) != 65_000:
            raise AssertionError(
                (arm, len(paths))
            )
        if len(set(paths)) != 65_000:
            raise AssertionError(
                f"Duplicate paths: {arm}"
            )

        output = (
            v2_incomplete
            / f"{arm}_train.txt"
        )
        output.write_text(
            "\n".join(paths) + "\n",
            encoding="utf-8",
        )

    (
        v2_incomplete
        / "dev_cache_safe.txt"
    ).write_text(
        "\n".join(dev_new) + "\n",
        encoding="utf-8",
    )

    final_cache_routes = {}

    for arm, paths in lists.items():
        final_cache_routes[arm] = str(
            predicted_cache(paths[0])
        )

        final_train_list = (
            V2 / f"{arm}_train.txt"
        )
        yaml_path = (
            v2_incomplete
            / f"{arm}_data.yaml"
        )
        yaml_path.write_text(
            yaml_text(
                final_train_list,
                final_dev_list,
            ),
            encoding="utf-8",
        )

    final_cache_routes["dev"] = str(
        predicted_cache(dev_new[0])
    )

    routes = list(
        final_cache_routes.values()
    )
    if len(routes) != len(set(routes)):
        raise AssertionError(
            final_cache_routes
        )

    for route in routes:
        if not route.startswith("/dev/shm/"):
            raise AssertionError(route)

    amendment = {
        "version": (
            "method_gate_cache_isolation_v1"
        ),
        "status": (
            "frozen_technical_amendment_before_training"
        ),
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_materialization": str(V1),
        "source_materialization_manifest_sha256": (
            EXPECTED[V1_MANIFEST]
        ),
        "method_gate_protocol": str(
            PROTOCOL
        ),
        "method_gate_protocol_sha256": (
            EXPECTED[PROTOCOL]
        ),
        "reason": (
            "Ultralytics derives label-cache location "
            "from the first label directory; all V1 "
            "lists resolved to the same original "
            "labels/train.cache."
        ),
        "changes": {
            "uniform": (
                "First image/label replaced by symlink "
                "aliases to the identical source files."
            ),
            "scalar_risk": (
                "Same 65k path set reordered with a "
                "Scalar crop first."
            ),
            "typed_cafr": (
                "Same 65k path set reordered with a "
                "Typed crop first."
            ),
            "dev": (
                "First image/label replaced by symlink "
                "aliases to the identical source files."
            ),
        },
        "sample_sets_changed": False,
        "labels_changed": False,
        "images_changed": False,
        "training_started": False,
        "cache_routes": final_cache_routes,
    }

    amendment_path = (
        v2_incomplete
        / "cache_isolation_amendment.json"
    )
    amendment_path.write_text(
        json.dumps(
            amendment,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    artifact_files = {}
    for path in sorted(
        v2_incomplete.iterdir()
    ):
        if path.is_file():
            artifact_files[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    artifact_manifest = {
        "version": (
            "method_gate_cache_isolation_v1"
        ),
        "status": "frozen",
        "files": artifact_files,
    }

    artifact_path = (
        v2_incomplete
        / "artifact_manifest.json"
    )
    artifact_path.write_text(
        json.dumps(
            artifact_manifest,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    anchor_incomplete.rename(ANCHORS)
    v2_incomplete.rename(V2)

    print()
    print("=== FINAL CACHE ROUTES ===")

    for arm, route in (
        final_cache_routes.items()
    ):
        print(f"{arm}: {route}")

        path = Path(route)
        if path.exists():
            raise AssertionError(
                f"Cache unexpectedly exists: {path}"
            )


    assert (
        Path(uniform_new[0]).resolve()
        == uniform_source.resolve()
    )
    assert (
        Path(dev_new[0]).resolve()
        == dev_source.resolve()
    )

    print()
    print("=" * 92)
    print("METHOD-GATE CACHE ISOLATION COMPLETE")
    print("=" * 92)
    print("output:", V2)
    print("anchors:", ANCHORS)
    print(
        "artifact_manifest_sha256:",
        sha256_file(
            V2 / "artifact_manifest.json"
        ),
    )
    print("PASS: four label caches are isolated in /dev/shm")
    print("PASS: image and label sample semantics are unchanged")
    print("NOTHING TRAINED")


if __name__ == "__main__":
    main()
