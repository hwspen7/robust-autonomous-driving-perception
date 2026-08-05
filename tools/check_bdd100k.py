from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def find_bdd_root() -> Path:
    candidates = [
        PROJECT_ROOT / "datasets" / "downloads" / "bdd100k" / "bdd100k:-images-100k",
        PROJECT_ROOT / "datasets" / "bdd100k" / "raw" / "supervisely",
        PROJECT_ROOT / "datasets" / "bdd100k" / "subsets" / "local_debug",
    ]

    for candidate in candidates:
        if (candidate / "train" / "img").exists():
            return candidate

    raise FileNotFoundError(
        "BDD100K root not found. Searched:\n"
        + "\n".join(f"  - {candidate}" for candidate in candidates)
    )

BDD_ROOT = find_bdd_root()

print("BDD root:", BDD_ROOT)

for split in [
    "train",
    "val",
    "test",
]:

    img_dir = (
            BDD_ROOT
            /
            split
            /
            "img"
    )

    ann_dir = (
            BDD_ROOT
            /
            split
            /
            "ann"
    )

    if img_dir.exists():

        images = list(
            img_dir.glob("*")
        )

        anns = list(
            ann_dir.glob("*.json")
        )

        print("=" * 40)

        print(
            split
        )

        print(
            "Images:",
            len(images)
        )

        print(
            "Annotations:",
            len(anns)
        )

    else:

        print(
            split,
            "Not found"
        )
