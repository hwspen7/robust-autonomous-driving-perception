import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download BDD100K Images 100K using Dataset Ninja."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "downloads" / "bdd100k",
        help="Output directory for downloads.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="BDD100K: Images 100K",
        help="Dataset Ninja dataset name.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run the download again even if the destination directory already exists.",
    )
    return parser.parse_args()

def looks_downloaded(
    output_dir: Path,
) -> bool:
    candidates = (
        output_dir / "bdd100k:-images-100k",
        output_dir / "bdd100k-images-100k",
    )

    return any(
        (candidate / "train" / "ann").exists()
        and (candidate / "train" / "img").exists()
        for candidate in candidates
    )

def main() -> None:
    args = parse_args()
    output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if looks_downloaded(output_dir) and not args.force:
        print("BDD100K already exists.")
        print("Location:", output_dir)
        return

    print("Downloading BDD100K...")

    try:
        import dataset_tools as dtools
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Dataset Ninja dependencies are not installed. "
            "Install them only when you need runtime downloading:\n"
            "  pip install -r requirements-dataset-ninja.txt"
        ) from exc

    dtools.download(
        dataset=args.dataset,
        dst_dir=str(output_dir),
    )

    print("\nBDD100K download finished.")
    print("Location:", output_dir)

if __name__ == "__main__":
    main()
