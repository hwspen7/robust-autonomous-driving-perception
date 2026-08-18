from pathlib import Path
import torch

from torch import device
from ultralytics import YOLO

def main()->None:
    project_root = Path(__file__).resolve().parents[3]
    input_path = project_root / "results" / "smoke_test" / "input.jpg"
    output_dir = project_root / "results" / "smoke_test"

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} does not exist"
        )
    model = YOLO("yolo26n.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = model.predict(
        source=str(input_path),
        device=device,
        conf=0.25,
        save=True,
        project=str(output_dir),
        name="yolo",
        exist_ok=True,
    )

    print(f"YOLO inference complete, processing {len(result)} images")

if __name__ == '__main__':
    main()

