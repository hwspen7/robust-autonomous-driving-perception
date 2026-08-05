from pathlib import Path
from mmdet.apis import DetInferencer

import torch


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "results" / "smoke_test" / "input.jpg"
    output_dir = project_root / "results" / "smoke_test" / "mask_rcnn"

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    inferencer = DetInferencer(
        model="mask-rcnn_r50_fpn_1x_coco",
        device=device,
    )

    inferencer(
        inputs=str(input_path),
        out_dir=str(output_dir),
        pred_score_thr=0.3,
        no_save_pred=False,
    )

    print(f"Mask R-CNN Inferencer Results ({device}): {output_dir}")


if __name__ == '__main__':
    main()
