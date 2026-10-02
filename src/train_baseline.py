"""Train a reproducible YOLOv8n baseline on a prepared YOLO dataset.

Run from the repository root (for example, in Google Colab):
    python src/train_baseline.py --data configs/data.yaml

Ultralytics writes the best checkpoint to runs/detect/baseline/weights/best.pt
and training metrics/plots beside it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "configs" / "data.yaml"
RUNS_DIR = ROOT / "runs" / "detect"
RUN_NAME = "baseline"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="YOLO dataset YAML (default: configs/data.yaml)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_yaml = args.data.expanduser()
    if not data_yaml.is_absolute():
        data_yaml = (Path.cwd() / data_yaml).resolve()
    else:
        data_yaml = data_yaml.resolve()

    if not data_yaml.is_file():
        print(
            f"ERROR: Dataset YAML not found: {data_yaml}\n"
            "Prepare the YOLO dataset and create configs/data.yaml before training.",
            file=sys.stderr,
        )
        return 2

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        print(
            f"ERROR: Missing training dependency ({exc}). In Colab run "
            "`pip install -r requirements.txt` first.",
            file=sys.stderr,
        )
        return 2

    if not torch.cuda.is_available():
        print(
            "ERROR: No CUDA GPU is available. Enable a GPU runtime in Colab "
            "(Runtime > Change runtime type > T4 GPU) and rerun.",
            file=sys.stderr,
        )
        return 2

    run_dir = RUNS_DIR / RUN_NAME
    if run_dir.exists():
        print(
            f"ERROR: Run directory already exists: {run_dir}\n"
            "Move it or choose a new run name so this baseline is not overwritten.",
            file=sys.stderr,
        )
        return 2

    print(f"Dataset YAML: {data_yaml}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Starting YOLOv8n COCO-pretrained training: 30 epochs, imgsz=640, seed=42")

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml),
        epochs=30,
        imgsz=640,
        seed=SEED,
        deterministic=True,
        pretrained=True,
        device=0,
        workers=2,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=False,
        save=True,
        plots=True,
    )

    best_weights = run_dir / "weights" / "best.pt"
    results_csv = run_dir / "results.csv"
    results_plot = run_dir / "results.png"
    if not best_weights.is_file():
        print(f"ERROR: Training finished without expected checkpoint: {best_weights}", file=sys.stderr)
        return 1

    print("\nTraining complete.")
    print(f"Best weights: {best_weights}")
    print(f"Metrics CSV:  {results_csv}")
    print(f"Curves plot:  {results_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
