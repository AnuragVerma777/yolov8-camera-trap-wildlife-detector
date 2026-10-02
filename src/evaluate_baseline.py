"""Evaluate the baseline checkpoint on full, day, and night test sets.

Run from the repository root after training and day/night tagging:
    python src/evaluate_baseline.py

Outputs a combined summary/per-class CSV and a summary metric chart in outputs/.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "baseline" / "weights" / "best.pt"
DEFAULT_CONFIGS = {
    "full_test": ROOT / "configs" / "data.yaml",
    "day": ROOT / "configs" / "day_test.yaml",
    "night": ROOT / "configs" / "night_test.yaml",
}
OUTPUT_DIR = ROOT / "outputs"
CSV_PATH = OUTPUT_DIR / "baseline_test_metrics.csv"
CHART_PATH = OUTPUT_DIR / "baseline_test_metrics.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--full-yaml", type=Path, default=DEFAULT_CONFIGS["full_test"])
    parser.add_argument("--day-yaml", type=Path, default=DEFAULT_CONFIGS["day"])
    parser.add_argument("--night-yaml", type=Path, default=DEFAULT_CONFIGS["night"])
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def as_float(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def evaluate(model: Any, label: str, yaml_path: Path, device: str, imgsz: int) -> tuple[dict[str, float], list[dict[str, Any]]]:
    result = model.val(
        data=str(yaml_path),
        split="test",
        imgsz=imgsz,
        device=device,
        plots=False,
        verbose=False,
    )
    box = result.box
    summary = {
        "mAP50": as_float(box.map50),
        "mAP50_95": as_float(box.map),
        "precision": as_float(box.mp),
        "recall": as_float(box.mr),
    }
    class_ids = list(box.ap_class_index)
    ap50_values = list(box.ap50)
    ap_values = list(box.ap)
    names = result.names
    per_class = []
    for i, class_id in enumerate(class_ids):
        class_id = int(class_id)
        class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
        per_class.append({
            "class_id": class_id,
            "class_name": class_name,
            "AP50": as_float(ap50_values[i]),
            "AP50_95": as_float(ap_values[i]),
        })
    return summary, per_class


def main() -> int:
    args = parse_args()
    weights = resolve(args.weights)
    configs = {label: resolve(path) for label, path in {
        "full_test": args.full_yaml,
        "day": args.day_yaml,
        "night": args.night_yaml,
    }.items()}
    missing = [str(path) for path in [weights, *configs.values()] if not path.is_file()]
    if missing:
        print("ERROR: Required evaluation input(s) are missing:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        print("Train the baseline and run src/tag_test_day_night.py first.", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"ERROR: Missing evaluation dependency ({exc}); install requirements.txt.", file=sys.stderr)
        return 2

    model = YOLO(str(weights))
    summaries: dict[str, dict[str, float]] = {}
    class_rows: dict[str, list[dict[str, Any]]] = {}
    for label, yaml_path in configs.items():
        print(f"Evaluating {label}: {yaml_path}")
        summary, per_class = evaluate(model, label, yaml_path, args.device, args.imgsz)
        summaries[label] = summary
        class_rows[label] = per_class

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["subset", "row_type", "class_id", "class_name", "mAP50", "mAP50_95", "precision", "recall", "AP50", "AP50_95"]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label in configs:
            summary = summaries[label]
            writer.writerow({"subset": label, "row_type": "overall", **summary})
            for item in class_rows[label]:
                writer.writerow({"subset": label, "row_type": "per_class", **item})

    labels = list(configs)
    metric_keys = [("mAP50", "mAP@0.5"), ("mAP50_95", "mAP@0.5:0.95"), ("precision", "Precision"), ("recall", "Recall")]
    x = list(range(len(metric_keys)))
    width = 0.24
    colors = {"full_test": "#4c78a8", "day": "#f2a541", "night": "#526b9b"}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, label in enumerate(labels):
        vals = [summaries[label][key] for key, _ in metric_keys]
        offsets = [i + (index - (len(labels) - 1) / 2) * width for i in x]
        bars = ax.bar(offsets, vals, width=width, label=label.replace("_", " "), color=colors[label])
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x, [title for _, title in metric_keys])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("YOLOv8n baseline: test-set metrics by lighting condition")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=180)
    plt.close(fig)

    print("\nOverall metrics")
    print(f"{'Subset':<12} {'mAP@0.5':>10} {'mAP@0.5:0.95':>15} {'Precision':>11} {'Recall':>9}")
    for label in labels:
        m = summaries[label]
        print(f"{label:<12} {m['mAP50']:>10.4f} {m['mAP50_95']:>15.4f} {m['precision']:>11.4f} {m['recall']:>9.4f}")
        print("  Per-class AP: " + ", ".join(
            f"{item['class_name']} (AP50={item['AP50']:.4f}, AP50-95={item['AP50_95']:.4f})"
            for item in class_rows[label]
        ))
    print(f"\nCSV: {CSV_PATH.resolve()}")
    print(f"Chart: {CHART_PATH.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
