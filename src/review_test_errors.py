"""Find and visualize false negatives/positives on the YOLO test split.

Run from the repository root:
    python src/review_test_errors.py

Matching is class-aware and uses greedy one-to-one matching at IoU >= 0.5.
Predictions below the configurable confidence threshold are ignored. Review
images and a per-error CSV are written under results/error_review/.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "baseline" / "weights" / "best.pt"
DEFAULT_DATA = ROOT / "configs" / "data.yaml"
DEFAULT_CONDITIONS = ROOT / "results" / "test_day_night.csv"
OUTPUT_DIR = ROOT / "results" / "error_review"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float | None = None

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--conditions", type=Path, default=DEFAULT_CONDITIONS,
                        help="Day/night CSV from src/tag_test_day_night.py; optional")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Minimum prediction confidence for FP/matching (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=50, help="Maximum reviewed examples of each error type")
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def read_yaml(path: Path) -> dict[str, Any]:
    import yaml
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def get_test_images(config: dict[str, Any], yaml_path: Path) -> tuple[Path, Path]:
    root_value = config.get("path", yaml_path.parent)
    root = Path(root_value)
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    test_value = config.get("test")
    if not test_value:
        raise ValueError("Dataset YAML must define a test path.")
    test_path = Path(test_value)
    if not test_path.is_absolute():
        test_path = (root / test_path).resolve()
    if test_path.is_file() and test_path.suffix.lower() == ".txt":
        image_paths = []
        for line in test_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                p = Path(line)
                image_paths.append(p if p.is_absolute() else (test_path.parent / p).resolve())
        label_root = None
    elif test_path.is_dir():
        image_paths = sorted(p for p in test_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        # Standard Ultralytics layout: dataset/images/test <-> dataset/labels/test.
        if test_path.parent.name == "images":
            label_root = test_path.parent.parent / "labels" / test_path.name
        else:
            label_root = test_path.parent.parent / "labels" / test_path.name
    else:
        raise FileNotFoundError(f"Test image path does not exist or is unsupported: {test_path}")
    if not image_paths:
        raise ValueError(f"No supported test images found under {test_path}")
    if label_root is None:
        raise ValueError("A TXT image list is not supported for labels; use an images/test directory in the YAML.")
    return image_paths, label_root


def load_gt(label_path: Path, width: int, height: int) -> list[Box]:
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing ground-truth label: {label_path}")
    boxes = []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"Malformed YOLO row at {label_path}:{line_no}: expected 5 values")
        cls, cx, cy, bw, bh = map(float, parts)
        boxes.append(Box(int(cls), (cx - bw / 2) * width, (cy - bh / 2) * height,
                         (cx + bw / 2) * width, (cy + bh / 2) * height))
    return boxes


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(gt_boxes: list[Box], pred_boxes: list[Box], threshold: float) -> tuple[list[int], list[int]]:
    candidates = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            if gt.cls == pred.cls:
                overlap = iou(gt, pred)
                if overlap >= threshold:
                    candidates.append((overlap, gi, pi))
    candidates.sort(reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for _, gi, pi in candidates:
        if gi not in matched_gt and pi not in matched_pred:
            matched_gt.add(gi)
            matched_pred.add(pi)
    return [i for i in range(len(gt_boxes)) if i not in matched_gt], [i for i in range(len(pred_boxes)) if i not in matched_pred]


def load_conditions(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        filename_key = next((key for key in ("filename", "file", "image") if key in (rows.fieldnames or [])), None)
        condition_key = next((key for key in ("condition", "day_night") if key in (rows.fieldnames or [])), None)
        if not filename_key or not condition_key:
            raise ValueError(f"Condition CSV needs filename and condition columns: {path}")
        return {Path(row[filename_key]).name: row[condition_key].strip().lower() for row in rows if row.get(filename_key)}


def size_label(box: Box, image_area: float) -> str:
    fraction = box.area / image_area if image_area else 0
    if fraction < 0.01:
        return "small (<1% image area)"
    if fraction < 0.1:
        return "medium (1-10% image area)"
    return "large (>=10% image area)"


def draw_image(image_path: Path, gt_boxes: list[Box], pred_boxes: list[Box], selected: list[tuple[str, int]], names: Any, save_path: Path) -> None:
    from PIL import Image, ImageDraw
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    # Context: all GTs in green, all predictions in blue; selected errors in red.
    for gt in gt_boxes:
        draw.rectangle((gt.x1, gt.y1, gt.x2, gt.y2), outline="#35a853", width=2)
    for pred in pred_boxes:
        draw.rectangle((pred.x1, pred.y1, pred.x2, pred.y2), outline="#4285f4", width=2)
    for kind, index in selected:
        box = gt_boxes[index] if kind == "false_negative" else pred_boxes[index]
        draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline="#ea4335", width=4)
        name = names.get(box.cls, str(box.cls)) if isinstance(names, dict) else names[box.cls]
        suffix = f" {box.confidence:.2f}" if box.confidence is not None else ""
        draw.text((box.x1, max(0, box.y1 - 14)), f"{kind[:2].upper()} {name}{suffix}", fill="#ea4335", stroke_width=2, stroke_fill="white")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)


def main() -> int:
    args = parse_args()
    weights, data_yaml = resolve(args.weights), resolve(args.data)
    conditions_path = resolve(args.conditions)
    missing = [str(p) for p in (weights, data_yaml) if not p.is_file()]
    if missing:
        print("ERROR: Required input(s) missing:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 2
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1 or args.limit < 0:
        print("ERROR: --conf and --iou must be within [0,1], and --limit cannot be negative.", file=sys.stderr)
        return 2
    try:
        from PIL import Image
        from ultralytics import YOLO
        config = read_yaml(data_yaml)
        image_paths, label_root = get_test_images(config, data_yaml)
        conditions = load_conditions(conditions_path)
    except (ImportError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not conditions:
        print(f"WARNING: No day/night metadata loaded from {conditions_path}; condition will be 'unknown'.")

    model = YOLO(str(weights))
    names = model.names
    events: list[dict[str, Any]] = []
    images_with_errors: dict[str, dict[str, Any]] = {}
    fn_sample_images: set[str] = set()
    fp_sample_images: set[str] = set()

    for start in range(0, len(image_paths), 16):
        batch_paths = image_paths[start:start + 16]
        predictions = model.predict(source=[str(p) for p in batch_paths], conf=args.conf, imgsz=args.imgsz,
                                    device=args.device, verbose=False, stream=False)
        for image_path, result in zip(batch_paths, predictions):
            with Image.open(image_path) as im:
                width, height = im.size
            label_path = label_root / f"{image_path.stem}.txt"
            gt_boxes = load_gt(label_path, width, height)
            pred_boxes = []
            if result.boxes is not None:
                xyxy = result.boxes.xyxy.cpu().tolist()
                classes = result.boxes.cls.cpu().tolist()
                confs = result.boxes.conf.cpu().tolist()
                pred_boxes = [Box(int(cls), *map(float, coords), float(conf)) for coords, cls, conf in zip(xyxy, classes, confs)]
            fn_indices, fp_indices = match_boxes(gt_boxes, pred_boxes, args.iou)
            if not fn_indices and not fp_indices:
                continue
            condition = conditions.get(image_path.name, "unknown")
            image_area = width * height
            save_fn_image = bool(fn_indices) and str(image_path) in fn_sample_images
            save_fp_image = bool(fp_indices) and str(image_path) in fp_sample_images
            if fn_indices and not save_fn_image and len(fn_sample_images) < args.limit:
                fn_sample_images.add(str(image_path))
                save_fn_image = True
            if fp_indices and not save_fp_image and len(fp_sample_images) < args.limit:
                fp_sample_images.add(str(image_path))
                save_fp_image = True
            for index in fn_indices:
                gt = gt_boxes[index]
                name = names.get(gt.cls, str(gt.cls)) if isinstance(names, dict) else names[gt.cls]
                events.append({"filename": image_path.name, "condition": condition, "error_type": "false_negative",
                               "class_id": gt.cls, "class_name": name, "box_width_px": round(gt.width, 2),
                               "box_height_px": round(gt.height, 2), "box_area_px2": round(gt.area, 2),
                               "box_area_fraction": round(gt.area / image_area, 6), "size_category": size_label(gt, image_area),
                               "confidence": "", "iou_threshold": args.iou})
            for index in fp_indices:
                pred = pred_boxes[index]
                name = names.get(pred.cls, str(pred.cls)) if isinstance(names, dict) else names[pred.cls]
                events.append({"filename": image_path.name, "condition": condition, "error_type": "false_positive",
                               "class_id": pred.cls, "class_name": name, "box_width_px": round(pred.width, 2),
                               "box_height_px": round(pred.height, 2), "box_area_px2": round(pred.area, 2),
                               "box_area_fraction": round(pred.area / image_area, 6), "size_category": size_label(pred, image_area),
                               "confidence": round(pred.confidence or 0, 6), "iou_threshold": args.iou})
            if save_fn_image:
                selected = [("false_negative", index) for index in fn_indices]
                draw_image(image_path, gt_boxes, pred_boxes, selected, names,
                           OUTPUT_DIR / "false_negatives" / f"{image_path.stem}.jpg")
            if save_fp_image:
                selected = [("false_positive", index) for index in fp_indices]
                draw_image(image_path, gt_boxes, pred_boxes, selected, names,
                           OUTPUT_DIR / "false_positives" / f"{image_path.stem}.jpg")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "test_errors.csv"
    fields = ["filename", "condition", "error_type", "class_id", "class_name", "box_width_px", "box_height_px",
              "box_area_px2", "box_area_fraction", "size_category", "confidence", "iou_threshold"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)

    print(f"Test images scanned: {len(image_paths)}")
    print(f"False negatives: {sum(e['error_type'] == 'false_negative' for e in events)}; reviewed images saved: {len(fn_sample_images)}")
    print(f"False positives: {sum(e['error_type'] == 'false_positive' for e in events)}; reviewed images saved: {len(fp_sample_images)}")
    print(f"Error CSV: {csv_path.resolve()}")
    print(f"Annotated images: {(OUTPUT_DIR.resolve())}")
    if not conditions:
        print("Condition breakdown unavailable (day/night CSV missing or empty).")
    else:
        for kind in ("false_negative", "false_positive"):
            rows = [e for e in events if e["error_type"] == kind]
            print(f"\n{kind.replace('_', ' ').title()} by condition:")
            for condition in ("day", "night", "unknown"):
                print(f"  {condition}: {sum(e['condition'] == condition for e in rows)}")
            print(f"  By size: {dict((category, sum(e['size_category'] == category for e in rows)) for category in ('small (<1% image area)', 'medium (1-10% image area)', 'large (>=10% image area)'))}")
            print("  By class: " + (str({name: sum(e['class_name'] == name for e in rows) for name in sorted({e['class_name'] for e in rows})}) if rows else "none"))
    print("These are descriptive counts only; inspect the saved images before attributing causes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
