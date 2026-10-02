"""Validate a YOLO detection dataset and create visual QA artifacts.

Defaults match ``src/convert_ena24.py``:

    python src/validate_yolo.py

The report is printed to the terminal. A class-distribution chart and up to 12
random label overlays are written under ``results/``. The source dataset is
read-only; only the results directory is created or updated.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ImageLabelPair:
    split: str
    image_path: Path
    label_path: Path
    boxes: list[tuple[int, float, float, float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yaml-path",
        type=Path,
        default=ROOT / "configs" / "data.yaml",
        help="Ultralytics dataset YAML (default: configs/data.yaml).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory for the chart and image overlays (default: results/).",
    )
    parser.add_argument("--samples", type=int, default=12, help="Number of random overlays to save (default: 12).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for overlay sampling (default: 42).")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required. Install project dependencies with `pip install -r requirements.txt`.") from exc
    if not path.is_file():
        raise SystemExit(f"Dataset YAML not found: {path}. Run src/convert_ena24.py first.")
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"Dataset YAML must contain a mapping: {path}")
    return config


def normalize_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, list):
        return {index: str(name) for index, name in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        normalized: dict[int, str] = {}
        for key, value in raw_names.items():
            try:
                normalized[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return normalized
    return {}


def resolve_split_path(config: dict[str, Any], yaml_path: Path, split: str) -> Path:
    value = config.get(split)
    if not value:
        raise ValueError(f"Dataset YAML has no '{split}' path.")
    dataset_root_value = config.get("path")
    if dataset_root_value:
        dataset_root = Path(str(dataset_root_value))
        if not dataset_root.is_absolute():
            dataset_root = (yaml_path.parent / dataset_root).resolve()
    else:
        dataset_root = yaml_path.parent.resolve()
    split_path = Path(str(value))
    return split_path.resolve() if split_path.is_absolute() else (dataset_root / split_path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_image(path: Path) -> tuple[bool, str | None]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        # verify() checks the file structure but does not necessarily decode pixels.
        with Image.open(path) as image:
            image.load()
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def parse_label_file(
    label_path: Path,
    split: str,
    valid_class_ids: set[int],
    class_counts: Counter[int],
    problems: list[str],
) -> list[tuple[int, float, float, float, float]]:
    parsed: list[tuple[int, float, float, float, float]] = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        problems.append(f"[{split}] cannot read label {label_path}: {exc}")
        return parsed

    for line_number, line in enumerate(lines, start=1):
        content = line.strip()
        if not content:
            continue
        parts = content.split()
        if len(parts) != 5:
            problems.append(
                f"[{split}] {label_path} line {line_number}: expected class_id and 4 coordinates, got {len(parts)} fields."
            )
            continue
        try:
            class_value = int(parts[0])
        except ValueError:
            problems.append(f"[{split}] {label_path} line {line_number}: class id is not an integer: {parts[0]!r}.")
            continue
        if class_value not in valid_class_ids:
            problems.append(
                f"[{split}] {label_path} line {line_number}: invalid class id {class_value}; "
                f"expected one of {sorted(valid_class_ids)}."
            )
            continue
        try:
            coordinates = tuple(float(value) for value in parts[1:])
        except ValueError:
            problems.append(f"[{split}] {label_path} line {line_number}: coordinate is not numeric.")
            continue
        if not all(math.isfinite(value) for value in coordinates):
            problems.append(f"[{split}] {label_path} line {line_number}: coordinates must be finite numbers.")
            continue
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            problems.append(
                f"[{split}] {label_path} line {line_number}: coordinates must be between 0 and 1; "
                f"got {coordinates}."
            )
            continue
        class_counts[class_value] += 1
        parsed.append((class_value, *coordinates))
    return parsed


def draw_overlays(pairs: list[ImageLabelPair], class_names: dict[int, str], output_dir: Path, count: int, seed: int) -> int:
    if count <= 0 or not pairs:
        return 0
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise SystemExit("Pillow is required to draw overlays. Install project dependencies with `pip install -r requirements.txt`.") from exc

    overlay_dir = output_dir / "label_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    sample = random.Random(seed).sample(pairs, min(count, len(pairs)))
    font = ImageFont.load_default()
    colors = ((255, 64, 64), (40, 230, 80), (30, 170, 255), (255, 180, 30), (220, 70, 255), (0, 230, 220))
    saved = 0

    for pair in sample:
        try:
            with Image.open(pair.image_path) as source:
                image = source.convert("RGB")
            width, height = image.size
            draw = ImageDraw.Draw(image)
            for class_id, center_x, center_y, box_width, box_height in pair.boxes:
                x1 = max(0, min(width, int((center_x - box_width / 2) * width)))
                y1 = max(0, min(height, int((center_y - box_height / 2) * height)))
                x2 = max(0, min(width, int((center_x + box_width / 2) * width)))
                y2 = max(0, min(height, int((center_y + box_height / 2) * height)))
                color = colors[class_id % len(colors)]
                draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, round(min(width, height) / 500)))
                label = class_names.get(class_id, f"class_{class_id}")
                text_y = max(0, y1 - 16)
                draw.rectangle((x1, text_y, x1 + int(len(label) * 7.5) + 8, text_y + 15), fill=color)
                draw.text((x1 + 3, text_y + 2), label, fill="black", font=font)

            destination = overlay_dir / f"{pair.split}_{pair.image_path.stem}_boxes.jpg"
            image.save(destination, format="JPEG", quality=92)
            saved += 1
        except Exception as exc:
            print(f"Overlay warning: could not draw {pair.image_path}: {type(exc).__name__}: {exc}")
    return saved


def save_class_chart(
    class_names: dict[int, str],
    split_counts: dict[str, Counter[int]],
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Matplotlib is required to save the chart. Install project dependencies with `pip install -r requirements.txt`.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    class_ids = sorted(class_names)
    labels = [class_names[class_id] for class_id in class_ids]
    x_positions = list(range(len(class_ids)))
    width = 0.8 / max(len(SPLITS), 1)
    fig_width = max(9, 1.1 * len(class_ids))
    fig, axis = plt.subplots(figsize=(fig_width, 6))
    for split_index, split in enumerate(SPLITS):
        offset = (split_index - (len(SPLITS) - 1) / 2) * width
        values = [split_counts[split][class_id] for class_id in class_ids]
        axis.bar([x + offset for x in x_positions], values, width=width, label=split)
    axis.set_title("YOLO label distribution by split")
    axis.set_xlabel("Class")
    axis.set_ylabel("Valid bounding boxes")
    axis.set_xticks(x_positions, labels, rotation=35, ha="right")
    axis.legend(title="Split")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.samples < 0:
        raise SystemExit("--samples cannot be negative.")
    config = load_yaml(args.yaml_path)
    class_names = normalize_names(config.get("names"))
    if not class_names:
        raise SystemExit(f"Dataset YAML must define a nonempty 'names' list or mapping: {args.yaml_path}")
    valid_class_ids = set(class_names)
    problems: list[str] = []
    split_counts: dict[str, Counter[int]] = {split: Counter() for split in SPLITS}
    pairs_for_overlays: list[ImageLabelPair] = []
    hash_splits: dict[str, dict[str, Path]] = {}
    split_image_counts: Counter[str] = Counter()

    for split in SPLITS:
        try:
            image_dir = resolve_split_path(config, args.yaml_path, split)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        label_dir = image_dir.parent.parent / "labels" / split
        if not image_dir.is_dir():
            problems.append(f"[{split}] image directory not found: {image_dir}")
            continue
        if not label_dir.is_dir():
            problems.append(f"[{split}] label directory not found: {label_dir}")
            continue

        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        )
        split_image_counts[split] = len(images)
        if not images:
            problems.append(f"[{split}] no supported images found in {image_dir}.")
        labels = {path.stem.casefold(): path for path in label_dir.glob("*.txt") if path.is_file()}
        image_stems: set[str] = set()

        for image_path in images:
            stem_key = image_path.stem.casefold()
            if stem_key in image_stems:
                problems.append(f"[{split}] multiple images share label stem {image_path.stem!r}.")
            image_stems.add(stem_key)
            label_path = labels.get(stem_key)
            image_ok, image_error = verify_image(image_path)
            if not image_ok:
                problems.append(f"[{split}] corrupted image {image_path}: {image_error}")
            try:
                digest = sha256_file(image_path)
                split_map = hash_splits.setdefault(digest, {})
                if split not in split_map:
                    for previous_split, previous_path in split_map.items():
                        if previous_split != split:
                            problems.append(
                                f"Duplicate image content appears in both [{previous_split}] {previous_path} "
                                f"and [{split}] {image_path}."
                            )
                    split_map[split] = image_path
            except OSError as exc:
                problems.append(f"[{split}] cannot hash image {image_path}: {exc}")
            if label_path is None:
                problems.append(f"[{split}] image has no matching label file: {image_path}")
                continue
            parsed = parse_label_file(
                label_path,
                split,
                valid_class_ids,
                split_counts[split],
                problems,
            )
            if image_ok and parsed:
                pairs_for_overlays.append(ImageLabelPair(split, image_path, label_path, parsed))

        for label_stem, label_path in labels.items():
            if label_stem not in image_stems:
                problems.append(f"[{split}] label has no matching image: {label_path}")

    chart_path = args.results_dir / "class_distribution.png"
    save_class_chart(class_names, split_counts, chart_path)
    overlays_saved = draw_overlays(
        pairs_for_overlays,
        class_names,
        args.results_dir,
        args.samples,
        args.seed,
    )

    print("Split totals (images / valid boxes):")
    for split in SPLITS:
        print(f"{split:<6} {split_image_counts[split]:>7} images  {sum(split_counts[split].values()):>7} boxes")
    print("\nClass counts by split (valid boxes):")
    print("class id  class name" + "".join(f"  {split:>10}" for split in SPLITS) + "       total")
    for class_id in sorted(class_names):
        counts = [split_counts[split][class_id] for split in SPLITS]
        total = sum(counts)
        print(
            f"{class_id:>8}  {class_names[class_id]:<24}"
            + "".join(f"  {count:>10}" for count in counts)
            + f"  {total:>10}"
        )
    print(f"\nClass distribution chart: {chart_path.resolve()}")
    print(f"Box overlay images saved: {overlays_saved} (requested {args.samples}) in {args.results_dir / 'label_overlays'}")

    if problems:
        print(f"\nValidation problems ({len(problems)}):")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("\nValidation passed: image-label pairs, label values, image integrity, and split duplicates look good.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Validation interrupted.", file=sys.stderr)
        raise SystemExit(130)
