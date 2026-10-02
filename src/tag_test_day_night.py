"""Tag YOLO test images as day/night and make condition-specific test YAMLs.

Run after converting ENA24 and creating configs/data.yaml:

    python src/tag_test_day_night.py

Grayscale/IR-looking images are classified as night first. Other images use
mean grayscale brightness and the configured threshold. Existing generated
outputs are preserved unless --overwrite is supplied.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class TaggedImage:
    source_image: Path
    source_label: Path
    condition: str
    brightness: float
    grayscale_fraction: float
    method: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yaml-path",
        type=Path,
        default=ROOT / "configs" / "data.yaml",
        help="Base YOLO dataset YAML (default: configs/data.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "day_night_test",
        help="Root for copied day/night test subsets.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory for CSV, histogram, and sample contact sheets.",
    )
    parser.add_argument("--brightness-threshold", type=float, default=80.0, help="Mean luminance cutoff on a 0-255 scale (default: 80).")
    parser.add_argument("--gray-fraction-threshold", type=float, default=0.95, help="Classify as grayscale/IR if this fraction of sampled pixels is near-gray (default: 0.95).")
    parser.add_argument("--gray-channel-delta", type=int, default=10, help="Per-pixel max(R,G,B)-min(R,G,B) cutoff for near-gray pixels (default: 10).")
    parser.add_argument("--samples-per-group", type=int, default=8, help="Random images shown per group (default: 8).")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42).")
    parser.add_argument("--overwrite", action="store_true", help="Replace outputs previously created by this script.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
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


def resolve_dataset_root(config: dict[str, Any], yaml_path: Path) -> Path:
    value = config.get("path")
    if value:
        root = Path(str(value))
        if not root.is_absolute():
            root = (yaml_path.parent / root).resolve()
        return root
    return yaml_path.parent.resolve()


def split_images_path(config: dict[str, Any], dataset_root: Path, split: str) -> Path:
    value = config.get(split)
    if not value or not isinstance(value, (str, Path)):
        raise ValueError(f"The base YAML needs a single image path for '{split}'.")
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (dataset_root / path).resolve()


def analyze_image(
    image_path: Path,
    brightness_threshold: float,
    gray_fraction_threshold: float,
    gray_channel_delta: int,
) -> tuple[str, float, float, str]:
    try:
        from PIL import Image, ImageStat
    except ImportError as exc:
        raise SystemExit("Pillow is required. Install project dependencies with `pip install -r requirements.txt`.") from exc

    with Image.open(image_path) as source:
        source_mode = source.mode
        rgb = source.convert("RGB")
        rgb.thumbnail((256, 256))
        pixels = list(rgb.getdata())
        brightness = float(ImageStat.Stat(rgb.convert("L")).mean[0])
    if not pixels:
        raise ValueError("image contains no pixels")

    near_gray_pixels = sum(1 for red, green, blue in pixels if max(red, green, blue) - min(red, green, blue) <= gray_channel_delta)
    gray_fraction = near_gray_pixels / len(pixels)
    grayscale_mode = source_mode in {"1", "L", "I", "F", "I;16", "I;16B", "I;16L"}

    # Treat explicit grayscale and strongly near-monochrome images as IR/night
    # before applying the brightness cutoff to the remaining color images.
    if grayscale_mode or gray_fraction >= gray_fraction_threshold:
        return "night", brightness, gray_fraction, "grayscale_ir"
    condition = "night" if brightness < brightness_threshold else "day"
    return condition, brightness, gray_fraction, "brightness"


def normalized_names(config: dict[str, Any]) -> dict[int, str]:
    names = config.get("names")
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    if isinstance(names, dict):
        result: dict[int, str] = {}
        for key, value in names.items():
            try:
                result[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return result
    raise ValueError("The base dataset YAML must define class names as a list or integer-keyed mapping.")


def safe_yaml_dump(path: Path, value: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(value, stream, allow_unicode=True, sort_keys=False)


def save_contact_sheet(
    tagged: list[TaggedImage],
    condition: str,
    destination: Path,
    samples_per_group: int,
    seed: int,
) -> int:
    from PIL import Image, ImageDraw, ImageFont

    group_images = [item for item in tagged if item.condition == condition]
    sample = random.Random(seed + (condition == "night")).sample(
        group_images, min(samples_per_group, len(group_images))
    )
    if not sample:
        return 0

    columns = 4
    cell_width, image_height, caption_height = 320, 190, 36
    rows = (len(sample) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * (image_height + caption_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, item in enumerate(sample):
        with Image.open(item.source_image) as source:
            image = source.convert("RGB")
        image.thumbnail((cell_width - 12, image_height - 12))
        x = (index % columns) * cell_width
        y = (index // columns) * (image_height + caption_height)
        image_x = x + (cell_width - image.width) // 2
        image_y = y + (image_height - image.height) // 2
        sheet.paste(image, (image_x, image_y))
        caption = f"{item.source_image.name} | {item.brightness:.1f}"
        draw.text((x + 5, y + image_height + 4), caption[:48], fill="black", font=font)

    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG", optimize=True)
    return len(sample)


def main() -> int:
    args = parse_args()
    if args.samples_per_group < 0:
        raise SystemExit("--samples-per-group cannot be negative.")
    if not 0.0 <= args.brightness_threshold <= 255.0:
        raise SystemExit("--brightness-threshold must be between 0 and 255.")
    if not 0.0 <= args.gray_fraction_threshold <= 1.0:
        raise SystemExit("--gray-fraction-threshold must be between 0 and 1.")
    if args.gray_channel_delta < 0 or args.gray_channel_delta > 255:
        raise SystemExit("--gray-channel-delta must be between 0 and 255.")

    config = load_config(args.yaml_path)
    dataset_root = resolve_dataset_root(config, args.yaml_path)
    try:
        test_images_dir = split_images_path(config, dataset_root, "test")
        train_images_dir = split_images_path(config, dataset_root, "train")
        val_images_dir = split_images_path(config, dataset_root, "val")
        names = normalized_names(config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    test_labels_dir = test_images_dir.parent.parent / "labels" / "test"
    if not test_images_dir.is_dir():
        raise SystemExit(f"Test image directory not found: {test_images_dir}. Run src/convert_ena24.py first.")
    if not test_labels_dir.is_dir():
        raise SystemExit(f"Test label directory not found: {test_labels_dir}.")
    if not names:
        raise SystemExit("No class names found in the base dataset YAML.")

    images = sorted(
        path for path in test_images_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    )
    if not images:
        raise SystemExit(f"No supported test images found in {test_images_dir}.")

    tagged: list[TaggedImage] = []
    missing_labels: list[Path] = []
    unreadable_images: list[tuple[Path, str]] = []
    for image_path in images:
        label_path = test_labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            missing_labels.append(image_path)
            continue
        try:
            condition, brightness, gray_fraction, method = analyze_image(
                image_path,
                args.brightness_threshold,
                args.gray_fraction_threshold,
                args.gray_channel_delta,
            )
        except Exception as exc:
            unreadable_images.append((image_path, f"{type(exc).__name__}: {exc}"))
            continue
        tagged.append(TaggedImage(image_path, label_path, condition, brightness, gray_fraction, method))
    if missing_labels:
        print(f"Problem: {len(missing_labels)} test image(s) have no matching label:")
        for path in missing_labels[:50]:
            print(f"- {path}")
        if len(missing_labels) > 50:
            print(f"- ... and {len(missing_labels) - 50} more")
        return 1
    if unreadable_images:
        print(f"Problem: {len(unreadable_images)} test image(s) could not be read:")
        for path, error in unreadable_images[:50]:
            print(f"- {path}: {error}")
        if len(unreadable_images) > 50:
            print(f"- ... and {len(unreadable_images) - 50} more")
        return 1
    if not tagged:
        raise SystemExit("No readable test images with matching labels were found.")

    results_dir = args.results_dir
    csv_path = results_dir / "test_day_night.csv"
    histogram_path = results_dir / "test_day_night_brightness.png"
    day_sheet_path = results_dir / "day_samples.png"
    night_sheet_path = results_dir / "night_samples.png"
    day_yaml_path = ROOT / "configs" / "day_test.yaml"
    night_yaml_path = ROOT / "configs" / "night_test.yaml"
    generated_files = (csv_path, histogram_path, day_sheet_path, night_sheet_path, day_yaml_path, night_yaml_path)
    group_output_dirs = (args.output_dir / "day", args.output_dir / "night")

    if not args.overwrite:
        existing = [path for path in generated_files if path.exists()]
        existing.extend(path for path in group_output_dirs if path.exists() and any(path.rglob("*")))
        if existing:
            raise SystemExit(
                "Generated outputs already exist. Choose a new --output-dir/--results-dir or pass --overwrite. "
                f"First existing path: {existing[0]}"
            )
    else:
        for path in group_output_dirs:
            if path.exists():
                shutil.rmtree(path)
        for path in generated_files:
            if path.exists() and path.is_file():
                path.unlink()

    counts = Counter(item.condition for item in tagged)
    for item in tagged:
        group_root = args.output_dir / item.condition
        image_destination = group_root / "images" / "test" / item.source_image.name
        label_destination = group_root / "labels" / "test" / item.source_label.name
        image_destination.parent.mkdir(parents=True, exist_ok=True)
        label_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.source_image, image_destination)
        shutil.copy2(item.source_label, label_destination)

    results_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("filename", "condition", "brightness"))
        writer.writeheader()
        for item in tagged:
            writer.writerow(
                {
                    "filename": item.source_image.name,
                    "condition": item.condition,
                    "brightness": f"{item.brightness:.4f}",
                }
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9, 5.5))
    bins = list(range(0, 261, 10))
    for condition, color in (("day", "#e5a500"), ("night", "#4267a9")):
        values = [item.brightness for item in tagged if item.condition == condition]
        if values:
            axis.hist(values, bins=bins, alpha=0.65, color=color, label=f"{condition} (n={len(values)})")
    axis.axvline(args.brightness_threshold, color="#d62728", linestyle="--", linewidth=1.5,
                 label=f"brightness threshold ({args.brightness_threshold:g})")
    axis.set(title="Test image mean brightness", xlabel="Mean grayscale brightness (0-255)", ylabel="Images")
    axis.set_xlim(0, 255)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(histogram_path, dpi=160)
    plt.close(fig)

    day_samples = save_contact_sheet(tagged, "day", day_sheet_path, args.samples_per_group, args.seed)
    night_samples = save_contact_sheet(tagged, "night", night_sheet_path, args.samples_per_group, args.seed)

    for condition, yaml_path in (("day", day_yaml_path), ("night", night_yaml_path)):
        group_root = args.output_dir / condition
        yaml_config = {
            "path": str(group_root.resolve()),
            "train": str(train_images_dir.resolve()),
            "val": str(val_images_dir.resolve()),
            "test": "images/test",
            "names": {class_id: names[class_id] for class_id in sorted(names)},
        }
        safe_yaml_dump(yaml_path, yaml_config)

    gray_night_count = sum(item.condition == "night" and item.method == "grayscale_ir" for item in tagged)
    print(f"Test images tagged: {len(tagged)}")
    print(f"Day: {counts['day']}; Night: {counts['night']} (grayscale/IR rule: {gray_night_count})")
    print(f"Brightness threshold: {args.brightness_threshold:g} / 255")
    print(f"CSV: {csv_path.resolve()}")
    print(f"Brightness histogram: {histogram_path.resolve()}")
    print(f"Day samples: {day_samples} in {day_sheet_path.resolve()}")
    print(f"Night samples: {night_samples} in {night_sheet_path.resolve()}")
    print(f"Day test YAML: {day_yaml_path.resolve()}")
    print(f"Night test YAML: {night_yaml_path.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Tagging interrupted.", file=sys.stderr)
        raise SystemExit(130)
