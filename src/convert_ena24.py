"""Convert the ENA24 Hugging Face Parquet export into a YOLO detection dataset.

Run from any working directory with:

    python src/convert_ena24.py

The default output is data/processed/ena24_yolo and configs/data.yaml. The
script never removes existing outputs; choose a new --output-dir if the target
already contains files.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
LOCATION_FIELDS = {
    "location",
    "location_id",
    "camera_location",
    "camera_site",
    "site",
    "site_id",
    "camera_id",
}
SEQUENCE_FIELDS = {
    "sequence",
    "sequence_id",
    "seq_id",
    "burst",
    "burst_id",
    "capture_sequence",
}
FOLDER_FIELDS = {
    "capture_folder",
    "camera_folder",
    "folder",
    "file_path",
    "filepath",
    "file_name",
    "filename",
    "path",
}
DIRECT_FOLDER_FIELDS = {"capture_folder", "camera_folder", "folder"}
PATH_FIELDS = {"file_path", "filepath", "file_name", "filename", "path"}


@dataclass
class ImageRecord:
    row_index: int
    image_id: Any
    width: int
    height: int
    boxes: list[tuple[int, float, float, float, float]]
    grouping_values: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=ROOT / "datset" / "data",
        help="Directory containing the ENA24 Parquet shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "ena24_yolo",
        help="Output root containing images/{train,val,test} and labels/{train,val,test}.",
    )
    parser.add_argument(
        "--yaml-path",
        type=Path,
        default=ROOT / "configs" / "data.yaml",
        help="Where to write the Ultralytics data YAML.",
    )
    parser.add_argument("--top-classes", type=int, default=5, help="Keep this many classes by box count (default: 5).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the split (default: 42).")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.70, 0.15, 0.15),
        help="Approximate split fractions (default: 0.70 0.15 0.15).",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Parquet rows per read batch.")
    return parser.parse_args()


def arrow_modules() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "Parquet support is missing. Install project dependencies with "
            "`pip install -r requirements.txt` (pyarrow is required)."
        ) from exc
    return pa, pq


def field_paths(schema: Any, pa: Any) -> list[str]:
    paths: list[str] = []

    def visit(field: Any, prefix: str) -> None:
        paths.append(prefix)
        if pa.types.is_struct(field.type):
            for child in field.type:
                visit(child, f"{prefix}.{child.name}")

    for field in schema:
        visit(field, field.name)
    return paths


def grouping_field_candidates(schema: Any, pa: Any) -> list[tuple[str, str]]:
    """Return grouping fields in preference order: location, sequence, folder."""
    paths = field_paths(schema, pa)
    result: list[tuple[str, str]] = []
    for group_type, names in (
        ("location", LOCATION_FIELDS),
        ("sequence", SEQUENCE_FIELDS),
        ("capture folder", FOLDER_FIELDS),
    ):
        for path in paths:
            if path.rsplit(".", 1)[-1].casefold() in names:
                if path not in {existing_path for existing_path, _ in result}:
                    result.append((path, group_type))
    return result


def get_nested(row: Mapping[str, Any], path: str) -> Any:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def scalar_group_value(value: Any, field_path: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        if field_path.rsplit(".", 1)[-1].casefold() in PATH_FIELDS:
            normalized = value.replace("\\", "/")
            path = PurePosixPath(normalized)
            if path.suffix:
                parent = path.parent.as_posix()
                # A bare filename has no usable capture-folder information.
                if parent in (".", ""):
                    return None
                return parent
            return path.as_posix()
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, (list, tuple, dict)):
        try:
            normalized = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return None
        return normalized or None
    return str(value)


def flatten_boxes(objects: Any) -> Iterable[tuple[Any, Any]]:
    if not isinstance(objects, Mapping):
        return ()
    boxes = objects.get("bbox") or []
    classes = objects.get("category") or []
    return zip(classes, boxes)


def normalize_box(class_id: Any, box: Any, width: int, height: int) -> tuple[int, float, float, float, float] | None:
    try:
        class_number = int(class_id)
        x, y, box_width, box_height = (float(value) for value in box[:4])
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
        return None
    if width <= 0 or height <= 0 or box_width <= 0 or box_height <= 0:
        return None

    # Clip COCO xywh boxes to the source image bounds before YOLO normalization.
    x_min = min(max(x, 0.0), float(width))
    y_min = min(max(y, 0.0), float(height))
    x_max = min(max(x + box_width, 0.0), float(width))
    y_max = min(max(y + box_height, 0.0), float(height))
    clipped_width = x_max - x_min
    clipped_height = y_max - y_min
    if clipped_width <= 0 or clipped_height <= 0:
        return None
    center_x = (x_min + clipped_width / 2.0) / width
    center_y = (y_min + clipped_height / 2.0) / height
    return class_number, center_x, center_y, clipped_width / width, clipped_height / height


def class_names_from_schema(schema: Any) -> dict[int, str]:
    """Use Hugging Face ClassLabel names when they are stored in Parquet metadata."""
    try:
        from datasets import Features

        features = Features.from_arrow_schema(schema)
    except (ImportError, AttributeError, ValueError, TypeError):
        return {}

    def find_names(feature: Any) -> list[str] | None:
        names = getattr(feature, "names", None)
        if names:
            return [str(name) for name in names]
        if isinstance(feature, Mapping):
            # Prefer the category feature rather than an unrelated ClassLabel.
            if "category" in feature:
                names = find_names(feature["category"])
                if names:
                    return names
            for child in feature.values():
                names = find_names(child)
                if names:
                    return names
        nested = getattr(feature, "feature", None)
        if nested is not None:
            return find_names(nested)
        return None

    names = find_names(features.get("objects", features))
    return {index: name for index, name in enumerate(names or [])}


def scan_shards(
    parquet_files: list[Path], batch_size: int, pa: Any, pq: Any
) -> tuple[list[ImageRecord], Counter[int], dict[int, str], list[tuple[str, str]]]:
    records: list[ImageRecord] = []
    class_counts: Counter[int] = Counter()
    all_candidates: list[tuple[str, str]] | None = None
    class_names: dict[int, str] = {}
    row_index = 0

    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        schema = parquet_file.schema_arrow
        required = [name for name in ("image_id", "width", "height", "objects") if name in schema.names]
        if "objects" not in required:
            raise ValueError(f"{parquet_path.name} has no 'objects' annotation column.")
        candidates = grouping_field_candidates(schema, pa)
        if all_candidates is None:
            all_candidates = candidates
            class_names = class_names_from_schema(schema)
        candidate_paths = [path for path, _ in candidates]
        columns = list(dict.fromkeys(required + candidate_paths))
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                try:
                    width = int(row.get("width") or 0)
                    height = int(row.get("height") or 0)
                except (TypeError, ValueError):
                    width = height = 0
                boxes = [
                    normalized
                    for class_id, box in flatten_boxes(row.get("objects"))
                    if (normalized := normalize_box(class_id, box, width, height)) is not None
                ]
                for class_id, *_ in boxes:
                    class_counts[class_id] += 1
                grouping_values = {path: get_nested(row, path) for path in candidate_paths}
                records.append(
                    ImageRecord(
                        row_index=row_index,
                        image_id=row.get("image_id"),
                        width=width,
                        height=height,
                        boxes=boxes,
                        grouping_values=grouping_values,
                    )
                )
                row_index += 1
    return records, class_counts, class_names, all_candidates or []


def choose_grouping(
    records: list[ImageRecord], candidates: list[tuple[str, str]]
) -> tuple[str | None, str | None, dict[int, str]]:
    for field_path, group_type in candidates:
        groups: dict[int, str] = {}
        counts: Counter[str] = Counter()
        complete = True
        for record in records:
            key = scalar_group_value(record.grouping_values.get(field_path), field_path)
            if key is None:
                complete = False
                break
            groups[record.row_index] = key
            counts[key] += 1
        if complete and len(counts) > 1 and max(counts.values(), default=0) > 1:
            return field_path, group_type, groups
    return None, None, {}


def split_randomly(
    records: list[ImageRecord], ratios: tuple[float, float, float], seed: int
) -> dict[int, str]:
    shuffled = [record.row_index for record in records]
    random.Random(seed).shuffle(shuffled)
    n_total = len(shuffled)
    n_train = int(round(ratios[0] * n_total))
    n_val = int(round(ratios[1] * n_total))
    if n_total >= 3:
        n_train = min(max(n_train, 1), n_total - 2)
        n_val = min(max(n_val, 1), n_total - n_train - 1)
    assignment: dict[int, str] = {}
    for row_id in shuffled[:n_train]:
        assignment[row_id] = "train"
    for row_id in shuffled[n_train : n_train + n_val]:
        assignment[row_id] = "val"
    for row_id in shuffled[n_train + n_val :]:
        assignment[row_id] = "test"
    return assignment


def split_by_group(
    records: list[ImageRecord],
    group_for_row: dict[int, str],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[int, str]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for record in records:
        grouped[group_for_row[record.row_index]].append(record.row_index)

    rng = random.Random(seed)
    groups = list(grouped.items())
    rng.shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)

    targets = [ratio * len(records) for ratio in ratios]
    counts = [0, 0, 0]
    names = ("train", "val", "test")
    assignment: dict[int, str] = {}
    for _, row_ids in groups:
        size = len(row_ids)

        def score(split_index: int) -> float:
            return sum(
                ((counts[index] + (size if index == split_index else 0) - targets[index]) ** 2)
                / max(targets[index], 1.0)
                for index in range(3)
            )

        target_split = min(range(3), key=score)
        counts[target_split] += size
        for row_id in row_ids:
            assignment[row_id] = names[target_split]
    return assignment


def image_payload(image_value: Any, parquet_dir: Path) -> bytes:
    if isinstance(image_value, Mapping):
        image_bytes = image_value.get("bytes")
        if image_bytes:
            return bytes(image_bytes)
        image_path = image_value.get("path")
    else:
        image_path = image_value
    if image_path:
        path = Path(str(image_path))
        if not path.is_absolute():
            path = parquet_dir / path
        if path.is_file():
            return path.read_bytes()
    raise ValueError("A Parquet row has neither embedded image bytes nor a readable image path.")


def safe_yaml_string(value: str) -> str:
    # JSON quoting is valid YAML and handles punctuation/unicode in species names.
    return json.dumps(value, ensure_ascii=False)


def write_data_yaml(yaml_path: Path, output_dir: Path, class_names: list[str]) -> None:
    lines = [
        f"path: {safe_yaml_string(str(output_dir.resolve()))}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines.extend(f"  {index}: {safe_yaml_string(name)}" for index, name in enumerate(class_names))
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.top_classes < 1:
        raise SystemExit("--top-classes must be at least 1.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    if len(args.ratios) != 3 or any(ratio <= 0 for ratio in args.ratios):
        raise SystemExit("Provide three positive split ratios for train, val, and test.")
    ratio_total = sum(args.ratios)
    if not math.isclose(ratio_total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise SystemExit(f"Split ratios must sum to 1.0; got {ratio_total:.6f}.")

    parquet_files = sorted(args.parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No Parquet files found in {args.parquet_dir}.")
        if args.output_dir.exists() and not args.output_dir.is_dir():
            raise SystemExit(f"Output path exists but is not a directory: {args.output_dir}.")
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(
            f"Output directory is not empty: {args.output_dir}. Choose a new --output-dir; "
            "this script does not delete existing files."
        )
    if args.yaml_path.exists():
        raise SystemExit(
            f"YAML already exists: {args.yaml_path}. Choose another --yaml-path; "
            "this script does not overwrite existing files."
        )

    pa, pq = arrow_modules()
    print(f"Reading {len(parquet_files)} Parquet shard(s) from {args.parquet_dir}...")
    records, counts, source_names, group_candidates = scan_shards(
        parquet_files, args.batch_size, pa, pq
    )
    if not records:
        raise SystemExit("No dataset rows were found.")
    if not counts:
        raise SystemExit("No valid bounding boxes were found in the objects column.")

    top_classes = [class_id for class_id, _ in counts.most_common(args.top_classes)]
    top_set = set(top_classes)
    kept_records: list[ImageRecord] = []
    kept_boxes: dict[int, list[tuple[int, float, float, float, float]]] = {}
    for record in records:
        boxes = [box for box in record.boxes if box[0] in top_set]
        if boxes:
            kept_records.append(record)
            kept_boxes[record.row_index] = boxes
    if not kept_records:
        raise SystemExit("No images contain boxes for the selected classes.")

    group_field, group_type, groups = choose_grouping(kept_records, group_candidates)
    if group_field:
        group_count = len(set(groups.values()))
        print(f"Splitting by {group_type} field '{group_field}' ({group_count} groups).")
        if group_count < 3:
            print(
                "WARNING: Fewer than three usable groups exist; keeping groups intact "
                "means one or more of train/val/test may be empty."
            )
        assignments = split_by_group(kept_records, groups, tuple(args.ratios), args.seed)
    else:
        print(
            "WARNING: No usable location, sequence, or capture-folder grouping metadata "
            "was found. Falling back to a seeded random image-level split; near-duplicate "
            "images from the same burst may leak across train/val/test."
        )
        assignments = split_randomly(kept_records, tuple(args.ratios), args.seed)

    split_names = ("train", "val", "test")
    for split in split_names:
        (args.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    selected_ids = {record.row_index for record in kept_records}
    split_counters: dict[str, Counter[int]] = {split: Counter() for split in split_names}
    split_image_class_counts: dict[str, Counter[int]] = {split: Counter() for split in split_names}
    split_image_counts: Counter[str] = Counter()
    record_by_row = {record.row_index: record for record in kept_records}
    row_index = 0

    print("Writing selected images and YOLO labels...")
    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=args.batch_size, columns=["image"]):
            for row in batch.to_pylist():
                if row_index in selected_ids:
                    record = record_by_row[row_index]
                    split = assignments[row_index]
                    image_bytes = image_payload(row.get("image"), args.parquet_dir)
                    try:
                        from PIL import Image

                        with Image.open(BytesIO(image_bytes)) as image:
                            image = image.convert("RGB")
                            image_width, image_height = image.size
                            if image_width <= 0 or image_height <= 0:
                                raise ValueError(f"Invalid image dimensions for row {row_index}.")
                            if record.width and record.height and (
                                image_width != record.width or image_height != record.height
                            ):
                                raise ValueError(
                                    f"Parquet dimensions {record.width}x{record.height} do not match "
                                    f"decoded dimensions {image_width}x{image_height}."
                                )
                            stem = f"image_{row_index:06d}"
                            image_path = args.output_dir / "images" / split / f"{stem}.jpg"
                            image.save(image_path, format="JPEG", quality=95)
                    except ImportError as exc:
                        raise SystemExit("Pillow is required to export images; install project requirements.") from exc
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to decode image_id={record.image_id!r} (row {row_index})."
                        ) from exc

                    label_lines: list[str] = []
                    image_classes: set[int] = set()
                    for source_class, center_x, center_y, box_width, box_height in kept_boxes[row_index]:
                        yolo_class = top_classes.index(source_class)
                        label_lines.append(
                            f"{yolo_class} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
                        )
                        split_counters[split][source_class] += 1
                        image_classes.add(source_class)
                    label_path = args.output_dir / "labels" / split / f"{stem}.txt"
                    label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
                    split_image_counts[split] += 1
                    for class_id in image_classes:
                        split_image_class_counts[split][class_id] += 1
                row_index += 1

    final_class_names = [source_names.get(class_id, f"class_{class_id}") for class_id in top_classes]
    write_data_yaml(args.yaml_path, args.output_dir, final_class_names)

    total_written_images = sum(split_image_counts.values())
    total_written_boxes = sum(sum(counter.values()) for counter in split_counters.values())
    dropped_empty_or_other = len(records) - total_written_images
    print("\nConversion summary")
    print(f"Source rows: {len(records)}")
    print(f"Images dropped (no boxes in kept classes): {dropped_empty_or_other}")
    print(f"Images written: {total_written_images}")
    print(f"Boxes written: {total_written_boxes}")
    print(f"Output: {args.output_dir.resolve()}")
    print(f"Dataset YAML: {args.yaml_path.resolve()}")
    print("\nPer-split totals")
    print("split  images  boxes")
    for split in split_names:
        box_total = sum(split_counters[split].values())
        print(f"{split:<6} {split_image_counts[split]:>6}  {box_total:>5}")

    print("\nPer-class totals in the written splits")
    print("YOLO ID  source ID  class name                     train boxes/images  val boxes/images  test boxes/images")
    for yolo_id, source_id in enumerate(top_classes):
        name = final_class_names[yolo_id]
        columns = []
        for split in split_names:
            boxes = split_counters[split][source_id]
            images = split_image_class_counts[split][source_id]
            columns.append(f"{boxes}/{images}")
        print(f"{yolo_id:>7}  {source_id:>9}  {name:<29} {columns[0]:>17} {columns[1]:>17} {columns[2]:>18}")

    unknown_names = [class_id for class_id in top_classes if class_id not in source_names]
    if unknown_names:
        print(
            "\nNOTE: ClassLabel names were not available in the Parquet metadata for source IDs "
            f"{unknown_names}; data.yaml uses class_<id> placeholders for those IDs."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; existing outputs were left in place.", file=sys.stderr)
        raise SystemExit(130)
