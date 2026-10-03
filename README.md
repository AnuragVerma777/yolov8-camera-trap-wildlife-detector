# Camera-Trap Wildlife Detector (YOLOv8)

## Project status

This repository contains a reproducible pipeline for preparing the ENA24 camera-trap detection dataset, training a YOLOv8n baseline, evaluating it by lighting condition, and reviewing detection errors. **The training/evaluation pipeline has not produced model outputs in this checkout yet.** There is no generated YOLO dataset YAML, baseline checkpoint, day/night evaluation, or error report. The comparison artifacts therefore contain `NA` and are placeholders, not measured results.

## Problem statement

Build a compact wildlife object detector for camera-trap images and assess its performance on held-out test images, including a day/night breakdown. The intended use is a research-internship project demonstration. This is an exploratory detector, not a validated wildlife monitoring system.

## Dataset and split

The repository's `datset/README.md` describes the source as ENA24 Detection from [Hugging Face](https://huggingface.co/datasets/davanstrien/ena24-detection), with COCO-style bounding boxes. The downloaded source files are under `datset/data/`. The conversion script is configured to retain the **5 most frequent categories by box count**, drop images without retained boxes, convert boxes to normalized YOLO coordinates, and target an approximate **70/15/15 train/validation/test** split.

The converter prefers to group by camera location, then sequence metadata, then capture folder. It warns if it must split randomly without grouping data. **The converter has not been run in this checkout, so the realized grouping method, class names, image/box totals, and per-split totals are not yet available.** The intended converted data path is `data/processed/ena24_yolo/`; the intended YAML is `configs/data.yaml`.

## Model and training settings

The baseline script initializes Ultralytics YOLOv8n from the COCO-pretrained `yolov8n.pt` checkpoint and specifies:

| Setting | Value |
|---|---|
| Epochs | 30 |
| Image size | 640 |
| Seed | 42 |
| Deterministic mode | Enabled |
| Device | CUDA device 0 |
| Run name | `baseline` |

If a run completes, the script expects `runs/detect/baseline/weights/best.pt`, `runs/detect/baseline/results.csv`, and `runs/detect/baseline/results.png`. Those files are not present yet. The training seed is a single seed; results from one run do not measure run-to-run variability.

## Results

The repository's `outputs/final_model_comparison.csv` and `.md` currently report all values as unavailable. No baseline or improved model evaluation has been completed, so no performance conclusion can be drawn.

| Test set | Metric | Baseline | Improved |
|---|---|---:|---:|
| All | mAP@0.5 | NA | NA |
| All | mAP@0.5:0.95 | NA | NA |
| All | Precision | NA | NA |
| All | Recall | NA | NA |
| Day | mAP@0.5 | NA | NA |
| Day | mAP@0.5:0.95 | NA | NA |
| Day | Precision | NA | NA |
| Day | Recall | NA | NA |
| Night | mAP@0.5 | NA | NA |
| Night | mAP@0.5:0.95 | NA | NA |
| Night | Precision | NA | NA |
| Night | Recall | NA | NA |

The generated comparison chart is `outputs/final_model_comparison.svg`; it is currently a clearly labeled placeholder because neither model has evaluation metrics.

## Failure analysis

The error-review script reports false negatives and false positives, with counts by day/night condition, box-size category, and class. It can save up to 50 annotated images for each error type and a CSV under `results/error_review/`.

**Failure-category counts not supplied:** the request included `[paste]` instead of counts, and no error-analysis CSV or report exists in the repository. No claim is made about whether errors concentrate at night, on small objects, or in particular classes. Review the saved examples before attributing individual errors to a cause.

## Limitations

- **One seed:** the baseline run is configured with seed 42. A single run cannot show whether a performance difference is stable across random seeds.
- **Reduced class subset:** conversion keeps the five most frequent classes by box count and removes images without boxes in those retained classes. Final dataset size and class balance are not known until conversion runs.
- **Pretrained initialization:** YOLOv8n starts from COCO-pretrained weights, so this is transfer learning rather than training from scratch.
- **Operational day/night definition:** the tagging script classifies grayscale/near-gray imagery as night first. Otherwise, mean brightness below 80 on a 0–255 scale is night; brighter images are day. The near-gray pixel fraction threshold is 0.95 with a channel-difference threshold of 10. These are heuristic labels, not verified capture-time metadata, and tagging covers the test set.
- **Unreported empirical results:** no converted split, training metrics, evaluation metrics, or error-category counts are available in this checkout yet.

## Exact reproduction steps

Run these commands from the repository root in a Colab session with a GPU runtime enabled. For a fresh Colab session, first clone or upload this repository and ensure `datset/data/` contains the source Parquet shards.

```bash
pip install -r requirements.txt
python src/convert_ena24.py
python src/validate_yolo.py
python src/train_baseline.py --data configs/data.yaml
python src/tag_test_day_night.py
python src/evaluate_baseline.py
python src/review_test_errors.py
```

The converter refuses to overwrite existing generated data/YAML outputs. The day/night tagging script also preserves existing outputs unless `--overwrite` is supplied; use that flag if intentionally regenerating its outputs. Training exits if the YAML or CUDA GPU is missing and refuses an existing `baseline` run directory. The expected training artifact is `runs/detect/baseline/weights/best.pt`.

The evaluation script expects the baseline checkpoint plus `configs/data.yaml`, `configs/day_test.yaml`, and `configs/night_test.yaml`; it writes `outputs/baseline_test_metrics.csv` and `outputs/baseline_test_metrics.png`. Error review defaults to `results/test_day_night.csv` for condition labels and writes its review CSV and annotated images below `results/error_review/`.

To reproduce the day/night labeling exactly, the script defaults to the grayscale/IR-first rule above, brightness threshold 80, and random contact-sheet sampling seed 42. Inspect `results/day_samples.png`, `results/night_samples.png`, and `results/test_day_night_brightness.png` before interpreting condition-specific scores.

## Repository map

```text
configs/                  Dataset YAMLs (generated by conversion/tagging)
datset/data/              Downloaded ENA24 Parquet source shards
data/processed/            Converted YOLO images and labels (generated)
outputs/                   Model comparison/evaluation artifacts
results/                   Validation, day/night, and error-review artifacts
runs/detect/baseline/      Ultralytics training run (generated)
src/convert_ena24.py       Source-to-YOLO conversion and grouped split
src/validate_yolo.py       Label/image/split validation
src/train_baseline.py      YOLOv8n baseline training
src/tag_test_day_night.py  Test-set lighting heuristic and subsets
src/evaluate_baseline.py   Full/day/night model evaluation
src/review_test_errors.py  False-negative/false-positive review

