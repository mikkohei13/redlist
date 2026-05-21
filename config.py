"""Shared paths and dataset identifiers. Edit here when switching inputs."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Folder name under ./data/ (FinBIF export / DwC archive folder).
DATASET_SLUG = "HBF.122199"

RAW_OCCURRENCES = ROOT / "data" / DATASET_SLUG / "occurrences.txt"
YKJ_CENTERPOINTS = ROOT / "data" / "ykj-centerpoints.csv"

OUTPUT_DIR = ROOT / "output" / DATASET_SLUG

# Written by preprocess_occurrences.py; read by stats and viz scripts.
PROCESSED_PARQUET = OUTPUT_DIR / "occurrences.parquet"
AGGREGATED_PARQUET = OUTPUT_DIR / "occurrences_aggregated.parquet"
