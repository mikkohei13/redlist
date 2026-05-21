# redlist
Tools for red list assessment

## Pipeline (technical)

**Runtime:** Python 3.11+, dependencies and venv via [uv](https://docs.astral.sh/uv/) (`uv sync`, `uv run`).

**Ingest:** FinBIF Darwin Core `occurrences.txt` is tab-delimited TSV with one English header row and three localized descriptor rows; preprocessing skips those rows after the header, then scans the file with **Polars** `scan_csv` (lazy, streaming-friendly) and materializes **Apache Parquet** (Zstd compression, column statistics) under `output/{dataset-slug}/`.

The same step also writes `occurrences_aggregated.parquet`: row counts grouped by `speciesName`, `year`, and `gridCellYKJ` (derived from `occurrences.parquet`), and prints a tiny random sample to stdout.

**Intermediate:** Downstream jobs use `scan_parquet` / `read_parquet` on the same Parquet artifact—columnar storage avoids re-parsing TSV and keeps memory bounded for large row counts.

**Egress:** Each analysis script is standalone; paths and dataset slug live in `config.py`. Outputs (CSVs, figures, etc.) are written beside the Parquet under `output/{dataset-slug}/`.
