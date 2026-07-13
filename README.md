# redlist

Tools for generating biodiversity occurrence statistics for red list assessment and other purposes.

## Usage

Requires uv and python 3.11+.

1. Download citable data from FinBIF
2. Unzip the data to `data/`
3. Run `uv run preprocess_occurrences.py <dataset-slug>` to preprocess the data
4. Run a selected script from the `scripts/` directory to generate the statistics

## Pipeline (technical)

Python 3.11+, dependencies via uv.

**Layout.** Raw FinBIF citable exports live under `data/{dataset-slug}/` (must contain `occurrences.txt`). Shared grid lookup: `data/ykj-centerpoints.csv`. All derived files go to `output/{dataset-slug}/`.

**Preprocessing** (`preprocess_occurrences.py`). Scans the DwC TSV with Polars (`scan_csv`, lazy/streaming), normalizes fields, and writes Parquet plus a small JSON sample per file:

- `occurrences.parquet` — one row per filtered occurrence
- `aggregate_yearly_10km.parquet` — counts by species, year, YKJ 10 km cell
- `aggregate_daily_10km.parquet` — counts by species, event date, YKJ cell (short date spans only)
- `aggregate_dayofyear_10km.parquet` — counts by species, year, day-of-year, YKJ cell

**Statistics** (`scripts/`). Each script takes `<dataset-slug>` as its only argument, reads one preprocessed Parquet (or, for grid taxon counts, the raw `occurrences.txt`), and writes results to the same `output/{dataset-slug}/` folder. Missing input files cause an immediate error and exit.
