# FinBIF data analysis (formerly known as redlist

Python tools for generating biodiversity occurrence statistics from [FinBIF](https://laji.fi) citable exports — for red list assessment and other purposes.

## Structure

```
preprocess_occurrences.py   # DwC TSV → normalized Parquet aggregates
scripts/                    # statistics, visualizations, GeoPackage exports
data/{dataset-slug}/        # raw FinBIF export (occurrences.txt)
data/ykj-centerpoints.csv   # shared YKJ grid lookup
output/{dataset-slug}/      # preprocessed Parquet and script outputs
sample_data/                # example output formats (for reference)
```

`data/` and `output/` are gitignored. Each dataset is identified by a slug (typically the FinBIF download name).

## Usage

1. Download a citable occurrence export from FinBIF and unzip it to `data/{dataset-slug}/`.
2. Preprocess: `uv run preprocess_occurrences.py <dataset-slug>`
3. Run analysis scripts as needed: `uv run scripts/<name>.py <dataset-slug>`

Preprocessing writes normalized Parquet files and small JSON samples to `output/{dataset-slug}/`. Scripts read from there (or, in a few cases, directly from the raw TSV) and write results to the same output folder.

- `occurrences.parquet` — one row per filtered occurrence
- `aggregate_yearly_10km.parquet` — counts by species, year, YKJ 10 km cell
- `aggregate_daily_10km.parquet` — counts by species, event date, YKJ cell (short date spans only)
- `aggregate_dayofyear_10km.parquet` — counts by species, year, day-of-year, YKJ cell

- Python 3.11+, dependencies managed with uv (`pyproject.toml`).
- New analysis scripts: copy `scripts/_template.py`, use `scripts/dataset_io.py` for paths and I/O.
- See `AGENTS.md` for conventions when working with a coding agent.

