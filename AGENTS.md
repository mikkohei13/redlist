
## Conventions

- Keep code simple. This is a small one-person project.
- Do not add features the user did not ask for.
- No CLI args beyond `<dataset-slug>` unless explicitly requested.
- No extensive error handling or fallback logic. Missing input → print an error and exit.
- Do not update `README.md` unless asked.
- Match existing style: Polars for tabular work, minimal abstractions, no one-off helpers.

## Architecture

All code runs with `uv`.

- **`preprocess_occurrences.py`** (repo root) — FinBIF DwC TSV → Parquet aggregates in `output/{slug}/`. Not in `scripts/`.
- **`scripts/`** — analysis only. One argument: `<dataset-slug>`. Use `scripts/dataset_io.py` for all paths and I/O.
- **`data/`**, **`output/`** — gitignored local data; never commit.
- **`sample_data/`** — sample data for agents: see this when thinking how scripts should read input data.

| Location | Purpose |
|---|---|
| `data/{slug}/occurrences.txt` | raw FinBIF export |
| `data/ykj-centerpoints.csv` | shared YKJ grid lookup |
| `output/{slug}/` | preprocessed Parquet + script outputs |

Inputs via `ds.path(...)`: `occurrences`, `aggregate_yearly_10km`, `aggregate_daily_10km`, `aggregate_daily_1km`, `aggregate_dayofyear_10km`, `raw_occurrences`, `ykj_centerpoints`.

Output basename defaults to script stem (`stats_foo.py` → `stats_foo.csv`); override when domain-specific.

## New script

Script files are in `scripts/`. They should be independent from each other.

1. Copy `scripts/_template.py` → `scripts/<name>.py`.
2. `dataset_from_argv` + `require_file(ds.path(...))` for inputs at top of `main()`.
3. Domain logic, then `output_path` and writers from `dataset_io`.
