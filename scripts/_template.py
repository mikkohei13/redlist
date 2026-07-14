"""
One-line description of what this script does.

Copy this file to scripts/<name>.py and fill in domain logic.
"""

from __future__ import annotations

import polars as pl

from dataset_io import Dataset, dataset_from_argv, output_path, require_file, write_csv

USAGE = "uv run scripts/_template.py <dataset-slug>"

# Inputs: list canonical names used by this script (for quick reference).
#   occurrences, aggregate_yearly_10km, aggregate_daily_10km,
#   aggregate_daily_1km, aggregate_dayofyear_10km, raw_occurrences
#
# Output: default basename matches script stem (stats_foo.py → stats_foo.csv).
# Override when the output name is domain-specific (e.g. species_count_1km.gpkg).
# Subfolders: output_path(ds, "trend/chart.png")
#
# Loading: scan_parquet for filter/aggregate on large inputs;
# read_parquet when you need full materialization or per-group iteration.


def main() -> None:
    ds = dataset_from_argv(usage=USAGE)

    # --- inputs ---
    occurrences = require_file(ds.path("occurrences"), label="parquet file")

    # --- domain logic ---
    result = (
        pl.scan_parquet(occurrences)
        .group_by("taxonRank")
        .len()
        .sort("len", descending=True)
        .collect()
    )

    # --- outputs ---
    out = output_path(ds, "_template.csv")
    write_csv(result, out)
    print(f"Wrote {out} ({result.height} rows)")


if __name__ == "__main__":
    main()
