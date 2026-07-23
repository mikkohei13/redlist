"""
Count rows per speciesName in aggregate_yearly_10km.
"""

import polars as pl

from dataset_io import dataset_from_argv, output_path, require_file, write_csv

USAGE = "uv run scripts/stats_species_name_counts.py <dataset-slug>"


def main() -> None:
    ds = dataset_from_argv(usage=USAGE)
    yearly = require_file(ds.path("aggregate_yearly_10km"), label="parquet file")

    counts = (
        pl.scan_parquet(yearly)
        .group_by("speciesName")
        .len()
        .sort("len", descending=True)
        .collect()
    )

    out = output_path(ds, "stats_species_name_counts.csv")
    write_csv(counts, out)
    print(f"Wrote {out} ({counts.height} rows)")


if __name__ == "__main__":
    main()
