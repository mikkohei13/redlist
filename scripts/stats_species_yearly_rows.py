"""
Count aggregate_yearly_10km rows per species, sorted descending.
"""

import polars as pl

from dataset_io import dataset_from_argv, output_path, require_file, write_csv

USAGE = "uv run scripts/stats_species_yearly_rows.py <dataset-slug>"


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

    out = output_path(ds, "stats_species_yearly_rows.csv")
    write_csv(counts, out)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=80):
        print(counts)
    print(f"Wrote {out} ({counts.height} rows)")


if __name__ == "__main__":
    main()
