"""
Count occurrences by taxonRank.
"""

import polars as pl

from dataset_io import dataset_from_argv, output_path, require_file, write_csv

USAGE = "uv run scripts/stats_taxon_ranks.py <dataset-slug>"


def main() -> None:
    ds = dataset_from_argv(usage=USAGE)
    occurrences = require_file(ds.path("occurrences"), label="parquet file")

    counts = (
        pl.scan_parquet(occurrences)
        .group_by("taxonRank")
        .len()
        .sort("len", descending=True)
        .collect()
    )

    out = output_path(ds, "stats_taxon_ranks.csv")
    write_csv(counts, out)
    print(f"Wrote {out} ({counts.height} rows)")


if __name__ == "__main__":
    main()
