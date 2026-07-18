"""
List species counts by host and guest taxa from occurrences.parquet.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dataset_io import dataset_from_argv, require_file
from host_guest.reports_md import write_host_guest_markdown

USAGE = "uv run scripts/host_taxa_species_counts.py <dataset-slug>"


def compute_pairs(occurrences: Path) -> pl.DataFrame:
    schema = pl.scan_parquet(occurrences).collect_schema()
    for col in ("host_taxon_normalized", "taxonConceptID", "scientificName", "vernacularName"):
        if col not in schema:
            print(f"error: occurrences.parquet missing column {col!r}")
            raise SystemExit(1)

    return (
        pl.scan_parquet(occurrences)
        .filter(pl.col("host_taxon_normalized").is_not_null())
        .group_by("host_taxon_normalized", "taxonConceptID")
        .agg(
            pl.len().alias("count"),
            pl.col("scientificName").first(),
            pl.col("vernacularName").first(),
        )
        .sort("host_taxon_normalized", "count", descending=[False, True])
        .collect()
    )


def main() -> None:
    ds = dataset_from_argv(usage=USAGE)
    occurrences = require_file(ds.path("occurrences"), label="parquet file")
    pairs = compute_pairs(occurrences)
    host_out, guest_out = write_host_guest_markdown(pairs, ds)

    host_count = pairs["host_taxon_normalized"].n_unique()
    guest_count = pairs["taxonConceptID"].n_unique()
    print(f"Wrote {host_out} ({host_count} host taxa, {pairs.height} species rows)")
    print(f"Wrote {guest_out} ({guest_count} guest taxa, {pairs.height} host rows)")


if __name__ == "__main__":
    main()
