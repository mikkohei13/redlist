"""
List species counts by host and guest taxa from occurrences.parquet.
"""

from __future__ import annotations

import polars as pl

from dataset_io import dataset_from_argv, output_path, require_file

USAGE = "uv run scripts/host_taxa_species_counts.py <dataset-slug>"


def main() -> None:
    ds = dataset_from_argv(usage=USAGE)
    occurrences = require_file(ds.path("occurrences"), label="parquet file")

    schema = pl.scan_parquet(occurrences).collect_schema()
    for col in ("host_taxon_normalized", "taxonConceptID", "scientificName", "vernacularName"):
        if col not in schema:
            print(f"error: occurrences.parquet missing column {col!r}")
            raise SystemExit(1)

    grouped = (
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

    host_lines = ["# Host taxa", ""]
    for host in grouped["host_taxon_normalized"].unique(maintain_order=True):
        host_lines.append(f"## {host}")
        host_lines.append("")
        host_rows = grouped.filter(pl.col("host_taxon_normalized") == host)
        for row in host_rows.iter_rows(named=True):
            scientific = row["scientificName"] or ""
            vernacular = row["vernacularName"] or ""
            host_lines.append(f"- {scientific} - {vernacular}: {row['count']}")
        host_lines.append("")

    guest_grouped = grouped.sort("scientificName", "count", descending=[False, True])
    guest_lines = ["# Guest taxa", ""]
    for taxon_id in guest_grouped["taxonConceptID"].unique(maintain_order=True):
        guest_rows = guest_grouped.filter(pl.col("taxonConceptID") == taxon_id)
        scientific = guest_rows["scientificName"][0] or ""
        vernacular = guest_rows["vernacularName"][0] or ""
        guest_lines.append(f"## {scientific} - {vernacular}")
        guest_lines.append("")
        for row in guest_rows.iter_rows(named=True):
            guest_lines.append(f"- {row['host_taxon_normalized']}: {row['count']}")
        guest_lines.append("")

    host_out = output_path(ds, "host_taxa_species_counts.md")
    host_out.write_text("\n".join(host_lines), encoding="utf-8")
    guest_out = output_path(ds, "guest_taxa_host_counts.md")
    guest_out.write_text("\n".join(guest_lines), encoding="utf-8")

    host_count = grouped["host_taxon_normalized"].n_unique()
    guest_count = grouped["taxonConceptID"].n_unique()
    print(f"Wrote {host_out} ({host_count} host taxa, {grouped.height} species rows)")
    print(f"Wrote {guest_out} ({guest_count} guest taxa, {grouped.height} host rows)")


if __name__ == "__main__":
    main()
