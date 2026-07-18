"""Markdown report writers for host-guest statistics."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dataset_io import Dataset, output_path


def write_host_guest_markdown(pairs: pl.DataFrame, ds: Dataset) -> tuple[Path, Path]:
    host_lines = ["# Host taxa", ""]
    for host in pairs["host_taxon_normalized"].unique(maintain_order=True):
        host_lines.append(f"## {host}")
        host_lines.append("")
        host_rows = pairs.filter(pl.col("host_taxon_normalized") == host)
        for row in host_rows.iter_rows(named=True):
            scientific = row["scientificName"] or ""
            vernacular = row["vernacularName"] or ""
            host_lines.append(f"- {scientific} - {vernacular}: {row['count']}")
        host_lines.append("")

    guest_grouped = pairs.sort("scientificName", "count", descending=[False, True])
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
    return host_out, guest_out
