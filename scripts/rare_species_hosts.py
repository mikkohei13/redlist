"""
Rank host plant families by value for finding rare insect species.
"""

from __future__ import annotations

import polars as pl

from dataset_io import dataset_from_argv, output_path, require_file, write_csv

USAGE = "uv run scripts/rare_species_hosts.py <dataset-slug>"

TOP_N = 20
EXCLUDE_THRESHOLD = 500


def main() -> None:
    ds = dataset_from_argv(usage=USAGE)

    # --- inputs ---
    aggregate = require_file(ds.path("aggregate_yearly_10km"), label="parquet file")
    occurrences = require_file(ds.path("occurrences"), label="parquet file")

    # --- domain logic ---
    species_counts = (
        pl.scan_parquet(aggregate)
        .group_by("speciesName")
        .len()
        .rename({"len": "total_count"})
        .collect()
    )
    n_excluded = species_counts.filter(pl.col("total_count") > EXCLUDE_THRESHOLD).height
    weights = species_counts.filter(pl.col("total_count") <= EXCLUDE_THRESHOLD).with_columns(
        ((EXCLUDE_THRESHOLD - pl.col("total_count") + 1) / EXCLUDE_THRESHOLD).alias("weight")
    )

    # Distribute each guest species' weight across host families by record proportion.
    species_family_scores = (
        pl.scan_parquet(occurrences)
        .filter(pl.col("host_family").is_not_null())
        .join(weights.lazy().select("speciesName", "weight"), on="speciesName", how="inner")
        .group_by("speciesName", "host_family")
        .agg(
            pl.len().alias("n_records"),
            pl.col("weight").first().alias("weight"),
        )
        .with_columns(
            (pl.col("n_records") / pl.col("n_records").sum().over("speciesName")).alias(
                "proportion"
            )
        )
        .with_columns((pl.col("weight") * pl.col("proportion")).alias("score"))
        .collect()
    )

    result = (
        species_family_scores.group_by("host_family")
        .agg(pl.col("score").sum().round(2).alias("family_score"))
        .sort("family_score", descending=True)
    )

    # --- diagnostics ---
    n_species = weights.height
    print(
        f"Species weights ({n_species} species with total_count ≤ {EXCLUDE_THRESHOLD}; "
        f"{n_excluded} excluded as too common):"
    )
    print(
        f"  total_count  min={weights['total_count'].min()}  "
        f"median={weights['total_count'].median():.0f}  "
        f"max={weights['total_count'].max()}"
    )
    print(
        f"  weight       min={weights['weight'].min():.4f}  "
        f"median={weights['weight'].median():.4f}  "
        f"max={weights['weight'].max():.4f}"
    )
    print(
        f"  (weight = ({EXCLUDE_THRESHOLD} - total_count + 1) / {EXCLUDE_THRESHOLD}; "
        f"distributed by host-family record proportion)"
    )
    print()
    print(f"  Rarest species (highest weight):")
    for row in weights.sort("weight", descending=True).head(TOP_N).iter_rows(named=True):
        print(
            f"    {row['speciesName']}: total_count={row['total_count']}  "
            f"weight={row['weight']:.4f}"
        )
    print(f"  Commonest included species (lowest weight):")
    for row in weights.sort("weight").head(TOP_N).iter_rows(named=True):
        print(
            f"    {row['speciesName']}: total_count={row['total_count']}  "
            f"weight={row['weight']:.4f}"
        )

    n_host_records = species_family_scores["n_records"].sum()
    n_guest_species = species_family_scores["speciesName"].n_unique()
    n_host_families = result.height
    print()
    print(
        f"Host-linked occurrences: {n_host_records} records, "
        f"{n_guest_species} guest species, {n_host_families} host families"
    )
    print(f"Total family_score sum: {result['family_score'].sum():.2f}")

    guest_contrib = (
        species_family_scores.group_by("speciesName")
        .agg(
            pl.col("n_records").sum().alias("n_records"),
            pl.col("weight").first().alias("weight"),
            pl.col("score").sum().alias("total_score"),
            pl.col("host_family").n_unique().alias("n_host_families"),
        )
        .sort("total_score", descending=True)
    )
    print()
    print(f"Top {TOP_N} guest species by total score contribution:")
    for row in guest_contrib.head(TOP_N).iter_rows(named=True):
        print(
            f"  {row['speciesName']}: score={row['total_score']:.2f}  "
            f"weight={row['weight']:.4f}  "
            f"records={row['n_records']}  "
            f"host_families={row['n_host_families']}"
        )

    print()
    print(f"Top {TOP_N} host families and their top guest contributors:")
    for fam_row in result.head(TOP_N).iter_rows(named=True):
        family = fam_row["host_family"]
        print(f"  {family}: family_score={fam_row['family_score']:.2f}")
        top_guests = (
            species_family_scores.filter(pl.col("host_family") == family)
            .sort("score", descending=True)
            .head(10)
        )
        for g in top_guests.iter_rows(named=True):
            print(
                f"    {g['speciesName']}: score={g['score']:.2f}  "
                f"weight={g['weight']:.4f}  "
                f"proportion={g['proportion']:.0%}  "
                f"records={g['n_records']}"
            )

    # --- outputs ---
    out = output_path(ds, "rare_species_hosts.csv")
    write_csv(result, out)
    print()
    print(f"Wrote {out} ({result.height} rows)")


if __name__ == "__main__":
    main()
