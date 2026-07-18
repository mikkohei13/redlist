"""
Score host plant families by weighted species rarity (log-compressed, proportional).
"""

from __future__ import annotations

import polars as pl

from dataset_io import dataset_from_argv, output_path, require_file, write_csv

USAGE = "uv run scripts/stats_host_family_rarity.py <dataset-slug>"


def main() -> None:
    ds = dataset_from_argv(usage=USAGE)

    yearly = require_file(ds.path("aggregate_yearly_10km"), label="parquet file")
    occurrences = require_file(ds.path("occurrences"), label="parquet file")

    rarity = (
        pl.scan_parquet(yearly)
        .group_by("speciesName")
        .agg(pl.col("occurrenceCount").sum().alias("total_count"))
        .with_columns(
            (1 / (pl.col("total_count") + 1).log()).alias("rarity_score")
        )
        .collect()
    )
    global_avg = rarity["rarity_score"].mean()

    host_props = (
        pl.scan_parquet(occurrences)
        .filter(pl.col("host_family").is_not_null())
        .group_by("speciesName", "host_family")
        .len()
        .with_columns(
            (pl.col("len") / pl.col("len").sum().over("speciesName")).alias(
                "proportion"
            )
        )
    )

    joined = (
        host_props.join(rarity.lazy(), on="speciesName")
        .with_columns(
            (pl.col("rarity_score") * pl.col("proportion")).alias("contribution")
        )
        .collect()
    )

    result = (
        joined.group_by("host_family")
        .agg(
            (pl.col("contribution").sum() / pl.col("proportion").sum()).alias(
                "raw_weighted_average"
            ),
            pl.col("speciesName").n_unique().alias("n"),
            pl.col("proportion").sum().alias("proportion_sum"),
        )
    )
    k = result["n"].median()
    result = result.with_columns(
        (
            (pl.col("n") * pl.col("raw_weighted_average") + k * global_avg)
            / (pl.col("n") + k)
        ).alias("adjusted_score")
    ).sort("adjusted_score", descending=True)

    out = output_path(ds, "stats_host_family_rarity.csv")
    write_csv(result, out)
    print(f"global_avg={global_avg:.6f}  k={k}")
    print(f"Wrote {out} ({result.height} rows)")

    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=80):
        for row in result.iter_rows(named=True):
            family = row["host_family"]
            print(
                f"\n=== {family}  "
                f"adjusted={row['adjusted_score']:.6f}  "
                f"raw={row['raw_weighted_average']:.6f}  "
                f"n={row['n']}  "
                f"proportion_sum={row['proportion_sum']:.4f} ==="
            )
            species = (
                joined.filter(pl.col("host_family") == family)
                .select(
                    "speciesName",
                    "total_count",
                    "rarity_score",
                    pl.col("len").alias("host_records"),
                    "proportion",
                    "contribution",
                )
                .sort("contribution", descending=True)
            )
            print(species)


if __name__ == "__main__":
    main()
