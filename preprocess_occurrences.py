"""
Load FinBIF occurrences.txt (tab-separated DwC), preprocess, write Parquet.

Also writes occurrences_aggregated.parquet (counts by speciesName, year,
gridCellYKJ, plus latitude/longitude from ykj-centerpoints.csv) from
occurrences.parquet and prints a small random sample.

Run from repo root: uv run python preprocess_occurrences.py
"""

import polars as pl

from config import (
    AGGREGATED_PARQUET,
    OUTPUT_DIR,
    PROCESSED_PARQUET,
    RAW_OCCURRENCES,
    YKJ_CENTERPOINTS,
)

# FinBIF occurrences export: English header row, then three translated label rows.
SKIP_ROWS_AFTER_HEADER = 3


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lf = pl.scan_csv(
        RAW_OCCURRENCES,
        separator="\t",
        skip_rows_after_header=SKIP_ROWS_AFTER_HEADER,
        infer_schema_length=10_000,
        try_parse_dates=True,
    )

    # Light normalization useful for downstream stats (extend as needed).
    lf = lf.with_columns(
        pl.col("decimalLatitude").cast(pl.Float64, strict=False),
        pl.col("decimalLongitude").cast(pl.Float64, strict=False),
        pl.col("eventDate")
        .cast(pl.Utf8, strict=False)
        .str.slice(0, 4)
        .alias("_year_prefix"),
    )

    # Binomial-style names only: drop bare genus (no space) or null.
    lf = lf.filter(
        pl.col("scientificName").is_not_null()
        & pl.col("scientificName").str.contains(" ")
        & (pl.col("_year_prefix").str.len_chars() == 4)
        & pl.col("_year_prefix").cast(pl.Int32, strict=False).is_not_null()
    )
    # Species name = part before the second space (binomial; trims trinomial+).
    lf = lf.with_columns(
        pl.col("scientificName")
        .str.split(" ")
        .list.slice(0, 2)
        .list.join(" ")
        .alias("speciesName"),
        pl.col("_year_prefix").cast(pl.Int32).alias("year"),
    ).drop("_year_prefix")

    lf.sink_parquet(
        PROCESSED_PARQUET,
        compression="zstd",
        statistics=True,
    )

    ykj = pl.read_csv(YKJ_CENTERPOINTS, separator=";").rename(
        {"lat": "latitude", "lon": "longitude"}
    )

    agg = (
        pl.scan_parquet(PROCESSED_PARQUET)
        .group_by("speciesName", "year", "gridCellYKJ")
        .agg(pl.len().alias("occurrenceCount"))
        .collect()
        .join(ykj, on="gridCellYKJ", how="left")
        .with_columns(
            pl.col("latitude").round(6),
            pl.col("longitude").round(6),
        )
    )
    agg.write_parquet(
        AGGREGATED_PARQUET,
        compression="zstd",
        statistics=True,
    )

    n_sample = min(5, agg.height)
    print(f"\nRandom sample of aggregated data ({n_sample} rows):")
    print(agg.sample(n_sample, seed=42))


if __name__ == "__main__":
    main()
