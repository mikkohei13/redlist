"""
Load FinBIF occurrences.txt (tab-separated DwC), preprocess, write Parquet.
"""

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent

# Only value you need to change when switching datasets (FinBIF export folder under data/).
dataset_id = "HBF.122105" # Heteroptera Suomi 10 km 1975-
dataset_id = "HBF.122199" #Pentatomidae Suomi 10 km

RAW_OCCURRENCES = ROOT / "data" / dataset_id / "occurrences.txt"
YKJ_CENTERPOINTS = ROOT / "data" / "ykj-centerpoints.csv"
OUTPUT_DIR = ROOT / "output" / dataset_id
PROCESSED_PARQUET = OUTPUT_DIR / "occurrences.parquet"
AGGREGATED_PARQUET = OUTPUT_DIR / "occurrences_aggregated.parquet"

# FinBIF occurrences export: English header row, then three translated label rows.
SKIP_ROWS_AFTER_HEADER = 3


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # FinBIF text fields sometimes contain raw " characters (e.g. "Name" W in
    # eventRemarks). Standard CSV quoting rules then fail; tabs still delimit.
    lf = pl.scan_csv(
        RAW_OCCURRENCES,
        separator="\t",
        quote_char=None,
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
    # DwC eventDate: single YYYY-MM-DD or ISO range "begin/end". Datetimes: use date part.
    _event_date_utf8 = pl.col("eventDate").cast(pl.Utf8, strict=False)
    _event_parts = _event_date_utf8.str.split("/")
    _begin_date_str = (
        _event_parts.list.first().str.strip_chars().str.slice(0, 10)
    )
    _end_date_str = (
        pl.when(_event_parts.list.len() > 1)
        .then(_event_parts.list.get(1).str.strip_chars().str.slice(0, 10))
        .otherwise(_begin_date_str)
    )
    _begin_date = _begin_date_str.str.strptime(pl.Date, "%Y-%m-%d", strict=False)
    _end_date = _end_date_str.str.strptime(pl.Date, "%Y-%m-%d", strict=False)

    # Species name = part before the second space (binomial; trims trinomial+).
    lf = lf.with_columns(
        pl.col("scientificName")
        .str.split(" ")
        .list.slice(0, 2)
        .list.join(" ")
        .alias("speciesName"),
        pl.col("_year_prefix").cast(pl.Int32).alias("year"),
        _begin_date.dt.ordinal_day().cast(pl.Int32).alias("day_of_year"),
        (_end_date - _begin_date).dt.total_days().cast(pl.Int32).alias("days_span"),
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
