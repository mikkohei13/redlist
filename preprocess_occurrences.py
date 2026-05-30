"""
Load FinBIF occurrences.txt (tab-separated DwC), preprocess, write Parquet.
"""

import json
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
AGGREGATE_YEARLY_10KM = OUTPUT_DIR / "aggregate_yearly_10km.parquet"
AGGREGATE_DAILY_10KM = OUTPUT_DIR / "aggregate_daily_10km.parquet"
AGGREGATE_DAYOFYEAR_10KM = OUTPUT_DIR / "aggregate_dayofyear_10km.parquet"
AGGREGATED_PARQUET = AGGREGATE_YEARLY_10KM

# FinBIF occurrences export: English header row, then three translated label rows.
SKIP_ROWS_AFTER_HEADER = 3

# Small on-disk row samples (JSON array of objects only).
SAMPLE_ROW_COUNT = 5
# For occurrences.parquet only: draw the random sample from the first N rows
# so we do not read the whole file back into memory on large exports.
SAMPLE_PARQUET_MAX_ROWS = 50_000


def write_sample_rows_json(path: Path, df: pl.DataFrame) -> None:
    """Write a JSON array of row objects (only sample rows, no metadata)."""
    sample_n = min(SAMPLE_ROW_COUNT, df.height)
    rows = df.head(sample_n).to_dicts() if sample_n else []
    path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")


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
        _begin_date.dt.ordinal_day().cast(pl.Int32).alias("dayOfYear"),
        (_end_date - _begin_date).dt.total_days().cast(pl.Int32).alias("daysSpan"),
        _event_parts.list.first().str.strip_chars().alias("eventDateBegin"),
    ).drop("_year_prefix")

    lf.sink_parquet(
        PROCESSED_PARQUET,
        compression="zstd",
        statistics=True,
    )

    proc_for_sample = (
        pl.scan_parquet(PROCESSED_PARQUET)
        .head(SAMPLE_PARQUET_MAX_ROWS)
        .collect()
    )
    proc_sample = proc_for_sample.sample(
        min(SAMPLE_ROW_COUNT, proc_for_sample.height),
        seed=42,
        shuffle=True,
    )
    write_sample_rows_json(OUTPUT_DIR / "occurrences_sample.json", proc_sample)

    ykj = pl.read_csv(YKJ_CENTERPOINTS, separator=";").rename(
        {"lat": "latitude", "lon": "longitude"}
    )

    agg_yearly = (
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
    agg_yearly.write_parquet(
        AGGREGATE_YEARLY_10KM,
        compression="zstd",
        statistics=True,
    )

    agg_yearly_sample = agg_yearly.sample(
        min(SAMPLE_ROW_COUNT, agg_yearly.height),
        seed=42,
        shuffle=True,
    )
    write_sample_rows_json(
        AGGREGATE_YEARLY_10KM.with_name(
            AGGREGATE_YEARLY_10KM.name.replace(".parquet", "_sample.json")
        ),
        agg_yearly_sample,
    )

    agg_daily = (
        pl.scan_parquet(PROCESSED_PARQUET)
        .filter(pl.col("daysSpan") <= 2)
        .group_by("speciesName", "eventDateBegin", "gridCellYKJ")
        .agg(pl.len().alias("occurrenceCount"))
        .collect()
        .join(ykj, on="gridCellYKJ", how="left")
        .with_columns(
            pl.col("latitude").round(6),
            pl.col("longitude").round(6),
        )
    )
    agg_daily.write_parquet(
        AGGREGATE_DAILY_10KM,
        compression="zstd",
        statistics=True,
    )

    agg_daily_sample = agg_daily.sample(
        min(SAMPLE_ROW_COUNT, agg_daily.height),
        seed=42,
        shuffle=True,
    )
    write_sample_rows_json(
        AGGREGATE_DAILY_10KM.with_name(
            AGGREGATE_DAILY_10KM.name.replace(".parquet", "_sample.json")
        ),
        agg_daily_sample,
    )

    agg_dayofyear = (
        pl.scan_parquet(PROCESSED_PARQUET)
        .filter(pl.col("daysSpan") <= 2)
        .group_by("speciesName", "dayOfYear", "gridCellYKJ")
        .agg(pl.len().alias("occurrenceCount"))
        .collect()
        .join(ykj, on="gridCellYKJ", how="left")
        .with_columns(
            pl.col("latitude").round(6),
            pl.col("longitude").round(6),
        )
    )
    agg_dayofyear.write_parquet(
        AGGREGATE_DAYOFYEAR_10KM,
        compression="zstd",
        statistics=True,
    )

    agg_dayofyear_sample = agg_dayofyear.sample(
        min(SAMPLE_ROW_COUNT, agg_dayofyear.height),
        seed=42,
        shuffle=True,
    )
    write_sample_rows_json(
        AGGREGATE_DAYOFYEAR_10KM.with_name(
            AGGREGATE_DAYOFYEAR_10KM.name.replace(".parquet", "_sample.json")
        ),
        agg_dayofyear_sample,
    )


if __name__ == "__main__":
    main()
