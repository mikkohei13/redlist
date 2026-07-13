"""
Load FinBIF occurrences.txt (tab-separated DwC), preprocess, write Parquet.
"""

import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent
YKJ_CENTERPOINTS = ROOT / "data" / "ykj-centerpoints.csv"

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
    if len(sys.argv) != 2:
        print("Usage: uv run preprocess_occurrences.py <dataset-slug>")
        sys.exit(1)

    dataset_id = sys.argv[1]
    raw_occurrences = ROOT / "data" / dataset_id / "occurrences.txt"
    output_dir = ROOT / "output" / dataset_id
    processed_parquet = output_dir / "occurrences.parquet"
    aggregate_yearly_10km = output_dir / "aggregate_yearly_10km.parquet"
    aggregate_daily_10km = output_dir / "aggregate_daily_10km.parquet"
    aggregate_dayofyear_10km = output_dir / "aggregate_dayofyear_10km.parquet"

    output_dir.mkdir(parents=True, exist_ok=True)

    # FinBIF text fields sometimes contain raw " characters (e.g. "Name" W in
    # eventRemarks). Standard CSV quoting rules then fail; tabs still delimit.
    lf = pl.scan_csv(
        raw_occurrences,
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
        processed_parquet,
        compression="zstd",
        statistics=True,
    )

    proc_for_sample = (
        pl.scan_parquet(processed_parquet)
        .head(SAMPLE_PARQUET_MAX_ROWS)
        .collect()
    )
    proc_sample = proc_for_sample.sample(
        min(SAMPLE_ROW_COUNT, proc_for_sample.height),
        seed=42,
        shuffle=True,
    )
    write_sample_rows_json(output_dir / "occurrences_sample.json", proc_sample)

    ykj = pl.read_csv(YKJ_CENTERPOINTS, separator=";").rename(
        {"lat": "latitude", "lon": "longitude"}
    )

    agg_yearly = (
        pl.scan_parquet(processed_parquet)
        .group_by("speciesName", "year", "gridCellYKJ")
        .agg(
            pl.len().alias("occurrenceCount"),
            pl.col("taxonConceptID").first(),
            pl.col("vernacularName").first(),
        )
        .collect()
        .join(ykj, on="gridCellYKJ", how="left")
        .with_columns(
            pl.col("latitude").round(6),
            pl.col("longitude").round(6),
        )
    )
    agg_yearly.write_parquet(
        aggregate_yearly_10km,
        compression="zstd",
        statistics=True,
    )

    agg_yearly_sample = agg_yearly.sample(
        min(SAMPLE_ROW_COUNT, agg_yearly.height),
        seed=42,
        shuffle=True,
    )
    write_sample_rows_json(
        aggregate_yearly_10km.with_name(
            aggregate_yearly_10km.name.replace(".parquet", "_sample.json")
        ),
        agg_yearly_sample,
    )

    agg_daily = (
        pl.scan_parquet(processed_parquet)
        .filter(pl.col("daysSpan") <= 2)
        .group_by("speciesName", "eventDateBegin", "gridCellYKJ")
        .agg(
            pl.len().alias("occurrenceCount"),
            pl.col("taxonConceptID").first(),
            pl.col("vernacularName").first(),
        )
        .collect()
        .join(ykj, on="gridCellYKJ", how="left")
        .with_columns(
            pl.col("latitude").round(6),
            pl.col("longitude").round(6),
        )
    )
    agg_daily.write_parquet(
        aggregate_daily_10km,
        compression="zstd",
        statistics=True,
    )

    agg_daily_sample = agg_daily.sample(
        min(SAMPLE_ROW_COUNT, agg_daily.height),
        seed=42,
        shuffle=True,
    )
    write_sample_rows_json(
        aggregate_daily_10km.with_name(
            aggregate_daily_10km.name.replace(".parquet", "_sample.json")
        ),
        agg_daily_sample,
    )

    agg_dayofyear = (
        pl.scan_parquet(processed_parquet)
        .filter(pl.col("daysSpan") <= 2)
        .group_by("speciesName", "year", "dayOfYear", "gridCellYKJ")
        .agg(
            pl.len().alias("occurrenceCount"),
            pl.col("taxonConceptID").first(),
            pl.col("vernacularName").first(),
        )
        .collect()
        .join(ykj, on="gridCellYKJ", how="left")
        .with_columns(
            pl.col("latitude").round(6),
            pl.col("longitude").round(6),
        )
    )
    agg_dayofyear.write_parquet(
        aggregate_dayofyear_10km,
        compression="zstd",
        statistics=True,
    )

    agg_dayofyear_sample = agg_dayofyear.sample(
        min(SAMPLE_ROW_COUNT, agg_dayofyear.height),
        seed=42,
        shuffle=True,
    )
    write_sample_rows_json(
        aggregate_dayofyear_10km.with_name(
            aggregate_dayofyear_10km.name.replace(".parquet", "_sample.json")
        ),
        agg_dayofyear_sample,
    )


if __name__ == "__main__":
    main()
