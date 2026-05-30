"""
Per-species weighted mean and median day-of-year from aggregate_dayofyear_10km.

Each row is a (speciesName, dayOfYear, grid cell, …) bucket with occurrenceCount;
counts are used as weights for mean and median. Output: integer rounded mean and
integer median DOY, ``day.month.`` calendar strings (anchor year 2024 for DOY
366), and per-species sum of occurrenceCount.

Run from repo root: uv run python stats_species_dayofyear.py
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl

from preprocess_occurrences import AGGREGATE_DAYOFYEAR_10KM, OUTPUT_DIR

OUTPUT_TSV = OUTPUT_DIR / "stats_species_dayofyear_10km.tsv"

# Calendar mapping for DOY → day.month. (supports DOY 366).
_DOY_ANCHOR_YEAR = 2024


def day_of_year_to_day_month_str(doy: int) -> str:
    """e.g. 172 → '20.6.'"""
    d = date(_DOY_ANCHOR_YEAR, 1, 1) + timedelta(days=doy - 1)
    return f"{d.day}.{d.month}."


def _weighted_mean_and_median_sorted_pairs(
    pairs: list[tuple[int, int]],
) -> tuple[float, int, int]:
    """pairs: (dayOfYear, occurrenceCount) sorted by dayOfYear. Returns mean (float), median day, total weight."""
    total_w = sum(w for _, w in pairs)
    if total_w == 0:
        print("error: zero total occurrenceCount in group")
        raise SystemExit(1)
    weighted_sum = sum(d * w for d, w in pairs)
    mean_val = weighted_sum / total_w
    half = total_w / 2.0
    cum = 0
    for d, w in pairs:
        cum += w
        if cum >= half:
            return mean_val, d, total_w
    return mean_val, pairs[-1][0], total_w


def species_weighted_day_stats(group: pl.DataFrame) -> pl.DataFrame:
    g = group.sort("dayOfYear")
    pairs = list(
        zip(
            g["dayOfYear"].to_list(),
            g["occurrenceCount"].to_list(),
            strict=True,
        )
    )
    mean_doy_f, median_doy, occurrence_count_sum = _weighted_mean_and_median_sorted_pairs(
        pairs
    )
    mean_doy = int(round(mean_doy_f))
    name = g["speciesName"][0]
    return pl.DataFrame(
        {
            "speciesName": [name],
            "mean_dayOfYear": [mean_doy],
            "median_dayOfYear": [median_doy],
            "mean_date": [day_of_year_to_day_month_str(mean_doy)],
            "median_date": [day_of_year_to_day_month_str(median_doy)],
            "occurrenceCount_sum": [occurrence_count_sum],
        },
        schema={
            "speciesName": pl.Utf8,
            "mean_dayOfYear": pl.Int64,
            "median_dayOfYear": pl.Int64,
            "mean_date": pl.Utf8,
            "median_date": pl.Utf8,
            "occurrenceCount_sum": pl.Int64,
        },
    )


def main() -> None:
    path: Path = AGGREGATE_DAYOFYEAR_10KM
    if not path.is_file():
        print(f"error: missing aggregate file {path}")
        raise SystemExit(1)
    df = pl.read_parquet(path)
    out = (
        df.group_by("speciesName", maintain_order=True)
        .map_groups(species_weighted_day_stats)
        .sort("speciesName")
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_csv(OUTPUT_TSV, separator="\t")
    print(f"Wrote {OUTPUT_TSV} ({out.height} species)")


if __name__ == "__main__":
    main()
