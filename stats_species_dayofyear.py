"""
Per-species weighted mean and median day-of-year from aggregate_dayofyear_10km.

Each row is a (speciesName, dayOfYear, grid cell, …) bucket with occurrenceCount;
counts are used as weights for mean and median. Output: integer rounded mean and
integer median DOY, ``day.month.`` calendar strings (anchor year 2024 for DOY
366), per-species sum of occurrenceCount, and the shortest linear DOY interval
that covers at least 60% of that sum (two-pointer over days with aggregated
counts).

Run from repo root: uv run python stats_species_dayofyear.py
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl

from preprocess_occurrences import AGGREGATE_DAYOFYEAR_10KM, OUTPUT_DIR

OUTPUT_TSV = OUTPUT_DIR / "stats_species_dayofyear_10km.tsv"

COVER_FRACTION = 0.6
COVER_NAME = f"cover{COVER_FRACTION*100:.0f}"

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


def shortest_linear_doy_window_covering(
    sorted_day_count: list[tuple[int, int]],
    total: int,
    fraction: float,
) -> tuple[int, int, int, float]:
    """
    ``sorted_day_count``: (dayOfYear, count) sorted by day, counts summed per day.
    Minimize ``end_day - start_day`` among windows with sum(count) >= fraction * total.
    Returns start_day, end_day, window_sum, window_sum / total.
    """
    target = fraction * total
    n = len(sorted_day_count)
    days = [d for d, _ in sorted_day_count]
    cnts = [c for _, c in sorted_day_count]
    j = 0
    cur = 0
    best_span: int | None = None
    best_start = 0
    best_end = 0
    best_sum = 0
    for i in range(n):
        while j < n and cur < target:
            cur += cnts[j]
            j += 1
        if cur >= target:
            start_d, end_d = days[i], days[j - 1]
            span = end_d - start_d
            if best_span is None or span < best_span or (
                span == best_span
                and (start_d < best_start or (start_d == best_start and end_d < best_end))
            ):
                best_span = span
                best_start, best_end = start_d, end_d
                best_sum = cur
        cur -= cnts[i]
    if best_span is None:
        print("error: no window reaches coverage target (empty day list?)")
        raise SystemExit(1)
    return best_start, best_end, best_sum, best_sum / total


def species_weighted_day_stats(group: pl.DataFrame) -> pl.DataFrame:
    g = group.sort("dayOfYear")
    by_day = (
        g.group_by("dayOfYear")
        .agg(pl.col("occurrenceCount").sum())
        .sort("dayOfYear")
    )
    pairs = list(
        zip(
            by_day["dayOfYear"].to_list(),
            by_day["occurrenceCount"].to_list(),
            strict=True,
        )
    )
    mean_doy_f, median_doy, occurrence_count_sum = _weighted_mean_and_median_sorted_pairs(
        pairs
    )
    mean_doy = int(round(mean_doy_f))
    c60_start, c60_end, c60_sum, c60_frac = shortest_linear_doy_window_covering(
        pairs, occurrence_count_sum, COVER_FRACTION
    )
    name = g["speciesName"][0]
    return pl.DataFrame(
        {
            "speciesName": [name],
            "mean_dayOfYear": [mean_doy],
            "median_dayOfYear": [median_doy],
            "mean_date": [day_of_year_to_day_month_str(mean_doy)],
            "median_date": [day_of_year_to_day_month_str(median_doy)],
            "occurrenceCount_sum": [occurrence_count_sum],
            f"{COVER_NAME}_start_day": [c60_start],
            f"{COVER_NAME}_end_day": [c60_end],
            f"{COVER_NAME}_days": [c60_end - c60_start + 1],
            f"{COVER_NAME}_start_date": [day_of_year_to_day_month_str(c60_start)],
            f"{COVER_NAME}_end_date": [day_of_year_to_day_month_str(c60_end)],
            f"{COVER_NAME}_window_sum": [c60_sum],
            f"{COVER_NAME}_fraction": [c60_frac],
        },
        schema={
            "speciesName": pl.Utf8,
            "mean_dayOfYear": pl.Int64,
            "median_dayOfYear": pl.Int64,
            "mean_date": pl.Utf8,
            "median_date": pl.Utf8,
            "occurrenceCount_sum": pl.Int64,
            f"{COVER_NAME}_start_day": pl.Int64,
            f"{COVER_NAME}_end_day": pl.Int64,
            f"{COVER_NAME}_days": pl.Int64,
            f"{COVER_NAME}_start_date": pl.Utf8,
            f"{COVER_NAME}_end_date": pl.Utf8,
            f"{COVER_NAME}_window_sum": pl.Int64,
            f"{COVER_NAME}_fraction": pl.Float64,
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
