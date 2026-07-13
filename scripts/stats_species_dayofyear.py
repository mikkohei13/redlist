"""
Per-species mean and median day-of-year & window coverage from aggregate_dayofyear_10km.

Output includes ``speciesName``, ``taxonConceptID``, and ``vernacularName`` (from the first
aggregate row per species), then DOY stats and cover window columns.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent

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
    """pairs: (dayOfYear, row_weight) sorted by day. Returns mean (float), median day, total weight."""
    total_w = sum(w for _, w in pairs)
    if total_w == 0:
        print("error: zero aggregate rows in group")
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
    ``sorted_day_count``: (dayOfYear, row_weight) sorted by day (weights per day).
    Minimize ``end_day - start_day`` among windows with sum(weight) >= fraction * total.
    Returns start_day, end_day, window_weight_sum, window_weight_sum / total.
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
    taxon_id = g["taxonConceptID"][0]
    vernacular = g["vernacularName"][0]
    by_day = (
        g.group_by("dayOfYear")
        .agg(pl.len().alias("row_count"))
        .sort("dayOfYear")
    )
    pairs = list(
        zip(
            by_day["dayOfYear"].to_list(),
            by_day["row_count"].to_list(),
            strict=True,
        )
    )
    mean_doy_f, median_doy, species_row_total = _weighted_mean_and_median_sorted_pairs(
        pairs
    )
    mean_doy = int(round(mean_doy_f))
    c60_start, c60_end, c60_sum, c60_frac = shortest_linear_doy_window_covering(
        pairs, species_row_total, COVER_FRACTION
    )
    name = g["speciesName"][0]
    return pl.DataFrame(
        {
            "speciesName": [name],
            "taxonConceptID": [taxon_id],
            "vernacularName": [vernacular],
            "mean_dayOfYear": [mean_doy],
            "median_dayOfYear": [median_doy],
            "mean_date": [day_of_year_to_day_month_str(mean_doy)],
            "median_date": [day_of_year_to_day_month_str(median_doy)],
            "aggregate_row_count": [species_row_total],
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
            "taxonConceptID": pl.Utf8,
            "vernacularName": pl.Utf8,
            "mean_dayOfYear": pl.Int64,
            "median_dayOfYear": pl.Int64,
            "mean_date": pl.Utf8,
            "median_date": pl.Utf8,
            "aggregate_row_count": pl.Int64,
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
    if len(sys.argv) != 2:
        print("Usage: uv run scripts/stats_species_dayofyear.py <dataset-slug>")
        sys.exit(1)

    dataset_slug = sys.argv[1]
    output_dir = ROOT / "output" / dataset_slug
    path = output_dir / "aggregate_dayofyear_10km.parquet"
    if not path.is_file():
        print(f"error: missing parquet file: {path}")
        sys.exit(1)
    df = pl.read_parquet(path)
    for col in ("taxonConceptID", "vernacularName"):
        if col not in df.columns:
            print(f"error: aggregate parquet missing column {col!r}; rerun preprocess_occurrences.py")
            raise SystemExit(1)
    out = (
        df.group_by("speciesName", maintain_order=True)
        .map_groups(species_weighted_day_stats)
        .sort("speciesName")
    )
    output_tsv = output_dir / "stats_species_dayofyear_10km.tsv"
    output_dir.mkdir(parents=True, exist_ok=True)
    out.write_csv(output_tsv, separator="\t")
    print(f"Wrote {output_tsv} ({out.height} species)")


if __name__ == "__main__":
    main()
