"""
Per row in occurrences_aggregated.parquet: convex hull area (km², EPSG:3067)
for that species if this row were removed vs full hull — i.e. how much
removing the row would shrink the hull (0 if the cell still appears on
another row).

Output XLSX keeps only rows with hull_area_decrease_km2 > 0 and
hull_area_if_row_removed_km2 <= 50000, plus crossed_threshold (VU/EN/CR)
when the hull crosses an EOO-style km² threshold (most severe category
wins).

Run from repo root: uv run python stats_species_hull_row_impact.py
"""

from __future__ import annotations

from collections import Counter

import polars as pl

from preprocess_occurrences import AGGREGATED_PARQUET, OUTPUT_DIR
from stats_species_convex_hull import convex_hull_area_km2, lon_lat_to_xy_m

OUTPUT_XLSX = OUTPUT_DIR / "stats_species_hull_row_impact.xlsx"

# EOO-style hull area thresholds (km²); crossed_threshold when full is above
# and area-if-removed is below the same threshold (CR > EN > VU).
THRESHOLD_VU_KM2 = 20_000
THRESHOLD_EN_KM2 = 5_000
THRESHOLD_CR_KM2 = 100
MAX_AREA_IF_REMOVED_KM2 = 25_000

def red_list_crossed_threshold(area_full: float, area_if_removed: float) -> str | None:
    """IUCN-style category if hull crosses from above to below a threshold."""
    if area_full > THRESHOLD_CR_KM2 and area_if_removed < THRESHOLD_CR_KM2:
        return "CR"
    if area_full > THRESHOLD_EN_KM2 and area_if_removed < THRESHOLD_EN_KM2:
        return "EN"
    if area_full > THRESHOLD_VU_KM2 and area_if_removed < THRESHOLD_VU_KM2:
        return "VU"
    return None


def main() -> None:
    df = pl.read_parquet(AGGREGATED_PARQUET)
    need = {"speciesName", "latitude", "longitude"}
    missing = need - set(df.columns)
    if missing:
        print(f"error: aggregated parquet missing columns: {sorted(missing)}")
        raise SystemExit(1)

    df = df.drop_nulls(["speciesName", "latitude", "longitude"])
    if df.is_empty():
        print("error: no rows with speciesName and coordinates")
        raise SystemExit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_rows: list[dict[str, object]] = []

    for sp in df["speciesName"].unique().sort().to_list():
        sub = df.filter(pl.col("speciesName") == sp)
        lons = sub["longitude"].to_list()
        lats = sub["latitude"].to_list()
        points = list(zip(lons, lats, strict=True))
        counts = Counter(points)
        unique_sorted = sorted(set(points))
        xy_full = lon_lat_to_xy_m(unique_sorted)
        area_full = convex_hull_area_km2(xy_full)

        for i in range(sub.height):
            p = points[i]
            if counts[p] > 1:
                area_after = area_full
            else:
                remaining = [q for q in unique_sorted if q != p]
                xy_after = lon_lat_to_xy_m(remaining)
                area_after = convex_hull_area_km2(xy_after)

            decrease = area_full - area_after
            if decrease == 0 or area_after > MAX_AREA_IF_REMOVED_KM2:
                continue

            row = sub.row(i, named=True)
            row["hull_area_full_km2"] = round(area_full, 6)
            row["hull_area_if_row_removed_km2"] = round(area_after, 6)
            row["hull_area_decrease_km2"] = round(decrease, 6)
            crossed = red_list_crossed_threshold(area_full, area_after)
            row["crossed_threshold"] = crossed if crossed is not None else ""
            out_rows.append(row)

    out = pl.DataFrame(out_rows)
    if "year" in out.columns:
        out = out.with_columns(pl.col("year").cast(pl.Int64))
    out.write_excel(OUTPUT_XLSX, float_precision=6, autofit=True)
    print(f"Wrote {OUTPUT_XLSX} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
