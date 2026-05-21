"""
Per-species convex hull over unique YKJ cell center coordinates (WGS84),
area in km² after projecting to ETRS-TM35FIN (EPSG:3067).

Reads occurrences_aggregated.parquet (speciesName, latitude, longitude).

Run from repo root: uv run python stats_species_convex_hull.py

Extension: import `hull_areas_table` (returns a stats frame plus polygon rows
for GeoPackage) and pass any subset of the same schema (e.g. after dropping
rows) to compare hull areas without changing the geometry helpers.
"""

from __future__ import annotations

import polars as pl
from pyproj import Transformer
from shapely.geometry import MultiPoint, Polygon

from config import AGGREGATED_PARQUET, OUTPUT_DIR
from gpkg_species_hulls import write_species_polygon_layers

OUTPUT_CSV = OUTPUT_DIR / "stats_species_convex_hull.csv"
OUTPUT_GPKG = OUTPUT_DIR / "stats_species_convex_hull.gpkg"

# Finland / Baltic — meters for planar hull area.
_WGS84_TO_3067 = Transformer.from_crs("EPSG:4326", "EPSG:3067", always_xy=True)


def lon_lat_to_xy_m(lon_lat_pairs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Project (lon, lat) WGS84 points to EPSG:3067 x/y in meters."""
    out: list[tuple[float, float]] = []
    for lon, lat in lon_lat_pairs:
        x, y = _WGS84_TO_3067.transform(lon, lat)
        out.append((x, y))
    return out


def convex_hull_polygon_xy_m(xy_m: list[tuple[float, float]]) -> Polygon | None:
    """Planar convex hull as a polygon in projected meters, or None if degenerate."""
    if len(xy_m) < 3:
        return None
    hull = MultiPoint(xy_m).convex_hull
    return hull if hull.geom_type == "Polygon" else None


def convex_hull_area_km2(xy_m: list[tuple[float, float]]) -> float:
    """
    Planar convex hull area in km². Degenerate hulls (fewer than 3 vertices,
    collinear points) yield 0.0.
    """
    poly = convex_hull_polygon_xy_m(xy_m)
    return poly.area / 1_000_000.0 if poly else 0.0


def unique_lon_lat_by_species(df: pl.DataFrame) -> pl.DataFrame:
    """One row per species with parallel lists of distinct cell center coords."""
    return (
        df.select(["speciesName", "longitude", "latitude"])
        .drop_nulls()
        .unique()
        .sort("speciesName", "longitude", "latitude")
        .group_by("speciesName", maintain_order=True)
        .agg(
            pl.col("longitude"),
            pl.col("latitude"),
        )
    )


def hull_areas_table(
    aggregated: pl.DataFrame,
) -> tuple[pl.DataFrame, list[tuple[str, Polygon | None]]]:
    """
    Build speciesName, n_unique_cells, hull_area_km2 (sorted by area descending),
    and parallel (speciesName, hull polygon in EPSG:3067 or None) for GeoPackage.

    Same columns as the aggregated parquet suffice; extra columns are ignored.
    """
    grouped = unique_lon_lat_by_species(aggregated)
    species = grouped["speciesName"].to_list()
    lons = grouped["longitude"].to_list()
    lats = grouped["latitude"].to_list()

    rows: list[dict[str, object]] = []
    gpkg_rows: list[tuple[str, Polygon | None]] = []
    for name, lon_series, lat_series in zip(species, lons, lats, strict=True):
        pairs = list(zip(lon_series, lat_series, strict=True))
        n = len(pairs)
        xy_m = lon_lat_to_xy_m(pairs) if n else []
        poly = convex_hull_polygon_xy_m(xy_m) if n else None
        area_km2 = poly.area / 1_000_000.0 if poly else 0.0
        rows.append(
            {
                "speciesName": name,
                "n_unique_cells": n,
                "hull_area_km2": round(area_km2, 1),
            }
        )
        gpkg_rows.append((name, poly))

    table = pl.DataFrame(rows).sort("hull_area_km2", descending=True, nulls_last=True)
    return table, gpkg_rows


def main() -> None:
    aggregated = pl.read_parquet(AGGREGATED_PARQUET)
    out, gpkg_rows = hull_areas_table(aggregated)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_csv(OUTPUT_CSV)
    print(f"Wrote {OUTPUT_CSV} ({out.height} species)")

    n_layers = write_species_polygon_layers(OUTPUT_GPKG, gpkg_rows)
    if n_layers == 0:
        print("error: no convex hull polygons to write to GeoPackage")
        raise SystemExit(1)
    print(f"Wrote {OUTPUT_GPKG} ({n_layers} layers)")


if __name__ == "__main__":
    main()
