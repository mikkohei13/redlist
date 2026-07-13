"""
GeoPackage of 1 km YKJ grid cells with distinct species counts.

Reads aggregate_daily_1km.parquet and writes one polygon layer (EPSG:2393).
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import polars as pl
from shapely.geometry import box

ROOT = Path(__file__).resolve().parent.parent

CELL_SIZE_M = 1000
CRS = "EPSG:2393"
GPKG_LAYER = "species_count_1km"


def ykj_cell_polygon(grid_cell: str) -> box:
    """1 km square for a YKJ cell ID (northing_km:easting_km) in EPSG:2393."""
    n_km, e_km = grid_cell.split(":")
    e = int(e_km) * CELL_SIZE_M
    n = int(n_km) * CELL_SIZE_M
    return box(e, n, e + CELL_SIZE_M, n + CELL_SIZE_M)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: uv run scripts/gpkg_species_count_1km.py <dataset-slug>")
        sys.exit(1)

    dataset_slug = sys.argv[1]
    output_dir = ROOT / "output" / dataset_slug
    input_parquet = output_dir / "aggregate_daily_1km.parquet"
    output_gpkg = output_dir / "species_count_1km.gpkg"

    if not input_parquet.is_file():
        print(f"error: missing parquet file: {input_parquet}")
        sys.exit(1)

    df = pl.read_parquet(input_parquet)

    agg = (
        df.group_by("gridCellYKJ")
        .agg(pl.col("speciesName").n_unique().alias("species_count"))
        .sort("gridCellYKJ")
    )

    if agg.is_empty():
        print("error: no grid cells after aggregation")
        sys.exit(1)

    grid_cells = agg["gridCellYKJ"].to_list()
    geometry = [ykj_cell_polygon(cell) for cell in grid_cells]
    cols = agg.to_dict(as_series=False)
    cols["geometry"] = geometry
    gdf = gpd.GeoDataFrame(cols, crs=CRS)

    output_dir.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_gpkg, driver="GPKG", layer=GPKG_LAYER)

    counts = agg["species_count"]
    print(f"wrote {output_gpkg} (layer {GPKG_LAYER!r})")
    print(f"grid cells: {agg.height}")
    print(
        "species per cell — "
        f"mean: {counts.mean():.4g}, median: {counts.median():.4g}, "
        f"min: {counts.min()}, max: {counts.max()}"
    )


if __name__ == "__main__":
    main()
