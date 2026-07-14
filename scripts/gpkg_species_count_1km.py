"""
GeoPackage of 1 km YKJ grid cells with distinct species counts.

Reads aggregate_daily_1km.parquet and writes one polygon layer (EPSG:2393).
"""

from __future__ import annotations

import geopandas as gpd
import polars as pl
from shapely.geometry import box

from dataset_io import dataset_from_argv, output_path, require_file

USAGE = "uv run scripts/gpkg_species_count_1km.py <dataset-slug>"

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
    ds = dataset_from_argv(usage=USAGE)
    input_parquet = require_file(ds.path("aggregate_daily_1km"), label="parquet file")
    output_gpkg = output_path(ds, "species_count_1km.gpkg")

    df = pl.read_parquet(input_parquet)

    agg = (
        df.group_by("gridCellYKJ")
        .agg(pl.col("speciesName").n_unique().alias("species_count"))
        .sort("gridCellYKJ")
    )

    if agg.is_empty():
        print("error: no grid cells after aggregation")
        raise SystemExit(1)

    grid_cells = agg["gridCellYKJ"].to_list()
    geometry = [ykj_cell_polygon(cell) for cell in grid_cells]
    cols = agg.to_dict(as_series=False)
    cols["geometry"] = geometry
    gdf = gpd.GeoDataFrame(cols, crs=CRS)

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
