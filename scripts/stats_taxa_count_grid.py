"""
Count distinct scientificName values per 1 km × 1 km cell from a FinBIF occurrences.txt export.

Coordinates are read as WGS84 (decimalLongitude, decimalLatitude), projected to EPSG:3067
(ETRS-TM35FIN / EUREF-FIN), then snapped to a 1 km grid (cell SW corner in metres).

Writes Parquet (tabular) and GeoPackage (1 km square polygons, CRS EPSG:3067) for QGIS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
from pyproj import Transformer
from shapely.geometry import box

ROOT = Path(__file__).resolve().parent.parent

# EPSG:3067 — ETRS-TM35FIN (EUREF-FIN), metres. Cell size on the projected plane.
CELL_SIZE_M = 500
CRS_LABEL = "EPSG:3067"
GPKG_LAYER = f"taxa_{CELL_SIZE_M}m"

# FinBIF occurrences export: English header row, then three translated label rows.
SKIP_ROWS_AFTER_HEADER = 3


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: uv run scripts/stats_taxa_count_grid.py <dataset-slug>")
        sys.exit(1)

    dataset_slug = sys.argv[1]
    raw_occurrences = ROOT / "data" / dataset_slug / "occurrences.txt"
    output_dir = ROOT / "output" / dataset_slug
    taxa_per_cell_parquet = output_dir / f"taxa_per_{CELL_SIZE_M}m_cell.parquet"
    taxa_per_cell_gpkg = output_dir / f"taxa_per_{CELL_SIZE_M}m_cell.gpkg"

    if not raw_occurrences.is_file():
        print(f"error: missing occurrences file: {raw_occurrences}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    lf = pl.scan_csv(
        raw_occurrences,
        separator="\t",
        quote_char=None,
        skip_rows_after_header=SKIP_ROWS_AFTER_HEADER,
        infer_schema_length=10_000,
        try_parse_dates=False,
    ).select(
        pl.col("decimalLatitude").cast(pl.Float64, strict=False).alias("decimalLatitude"),
        pl.col("decimalLongitude").cast(pl.Float64, strict=False).alias("decimalLongitude"),
        pl.col("scientificName").cast(pl.Utf8, strict=False).alias("scientificName"),
    )

    df = lf.collect()

    df = df.filter(
        pl.col("scientificName").is_not_null()
        & pl.col("decimalLatitude").is_not_null()
        & pl.col("decimalLongitude").is_not_null()
    )

    lon = df["decimalLongitude"].to_numpy()
    lat = df["decimalLatitude"].to_numpy()

    transformer = Transformer.from_crs(4326, 3067, always_xy=True)
    e, n = transformer.transform(lon, lat)
    e = np.asarray(e, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)

    finite = np.isfinite(e) & np.isfinite(n)
    df = df.with_columns(pl.Series("_finite", finite)).filter(pl.col("_finite")).drop(
        "_finite"
    )
    e = e[finite]
    n = n[finite]

    cell_e = np.floor(e / CELL_SIZE_M) * CELL_SIZE_M
    cell_n = np.floor(n / CELL_SIZE_M) * CELL_SIZE_M

    df = df.with_columns(
        pl.Series("cell_easting_m", cell_e.astype(np.int64)),
        pl.Series("cell_northing_m", cell_n.astype(np.int64)),
    ).with_columns(
        (
            pl.lit(f"{CRS_LABEL}:")
            + pl.col("cell_easting_m").cast(pl.Utf8)
            + pl.lit(":")
            + pl.col("cell_northing_m").cast(pl.Utf8)
        ).alias("cell_id")
    )

    agg = (
        df.group_by("cell_id", "cell_easting_m", "cell_northing_m")
        .agg(pl.col("scientificName").n_unique().alias("distinct_scientificName_count"))
        .sort("cell_id")
    )

    agg = agg.with_columns(pl.lit(CRS_LABEL).alias("grid_crs"))

    if agg.is_empty():
        print("error: no grid cells after aggregation")
        raise SystemExit(1)

    count_col = pl.col("distinct_scientificName_count")
    mean_c, median_c, min_c, max_c = agg.select(
        count_col.mean().alias("_mean"),
        count_col.median().alias("_median"),
        count_col.min().alias("_min"),
        count_col.max().alias("_max"),
    ).row(0)

    cap_m = int(round(10 * float(mean_c)))

    agg = agg.with_columns(
        pl.col("distinct_scientificName_count").alias(
            "distinct_scientificName_count_uncapped"
        ),
    ).with_columns(
        pl.min_horizontal(
            pl.col("distinct_scientificName_count_uncapped"),
            pl.lit(cap_m),
        )
        .cast(pl.Int64)
        .alias("distinct_scientificName_count"),
    )

    capped_cells = agg.filter(
        pl.col("distinct_scientificName_count_uncapped") > pl.lit(cap_m)
    ).sort("distinct_scientificName_count_uncapped", descending=True)

    agg.write_parquet(taxa_per_cell_parquet, compression="zstd", statistics=True)

    ee = agg["cell_easting_m"].to_numpy()
    nn = agg["cell_northing_m"].to_numpy()
    geometry = [
        box(float(e), float(n), float(e) + CELL_SIZE_M, float(n) + CELL_SIZE_M)
        for e, n in zip(ee, nn, strict=True)
    ]
    cols = agg.to_dict(as_series=False)
    cols["geometry"] = geometry
    gdf = gpd.GeoDataFrame(cols, crs="EPSG:3067")
    gdf.to_file(taxa_per_cell_gpkg, driver="GPKG", layer=GPKG_LAYER)

    print(f"wrote {taxa_per_cell_parquet}")
    print(f"wrote {taxa_per_cell_gpkg} (layer {GPKG_LAYER!r})")
    print(f"occurrence rows used: {df.height}")
    print(f"distinct grid cells: {agg.height}")
    print(
        "distinct scientificName per occupied cell (uncapped) — "
        f"mean: {mean_c:.4g}, median: {median_c:.4g}, min: {min_c}, max: {max_c}"
    )
    print(f"taxon count cap (10× mean, rounded): {cap_m}")
    if capped_cells.is_empty():
        print("capped cells: none")
    else:
        print(f"capped cells ({capped_cells.height}), original distinct_scientificName_count:")
        for row in capped_cells.iter_rows(named=True):
            print(
                f"  {row['cell_id']}: {row['distinct_scientificName_count_uncapped']}"
            )


if __name__ == "__main__":
    main()
