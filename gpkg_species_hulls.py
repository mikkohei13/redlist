"""GeoPackage I/O: one vector layer per species polygon (EPSG:3067)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import geopandas as gpd
from shapely.geometry.base import BaseGeometry


def sanitize_gpkg_layer_name(species_name: str) -> str:
    """
    GeoPackage / SQLite table name: letters, digits, underscore; must not start
    with a digit. Keep names stable across runs.
    """
    s = species_name.strip().replace(" ", "_")
    s = re.sub(r"[^0-9A-Za-z_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "species"
    if s[0].isdigit():
        s = f"L_{s}"
    if len(s) > 58:
        digest = hashlib.md5(species_name.encode()).hexdigest()[:8]
        s = f"{s[:48]}_{digest}"
    return s


def write_species_polygon_layers(
    path: Path,
    species_geometries: list[tuple[str, BaseGeometry | None]],
    *,
    crs: str = "EPSG:3067",
) -> int:
    """
    Write one GeoPackage layer per species. Skips non-polygon or empty hulls.

    Returns the number of layers written. Overwrites `path` if it exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    used_layers: set[str] = set()
    written = 0
    first = True

    for species_name, geom in species_geometries:
        if geom is None or geom.is_empty or geom.geom_type != "Polygon":
            continue
        base = sanitize_gpkg_layer_name(species_name)
        layer = base
        suffix = 2
        while layer in used_layers:
            layer = f"{base}_{suffix}"
            suffix += 1
        used_layers.add(layer)

        gdf = gpd.GeoDataFrame(
            {"species_name": [species_name]},
            geometry=[geom],
            crs=crs,
        )
        gdf.to_file(path, layer=layer, driver="GPKG", mode="w" if first else "a")
        first = False
        written += 1

    return written
