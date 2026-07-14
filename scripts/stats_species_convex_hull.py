"""
Per-species convex hull over unique YKJ cell center coordinates (WGS84),
area in km² after projecting to ETRS-TM35FIN (EPSG:3067).

Reads occurrences_aggregated.parquet (speciesName, latitude, longitude).

Extension: import `hull_areas_table` (returns a stats frame plus polygon rows
for GeoPackage) and pass any subset of the same schema (e.g. after dropping
rows) to compare hull areas without changing the geometry helpers.

Also writes per-species charts under output/<slug>/hull_removal_sim/: for each
species with more than 10 unique YKJ cells, 10 Monte Carlo runs remove random
cells one at a time (10 removals) and plot hull area vs removals.
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from pyproj import Transformer
from shapely.geometry import MultiPoint, Polygon

from dataset_io import dataset_from_argv, output_path, require_file, write_csv
from gpkg_species_hulls import write_species_polygon_layers

USAGE = "uv run scripts/stats_species_convex_hull.py <dataset-slug>"

N_REMOVAL_STEPS = 10
N_SIM_RUNS = 50
SIM_RNG_SEED = 42

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


def simulate_random_removal_hull_areas(
    xy_m: list[tuple[float, float]],
    n_removals: int,
    n_runs: int,
    rng: random.Random,
) -> list[list[float]]:
    """
    For each run, repeatedly remove one random point and record hull area (km²).
    Returns ``n_runs`` lists of length ``n_removals + 1`` (areas after 0..n removals).
    """
    if len(xy_m) <= n_removals:
        raise ValueError("need more points than removals")
    curves: list[list[float]] = []
    for _ in range(n_runs):
        pts = list(xy_m)
        areas = [convex_hull_area_km2(pts)]
        for _ in range(n_removals):
            del pts[rng.randrange(len(pts))]
            areas.append(convex_hull_area_km2(pts))
        curves.append(areas)
    return curves


def species_name_to_chart_filename(species_name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in species_name)
    safe = "_".join(safe.split())
    return (safe[:180] if len(safe) > 180 else safe) + ".png"


def plot_hull_removal_simulation(
    species_name: str,
    curves: list[list[float]],
    out_path: Path,
) -> None:
    n_steps = len(curves[0])
    x = list(range(n_steps))
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, ys in enumerate(curves):
        ax.plot(
            x,
            ys,
            linewidth=1.2,
            alpha=0.5,
            color=f"C{i}",
        )
    mean_y = [sum(vals) / len(vals) for vals in zip(*curves)]
    ax.plot(
        x,
        mean_y,
        color="black",
        linewidth=2.5,
        alpha=1.0,
        label="Mean",
        zorder=10,
    )
    ax.legend(loc="best")
    ax.set_xlabel("Records removed")
    ax.set_ylabel("Convex hull area (km²)")
    ax.set_title(f"{species_name}: hull area vs random cell removals")
    ax.set_xticks(x)
    ax.set_ylim(bottom=0)
    for y_ref in (2000, 20_000, 50_000):
        ax.axhline(
            y_ref,
            linestyle="--",
            color="gray",
            linewidth=1,
            alpha=0.75,
            zorder=0,
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_removal_simulation_charts(
    aggregated: pl.DataFrame,
    chart_dir: Path,
    n_removals: int,
    n_runs: int,
    rng_seed: int,
) -> None:
    """One chart per species with enough unique cells (n > n_removals)."""
    grouped = unique_lon_lat_by_species(aggregated)
    species = grouped["speciesName"].to_list()
    lons = grouped["longitude"].to_list()
    lats = grouped["latitude"].to_list()
    rng = random.Random(rng_seed)
    n_charts = 0
    for name, lon_series, lat_series in zip(species, lons, lats, strict=True):
        pairs = list(zip(lon_series, lat_series, strict=True))
        if len(pairs) <= n_removals:
            continue
        xy_m = lon_lat_to_xy_m(pairs)
        curves = simulate_random_removal_hull_areas(xy_m, n_removals, n_runs, rng)
        out_path = chart_dir / species_name_to_chart_filename(name)
        plot_hull_removal_simulation(name, curves, out_path)
        n_charts += 1
    print(
        f"Wrote {n_charts} removal-simulation charts under {chart_dir} "
        f"(species with >{n_removals} unique cells)"
    )


def main() -> None:
    ds = dataset_from_argv(usage=USAGE)
    aggregate_parquet = require_file(ds.path("aggregate_yearly_10km"), label="parquet file")

    aggregated = pl.read_parquet(aggregate_parquet)
    out, gpkg_rows = hull_areas_table(aggregated)
    output_csv = output_path(ds, "stats_species_convex_hull.csv")
    output_gpkg = output_path(ds, "stats_species_convex_hull.gpkg")
    sim_chart_dir = output_path(ds, "hull_removal_sim/chart.png").parent
    write_csv(out, output_csv)
    print(f"Wrote {output_csv} ({out.height} species)")

    n_layers = write_species_polygon_layers(output_gpkg, gpkg_rows)
    if n_layers == 0:
        print("error: no convex hull polygons to write to GeoPackage")
        raise SystemExit(1)
    print(f"Wrote {output_gpkg} ({n_layers} layers)")

    run_removal_simulation_charts(
        aggregated,
        sim_chart_dir,
        N_REMOVAL_STEPS,
        N_SIM_RUNS,
        SIM_RNG_SEED,
    )


if __name__ == "__main__":
    main()
