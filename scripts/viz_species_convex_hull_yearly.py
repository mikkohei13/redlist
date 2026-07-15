"""
Per-species cumulative convex hull area by year (baseline 2010–2016, chart from 2016).

Reads aggregate_yearly_10km.parquet, keeps years >= 2010, and for each species
with at least 3 records writes a chart under output/<slug>/convex_hull/ and a
CSV summary of hull area change from 2016 to the latest year.
Hull area at 2016 uses all records through 2016; later years add cumulatively.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from dataset_io import dataset_from_argv, output_path, require_file, write_csv
from stats_species_convex_hull import convex_hull_area_km2, lon_lat_to_xy_m

USAGE = "uv run scripts/viz_species_convex_hull_yearly.py <dataset-slug>"

MIN_RECORDS = 3
FIRST_YEAR = 2010
CHART_FIRST_YEAR = 2016
INCLUDE_CURRENT_YEAR = True

if INCLUDE_CURRENT_YEAR:
    LAST_YEAR = date.today().year
else:
    LAST_YEAR = date.today().year - 1


def species_name_to_chart_filename(species_name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in species_name)
    safe = "_".join(safe.split())
    if len(safe) > 120:
        safe = safe[:120]
    return f"{safe}.png"


def plot_cumulative_hull_area(
    species_name: str,
    years: list[int],
    areas_km2: list[float],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(years, areas_km2, color="lightblue", alpha=0.5)
    ax.plot(years, areas_km2, marker="o", linewidth=1.5, color="steelblue")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative convex hull area (km²)")
    ax.set_title(species_name)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.set_ylim(0, 500_000)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def cumulative_hull_areas_by_year(
    sp_df: pl.DataFrame,
    chart_years: list[int],
) -> list[float]:
    """
    Hull area (km²) from unique cell coords accumulated from FIRST_YEAR through
    each chart year. Data from 2010–2016 forms the baseline at 2016.
    """
    cumulative: set[tuple[float, float]] = set()
    areas: list[float] = []
    for year in range(FIRST_YEAR, chart_years[-1] + 1):
        year_cells = (
            sp_df.filter(pl.col("year") == year)
            .select("longitude", "latitude")
            .drop_nulls()
            .unique()
        )
        for lon, lat in zip(
            year_cells["longitude"].to_list(),
            year_cells["latitude"].to_list(),
            strict=True,
        ):
            cumulative.add((lon, lat))
        if year >= CHART_FIRST_YEAR:
            areas.append(convex_hull_area_km2(lon_lat_to_xy_m(list(cumulative))))
    return areas


def main() -> None:
    ds = dataset_from_argv(usage=USAGE)
    path = require_file(ds.path("aggregate_yearly_10km"), label="parquet file")

    chart_years = list(range(CHART_FIRST_YEAR, LAST_YEAR + 1))

    df = pl.read_parquet(path).filter(
        (pl.col("year") >= FIRST_YEAR) & (pl.col("year") <= LAST_YEAR)
    )

    species_totals = (
        df.group_by("speciesName")
        .agg(pl.col("occurrenceCount").sum().alias("total_records"))
        .filter(pl.col("total_records") >= MIN_RECORDS)
        .sort("speciesName")
    )

    chart_dir = output_path(ds, "convex_hull/chart.png").parent
    summary_rows: list[dict[str, object]] = []
    for species_name, total_records in zip(
        species_totals["speciesName"].to_list(),
        species_totals["total_records"].to_list(),
        strict=True,
    ):
        sp_df = df.filter(pl.col("speciesName") == species_name)
        areas = cumulative_hull_areas_by_year(sp_df, chart_years)
        area_2016 = areas[0]
        area_latest = areas[-1]
        summary_rows.append(
            {
                "speciesName": species_name,
                "total_records": total_records,
                "hull_area_2016_km2": round(area_2016, 1),
                f"hull_area_{LAST_YEAR}_km2": round(area_latest, 1),
                "hull_area_change_km2": round(area_latest - area_2016, 1),
            }
        )
        out_path = chart_dir / species_name_to_chart_filename(species_name)
        plot_cumulative_hull_area(species_name, chart_years, areas, out_path)

    summary_csv = output_path(ds, "viz_species_convex_hull_yearly.csv")
    summary_table = pl.DataFrame(summary_rows).sort("hull_area_change_km2", descending=True)
    write_csv(summary_table, summary_csv)

    print(
        f"Wrote {summary_table.height} cumulative hull charts under {chart_dir} "
        f"(species with >={MIN_RECORDS} records, {CHART_FIRST_YEAR}–{LAST_YEAR})"
    )
    print(f"Wrote {summary_csv} ({summary_table.height} species)")


if __name__ == "__main__":
    main()
