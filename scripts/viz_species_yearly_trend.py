"""
Per-species yearly observation proportion (2010–current) and linear trend.

Reads aggregate_yearly_10km.parquet, keeps years >= 2010, and for each species
with at least 10 records writes a chart under output/<slug>/trend/ and a CSV
summary sorted by trend descending.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from dataset_io import dataset_from_argv, output_path, require_file, write_csv

USAGE = "uv run scripts/viz_species_yearly_trend.py <dataset-slug>"

MIN_RECORDS = 10
FIRST_YEAR = 2010


def linear_slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


def species_name_to_chart_filename(species_name: str, trend: float) -> str:
    safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in species_name)
    safe = "_".join(safe.split())
    if len(safe) > 120:
        safe = safe[:120]
    return f"{safe}-{trend:.6f}.png"


def plot_species_trend(
    species_name: str,
    years: list[int],
    proportions: list[float],
    trend: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, proportions, marker="o", linewidth=1.5, color="steelblue")
    ax.set_xlabel("Year")
    ax.set_ylabel("Proportion of yearly records")
    ax.set_title(species_name)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.set_ylim(bottom=0)
    ax.text(
        0.02,
        0.98,
        f"trend = {trend:.6f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ds = dataset_from_argv(usage=USAGE)
    path = require_file(ds.path("aggregate_yearly_10km"), label="parquet file")

    current_year = date.today().year
    years = list(range(FIRST_YEAR, current_year + 1))

    df = pl.read_parquet(path).filter(pl.col("year") >= FIRST_YEAR)

    yearly_totals = (
        df.group_by("year")
        .agg(pl.col("occurrenceCount").sum().alias("total_count"))
        .sort("year")
    )
    totals_by_year = dict(
        zip(
            yearly_totals["year"].to_list(),
            yearly_totals["total_count"].to_list(),
            strict=True,
        )
    )

    species_yearly = (
        df.group_by("speciesName", "year")
        .agg(pl.col("occurrenceCount").sum().alias("species_count"))
        .sort("speciesName", "year")
    )
    species_totals = (
        species_yearly.group_by("speciesName")
        .agg(pl.col("species_count").sum().alias("total_records"))
        .filter(pl.col("total_records") >= MIN_RECORDS)
        .sort("speciesName")
    )

    chart_dir = output_path(ds, "trend/chart.png").parent
    trend_rows: list[dict[str, object]] = []
    for species_name, total_records in zip(
        species_totals["speciesName"].to_list(),
        species_totals["total_records"].to_list(),
        strict=True,
    ):
        sp_rows = species_yearly.filter(pl.col("speciesName") == species_name)
        counts_by_year = dict(
            zip(
                sp_rows["year"].to_list(),
                sp_rows["species_count"].to_list(),
                strict=True,
            )
        )
        proportions = []
        for year in years:
            total = totals_by_year.get(year, 0)
            count = counts_by_year.get(year, 0)
            proportions.append(count / total if total else 0.0)

        trend = linear_slope([float(y) for y in years], proportions)
        trend_rows.append({"speciesName": species_name, "trend": trend})
        out_path = chart_dir / species_name_to_chart_filename(species_name, trend)
        plot_species_trend(species_name, years, proportions, trend, out_path)

    trend_csv = output_path(ds, "viz_species_yearly_trend.csv")
    trend_table = pl.DataFrame(trend_rows).sort("trend", descending=True)
    write_csv(trend_table, trend_csv)

    print(
        f"Wrote {trend_table.height} trend charts under {chart_dir} "
        f"(species with >={MIN_RECORDS} records, {FIRST_YEAR}–{current_year})"
    )
    print(f"Wrote {trend_csv} ({trend_table.height} species)")


if __name__ == "__main__":
    main()
