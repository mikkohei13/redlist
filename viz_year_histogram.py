"""
Histogram of occurrence years (from eventDate string, first year in range).

Run from repo root: uv run python viz_year_histogram.py
"""

import matplotlib.pyplot as plt
import polars as pl

from preprocess_occurrences import OUTPUT_DIR, PROCESSED_PARQUET

OUTPUT_FIG = OUTPUT_DIR / "viz_year_histogram.png"


def main() -> None:
    years = (
        pl.scan_parquet(PROCESSED_PARQUET)
        .select(
            pl.col("eventDate")
            .str.split("/")
            .list.first()
            .str.slice(0, 4)
            .cast(pl.Int32, strict=False)
            .alias("year")
        )
        .filter(pl.col("year").is_not_null())
        .collect()
    )["year"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(years.to_numpy(), bins=50, color="steelblue", edgecolor="white")
    ax.set_xlabel("Year")
    ax.set_ylabel("Occurrences")
    ax.set_title("Occurrences by year (from eventDate)")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
