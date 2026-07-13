"""
Histogram of occurrence years (from eventDate string, first year in range).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: uv run scripts/viz_year_histogram.py <dataset-slug>")
        sys.exit(1)

    dataset_slug = sys.argv[1]
    output_dir = ROOT / "output" / dataset_slug
    processed_parquet = output_dir / "occurrences.parquet"
    if not processed_parquet.is_file():
        print(f"error: missing parquet file: {processed_parquet}")
        sys.exit(1)

    years = (
        pl.scan_parquet(processed_parquet)
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

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(years.to_numpy(), bins=50, color="steelblue", edgecolor="white")
    ax.set_xlabel("Year")
    ax.set_ylabel("Occurrences")
    ax.set_title("Occurrences by year (from eventDate)")
    fig.tight_layout()
    fig.savefig(output_dir / "viz_year_histogram.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
