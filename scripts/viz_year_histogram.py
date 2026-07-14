"""
Histogram of occurrence years (from eventDate string, first year in range).
"""

import matplotlib.pyplot as plt
import polars as pl

from dataset_io import dataset_from_argv, output_path, require_file

USAGE = "uv run scripts/viz_year_histogram.py <dataset-slug>"


def main() -> None:
    ds = dataset_from_argv(usage=USAGE)
    processed_parquet = require_file(ds.path("occurrences"), label="parquet file")

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

    out = output_path(ds, "viz_year_histogram.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(years.to_numpy(), bins=50, color="steelblue", edgecolor="white")
    ax.set_xlabel("Year")
    ax.set_ylabel("Occurrences")
    ax.set_title("Occurrences by year (from eventDate)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
