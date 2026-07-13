"""
Count occurrences by taxonRank.
"""

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: uv run scripts/stats_taxon_ranks.py <dataset-slug>")
        sys.exit(1)

    dataset_slug = sys.argv[1]
    output_dir = ROOT / "output" / dataset_slug
    processed_parquet = output_dir / "occurrences.parquet"
    if not processed_parquet.is_file():
        print(f"error: missing parquet file: {processed_parquet}")
        sys.exit(1)

    counts = (
        pl.scan_parquet(processed_parquet)
        .group_by("taxonRank")
        .len()
        .sort("len", descending=True)
        .collect()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    counts.write_csv(output_dir / "stats_taxon_ranks.csv")


if __name__ == "__main__":
    main()
