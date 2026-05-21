"""
Count occurrences by taxonRank.

Run from repo root: uv run python stats_taxon_ranks.py
"""

import polars as pl

from config import OUTPUT_DIR, PROCESSED_PARQUET

OUTPUT_CSV = OUTPUT_DIR / "stats_taxon_ranks.csv"


def main() -> None:
    counts = (
        pl.scan_parquet(PROCESSED_PARQUET)
        .group_by("taxonRank")
        .len()
        .sort("len", descending=True)
        .collect()
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts.write_csv(OUTPUT_CSV)


if __name__ == "__main__":
    main()
