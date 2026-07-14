"""
Path layout and I/O helpers for scripts.

Run: uv run scripts/<script>.py <dataset-slug>
Data: data/{slug}/occurrences.txt
Output: output/{slug}/
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent

FINBIF_SKIP_ROWS_AFTER_HEADER = 3
SAMPLE_ROW_COUNT = 5

# Canonical input names → path relative to data_dir or output_dir.
_OUTPUT_PATHS: dict[str, str] = {
    "occurrences": "occurrences.parquet",
    "aggregate_yearly_10km": "aggregate_yearly_10km.parquet",
    "aggregate_daily_10km": "aggregate_daily_10km.parquet",
    "aggregate_daily_1km": "aggregate_daily_1km.parquet",
    "aggregate_dayofyear_10km": "aggregate_dayofyear_10km.parquet",
}
_DATA_PATHS: dict[str, str] = {
    "raw_occurrences": "occurrences.txt",
    "ykj_centerpoints": "ykj-centerpoints.csv",
}


@dataclass(frozen=True)
class Dataset:
    slug: str
    root: Path = ROOT

    @property
    def data_dir(self) -> Path:
        return self.root / "data" / self.slug

    @property
    def output_dir(self) -> Path:
        return self.root / "output" / self.slug

    def path(self, name: str) -> Path:
        """Resolve a canonical input path by logical name."""
        if name in _OUTPUT_PATHS:
            return self.output_dir / _OUTPUT_PATHS[name]
        if name in _DATA_PATHS:
            if name == "ykj_centerpoints":
                return self.root / "data" / _DATA_PATHS[name]
            return self.data_dir / _DATA_PATHS[name]
        print(f"error: unknown dataset path name: {name!r}")
        sys.exit(1)


def dataset_from_argv(*, usage: str, argv: list[str] | None = None) -> Dataset:
    args = argv if argv is not None else sys.argv
    if len(args) != 2:
        print(f"Usage: {usage}")
        sys.exit(1)
    return Dataset(slug=args[1])


def require_file(path: Path, *, label: str | None = None) -> Path:
    if not path.is_file():
        kind = label or "file"
        print(f"error: missing {kind}: {path}")
        sys.exit(1)
    return path


def output_path(ds: Dataset, filename: str) -> Path:
    path = ds.output_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def scan_finbif_occurrences(path: Path, *, try_parse_dates: bool = False) -> pl.LazyFrame:
    """FinBIF DwC TSV: tab-separated, no quote char, three label rows after header."""
    return pl.scan_csv(
        path,
        separator="\t",
        quote_char=None,
        skip_rows_after_header=FINBIF_SKIP_ROWS_AFTER_HEADER,
        infer_schema_length=10_000,
        try_parse_dates=try_parse_dates,
    )


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd", statistics=True)


def write_csv(df: pl.DataFrame, path: Path, *, separator: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path, separator=separator)


def write_sample_json(df: pl.DataFrame, parquet_path: Path) -> None:
    """Write {parquet_stem}_sample.json with up to SAMPLE_ROW_COUNT rows."""
    sample_n = min(SAMPLE_ROW_COUNT, df.height)
    rows = df.head(sample_n).to_dicts() if sample_n else []
    sample_path = parquet_path.with_name(
        parquet_path.name.replace(".parquet", "_sample.json")
    )
    sample_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
