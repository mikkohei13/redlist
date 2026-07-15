"""
Resolve host names via FinBIF /taxa/search and add host_taxon_normalized to occurrences.parquet.

Requires FINBIF_ACCESS_TOKEN in the environment.
API responses are cached under cache/ (gitignored).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_io import ROOT, dataset_from_argv, require_file
from host_taxa import FinBifApiError, TOKEN_ENV, normalize_hosts_in_parquet

USAGE = "uv run scripts/normalize_host_taxa.py <dataset-slug>"


def main() -> None:
    ds = dataset_from_argv(usage=USAGE)
    occurrences_path = require_file(ds.path("occurrences"), label="parquet file")

    token = os.environ.get(TOKEN_ENV)
    if not token:
        print(f"error: missing environment variable {TOKEN_ENV}")
        sys.exit(1)

    try:
        normalize_hosts_in_parquet(
            occurrences_path,
            token=token,
            cache_dir=ROOT / "cache",
        )
    except FinBifApiError as exc:
        print(f"error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
