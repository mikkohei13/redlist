"""
FinBIF /taxa/search lookups for host names. Responses cached under cache/.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl

API_URL = "https://api.laji.fi/taxa/search"
TOKEN_ENV = "FINBIF_ACCESS_TOKEN"
RATE_LIMIT_SECONDS = 0.1


class FinBifApiError(RuntimeError):
    pass


def cache_path(query: str, cache_dir: Path) -> Path:
    key = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.json"


def fetch_taxa_search(query: str, token: str, *, cache_dir: Path) -> tuple[dict, bool]:
    """Return (response_json, from_cache)."""
    path = cache_path(query, cache_dir)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8")), True

    params = urllib.parse.urlencode(
        {"query": query, "limit": 10, "includeHidden": "false"}
    )
    req = urllib.request.Request(
        f"{API_URL}?{params}",
        headers={
            "accept": "application/json",
            "Authorization": f"Bearer {token}",
            "Accept-Language": "fi",
            "API-Version": "1",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise FinBifApiError(
            f"FinBIF API HTTP {exc.code} for query {query!r}: {body}"
        ) from exc

    if "errorCode" in data:
        raise FinBifApiError(
            f"FinBIF API {data['errorCode']} for query {query!r}: "
            f"{data.get('message', '')}"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    time.sleep(RATE_LIMIT_SECONDS)
    return data, False


def taxon_normalized_from_response(data: dict) -> str | None:
    for hit in data.get("results", []):
        if hit.get("type") != "exactMatches":
            continue
        scientific_name = hit.get("scientificName")
        vernacular_name = hit.get("vernacularName")
        if scientific_name and vernacular_name:
            return f"{scientific_name} - {vernacular_name}"
    return None


def normalize_hosts_in_parquet(
    parquet_path: Path,
    *,
    token: str,
    cache_dir: Path,
) -> dict[str, int]:
    print(f"normalizing host names via FinBIF API: {parquet_path}")

    occurrences = pl.read_parquet(parquet_path)
    if "host" not in occurrences.columns:
        raise ValueError("occurrences.parquet has no host column")

    hosts = (
        occurrences.filter(pl.col("host").is_not_null())
        .select("host")
        .unique()
        .sort("host")
        .to_series()
        .to_list()
    )

    if not hosts:
        print("  no host values to normalize")
        return {
            "distinct_hosts": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "exact_matches": 0,
            "rows_with_taxon_normalized": 0,
        }

    print(f"  {len(hosts)} distinct host values")

    normalized_by_host: dict[str, str | None] = {}
    cache_hits = 0
    api_calls = 0
    exact_matches = 0

    for i, host in enumerate(hosts, start=1):
        response, from_cache = fetch_taxa_search(host, token, cache_dir=cache_dir)
        if from_cache:
            cache_hits += 1
            source = "cache"
        else:
            api_calls += 1
            source = "API"

        normalized = taxon_normalized_from_response(response)
        normalized_by_host[host] = normalized
        if normalized is not None:
            exact_matches += 1
            print(f"  [{i}/{len(hosts)}] {host!r} ({source}) -> {normalized}")
        else:
            print(f"  [{i}/{len(hosts)}] {host!r} ({source}) -> no exact match")

    mapping = pl.DataFrame(
        {
            "host": list(normalized_by_host.keys()),
            "taxon_normalized": list(normalized_by_host.values()),
        }
    )

    if "taxon_normalized" in occurrences.columns:
        occurrences = occurrences.drop("taxon_normalized")

    occurrences = occurrences.join(mapping, on="host", how="left")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    occurrences.write_parquet(parquet_path, compression="zstd", statistics=True)

    rows_with_normalized = occurrences.filter(
        pl.col("taxon_normalized").is_not_null()
    ).height
    print(
        f"  done: {cache_hits} cache hits, {api_calls} API calls, "
        f"{exact_matches} exact matches, {rows_with_normalized} rows with taxon_normalized"
    )

    return {
        "distinct_hosts": len(hosts),
        "cache_hits": cache_hits,
        "api_calls": api_calls,
        "exact_matches": exact_matches,
        "rows_with_taxon_normalized": rows_with_normalized,
    }
