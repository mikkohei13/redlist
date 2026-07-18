"""
FinBIF /taxa/search and /taxa/{id} lookups for host names. Responses cached under cache/.
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

TAXA_SEARCH_URL = "https://api.laji.fi/taxa/search"
TAXA_URL = "https://api.laji.fi/taxa"
TOKEN_ENV = "FINBIF_ACCESS_TOKEN"
RATE_LIMIT_SECONDS = 0.1

_API_HEADERS = {
    "accept": "application/json",
    "Accept-Language": "fi",
    "API-Version": "1",
}


class FinBifApiError(RuntimeError):
    pass


def cache_path(key: str, cache_dir: Path) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def _get_json(url: str, token: str, *, label: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={**_API_HEADERS, "Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise FinBifApiError(
            f"FinBIF API HTTP {exc.code} for {label}: {body}"
        ) from exc

    if "errorCode" in data:
        raise FinBifApiError(
            f"FinBIF API {data['errorCode']} for {label}: "
            f"{data.get('message', '')}"
        )
    return data


def _cached_get(
    cache_key: str,
    url: str,
    token: str,
    *,
    cache_dir: Path,
    label: str,
) -> tuple[dict, bool]:
    """Return (response_json, from_cache)."""
    path = cache_path(cache_key, cache_dir)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8")), True

    data = _get_json(url, token, label=label)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    time.sleep(RATE_LIMIT_SECONDS)
    return data, False


def fetch_taxa_search(query: str, token: str, *, cache_dir: Path) -> tuple[dict, bool]:
    """Return (response_json, from_cache)."""
    params = urllib.parse.urlencode(
        {"query": query, "limit": 10, "includeHidden": "false"}
    )
    return _cached_get(
        query,
        f"{TAXA_SEARCH_URL}?{params}",
        token,
        cache_dir=cache_dir,
        label=f"query {query!r}",
    )


def fetch_taxon(taxon_id: str, token: str, *, cache_dir: Path) -> tuple[dict, bool]:
    """Return (response_json, from_cache) for GET /taxa/{id}."""
    params = urllib.parse.urlencode(
        {
            "includeMedia": "false",
            "includeDescriptions": "false",
            "includeRedListEvaluations": "true",
            "checklistVersion": "current",
        }
    )
    return _cached_get(
        f"taxon:{taxon_id}",
        f"{TAXA_URL}/{urllib.parse.quote(taxon_id, safe='')}?{params}",
        token,
        cache_dir=cache_dir,
        label=f"taxon {taxon_id!r}",
    )


GENUS_RANKS = frozenset(
    {"MX.species", "MX.aggregate", "MX.subgenus", "MX.genus"}
)


def host_fields_from_response(data: dict) -> dict[str, str | None]:
    for hit in data.get("results", []):
        if hit.get("type") != "exactMatches":
            continue

        scientific_name = hit.get("scientificName")
        vernacular_name = hit.get("vernacularName")
        taxon_rank = hit.get("taxonRank")

        host_taxon_normalized = None
        if scientific_name and vernacular_name:
            host_taxon_normalized = f"{scientific_name} - {vernacular_name}"

        host_species = (
            scientific_name if taxon_rank == "MX.species" and scientific_name else None
        )

        host_genus = None
        if taxon_rank in GENUS_RANKS and scientific_name:
            host_genus = scientific_name.split(" ", 1)[0]

        return {
            "host_taxon_normalized": host_taxon_normalized,
            "host_taxon_id": hit.get("id"),
            "host_species": host_species,
            "host_genus": host_genus,
        }

    return {
        "host_taxon_normalized": None,
        "host_taxon_id": None,
        "host_species": None,
        "host_genus": None,
    }


def family_fields_from_taxon(data: dict) -> dict[str, str | None]:
    family = (data.get("parent") or {}).get("family") or {}
    return {
        "host_family": family.get("scientificName"),
        "host_family_id": family.get("id"),
    }


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
            "rows_with_host_taxon_normalized": 0,
        }

    print(f"  {len(hosts)} distinct host values")

    normalized_by_host: dict[str, dict[str, str | None]] = {}
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

        fields = host_fields_from_response(response)
        normalized_by_host[host] = fields
        if fields["host_taxon_normalized"] is not None:
            exact_matches += 1
            print(
                f"  [{i}/{len(hosts)}] {host!r} ({source}) -> "
                f"{fields['host_taxon_normalized']}"
            )
        else:
            print(f"  [{i}/{len(hosts)}] {host!r} ({source}) -> no exact match")

    taxon_ids = sorted(
        {
            fields["host_taxon_id"]
            for fields in normalized_by_host.values()
            if fields["host_taxon_id"]
        }
    )
    print(f"  resolving family for {len(taxon_ids)} distinct host taxa")

    family_by_taxon_id: dict[str, dict[str, str | None]] = {}
    for i, taxon_id in enumerate(taxon_ids, start=1):
        response, from_cache = fetch_taxon(taxon_id, token, cache_dir=cache_dir)
        if from_cache:
            cache_hits += 1
            source = "cache"
        else:
            api_calls += 1
            source = "API"

        fields = family_fields_from_taxon(response)
        family_by_taxon_id[taxon_id] = fields
        if fields["host_family"] is not None:
            print(
                f"  [{i}/{len(taxon_ids)}] {taxon_id} ({source}) -> "
                f"{fields['host_family']}"
            )
        else:
            print(f"  [{i}/{len(taxon_ids)}] {taxon_id} ({source}) -> no family")

    for fields in normalized_by_host.values():
        taxon_id = fields["host_taxon_id"]
        if taxon_id and taxon_id in family_by_taxon_id:
            fields.update(family_by_taxon_id[taxon_id])
        else:
            fields["host_family"] = None
            fields["host_family_id"] = None

    mapping = pl.DataFrame(
        {
            "host": list(normalized_by_host.keys()),
            "host_taxon_normalized": [
                fields["host_taxon_normalized"]
                for fields in normalized_by_host.values()
            ],
            "host_taxon_id": [
                fields["host_taxon_id"] for fields in normalized_by_host.values()
            ],
            "host_species": [
                fields["host_species"] for fields in normalized_by_host.values()
            ],
            "host_genus": [
                fields["host_genus"] for fields in normalized_by_host.values()
            ],
            "host_family": [
                fields["host_family"] for fields in normalized_by_host.values()
            ],
            "host_family_id": [
                fields["host_family_id"] for fields in normalized_by_host.values()
            ],
        }
    )

    drop_cols = [
        c
        for c in (
            "host_taxon_normalized",
            "host_taxon_id",
            "host_species",
            "host_genus",
            "host_family",
            "host_family_id",
            "taxon_normalized",
            "species_scientific",
            "genus_scientific",
        )
        if c in occurrences.columns
    ]
    if drop_cols:
        occurrences = occurrences.drop(drop_cols)

    occurrences = occurrences.join(mapping, on="host", how="left")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    occurrences.write_parquet(parquet_path, compression="zstd", statistics=True)

    rows_with_normalized = occurrences.filter(
        pl.col("host_taxon_normalized").is_not_null()
    ).height
    print(
        f"  done: {cache_hits} cache hits, {api_calls} API calls, "
        f"{exact_matches} exact matches, "
        f"{rows_with_normalized} rows with host_taxon_normalized"
    )

    return {
        "distinct_hosts": len(hosts),
        "cache_hits": cache_hits,
        "api_calls": api_calls,
        "exact_matches": exact_matches,
        "rows_with_host_taxon_normalized": rows_with_normalized,
    }
