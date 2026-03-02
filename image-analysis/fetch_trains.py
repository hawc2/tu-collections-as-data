#!/usr/bin/env python3
"""
fetch_trains.py — Fetch train-related items from Temple ContentDM collections.

Pulls from two sources:
  1. Frank G. Zahn Railroad Photograph Collection (p16002coll26) — all items
  2. Allied Posters of World War I (p16002coll9) — filtered for train/railroad content

Uses IIIF collection manifest pagination (from contentdm-iiif-api notebook)
to enumerate items, then fetches rich metadata via dmGetItemInfo.

Usage:
  python3 image-analysis/fetch_trains.py --output data/trains.json
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests


CONTENTDM_BASE = "https://cdm16002.contentdm.oclc.org"
IIIF_BASE      = f"{CONTENTDM_BASE}/iiif"
TEMPLE_DIGITAL = "https://digital.library.temple.edu/digital"
REQUEST_DELAY  = 0.4

# Keywords that indicate train/railroad content in war posters
TRAIN_KEYWORDS = re.compile(
    r"railroad|railway|train|locomotive|rail road|freight car|box car|"
    r"rolling stock|pullman|engine|caboose|depot|station.*rail",
    re.IGNORECASE,
)


def get_all_manifest_urls(collection: str) -> list:
    """Get all item manifest URLs via IIIF collection pagination.

    Uses the same approach as the CDM_IIIF_Image_Download notebook:
    fetch the collection manifest, follow first/next page links.
    """
    coll_url = f"{IIIF_BASE}/{collection}/manifest.json"
    try:
        r = requests.get(coll_url, timeout=30)
        r.raise_for_status()
        coll = r.json()
    except Exception as e:
        print(f"  [ERROR] Failed to fetch collection manifest: {e}")
        return []

    page_url = coll.get("first")
    if isinstance(page_url, dict):
        page_url = page_url["@id"]

    all_urls = []
    page_num = 0
    while page_url:
        page_num += 1
        try:
            r = requests.get(page_url, timeout=30)
            r.raise_for_status()
            page = r.json()
        except Exception as e:
            print(f"  [WARN] Failed to fetch page {page_num}: {e}")
            break

        manifests = page.get("manifests", [])
        if not manifests:
            break

        for m in manifests:
            all_urls.append(m["@id"])

        print(f"  Page {page_num}: {len(manifests)} items (total: {len(all_urls)})")
        page_url = page.get("next", None)

    return all_urls


def pointer_from_manifest_url(url: str) -> int:
    """Extract the ContentDM pointer from an IIIF manifest URL.

    e.g. '.../iiif/p16002coll26:148/manifest.json' -> 148
    """
    match = re.search(r":(\d+)/manifest", url)
    if match:
        return int(match.group(1))
    return -1


def fetch_item_info(collection: str, pointer: int) -> dict:
    """Fetch full metadata for a single item via dmGetItemInfo."""
    url = (
        f"{CONTENTDM_BASE}/digital/bl/dmwebservices/index.php"
        f"?q=dmGetItemInfo/{collection}/{pointer}/json"
    )
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                print(f"    [WARN] dmGetItemInfo failed for {collection}/{pointer}: {e}")
    return {}


def parse_subjects(raw: str) -> list:
    """Split a ContentDM semicolon-delimited subjects string into a list."""
    if not raw or not isinstance(raw, str):
        return []
    return [s.strip().rstrip(";").strip() for s in raw.split(";") if s.strip().rstrip(";").strip()]


def has_train_content(info: dict) -> bool:
    """Check whether a ContentDM item record mentions trains/railroads."""
    fields_to_check = [
        info.get("title", ""),
        info.get("descri", ""),
        info.get("subjec", ""),
        info.get("notes", ""),
    ]
    text = " ".join(str(f) for f in fields_to_check if f)
    return bool(TRAIN_KEYWORDS.search(text))


def build_item(collection: str, pointer: int, info: dict, source_label: str) -> dict:
    """Build a trains item dict from ContentDM metadata."""
    title = info.get("title", "").strip() or f"Untitled ({collection}/{pointer})"
    item = {
        "title":             title,
        "featured":          False,
        "source_collection": source_label,
        "record":            f"{TEMPLE_DIGITAL}/collection/{collection}/id/{pointer}",
        "manifest":          f"{CONTENTDM_BASE}/iiif/{collection}:{pointer}/manifest.json",
        "date":              info.get("date", "Undated").strip() or "Undated",
        "subjects":          parse_subjects(info.get("subjec", "")),
        "description":       (info.get("descri", "") or "").strip(),
    }
    photographer = (info.get("creato", "") or "").strip()
    if photographer:
        item["photographer"] = photographer
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="Path to trains.json")
    args = parser.parse_args()

    out_path = Path(args.output)
    if out_path.exists():
        with open(out_path) as f:
            data = json.load(f)
    else:
        data = {
            "collection": {
                "title": "Trains at Temple",
                "description": "Train and railroad imagery drawn from Temple University Libraries' digital collections, combining the Frank G. Zahn Railroad Photograph Collection with train-related war posters.",
                "source": "Temple University Libraries, Digital Collections",
            },
            "items": [],
        }

    existing_manifests = {item["manifest"] for item in data.get("items", [])}
    new_items = []

    # --- Collection 1: Railroad Photographs (all items via IIIF pagination) ---
    railroad_coll = "p16002coll26"
    print(f"\n=== Railroad Photographs ({railroad_coll}) ===")
    print("  Enumerating items via IIIF collection manifest...")
    manifest_urls = get_all_manifest_urls(railroad_coll)
    print(f"  Found {len(manifest_urls)} items total")

    skipped = 0
    for i, murl in enumerate(manifest_urls):
        if murl in existing_manifests:
            skipped += 1
            continue
        ptr = pointer_from_manifest_url(murl)
        if ptr < 0:
            continue
        print(f"  [{i+1}/{len(manifest_urls)}] Fetching {railroad_coll}/{ptr}...")
        info = fetch_item_info(railroad_coll, ptr)
        time.sleep(REQUEST_DELAY)
        if not info or info.get("code") == -2:
            print(f"    [SKIP] not found")
            continue
        new_items.append(build_item(railroad_coll, ptr, info, "railroad"))

    print(f"  Skipped {skipped} already-fetched items")

    # --- Collection 2: War Posters — train-related only ---
    posters_coll = "p16002coll9"
    print(f"\n=== War Posters — train filter ({posters_coll}) ===")
    print("  Enumerating items via IIIF collection manifest...")
    manifest_urls = get_all_manifest_urls(posters_coll)
    print(f"  Found {len(manifest_urls)} total items, filtering for train content...")

    train_poster_count = 0
    for murl in manifest_urls:
        if murl in existing_manifests:
            continue
        ptr = pointer_from_manifest_url(murl)
        if ptr < 0:
            continue
        info = fetch_item_info(posters_coll, ptr)
        time.sleep(REQUEST_DELAY)
        if not info or info.get("code") == -2:
            continue
        if not has_train_content(info):
            continue
        print(f"  [MATCH] {info.get('title', '?')}")
        new_items.append(build_item(posters_coll, ptr, info, "warposters"))
        train_poster_count += 1

    print(f"\n  Train-related posters found: {train_poster_count}")

    # Clean None values
    for item in new_items:
        for k in list(item.keys()):
            if item[k] is None:
                del item[k]

    existing_items = data.get("items", [])
    data["items"] = existing_items + new_items

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nExisting items: {len(existing_items)}")
    print(f"New items: {len(new_items)}")
    print(f"Total: {len(data['items'])}")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
