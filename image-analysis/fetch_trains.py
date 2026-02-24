#!/usr/bin/env python3
"""
fetch_trains.py — Fetch train-related items from Temple ContentDM collections.

Pulls from two sources:
  1. Frank G. Zahn Railroad Photograph Collection (p16002coll26) — all items
  2. Allied Posters of World War I (p16002coll9) — filtered for train/railroad content

Outputs a trains.json with the same schema used by enrich_trains.py.

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
TEMPLE_DIGITAL  = "https://digital.library.temple.edu/digital"
REQUEST_DELAY   = 0.4

# Keywords that indicate train/railroad content in war posters
TRAIN_KEYWORDS = re.compile(
    r"railroad|railway|train|locomotive|rail road|freight car|box car|"
    r"rolling stock|pullman|engine|caboose|depot|station.*rail",
    re.IGNORECASE,
)


def get_all_pointers(collection: str) -> list:
    """Return all item pointers in a ContentDM collection."""
    url = (
        f"{CONTENTDM_BASE}/digital/bl/dmwebservices/index.php"
        f"?q=dmQuery/{collection}/0/pointer!descri/pointer/1000/0/1/0/0/0/json"
    )
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        d = r.json()
        recs = d.get("records", [])
        pointers = []
        for rec in recs:
            if rec.get("parentobject", -1) != -1:
                continue
            ptr = rec.get("pointer")
            if ptr != "" and ptr is not None:
                pointers.append(int(ptr))
                continue
            find = rec.get("find", "")
            if find:
                try:
                    pointers.append(int(find.split(".")[0]) - 1)
                except ValueError:
                    pass
        return pointers
    except Exception as e:
        print(f"  [WARN] failed to list {collection}: {e}")
        return []


def fetch_item_info(collection: str, pointer: int) -> dict:
    """Fetch full metadata for a single item via dmGetItemInfo."""
    url = (
        f"{CONTENTDM_BASE}/digital/bl/dmwebservices/index.php"
        f"?q=dmGetItemInfo/{collection}/{pointer}/json"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
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

    # --- Collection 1: Railroad Photographs (all items) ---
    railroad_coll = "p16002coll26"
    print(f"\n=== Railroad Photographs ({railroad_coll}) ===")
    print("  Fetching pointer list...")
    pointers = get_all_pointers(railroad_coll)
    print(f"  Found {len(pointers)} items")

    for ptr in pointers:
        manifest_url = f"{CONTENTDM_BASE}/iiif/{railroad_coll}:{ptr}/manifest.json"
        if manifest_url in existing_manifests:
            continue
        print(f"  Fetching {railroad_coll}/{ptr}...")
        info = fetch_item_info(railroad_coll, ptr)
        time.sleep(REQUEST_DELAY)
        if not info or info.get("code") == -2:
            print(f"    [SKIP] not found")
            continue
        new_items.append(build_item(railroad_coll, ptr, info, "railroad"))

    # --- Collection 2: War Posters — train-related only ---
    posters_coll = "p16002coll9"
    print(f"\n=== War Posters — train filter ({posters_coll}) ===")
    print("  Fetching pointer list...")
    pointers = get_all_pointers(posters_coll)
    print(f"  Found {len(pointers)} total items, filtering for train content...")

    train_poster_count = 0
    for ptr in pointers:
        manifest_url = f"{CONTENTDM_BASE}/iiif/{posters_coll}:{ptr}/manifest.json"
        if manifest_url in existing_manifests:
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
