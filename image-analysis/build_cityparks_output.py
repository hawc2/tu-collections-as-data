#!/usr/bin/env python3
"""Build cityparks_train_results.json from the scan log data."""

import json
import requests
import time

CONTENTDM_BASE = "https://cdm16002.contentdm.oclc.org"
TEMPLE_DIGITAL = "https://digital.library.temple.edu/digital"
COLL = "p15037coll5"

TRAIN_KEYWORDS = ["railroad", "railway", "train", "locomotive", "rail yard",
                   "rail line", "trolley", "elevated line", "freight"]

# CLIP matches from scan log
CLIP_TITLES = {
    "Freight lines crossing the passenger rail lines and Market-Frankford El at 31st Street": 0.61,
    "Train tracks on West Chester Pike": 0.87,
    "Road leading to the Spring Garden Bridge": 0.74,
    "Bus at Delaware and Oregon Avenues": 0.80,
    "A residential tree-lined street in Bornville": 0.79,
    "Grain elevator and railroad tracks": 0.60,
    "Filbert Street": 0.60,
    "Trolley tracks along unidentified road": 0.77,
    "Awbury Arboretum": 0.57,
    "Billboards on Southern Boulevard": 0.61,
    "First National Bank, Lake Forest": 0.63,
    "Property at Fourth snd Shunk Streets": 0.57,
    "Yorkship Village construction": 0.74,
    "Yorkship Village street construction": 0.64,
}


def main():
    print("Fetching item list...")
    url = (
        f"{CONTENTDM_BASE}/digital/bl/dmwebservices/index.php"
        f"?q=dmQuery/{COLL}/0/pointer!title!descri!subjec/pointer/1500/0/1/0/0/0/json"
    )
    r = requests.get(url, timeout=60)
    recs = r.json().get("records", [])

    items_by_ptr = {}
    metadata_matches = set()
    for rec in recs:
        if rec.get("parentobject", -1) != -1:
            continue
        ptr = rec.get("pointer")
        if ptr == "" or ptr is None:
            find = rec.get("find", "")
            if find:
                try:
                    ptr = int(find.split(".")[0]) - 1
                except ValueError:
                    continue
            else:
                continue
        ptr = int(ptr)
        title = str(rec.get("title", ""))
        descr = str(rec.get("descri", ""))
        subj = str(rec.get("subjec", ""))
        items_by_ptr[ptr] = {"title": title, "description": descr, "subjects": subj}
        text = f"{title} {descr} {subj}".lower()
        if any(kw in text for kw in TRAIN_KEYWORDS):
            metadata_matches.add(ptr)

    # Find pointers for CLIP matches
    clip_matches = {}
    for ptr, info in items_by_ptr.items():
        if info["title"] in CLIP_TITLES:
            clip_matches[ptr] = CLIP_TITLES[info["title"]]

    clip_only = set(clip_matches.keys()) - metadata_matches
    both = metadata_matches & set(clip_matches.keys())
    metadata_only = metadata_matches - set(clip_matches.keys())

    print(f"Metadata matches: {len(metadata_matches)}")
    print(f"CLIP matches: {len(clip_matches)}")
    print(f"Both: {len(both)}")
    print(f"Metadata only: {len(metadata_only)}")
    print(f"CLIP only: {len(clip_only)}")

    # Build results for all matches
    all_ptrs = metadata_matches | set(clip_matches.keys())
    results = []
    for ptr in sorted(all_ptrs):
        info_url = (
            f"{CONTENTDM_BASE}/digital/bl/dmwebservices/index.php"
            f"?q=dmGetItemInfo/{COLL}/{ptr}/json"
        )
        try:
            info = requests.get(info_url, timeout=15).json()
        except Exception:
            info = {}

        raw_subj = info.get("subjec", "")
        if not isinstance(raw_subj, str):
            raw_subj = ""
        subjects = [s.strip().rstrip(";").strip()
                    for s in raw_subj.split(";")
                    if s.strip().rstrip(";").strip()]
        title = info.get("title", "").strip() or f"Untitled ({COLL}/{ptr})"

        found_by = []
        if ptr in metadata_matches:
            found_by.append("metadata")
        if ptr in clip_matches:
            found_by.append("clip")

        item = {
            "title": title,
            "featured": False,
            "source_collection": "cityparks",
            "record": f"{TEMPLE_DIGITAL}/collection/{COLL}/id/{ptr}",
            "manifest": f"{CONTENTDM_BASE}/iiif/{COLL}:{ptr}/manifest.json",
            "date": (info.get("date", "") or "Undated").strip() or "Undated",
            "subjects": subjects,
            "description": (info.get("descri", "") or "").strip(),
            "found_by": found_by,
            "train_confidence": clip_matches.get(ptr, None),
        }
        creator = (info.get("creato", "") or "").strip()
        if creator:
            item["photographer"] = creator
        results.append(item)
        time.sleep(0.2)

    output = {
        "collection": {
            "title": "City Parks Association Photographs",
            "collection_id": COLL,
            "total_scanned": len(items_by_ptr),
        },
        "summary": {
            "metadata_matches": len(metadata_matches),
            "clip_matches": len(clip_matches),
            "overlap": len(both),
            "metadata_only": len(metadata_only),
            "clip_only": len(clip_only),
        },
        "items": results,
    }

    with open("image-analysis/cityparks_train_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(results)} items to cityparks_train_results.json")

    print("\nCLIP-ONLY DISCOVERIES:")
    for item in results:
        if item["found_by"] == ["clip"]:
            print(f"  ({item['train_confidence']:.2f}) {item['title']}")
            print(f"    {item['description'][:120]}")


if __name__ == "__main__":
    main()
