#!/usr/bin/env python3
"""
Scan Temple History in Photographs for train/railroad content.
Compares metadata keyword matches vs CLIP visual detection.
"""

import json
import sys
import time
from io import BytesIO
from pathlib import Path

import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CONTENTDM_BASE = "https://cdm16002.contentdm.oclc.org"
TEMPLE_DIGITAL = "https://digital.library.temple.edu/digital"
COLL = "p245801coll0"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

TRAIN_KEYWORDS = ["railroad", "railway", "train", "locomotive", "rail yard",
                   "rail line", "trolley", "elevated line", "freight",
                   "rail road", "depot", "station"]

TRAIN_PROMPTS = [
    "a train, locomotive, railroad tracks, railway, or trolley",
    "a photograph that does not contain trains, railroads, or railway tracks",
]

THRESHOLD = 0.55

OUTPUT_PATH = Path("image-analysis/temple_history_train_results.json")


def get_all_items():
    url = (
        f"{CONTENTDM_BASE}/digital/bl/dmwebservices/index.php"
        f"?q=dmQuery/{COLL}/0/pointer!title!descri!subjec/pointer/1500/0/1/0/0/0/json"
    )
    r = requests.get(url, timeout=60)
    recs = r.json().get("records", [])
    items = []
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
        items.append({
            "pointer": int(ptr),
            "title": str(rec.get("title", "")),
            "description": str(rec.get("descri", "")),
            "subjects": str(rec.get("subjec", "")),
        })
    return items


def has_train_metadata(item):
    text = f"{item['title']} {item['description']} {item['subjects']}".lower()
    return any(kw in text for kw in TRAIN_KEYWORDS)


def fetch_image_from_manifest(pointer):
    manifest_url = f"{CONTENTDM_BASE}/iiif/{COLL}:{pointer}/manifest.json"
    try:
        r = requests.get(manifest_url, timeout=20)
        r.raise_for_status()
        m = r.json()
        seq = m.get("sequences", [])
        if seq:
            canvas = seq[0].get("canvases", [])[0]
            img = canvas.get("images", [])[0]
            res = img.get("resource", {})
            url = res.get("@id", "")
            if not url:
                svc = res.get("service", {})
                svc_id = svc.get("@id", "") if isinstance(svc, dict) else ""
                if svc_id:
                    url = f"{svc_id}/full/400,/0/default.jpg"
            if url:
                if "/full/" in url:
                    url = url.replace("/full/800,/", "/full/400,/").replace("/full/full/", "/full/400,/")
                r2 = requests.get(url, timeout=25)
                r2.raise_for_status()
                return Image.open(BytesIO(r2.content)).convert("RGB")
        thumb = m.get("thumbnail")
        if thumb:
            if isinstance(thumb, list):
                thumb = thumb[0]
            url = thumb.get("@id", "")
            if url:
                r2 = requests.get(url, timeout=25)
                r2.raise_for_status()
                return Image.open(BytesIO(r2.content)).convert("RGB")
    except Exception as e:
        print(f"  [WARN] {pointer}: {e}", flush=True)
    return None


def get_full_metadata(pointer):
    url = (
        f"{CONTENTDM_BASE}/digital/bl/dmwebservices/index.php"
        f"?q=dmGetItemInfo/{COLL}/{pointer}/json"
    )
    try:
        return requests.get(url, timeout=15).json()
    except Exception:
        return {}


def main():
    print(f"Loading CLIP ({CLIP_MODEL_ID})...", flush=True)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    print("Model ready.\n", flush=True)

    print("Fetching Temple History item list...", flush=True)
    items = get_all_items()
    print(f"Found {len(items)} items.\n", flush=True)

    metadata_matches = set()
    for item in items:
        if has_train_metadata(item):
            metadata_matches.add(item["pointer"])

    print(f"Metadata keyword matches: {len(metadata_matches)}", flush=True)
    print("Now scanning ALL items with CLIP...\n", flush=True)

    clip_matches = {}
    for i, item in enumerate(items):
        ptr = item["pointer"]

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(items)} scanned, "
                  f"{len(clip_matches)} CLIP matches so far", flush=True)

        img = fetch_image_from_manifest(ptr)
        if img is None:
            continue

        inputs = processor(text=TRAIN_PROMPTS, images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits_per_image[0]
        probs = logits.softmax(dim=-1).cpu().numpy()
        train_prob = float(probs[0])

        if train_prob >= THRESHOLD:
            in_metadata = ptr in metadata_matches
            label = "BOTH" if in_metadata else "CLIP-ONLY"
            print(f"  {label} ({train_prob:.2f}): {item['title']}", flush=True)
            clip_matches[ptr] = train_prob

        time.sleep(0.3)

    metadata_only = metadata_matches - set(clip_matches.keys())
    clip_only = set(clip_matches.keys()) - metadata_matches
    both = metadata_matches & set(clip_matches.keys())

    print(f"\n{'='*60}", flush=True)
    print(f"COMPARISON RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total items scanned:    {len(items)}", flush=True)
    print(f"Metadata matches:       {len(metadata_matches)}", flush=True)
    print(f"CLIP matches:           {len(clip_matches)}", flush=True)
    print(f"Both (overlap):         {len(both)}", flush=True)
    print(f"Metadata only:          {len(metadata_only)}", flush=True)
    print(f"CLIP only (new finds):  {len(clip_only)}", flush=True)
    print(f"{'='*60}\n", flush=True)

    all_match_ptrs = metadata_matches | set(clip_matches.keys())
    results = []
    for ptr in sorted(all_match_ptrs):
        info = get_full_metadata(ptr)
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
            "source_collection": "templehistory",
            "record": f"{TEMPLE_DIGITAL}/collection/{COLL}/id/{ptr}",
            "manifest": f"{CONTENTDM_BASE}/iiif/{COLL}:{ptr}/manifest.json",
            "image_url": f"{CONTENTDM_BASE}/iiif/2/{COLL}:{ptr}/full/full/0/default.jpg",
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
        time.sleep(0.3)

    output = {
        "collection": {
            "title": "Temple History in Photographs",
            "collection_id": COLL,
            "total_scanned": len(items),
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

    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Results saved to {OUTPUT_PATH}", flush=True)

    if clip_only:
        print(f"\n=== CLIP-ONLY DISCOVERIES ===", flush=True)
        for item in results:
            if item["found_by"] == ["clip"]:
                print(f"  ({item['train_confidence']:.2f}) {item['title']}", flush=True)
                print(f"    {item['description'][:120]}", flush=True)


if __name__ == "__main__":
    main()
