#!/usr/bin/env python3
"""
Scan ALL war posters via CLIP to find train/railroad imagery.
Outputs a JSON file of matches with confidence scores.
"""

import json
import time
from io import BytesIO
from pathlib import Path

import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CONTENTDM_BASE = "https://cdm16002.contentdm.oclc.org"
TEMPLE_DIGITAL = "https://digital.library.temple.edu/digital"
COLL = "p16002coll9"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# Binary classification: is this train-related or not?
TRAIN_PROMPTS = [
    "a train, locomotive, railroad, or railway",
    "a poster or illustration that does not depict trains or railroads",
]

# Threshold for the "train" class probability
THRESHOLD = 0.55

OUTPUT_PATH = Path("image-analysis/warposters_train_matches.json")


def get_all_pointers():
    url = (
        f"{CONTENTDM_BASE}/digital/bl/dmwebservices/index.php"
        f"?q=dmQuery/{COLL}/0/pointer!title!descri!subjec/pointer/1200/0/1/0/0/0/json"
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
            "title": rec.get("title", ""),
            "description": str(rec.get("descri", "")),
            "subjects": str(rec.get("subjec", "")),
        })
    return items


def fetch_image_from_manifest(pointer):
    manifest_url = f"{CONTENTDM_BASE}/iiif/{COLL}:{pointer}/manifest.json"
    try:
        r = requests.get(manifest_url, timeout=20)
        r.raise_for_status()
        m = r.json()
        # Try sequences -> canvases -> images -> resource
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
                # Use smaller size for speed
                if "/full/" in url:
                    url = url.replace("/full/800,/", "/full/400,/")
                    url = url.replace("/full/full/", "/full/400,/")
                r2 = requests.get(url, timeout=25)
                r2.raise_for_status()
                return Image.open(BytesIO(r2.content)).convert("RGB")
        # Fallback to thumbnail
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
        print(f"  [WARN] {pointer}: {e}")
    return None


def main():
    print(f"Loading CLIP ({CLIP_MODEL_ID})...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    print("Model ready.\n")

    print("Fetching war poster list...")
    items = get_all_pointers()
    print(f"Found {len(items)} posters. Scanning with CLIP...\n")

    matches = []
    for i, item in enumerate(items):
        ptr = item["pointer"]
        title = item["title"] or f"(ptr {ptr})"

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(items)} scanned, {len(matches)} matches so far")

        img = fetch_image_from_manifest(ptr)
        if img is None:
            continue

        inputs = processor(text=TRAIN_PROMPTS, images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits_per_image[0]
        probs = logits.softmax(dim=-1).cpu().numpy()
        train_prob = float(probs[0])

        if train_prob >= THRESHOLD:
            print(f"  MATCH ({train_prob:.2f}): {title}")
            # Get full metadata
            info_url = (
                f"{CONTENTDM_BASE}/digital/bl/dmwebservices/index.php"
                f"?q=dmGetItemInfo/{COLL}/{ptr}/json"
            )
            try:
                info = requests.get(info_url, timeout=15).json()
            except Exception:
                info = {}

            subjects = [s.strip().rstrip(";").strip()
                        for s in info.get("subjec", "").split(";")
                        if s.strip().rstrip(";").strip()]

            matches.append({
                "title": info.get("title", title).strip(),
                "featured": False,
                "source_collection": "warposters",
                "record": f"{TEMPLE_DIGITAL}/collection/{COLL}/id/{ptr}",
                "manifest": f"{CONTENTDM_BASE}/iiif/{COLL}:{ptr}/manifest.json",
                "date": (info.get("date", "") or "Undated").strip() or "Undated",
                "subjects": subjects,
                "description": (info.get("descri", "") or "").strip(),
                "train_confidence": train_prob,
            })
            creator = (info.get("creato", "") or "").strip()
            if creator:
                matches[-1]["photographer"] = creator

        time.sleep(0.3)

    # Save matches
    OUTPUT_PATH.write_text(json.dumps(matches, indent=2))
    print(f"\nDone. {len(matches)} train-related posters found.")
    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
