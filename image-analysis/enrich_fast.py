#!/usr/bin/env python3
"""Fast CLIP enrichment — concurrent fetching, short timeouts, incremental saves."""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CACHE_DIR = Path(".manifest_cache")
DATA_PATH = Path("data/trains.json")
FETCH_TIMEOUT = 10
WORKERS = 8
SAVE_EVERY = 100
TOP_TAGS = 4
TOP_SIMILAR = 3

TAG_PROMPTS = {
    "steam locomotive":    "a steam locomotive with visible smoke, steam, or smokestack",
    "diesel locomotive":   "a diesel locomotive or diesel-electric train engine",
    "electric locomotive": "an electric locomotive or electric train with overhead wires or third rail",
    "passenger train":     "a passenger train with passenger cars or coaches",
    "freight train":       "a freight train with boxcars, tankers, or flatcars",
    "yard / switching":    "a rail yard with multiple tracks, switching operations, or parked rolling stock",
    "station / depot":     "a railroad station, train depot, or platform where passengers board",
    "rail yard":           "a railroad yard with many tracks, signals, and parked trains",
    "open track":          "a train on open track through countryside, fields, or rural landscape",
    "urban / industrial":  "a train in an urban or industrial setting with buildings and infrastructure",
    "bridge / trestle":    "a railroad bridge, trestle, or viaduct",
    "vintage (pre-1920s)": "a very old photograph from the early 1900s or 19th century, sepia toned",
    "mid-century":         "a mid-20th century photograph from the 1930s to 1950s",
    "black and white":     "a black and white photograph with no color",
    "color photo":         "a color photograph",
    "close-up detail":     "a close-up or detail shot showing mechanical parts, wheels, or equipment",
    "photograph":          "a photographic image, a real photograph of a scene",
    "poster / illustration": "an illustrated poster, drawing, or graphic design, not a photograph",
}


def slugify(text):
    s = text.lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s.strip())
    return s[:80]


def fetch_manifest(url):
    slug = slugify(url[-60:])
    cache_path = CACHE_DIR / f"{slug}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass
    try:
        r = requests.get(url, timeout=FETCH_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        cache_path.write_text(json.dumps(data))
        return data
    except Exception:
        return None


def extract_image_url(manifest):
    try:
        seq = manifest.get("sequences", [])
        if seq:
            canvas = seq[0].get("canvases", [])[0]
            img = canvas.get("images", [])[0]
            res = img.get("resource", {})
            url = res.get("@id", "")
            if url:
                return url
            svc = res.get("service", {})
            svc_id = svc.get("@id", "") if isinstance(svc, dict) else ""
            if svc_id:
                return f"{svc_id}/full/800,/0/default.jpg"
    except (IndexError, KeyError, TypeError):
        pass
    thumb = manifest.get("thumbnail")
    if thumb:
        if isinstance(thumb, list):
            thumb = thumb[0]
        return thumb.get("@id", "")
    return ""


def fetch_image(url):
    try:
        r = requests.get(url, timeout=FETCH_TIMEOUT)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def fetch_item_image(item):
    """Fetch manifest + image for a single item. Returns (index, image) or (index, None)."""
    idx = item["_idx"]
    manifest = fetch_manifest(item["manifest"])
    if not manifest:
        return idx, None, ""
    image_url = extract_image_url(manifest)
    if not image_url:
        return idx, None, ""
    img = fetch_image(image_url)
    return idx, img, image_url


def main():
    CACHE_DIR.mkdir(exist_ok=True)

    print(f"Loading CLIP ({CLIP_MODEL_ID})...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    print("Model ready.\n")

    with open(DATA_PATH) as f:
        data = json.load(f)
    items = data["items"]

    prompts = list(TAG_PROMPTS.values())
    prompt_to_label = {v: k for k, v in TAG_PROMPTS.items()}

    # Phase 1: Tag from cached embeddings first (fast, no network)
    all_embeddings = [None] * len(items)
    needs_fetch = []

    for i, item in enumerate(items):
        item["id"] = slugify(item["title"])
        emb_path = CACHE_DIR / f"{item['id']}_emb.npy"
        if emb_path.exists():
            emb = np.load(str(emb_path))
            all_embeddings[i] = emb
            if "visual_tags" not in item:
                # Tag from cached embedding
                img_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
                inputs = processor(text=prompts, return_tensors="pt", padding=True)
                with torch.no_grad():
                    text_feats = model.get_text_features(**inputs)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                logits = (img_tensor @ text_feats.T) * model.logit_scale.exp()
                probs = logits[0].softmax(dim=-1).detach().cpu().numpy()
                ranked = sorted(zip(prompts, probs.tolist()), key=lambda x: -x[1])
                item["visual_tags"] = [prompt_to_label.get(p, p) for p, _ in ranked[:TOP_TAGS]]
        else:
            needs_fetch.append(i)

    cached_count = len(items) - len(needs_fetch)
    print(f"Cached embeddings: {cached_count}/{len(items)}")
    print(f"Need to fetch: {len(needs_fetch)}")

    # Precompute text features once for reuse
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_feats = model.get_text_features(**text_inputs)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    # Phase 2: Concurrent fetch + sequential CLIP inference
    tagged = 0
    skipped = 0
    batch = []

    for idx in needs_fetch:
        items[idx]["_idx"] = idx
    fetch_items = [items[idx] for idx in needs_fetch]

    print(f"\nFetching {len(fetch_items)} images with {WORKERS} workers...\n")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch_item_image, item): item for item in fetch_items}
        for future in as_completed(futures):
            idx, img, image_url = future.result()
            item = items[idx]
            if "_idx" in item:
                del item["_idx"]

            if image_url:
                item["image_url"] = image_url

            if img is None:
                skipped += 1
                continue

            # CLIP inference (sequential, on CPU)
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            emb = feats[0].cpu().numpy()

            # Tag
            img_tensor = feats
            logits = (img_tensor @ text_feats.T) * model.logit_scale.exp()
            probs = logits[0].softmax(dim=-1).detach().cpu().numpy()
            ranked = sorted(zip(prompts, probs.tolist()), key=lambda x: -x[1])
            item["visual_tags"] = [prompt_to_label.get(p, p) for p, _ in ranked[:TOP_TAGS]]

            # Cache embedding
            np.save(str(CACHE_DIR / f"{item['id']}_emb.npy"), emb)
            all_embeddings[idx] = emb

            tagged += 1
            total_done = cached_count + tagged
            if tagged % 10 == 0:
                print(f"  Tagged {tagged}/{len(needs_fetch)} (total: {total_done}/{len(items)}, skipped: {skipped})")

            # Incremental save
            if tagged % SAVE_EVERY == 0:
                data["items"] = items
                with open(DATA_PATH, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"  [saved checkpoint at {total_done} items]")

    # Clean up _idx from any skipped items
    for item in items:
        item.pop("_idx", None)

    print(f"\nPhase 1 complete: {cached_count + tagged} tagged, {skipped} skipped")

    # Phase 3: Visual similarity
    print("\nComputing visual similarity...")
    valid = [(i, e) for i, e in enumerate(all_embeddings) if e is not None]
    if len(valid) >= 2:
        valid_indices, valid_embs = zip(*valid)
        mat = np.array(list(valid_embs))
        sim = mat @ mat.T
        ids = [item["id"] for item in items]
        for rank, i in enumerate(valid_indices):
            row = sim[rank]
            sorted_j = sorted(enumerate(row), key=lambda x: -x[1])
            similar = [ids[valid_indices[j]] for j, _ in sorted_j if valid_indices[j] != i][:TOP_SIMILAR]
            items[i]["similar_items"] = similar

    # Final save
    data["items"] = items
    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)

    # Tag distribution
    tag_counts = {}
    for item in items:
        for tag in item.get("visual_tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    print("\n=== Tag Distribution ===")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {count}")

    total_tagged = sum(1 for i in items if i.get("visual_tags"))
    total_similar = sum(1 for i in items if i.get("similar_items"))
    print(f"\nDone. {total_tagged}/{len(items)} tagged, {total_similar} with similarity.")
    print(f"Written to {DATA_PATH}")


if __name__ == "__main__":
    main()
