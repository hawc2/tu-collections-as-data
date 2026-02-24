#!/usr/bin/env python3
"""
enrich_trains.py — CLIP-based visual tagging and similarity pipeline
for the Trains at Temple exhibit.

Adds to each item in trains.json:
  id             str   URL-safe slug from title
  image_url      str   Primary image URL resolved from IIIF manifest
  visual_tags    list  Top-4 descriptive labels via CLIP zero-shot classification
  similar_items  list  IDs of top-3 visually similar items (cosine similarity)

Usage:
  python3 image-analysis/enrich_trains.py --input data/trains.json --output data/trains.json

  # Preview without writing:
  python3 image-analysis/enrich_trains.py --input data/trains.json --output data/trains.json --dry-run

  # Force re-tag all items (even cached):
  python3 image-analysis/enrich_trains.py --input data/trains.json --output data/trains.json --force-retag
"""

import argparse
import json
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor


CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# Train-specific tag schema
TAG_PROMPTS = {
    # -- Locomotive type --
    "steam locomotive":    "a steam locomotive with visible smoke, steam, or smokestack",
    "diesel locomotive":   "a diesel locomotive or diesel-electric train engine",
    "electric locomotive": "an electric locomotive or electric train with overhead wires or third rail",

    # -- Train type --
    "passenger train":     "a passenger train with passenger cars or coaches",
    "freight train":       "a freight train with boxcars, tankers, or flatcars",
    "yard / switching":    "a rail yard with multiple tracks, switching operations, or parked rolling stock",

    # -- Setting --
    "station / depot":     "a railroad station, train depot, or platform where passengers board",
    "rail yard":           "a railroad yard with many tracks, signals, and parked trains",
    "open track":          "a train on open track through countryside, fields, or rural landscape",
    "urban / industrial":  "a train in an urban or industrial setting with buildings and infrastructure",
    "bridge / trestle":    "a railroad bridge, trestle, or viaduct",

    # -- Era --
    "vintage (pre-1920s)": "a very old photograph from the early 1900s or 19th century, sepia toned",
    "mid-century":         "a mid-20th century photograph from the 1930s to 1950s",

    # -- Photo characteristics --
    "black and white":     "a black and white photograph with no color",
    "color photo":         "a color photograph",
    "close-up detail":     "a close-up or detail shot showing mechanical parts, wheels, or equipment",

    # -- Content type --
    "photograph":          "a photographic image, a real photograph of a scene",
    "poster / illustration": "an illustrated poster, drawing, or graphic design, not a photograph",
}

TOP_TAGS = 4
TOP_SIMILAR = 3
REQUEST_DELAY = 0.5


def slugify(text: str) -> str:
    s = text.lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s.strip())
    return s[:80]


def fetch_manifest(url: str, cache_dir: Path) -> Optional[dict]:
    slug = slugify(url[-60:])
    cache_path = cache_dir / f"{slug}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        cache_path.write_text(json.dumps(data))
        return data
    except Exception as e:
        print(f"    [WARN] manifest fetch failed: {url} — {e}")
        return None


def extract_image_url(manifest: dict) -> str:
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


def fetch_image(url: str) -> Optional[Image.Image]:
    try:
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as e:
        print(f"    [WARN] image fetch failed: {url} — {e}")
        return None


def get_embedding(model, processor, image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].cpu().numpy()


def zero_shot(model, processor, image, labels, top_k):
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image[0]
    probs = logits.softmax(dim=-1).cpu().numpy()
    ranked = sorted(zip(labels, probs.tolist()), key=lambda x: -x[1])
    return [lbl for lbl, _ in ranked[:top_k]]


def zero_shot_from_embedding(model, processor, image_emb, labels, top_k):
    inputs = processor(text=labels, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_feats = model.get_text_features(**inputs)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    img_tensor = torch.tensor(image_emb, dtype=torch.float32).unsqueeze(0)
    logits = (img_tensor @ text_feats.T) * model.logit_scale.exp()
    probs = logits[0].softmax(dim=-1).detach().cpu().numpy()
    ranked = sorted(zip(labels, probs.tolist()), key=lambda x: -x[1])
    return [lbl for lbl, _ in ranked[:top_k]]


def save_embedding(cache_dir, item_id, emb):
    np.save(str(cache_dir / f"{item_id}_emb.npy"), emb)


def load_embedding(cache_dir, item_id):
    path = cache_dir / f"{item_id}_emb.npy"
    if path.exists():
        return np.load(str(path))
    return None


def cosine_sim_matrix(embeddings):
    mat = np.array(embeddings)
    return mat @ mat.T


def phase1_fetch_and_tag(items, model, processor, cache_dir, force_retag=False):
    prompts = list(TAG_PROMPTS.values())
    prompt_to_label = {v: k for k, v in TAG_PROMPTS.items()}
    embeddings = []

    for item in items:
        item["id"] = slugify(item["title"])
        print(f"  {item['title']}")

        cached_emb = load_embedding(cache_dir, item["id"])
        if cached_emb is not None:
            if force_retag or "visual_tags" not in item:
                top_prompts = zero_shot_from_embedding(model, processor, cached_emb, prompts, TOP_TAGS)
                item["visual_tags"] = [prompt_to_label.get(p, p) for p in top_prompts]
                print("    [cached embedding, tags updated]")
            else:
                print("    [cached, skipping]")
            embeddings.append(cached_emb)
            continue

        manifest = fetch_manifest(item["manifest"], cache_dir)
        if not manifest:
            item["image_url"] = ""
            embeddings.append(None)
            continue

        image_url = extract_image_url(manifest)
        item["image_url"] = image_url

        if not image_url:
            print("    [WARN] no image URL in manifest")
            embeddings.append(None)
            continue

        image = fetch_image(image_url)
        if image is None:
            embeddings.append(None)
            continue

        top_prompts = zero_shot(model, processor, image, prompts, TOP_TAGS)
        item["visual_tags"] = [prompt_to_label.get(p, p) for p in top_prompts]

        emb = get_embedding(model, processor, image)
        save_embedding(cache_dir, item["id"], emb)
        embeddings.append(emb)
        time.sleep(REQUEST_DELAY)

    return embeddings


def phase2_assign_similar(items, embeddings, ids):
    valid = [(i, e) for i, e in enumerate(embeddings) if e is not None]
    if len(valid) < 2:
        return

    valid_indices, valid_embs = zip(*valid)
    sim = cosine_sim_matrix(list(valid_embs))

    for rank, i in enumerate(valid_indices):
        row = sim[rank]
        sorted_j = sorted(enumerate(row), key=lambda x: -x[1])
        similar = [
            ids[valid_indices[j]]
            for j, _ in sorted_j
            if valid_indices[j] != i
        ][:TOP_SIMILAR]
        items[i]["similar_items"] = similar


def main():
    global TOP_TAGS, TOP_SIMILAR
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",   required=True, help="Input trains.json")
    parser.add_argument("--output",  required=True, help="Output trains.json")
    parser.add_argument("--cache-dir", default=".manifest_cache",
                        help="Directory for caching manifests and embeddings")
    parser.add_argument("--top-tags",    type=int, default=TOP_TAGS)
    parser.add_argument("--top-similar", type=int, default=TOP_SIMILAR)
    parser.add_argument("--force-retag", action="store_true",
                        help="Recompute visual_tags for all items, even cached ones")
    parser.add_argument("--dry-run", action="store_true", help="Process but do not write output")
    args = parser.parse_args()

    TOP_TAGS = args.top_tags
    TOP_SIMILAR = args.top_similar

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True)

    print(f"Loading CLIP ({CLIP_MODEL_ID})...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    print("Model ready.\n")

    with open(args.input) as f:
        data = json.load(f)

    items = data["items"]
    print(f"Processing {len(items)} items...\n")

    embeddings = phase1_fetch_and_tag(items, model, processor, cache_dir, args.force_retag)
    ids = [item["id"] for item in items]
    phase2_assign_similar(items, embeddings, ids)

    # Print tag distribution summary
    tag_counts = {}
    for item in items:
        for tag in item.get("visual_tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    print("\n=== Tag Distribution ===")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {count}")

    if args.dry_run:
        print("\nDry run — no file written.")
        return

    data["items"] = items
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nDone. Enriched data written to {args.output}")


if __name__ == "__main__":
    main()
