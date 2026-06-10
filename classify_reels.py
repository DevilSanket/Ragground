#!/usr/bin/env python3
"""
classify_reels.py
─────────────────
Uses Gemini to classify each reel transcript into one of five content types
and extract structured, type-specific data:

  Content types:
    recipe          — cooking how-to, ingredients, steps
    travel_vlog     — place visits, journeys, street food exploration
    informational   — educational, tips, facts, brand story, process
    product_showcase — product reveals, reviews, launches
    other           — lifestyle, personal, unclassifiable

  Per type, generates a rich Markdown file in:
    markdown/recipes/           (recipe type)
    markdown/travel_vlogs/      (travel type)
    markdown/informational/     (informational type)
    markdown/product_showcase/  (product_showcase type)
    markdown/other/             (other type)

  All content is also ingested into the SQLite vector DB under the
  'instagram_reels' collection with a 'content_type' metadata field
  enabling filtered semantic search.

Usage:
    python classify_reels.py                   # classify all transcripts
    python classify_reels.py --id DPymeb6Cpgn  # single reel
    python classify_reels.py --ingest-only     # skip classification, re-ingest
    python classify_reels.py --show-all        # show classification results
    python classify_reels.py --type recipe     # show only recipes
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from collections import defaultdict

# ── Shared config ──────────────────────────────────────────────────────────────
from config import cfg

BASE_DIR   = cfg.BASE_DIR
MARKDOWN   = cfg.MARKDOWN_DIR
CACHE_FILE = cfg.CACHE_FILE

# Per content-type output directories
CONTENT_DIRS = {
    "recipe":           MARKDOWN / "recipes",
    "travel_vlog":      MARKDOWN / "travel_vlogs",
    "informational":    MARKDOWN / "informational",
    "product_showcase": MARKDOWN / "product_showcase",
    "other":            MARKDOWN / "other",
}

GEMINI_API_KEY = cfg.GEMINI_API_KEY
GEMINI_MODEL   = cfg.GEMINI_MODEL_CLASSIFY


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI MULTI-TYPE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

CLASSIFY_PROMPT_TEMPLATE = """You are analyzing Instagram Reel transcripts and captions for "Ground Up" — an artisan Indian food brand known for miso, fermented products, tofu, and healthy eating.

Classify the reel into EXACTLY ONE of these content types:
  - recipe          : Shows/describes how to make a dish (cooking steps, ingredients)
  - travel_vlog     : Documents a visit to a place, market, restaurant, or journey
  - informational   : Educational content — tips, facts, brand story, process, science
  - product_showcase: Focus on showcasing/launching/reviewing a specific product
  - other           : Lifestyle, personal, unclassifiable

Respond ONLY with valid JSON. No markdown fences. No explanation outside the JSON.

━━━ OUTPUT FORMAT ━━━

For ALL types, always include these base fields:
{
  "content_type": "<one of the 5 types>",
  "confidence": "high | medium | low",
  "summary": "2-3 sentence summary of what this reel is about",
  "topics": ["topic1", "topic2"],
  "ground_up_products": ["product names mentioned, if any"]
}

Then ADD type-specific fields:

If content_type == "recipe":
  "recipe_name": "Name of the dish",
  "description": "1-2 sentence description",
  "cuisine_type": "Indian | Asian | Mediterranean | etc or null",
  "meal_type": "breakfast | lunch | dinner | snack | dessert or null",
  "dietary_tags": ["vegan", "vegetarian", "gluten-free" — only if clearly stated],
  "difficulty": "easy | medium | hard",
  "estimated_time": "time or null",
  "ingredients": ["ingredient 1", "ingredient 2"],
  "steps": ["Step 1: ...", "Step 2: ..."]

If content_type == "travel_vlog":
  "location": "City, State/Country",
  "place_type": "market | restaurant | city | region | street | farm | other",
  "highlights": ["highlight 1", "highlight 2"],
  "food_spots": ["Any food places mentioned"],
  "mood": "adventurous | cultural | foodie | relaxed"

If content_type == "informational":
  "subject": "Main topic of the video",
  "key_points": ["point 1", "point 2", "point 3"],
  "takeaway": "The main actionable or educational takeaway"

If content_type == "product_showcase":
  "product_name": "Name of the product being showcased",
  "product_category": "fermented | condiment | snack | ingredient | other",
  "use_cases": ["use case 1", "use case 2"],
  "key_features": ["feature 1", "feature 2"]

━━━ CONTENT TO ANALYZE ━━━

Transcript:
__TRANSCRIPT__

Caption:
__CAPTION__"""


def build_classify_prompt(transcript: str, caption: str) -> str:
    """Build classification prompt (safe placeholder replacement, no .format())."""
    return (
        CLASSIFY_PROMPT_TEMPLATE
        .replace("__TRANSCRIPT__", transcript[:3000])
        .replace("__CAPTION__", caption[:1000])
    )


def classify_with_gemini(transcript: str, caption: str, reel_id: str) -> dict:
    """Call Gemini to classify a reel transcript/caption into one of 5 content types."""
    if not GEMINI_API_KEY:
        print("  ERROR: GEMINI_API_KEY is not configured in .env.")
        return {"content_type": "other", "confidence": "low", "reason": "No API key"}

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = build_classify_prompt(transcript, caption)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=4096,
            ),
        )
        raw = response.text.strip()

        # Strip markdown code fences if Gemini adds them anyway
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)

        # Validate content_type is one of our 5 types
        valid_types = set(cfg.CONTENT_TYPES)
        if result.get("content_type") not in valid_types:
            result["content_type"] = "other"

        return result

    except json.JSONDecodeError as e:
        print(f"  JSON parse error for {reel_id}: {e}")
        return {"content_type": "other", "confidence": "low", "reason": f"Parse error: {e}"}
    except Exception as e:
        print(f"  Gemini error for {reel_id}: {e}")
        return {"content_type": "other", "confidence": "low", "reason": f"API error: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_markdown(md_path: Path) -> dict:
    """Extract key sections from a reel markdown file."""
    import re
    text = md_path.read_text(encoding="utf-8")

    meta = {}
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                meta[k.strip()] = v.strip().strip('"')

    def section(heading):
        pat = rf"## {re.escape(heading)}\n+(.*?)(?=\n## |\Z)"
        s = re.search(pat, text, re.DOTALL)
        return s.group(1).strip() if s else ""

    return {
        "reel_id":    meta.get("reel_id", md_path.stem),
        "url":        meta.get("url", ""),
        "author":     meta.get("author", ""),
        "date":       meta.get("date", ""),
        "likes":      meta.get("likes", ""),
        "duration":   meta.get("duration_seconds", ""),
        "transcript": section("Transcript"),
        "caption":    section("Caption"),
        "segments":   section("Timestamped Segments"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN BUILDERS — per content type
# ══════════════════════════════════════════════════════════════════════════════

def _clean(val) -> str:
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    return str(val) if val is not None else ""


def _frontmatter_base(reel_data: dict, result: dict) -> list[str]:
    """Build the common YAML frontmatter lines."""
    lines = ["---"]
    lines.append(f'reel_id: "{reel_data["reel_id"]}"')
    lines.append(f'url: "{reel_data["url"]}"')
    lines.append(f'author: "{reel_data["author"]}"')
    lines.append(f'date: "{reel_data["date"]}"')
    lines.append(f'likes: {reel_data["likes"]}')
    lines.append(f'duration_seconds: {reel_data["duration"]}')
    lines.append(f'content_type: {result.get("content_type", "other")}')
    lines.append(f'confidence: "{result.get("confidence", "")}"')
    products = result.get("ground_up_products", [])
    if products:
        lines.append(f'ground_up_products: [{", ".join(str(p) for p in products)}]')
    topics = result.get("topics", [])
    if topics:
        lines.append(f'topics: [{", ".join(str(t) for t in topics)}]')
    return lines


def build_recipe_markdown(reel_data: dict, result: dict) -> str:
    lines = _frontmatter_base(reel_data, result)
    lines.append(f'recipe_name: "{result.get("recipe_name", "")}"')
    for field in ["cuisine_type", "meal_type", "difficulty", "estimated_time"]:
        val = result.get(field)
        if val:
            lines.append(f'{field}: "{_clean(val)}"')
    tags = result.get("dietary_tags", [])
    if tags:
        lines.append(f'dietary_tags: [{", ".join(str(t) for t in tags)}]')
    lines.append("---\n")

    name = result.get("recipe_name", f"Recipe from {reel_data['reel_id']}")
    lines.append(f"# {name}\n")

    desc = result.get("description", "")
    if desc:
        lines.append(f"_{desc}_\n")

    info = []
    for field, icon in [("cuisine_type", "🍽️"), ("meal_type", "🕐"), ("difficulty", "📊"), ("estimated_time", "⏱️")]:
        val = result.get(field)
        if val:
            info.append(f"{icon} {_clean(val).capitalize()}")
    if info:
        lines.append("  |  ".join(info) + "\n")

    lines.append("## Source\n")
    lines.append(f"- **Instagram Reel:** [{reel_data['author']}]({reel_data['url']})")
    lines.append(f"- **Date:** {reel_data['date']}  |  **Likes:** {reel_data['likes']}\n")

    ingredients = result.get("ingredients", [])
    if ingredients:
        lines.append("## Ingredients\n")
        for ing in ingredients:
            lines.append(f"- {ing}")
        lines.append("")

    products = result.get("ground_up_products", [])
    if products:
        lines.append("## Ground Up Products Used\n")
        for p in products:
            lines.append(f"- **{p}** — [shop via link in bio]({reel_data['url']})")
        lines.append("")

    steps = result.get("steps", [])
    if steps:
        lines.append("## Instructions\n")
        for i, step in enumerate(steps, 1):
            step_text = step.split(":", 1)[-1].strip() if step.lower().startswith("step") else step
            lines.append(f"{i}. {step_text}")
        lines.append("")

    if tags:
        lines.append("## Dietary Tags\n")
        lines.append("  ".join(f"`{t}`" for t in tags) + "\n")

    summary = result.get("summary", "")
    if summary:
        lines.append("## Summary\n")
        lines.append(summary + "\n")

    if reel_data["transcript"]:
        lines.append("## Original Transcript\n")
        lines.append(reel_data["transcript"] + "\n")

    if reel_data["caption"]:
        lines.append("## Original Caption\n")
        lines.append(reel_data["caption"] + "\n")

    return "\n".join(lines)


def build_travel_markdown(reel_data: dict, result: dict) -> str:
    lines = _frontmatter_base(reel_data, result)
    location = result.get("location", "")
    if location:
        lines.append(f'location: "{_clean(location)}"')
    place_type = result.get("place_type", "")
    if place_type:
        lines.append(f'place_type: "{_clean(place_type)}"')
    mood = result.get("mood", "")
    if mood:
        lines.append(f'mood: "{_clean(mood)}"')
    lines.append("---\n")

    title = location or f"Travel Vlog — {reel_data['reel_id']}"
    lines.append(f"# {title}\n")

    summary = result.get("summary", "")
    if summary:
        lines.append(f"_{summary}_\n")

    lines.append("## Source\n")
    lines.append(f"- **Instagram Reel:** [{reel_data['author']}]({reel_data['url']})")
    lines.append(f"- **Date:** {reel_data['date']}  |  **Likes:** {reel_data['likes']}\n")

    highlights = result.get("highlights", [])
    if highlights:
        lines.append("## Highlights\n")
        for h in highlights:
            lines.append(f"- {h}")
        lines.append("")

    food_spots = result.get("food_spots", [])
    if food_spots:
        lines.append("## Food Spots\n")
        for f in food_spots:
            lines.append(f"- {f}")
        lines.append("")

    products = result.get("ground_up_products", [])
    if products:
        lines.append("## Ground Up Products Mentioned\n")
        for p in products:
            lines.append(f"- **{p}**")
        lines.append("")

    if reel_data["transcript"]:
        lines.append("## Original Transcript\n")
        lines.append(reel_data["transcript"] + "\n")

    if reel_data["caption"]:
        lines.append("## Original Caption\n")
        lines.append(reel_data["caption"] + "\n")

    return "\n".join(lines)


def build_informational_markdown(reel_data: dict, result: dict) -> str:
    lines = _frontmatter_base(reel_data, result)
    subject = result.get("subject", "")
    if subject:
        lines.append(f'subject: "{_clean(subject)}"')
    lines.append("---\n")

    title = subject or f"Educational Reel — {reel_data['reel_id']}"
    lines.append(f"# {title}\n")

    summary = result.get("summary", "")
    if summary:
        lines.append(f"_{summary}_\n")

    lines.append("## Source\n")
    lines.append(f"- **Instagram Reel:** [{reel_data['author']}]({reel_data['url']})")
    lines.append(f"- **Date:** {reel_data['date']}  |  **Likes:** {reel_data['likes']}\n")

    key_points = result.get("key_points", [])
    if key_points:
        lines.append("## Key Points\n")
        for pt in key_points:
            lines.append(f"- {pt}")
        lines.append("")

    takeaway = result.get("takeaway", "")
    if takeaway:
        lines.append("## Main Takeaway\n")
        lines.append(f"> {takeaway}\n")

    products = result.get("ground_up_products", [])
    if products:
        lines.append("## Ground Up Products Mentioned\n")
        for p in products:
            lines.append(f"- **{p}**")
        lines.append("")

    if reel_data["transcript"]:
        lines.append("## Original Transcript\n")
        lines.append(reel_data["transcript"] + "\n")

    if reel_data["caption"]:
        lines.append("## Original Caption\n")
        lines.append(reel_data["caption"] + "\n")

    return "\n".join(lines)


def build_product_markdown(reel_data: dict, result: dict) -> str:
    lines = _frontmatter_base(reel_data, result)
    product_name = result.get("product_name", "")
    if product_name:
        lines.append(f'product_name: "{_clean(product_name)}"')
    product_cat = result.get("product_category", "")
    if product_cat:
        lines.append(f'product_category: "{_clean(product_cat)}"')
    lines.append("---\n")

    title = product_name or f"Product Showcase — {reel_data['reel_id']}"
    lines.append(f"# {title}\n")

    summary = result.get("summary", "")
    if summary:
        lines.append(f"_{summary}_\n")

    lines.append("## Source\n")
    lines.append(f"- **Instagram Reel:** [{reel_data['author']}]({reel_data['url']})")
    lines.append(f"- **Date:** {reel_data['date']}  |  **Likes:** {reel_data['likes']}\n")

    features = result.get("key_features", [])
    if features:
        lines.append("## Key Features\n")
        for f in features:
            lines.append(f"- {f}")
        lines.append("")

    use_cases = result.get("use_cases", [])
    if use_cases:
        lines.append("## Use Cases\n")
        for u in use_cases:
            lines.append(f"- {u}")
        lines.append("")

    if reel_data["transcript"]:
        lines.append("## Original Transcript\n")
        lines.append(reel_data["transcript"] + "\n")

    if reel_data["caption"]:
        lines.append("## Original Caption\n")
        lines.append(reel_data["caption"] + "\n")

    return "\n".join(lines)


def build_other_markdown(reel_data: dict, result: dict) -> str:
    lines = _frontmatter_base(reel_data, result)
    lines.append("---\n")

    lines.append(f"# Reel — {reel_data['reel_id']}\n")

    summary = result.get("summary", "")
    if summary:
        lines.append(f"_{summary}_\n")

    lines.append("## Source\n")
    lines.append(f"- **Instagram Reel:** [{reel_data['author']}]({reel_data['url']})")
    lines.append(f"- **Date:** {reel_data['date']}  |  **Likes:** {reel_data['likes']}\n")

    if reel_data["transcript"]:
        lines.append("## Original Transcript\n")
        lines.append(reel_data["transcript"] + "\n")

    if reel_data["caption"]:
        lines.append("## Original Caption\n")
        lines.append(reel_data["caption"] + "\n")

    return "\n".join(lines)


MARKDOWN_BUILDERS = {
    "recipe":           build_recipe_markdown,
    "travel_vlog":      build_travel_markdown,
    "informational":    build_informational_markdown,
    "product_showcase": build_product_markdown,
    "other":            build_other_markdown,
}


def sanitize_filename(name: str, fallback: str) -> str:
    """Make a safe filename from a title."""
    cleaned = "".join(c if c.isalnum() or c in (" ", "_", "-") else "" for c in name)
    cleaned = cleaned.strip().lower().replace(" ", "_")
    return cleaned if cleaned else fallback


# ══════════════════════════════════════════════════════════════════════════════
# CACHE
# ══════════════════════════════════════════════════════════════════════════════

def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict):
    CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# VECTOR DB INGESTION (all content types → 'instagram_reels' collection)
# ══════════════════════════════════════════════════════════════════════════════

def ingest_all_content(all_md_files: dict[str, list[Path]], reset: bool = False):
    """
    Ingest all classified markdown files into the 'instagram_reels' collection.
    Each chunk carries a 'content_type' metadata field for filtered retrieval.

    all_md_files: {content_type: [Path, ...]}
    """
    import re
    import postgres_db

    flat_files = [(ct, path) for ct, paths in all_md_files.items() for path in paths]
    print(f"\nIngesting {len(flat_files)} file(s) across all content types...")

    # Ensure collection exists
    postgres_db.init_db(reset=reset, collection_name="instagram_reels")

    def extract_meta(text, key):
        m = re.search(rf'^{key}:\s*"?([^"\n]+)"?', text, re.MULTILINE)
        return m.group(1).strip() if m else ""

    all_ids, all_docs, all_metas = [], [], []

    for content_type, md_path in flat_files:
        text          = md_path.read_text(encoding="utf-8")
        filename_stem = md_path.stem
        reel_id       = extract_meta(text, "reel_id") or filename_stem
        author        = extract_meta(text, "author")
        url           = extract_meta(text, "url")
        date          = extract_meta(text, "date")
        likes         = extract_meta(text, "likes")

        base_meta = {
            "reel_id":      reel_id,
            "author":       author,
            "url":          url,
            "date":         date,
            "likes":        likes,
            "content_type": content_type,
        }

        # Pull type-specific metadata fields
        for field in ["recipe_name", "location", "subject", "product_name",
                      "cuisine_type", "meal_type", "difficulty"]:
            val = extract_meta(text, field)
            if val:
                base_meta[field] = val

        def get_section(heading):
            m = re.search(rf"## {re.escape(heading)}\n+(.*?)(?=\n## |\Z)", text, re.DOTALL)
            return m.group(1).strip() if m else ""

        summary_text = get_section("Summary") or get_section("Original Transcript")[:500]

        # Chunk 1: Full content overview (used for discovery)
        title_field = {
            "recipe":           extract_meta(text, "recipe_name"),
            "travel_vlog":      extract_meta(text, "location"),
            "informational":    extract_meta(text, "subject"),
            "product_showcase": extract_meta(text, "product_name"),
        }.get(content_type, filename_stem)

        overview_parts = [
            f"Content Type: {content_type}",
            f"Title: {title_field}" if title_field else "",
            f"Summary: {summary_text}" if summary_text else "",
        ]

        # Type-specific section chunks
        if content_type == "recipe":
            ingredients = get_section("Ingredients")
            steps = get_section("Instructions")
            if ingredients:
                overview_parts.append(f"Ingredients:\n{ingredients}")
            if steps:
                overview_parts.append(f"Instructions:\n{steps}")

        elif content_type == "travel_vlog":
            highlights = get_section("Highlights")
            food_spots = get_section("Food Spots")
            if highlights:
                overview_parts.append(f"Highlights:\n{highlights}")
            if food_spots:
                overview_parts.append(f"Food Spots:\n{food_spots}")

        elif content_type == "informational":
            key_points = get_section("Key Points")
            takeaway = get_section("Main Takeaway")
            if key_points:
                overview_parts.append(f"Key Points:\n{key_points}")
            if takeaway:
                overview_parts.append(f"Takeaway: {takeaway}")

        elif content_type == "product_showcase":
            features = get_section("Key Features")
            use_cases = get_section("Use Cases")
            if features:
                overview_parts.append(f"Key Features:\n{features}")
            if use_cases:
                overview_parts.append(f"Use Cases:\n{use_cases}")

        overview = "\n\n".join(p for p in overview_parts if p)
        all_ids.append(f"{filename_stem}__overview")
        all_docs.append(overview)
        all_metas.append({**base_meta, "chunk_type": "overview"})

        # Chunk 2: Raw transcript (for verbatim search)
        transcript = get_section("Original Transcript")
        if transcript:
            all_ids.append(f"{filename_stem}__transcript")
            all_docs.append(f"[{content_type}] {title_field}\n\n{transcript}")
            all_metas.append({**base_meta, "chunk_type": "transcript"})

        # Chunk 3: Caption (for metadata/hashtag search)
        caption = get_section("Original Caption")
        if caption:
            all_ids.append(f"{filename_stem}__caption")
            all_docs.append(f"[{content_type}] {title_field}\n\n{caption}")
            all_metas.append({**base_meta, "chunk_type": "caption"})

    if all_ids:
        postgres_db.upsert_chunks(
            collection_name="instagram_reels",
            ids=all_ids,
            documents=all_docs,
            metadatas=all_metas,
        )
        total = postgres_db.get_collection_count("instagram_reels")
        print(f"Upserted {len(all_ids)} chunks. Total in 'instagram_reels': {total}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Multi-type reel classifier: recipe | travel_vlog | informational | product_showcase | other"
    )
    parser.add_argument("--id",          help="Classify a single reel by ID")
    parser.add_argument("--ingest-only", action="store_true",
                        help="Skip classification, re-ingest existing classified markdowns")
    parser.add_argument("--show-all",    action="store_true",
                        help="Print classification results and exit")
    parser.add_argument("--type",        choices=cfg.CONTENT_TYPES,
                        help="Filter --show-all to a specific content type")
    parser.add_argument("--force",       action="store_true",
                        help="Re-classify even if cached results exist")
    parser.add_argument("--reset",       action="store_true",
                        help="Wipe and recreate the vector DB collection")
    args = parser.parse_args()

    # Ensure all output directories exist
    for d in CONTENT_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    # ── Show cached results ───────────────────────────────────────────────────
    if args.show_all:
        cache = load_cache()
        print(f"\nClassification results ({len(cache)} reels):\n{'-'*60}")
        by_type = defaultdict(list)
        for rid, data in cache.items():
            ct = data.get("content_type", "other")
            if args.type and ct != args.type:
                continue
            by_type[ct].append((rid, data))

        for ct in cfg.CONTENT_TYPES:
            items = by_type.get(ct, [])
            label = cfg.CONTENT_TYPE_LABELS.get(ct, ct)
            print(f"\n  {label} ({len(items)}):")
            for rid, data in items:
                title = (
                    data.get("recipe_name") or data.get("location") or
                    data.get("subject")     or data.get("product_name") or
                    data.get("summary", "")[:50] or rid
                )
                conf = data.get("confidence", "?")
                print(f"    [{conf}] {rid}: {title}")
        return

    # ── Ingest-only mode ──────────────────────────────────────────────────────
    if args.ingest_only:
        all_md_files = {}
        for ct, d in CONTENT_DIRS.items():
            files = sorted(d.glob("*.md"))
            if files:
                all_md_files[ct] = files
                print(f"  {cfg.CONTENT_TYPE_LABELS.get(ct, ct)}: {len(files)} file(s)")

        if not all_md_files:
            print("No classified markdown files found. Run without --ingest-only to classify first.")
            sys.exit(1)

        ingest_all_content(all_md_files, reset=args.reset)
        return

    # ── Classification ────────────────────────────────────────────────────────
    if args.id:
        md_files = [MARKDOWN / f"{args.id}.md"]
    else:
        # Only direct children of markdown/ (not already classified subdirectories)
        md_files = sorted(f for f in MARKDOWN.glob("*.md") if f.parent == MARKDOWN)

    if not md_files:
        print("No markdown files found. Run transcribe_reels.py first.")
        sys.exit(1)

    print(f"Classifying {len(md_files)} reel(s) with Gemini {GEMINI_MODEL}...\n")

    cache   = load_cache()
    results = defaultdict(list)  # content_type → [md_path]
    skipped = []

    for i, md_path in enumerate(md_files, 1):
        reel_id = md_path.stem
        print(f"[{i}/{len(md_files)}] {reel_id}", end="", flush=True)

        # Check cache
        if not args.force and reel_id in cache:
            result = cache[reel_id]
            ct = result.get("content_type", "other")
            print(f" [cached] -> {cfg.CONTENT_TYPE_LABELS.get(ct, ct)}")
        else:
            reel_data = parse_markdown(md_path)
            if not reel_data["transcript"] and not reel_data["caption"]:
                print(f" [no content] -> skip")
                skipped.append(reel_id)
                continue

            print(" -> classifying...", end="", flush=True)
            result = classify_with_gemini(
                reel_data["transcript"], reel_data["caption"], reel_id
            )
            cache[reel_id] = result
            save_cache(cache)

            ct = result.get("content_type", "other")
            label = cfg.CONTENT_TYPE_LABELS.get(ct, ct)
            # Show type-specific title in output
            title = (
                result.get("recipe_name") or result.get("location") or
                result.get("subject")     or result.get("product_name") or
                result.get("summary", "")[:45] or "—"
            )
            print(f" -> {label}: {title}")

            time.sleep(1.5)  # polite Gemini rate limiting

        result = cache[reel_id]
        ct = result.get("content_type", "other")

        # Build and save the type-specific markdown
        reel_data = parse_markdown(md_path)
        builder = MARKDOWN_BUILDERS.get(ct, build_other_markdown)
        md_content = builder(reel_data, result)

        # Determine filename from type-specific title field
        title_raw = (
            result.get("recipe_name") or result.get("location") or
            result.get("subject")     or result.get("product_name") or reel_id
        )
        out_name = sanitize_filename(title_raw, reel_id)
        out_path = CONTENT_DIRS[ct] / f"{out_name}.md"
        out_path.write_text(md_content, encoding="utf-8")
        results[ct].append(out_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("Classification complete:\n")
    total_classified = 0
    for ct in cfg.CONTENT_TYPES:
        count = len(results[ct])
        total_classified += count
        label = cfg.CONTENT_TYPE_LABELS.get(ct, ct)
        print(f"  {label:30s} : {count}")
    print(f"  {'Skipped (no content)':30s} : {len(skipped)}")

    # ── Ingest into vector DB ─────────────────────────────────────────────────
    if total_classified > 0:
        ingest_all_content(dict(results), reset=args.reset)

        print(f"\nMarkdown files saved to:")
        for ct, paths in results.items():
            if paths:
                print(f"  {CONTENT_DIRS[ct]}")

        print(f"\nTest the knowledge base:")
        print(f"  python rag_chat.py                          # chat across all content")
        print(f"  python rag_chat.py --content-type recipe    # recipes only")
        print(f"  python rag_chat.py --content-type travel_vlog")
    else:
        print("\nNo content was classified. Check your markdown files.")


if __name__ == "__main__":
    main()
