#!/usr/bin/env python3
"""
classify_reels.py
─────────────────
Uses Gemini to:
  1. Classify each reel transcript as "recipe" or "non-recipe"
  2. For recipe reels: extract structured data (name, ingredients, steps, tags)
  3. Save enhanced recipe markdown files to markdown/recipes/
  4. Build a separate "recipes" PostgreSQL collection

This creates a high-quality, recipe-only knowledge base from your content.

Usage:
    python classify_reels.py                  # classify all transcripts
    python classify_reels.py --id DPymeb6Cpgn # single reel
    python classify_reels.py --ingest-only    # skip classification, just re-ingest
    python classify_reels.py --show-all       # show classification results
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

BASE_DIR    = Path(__file__).parent
MARKDOWN    = BASE_DIR / "markdown"
RECIPES_DIR = MARKDOWN / "recipes"
VECTOR_DB   = BASE_DIR / "vectordb"
CACHE_FILE  = BASE_DIR / "classification_cache.json"

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Load .env
_env = BASE_DIR / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-3.5-flash"


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

CLASSIFY_PROMPT_TEMPLATE = """You are analyzing Instagram Reel transcripts and captions for a food brand called "Ground Up".

Given the transcript and caption below (note: the recipe details may be written in the caption, spoken in the transcript, or both), determine:
1. Is this a COOKING RECIPE reel? (i.e., does it show/describe how to make a dish)
2. If YES, extract the recipe details.

Respond ONLY with valid JSON. No markdown fences, no explanation outside the JSON.

If NOT a recipe, return: {"is_recipe": false, "reason": "brief reason"}

If it IS a recipe, return:
{"is_recipe": true, "recipe_name": "Name of dish", "description": "1-2 sentence description",
 "ingredients": ["item1", "item2"], "steps": ["Step 1: ...", "Step 2: ..."],
 "cuisine_type": "Indian/Asian/etc or null", "meal_type": "breakfast/lunch/dinner/snack or null",
 "dietary_tags": ["vegan","vegetarian","gluten-free" — only if clearly stated],
 "key_products": ["Ground Up product names mentioned"],
 "difficulty": "easy/medium/hard", "estimated_time": "time or null", "confidence": "high/medium/low"}

Transcript:
__TRANSCRIPT__

Caption:
__CAPTION__"""


def build_classify_prompt(transcript: str, caption: str) -> str:
    """Build classification prompt using safe placeholder replacement (avoids .format() KeyError on JSON braces)."""
    return (
        CLASSIFY_PROMPT_TEMPLATE
        .replace("__TRANSCRIPT__", transcript[:3000])
        .replace("__CAPTION__", caption[:1000])
    )


def classify_with_gemini(transcript: str, caption: str, reel_id: str) -> dict:
    """Call Gemini to classify a reel transcript/caption."""
    if not GEMINI_API_KEY:
        print("  ERROR: GEMINI_API_KEY is not configured in .env.")
        return {"is_recipe": False, "reason": "No API key"}

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = build_classify_prompt(transcript, caption)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(
                temperature=0.1,       # low temp for consistent structured output
                max_output_tokens=4096,
            ),
        )
        raw = response.text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        return json.loads(raw)

    except json.JSONDecodeError as e:
        print(f"  JSON parse error for {reel_id}: {e}")
        print(f"  Raw response: {raw[:200]}")
        return {"is_recipe": False, "reason": f"Parse error: {e}"}
    except Exception as e:
        print(f"  Gemini error for {reel_id}: {e}")
        return {"is_recipe": False, "reason": f"API error: {e}"}



# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_markdown(md_path: Path) -> dict:
    """Extract key sections from a reel markdown file."""
    import re
    text = md_path.read_text(encoding="utf-8")

    # Frontmatter
    meta = {}
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                meta[k.strip()] = v.strip().strip('"')

    # Sections
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
# RECIPE MARKDOWN BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_recipe_markdown(reel_data: dict, recipe: dict) -> str:
    """Build an enhanced recipe markdown file from reel data + Gemini extraction."""
    reel_id    = reel_data["reel_id"]
    name       = recipe.get("recipe_name", f"Recipe from {reel_id}")
    desc       = recipe.get("description", "")
    ingredients = recipe.get("ingredients", [])
    steps      = recipe.get("steps", [])
    products   = recipe.get("key_products", [])
    tags       = recipe.get("dietary_tags", [])
    cuisine    = recipe.get("cuisine_type", "")
    meal_type  = recipe.get("meal_type", "")
    difficulty = recipe.get("difficulty", "")
    est_time   = recipe.get("estimated_time", "")
    confidence = recipe.get("confidence", "")

    lines = []

    # YAML frontmatter
    lines.append("---")
    lines.append(f'reel_id: "{reel_id}"')
    lines.append(f'recipe_name: "{name}"')
    lines.append(f'url: "{reel_data["url"]}"')
    lines.append(f'author: "{reel_data["author"]}"')
    lines.append(f'date: "{reel_data["date"]}"')
    lines.append(f'likes: {reel_data["likes"]}')
    lines.append(f'duration_seconds: {reel_data["duration"]}')
    def clean_str(val):
        if isinstance(val, list):
            return ", ".join(str(v) for v in val)
        return str(val) if val is not None else ""

    if cuisine:
        lines.append(f'cuisine_type: "{clean_str(cuisine)}"')
    if meal_type:
        lines.append(f'meal_type: "{clean_str(meal_type)}"')
    if difficulty:
        lines.append(f'difficulty: "{clean_str(difficulty)}"')
    if est_time:
        lines.append(f'estimated_time: "{clean_str(est_time)}"')
    if tags:
        lines.append(f'dietary_tags: [{", ".join(f"{t}" for t in tags)}]')
    if products:
        lines.append(f'ground_up_products: [{", ".join(f"{p}" for p in products)}]')
    lines.append(f'classification_confidence: "{confidence}"')
    lines.append("content_type: recipe")
    lines.append("---")
    lines.append("")

    # Title
    lines.append(f"# {name}")
    lines.append("")

    if desc:
        lines.append(f"_{desc}_")
        lines.append("")

    # Quick info
    info_parts = []
    if cuisine:   info_parts.append(f"🍽️ {clean_str(cuisine)}")
    if meal_type: info_parts.append(f"🕐 {clean_str(meal_type).capitalize()}")
    if difficulty: info_parts.append(f"📊 {clean_str(difficulty).capitalize()}")
    if est_time:  info_parts.append(f"⏱️ {clean_str(est_time)}")
    if info_parts:
        lines.append("  |  ".join(info_parts))
        lines.append("")

    # Source
    lines.append("## Source")
    lines.append("")
    lines.append(f"- **Instagram Reel:** [{reel_data['author']}]({reel_data['url']})")
    lines.append(f"- **Date:** {reel_data['date']}  |  **Likes:** {reel_data['likes']}")
    lines.append("")

    # Ingredients
    if ingredients:
        lines.append("## Ingredients")
        lines.append("")
        for ing in ingredients:
            lines.append(f"- {ing}")
        lines.append("")

    # Ground Up Products
    if products:
        lines.append("## Ground Up Products Used")
        lines.append("")
        for p in products:
            lines.append(f"- **{p}** — [shop via link in bio]({reel_data['url']})")
        lines.append("")

    # Steps
    if steps:
        lines.append("## Instructions")
        lines.append("")
        for i, step in enumerate(steps, 1):
            # Remove leading "Step N:" if already present
            step_text = step
            if step.lower().startswith("step"):
                step_text = step.split(":", 1)[-1].strip()
            lines.append(f"{i}. {step_text}")
        lines.append("")

    # Tags
    if tags:
        lines.append("## Dietary Tags")
        lines.append("")
        lines.append("  ".join(f"`{t}`" for t in tags))
        lines.append("")

    # Original transcript
    if reel_data["transcript"]:
        lines.append("## Original Transcript")
        lines.append("")
        lines.append(reel_data["transcript"])
        lines.append("")

    # Original caption
    if reel_data["caption"]:
        lines.append("## Original Caption")
        lines.append("")
        lines.append(reel_data["caption"])
        lines.append("")

    return "\n".join(lines)


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
# VECTOR DB INGESTION (RECIPES COLLECTION)
# ══════════════════════════════════════════════════════════════════════════════

def ingest_recipes(recipe_files: list[Path], reset: bool = False):
    """Ingest recipe markdown files into a dedicated 'recipes' PostgreSQL collection."""
    import postgres_db, re

    print(f"\nIngesting {len(recipe_files)} recipe(s) into 'recipes' collection...")

    # Initialize PostgreSQL for 'recipes' collection
    postgres_db.init_db(reset=reset, collection_name="recipes")

    def extract_meta(text, key):
        m = re.search(rf'^{key}:\s*"?([^"\n]+)"?', text, re.MULTILINE)
        return m.group(1).strip() if m else ""

    all_ids, all_docs, all_metas = [], [], []

    for md_path in recipe_files:
        text       = md_path.read_text(encoding="utf-8")
        filename_stem = md_path.stem
        reel_id    = extract_meta(text, "reel_id")
        if not reel_id:
            reel_id = filename_stem
        recipe_name = extract_meta(text, "recipe_name")
        author     = extract_meta(text, "author")
        url        = extract_meta(text, "url")
        date       = extract_meta(text, "date")
        cuisine    = extract_meta(text, "cuisine_type")
        meal_type  = extract_meta(text, "meal_type")
        difficulty = extract_meta(text, "difficulty")
        likes      = extract_meta(text, "likes")

        base_meta = {
            "reel_id":     reel_id,
            "recipe_name": recipe_name,
            "author":      author,
            "url":         url,
            "date":        date,
            "cuisine":     cuisine,
            "meal_type":   meal_type,
            "difficulty":  difficulty,
            "likes":       likes,
            "content_type": "recipe",
        }

        def get_section(heading):
            m = re.search(rf"## {re.escape(heading)}\n+(.*?)(?=\n## |\Z)", text, re.DOTALL)
            return m.group(1).strip() if m else ""

        # Chunk 1: Full recipe description chunk (for discovery)
        ingredients = get_section("Ingredients")
        steps       = get_section("Instructions")
        overview    = f"Recipe: {recipe_name}\nSection: Full Overview\n\nIngredients:\n{ingredients}\n\nInstructions:\n{steps}"
        all_ids.append(f"{filename_stem}__recipe_overview")
        all_docs.append(overview)
        all_metas.append({**base_meta, "chunk_type": "recipe_overview"})

        # Chunk 2: Ingredients only
        if ingredients:
            all_ids.append(f"{filename_stem}__ingredients")
            all_docs.append(f"Recipe: {recipe_name}\nSection: Ingredients\n\n{ingredients}")
            all_metas.append({**base_meta, "chunk_type": "ingredients"})

        # Chunk 3: Steps only
        if steps:
            all_ids.append(f"{filename_stem}__steps")
            all_docs.append(f"Recipe: {recipe_name}\nSection: Instructions\n\n{steps}")
            all_metas.append({**base_meta, "chunk_type": "steps"})

        # Chunk 4: Original transcript
        transcript = get_section("Original Transcript")
        if transcript:
            all_ids.append(f"{filename_stem}__transcript")
            all_docs.append(f"Recipe: {recipe_name}\nSection: Original Transcript\n\n{transcript}")
            all_metas.append({**base_meta, "chunk_type": "transcript"})

        # Chunk 5: Original caption
        caption = get_section("Original Caption")
        if caption:
            all_ids.append(f"{filename_stem}__caption")
            all_docs.append(f"Recipe: {recipe_name}\nSection: Original Caption\n\n{caption}")
            all_metas.append({**base_meta, "chunk_type": "caption"})

    if all_ids:
        postgres_db.upsert_chunks(
            collection_name="recipes",
            ids=all_ids,
            documents=all_docs,
            metadatas=all_metas
        )
        print(f"Upserted {len(all_ids)} chunks into 'recipes' collection.")
        print(f"Total chunks in 'recipes': {postgres_db.get_collection_count('recipes')}")



# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Classify reels as recipe/non-recipe using Gemini and build a recipe knowledge base"
    )
    parser.add_argument("--id",          help="Classify a single reel by ID")
    parser.add_argument("--ingest-only", action="store_true",
                        help="Skip classification, just re-ingest existing recipe markdowns")
    parser.add_argument("--show-all",    action="store_true",
                        help="Print classification results and exit")
    parser.add_argument("--force",       action="store_true",
                        help="Re-classify even if cached results exist")
    parser.add_argument("--reset",       action="store_true",
                        help="Wipe and recreate recipes vector DB collection")
    args = parser.parse_args()

    RECIPES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Show cached results ───────────────────────────────────────────────────
    if args.show_all:
        cache = load_cache()
        print(f"\nClassification results ({len(cache)} reels):\n{'─'*55}")
        recipes    = {k: v for k, v in cache.items() if v.get("is_recipe")}
        non_recipes = {k: v for k, v in cache.items() if not v.get("is_recipe")}
        print(f"  RECIPES ({len(recipes)}):")
        for rid, data in recipes.items():
            print(f"    ✅ {rid}: {data.get('recipe_name', '?')}")
        print(f"\n  NON-RECIPES ({len(non_recipes)}):")
        for rid, data in non_recipes.items():
            print(f"    ❌ {rid}: {data.get('reason', '?')}")
        return

    # ── Ingest only ───────────────────────────────────────────────────────────
    if args.ingest_only:
        recipe_files = sorted(RECIPES_DIR.glob("*.md"))
        if not recipe_files:
            print("No recipe markdown files found in markdown/recipes/")
            print("Run without --ingest-only to classify first.")
            sys.exit(1)
        ingest_recipes(recipe_files, reset=args.reset)
        return

    # ── Classification ────────────────────────────────────────────────────────
    if args.id:
        md_files = [MARKDOWN / f"{args.id}.md"]
    else:
        md_files = sorted(MARKDOWN.glob("*.md"))
        # Exclude files already in recipes/ subdirectory
        md_files = [f for f in md_files if f.parent == MARKDOWN]

    if not md_files:
        print("No markdown files found. Run transcribe_reels.py first.")
        sys.exit(1)

    print(f"Classifying {len(md_files)} reel(s) with Gemini {GEMINI_MODEL}...\n")

    cache    = load_cache()
    recipes  = []
    skipped  = []
    rejected = []

    for i, md_path in enumerate(md_files, 1):
        reel_id = md_path.stem
        print(f"[{i}/{len(md_files)}] {reel_id}", end="", flush=True)

        # Check cache
        if not args.force and reel_id in cache:
            result = cache[reel_id]
            print(f" [cached] → {'RECIPE' if result.get('is_recipe') else 'skip'}")
        else:
            # Parse markdown
            reel_data = parse_markdown(md_path)
            if not reel_data["transcript"] and not reel_data["caption"]:
                print(f" [no transcript & no caption] → skip")
                skipped.append(reel_id)
                continue

            # Classify with Gemini
            print(" → classifying...", end="", flush=True)
            result = classify_with_gemini(
                reel_data["transcript"], reel_data["caption"], reel_id
            )
            cache[reel_id] = result
            save_cache(cache)

            is_recipe = result.get("is_recipe", False)
            print(f" → {'✅ RECIPE: ' + result.get('recipe_name','?') if is_recipe else '❌ ' + result.get('reason','non-recipe')}")

            # Rate limit: be gentle with Gemini
            time.sleep(1.5)

        result = cache[reel_id]

        if result.get("is_recipe"):
            # Build recipe markdown
            reel_data = parse_markdown(md_path)
            recipe_md = build_recipe_markdown(reel_data, result)
            
            # Sanitize recipe name for filename
            recipe_name = result.get("recipe_name", f"Recipe from {reel_id}")
            sanitized_name = "".join(c if c.isalnum() or c in (" ", "_", "-") else "" for c in recipe_name)
            sanitized_name = sanitized_name.strip().lower().replace(" ", "_")
            filename_stem = sanitized_name if sanitized_name else reel_id
            
            out_path  = RECIPES_DIR / f"{filename_stem}.md"
            out_path.write_text(recipe_md, encoding="utf-8")
            recipes.append(out_path)
        else:
            rejected.append(reel_id)

    # ── Ingest recipes into vector DB ─────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"Classification complete:")
    print(f"  Recipes found  : {len(recipes)}")
    print(f"  Non-recipes    : {len(rejected)}")
    print(f"  Skipped        : {len(skipped)}")

    if recipes:
        ingest_recipes(recipes, reset=args.reset)
        print(f"\nRecipe markdown files saved to: {RECIPES_DIR}")
        print(f"\nTest the recipe knowledge base:")
        print(f"  python rag_chat.py --collection recipes")
    else:
        print("\nNo recipes found in the transcripts.")
        print("Try adding more cooking/recipe reels to urls.txt")


if __name__ == "__main__":
    main()
