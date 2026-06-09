#!/usr/bin/env python3
"""
api.py — Flask REST API for Ground Up RAG Frontend
────────────────────────────────────────────────────
Bridges the React frontend with the rag_chat.py / postgres_db.py backend.

Endpoints:
  GET  /api/health          — Health check + DB status
  POST /api/chat            — RAG query → answer + sources
  GET  /api/recipes         — List recipe titles (from markdown/ dir)
  GET  /api/collections     — List available vector DB collections

Run: python api.py
"""

import os
import sys
import json
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
_env = BASE_DIR / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(BASE_DIR))

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"])

# Lazy-loaded RAG modules (avoid startup cost if DB not ready)
_rag = None
_pg  = None

def get_rag():
    global _rag
    if _rag is None:
        import rag_chat
        _rag = rag_chat
    return _rag

def get_pg():
    global _pg
    if _pg is None:
        import postgres_db
        _pg = postgres_db
    return _pg


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    status = {"status": "ok", "db": "unknown", "collections": []}
    try:
        pg = get_pg()
        collections = pg.list_collections()
        status["db"] = "connected"
        status["collections"] = collections
    except Exception as e:
        status["db"] = f"error: {e}"
    return jsonify(status)


# ── Chat ──────────────────────────────────────────────────────────────────────
@app.post("/api/chat")
def chat():
    body = request.get_json(force=True, silent=True) or {}
    query       = body.get("query", "").strip()
    collection  = body.get("collection", "recipes")
    top_k       = int(body.get("top_k", 5))
    temperature = float(body.get("temperature", 0.3))
    history     = body.get("history", [])

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        rag = get_rag()
        pg  = get_pg()

        # Load / validate collection
        try:
            rag.load_collection(collection_name=collection)
        except SystemExit:
            return jsonify({
                "answer": f"⚠️ Collection '{collection}' not found in the database. "
                          f"Please run the ingestion pipeline first.",
                "sources": []
            })

        # Retrieve relevant chunks
        chunks = pg.retrieve(collection, query, k=top_k)
        context = rag.build_context(chunks)

        # Build LLM history
        llm_history = []
        for turn in history[-8:]:
            role = turn.get("role", "user")
            text = turn.get("text", "")
            if role in ("chef", "model", "assistant"):
                llm_history.append({"role": "model", "text": text})
            else:
                llm_history.append({"role": "user", "text": text})

        # Generate answer
        answer = rag.ask_gemini(query, context, llm_history, temperature=temperature)

        # Build source dicts (clean for JSON serialisation)
        sources = []
        seen = set()
        for c in chunks:
            rid = c.get("reel_id", "")
            if rid and rid in seen:
                continue
            seen.add(rid)
            sources.append({
                "reel_id":    rid,
                "text":       c.get("text", "")[:500],
                "url":        c.get("url", ""),
                "score":      c.get("score", 0),
                "date":       c.get("date", ""),
                "likes":      c.get("likes", ""),
                "chunk_type": c.get("chunk_type", ""),
                "author":     c.get("author", ""),
            })

        return jsonify({"answer": answer, "sources": sources})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Recipes list ──────────────────────────────────────────────────────────────
@app.get("/api/recipes")
def recipes():
    """Return list of recipe titles parsed from the markdown/recipes directory."""
    recipes_dir = BASE_DIR / "markdown" / "recipes"
    result = []

    TAGS_MAP = {
        "egg": ["eggs"], "miso": ["miso"], "soup": ["soup"], "devilled": ["appetiser"],
        "scrambled": ["breakfast"], "paneer": ["paneer", "dip"], "chocolate": ["dessert"],
        "noodle": ["noodles", "quick"], "ramen": ["ramen"], "tofu": ["tofu", "vegetarian"],
        "prawn": ["seafood"], "broccoli": ["vegetarian", "sides"], "potato": ["vegetarian", "sides"],
        "pasta": ["pasta"], "caramel": ["dessert", "sweet"], "butter": ["condiment"],
    }

    EMOJI_MAP = {
        "egg": "🥚", "miso": "🫙", "soup": "🍲", "chocolate": "🍫",
        "noodle": "🍜", "ramen": "🍲", "tofu": "🥛", "prawn": "🍤",
        "broccoli": "🥦", "potato": "🍠", "pasta": "🍝", "caramel": "🍮",
        "dip": "🧀", "paneer": "🧀", "pumpkin": "🎃", "garlic": "🧄",
    }

    TIME_MAP = {
        "easy": "10-20 min", "medium": "30-45 min", "hard": "60+ min"
    }

    if recipes_dir.exists():
        for md_file in sorted(recipes_dir.glob("*.md")):
            name = md_file.stem.replace("_", " ").replace("-", " ").title()
            low  = name.lower()
            tags = ["miso"]
            for kw, t in TAGS_MAP.items():
                if kw in low:
                    tags.extend(t)
            tags = list(dict.fromkeys(tags))  # deduplicate

            emoji = "🍽️"
            for kw, e in EMOJI_MAP.items():
                if kw in low:
                    emoji = e
                    break

            result.append({
                "title":      name,
                "tags":       tags,
                "emoji":      emoji,
                "time":       "20-30 min",
                "difficulty": "Easy",
                "file":       md_file.name,
            })

    return jsonify({"recipes": result, "total": len(result)})


# ── Collections ───────────────────────────────────────────────────────────────
@app.get("/api/collections")
def collections():
    try:
        pg = get_pg()
        cols = pg.list_collections()
        counts = {}
        for c in cols:
            try:
                counts[c] = pg.get_collection_count(c)
            except Exception:
                counts[c] = -1
        return jsonify({"collections": cols, "counts": counts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("API_PORT", 5050))
    print(f"\n🍳  Ground Up RAG API  →  http://localhost:{port}")
    print(f"   CORS origins: http://localhost:5173, http://localhost:3000\n")
    app.run(host="0.0.0.0", port=port, debug=True)
