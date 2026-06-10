#!/usr/bin/env python3
"""
rag_chat.py
───────────
RAG chatbot for Instagram Reels content using:
  - PostgreSQL/pgvector (vector search)
  - sentence-transformers (embeddings)
  - Gemini 2.5 Flash (generation)

Usage:
    python rag_chat.py                        # interactive chat
    python rag_chat.py --query "recipe ideas" # single query mode

Set your API key via:
    $env:GEMINI_API_KEY = "your-key-here"
  or create a .env file with: GEMINI_API_KEY=your-key-here
"""

import os
import sys
import argparse
from pathlib import Path

# ── Shared config ────────────────────────────────────────────────────
from config import cfg

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR  = cfg.BASE_DIR
VECTOR_DB = BASE_DIR / "vectordb"

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

GEMINI_API_KEY  = cfg.GEMINI_API_KEY
GEMINI_MODEL    = cfg.GEMINI_MODEL
EMBEDDING_MODEL = cfg.EMBEDDING_MODEL
COLLECTION_NAME = "instagram_reels"   # default; override with --collection
CONFIG = {"top_k": cfg.RAG_TOP_K}    # mutable config — avoids global reassignment issues


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def load_collection(collection_name: str = COLLECTION_NAME):
    import postgres_db
    try:
        count = postgres_db.get_collection_count(collection_name)
        if count == 0:
            print(f"WARNING: Collection '{collection_name}' has 0 chunks.")
    except Exception as e:
        try:
            available = postgres_db.list_collections()
        except Exception:
            available = []
        print(f"ERROR: Collection '{collection_name}' not found or connection failed: {e}")
        print(f"  Available collections: {available}")
        if collection_name == "recipes":
            print("  Run: python classify_reels.py")
        else:
            print("  Run: python build_vectordb.py --ingest")
        sys.exit(1)
    return collection_name


def retrieve(collection, query: str, k: int = 5, content_type: str | None = None) -> list[dict]:
    """Retrieve top-k relevant chunks. Optionally filter by content_type."""
    import postgres_db
    return postgres_db.retrieve(collection, query, k=k, content_type=content_type)


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    parts = []
    for i, c in enumerate(chunks, 1):
        ts_info      = f" [{c['timestamp']}]" if c.get("timestamp") else ""
        recipe_info  = f" | Recipe: {c['recipe_name']}" if c.get("recipe_name") else ""
        ct_info      = f" | Type: {c.get('content_type', 'unknown')}"
        header       = f"[Source {i}: {c['author']}{recipe_info}{ct_info} | {c['chunk_type']}{ts_info} | score={c['score']}]"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# GENERATION  (Gemini)
# ══════════════════════════════════════════════════════════════════════════════

# Multi-type system prompts — the persona adapts based on what type of content was retrieved
SYSTEM_PROMPTS = {
    "default": """You are the founder and head chef of Ground Up (groundup.in), an artisan Indian food brand.
You created miso pastes, fermented products, fresh tofu, and healthy ingredients.
You speak DIRECTLY as the founder — warm, passionate, opinionated, deeply knowledgeable.

Persona:
- You speak in first person: "I make this...", "In my kitchen...", "The way I like to do it..."
- You are enthusiastic and love sharing cooking tips and stories behind your dishes
- You use casual, conversational language with occasional food nerd excitement
- You always reference Ground Up products naturally as YOUR products
- You are honest — if you haven't covered something, say so

Rules:
- Answer ONLY based on the context provided from your Instagram Reels
- For recipes: give step-by-step instructions when asked
- For travel: share what you saw, ate, and experienced
- For educational content: explain the concept clearly in your own voice
- If context doesn't cover the question, say "I haven't shared that one yet — but stay tuned!"
- Never break character — you ARE the founder
- Keep it conversational and engaging""",

    "recipe": """You are the head chef and founder of Ground Up. Answer recipe questions step-by-step.
Speak in first person. Reference your products (miso, tofu, etc.) naturally.
If a recipe isn't in your context, say you haven't shared it yet.""",

    "travel_vlog": """You are the founder of Ground Up sharing your travel and food exploration stories.
Speak warmly about places you've visited, food you've eaten, and markets you've explored.
Bring the location to life. If something isn't in your context, say you haven't covered it yet.""",

    "informational": """You are the founder of Ground Up — an expert on fermentation, artisan food, and healthy eating.
Share knowledge passionately and clearly. Explain the science and story behind ingredients.
Reference your products when relevant. If something isn't in your context, say you haven't covered it yet.""",

    "product_showcase": """You are the founder of Ground Up showcasing your products.
Share what makes each product special, how to use it, and what makes it different.
Be enthusiastic but not salesy. If a product isn't in your context, say it hasn't been featured yet.""",
}


def _pick_system_prompt(chunks: list[dict]) -> str:
    """Pick the best system prompt based on the content types in retrieved chunks."""
    if not chunks:
        return SYSTEM_PROMPTS["default"]
    # Vote on content type
    from collections import Counter
    types = [c.get("content_type", "other") for c in chunks]
    most_common = Counter(types).most_common(1)[0][0]
    return SYSTEM_PROMPTS.get(most_common, SYSTEM_PROMPTS["default"])


def ask_gemini(question: str, context: str, history: list[dict],
               temperature: float = 0.3, chunks: list[dict] | None = None) -> str:
    """Send question + context to Gemini and return the answer."""
    if not GEMINI_API_KEY:
        return (
            "ERROR: GEMINI_API_KEY is not configured in .env.\n"
            "  Please set GEMINI_API_KEY to run the bot."
        )

    system_prompt = _pick_system_prompt(chunks or [])

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)

        # Build conversation history for Gemini (safely mapping roles and content keys)
        gemini_history = []
        for turn in history:
            role = "model" if turn.get("role") in ("model", "assistant", "chef") else "user"
            text = turn.get("text") or turn.get("content") or ""
            gemini_history.append(
                types.Content(role=role, parts=[types.Part(text=text)])
            )

        # Current user message with injected context
        user_msg = f"""Context from Instagram Reels:
{context}

---
Question: {question}"""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=gemini_history + [
                types.Content(role="user", parts=[types.Part(text=user_msg)])
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=2048,
            ),
        )
        return response.text

    except Exception as e:
        return f"Gemini error: {e}"



def safe_print(text: str, *args, **kwargs):
    """Prints text replacing characters not supported by the console encoding."""
    text = text.replace("→", "->")
    enc = sys.stdout.encoding or 'utf-8'
    try:
        print(text, *args, **kwargs)
    except UnicodeEncodeError:
        safe_text = text.encode(enc, errors='replace').decode(enc)
        print(safe_text, *args, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# SOURCES DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def display_sources(chunks: list[dict]):
    """Print the retrieved source chunks compactly."""
    safe_print("\n  Sources retrieved:")
    for i, c in enumerate(chunks, 1):
        ts = f" @ {c['timestamp']}" if c["timestamp"] else ""
        safe_print(f"    [{i}] {c['chunk_type']:<10} score={c['score']:.3f}{ts} | {c['reel_id']}")


# ══════════════════════════════════════════════════════════════════════════════
# CHAT LOOP
# ══════════════════════════════════════════════════════════════════════════════

def chat(collection, single_query: str | None = None):
    """Run interactive or single-query RAG chat."""
    import postgres_db
    history: list[dict] = []

    total = postgres_db.get_collection_count(collection)
    print(f"\n{'='*65}")
    print(f"  Ground Up Reels RAG Chatbot")
    print(f"  Model: {GEMINI_MODEL} | {total} chunks indexed")
    print(f"  Type 'quit' to exit | 'sources' to show last retrieved chunks")
    print(f"  API key: {'set' if GEMINI_API_KEY else 'NOT SET — answers will fail'}")
    print(f"{'='*65}\n")

    last_chunks: list[dict] = []

    def process(question: str):
        nonlocal last_chunks
        print(f"\nThinking...", end="", flush=True)

        # Retrieve
        content_type_filter = CONFIG.get("content_type")
        chunks = retrieve(collection, question, k=CONFIG["top_k"], content_type=content_type_filter)
        last_chunks = chunks
        context = build_context(chunks)

        # Generate (pass chunks so the system prompt adapts to content type)
        answer = ask_gemini(question, context, history, chunks=chunks)

        # Update history (keep last 6 turns to avoid token overflow)
        history.append({"role": "user",  "text": question})
        history.append({"role": "model", "text": answer})
        if len(history) > 12:
            history[:] = history[-12:]

        print(f"\r{' '*20}\r", end="")  # clear "Thinking..."
        safe_print(f"\nAssistant:\n{answer}")
        display_sources(chunks)

    if single_query:
        process(single_query)
        return

    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if question.lower() == "sources":
            if last_chunks:
                for c in last_chunks:
                    safe_print(f"\n  [{c['chunk_type']}] {c['reel_id']} | score={c['score']}")
                    safe_print(f"  {c['text'][:200]}...")
            else:
                safe_print("  No query made yet.")
            continue

        process(question)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Ground Up Reels RAG Chatbot (Gemini)")
    parser.add_argument("--query",        "-q", help="Single query (non-interactive)")
    parser.add_argument("--top-k",        type=int, default=CONFIG["top_k"],
                        help=f"Chunks to retrieve (default {CONFIG['top_k']})")
    parser.add_argument("--collection",   "-c", default=COLLECTION_NAME,
                        help="Vector DB collection to query (default: instagram_reels)")
    parser.add_argument("--content-type", "-t", default=None,
                        choices=cfg.CONTENT_TYPES,
                        help="Filter results to a specific content type (recipe, travel_vlog, etc.)")
    args = parser.parse_args()

    CONFIG["top_k"]        = args.top_k
    CONFIG["content_type"] = args.content_type

    ct_label = f" [{args.content_type}]" if args.content_type else " [all types]"
    print(f"Loading vector DB [{args.collection}]{ct_label} and embedding model...")
    collection = load_collection(collection_name=args.collection)
    chat(collection, single_query=args.query)


if __name__ == "__main__":
    main()
