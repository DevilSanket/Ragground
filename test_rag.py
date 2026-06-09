#!/usr/bin/env python3
"""Test the full RAG pipeline with Gemini — runs 3 questions automatically."""
import os, sys
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from pathlib import Path

BASE = Path(__file__).parent

# Load .env
_env = BASE / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY or GEMINI_API_KEY == "your-gemini-api-key-here":
    print("ERROR: GEMINI_API_KEY not set in .env")
    sys.exit(1)

print(f"API key loaded: {GEMINI_API_KEY[:8]}...")

import postgres_db
from google import genai
from google.genai import types

# Load vector DB
count = postgres_db.get_collection_count("instagram_reels")
print(f"Vector DB loaded: {count} chunks\n")

# Gemini client
gemini = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM = """You are a helpful assistant for Ground Up (groundup.in), an Indian food brand.
Answer questions based ONLY on the Instagram Reel transcripts provided as context.
Be concise, friendly, and cite the source reel when relevant."""

def ask(question: str) -> str:
    # Retrieve
    chunks = postgres_db.retrieve("instagram_reels", question, k=4)
    context_parts = []
    for c in chunks:
        ts = f" [{c['timestamp']}]" if c['timestamp'] else ""
        context_parts.append(f"[{c['chunk_type']}{ts} | score={c['score']:.2f}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_parts)

    # Generate
    resp = gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Content(role="user", parts=[types.Part(
            text=f"Context:\n{context}\n\nQuestion: {question}"
        )])],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=0.3,
            max_output_tokens=512,
        ),
    )
    return resp.text

# Test questions
questions = [
    "What is the secret ingredient in the noodle soup recipe and why is it important?",
    "How should I prepare sweet water prawns before cooking them?",
    "What products can I buy from Ground Up and where?",
]

for i, q in enumerate(questions, 1):
    print(f"{'='*60}")
    print(f"Q{i}: {q}")
    print(f"{'─'*60}")
    answer = ask(q)
    print(f"A: {answer}")
    print()
