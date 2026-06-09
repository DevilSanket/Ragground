#!/usr/bin/env python3
"""Test retrieval part of RAG (no API key needed)."""
import os, sys
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from pathlib import Path
import postgres_db

questions = [
    "What is the secret ingredient in the noodle recipe?",
    "How should I prepare prawns before cooking?",
    "What products does Ground Up sell?",
]

for q in questions:
    print(f"\nQ: {q}")
    chunks = postgres_db.retrieve("instagram_reels", q, k=3)
    print("  Top chunks:")
    for c in chunks:
        ts = f"[{c['timestamp']}] " if c['timestamp'] else ""
        print(f"    score={c['score']:.3f} | {c['chunk_type']:<10} | {ts}{c['text'][:100]}...")

