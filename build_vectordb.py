#!/usr/bin/env python3
"""
build_vectordb.py
─────────────────
Ingest transcribed Reel markdown files into PostgreSQL (pgvector) for semantic search.

What it does:
  1. Parses each .md file in markdown/
  2. Splits into smart chunks (caption, full transcript, timestamped segments)
  3. Embeds with sentence-transformers (all-MiniLM-L6-v2, runs locally)
  4. Stores in PostgreSQL (using the pgvector extension)
  5. Runs an interactive search loop to test queries

Usage:
    python build_vectordb.py            # ingest + search
    python build_vectordb.py --ingest   # ingest only
    python build_vectordb.py --search   # search only (DB must exist)
    python build_vectordb.py --reset    # wipe DB and re-ingest
"""

import argparse
import re
import sys
from pathlib import Path

# ── Shared config ──────────────────────────────────────────────────────
from config import cfg

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR  = cfg.BASE_DIR
MARKDOWN  = cfg.MARKDOWN_DIR
VECTOR_DB = BASE_DIR / "vectordb"

COLLECTION_NAME  = "instagram_reels"
EMBEDDING_MODEL  = cfg.EMBEDDING_MODEL


# ══════════════════════════════════════════════════════════════════════════════
# PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_frontmatter(text: str) -> dict:
    """Extract YAML frontmatter as a flat dict of strings."""
    meta = {}
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return meta
    for line in m.group(1).splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip().strip('"')
    return meta


def extract_section(text: str, heading: str) -> str:
    """Extract content under a ## heading."""
    pattern = rf"## {re.escape(heading)}\n+(.*?)(?=\n## |\Z)"
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_timestamped_segments(text: str) -> list[dict]:
    """Parse '**[MM:SS → MM:SS]** text' lines into segment dicts."""
    segments = []
    pattern = r"\*\[\s*(\d+:\d+)\s*→\s*(\d+:\d+)\]\*\*\s+(.+)"
    for m in re.finditer(pattern, text):
        segments.append({
            "start": m.group(1),
            "end":   m.group(2),
            "text":  m.group(3).strip(),
        })
    return segments


def chunk_markdown(md_path: Path) -> list[dict]:
    """
    Parse a reel markdown file and return a list of chunks.
    Each chunk = {"id": str, "text": str, "metadata": dict}

    Chunk types:
      - caption      : Instagram caption text
      - transcript   : full transcript as one chunk
      - segment_N    : individual timestamped segment (if > 20 words)
    """
    text = md_path.read_text(encoding="utf-8")
    meta = parse_frontmatter(text)
    reel_id = meta.get("reel_id", md_path.stem)

    base_meta = {
        "reel_id":      reel_id,
        "author":       meta.get("author", ""),
        "date":         meta.get("date", ""),
        "url":          meta.get("url", ""),
        "likes":        meta.get("likes", ""),
        "duration":     meta.get("duration_seconds", ""),
        "source":       str(md_path.name),
        "content_type": meta.get("content_type", "other"),  # ← new: enables filtered search
    }

    chunks = []

    # ── 1. Caption chunk ──────────────────────────────────────────────────────
    caption = extract_section(text, "Caption")
    if caption and len(caption.split()) >= 5:
        chunks.append({
            "id":       f"{reel_id}__caption",
            "text":     caption,
            "metadata": {**base_meta, "chunk_type": "caption"},
        })

    # ── 2. Full transcript chunk ───────────────────────────────────────────────
    transcript = extract_section(text, "Transcript")
    if transcript and len(transcript.split()) >= 10:
        chunks.append({
            "id":       f"{reel_id}__transcript",
            "text":     transcript,
            "metadata": {**base_meta, "chunk_type": "transcript"},
        })

    # ── 3. Individual timestamped segments ────────────────────────────────────
    ts_section = extract_section(text, "Timestamped Segments")
    segments = parse_timestamped_segments(ts_section)
    for i, seg in enumerate(segments):
        if len(seg["text"].split()) < 6:   # skip very short segments
            continue
        chunks.append({
            "id":   f"{reel_id}__seg_{i:03d}",
            "text": seg["text"],
            "metadata": {
                **base_meta,
                "chunk_type": "segment",
                "timestamp":  f"{seg['start']} → {seg['end']}",
            },
        })

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# INGESTION
# ══════════════════════════════════════════════════════════════════════════════

def get_collection(reset: bool = False):
    """Initialize PostgreSQL database and return collection name."""
    import postgres_db
    postgres_db.init_db(reset=reset, collection_name=COLLECTION_NAME)
    return COLLECTION_NAME


def ingest(collection, reset: bool = False):
    """Parse all markdown files and upsert chunks into PostgreSQL."""
    import postgres_db
    md_files = sorted(MARKDOWN.glob("*.md"))
    if not md_files:
        print(f"No .md files found in {MARKDOWN}. Run transcribe_reels.py first.")
        sys.exit(1)

    print(f"\nFound {len(md_files)} markdown file(s): {[f.name for f in md_files]}")

    all_chunks = []
    for md_file in md_files:
        chunks = chunk_markdown(md_file)
        print(f"  {md_file.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks to ingest: {len(all_chunks)}")

    # Check existing IDs to skip duplicates (unless reset)
    existing_ids = set()
    if not reset:
        try:
            existing_ids = postgres_db.get_existing_ids(collection)
        except Exception:
            pass

    new_chunks = [c for c in all_chunks if c["id"] not in existing_ids]
    if not new_chunks:
        print("All chunks already in DB. Use --reset to re-ingest.")
        return len(all_chunks)

    print(f"Ingesting {len(new_chunks)} new chunks (skipping {len(all_chunks) - len(new_chunks)} existing)...")

    # Batch upsert
    BATCH = 50
    for i in range(0, len(new_chunks), BATCH):
        batch = new_chunks[i:i+BATCH]
        postgres_db.upsert_chunks(
            collection_name=collection,
            ids       = [c["id"] for c in batch],
            documents = [c["text"] for c in batch],
            metadatas = [c["metadata"] for c in batch],
        )
        print(f"  Upserted batch {i//BATCH + 1}/{(len(new_chunks)-1)//BATCH + 1}")

    count = postgres_db.get_collection_count(collection)
    print(f"\nDone! Collection '{COLLECTION_NAME}' now has {count} chunks total.")
    return count


# ══════════════════════════════════════════════════════════════════════════════
# SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def format_result(rank: int, doc: str, meta: dict, score: float) -> str:
    """Pretty-print one search result."""
    reel_url = meta.get("url", "")
    author   = meta.get("author", "")
    ctype    = meta.get("chunk_type", "")
    ts       = meta.get("timestamp", "")

    lines = [
        f"\n{'-'*60}",
        f"  #{rank}  [{ctype.upper()}]  Score: {score:.3f}",
        f"  Author: {author}  |  Date: {meta.get('date','')}",
        f"  URL: {reel_url}",
    ]
    if ts:
        lines.append(f"  Timestamp: {ts}")
    lines.append(f"\n  {doc[:400]}{'...' if len(doc) > 400 else ''}")
    return "\n".join(lines)


def search(collection, query: str, n_results: int = 3, chunk_type: str | None = None):
    """Run a semantic search and print results from PostgreSQL."""
    import postgres_db
    chunks = postgres_db.retrieve(collection, query, k=n_results, chunk_type=chunk_type)

    print(f"\nQuery: \"{query}\"")
    print(f"Top {len(chunks)} results:")

    for rank, c in enumerate(chunks, 1):
        meta = {
            "url":        c["url"],
            "author":     c["author"],
            "chunk_type": c["chunk_type"],
            "timestamp":  c["timestamp"],
            "date":       c["date"]
        }
        print(format_result(rank, c["text"], meta, c["score"]))

    print(f"\n{'-'*60}")
    return chunks


def interactive_search(collection):
    """Run an interactive search REPL."""
    import postgres_db
    total = postgres_db.get_collection_count(collection)
    print(f"\n{'='*60}")
    print(f"  Reels Vector DB — {total} chunks indexed")
    print(f"  Model: {EMBEDDING_MODEL}")
    print(f"  Commands: 'quit' to exit, 'n=5' to change result count")
    print(f"            'type=transcript' to filter by chunk type")
    print(f"            (types: transcript, caption, segment)")
    print(f"{'='*60}\n")

    n_results  = 3
    chunk_type = None

    while True:
        try:
            raw = input("Search > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        # Parse special commands inline
        query = raw
        for token in raw.split():
            if token.startswith("n="):
                try:
                    n_results = int(token[2:])
                    query = query.replace(token, "").strip()
                    print(f"  [Results set to {n_results}]")
                except ValueError:
                    pass
            elif token.startswith("type="):
                chunk_type = token[5:] or None
                query = query.replace(token, "").strip()
                print(f"  [Filter: chunk_type = {chunk_type}]")

        if not query:
            continue

        try:
            search(collection, query, n_results=n_results, chunk_type=chunk_type)
        except Exception as e:
            print(f"  Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build and query the Reels Vector DB")
    parser.add_argument("--ingest", action="store_true", help="Ingest markdown files only")
    parser.add_argument("--search", action="store_true", help="Search only (skip ingestion)")
    parser.add_argument("--reset",  action="store_true", help="Wipe and re-ingest the DB")
    args = parser.parse_args()

    collection = get_collection(reset=args.reset)

    if not args.search:
        ingest(collection, reset=args.reset)

    if not args.ingest:
        interactive_search(collection)


if __name__ == "__main__":
    main()

