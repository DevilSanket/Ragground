#!/usr/bin/env python3
"""
transcribe_reels.py
───────────────────
Transcribes all downloaded Instagram Reels using faster-whisper
and saves structured Markdown files ready for vector DB ingestion.

Output per reel:
  markdown/<reel_id>.md   — structured markdown with metadata + transcript
  transcripts/<reel_id>.txt — raw transcript only

Usage:
    python transcribe_reels.py                      # transcribe all in downloads/
    python transcribe_reels.py --id DY_sr7kKS0x    # single reel
    python transcribe_reels.py --model large-v3     # use a larger model
"""

import argparse
import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DOWNLOADS   = BASE_DIR / "downloads"
TRANSCRIPTS = BASE_DIR / "transcripts"
MARKDOWN    = BASE_DIR / "markdown"

TRANSCRIPTS.mkdir(exist_ok=True)
MARKDOWN.mkdir(exist_ok=True)

# ─── Model config ─────────────────────────────────────────────────────────────
DEFAULT_MODEL    = "base"       # tiny | base | small | medium | large-v3
DEFAULT_DEVICE   = "cpu"        # cpu | cuda (if you have GPU)
DEFAULT_COMPUTE  = "int8"       # int8 (fast) | float16 (GPU) | float32


def load_metadata(reel_id: str) -> dict:
    """Load yt-dlp metadata JSON for a reel."""
    meta_file = DOWNLOADS / f"{reel_id}.info.json"
    if meta_file.exists():
        with open(meta_file, encoding="utf-8") as f:
            return json.load(f)
    return {}


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def transcribe(video_path: Path, model, language: str | None = None) -> list[dict]:
    """Run faster-whisper on a video file. Returns list of segment dicts."""
    print(f"  Transcribing: {video_path.name}")
    segments, info = model.transcribe(
        str(video_path),
        language=language,
        beam_size=5,
        vad_filter=True,           # remove silence
        vad_parameters={"min_silence_duration_ms": 500},
    )
    print(f"  Detected language: {info.language} (confidence: {info.language_probability:.0%})")

    result = []
    for seg in segments:
        result.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        })
    return result


def build_markdown(reel_id: str, segments: list[dict], meta: dict) -> str:
    """Build a structured markdown document for one reel."""

    # ── Extract metadata ──────────────────────────────────────────────────────
    title       = meta.get("title") or meta.get("description", "")[:80] or f"Reel {reel_id}"
    uploader    = meta.get("uploader") or meta.get("channel", "Unknown")
    upload_date = meta.get("upload_date", "")          # YYYYMMDD
    url         = meta.get("webpage_url", f"https://www.instagram.com/reel/{reel_id}/")
    duration    = meta.get("duration", 0)
    like_count  = meta.get("like_count", "N/A")
    view_count  = meta.get("view_count", "N/A")
    description = meta.get("description", "").strip()
    tags        = meta.get("tags", [])

    # Format date
    if upload_date and len(upload_date) == 8:
        date_str = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"
    else:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Full transcript text
    full_text = " ".join(seg["text"] for seg in segments)

    # ── Build markdown ────────────────────────────────────────────────────────
    lines = []

    # YAML frontmatter
    lines.append("---")
    lines.append(f'reel_id: "{reel_id}"')
    lines.append(f'url: "{url}"')
    lines.append(f'author: "@{uploader}"')
    lines.append(f'date: "{date_str}"')
    lines.append(f'duration_seconds: {int(duration)}')
    lines.append(f'likes: {like_count}')
    lines.append(f'views: {view_count}')
    if tags:
        tags_str = ", ".join(f'"{t}"' for t in tags[:10])
        lines.append(f'tags: [{tags_str}]')
    lines.append(f'transcribed_at: "{datetime.now(timezone.utc).isoformat()}"')
    lines.append("---")
    lines.append("")

    # Title
    lines.append(f"# {title}")
    lines.append("")

    # Meta block
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- **Author:** @{uploader}")
    lines.append(f"- **Date:** {date_str}")
    lines.append(f"- **Duration:** {format_timestamp(duration)}")
    lines.append(f"- **URL:** {url}")
    if like_count != "N/A":
        lines.append(f"- **Likes:** {like_count:,}" if isinstance(like_count, int) else f"- **Likes:** {like_count}")
    lines.append("")

    # Description (caption)
    if description:
        lines.append("## Caption")
        lines.append("")
        lines.append(description)
        lines.append("")

    # Full transcript
    lines.append("## Transcript")
    lines.append("")
    lines.append(full_text)
    lines.append("")

    # Timestamped segments
    if segments:
        lines.append("## Timestamped Segments")
        lines.append("")
        for seg in segments:
            start = format_timestamp(seg["start"])
            end   = format_timestamp(seg["end"])
            lines.append(f"**[{start} → {end}]** {seg['text']}")
            lines.append("")

    return "\n".join(lines)


def process_reel(reel_id: str, model, language: str | None = None) -> bool:
    """Transcribe one reel and save markdown + raw transcript."""
    md_path  = MARKDOWN    / f"{reel_id}.md"
    txt_path = TRANSCRIPTS / f"{reel_id}.txt"

    if md_path.exists():
        print(f"  [SKIP] {reel_id} already processed — delete .md to redo")
        return True

    video_path = DOWNLOADS / f"{reel_id}.mp4"
    segments = []

    if video_path.exists():
        # Transcribe
        segments = transcribe(video_path, model, language)
        if not segments:
            print(f"  [WARN] No speech detected in {reel_id}")
    else:
        print(f"  [INFO] No video file (.mp4) found for {reel_id}. Using caption/metadata only.")

    # Load metadata
    meta = load_metadata(reel_id)

    # Save raw transcript
    raw_text = "\n".join(seg["text"] for seg in segments)
    txt_path.write_text(raw_text, encoding="utf-8")
    print(f"  Saved: transcripts/{reel_id}.txt")

    # Save markdown
    md_content = build_markdown(reel_id, segments, meta)
    md_path.write_text(md_content, encoding="utf-8")
    print(f"  Saved: markdown/{reel_id}.md")

    return True


def main():
    parser = argparse.ArgumentParser(description="Transcribe Instagram Reels to Markdown")
    parser.add_argument("--id",       help="Transcribe a single reel by ID (e.g. DY_sr7kKS0x)")
    parser.add_argument("--model",    default=DEFAULT_MODEL,
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                        help=f"Whisper model size (default: {DEFAULT_MODEL})")
    parser.add_argument("--language", default=None,
                        help="Force language code e.g. 'en', 'hi'. Auto-detect if omitted.")
    parser.add_argument("--device",   default=DEFAULT_DEVICE, choices=["cpu", "cuda"],
                        help="Device to run on (default: cpu)")
    args = parser.parse_args()

    # ── Find reels to process ─────────────────────────────────────────────────
    if args.id:
        reel_ids = [args.id]
    else:
        stems = set(p.stem for p in DOWNLOADS.glob("*.mp4")).union(
            p.name[:-10] for p in DOWNLOADS.glob("*.info.json") if p.name.endswith(".info.json")
        )
        reel_ids = sorted(list(stems))

    if not reel_ids:
        print("No downloaded files (.mp4 or .info.json) found in downloads/. Run download_reels.py first.")
        sys.exit(1)

    # ── Load model (only if we need to transcribe any .mp4 files) ─────────────
    needs_whisper = any((DOWNLOADS / f"{rid}.mp4").exists() for rid in reel_ids)
    model = None
    if needs_whisper:
        print(f"\nLoading faster-whisper model: {args.model} on {args.device}...")
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("ERROR: faster-whisper not installed. Run: pip install faster-whisper")
            sys.exit(1)
        compute_type = "float16" if args.device == "cuda" else "int8"
        model = WhisperModel(args.model, device=args.device, compute_type=compute_type)
        print(f"Model loaded.\n{'─'*50}")
    else:
        print(f"\nNo video files (.mp4) to transcribe. Skipping model loading.\n{'─'*50}")

    print(f"Found {len(reel_ids)} reel(s)/post(s) to process: {', '.join(reel_ids)}\n")

    # ── Process each reel ─────────────────────────────────────────────────────
    success, failed = 0, 0
    for i, reel_id in enumerate(reel_ids, 1):
        print(f"[{i}/{len(reel_ids)}] Processing: {reel_id}")
        try:
            ok = process_reel(reel_id, model, args.language)
            if ok:
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [ERROR] {reel_id}: {e}")
            failed += 1
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 50)
    print(f"Done! {success} transcribed, {failed} failed/skipped")
    print(f"Markdown files: {MARKDOWN}")
    print(f"Raw transcripts: {TRANSCRIPTS}")


if __name__ == "__main__":
    main()
