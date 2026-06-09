#!/usr/bin/env python3
"""
run_pipeline.py
───────────────
One-command pipeline: Fetch → Download → Transcribe → Classify → Ingest → Chat

Stages:
  1. fetch      — auto-discover all reel URLs from an Instagram profile
  2. download   — yt-dlp downloads from urls.txt
  3. transcribe — faster-whisper → markdown files
  4. classify   — Gemini separates recipe vs non-recipe content
  5. ingest     — PostgreSQL vector DB ingestion (all reels)
  6. chat       — Gemini RAG chatbot

Usage:
    python run_pipeline.py                                    # full pipeline
    python run_pipeline.py --stages fetch,download            # fetch + download only
    python run_pipeline.py --stages classify,ingest,chat      # from classify onwards
    python run_pipeline.py --profile groundup.in --limit 20  # limit to 20 reels
    python run_pipeline.py --model small                      # better transcription
    python run_pipeline.py --reset-db                         # wipe & re-ingest
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Reconfigure stdout to support UTF-8 characters on Windows console
sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).parent

# ANSI colours (Windows 10+ supports these)
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def banner(text: str):
    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}\n")


def success(text: str):
    print(f"{GREEN}  ✔ {text}{RESET}")


def warn(text: str):
    print(f"{YELLOW}  ⚠ {text}{RESET}")


def error(text: str):
    print(f"{RED}  ✘ {text}{RESET}")


def run_stage(label: str, cmd: list[str], cwd: Path) -> bool:
    """Run a subprocess stage. Returns True on success."""
    banner(f"Stage: {label}")
    start = time.time()

    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env={**__import__("os").environ, "PYTHONUTF8": "1", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"},
    )

    elapsed = time.time() - start
    if result.returncode == 0:
        success(f"{label} completed in {elapsed:.1f}s")
        return True
    else:
        error(f"{label} failed (exit code {result.returncode})")
        return False


def check_urls() -> int:
    """Count non-comment URLs in urls.txt."""
    urls_file = BASE_DIR / "urls.txt"
    if not urls_file.exists():
        return 0
    count = 0
    for line in urls_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            count += 1
    return count


def check_downloads() -> list[Path]:
    return sorted((BASE_DIR / "downloads").glob("*.mp4"))


def check_markdowns() -> list[Path]:
    return sorted((BASE_DIR / "markdown").glob("*.md"))


def main():
    parser = argparse.ArgumentParser(
        description="Ground Up Reels Pipeline — Fetch → Download → Transcribe → Classify → Ingest → Chat"
    )
    parser.add_argument(
        "--stages",
        default="fetch,download,transcribe,classify,chat",
        help="Comma-separated stages: fetch,download,transcribe,classify,chat",
    )
    parser.add_argument("--profile", "-p", default="groundup.in",
                        help="Instagram profile to fetch reels from (default: groundup.in)")
    parser.add_argument("--limit", "-n", type=int, default=None,
                        help="Max reels to fetch from profile")
    parser.add_argument("--skip-chat",  action="store_true", help="Skip the chat stage")
    parser.add_argument("--model",      default="base",
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                        help="Whisper model for transcription (default: base)")
    parser.add_argument("--cookies-file",    default="cookies.txt",    help="Path to cookies.txt")
    parser.add_argument("--cookies-browser", default=None,
                        choices=["chrome", "firefox", "edge"],
                        help="Use cookies from browser instead of file")
    parser.add_argument("--reset-db",   action="store_true", help="Wipe and re-ingest vector DB")
    parser.add_argument("--language",   default=None, help="Force transcription language (e.g. en, hi)")
    args = parser.parse_args()

    stages = [s.strip().lower() for s in args.stages.split(",")]
    if args.skip_chat and "chat" in stages:
        stages.remove("chat")

    print(f"\n{BOLD}Ground Up Reels Pipeline{RESET}")
    print(f"  Stages: {' → '.join(stages)}")
    print(f"  Whisper model: {args.model}")

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    url_count = check_urls()
    dl_count  = len(check_downloads())
    md_count  = len(check_markdowns())
    print(f"\n  URLs in urls.txt: {url_count}")
    print(f"  Videos in downloads/: {dl_count}")
    print(f"  Markdown files: {md_count}")

    all_ok = True

    # ── STAGE 0: Fetch profile URLs ───────────────────────────────────────────
    if "fetch" in stages:
        fetch_cmd = [sys.executable, "fetch_profile.py", "--profile", args.profile, "--append"]
        if (BASE_DIR / args.cookies_file).exists():
            fetch_cmd += ["--cookies-file", args.cookies_file]
        else:
            warn("cookies.txt not found — fetch may fail")
        if args.limit:
            fetch_cmd += ["--limit", str(args.limit)]
        ok = run_stage(f"Fetch Profile @{args.profile}", fetch_cmd, BASE_DIR)
        all_ok = all_ok and ok

    # ── STAGE 1: Download ─────────────────────────────────────────────────────
    if "download" in stages:
        if url_count == 0:
            warn("No URLs found in urls.txt — skipping download stage")
        else:
            # Build download command
            dl_cmd = [sys.executable, "download_reels.py"]
            if args.cookies_browser:
                dl_cmd += ["--cookies-browser", args.cookies_browser]
            elif (BASE_DIR / args.cookies_file).exists():
                dl_cmd += ["--cookies-file", args.cookies_file]
            else:
                warn(f"cookies.txt not found — trying without auth (public reels only)")

            ok = run_stage("Download", dl_cmd, BASE_DIR)
            all_ok = all_ok and ok

    # ── STAGE 2: Transcribe ───────────────────────────────────────────────────
    if "transcribe" in stages and all_ok:
        videos = check_downloads()
        if not videos:
            warn("No .mp4 files in downloads/ — skipping transcription")
        else:
            tx_cmd = [sys.executable, "transcribe_reels.py", "--model", args.model]
            if args.language:
                tx_cmd += ["--language", args.language]
            ok = run_stage("Transcribe", tx_cmd, BASE_DIR)
            all_ok = all_ok and ok

    # ── STAGE 3: Ingest (Skipped) ─────────────────────────────────────────────
    if "ingest" in stages and all_ok:
        warn("Raw reels ingestion stage is disabled to focus only on recipe embeddings.")
        print("  Recipes will be ingested during the 'classify' stage.")

    # ── STAGE 4: Classify (recipe vs non-recipe) ──────────────────────────────
    if "classify" in stages and all_ok:
        mds = check_markdowns()
        if not mds:
            warn("No .md files to classify — skipping")
        else:
            classify_cmd = [sys.executable, "classify_reels.py"]
            if args.reset_db:
                classify_cmd.append("--reset")
            ok = run_stage("Classify Reels (Recipe Detection)", classify_cmd, BASE_DIR)
            all_ok = all_ok and ok

    # ── STAGE 5: Chat ─────────────────────────────────────────────────────────
    if "chat" in stages and all_ok:
        banner("Stage: RAG Chat")
        chat_cmd = [sys.executable, "rag_chat.py"]
        subprocess.run(chat_cmd, cwd=str(BASE_DIR),
                       env={**__import__("os").environ, "PYTHONUTF8": "1",
                            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"})

    # ── Summary ───────────────────────────────────────────────────────────────
    if "chat" not in stages:
        banner("Pipeline Complete")
        videos  = check_downloads()
        mds     = check_markdowns()
        recipes = sorted((BASE_DIR / "markdown" / "recipes").glob("*.md")) if (BASE_DIR / "markdown" / "recipes").exists() else []
        print(f"  Videos downloaded   : {len(videos)}")
        print(f"  Markdowns created   : {len(mds)}")
        print(f"  Recipes extracted   : {len(recipes)}")
        print(f"\n  Run the full chatbot:")
        print(f"    python rag_chat.py")
        print(f"  Run the recipe-only chatbot:")
        print(f"    python rag_chat.py --collection recipes\n")


if __name__ == "__main__":
    main()
