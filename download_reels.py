#!/usr/bin/env python3
"""
download_reels.py
─────────────────
Batch download Instagram Reels using yt-dlp.
Reads URLs from urls.txt and saves videos + metadata JSON.

Usage:
    python download_reels.py                  # uses urls.txt
    python download_reels.py --url <URL>      # single reel
    python download_reels.py --cookies-browser chrome
"""

import subprocess
import argparse
import sys
import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
URLS_FILE   = BASE_DIR / "urls.txt"
DOWNLOADS   = BASE_DIR / "downloads"
CONFIG_FILE = BASE_DIR / "yt-dlp.conf"

DOWNLOADS.mkdir(exist_ok=True)


def build_command(urls: list[str], cookies_browser: str | None, cookies_file: str | None) -> list[str]:
    import sys
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--config-location", str(CONFIG_FILE),
    ]

    # Auth: cookies
    if cookies_browser:
        cmd += ["--cookies-from-browser", cookies_browser]
        print(f"🍪  Using cookies from browser: {cookies_browser}")
    elif cookies_file:
        cmd += ["--cookies", cookies_file]
        print(f"🍪  Using cookies file: {cookies_file}")
    else:
        print("⚠️  No cookies set — public reels only. Private/auth-gated content will fail.")

    cmd += urls
    return cmd


def load_urls_from_file(path: Path) -> list[str]:
    if not path.exists():
        print(f"❌ urls.txt not found at {path}")
        sys.exit(1)
    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def download(urls: list[str], cookies_browser: str | None = None, cookies_file: str | None = None):
    if not urls:
        print("❌ No URLs provided. Add URLs to urls.txt or pass --url <URL>")
        sys.exit(1)

    print(f"\n📥 Downloading {len(urls)} reel(s)...\n{'─'*50}")

    cmd = build_command(urls, cookies_browser, cookies_file)
    print(f"▶  Running: {' '.join(cmd[:4])} ... [{len(urls)} URL(s)]\n")

    result = subprocess.run(cmd, cwd=str(BASE_DIR))

    if result.returncode == 0:
        print(f"\n✅ Download complete! Files saved to: {DOWNLOADS}")
        # List downloaded files
        files = list(DOWNLOADS.glob("*.mp4")) + list(DOWNLOADS.glob("*.webm"))
        print(f"📁 {len(files)} video file(s) found in downloads/")
    else:
        print(f"\n⚠️  yt-dlp exited with code {result.returncode}")
        print("   Check the output above for details.")


def main():
    parser = argparse.ArgumentParser(
        description="Batch download Instagram Reels with yt-dlp"
    )
    parser.add_argument("--url", nargs="+", help="One or more Reel URLs (overrides urls.txt)")
    parser.add_argument("--cookies-browser", choices=["chrome", "firefox", "edge", "safari", "brave"],
                        help="Pull cookies from an installed browser (requires browser to be closed or bg)")
    parser.add_argument("--cookies-file", help="Path to exported cookies.txt (Netscape format)")
    args = parser.parse_args()

    # Resolve URLs
    if args.url:
        urls = args.url
        print(f"🔗 Single/inline mode: {len(urls)} URL(s)")
    else:
        urls = load_urls_from_file(URLS_FILE)
        print(f"📄 Loaded {len(urls)} URL(s) from urls.txt")

    download(urls, cookies_browser=args.cookies_browser, cookies_file=args.cookies_file)


if __name__ == "__main__":
    main()
