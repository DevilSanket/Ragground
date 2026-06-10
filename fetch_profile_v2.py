#!/usr/bin/env python3
"""
fetch_profile_v2.py
───────────────────
Fetch all video/reel URLs from an Instagram profile using instaloader.
This replaces the brittle requests+GraphQL approach in fetch_profile.py.

Why instaloader?
  - Actively maintained (unlike custom GraphQL hashes that break every few weeks)
  - Handles session auth, rate limiting, and pagination reliably
  - Returns rich metadata: caption, likes, views, typename, date

Setup (one-time):
    pip install instaloader
    instaloader --login <your_ig_username>   # saves a session file locally

Usage:
    python fetch_profile_v2.py --profile groundup.in
    python fetch_profile_v2.py --profile groundup.in --limit 30 --append
    python fetch_profile_v2.py --profile groundup.in --dry-run
    python fetch_profile_v2.py --profile groundup.in --no-login  # public only
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from config import cfg

BASE_DIR  = cfg.BASE_DIR
URLS_FILE = cfg.URLS_FILE

# ── Instaloader availability check ────────────────────────────────────────────

def _check_instaloader():
    try:
        import instaloader
        return instaloader
    except ImportError:
        print("ERROR: instaloader is not installed.")
        print("  Run: pip install instaloader")
        print("\nFirst-time setup:")
        print("  instaloader --login <your_instagram_username>")
        print("  (This saves a session file so you don't need to log in each time)")
        sys.exit(1)


# ── Session loading ────────────────────────────────────────────────────────────

def build_loader(ig_username: str | None = None, use_login: bool = True):
    """Build and return a configured Instaloader instance."""
    IL = _check_instaloader()

    loader = IL.Instaloader(
        # We don't want instaloader to download files — yt-dlp handles that
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
        post_metadata_txt_pattern="",
        # Quiet mode to suppress instaloader's own output
        quiet=True,
        # Respect rate limits
        request_timeout=15,
    )

    if use_login:
        username = ig_username or cfg.IG_USERNAME
        if not username:
            print("WARNING: No Instagram username set.")
            print("  Set IG_USERNAME in .env or pass --ig-user <username>")
            print("  Continuing without login (public reels only, higher rate limit risk)")
            return loader

        try:
            loader.load_session_from_file(username)
            print(f"✔ Loaded session for @{username}")
        except FileNotFoundError:
            print(f"WARNING: No saved session found for @{username}")
            print(f"  Run this once: instaloader --login {username}")
            print("  Continuing without auth (public content only)")
        except Exception as e:
            print(f"WARNING: Session load failed: {e}")
            print("  Continuing without auth")

    return loader


# ── Profile fetching ───────────────────────────────────────────────────────────

def fetch_reels(
    profile_name: str,
    loader,
    limit: int | None = None,
    include_types: list[str] | None = None,
) -> list[dict]:
    """
    Fetch video posts from an Instagram profile.

    Returns a list of dicts:
        {id, url, title, date, likes, views, typename, content_type_hint}
    """
    IL = _check_instaloader()

    print(f"\nFetching profile @{profile_name}...")

    try:
        profile = IL.Profile.from_username(loader.context, profile_name)
    except IL.exceptions.ProfileNotExistsException:
        print(f"ERROR: Profile @{profile_name} does not exist or is not accessible.")
        sys.exit(1)
    except IL.exceptions.LoginRequiredException:
        print(f"ERROR: Profile @{profile_name} is private. Login required.")
        print(f"  Run: instaloader --login <your_username>")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR fetching profile: {e}")
        sys.exit(1)

    print(f"  Profile  : {profile.full_name} (@{profile_name})")
    print(f"  Followers: {profile.followers:,}")
    print(f"  Posts    : {profile.mediacount}")
    print()

    reels = []
    skipped = 0

    try:
        posts_iter = profile.get_posts()

        for post in posts_iter:
            # Only video posts (reels are videos)
            if not post.is_video:
                skipped += 1
                continue

            shortcode = post.shortcode
            url = f"https://www.instagram.com/p/{shortcode}/"
            caption = (post.caption or "").replace("\n", " ")[:120]
            date_str = str(post.date_local.date()) if post.date_local else ""

            try:
                views = post.video_view_count or 0
            except Exception:
                views = 0

            try:
                likes = post.likes or 0
            except Exception:
                likes = 0

            # Guess content type from caption keywords (for url.txt annotation)
            content_hint = _guess_content_type(caption)

            reel = {
                "id":               shortcode,
                "url":              url,
                "title":            caption,
                "date":             date_str,
                "likes":            likes,
                "views":            views,
                "typename":         post.typename,
                "content_type_hint": content_hint,
            }
            reels.append(reel)

            print(
                f"  [{len(reels):>3}] {shortcode}  "
                f"👁 {views:>8,}  ❤ {likes:>6,}  "
                f"[{content_hint}]  {caption[:45]}"
            )

            if limit and len(reels) >= limit:
                print(f"\n  Reached limit of {limit} reels.")
                break

            # Polite delay between posts to avoid rate limiting
            time.sleep(0.8)

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
    except Exception as e:
        print(f"\n  Error while iterating posts: {e}")
        print(f"  Collected {len(reels)} reels before error.")

    print(f"\n  Total video posts found: {len(reels)} | Skipped non-videos: {skipped}")
    return reels


def _guess_content_type(caption: str) -> str:
    """
    Heuristic content type guess from caption text.
    Used only for URL file annotations — actual classification uses Gemini.
    """
    low = caption.lower()

    recipe_kws     = ["recipe", "ingredients", "how to make", "cook", "tablespoon",
                      "cup of", "tsp", "tbsp", "stir", "fry", "bake", "boil", "simmer",
                      "miso", "tofu", "paneer", "dal", "roti", "sabzi", "masala"]
    travel_kws     = ["travel", "vlog", "visited", "trip", "journey", "place",
                      "city", "street food", "market", "explore", "tour"]
    info_kws       = ["did you know", "fact", "learn", "tip", "why", "how does",
                      "benefits", "ferment", "process", "science", "story behind",
                      "about us", "brand", "origin", "history"]
    product_kws    = ["new product", "launch", "available now", "shop", "link in bio",
                      "order now", "miso butter", "seaweed", "available at"]

    score = {
        "recipe":           sum(1 for k in recipe_kws  if k in low),
        "travel_vlog":      sum(1 for k in travel_kws  if k in low),
        "informational":    sum(1 for k in info_kws    if k in low),
        "product_showcase": sum(1 for k in product_kws if k in low),
    }

    best = max(score, key=score.get)
    return best if score[best] > 0 else "other"


# ── URLs file management ───────────────────────────────────────────────────────

def load_existing_urls() -> set[str]:
    if not URLS_FILE.exists():
        return set()
    return {
        line.split()[0].strip()   # strip inline comments
        for line in URLS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def save_urls(reels: list[dict], append: bool, profile: str) -> tuple[int, int]:
    """Save reel URLs to urls.txt with content-type annotations."""
    existing  = load_existing_urls() if append else set()
    new_reels = [r for r in reels if r["url"] not in existing]
    skipped   = len(reels) - len(new_reels)

    if not new_reels:
        return 0, skipped

    ts = datetime.now().strftime("%Y-%m-%d")

    if append and URLS_FILE.exists():
        with open(URLS_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n# Fetched from @{profile} on {ts} ({len(new_reels)} new)\n")
            for r in new_reels:
                comment = f"  # [{r['content_type_hint']}] {r['title'][:55]}"
                f.write(f"{r['url']}{comment}\n")
    else:
        lines = [
            f"# Fetched from @{profile} on {ts}",
            f"# Total video reels: {len(reels)}",
            "# Content-type hints are auto-detected (Gemini will reclassify accurately)",
            "# ─────────────────────────────────────",
            "",
        ]
        for r in reels:
            comment = f"  # [{r['content_type_hint']}] {r['title'][:55]}"
            lines.append(f"{r['url']}{comment}")
        URLS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return len(new_reels), skipped


# ── Metadata export ────────────────────────────────────────────────────────────

def save_metadata(reels: list[dict], profile: str):
    """Save full reel metadata as JSON for downstream use."""
    meta_file = BASE_DIR / f"profile_meta_{profile.replace('.', '_')}.json"
    data = {
        "profile":    profile,
        "fetched_at": datetime.now().isoformat(),
        "total":      len(reels),
        "reels":      reels,
    }
    meta_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Metadata saved → {meta_file.name}")
    return meta_file


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch Instagram profile reels using instaloader"
    )
    parser.add_argument("--profile",  "-p", default=cfg.IG_DEFAULT_PROFILE,
                        help=f"Instagram profile name (default: {cfg.IG_DEFAULT_PROFILE})")
    parser.add_argument("--limit",    "-n", type=int, default=None,
                        help="Max reels to fetch (default: all)")
    parser.add_argument("--append",   "-a", action="store_true",
                        help="Append new URLs to existing urls.txt (skip duplicates)")
    parser.add_argument("--ig-user",  default=None,
                        help="Instagram username for session auth (default: IG_USERNAME from .env)")
    parser.add_argument("--no-login", action="store_true",
                        help="Skip login (public posts only, more rate limit risk)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Fetch and print, but do NOT write urls.txt")
    parser.add_argument("--save-meta", action="store_true",
                        help="Also save full metadata JSON file")
    args = parser.parse_args()

    profile = args.profile.lstrip("@")

    print(f"\n{'='*55}")
    print(f"  Ground Up — Instagram Profile Fetcher v2 (instaloader)")
    print(f"  Profile : @{profile}")
    print(f"  Limit   : {args.limit or 'all'}")
    print(f"  Mode    : {'append' if args.append else 'overwrite'}")
    print(f"{'='*55}")

    # Build loader
    loader = build_loader(
        ig_username=args.ig_user,
        use_login=not args.no_login,
    )

    # Fetch reels
    reels = fetch_reels(profile, loader, limit=args.limit)

    if not reels:
        print("\nNo video reels found.")
        print("  Check: Is the profile public? Is your session valid?")
        sys.exit(1)

    print(f"\nContent-type breakdown (heuristic):")
    from collections import Counter
    counts = Counter(r["content_type_hint"] for r in reels)
    for ct, n in sorted(counts.items()):
        label = cfg.CONTENT_TYPE_LABELS.get(ct, ct)
        print(f"  {label}: {n}")

    if args.save_meta:
        save_metadata(reels, profile)

    if args.dry_run:
        print(f"\n[Dry run] Would write {len(reels)} URLs to {URLS_FILE.name}")
        print("  Re-run without --dry-run to save.")
        return

    # Save URLs
    new_count, skipped = save_urls(reels, append=args.append, profile=profile)
    mode = "Appended" if args.append else "Saved"
    print(f"\n{mode} {new_count} new URL(s) → {URLS_FILE.name}")
    if skipped:
        print(f"Skipped {skipped} duplicate(s) already in urls.txt")

    if new_count > 0:
        print(f"\nNext step:")
        print(f"  python download_reels.py --cookies-browser chrome")
        print(f"  python run_pipeline.py --stages download,transcribe,classify")


if __name__ == "__main__":
    main()
