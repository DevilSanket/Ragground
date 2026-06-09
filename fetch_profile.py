#!/usr/bin/env python3
"""
fetch_profile.py
────────────────
Fetch all video/reel URLs from an Instagram profile
using Instagram's web API with your sessionid cookie.

Usage:
    python fetch_profile.py --profile groundup.in
    python fetch_profile.py --profile groundup.in --limit 30 --append
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from http.cookiejar import MozillaCookieJar

BASE_DIR  = Path(__file__).parent
URLS_FILE = BASE_DIR / "urls.txt"

# Instagram app/web headers (mimics the browser)
IG_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "X-Ig-App-Id":    "936619743392459",
    "X-Requested-With": "XMLHttpRequest",
    "Referer":         "https://www.instagram.com/",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Origin":          "https://www.instagram.com",
    "Sec-Fetch-Site":  "same-origin",
    "Sec-Fetch-Mode":  "cors",
    "Sec-Fetch-Dest":  "empty",
}


def load_cookies(cookies_file: str) -> dict:
    """Parse cookies.txt (Netscape format) into a dict."""
    jar = MozillaCookieJar()
    try:
        jar.load(cookies_file, ignore_discard=True, ignore_expires=True)
        cookies = {c.name: c.value for c in jar if "instagram.com" in c.domain}
        print(f"Loaded {len(cookies)} Instagram cookies from {cookies_file}")
        return cookies
    except Exception as e:
        print(f"Error loading cookies: {e}")
        return {}


def get_user_id(session, username: str) -> str | None:
    """Resolve Instagram username to user ID via web API."""
    url = f"https://www.instagram.com/api/v1/users/web_profile_info/?username={username}"
    try:
        resp = session.get(url, headers=IG_HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"Failed to get user info: HTTP {resp.status_code}")
            print(f"Response: {resp.text[:200]}")
            return None
        data = resp.json()
        user = data.get("data", {}).get("user", {})
        uid  = user.get("id")
        name = user.get("full_name", username)
        count = user.get("edge_owner_to_timeline_media", {}).get("count", "?")
        print(f"  Profile: {name} (@{username}) | ID: {uid} | Posts: {count}")
        return uid
    except Exception as e:
        print(f"Error getting user ID: {e}")
        return None


def fetch_user_media(session, user_id: str, limit: int | None) -> list[dict]:
    """Paginate through a user's media using the graphql edge API."""
    reels = []
    cursor = None
    page   = 0

    # GraphQL query hash for user timeline media
    QUERY_HASH = "69cba40317214236af40e7efa697781d"

    while True:
        page += 1
        variables = {
            "id": user_id,
            "first": 12,
        }
        if cursor:
            variables["after"] = cursor

        params = {
            "query_hash": QUERY_HASH,
            "variables":  json.dumps(variables),
        }

        try:
            resp = session.get(
                "https://www.instagram.com/graphql/query/",
                params=params,
                headers=IG_HEADERS,
                timeout=15,
            )
        except Exception as e:
            print(f"  Request error on page {page}: {e}")
            break

        if resp.status_code == 429:
            wait = 30
            print(f"  Rate limited — waiting {wait}s...")
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} on page {page}: {resp.text[:150]}")
            break

        try:
            data = resp.json()
        except Exception as e:
            print(f"  JSON parse error: {e}")
            break

        timeline = (
            data.get("data", {})
                .get("user", {})
                .get("edge_owner_to_timeline_media", {})
        )
        edges      = timeline.get("edges", [])
        page_info  = timeline.get("page_info", {})
        has_next   = page_info.get("has_next_page", False)
        cursor     = page_info.get("end_cursor")

        if not edges:
            print("  No more posts found.")
            break

        for edge in edges:
            node = edge.get("node", {})
            typename = node.get("__typename", "")

            # Only video posts (GraphSidecar may also contain videos)
            if node.get("is_video") or typename == "GraphVideo":
                shortcode = node.get("shortcode", "")
                url       = f"https://www.instagram.com/p/{shortcode}/"
                caption   = ""
                edges_cap = node.get("edge_media_to_caption", {}).get("edges", [])
                if edges_cap:
                    caption = edges_cap[0].get("node", {}).get("text", "")[:80].replace("\n", " ")

                reel = {
                    "id":    shortcode,
                    "url":   url,
                    "title": caption,
                    "date":  str(node.get("taken_at_timestamp", "")),
                    "likes": node.get("edge_liked_by", {}).get("count", 0),
                    "views": node.get("video_view_count", 0),
                }
                reels.append(reel)
                print(f"  [{len(reels):>3}] {shortcode}  👁 {reel['views']:>8,}  {caption[:50]}")

        print(f"  Page {page} done | Reels so far: {len(reels)} | Has more: {has_next}")

        if limit and len(reels) >= limit:
            print(f"  Reached limit of {limit}.")
            break

        if not has_next:
            break

        time.sleep(2)  # polite delay between pages

    return reels


def fetch_via_api_v1(session, username: str, limit: int | None) -> list[dict]:
    """Alternative: use Instagram's /api/v1/feed/user/ endpoint."""
    reels = []
    max_id = None
    page   = 0

    while True:
        page += 1
        url    = f"https://www.instagram.com/api/v1/feed/user/{username}/"
        params = {"count": 12}
        if max_id:
            params["max_id"] = max_id

        try:
            resp = session.get(url, params=params, headers=IG_HEADERS, timeout=15)
        except Exception as e:
            print(f"  Request error: {e}")
            break

        if resp.status_code == 404:
            print(f"  Profile @{username} not found via v1 API.")
            break

        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
            break

        try:
            data = resp.json()
        except Exception:
            print("  JSON parse error")
            break

        items     = data.get("items", [])
        more      = data.get("more_available", False)
        max_id    = data.get("next_max_id")

        if not items:
            break

        for item in items:
            if item.get("media_type") not in (2,):  # 2 = video
                continue
            shortcode = item.get("code", "")
            url_post  = f"https://www.instagram.com/p/{shortcode}/"
            caption   = ""
            caps      = item.get("caption")
            if caps and isinstance(caps, dict):
                caption = caps.get("text", "")[:80].replace("\n", " ")
            elif isinstance(caps, list) and caps:
                caption = caps[0].get("text", "")[:80].replace("\n", " ")

            reel = {
                "id":    shortcode,
                "url":   url_post,
                "title": caption,
                "date":  str(item.get("taken_at", "")),
                "likes": item.get("like_count", 0),
                "views": item.get("view_count", 0),
            }
            reels.append(reel)
            print(f"  [{len(reels):>3}] {shortcode}  👁 {reel['views']:>8,}  {caption[:50]}")

        print(f"  Page {page}: {len(items)} items, {len(reels)} video reels total | more={more}")

        if limit and len(reels) >= limit:
            break
        if not more or not max_id:
            break

        time.sleep(1.5)

    return reels


def load_existing_urls() -> set[str]:
    if not URLS_FILE.exists():
        return set()
    return {
        line.strip()
        for line in URLS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def save_urls(reels: list[dict], append: bool, profile: str) -> tuple[int, int]:
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
                comment = f"  # {r['title'][:60]}" if r.get("title") else ""
                f.write(f"{r['url']}{comment}\n")
    else:
        lines = [
            f"# Fetched from @{profile} on {ts}",
            f"# Total video reels: {len(reels)}",
            "# ─────────────────────────────────────",
            "",
        ]
        for r in reels:
            comment = f"  # {r['title'][:60]}" if r.get("title") else ""
            lines.append(f"{r['url']}{comment}")
        URLS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return len(new_reels), skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", "-p", required=True)
    parser.add_argument("--limit",   "-n", type=int, default=None)
    parser.add_argument("--cookies-file", default="cookies.txt")
    parser.add_argument("--append",  "-a", action="store_true")
    args = parser.parse_args()

    import requests

    # Build authenticated session
    session = requests.Session()
    cookies_path = BASE_DIR / args.cookies_file
    if cookies_path.exists():
        cookies = load_cookies(str(cookies_path))
        session.cookies.update(cookies)
        if "csrftoken" in cookies:
            IG_HEADERS["X-CSRFToken"] = cookies["csrftoken"]
    else:
        print(f"Warning: {cookies_path} not found — requests may be blocked")

    profile = args.profile.lstrip("@")
    print(f"\nFetching reels from @{profile}...")

    # Strategy 1: web profile info + graphql
    print("\n[Strategy 1] Fetching via GraphQL API...")
    user_id = get_user_id(session, profile)
    reels = []

    if user_id:
        reels = fetch_user_media(session, user_id, args.limit)

    # Strategy 2: mobile API v1 fallback
    if not reels:
        print("\n[Strategy 2] Trying mobile API v1...")
        reels = fetch_via_api_v1(session, profile, args.limit)

    if not reels:
        print("\nCould not fetch reels. Possible reasons:")
        print("  - Session cookies have expired (re-export from browser)")
        print("  - Instagram is blocking the request")
        print("  - The profile is private")
        print("\nWorkaround: Manually add reel URLs to urls.txt")
        print("  Just paste the reel URLs (one per line) from the browser URL bar")
        sys.exit(1)

    print(f"\nFound {len(reels)} video reel(s)")
    new_count, skipped = save_urls(reels, append=args.append, profile=profile)
    mode = "Appended" if args.append else "Saved"
    print(f"{mode} {new_count} new URL(s) → {URLS_FILE.name}")
    if skipped:
        print(f"Skipped {skipped} duplicate(s)")

    if new_count > 0:
        print(f"\nNext: python download_reels.py --cookies-file {args.cookies_file}")


if __name__ == "__main__":
    main()
