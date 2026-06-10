#!/usr/bin/env python3
"""
config.py
─────────
Shared configuration loader for the Ground Up Reels Pipeline.
All scripts import from here instead of each re-implementing env loading.

Usage:
    from config import cfg

    api_key = cfg.GEMINI_API_KEY
    model   = cfg.GEMINI_MODEL
    db_type = cfg.DB_TYPE
"""

import os
from pathlib import Path

# ── Resolve paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

# Silence HuggingFace symlink warnings on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _load_env(path: Path) -> None:
    """Load key=value pairs from a .env file into os.environ.
    .env file values take precedence over existing shell env vars
    (so editing .env always takes effect without restarting the terminal).
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ[key.strip()] = val.strip()  # override — .env wins


# Load .env automatically when this module is imported
_load_env(BASE_DIR / ".env")


class Config:
    """Central config object. All values are read lazily from os.environ."""

    # ── Gemini ────────────────────────────────────────────────────────────────
    @property
    def GEMINI_API_KEY(self) -> str:
        return os.environ.get("GEMINI_API_KEY", "")

    @property
    def GEMINI_MODEL(self) -> str:
        """Default Gemini model. Override via GEMINI_MODEL env var."""
        return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    @property
    def GEMINI_MODEL_CLASSIFY(self) -> str:
        """Model used for classification (may differ from chat model)."""
        return os.environ.get("GEMINI_MODEL_CLASSIFY", self.GEMINI_MODEL)

    # ── Embedding ─────────────────────────────────────────────────────────────
    @property
    def EMBEDDING_MODEL(self) -> str:
        return os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ── Database ──────────────────────────────────────────────────────────────
    @property
    def DB_TYPE(self) -> str:
        """'sqlite' or 'postgres'"""
        return os.environ.get("DB_TYPE", "sqlite").lower()

    @property
    def POSTGRES_HOST(self) -> str:
        return os.environ.get("POSTGRES_HOST", "localhost")

    @property
    def POSTGRES_PORT(self) -> int:
        return int(os.environ.get("POSTGRES_PORT", "5432"))

    @property
    def POSTGRES_DB(self) -> str:
        return os.environ.get("POSTGRES_DB", "groundup_reels")

    @property
    def POSTGRES_USER(self) -> str:
        return os.environ.get("POSTGRES_USER", "postgres")

    @property
    def POSTGRES_PASSWORD(self) -> str:
        return os.environ.get("POSTGRES_PASSWORD", "")

    # ── Instagram ─────────────────────────────────────────────────────────────
    @property
    def IG_USERNAME(self) -> str:
        """Instagram username for instaloader session."""
        return os.environ.get("IG_USERNAME", "")

    @property
    def IG_DEFAULT_PROFILE(self) -> str:
        return os.environ.get("IG_DEFAULT_PROFILE", "groundup.in")

    # ── Pipeline paths ────────────────────────────────────────────────────────
    @property
    def BASE_DIR(self) -> Path:
        return BASE_DIR

    @property
    def DOWNLOADS_DIR(self) -> Path:
        return BASE_DIR / "downloads"

    @property
    def MARKDOWN_DIR(self) -> Path:
        return BASE_DIR / "markdown"

    @property
    def TRANSCRIPTS_DIR(self) -> Path:
        return BASE_DIR / "transcripts"

    @property
    def URLS_FILE(self) -> Path:
        return BASE_DIR / "urls.txt"

    @property
    def CACHE_FILE(self) -> Path:
        return BASE_DIR / "classification_cache.json"

    # ── Content types ─────────────────────────────────────────────────────────
    CONTENT_TYPES = [
        "recipe",
        "travel_vlog",
        "informational",
        "product_showcase",
        "other",
    ]

    CONTENT_TYPE_LABELS = {
        "recipe":           "[Recipe]",
        "travel_vlog":      "[Travel Vlog]",
        "informational":    "[Informational]",
        "product_showcase": "[Product Showcase]",
        "other":            "[Other]",
    }

    # ── RAG ───────────────────────────────────────────────────────────────────
    @property
    def RAG_TOP_K(self) -> int:
        return int(os.environ.get("RAG_TOP_K", "5"))

    @property
    def RAG_TEMPERATURE(self) -> float:
        return float(os.environ.get("RAG_TEMPERATURE", "0.3"))


# Singleton instance — import this everywhere
cfg = Config()


if __name__ == "__main__":
    print("Ground Up Pipeline — Config Check")
    print(f"  Gemini model   : {cfg.GEMINI_MODEL}")
    print(f"  API key set    : {'yes' if cfg.GEMINI_API_KEY else 'NO — set GEMINI_API_KEY in .env'}")
    print(f"  DB type        : {cfg.DB_TYPE}")
    print(f"  Embedding model: {cfg.EMBEDDING_MODEL}")
    print(f"  IG username    : {cfg.IG_USERNAME or '(not set)'}")
    print(f"  BASE_DIR       : {cfg.BASE_DIR}")
    print(f"\n  Content types  : {', '.join(cfg.CONTENT_TYPES)}")
