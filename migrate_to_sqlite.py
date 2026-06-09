#!/usr/bin/env python3
"""
migrate_to_sqlite.py
────────────────────
Utility to migrate data from PostgreSQL vector database to SQLite vector database.
Avoids having to re-run classification and ingestion scripts.
"""

import os
import json
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).parent
SQLITE_DB_PATH = BASE_DIR / "reels_vector.db"

def main():
    print("Starting migration from PostgreSQL to SQLite...")
    
    # 1. Check if we can import psycopg2
    try:
        import psycopg2
    except ImportError:
        print("Error: psycopg2 is not installed. PostgreSQL migration is not possible without it.")
        print("Please run: pip install psycopg2-binary")
        return

    # 2. Check if .env exists to get credentials
    _env = BASE_DIR / ".env"
    if _env.exists():
        print("Loading environment from .env...")
        for line in _env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    # Ensure DB_TYPE is set to postgres for this extraction session
    os.environ["DB_TYPE"] = "postgres"
    
    # Import our postgres_db module
    import postgres_db
    
    # 3. Connect to PostgreSQL and fetch all records
    try:
        pg_conn = postgres_db.get_connection()
    except Exception as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        print("Make sure PostgreSQL is running and credentials in .env are correct.")
        return

    print("Connected to PostgreSQL database.")
    
    try:
        with pg_conn.cursor() as cur:
            cur.execute("SELECT id, collection, document, embedding, metadata FROM reels_embeddings;")
            rows = cur.fetchall()
            print(f"Retrieved {len(rows)} records from PostgreSQL reels_embeddings table.")
    except Exception as e:
        print(f"Error reading from PostgreSQL table: {e}")
        pg_conn.close()
        return
    finally:
        pg_conn.close()

    if not rows:
        print("No records found in PostgreSQL table to migrate.")
        return

    # 4. Initialize SQLite and write records
    print("Writing to SQLite database...")
    
    # Set DB_TYPE back to sqlite to initialize the SQLite DB correctly
    os.environ["DB_TYPE"] = "sqlite"
    import importlib
    importlib.reload(postgres_db)
    
    # Initialize the SQLite table
    postgres_db.init_db(reset=True)
    
    # Connect to SQLite
    sqlite_conn = sqlite3.connect(SQLITE_DB_PATH)
    try:
        with sqlite_conn:
            cur = sqlite_conn.cursor()
            
            sqlite_rows = []
            for r_id, col, doc, emb, meta in rows:
                if isinstance(emb, str):
                    emb_str = emb
                else:
                    emb_str = json.dumps(list(emb) if hasattr(emb, '__iter__') else emb)
                
                if isinstance(meta, str):
                    meta_str = meta
                else:
                    meta_str = json.dumps(meta)
                
                sqlite_rows.append((r_id, col, doc, emb_str, meta_str))
                
            cur.executemany("""
                INSERT OR REPLACE INTO reels_embeddings (id, collection, document, embedding, metadata)
                VALUES (?, ?, ?, ?, ?);
            """, sqlite_rows)
            
        print(f"Successfully migrated {len(rows)} records to SQLite database at: {SQLITE_DB_PATH}")
    except Exception as e:
        print(f"Error writing to SQLite database: {e}")
    finally:
        sqlite_conn.close()

    print("\nMigration finished!")
    print("To use SQLite as your default database, make sure DB_TYPE=sqlite is in your .env (or leave it unset as SQLite is now the default).")

if __name__ == "__main__":
    main()
