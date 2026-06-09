import os
import sys
import json
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

BASE_DIR = Path(__file__).parent

# Load .env variables
_env = BASE_DIR / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

# Global lazy-loaded embedding model
_model = None

def get_connection():
    """Returns a connection to the PostgreSQL database."""
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "groundup_reels")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    
    # Try using DATABASE_URL first if it exists
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        return psycopg2.connect(db_url)
    
    return psycopg2.connect(
        host=host,
        port=port,
        database=db,
        user=user,
        password=password
    )

def init_db(reset: bool = False, collection_name: str = None):
    """Initializes the database, creating the vector extension and the table if they do not exist."""
    conn = get_connection()
    conn.autocommit = True
    with conn.cursor() as cur:
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except Exception as e:
            print(f"Error creating vector extension: {e}")
            print("Please ensure pgvector is installed in your PostgreSQL instance.")
            conn.close()
            sys.exit(1)

        # Check if table exists and has a dimension mismatch for the vector column
        cur.execute("SELECT EXISTS (SELECT FROM pg_tables WHERE tablename = 'reels_embeddings');")
        table_exists = cur.fetchone()[0]
        if table_exists:
            try:
                cur.execute("""
                    SELECT atttypmod 
                    FROM pg_attribute 
                    WHERE attrelid = 'reels_embeddings'::regclass AND attname = 'embedding';
                """)
                row = cur.fetchone()
                if row:
                    existing_dim = row[0]
                    if existing_dim != 384:
                        print(f"Detected vector dimension mismatch (DB has {existing_dim}, expected 384). Dropping table to recreate...")
                        cur.execute("DROP TABLE IF EXISTS reels_embeddings CASCADE;")
            except Exception as e:
                print(f"Warning during vector dimension check: {e}")

        # Create table with 384 dimensions vector (all-MiniLM-L6-v2)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS reels_embeddings (
                id VARCHAR(255) PRIMARY KEY,
                collection VARCHAR(50) NOT NULL,
                document TEXT NOT NULL,
                embedding VECTOR(384) NOT NULL,
                metadata JSONB NOT NULL
            );
        """)

        # Indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_reels_embeddings_collection ON reels_embeddings(collection);")
        
        # HNSW index for fast similarity search using cosine distance operator (<=>)
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_reels_embeddings_vector ON reels_embeddings USING hnsw (embedding vector_cosine_ops);")
        except Exception as e:
            # Fallback to simple IVFFlat index or no index if HNSW is not supported on older pgvector versions
            print(f"Warning: HNSW index creation failed: {e}. Trying IVFFlat or skipping index...")
            try:
                cur.execute("CREATE INDEX IF NOT EXISTS idx_reels_embeddings_vector ON reels_embeddings USING ivfflat (embedding vector_cosine_ops);")
            except Exception:
                pass

        if reset:
            if collection_name:
                cur.execute("DELETE FROM reels_embeddings WHERE collection = %s;", (collection_name,))
                print(f"Cleared collection: '{collection_name}' in PostgreSQL.")
            else:
                cur.execute("TRUNCATE TABLE reels_embeddings;")
                print("Truncated all collections in reels_embeddings table.")
    conn.close()

def get_embedding_model():
    """Lazily loads the SentenceTransformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        # Disable hub warning
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Encodes texts to vector embeddings."""
    model = get_embedding_model()
    embeddings = model.encode(texts)
    return [emb.tolist() for emb in embeddings]

def upsert_chunks(collection_name: str, ids: list[str], documents: list[str], metadatas: list[dict]):
    """Upserts chunks into the PostgreSQL table."""
    if not ids:
        return

    # Generate embeddings
    embeddings = get_embeddings(documents)

    conn = get_connection()
    conn.autocommit = True
    with conn.cursor() as cur:
        rows = []
        for cid, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
            rows.append((cid, collection_name, doc, emb, json.dumps(meta)))

        query = """
            INSERT INTO reels_embeddings (id, collection, document, embedding, metadata)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                document = EXCLUDED.document,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata;
        """
        execute_values(cur, query, rows)
    conn.close()

def get_collection_count(collection_name: str) -> int:
    """Returns the total number of documents in a collection."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM reels_embeddings WHERE collection = %s;", (collection_name,))
        count = cur.fetchone()[0]
    conn.close()
    return count

def get_existing_ids(collection_name: str) -> set[str]:
    """Retrieves all document IDs currently in the collection."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM reels_embeddings WHERE collection = %s;", (collection_name,))
        ids = {row[0] for row in cur.fetchall()}
    conn.close()
    return ids

def list_collections() -> list[str]:
    """Returns list of distinct collection names in database."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT collection FROM reels_embeddings;")
        collections = [row[0] for row in cur.fetchall()]
    conn.close()
    return collections

def retrieve(collection_name: str, query: str, k: int = 5, chunk_type: str | None = None) -> list[dict]:
    """Retrieves top-K relevant chunks using cosine similarity."""
    query_emb = get_embeddings([query])[0]

    conn = get_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        query_sql = """
            SELECT id, document, metadata, (1 - (embedding <=> %s::vector)) AS score
            FROM reels_embeddings
            WHERE collection = %s
        """
        params = [query_emb, collection_name]
        
        if chunk_type:
            query_sql += " AND metadata->>'chunk_type' = %s"
            params.append(chunk_type)
            
        query_sql += """
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        params.extend([query_emb, k])
        
        cur.execute(query_sql, params)
        results = cur.fetchall()
    conn.close()

    chunks = []
    for r in results:
        meta = r["metadata"]
        chunks.append({
            "text":       r["document"],
            "reel_id":    meta.get("reel_id", ""),
            "recipe_name": meta.get("recipe_name", ""),
            "author":     meta.get("author", ""),
            "url":        meta.get("url", ""),
            "chunk_type": meta.get("chunk_type", ""),
            "timestamp":  meta.get("timestamp", ""),
            "date":       meta.get("date", ""),
            "score":      round(float(r["score"]), 3) if r["score"] is not None else 0.0,
        })
    return chunks

