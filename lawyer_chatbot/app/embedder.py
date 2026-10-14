"""Production embedding pipeline for Pakistan Legal Knowledge Base.

Converts all extracted statutory chunks into FAISS vector representations using
SentenceTransformer (all-MiniLM-L6-v2) with L2-normalized Inner Product metric
for exact cosine similarity search.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import (
    BASE_DIR,
    EMBEDDINGS_DIR,
    OUTPUT_JSON,
    EMBEDDING_MODEL,
)

INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"
CHUNKS_PATH = EMBEDDINGS_DIR / "chunks.pkl"
MANIFEST_PATH = EMBEDDINGS_DIR / "index_manifest.json"


def build_embeddings(batch_size: int = 128) -> None:
    """Build or rebuild FAISS index from kb_chunks.json with cosine similarity."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    if not OUTPUT_JSON.exists():
        print(f"Error: {OUTPUT_JSON} not found. Run scripts/extract_all_pdfs.py first.")
        sys.exit(1)

    print(f"Loading legal chunks from {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print("Error: No chunks found in JSON.")
        return

    print(f"Loaded {len(chunks):,} statutory chunks.")

    # Deduplicate by text content
    unique_chunks = []
    seen_texts = set()
    for c in chunks:
        norm = " ".join(c.get("text", "").split())
        if norm and norm not in seen_texts:
            seen_texts.add(norm)
            unique_chunks.append(c)

    chunks = unique_chunks
    texts = [c["text"] for c in chunks]
    print(f"Total deduplicated chunks to index: {len(texts):,}")

    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Encoding {len(texts):,} passages with batch_size={batch_size}...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    encoding_time = time.time() - t0
    print(f"Encoding complete in {encoding_time:.1f}s ({len(texts)/encoding_time:.1f} chunks/sec).")

    # L2-normalize vectors for exact Cosine Similarity with Inner Product index
    print("Normalizing vectors for cosine similarity...")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    print(f"Constructing FAISS IndexFlatIP (dimension {dim})...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Indexed {index.ntotal:,} vectors in FAISS.")

    # Atomic persistence
    temp_index = EMBEDDINGS_DIR / ".faiss.index.tmp"
    temp_chunks = EMBEDDINGS_DIR / ".chunks.pkl.tmp"
    temp_manifest = EMBEDDINGS_DIR / ".index_manifest.tmp"

    faiss.write_index(index, str(temp_index))
    with open(temp_chunks, "wb") as f:
        pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": EMBEDDING_MODEL,
        "metric": "cosine",
        "dimension": dim,
        "chunk_count": len(chunks),
    }
    with open(temp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    os.replace(temp_index, INDEX_PATH)
    os.replace(temp_chunks, CHUNKS_PATH)
    os.replace(temp_manifest, MANIFEST_PATH)

    print(f"Saved FAISS index to {INDEX_PATH}")
    print(f"Saved chunk metadata to {CHUNKS_PATH}")
    print(f"Saved index manifest to {MANIFEST_PATH}")
    print("Knowledge base indexing complete!")


if __name__ == "__main__":
    build_embeddings()
