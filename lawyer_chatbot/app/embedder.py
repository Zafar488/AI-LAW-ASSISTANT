import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path

from app.config import (
    EMBEDDINGS_DIR, OUTPUT_JSON, OFFICIAL_OUTPUT_JSON, PDF_UPDATES_JSON,
    EMBEDDING_MODEL,
)

# -------------------------------
# Build FAISS embeddings from kb_chunks.json
# -------------------------------
def build_embeddings():
    # Create embeddings dir if not exists
    os.makedirs(str(EMBEDDINGS_DIR), exist_ok=True)

    KB_FILE = str(OUTPUT_JSON)
    INDEX_FILE = os.path.join(str(EMBEDDINGS_DIR), "faiss.index")
    META_FILE = os.path.join(str(EMBEDDINGS_DIR), "chunks.pkl")

    # Load chunks from JSON
    if not os.path.exists(KB_FILE):
        print(f"Error: KB file not found at {KB_FILE}")
        print("Run loader.py first: python -m app.loader")
        return

    print(f"Loading chunks from {KB_FILE}...")
    with open(KB_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    official_chunks = []
    if OFFICIAL_OUTPUT_JSON.exists():
        with open(OFFICIAL_OUTPUT_JSON, "r", encoding="utf-8") as f:
            official_chunks = json.load(f)
        chunks.extend(official_chunks)
        print(f"Added {len(official_chunks)} official web chunks.")

    if PDF_UPDATES_JSON.exists():
        with open(PDF_UPDATES_JSON, "r", encoding="utf-8") as f:
            pdf_updates = json.load(f)
        chunks.extend(pdf_updates)
        print(f"Added {len(pdf_updates)} newly extracted PDF chunks.")

    if not chunks:
        print("No chunks found in KB file.")
        return

    # Filter empty and duplicate chunks across PDF and official web sources.
    unique_chunks = []
    seen_text = set()
    for chunk in chunks:
        normalized = " ".join(chunk.get("text", "").split())
        if normalized and normalized not in seen_text:
            seen_text.add(normalized)
            unique_chunks.append(chunk)
    chunks = unique_chunks
    print(f"Loaded {len(chunks)} valid chunks.")

    texts = [c['text'] for c in chunks]

    # Reuse the existing base vectors when they exactly match the local JSON.
    # This makes adding newly scraped official laws much faster.
    existing_index = None
    existing_chunks = []
    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        try:
            existing_index = faiss.read_index(INDEX_FILE)
            with open(META_FILE, "rb") as f:
                existing_chunks = pickle.load(f)
            local_normalized = {" ".join(c.get("text", "").split()) for c in json.load(open(KB_FILE, encoding="utf-8"))}
            existing_normalized = {" ".join(c.get("text", "").split()) for c in existing_chunks}
            if (
                existing_index.ntotal != len(existing_chunks)
                or not local_normalized.issubset(existing_normalized)
            ):
                existing_index = None
                existing_chunks = []
        except Exception as exc:
            print(f"Existing index cannot be reused: {exc}")
            existing_index = None
            existing_chunks = []

    # Load embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Generate embeddings
    if existing_index is not None:
        known = {" ".join(c.get("text", "").split()) for c in existing_chunks}
        chunks_to_add = [c for c in chunks if " ".join(c.get("text", "").split()) not in known]
        texts = [c["text"] for c in chunks_to_add]
        print(f"Reusing {existing_index.ntotal} vectors; encoding {len(texts)} new chunks...")
    else:
        chunks_to_add = chunks
        print(f"Encoding {len(texts)} chunks into embeddings...")

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=64)

    # Ensure float32 for FAISS
    embeddings = np.array(embeddings).astype('float32')

    # Create FAISS index
    dim = embeddings.shape[1]
    print(f"Creating FAISS index with dimension {dim}...")
    if existing_index is not None:
        if existing_index.d != dim:
            raise ValueError(f"Embedding dimension changed: {existing_index.d} != {dim}")
        index = existing_index
        index.add(embeddings)
        chunks = existing_chunks + chunks_to_add
    else:
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

    # Save FAISS index and metadata
    index_temp = INDEX_FILE + ".tmp"
    metadata_temp = META_FILE + ".tmp"
    faiss.write_index(index, index_temp)
    with open(metadata_temp, 'wb') as f:
        pickle.dump(chunks, f)
    os.replace(index_temp, INDEX_FILE)
    os.replace(metadata_temp, META_FILE)
    print(f"FAISS index saved at: {INDEX_FILE}")
    print(f"Chunk metadata saved at: {META_FILE}")
    print(f"Successfully processed {len(chunks)} chunks.")


if __name__ == "__main__":
    build_embeddings()
