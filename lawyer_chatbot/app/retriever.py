import os
import pickle
from copy import deepcopy
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "faiss.index")
CHUNKS_PKL_PATH = os.path.join(EMBEDDINGS_DIR, "chunks.pkl")
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

class Retriever:
    def __init__(self):
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_PKL_PATH):
            raise FileNotFoundError("FAISS index or chunks.pkl not found. Run app/loader.py first.")
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNKS_PKL_PATH, 'rb') as f:
            self.chunks = pickle.load(f)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.dimension = self.index.d
        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"FAISS index/chunk mismatch: {self.index.ntotal} vectors, "
                f"{len(self.chunks)} chunks. Rebuild embeddings."
            )

    def retrieve(self, query, top_k=8):
        if not isinstance(query, str) or not query.strip():
            return []
        top_k = max(1, min(int(top_k), len(self.chunks), self.index.ntotal))
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                result = deepcopy(self.chunks[idx])
                result["distance"] = float(distance)
                results.append(result)
        return results
