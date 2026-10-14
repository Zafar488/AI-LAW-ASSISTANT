import unittest
import numpy as np
import faiss
from app.retriever import Retriever


class FakeEmbeddingModel:
    """Fake embedding model that returns predefined normalized vectors for testing."""
    def __init__(self, dim=4):
        self.dim = dim

    def encode(self, texts, **kwargs):
        # Return a fixed unit vector along dimension 0 for queries
        arr = np.zeros((len(texts), self.dim), dtype=np.float32)
        arr[:, 0] = 1.0
        return arr


class TestRetrieverMetricAndValidation(unittest.TestCase):
    """Focused unit tests for FAISS metric consistency, manifest validation, and score ordering.
    Uses strictly synthetic in-memory data; does not read or mutate production chunks.pkl.
    """

    def setUp(self):
        self.dim = 4
        # Create an in-memory IndexFlatIP (cosine similarity when vectors are normalized)
        self.index = faiss.IndexFlatIP(self.dim)

        # 3 normalized synthetic vectors:
        # v0: highly aligned with query [1, 0, 0, 0] -> cosine = 1.0
        # v1: partially aligned -> cosine = ~0.707
        # v2: orthogonal -> cosine = 0.0
        v0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v1 = np.array([0.70710677, 0.70710677, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        vectors = np.stack([v0, v1, v2])
        self.index.add(vectors)

        self.fake_chunks = [
            {"id": "doc_0", "text": "Passage zero text", "document": "Act A"},
            {"id": "doc_1", "text": "Passage one text", "document": "Act B"},
            {"id": "doc_2", "text": "Passage two text", "document": "Act C"},
        ]

        self.valid_manifest = {
            "chunk_count": 3,
            "vector_count": 3,
            "dimension": 4,
            "metric": "cosine",
            "model_name": "all-MiniLM-L6-v2",
            "created_at": "2026-09-26T00:00:00"
        }

    def test_manifest_validation_success(self):
        """Manifest validation must succeed when counts, dimension, and metric align."""
        retriever = Retriever(
            index=self.index,
            chunks=self.fake_chunks,
            model=FakeEmbeddingModel(self.dim),
            manifest=self.valid_manifest
        )
        self.assertIsNotNone(retriever.manifest)
        self.assertEqual(retriever.manifest["metric"], "cosine")
        self.assertEqual(len(retriever.chunks), 3)

    def test_manifest_validation_dimension_mismatch(self):
        """Retriever must fail with actionable ValueError if vector dimension differs from manifest."""
        bad_manifest = dict(self.valid_manifest, dimension=384)  # Actual is 4
        with self.assertRaises(ValueError) as ctx:
            Retriever(
                index=self.index,
                chunks=self.fake_chunks,
                model=FakeEmbeddingModel(self.dim),
                manifest=bad_manifest
            )
        self.assertIn("Dimension mismatch", str(ctx.exception))

    def test_manifest_validation_count_mismatch(self):
        """Retriever must fail if FAISS vector count != chunk count."""
        with self.assertRaises(ValueError) as ctx:
            Retriever(
                index=self.index,
                chunks=self.fake_chunks[:2],  # Only 2 chunks for 3 vectors
                model=FakeEmbeddingModel(self.dim),
                manifest=self.valid_manifest
            )
        self.assertIn("Knowledge base vector mismatch", str(ctx.exception))

    def test_manifest_validation_incompatible_metric(self):
        """Retriever must fail if manifest metric is incompatible (e.g. L2 distance)."""
        bad_manifest = dict(self.valid_manifest, metric="l2")
        with self.assertRaises(ValueError) as ctx:
            Retriever(
                index=self.index,
                chunks=self.fake_chunks,
                model=FakeEmbeddingModel(self.dim),
                manifest=bad_manifest
            )
        self.assertIn("Incompatible FAISS metric in manifest", str(ctx.exception))

    def test_cosine_similarity_ranking_order(self):
        """Verifies that IndexFlatIP ranks highest cosine similarity first and scores are monotonic."""
        retriever = Retriever(
            index=self.index,
            chunks=self.fake_chunks,
            model=FakeEmbeddingModel(self.dim),
            manifest=self.valid_manifest
        )

        results = retriever.retrieve("neutral general query", top_k=3)
        self.assertEqual(len(results), 3)

        # Most aligned doc_0 should be ranked first
        self.assertEqual(results[0]["id"], "doc_0")
        self.assertEqual(results[1]["id"], "doc_1")
        self.assertEqual(results[2]["id"], "doc_2")

        # Semantic scores should be strictly descending: doc_0 (1.0) > doc_1 (~0.707) > doc_2 (0.0)
        self.assertAlmostEqual(results[0]["semantic_score"], 1.0, places=4)
        self.assertAlmostEqual(results[1]["semantic_score"], 0.7071, places=3)
        self.assertAlmostEqual(results[2]["semantic_score"], 0.0, places=4)

        self.assertGreater(results[0]["hybrid_score"], results[1]["hybrid_score"])
        self.assertGreater(results[1]["hybrid_score"], results[2]["hybrid_score"])


if __name__ == "__main__":
    unittest.main()
