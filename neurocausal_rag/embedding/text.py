"""
NeuroCausal RAG - Text Embedding Engine
Sentence Transformers based text embedding
"""

import numpy as np
from typing import List, Optional
import logging
from functools import lru_cache

from ..config import EmbeddingConfig
from ..interfaces import IEmbeddingEngine

logger = logging.getLogger(__name__)


class TextEmbedding(IEmbeddingEngine):
    """
    Text embedding engine using Sentence Transformers.

    Features:
    - Multilingual support
    - Optional L2 normalization
    - LRU caching for repeated queries
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._model = None
        self._dimension = self.config.dimension

    def _load_model(self):
        """Lazy load the model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.config.model_name)
                logger.info(f"Loaded embedding model: {self.config.model_name}")
            except ImportError:
                logger.warning("sentence-transformers not installed, using hash-based fallback")
                self._model = "fallback"

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Generate text embedding"""
        self._load_model()

        if self._model == "fallback":
            return self._fallback_embedding(text)

        embedding = self._model.encode(text, convert_to_numpy=True)

        if self.config.normalize:
            embedding = self._normalize(embedding)

        return embedding.astype(np.float32)

    def get_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        self._load_model()

        if self._model == "fallback":
            return np.array([self._fallback_embedding(t) for t in texts])

        embeddings = self._model.encode(texts, convert_to_numpy=True)

        if self.config.normalize:
            embeddings = np.array([self._normalize(e) for e in embeddings])

        return embeddings.astype(np.float32)

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        """L2 normalize embedding"""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def _fallback_embedding(self, text: str) -> np.ndarray:
        """Hash-based fallback embedding (for demo purposes)"""
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self._dimension).astype(np.float32)
        if self.config.normalize:
            embedding = self._normalize(embedding)
        return embedding

    @property
    def dimension(self) -> int:
        """Return embedding dimension"""
        return self._dimension


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))
