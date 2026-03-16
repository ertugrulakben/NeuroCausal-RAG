"""
NeuroCausal RAG - Abstract Interfaces
Strategy Pattern için Abstract Base Classes

Yazar: Ertugrul Akben
E-posta: i@ertugrulakben.com
Versiyon: 4.0.0
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np


# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class SearchResult:
    """Standard search result format"""
    node_id: str
    content: str
    score: float
    similarity_score: float
    causal_score: float
    importance_score: float
    causal_chain: Optional[List[str]] = None
    metadata: Optional[Dict] = None
    resolved_entities: Optional[List[str]] = None  # Entity linking sonuclari


@dataclass
class EntityResolution:
    """Entity linking sonucu"""
    original_query: str
    enriched_query: str
    resolved_aliases: Dict[str, str]  # alias -> canonical
    entities_found: List[str]


@dataclass
class EvaluationResult:
    """LLM evaluation result"""
    answer: str
    score: float
    context_quality: float
    reasoning: str
    tokens_used: int


# ============================================================================
# EMBEDDING INTERFACE
# ============================================================================
class IEmbeddingEngine(ABC):
    """Abstract interface for embedding engines"""

    @abstractmethod
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Generate text embedding"""
        pass

    @abstractmethod
    def get_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension"""
        pass


# ============================================================================
# GRAPH INTERFACE
# ============================================================================
class IGraphEngine(ABC):
    """Abstract interface for graph engines"""

    @abstractmethod
    def add_node(self, node_id: str, content: str, embedding: np.ndarray,
                 metadata: Optional[Dict] = None) -> None:
        """Add node to graph"""
        pass

    @abstractmethod
    def add_edge(self, source: str, target: str, relation_type: str,
                 strength: float = 1.0) -> None:
        """Add edge between nodes"""
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Dict]:
        """Get node by ID"""
        pass

    @abstractmethod
    def get_neighbors(self, node_id: str,
                      relation_types: Optional[List[str]] = None) -> List[str]:
        """Get neighboring node IDs"""
        pass

    @abstractmethod
    def get_causal_chain(self, node_id: str, max_depth: int = 3) -> List[str]:
        """Get causal chain starting from node"""
        pass

    @abstractmethod
    def get_importance(self, node_id: str) -> float:
        """Get node importance (PageRank)"""
        pass

    @abstractmethod
    def export(self, path: str) -> None:
        """Export graph to file"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load graph from file"""
        pass

    @property
    @abstractmethod
    def node_count(self) -> int:
        """Return number of nodes"""
        pass

    @property
    @abstractmethod
    def edge_count(self) -> int:
        """Return number of edges"""
        pass


# ============================================================================
# INDEX INTERFACE (Strategy Pattern)
# ============================================================================
class IIndexBackend(ABC):
    """
    Abstract interface for index backends.
    Implements Strategy Pattern for different index implementations.
    """

    @abstractmethod
    def build(self, embeddings: np.ndarray, ids: List[str]) -> None:
        """Build index from embeddings"""
        pass

    @abstractmethod
    def add(self, embedding: np.ndarray, node_id: str) -> None:
        """Add single embedding to index"""
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """
        Search for k nearest neighbors.
        Returns: List of (node_id, similarity_score) tuples
        """
        pass

    @abstractmethod
    def remove(self, node_id: str) -> bool:
        """Remove node from index"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save index to disk"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load index from disk"""
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """Return number of vectors in index"""
        pass


# ============================================================================
# RETRIEVER INTERFACE
# ============================================================================
class IRetriever(ABC):
    """Abstract interface for retrieval engines"""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Search for relevant documents.
        Combines similarity, causal, and importance scores.
        """
        pass

    @abstractmethod
    def search_by_embedding(self, embedding: np.ndarray,
                            top_k: int = 5) -> List[SearchResult]:
        """Search using pre-computed embedding"""
        pass

    @abstractmethod
    def rebuild_index(self) -> None:
        """Rebuild the search index"""
        pass


# ============================================================================
# LLM INTERFACE
# ============================================================================
class ILLMClient(ABC):
    """Abstract interface for LLM clients"""

    @abstractmethod
    def generate(self, prompt: str, context: str) -> str:
        """Generate answer given prompt and context"""
        pass

    @abstractmethod
    def evaluate(self, query: str, answer: str, context: str) -> EvaluationResult:
        """Evaluate answer quality"""
        pass

    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Count tokens in text"""
        pass


# ============================================================================
# LEARNING INTERFACE
# ============================================================================
class ILearningEngine(ABC):
    """Abstract interface for self-learning engine"""

    @abstractmethod
    def record_feedback(self, query: str, result_ids: List[str],
                        rating: float, comment: Optional[str] = None) -> None:
        """Record user feedback"""
        pass

    @abstractmethod
    def discover_links(self, min_confidence: float = 0.7) -> List[Dict]:
        """Discover potential new causal links"""
        pass

    @abstractmethod
    def update_weights(self) -> Dict[str, float]:
        """Update node weights based on feedback"""
        pass

    @abstractmethod
    def get_statistics(self) -> Dict:
        """Get learning statistics"""
        pass
