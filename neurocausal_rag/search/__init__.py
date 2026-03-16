"""
NeuroCausal RAG - Search Module
Index backends and retrieval engine

Includes Multi-Hop, Optimizer, and Query Decomposition
"""

from .index import BruteForceIndex, create_index_backend
from .retriever import Retriever
from .multi_hop import (
    MultiHopRetriever,
    MultiHopResult,
    HopPath,
    create_multi_hop_retriever
)
from .optimizer import (
    SearchOptimizer,
    SearchMode,
    SearchWeights,
    QueryAnalyzer,
    QueryAnalysis,
    get_mode_preset,
    create_optimizer
)
from .decomposer import (
    QueryDecomposer,
    ResultMerger,
    DecomposedSearch,
    DecompositionResult,
    DecompositionStrategy,
    SubQuery,
    MergedResult,
    create_decomposer,
    create_merger
)

__all__ = [
    # Index
    "BruteForceIndex",
    "create_index_backend",
    # Retriever
    "Retriever",
    # Multi-Hop
    "MultiHopRetriever",
    "MultiHopResult",
    "HopPath",
    "create_multi_hop_retriever",
    # Optimizer
    "SearchOptimizer",
    "SearchMode",
    "SearchWeights",
    "QueryAnalyzer",
    "QueryAnalysis",
    "get_mode_preset",
    "create_optimizer",
    # Decomposer
    "QueryDecomposer",
    "ResultMerger",
    "DecomposedSearch",
    "DecompositionResult",
    "DecompositionStrategy",
    "SubQuery",
    "MergedResult",
    "create_decomposer",
    "create_merger"
]
