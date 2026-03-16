"""
NeuroCausal RAG - Retriever Engine
Combines similarity, causal, and importance scores

Includes Entity Linking integration
"""

import numpy as np
from typing import List, Optional, Tuple
import logging

from ..config import SearchConfig, IndexConfig
from ..interfaces import IRetriever, SearchResult, EntityResolution
from ..core.graph import GraphEngine
from ..embedding.text import TextEmbedding
from .index import create_index_backend, IIndexBackend

logger = logging.getLogger(__name__)


class Retriever(IRetriever):
    """
    NeuroCausal Retriever Engine.

    Combines three scoring components:
    - Similarity score: Vector similarity from index
    - Causal score: Based on causal relationships in graph
    - Importance score: PageRank-based node importance

    Final score = alpha * similarity + beta * causal + gamma * importance

    Entity Linking support:
    - Query enrichment (alias -> canonical)
    - Entity-based document boosting
    """

    def __init__(
        self,
        graph: GraphEngine,
        embedding: TextEmbedding,
        config: Optional[SearchConfig] = None,
        index_config: Optional[IndexConfig] = None,
        entity_linker: Optional['EntityLinker'] = None
    ):
        self.graph = graph
        self.embedding = embedding
        self.config = config or SearchConfig()
        self.index_config = index_config or IndexConfig()
        self.entity_linker = entity_linker

        self._index: Optional[IIndexBackend] = None

    def _ensure_index(self) -> None:
        """Ensure index is created"""
        if self._index is None:
            self._index = create_index_backend(self.index_config)

    def rebuild_index(self) -> None:
        """Rebuild the search index from graph"""
        self._ensure_index()

        embeddings, ids = self.graph.get_all_embeddings()
        if len(ids) > 0:
            self._index.build(embeddings, ids)
            logger.info(f"Index rebuilt with {len(ids)} documents")

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        use_entity_linking: bool = True
    ) -> List[SearchResult]:
        """
        Search for relevant documents.

        Args:
            query: Search query
            top_k: Number of results
            alpha: Similarity weight (optional, uses config default)
            beta: Causal weight (optional, uses config default)
            gamma: Importance weight (optional, uses config default)
            use_entity_linking: Entity linking ile sorgu zenginleştirme
        """
        # Entity Linking ile sorgu zenginleştirme
        enriched_query = query
        entity_resolution = None

        if use_entity_linking and self.entity_linker:
            entity_resolution = self._resolve_entities(query)
            if entity_resolution and entity_resolution.enriched_query != query:
                enriched_query = entity_resolution.enriched_query
                logger.info(f"Query enriched: '{query}' → '{enriched_query}'")

        query_embedding = self.embedding.get_text_embedding(enriched_query)
        results = self.search_by_embedding(
            query_embedding, top_k,
            alpha=alpha, beta=beta, gamma=gamma
        )

        # Entity bilgisini sonuclara ekle
        if entity_resolution and entity_resolution.resolved_aliases:
            for result in results:
                result.resolved_entities = list(entity_resolution.resolved_aliases.values())

        return results

    def _resolve_entities(self, query: str) -> Optional[EntityResolution]:
        """Entity linking ile sorgu zenginleştirme"""
        if not self.entity_linker:
            return None

        try:
            enriched = self.entity_linker.enrich_query(query)
            resolved = self.entity_linker.resolve_text(query)

            # Alias'ları topla
            aliases = {}
            for original, resolved_text in resolved.items():
                if original != resolved_text:
                    aliases[original] = resolved_text

            return EntityResolution(
                original_query=query,
                enriched_query=enriched,
                resolved_aliases=aliases,
                entities_found=list(set(resolved.values()))
            )
        except Exception as e:
            logger.warning(f"Entity resolution failed: {e}")
            return None

    def search_by_embedding(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        inject_chain: bool = True,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search using pre-computed embedding.

        Args:
            embedding: Query embedding
            top_k: Number of results to return
            inject_chain: If True, forcefully inject chain documents
            alpha: Similarity weight (optional, uses config default)
            beta: Causal weight (optional, uses config default)
            gamma: Importance weight (optional, uses config default)
        """
        self._ensure_index()

        if self._index.size == 0:
            return []

        # Use provided weights or defaults from config
        a = alpha if alpha is not None else self.config.alpha
        b = beta if beta is not None else self.config.beta
        g = gamma if gamma is not None else self.config.gamma

        # Get more candidates for re-ranking
        k_candidates = min(top_k * 3, self._index.size)
        candidates = self._index.search(embedding, k_candidates)

        results = []
        seen_ids = set()

        for node_id, similarity in candidates:
            if node_id in seen_ids:
                continue
            seen_ids.add(node_id)

            node = self.graph.get_node(node_id)
            if node is None:
                continue

            # Calculate causal score
            causal_score = self._compute_causal_score(node_id)

            # Get importance score
            importance_score = self.graph.get_importance(node_id)

            # Combined score with dynamic weights
            final_score = (
                a * similarity +
                b * causal_score +
                g * importance_score
            )

            # Get causal chain
            causal_chain = self.graph.get_causal_chain(node_id, max_depth=3)

            results.append(SearchResult(
                node_id=node_id,
                content=node['content'],
                score=final_score,
                similarity_score=similarity,
                causal_score=causal_score,
                importance_score=importance_score,
                causal_chain=causal_chain if len(causal_chain) > 1 else None,
                metadata=node.get('metadata')
            ))

        # Sort by final score
        results.sort(key=lambda x: x.score, reverse=True)

        # CHAIN INJECTION: Forcefully inject chain documents
        if inject_chain and len(results) > 0:
            injected = []
            for result in results[:min(3, len(results))]:  # Top 3 sonuc icin zincir ekle
                if result.causal_chain and len(result.causal_chain) > 1:
                    for chain_node_id in result.causal_chain[1:3]:  # Zincirdeki 2 dokuman
                        if chain_node_id not in seen_ids:
                            seen_ids.add(chain_node_id)
                            chain_node = self.graph.get_node(chain_node_id)
                            if chain_node:
                                # Zincir dokumani icin ozel skor
                                chain_result = SearchResult(
                                    node_id=chain_node_id,
                                    content=chain_node['content'],
                                    score=result.score * 0.8,  # Ana sonucun %80'i
                                    similarity_score=0.5,  # Benzerlik dusuk olabilir
                                    causal_score=1.0,  # Nedensel skor yuksek
                                    importance_score=self.graph.get_importance(chain_node_id),
                                    causal_chain=[chain_node_id],  # Kendisi
                                    metadata={'injected_from': result.node_id}  # Nereden enjekte edildi
                                )
                                injected.append(chain_result)

            # Enjekte edilen dokumanlari ekle
            results.extend(injected)
            results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    def _compute_causal_score(self, node_id: str) -> float:
        """
        Compute causal score for a node.

        Based on:
        - Number of causal connections (incoming and outgoing)
        - Strength of causal connections
        """
        # Get outgoing CAUSES relations
        forward_neighbors = self.graph.get_neighbors(node_id, ['causes'])

        # Get incoming CAUSES relations
        backward_neighbors = self.graph.get_predecessors(node_id, ['causes'])

        # Score based on causal connectivity
        total_connections = len(forward_neighbors) + len(backward_neighbors)

        if total_connections == 0:
            return 0.0

        # Normalize to 0-1 range with soft ceiling
        causal_score = min(1.0, total_connections / 5.0)

        return causal_score

    def save_index(self, path: str) -> None:
        """Save index to disk"""
        if self._index:
            self._index.save(path)

    def load_index(self, path: str) -> None:
        """Load index from disk"""
        self._ensure_index()
        self._index.load(path)
