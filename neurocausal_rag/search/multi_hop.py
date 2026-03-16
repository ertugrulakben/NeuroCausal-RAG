"""
NeuroCausal RAG - Multi-Hop Retrieval
v5.2 - FAZ 2.1

Multi-hop retrieval ile dolayli baglantilar uzerinden dokuman bulma.

Ornek:
    Sorgu: "Sera gazlari iklim degisikligine nasil neden olur?"
    Direkt eslesen yok, ama zincir var:

    [Sera Gazlari] --causes--> [Sicaklik Artisi] --causes--> [Buzul Erimesi]

    Multi-hop retrieval bu zinciri kesfeder ve ara dokumanlari da getirir.

Yazar: Ertugrul Akben
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from ..core.graph import GraphEngine
from ..embedding.text import TextEmbedding
from ..interfaces import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class HopPath:
    """Cok atlamali yol temsili"""
    nodes: List[str]  # Yoldaki node ID'leri
    edges: List[str]  # Yoldaki edge tipleri
    total_weight: float  # Toplam yol agirligi
    hop_count: int  # Atlama sayisi

    @property
    def start(self) -> str:
        return self.nodes[0] if self.nodes else ""

    @property
    def end(self) -> str:
        return self.nodes[-1] if self.nodes else ""


@dataclass
class MultiHopResult:
    """Multi-hop arama sonucu"""
    node_id: str
    content: str
    score: float
    hop_distance: int  # Sorgudan kac hop uzakta
    path: Optional[HopPath] = None  # Nasil ulasildigi
    bridge_nodes: List[str] = field(default_factory=list)  # Ara koprü nodelar
    similarity_score: float = 0.0
    causal_score: float = 0.0
    importance_score: float = 0.0
    metadata: Optional[Dict] = None


class MultiHopRetriever:
    """
    Multi-Hop Retrieval Engine

    Ozellikler:
    1. N-hop path finding: Direkt baglanti olmayan dokumanlar arasi yol bulma
    2. Path-based scoring: Yol kalitesine gore skorlama
    3. Bridge discovery: Kopru dokumanlari kesfi
    4. Decay factor: Her hop icin skor azaltma

    Kullanim:
        retriever = MultiHopRetriever(graph, embedding, max_hops=3)
        results = retriever.search("sorgu", top_k=10)
    """

    # Edge tipi agirliklari
    EDGE_WEIGHTS = {
        'causes': 1.0,      # Dogrudan nedensellik
        'supports': 0.8,    # Destekleyici kanit
        'requires': 0.7,    # On kosul
        'related': 0.5,     # Genel iliski
        'contradicts': 0.3  # Celiski (dusuk agirlik)
    }

    # Hop basina decay faktoru
    HOP_DECAY = 0.7

    def __init__(
        self,
        graph: GraphEngine,
        embedding: TextEmbedding,
        max_hops: int = 3,
        min_path_score: float = 0.3,
        use_bidirectional: bool = True
    ):
        """
        Args:
            graph: Graf motoru
            embedding: Embedding motoru
            max_hops: Maksimum atlama sayisi
            min_path_score: Minimum yol skoru (altindakiler elenır)
            use_bidirectional: Cift yonlu arama
        """
        self.graph = graph
        self.embedding = embedding
        self.max_hops = max_hops
        self.min_path_score = min_path_score
        self.use_bidirectional = use_bidirectional

    def search(
        self,
        query: str,
        top_k: int = 10,
        seed_top_k: int = 5,
        alpha: float = 0.4,  # Similarity weight
        beta: float = 0.4,   # Path weight
        gamma: float = 0.2   # Importance weight
    ) -> List[MultiHopResult]:
        """
        Multi-hop arama yap.

        Args:
            query: Arama sorgusu
            top_k: Donulecek sonuc sayisi
            seed_top_k: Baslangic noktasi olarak alinacak seed sayisi
            alpha: Benzerlik agirligi
            beta: Yol agirligi
            gamma: Onem agirligi

        Returns:
            Skorlarina gore siralanmis sonuclar
        """
        # 1. Query embedding
        query_embedding = self.embedding.get_text_embedding(query)

        # 2. Seed node'lari bul (direkt benzerlik)
        seeds = self._find_seed_nodes(query_embedding, seed_top_k)

        if not seeds:
            return []

        # 3. Multi-hop expansion
        all_candidates = {}  # node_id -> MultiHopResult

        for seed_id, seed_similarity in seeds:
            # Seed node'u ekle
            if seed_id not in all_candidates:
                seed_node = self.graph.get_node(seed_id)
                if seed_node:
                    all_candidates[seed_id] = MultiHopResult(
                        node_id=seed_id,
                        content=seed_node['content'],
                        score=0.0,
                        hop_distance=0,
                        similarity_score=seed_similarity,
                        importance_score=self.graph.get_importance(seed_id),
                        metadata=seed_node.get('metadata')
                    )

            # Seed'den coklu hop expansion
            expanded = self._expand_from_node(seed_id, query_embedding)

            for node_id, result in expanded.items():
                if node_id not in all_candidates:
                    all_candidates[node_id] = result
                elif result.score > all_candidates[node_id].score:
                    # Daha iyi yol bulundu
                    all_candidates[node_id] = result

        # 4. Final skorlama
        results = []
        for node_id, result in all_candidates.items():
            # Path score (decay dahil)
            path_score = self._compute_path_score(result)

            # Final score
            final_score = (
                alpha * result.similarity_score +
                beta * path_score +
                gamma * result.importance_score
            )

            result.score = final_score
            result.causal_score = path_score
            results.append(result)

        # 5. Siralama ve filtreleme
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    def _find_seed_nodes(
        self,
        query_embedding: np.ndarray,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """Similarity ile seed node'lari bul"""
        embeddings, ids = self.graph.get_all_embeddings()

        if len(ids) == 0:
            return []

        # Cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        embed_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        similarities = np.dot(embed_norms, query_norm)

        # Top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [(ids[i], float(similarities[i])) for i in top_indices]

    def _expand_from_node(
        self,
        start_node: str,
        query_embedding: np.ndarray
    ) -> Dict[str, MultiHopResult]:
        """Bir node'dan multi-hop expansion yap"""
        results = {}
        visited = {start_node}

        # BFS ile expansion
        # Her seviye: (node_id, path, current_weight, hop_count)
        current_level = [(start_node, [start_node], [], 1.0, 0)]

        for hop in range(1, self.max_hops + 1):
            next_level = []

            for node_id, path, edge_types, path_weight, hop_count in current_level:
                # Komsulari al
                neighbors = self._get_weighted_neighbors(node_id)

                for neighbor_id, edge_type, edge_weight in neighbors:
                    if neighbor_id in visited:
                        continue

                    visited.add(neighbor_id)

                    # Yeni yol
                    new_path = path + [neighbor_id]
                    new_edges = edge_types + [edge_type]
                    new_weight = path_weight * edge_weight * self.HOP_DECAY

                    # Minimum threshold kontrolu
                    if new_weight < self.min_path_score:
                        continue

                    # Node bilgisi
                    neighbor_node = self.graph.get_node(neighbor_id)
                    if not neighbor_node:
                        continue

                    # Similarity hesapla
                    neighbor_embedding = neighbor_node.get('embedding')
                    similarity = 0.0
                    if neighbor_embedding is not None:
                        similarity = self._cosine_similarity(
                            query_embedding, neighbor_embedding
                        )

                    # Bridge node'lari bul (ara nodelar)
                    bridge_nodes = new_path[1:-1] if len(new_path) > 2 else []

                    # Result olustur
                    hop_path = HopPath(
                        nodes=new_path,
                        edges=new_edges,
                        total_weight=new_weight,
                        hop_count=hop
                    )

                    result = MultiHopResult(
                        node_id=neighbor_id,
                        content=neighbor_node['content'],
                        score=new_weight,  # Gecici skor
                        hop_distance=hop,
                        path=hop_path,
                        bridge_nodes=bridge_nodes,
                        similarity_score=similarity,
                        importance_score=self.graph.get_importance(neighbor_id),
                        metadata=neighbor_node.get('metadata')
                    )

                    results[neighbor_id] = result

                    # Sonraki seviye icin ekle
                    if hop < self.max_hops:
                        next_level.append((
                            neighbor_id, new_path, new_edges, new_weight, hop
                        ))

            current_level = next_level

        return results

    def _get_weighted_neighbors(
        self,
        node_id: str
    ) -> List[Tuple[str, str, float]]:
        """Agirlikli komsulari getir"""
        neighbors = []

        # Ileri yonlu (successors)
        for edge_type in self.EDGE_WEIGHTS.keys():
            forward = self.graph.get_neighbors(node_id, [edge_type])
            weight = self.EDGE_WEIGHTS.get(edge_type, 0.5)
            for n in forward:
                neighbors.append((n, edge_type, weight))

        # Geri yonlu (predecessors) - opsiyonel
        if self.use_bidirectional:
            for edge_type in self.EDGE_WEIGHTS.keys():
                backward = self.graph.get_predecessors(node_id, [edge_type])
                weight = self.EDGE_WEIGHTS.get(edge_type, 0.5) * 0.8  # Geri yon biraz dusuk
                for n in backward:
                    neighbors.append((n, f"rev_{edge_type}", weight))

        return neighbors

    def _compute_path_score(self, result: MultiHopResult) -> float:
        """Yol kalite skoru hesapla"""
        if result.path is None:
            return result.similarity_score  # Direkt baglanti

        # Decay ile path weight
        return result.path.total_weight

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity hesapla"""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)

        if a_norm == 0 or b_norm == 0:
            return 0.0

        return float(np.dot(a, b) / (a_norm * b_norm))

    def find_paths_between(
        self,
        source_id: str,
        target_id: str,
        max_paths: int = 5
    ) -> List[HopPath]:
        """
        Iki node arasindaki tum yollari bul.

        Kullanim:
            paths = retriever.find_paths_between("doc_a", "doc_b")
            for path in paths:
                print(f"Yol: {' -> '.join(path.nodes)}")
        """
        all_paths = []

        def dfs(current: str, target: str, path: List[str], edges: List[str],
                weight: float, depth: int, visited: Set[str]):

            if len(all_paths) >= max_paths:
                return

            if current == target:
                all_paths.append(HopPath(
                    nodes=path.copy(),
                    edges=edges.copy(),
                    total_weight=weight,
                    hop_count=depth
                ))
                return

            if depth >= self.max_hops:
                return

            # Komsulari gez
            neighbors = self._get_weighted_neighbors(current)
            for neighbor_id, edge_type, edge_weight in neighbors:
                if neighbor_id in visited:
                    continue

                new_weight = weight * edge_weight * self.HOP_DECAY
                if new_weight < self.min_path_score:
                    continue

                visited.add(neighbor_id)
                path.append(neighbor_id)
                edges.append(edge_type)

                dfs(neighbor_id, target, path, edges, new_weight, depth + 1, visited)

                path.pop()
                edges.pop()
                visited.remove(neighbor_id)

        # DFS baslat
        dfs(source_id, target_id, [source_id], [], 1.0, 0, {source_id})

        # Agirliga gore sirala
        all_paths.sort(key=lambda p: p.total_weight, reverse=True)

        return all_paths[:max_paths]

    def explain_connection(
        self,
        source_id: str,
        target_id: str
    ) -> Optional[str]:
        """
        Iki dokuman arasindaki baglantıyi acikla.

        Returns:
            Aciklama metni veya None
        """
        paths = self.find_paths_between(source_id, target_id, max_paths=1)

        if not paths:
            return None

        path = paths[0]

        # Aciklama olustur
        explanation_parts = []

        for i, node_id in enumerate(path.nodes):
            node = self.graph.get_node(node_id)
            if node:
                content_preview = node['content'][:100] + "..." \
                    if len(node['content']) > 100 else node['content']

                if i == 0:
                    explanation_parts.append(f"Baslangic: [{node_id}] {content_preview}")
                elif i == len(path.nodes) - 1:
                    edge = path.edges[i-1] if i > 0 else ""
                    explanation_parts.append(f"  --({edge})--> [{node_id}] {content_preview}")
                else:
                    edge = path.edges[i-1] if i > 0 else ""
                    explanation_parts.append(f"  --({edge})--> [{node_id}] {content_preview}")

        return "\n".join(explanation_parts)


def create_multi_hop_retriever(
    graph: GraphEngine,
    embedding: TextEmbedding,
    **kwargs
) -> MultiHopRetriever:
    """Factory function for MultiHopRetriever"""
    return MultiHopRetriever(graph, embedding, **kwargs)
