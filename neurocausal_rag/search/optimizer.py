"""
NeuroCausal RAG - Search Optimizer
v5.2 - FAZ 2.2

Hibrit arama optimizasyonu:
1. Sorgu analizi ile otomatik agirlik ayarlama
2. Arama modu presetleri
3. Sonuc cesitlendirme (diversification)
4. Adaptif re-ranking

Yazar: Ertugrul Akben
"""

import re
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..interfaces import SearchResult

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    """Arama modu presetleri"""
    BALANCED = "balanced"           # Dengeli (varsayilan)
    ENCYCLOPEDIA = "encyclopedia"   # Bilgi odakli (yuksek similarity)
    DETECTIVE = "detective"         # Nedensellik odakli (yuksek causal)
    HUB = "hub"                     # Hub odakli (yuksek importance)
    EXPLORER = "explorer"           # Kesfedici (diversification)
    FACT_CHECKER = "fact_checker"   # Dogrulama odakli


@dataclass
class SearchWeights:
    """Arama agirliklari"""
    alpha: float = 0.5   # Similarity weight
    beta: float = 0.3    # Causal weight
    gamma: float = 0.2   # Importance weight

    def normalize(self) -> 'SearchWeights':
        """Agirliklari normalize et (toplam = 1.0)"""
        total = self.alpha + self.beta + self.gamma
        if total == 0:
            return SearchWeights(0.33, 0.33, 0.34)
        return SearchWeights(
            alpha=self.alpha / total,
            beta=self.beta / total,
            gamma=self.gamma / total
        )

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.alpha, self.beta, self.gamma)


@dataclass
class QueryAnalysis:
    """Sorgu analizi sonucu"""
    original_query: str
    query_type: str  # factual, causal, exploratory, specific
    has_causal_intent: bool
    has_temporal_markers: bool
    is_question: bool
    key_entities: List[str] = field(default_factory=list)
    suggested_mode: SearchMode = SearchMode.BALANCED
    suggested_weights: SearchWeights = field(default_factory=SearchWeights)
    confidence: float = 0.5


class QueryAnalyzer:
    """
    Sorgu analiz edici.

    Sorguyu analiz ederek en uygun arama stratejisini belirler.
    """

    # Nedensellik ifadeleri
    CAUSAL_PATTERNS = [
        r'\b(neden|sebep|yol ac|neden ol|sonuc|etki|nasil)\b',
        r'\b(causes?|leads?\s+to|results?\s+in|because|why|how)\b',
        r'\b(nedeniyle|sonucunda|yuzunden|sebebiyle)\b',
        r'(\?|nasil|neden|nicin)',
    ]

    # Zamansal ifadeler
    TEMPORAL_PATTERNS = [
        r'\b(once|sonra|sirasinda|when|before|after)\b',
        r'\b(\d{4}|\d{2}/\d{2}|\d{1,2}\.\d{1,2}\.\d{4})\b',
        r'\b(yil|ay|hafta|gun|saat)\b',
    ]

    # Soru kaliplari
    QUESTION_PATTERNS = [
        r'\?$',
        r'^(ne|kim|nerede|nasil|nicin|neden|hangi|kac)\b',
        r'^(what|who|where|how|why|which|when)\b',
    ]

    # Spesifik sorgu kaliplari
    SPECIFIC_PATTERNS = [
        r'"[^"]+"',  # Tirnak icinde
        r'\b(tam olarak|spesifik|exactly|specific)\b',
    ]

    def analyze(self, query: str) -> QueryAnalysis:
        """Sorguyu analiz et"""
        query_lower = query.lower()

        # Nedensellik niyeti
        has_causal = any(
            re.search(p, query_lower) for p in self.CAUSAL_PATTERNS
        )

        # Zamansal isaretler
        has_temporal = any(
            re.search(p, query_lower) for p in self.TEMPORAL_PATTERNS
        )

        # Soru mu?
        is_question = any(
            re.search(p, query_lower) for p in self.QUESTION_PATTERNS
        )

        # Spesifik mi?
        is_specific = any(
            re.search(p, query_lower) for p in self.SPECIFIC_PATTERNS
        )

        # Sorgu tipi belirleme
        if has_causal:
            query_type = "causal"
            suggested_mode = SearchMode.DETECTIVE
            suggested_weights = SearchWeights(0.3, 0.5, 0.2)
            confidence = 0.8
        elif is_specific:
            query_type = "specific"
            suggested_mode = SearchMode.ENCYCLOPEDIA
            suggested_weights = SearchWeights(0.6, 0.2, 0.2)
            confidence = 0.7
        elif is_question:
            query_type = "factual"
            suggested_mode = SearchMode.BALANCED
            suggested_weights = SearchWeights(0.5, 0.3, 0.2)
            confidence = 0.6
        else:
            query_type = "exploratory"
            suggested_mode = SearchMode.EXPLORER
            suggested_weights = SearchWeights(0.4, 0.3, 0.3)
            confidence = 0.5

        return QueryAnalysis(
            original_query=query,
            query_type=query_type,
            has_causal_intent=has_causal,
            has_temporal_markers=has_temporal,
            is_question=is_question,
            suggested_mode=suggested_mode,
            suggested_weights=suggested_weights.normalize(),
            confidence=confidence
        )


class SearchOptimizer:
    """
    Arama optimizasyonu motoru.

    Ozellikler:
    1. Mod bazli agirlik presetleri
    2. Otomatik sorgu analizi
    3. Sonuc cesitlendirme (MMR)
    4. Re-ranking stratejileri
    """

    # Mod presetleri
    MODE_PRESETS: Dict[SearchMode, SearchWeights] = {
        SearchMode.BALANCED: SearchWeights(0.5, 0.3, 0.2),
        SearchMode.ENCYCLOPEDIA: SearchWeights(0.7, 0.2, 0.1),
        SearchMode.DETECTIVE: SearchWeights(0.3, 0.5, 0.2),
        SearchMode.HUB: SearchWeights(0.3, 0.2, 0.5),
        SearchMode.EXPLORER: SearchWeights(0.4, 0.3, 0.3),
        SearchMode.FACT_CHECKER: SearchWeights(0.6, 0.3, 0.1),
    }

    def __init__(
        self,
        auto_analyze: bool = True,
        diversity_threshold: float = 0.7,
        enable_mmr: bool = True
    ):
        """
        Args:
            auto_analyze: Sorgu otomatik analiz edilsin mi
            diversity_threshold: MMR cesitlilik esigi
            enable_mmr: MMR (Maximal Marginal Relevance) aktif mi
        """
        self.auto_analyze = auto_analyze
        self.diversity_threshold = diversity_threshold
        self.enable_mmr = enable_mmr
        self.analyzer = QueryAnalyzer()

    def get_weights(
        self,
        query: str,
        mode: Optional[SearchMode] = None
    ) -> SearchWeights:
        """
        Sorgu icin optimal agirliklari dondur.

        Args:
            query: Arama sorgusu
            mode: Manuel mod secimi (None ise otomatik)

        Returns:
            Optimize edilmis agirliklar
        """
        if mode:
            return self.MODE_PRESETS[mode].normalize()

        if self.auto_analyze:
            analysis = self.analyzer.analyze(query)
            logger.info(
                f"Query analyzed: type={analysis.query_type}, "
                f"mode={analysis.suggested_mode.value}, "
                f"confidence={analysis.confidence:.2f}"
            )
            return analysis.suggested_weights

        return self.MODE_PRESETS[SearchMode.BALANCED]

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Sorguyu analiz et"""
        return self.analyzer.analyze(query)

    def diversify_results(
        self,
        results: List[SearchResult],
        embeddings: Dict[str, np.ndarray],
        top_k: int = 10,
        lambda_param: float = 0.5
    ) -> List[SearchResult]:
        """
        MMR (Maximal Marginal Relevance) ile sonuclari cesitlendir.

        Bu algoritma, hem relevance hem de diversity'yi dengeler.

        Args:
            results: Orijinal sonuclar
            embeddings: Node ID -> embedding mapping
            top_k: Donulecek sonuc sayisi
            lambda_param: Relevance vs diversity dengesi (0-1)
                         1.0 = sadece relevance
                         0.0 = sadece diversity

        Returns:
            Cesitlendirilmis sonuclar
        """
        if not self.enable_mmr or len(results) <= top_k:
            return results[:top_k]

        selected = []
        remaining = list(results)

        # Ilk sonucu ekle (en yuksek skor)
        selected.append(remaining.pop(0))

        while len(selected) < top_k and remaining:
            best_idx = -1
            best_mmr = float('-inf')

            for i, candidate in enumerate(remaining):
                # Relevance skoru
                relevance = candidate.score

                # Diversity: secilenlere olan max benzerlik
                max_sim = 0.0
                cand_emb = embeddings.get(candidate.node_id)

                if cand_emb is not None:
                    for sel in selected:
                        sel_emb = embeddings.get(sel.node_id)
                        if sel_emb is not None:
                            sim = self._cosine_similarity(cand_emb, sel_emb)
                            max_sim = max(max_sim, sim)

                # MMR skoru
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
            else:
                break

        return selected

    def rerank_by_coverage(
        self,
        results: List[SearchResult],
        query_terms: List[str]
    ) -> List[SearchResult]:
        """
        Query term coverage'a gore yeniden sirala.

        Args:
            results: Orijinal sonuclar
            query_terms: Sorgu terimleri

        Returns:
            Coverage'a gore yeniden siralanmis sonuclar
        """
        if not query_terms:
            return results

        scored_results = []

        for result in results:
            content_lower = result.content.lower()
            coverage = sum(
                1 for term in query_terms
                if term.lower() in content_lower
            ) / len(query_terms)

            # Orijinal skor + coverage bonus
            boosted_score = result.score * (1 + 0.2 * coverage)

            # Yeni result objesi olustur
            new_result = SearchResult(
                node_id=result.node_id,
                content=result.content,
                score=boosted_score,
                similarity_score=result.similarity_score,
                causal_score=result.causal_score,
                importance_score=result.importance_score,
                causal_chain=result.causal_chain,
                metadata=result.metadata
            )
            scored_results.append(new_result)

        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results

    def combine_multi_hop_results(
        self,
        direct_results: List[SearchResult],
        multi_hop_results: List,  # MultiHopResult
        hop_penalty: float = 0.15
    ) -> List[SearchResult]:
        """
        Direkt ve multi-hop sonuclari birlestir.

        Args:
            direct_results: Direkt arama sonuclari
            multi_hop_results: Multi-hop sonuclari
            hop_penalty: Her hop icin skor penalti

        Returns:
            Birlestirilmis ve siralanmis sonuclar
        """
        combined = {}

        # Direkt sonuclari ekle
        for result in direct_results:
            combined[result.node_id] = result

        # Multi-hop sonuclari ekle/guncelle
        for mh_result in multi_hop_results:
            # Hop bazli penalti
            adjusted_score = mh_result.score * (1 - hop_penalty * mh_result.hop_distance)

            if mh_result.node_id in combined:
                # Yuksek skoru al
                existing = combined[mh_result.node_id]
                if adjusted_score > existing.score:
                    combined[mh_result.node_id] = SearchResult(
                        node_id=mh_result.node_id,
                        content=mh_result.content,
                        score=adjusted_score,
                        similarity_score=mh_result.similarity_score,
                        causal_score=mh_result.causal_score,
                        importance_score=mh_result.importance_score,
                        metadata={
                            **(mh_result.metadata or {}),
                            'hop_distance': mh_result.hop_distance,
                            'bridge_nodes': mh_result.bridge_nodes
                        }
                    )
            else:
                combined[mh_result.node_id] = SearchResult(
                    node_id=mh_result.node_id,
                    content=mh_result.content,
                    score=adjusted_score,
                    similarity_score=mh_result.similarity_score,
                    causal_score=mh_result.causal_score,
                    importance_score=mh_result.importance_score,
                    metadata={
                        **(mh_result.metadata or {}),
                        'hop_distance': mh_result.hop_distance,
                        'bridge_nodes': mh_result.bridge_nodes
                    }
                )

        # Sirala
        results = list(combined.values())
        results.sort(key=lambda x: x.score, reverse=True)

        return results

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity hesapla"""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)

        if a_norm == 0 or b_norm == 0:
            return 0.0

        return float(np.dot(a, b) / (a_norm * b_norm))


def get_mode_preset(mode: SearchMode) -> SearchWeights:
    """Mod presetini dondur"""
    return SearchOptimizer.MODE_PRESETS.get(mode, SearchWeights())


def create_optimizer(**kwargs) -> SearchOptimizer:
    """Factory function"""
    return SearchOptimizer(**kwargs)
