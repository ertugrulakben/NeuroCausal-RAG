"""
NeuroCausal RAG - Query Decomposer
v5.2 - FAZ 2.3

Karmasik sorgulari alt sorgulara ayirma ve birlestirme.

Ornek:
    Orijinal: "Sera gazlari buzullari nasil etkiler ve bu deniz seviyesini nasil yukseltir?"

    Alt sorgular:
    1. "Sera gazlari buzullari nasil etkiler?"
    2. "Buzul erimesi deniz seviyesini nasil etkiler?"

    Her alt sorgu ayri aranir, sonuclar birlestirilir.

Yazar: Ertugrul Akben
"""

import re
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..interfaces import SearchResult

logger = logging.getLogger(__name__)


class DecompositionStrategy(Enum):
    """Ayristirma stratejileri"""
    CONJUNCTION = "conjunction"       # "ve", "and" ile ayristirma
    CAUSAL_CHAIN = "causal_chain"    # Nedensel zincir ayristirma
    MULTI_ASPECT = "multi_aspect"    # Coklu boyut ayristirma
    TEMPORAL = "temporal"            # Zamansal ayristirma
    ENTITY_FOCUS = "entity_focus"    # Entity bazli ayristirma


@dataclass
class SubQuery:
    """Alt sorgu temsili"""
    text: str
    original_part: str  # Orijinal sorgudaki karsilik
    query_type: str  # factual, causal, temporal
    weight: float = 1.0  # Birlestirme agirligi
    entities: List[str] = field(default_factory=list)


@dataclass
class DecompositionResult:
    """Ayristirma sonucu"""
    original_query: str
    strategy_used: DecompositionStrategy
    sub_queries: List[SubQuery]
    is_decomposed: bool  # Ayristirma yapildi mi
    decomposition_reason: str = ""


@dataclass
class MergedResult:
    """Birlestirilmis arama sonucu"""
    node_id: str
    content: str
    final_score: float
    contributing_queries: List[str]  # Hangi sub-query'lerden geldi
    individual_scores: Dict[str, float]  # Sub-query basina skor
    coverage_score: float  # Kac sub-query'yi karsiladigi
    metadata: Optional[Dict] = None


class QueryDecomposer:
    """
    Sorgu ayristirici.

    Karmasik sorgulari alt sorgulara ayirir.
    """

    # "ve", "and" baglacları
    CONJUNCTION_PATTERNS = [
        r'\s+ve\s+',
        r'\s+and\s+',
        r'\s+ayrica\s+',
        r'\s+also\s+',
        r',\s*(?=\w)',  # Virgül ile ayrilmis
    ]

    # Nedensel zincir kaliplari
    CAUSAL_CHAIN_PATTERNS = [
        r'(.+?)\s+sonucunda\s+(.+)',
        r'(.+?)\s+nedeniyle\s+(.+)',
        r'(.+?)\s+causes\s+(.+)',
        r'(.+?)\s+leads?\s+to\s+(.+)',
        r'(.+?)\s+(?:ve|and)\s+bu\s+(.+)',  # "ve bu sonuc..."
    ]

    # Soru kaliplari (ayristirma icin)
    MULTI_QUESTION_PATTERNS = [
        r'\?[^?]*\?',  # Birden fazla soru isareti
        r'(ne|nasil|neden|kim|nerede).+?(ne|nasil|neden|kim|nerede)',
        r'(what|how|why|who|where).+?(what|how|why|who|where)',
    ]

    def __init__(
        self,
        min_subquery_length: int = 10,
        max_subqueries: int = 5
    ):
        """
        Args:
            min_subquery_length: Minimum alt sorgu uzunlugu
            max_subqueries: Maksimum alt sorgu sayisi
        """
        self.min_subquery_length = min_subquery_length
        self.max_subqueries = max_subqueries

    def decompose(self, query: str) -> DecompositionResult:
        """
        Sorguyu alt sorgulara ayir.

        Args:
            query: Orijinal sorgu

        Returns:
            Ayristirma sonucu
        """
        query = query.strip()

        # 1. Nedensel zincir kontrolu
        causal_result = self._try_causal_decomposition(query)
        if causal_result:
            return causal_result

        # 2. Baglac kontrolu ("ve", "and")
        conjunction_result = self._try_conjunction_decomposition(query)
        if conjunction_result:
            return conjunction_result

        # 3. Coklu soru kontrolu
        multi_result = self._try_multi_question_decomposition(query)
        if multi_result:
            return multi_result

        # Ayristirma yapilmadi
        return DecompositionResult(
            original_query=query,
            strategy_used=DecompositionStrategy.CONJUNCTION,
            sub_queries=[SubQuery(
                text=query,
                original_part=query,
                query_type="general",
                weight=1.0
            )],
            is_decomposed=False,
            decomposition_reason="Sorgu ayristirma gerektirmiyor"
        )

    def _try_causal_decomposition(
        self,
        query: str
    ) -> Optional[DecompositionResult]:
        """Nedensel zincir ayristirma dene"""
        for pattern in self.CAUSAL_CHAIN_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    cause_part = groups[0].strip()
                    effect_part = groups[1].strip()

                    if (len(cause_part) >= self.min_subquery_length and
                        len(effect_part) >= self.min_subquery_length):

                        sub_queries = [
                            SubQuery(
                                text=f"{cause_part}?",
                                original_part=cause_part,
                                query_type="causal_cause",
                                weight=1.0
                            ),
                            SubQuery(
                                text=f"{effect_part}?",
                                original_part=effect_part,
                                query_type="causal_effect",
                                weight=1.0
                            )
                        ]

                        return DecompositionResult(
                            original_query=query,
                            strategy_used=DecompositionStrategy.CAUSAL_CHAIN,
                            sub_queries=sub_queries,
                            is_decomposed=True,
                            decomposition_reason=f"Nedensel zincir tespit edildi: '{cause_part}' -> '{effect_part}'"
                        )

        return None

    def _try_conjunction_decomposition(
        self,
        query: str
    ) -> Optional[DecompositionResult]:
        """Baglac ile ayristirma dene"""
        for pattern in self.CONJUNCTION_PATTERNS:
            parts = re.split(pattern, query, flags=re.IGNORECASE)
            if len(parts) >= 2:
                # Minimum uzunluk kontrolu
                valid_parts = [
                    p.strip() for p in parts
                    if len(p.strip()) >= self.min_subquery_length
                ]

                if len(valid_parts) >= 2:
                    sub_queries = [
                        SubQuery(
                            text=self._clean_subquery(part),
                            original_part=part,
                            query_type="conjunction_part",
                            weight=1.0 / len(valid_parts)  # Esit agirlik
                        )
                        for part in valid_parts[:self.max_subqueries]
                    ]

                    return DecompositionResult(
                        original_query=query,
                        strategy_used=DecompositionStrategy.CONJUNCTION,
                        sub_queries=sub_queries,
                        is_decomposed=True,
                        decomposition_reason=f"{len(sub_queries)} alt sorgu ayristirildi"
                    )

        return None

    def _try_multi_question_decomposition(
        self,
        query: str
    ) -> Optional[DecompositionResult]:
        """Coklu soru ayristirma dene"""
        # Soru isareti ile ayir
        if query.count('?') > 1:
            parts = [p.strip() + '?' for p in query.split('?') if p.strip()]
            if len(parts) >= 2:
                sub_queries = [
                    SubQuery(
                        text=part,
                        original_part=part,
                        query_type="multi_question",
                        weight=1.0 / len(parts)
                    )
                    for part in parts[:self.max_subqueries]
                ]

                return DecompositionResult(
                    original_query=query,
                    strategy_used=DecompositionStrategy.MULTI_ASPECT,
                    sub_queries=sub_queries,
                    is_decomposed=True,
                    decomposition_reason=f"{len(parts)} soru tespit edildi"
                )

        return None

    def _clean_subquery(self, part: str) -> str:
        """Alt sorguyu temizle ve sorguya cevir"""
        part = part.strip()

        # Boslukla baslayan/biten kelimeleri temizle
        part = re.sub(r'^\W+', '', part)
        part = re.sub(r'\W+$', '', part)

        # Soru isareti ekle (yoksa)
        if not part.endswith('?'):
            part = part + '?'

        return part


class ResultMerger:
    """
    Alt sorgu sonuclarini birlestirici.
    """

    def __init__(
        self,
        coverage_weight: float = 0.3,
        score_aggregation: str = "weighted_sum"  # weighted_sum, max, min
    ):
        """
        Args:
            coverage_weight: Coverage skorunun final skora katkisi
            score_aggregation: Skor birlestirme yontemi
        """
        self.coverage_weight = coverage_weight
        self.score_aggregation = score_aggregation

    def merge(
        self,
        sub_results: Dict[str, List[SearchResult]],
        sub_queries: List[SubQuery]
    ) -> List[MergedResult]:
        """
        Alt sorgu sonuclarini birlestir.

        Args:
            sub_results: Sub-query text -> sonuclar mapping
            sub_queries: Alt sorgu bilgileri

        Returns:
            Birlestirilmis sonuclar
        """
        # Node ID -> sonuc bilgileri
        merged_map: Dict[str, Dict] = {}

        # Her sub-query sonucunu isle
        for sub_query in sub_queries:
            results = sub_results.get(sub_query.text, [])

            for result in results:
                node_id = result.node_id

                if node_id not in merged_map:
                    merged_map[node_id] = {
                        'content': result.content,
                        'scores': {},
                        'queries': [],
                        'metadata': result.metadata
                    }

                # Agirlikli skor
                weighted_score = result.score * sub_query.weight
                merged_map[node_id]['scores'][sub_query.text] = weighted_score
                merged_map[node_id]['queries'].append(sub_query.text)

        # Final sonuclari olustur
        merged_results = []
        total_queries = len(sub_queries)

        for node_id, data in merged_map.items():
            # Coverage: Kac sub-query'yi karsiladigi
            coverage = len(set(data['queries'])) / total_queries

            # Skor agregasyonu
            scores = list(data['scores'].values())
            if self.score_aggregation == "weighted_sum":
                base_score = sum(scores)
            elif self.score_aggregation == "max":
                base_score = max(scores) if scores else 0
            elif self.score_aggregation == "min":
                base_score = min(scores) if scores else 0
            else:
                base_score = sum(scores) / len(scores) if scores else 0

            # Coverage bonus
            final_score = (
                (1 - self.coverage_weight) * base_score +
                self.coverage_weight * coverage
            )

            merged_results.append(MergedResult(
                node_id=node_id,
                content=data['content'],
                final_score=final_score,
                contributing_queries=list(set(data['queries'])),
                individual_scores=data['scores'],
                coverage_score=coverage,
                metadata=data['metadata']
            ))

        # Final skora gore sirala
        merged_results.sort(key=lambda x: x.final_score, reverse=True)

        return merged_results


class DecomposedSearch:
    """
    Tam ayristirmali arama pipeline'i.

    Kullanim:
        decomposed = DecomposedSearch(retriever)
        results = decomposed.search("Sera gazlari ve buzul erimesi")
    """

    def __init__(
        self,
        retriever,  # Retriever veya MultiHopRetriever
        decomposer: Optional[QueryDecomposer] = None,
        merger: Optional[ResultMerger] = None
    ):
        self.retriever = retriever
        self.decomposer = decomposer or QueryDecomposer()
        self.merger = merger or ResultMerger()

    def search(
        self,
        query: str,
        top_k: int = 10,
        sub_query_top_k: int = 5
    ) -> Tuple[List[MergedResult], DecompositionResult]:
        """
        Ayristirmali arama yap.

        Args:
            query: Arama sorgusu
            top_k: Donulecek toplam sonuc sayisi
            sub_query_top_k: Her alt sorgu icin sonuc sayisi

        Returns:
            (Birlestirilmis sonuclar, Ayristirma bilgisi)
        """
        # 1. Sorguyu ayristir
        decomposition = self.decomposer.decompose(query)

        # 2. Her sub-query icin arama yap
        sub_results = {}
        for sub_query in decomposition.sub_queries:
            results = self.retriever.search(
                sub_query.text,
                top_k=sub_query_top_k
            )
            sub_results[sub_query.text] = results

            logger.info(
                f"Sub-query '{sub_query.text[:50]}...' -> {len(results)} sonuc"
            )

        # 3. Sonuclari birlestir
        merged = self.merger.merge(sub_results, decomposition.sub_queries)

        return merged[:top_k], decomposition


def create_decomposer(**kwargs) -> QueryDecomposer:
    """Factory function"""
    return QueryDecomposer(**kwargs)


def create_merger(**kwargs) -> ResultMerger:
    """Factory function"""
    return ResultMerger(**kwargs)
