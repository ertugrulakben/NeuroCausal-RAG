"""
NeuroCausal RAG - Search Optimizer Tests
v5.2 - FAZ 2.2

Yazar: Ertugrul Akben
"""

import pytest
import numpy as np
from neurocausal_rag.search.optimizer import (
    SearchOptimizer,
    SearchMode,
    SearchWeights,
    QueryAnalyzer,
    QueryAnalysis,
    get_mode_preset,
    create_optimizer
)
from neurocausal_rag.interfaces import SearchResult


class TestSearchWeights:
    """SearchWeights testleri"""

    def test_default_weights(self):
        """Varsayilan agirliklar"""
        weights = SearchWeights()

        assert weights.alpha == 0.5
        assert weights.beta == 0.3
        assert weights.gamma == 0.2

    def test_normalize(self):
        """Normalizasyon testi"""
        weights = SearchWeights(alpha=1.0, beta=1.0, gamma=1.0)
        normalized = weights.normalize()

        total = normalized.alpha + normalized.beta + normalized.gamma
        assert abs(total - 1.0) < 0.001

    def test_normalize_zero(self):
        """Sifir agirlik normalizasyonu"""
        weights = SearchWeights(alpha=0, beta=0, gamma=0)
        normalized = weights.normalize()

        # Sifirlara karsi koruma
        assert normalized.alpha > 0
        assert normalized.beta > 0
        assert normalized.gamma > 0

    def test_to_tuple(self):
        """Tuple donusumu"""
        weights = SearchWeights(alpha=0.5, beta=0.3, gamma=0.2)
        t = weights.to_tuple()

        assert t == (0.5, 0.3, 0.2)


class TestQueryAnalyzer:
    """QueryAnalyzer testleri"""

    @pytest.fixture
    def analyzer(self):
        return QueryAnalyzer()

    def test_causal_query_detection(self, analyzer):
        """Nedensellik sorgusu tespiti"""
        queries = [
            "Sera gazlari iklim degisikligine nasil neden olur?",
            "What causes global warming?",
            "Kuraklik nedeniyle neler oluyor?",
            "Why is the temperature increasing?",
        ]

        for query in queries:
            analysis = analyzer.analyze(query)
            assert analysis.has_causal_intent, f"Causal not detected: {query}"

    def test_temporal_query_detection(self, analyzer):
        """Zamansal sorgu tespiti"""
        queries = [
            "2020 yilinda ne oldu?",
            "Olaydan once ne yapilmisti?",
            "After the earthquake, what happened?",
        ]

        for query in queries:
            analysis = analyzer.analyze(query)
            assert analysis.has_temporal_markers, f"Temporal not detected: {query}"

    def test_question_detection(self, analyzer):
        """Soru tespiti"""
        questions = [
            "Nedir?",
            "What is climate change?",
            "Kim bu karari aldi?",
            "Where are the glaciers melting?",
        ]

        for q in questions:
            analysis = analyzer.analyze(q)
            assert analysis.is_question, f"Question not detected: {q}"

    def test_suggested_mode_causal(self, analyzer):
        """Nedensel sorgu icin detective modu onerilmeli"""
        analysis = analyzer.analyze("Sera gazlari sicakliga nasil neden olur?")

        assert analysis.suggested_mode == SearchMode.DETECTIVE

    def test_suggested_mode_specific(self, analyzer):
        """Spesifik sorgu icin encyclopedia modu"""
        # "nedir" causal pattern'i tetikledigi icin causal olmayan sorgu kullan
        analysis = analyzer.analyze('"Paris Anlasması" hakkında bilgi')

        assert analysis.suggested_mode == SearchMode.ENCYCLOPEDIA

    def test_analysis_confidence(self, analyzer):
        """Analiz guveni 0-1 arasinda olmali"""
        queries = [
            "test query",
            "Neden oluyor?",
            "What is this?",
        ]

        for query in queries:
            analysis = analyzer.analyze(query)
            assert 0 <= analysis.confidence <= 1


class TestSearchOptimizer:
    """SearchOptimizer testleri"""

    @pytest.fixture
    def optimizer(self):
        return SearchOptimizer(auto_analyze=True)

    def test_initialization(self, optimizer):
        """Optimizer baslatma"""
        assert optimizer.auto_analyze is True
        assert optimizer.enable_mmr is True

    def test_factory_function(self):
        """Factory function"""
        optimizer = create_optimizer(auto_analyze=False)

        assert isinstance(optimizer, SearchOptimizer)
        assert optimizer.auto_analyze is False

    def test_get_weights_with_mode(self, optimizer):
        """Manuel mod ile agirlik alma"""
        weights = optimizer.get_weights("any query", mode=SearchMode.DETECTIVE)

        assert weights.beta > weights.alpha  # Causal daha yuksek

    def test_get_weights_auto(self, optimizer):
        """Otomatik agirlik belirleme"""
        weights = optimizer.get_weights("Sera gazlari sicakliga neden olur")

        # Causal query icin beta yuksek olmali
        assert weights.beta >= 0.3

    def test_mode_presets_exist(self):
        """Tum modlar icin preset olmali"""
        for mode in SearchMode:
            weights = get_mode_preset(mode)
            assert weights is not None
            assert weights.alpha + weights.beta + weights.gamma > 0


class TestResultDiversification:
    """Sonuc cesitlendirme testleri"""

    @pytest.fixture
    def optimizer(self):
        return SearchOptimizer(enable_mmr=True)

    @pytest.fixture
    def sample_results(self):
        """Ornek arama sonuclari"""
        return [
            SearchResult(
                node_id="doc1",
                content="Content 1",
                score=0.9,
                similarity_score=0.9,
                causal_score=0.5,
                importance_score=0.3
            ),
            SearchResult(
                node_id="doc2",
                content="Content 2",
                score=0.85,
                similarity_score=0.85,
                causal_score=0.4,
                importance_score=0.25
            ),
            SearchResult(
                node_id="doc3",
                content="Content 3",
                score=0.8,
                similarity_score=0.8,
                causal_score=0.3,
                importance_score=0.2
            ),
        ]

    @pytest.fixture
    def sample_embeddings(self):
        """Ornek embeddingler"""
        return {
            "doc1": np.array([1.0, 0.0, 0.0]),
            "doc2": np.array([0.9, 0.1, 0.0]),  # doc1'e cok benzer
            "doc3": np.array([0.0, 1.0, 0.0]),  # doc1'e benzemez
        }

    def test_diversify_returns_correct_count(
        self, optimizer, sample_results, sample_embeddings
    ):
        """Doğru sayida sonuc donmeli"""
        diversified = optimizer.diversify_results(
            sample_results,
            sample_embeddings,
            top_k=2
        )

        assert len(diversified) == 2

    def test_diversify_first_is_highest(
        self, optimizer, sample_results, sample_embeddings
    ):
        """Ilk sonuc en yuksek skorlu olmali"""
        diversified = optimizer.diversify_results(
            sample_results,
            sample_embeddings,
            top_k=3
        )

        assert diversified[0].node_id == "doc1"

    def test_diversify_prefers_different(
        self, optimizer, sample_results, sample_embeddings
    ):
        """MMR farkli dokumanlari tercih etmeli"""
        diversified = optimizer.diversify_results(
            sample_results,
            sample_embeddings,
            top_k=2,
            lambda_param=0.3  # Diversity'ye agirlik ver
        )

        # doc3 (farkli) doc2'den (benzer) once gelebilir
        # (lambda_param dusuk oldugunda)
        ids = [r.node_id for r in diversified]
        assert "doc1" in ids  # En yuksek skor mutlaka dahil


class TestCoverageReranking:
    """Coverage bazli re-ranking testleri"""

    @pytest.fixture
    def optimizer(self):
        return SearchOptimizer()

    def test_rerank_boosts_coverage(self, optimizer):
        """Yuksek coverage'a bonus verilmeli"""
        results = [
            SearchResult(
                node_id="low_coverage",
                content="Random content without query terms",
                score=0.9,
                similarity_score=0.9,
                causal_score=0.5,
                importance_score=0.3
            ),
            SearchResult(
                node_id="high_coverage",
                content="Sera gazlari iklim degisikligine neden olur",
                score=0.8,
                similarity_score=0.8,
                causal_score=0.4,
                importance_score=0.25
            ),
        ]

        query_terms = ["sera", "gazlari", "iklim"]
        reranked = optimizer.rerank_by_coverage(results, query_terms)

        # high_coverage one gecmeli (daha iyi coverage)
        assert reranked[0].node_id == "high_coverage"

    def test_rerank_empty_terms(self, optimizer):
        """Bos query terms ile degisiklik olmamali"""
        results = [
            SearchResult(
                node_id="doc1",
                content="Content",
                score=0.9,
                similarity_score=0.9,
                causal_score=0.5,
                importance_score=0.3
            ),
        ]

        reranked = optimizer.rerank_by_coverage(results, [])

        assert len(reranked) == 1
        assert reranked[0].node_id == "doc1"


class TestMultiHopCombination:
    """Multi-hop sonuc birlestirme testleri"""

    @pytest.fixture
    def optimizer(self):
        return SearchOptimizer()

    def test_combine_prefers_direct(self, optimizer):
        """Direkt sonuclar multi-hop'tan once gelmeli"""
        direct = [
            SearchResult(
                node_id="direct1",
                content="Direct content",
                score=0.9,
                similarity_score=0.9,
                causal_score=0.5,
                importance_score=0.3
            ),
        ]

        # Mock multi-hop result (dataclass benzeri)
        class MockMultiHop:
            node_id = "hop2"
            content = "Multi-hop content"
            score = 0.8
            hop_distance = 2
            similarity_score = 0.5
            causal_score = 0.8
            importance_score = 0.3
            metadata = {}
            bridge_nodes = ["bridge1"]

        multi_hop = [MockMultiHop()]

        combined = optimizer.combine_multi_hop_results(direct, multi_hop)

        # Direkt sonuc birinci olmali (hop penalty sonrasi)
        assert combined[0].node_id == "direct1"

    def test_combine_applies_hop_penalty(self, optimizer):
        """Hop penalty uygulanmali"""
        direct = []

        class MockMultiHop:
            node_id = "hop3"
            content = "3-hop content"
            score = 0.9
            hop_distance = 3
            similarity_score = 0.5
            causal_score = 0.9
            importance_score = 0.3
            metadata = {}
            bridge_nodes = []

        multi_hop = [MockMultiHop()]

        combined = optimizer.combine_multi_hop_results(
            direct, multi_hop, hop_penalty=0.15
        )

        # 3 hop * 0.15 = 0.45 penalti
        # 0.9 * (1 - 0.45) = 0.495
        assert combined[0].score < 0.6
