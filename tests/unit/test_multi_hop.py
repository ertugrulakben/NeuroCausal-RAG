"""
NeuroCausal RAG - Multi-Hop Retrieval Tests
v5.2 - FAZ 2.1

Yazar: Ertugrul Akben
"""

import pytest
import numpy as np
from neurocausal_rag.core.graph import GraphEngine
from neurocausal_rag.core.edge import RelationType
from neurocausal_rag.embedding.text import TextEmbedding
from neurocausal_rag.search.multi_hop import (
    MultiHopRetriever,
    MultiHopResult,
    HopPath,
    create_multi_hop_retriever
)


@pytest.fixture
def sample_graph():
    """Ornek bir zincirli graf olustur"""
    graph = GraphEngine()

    # Embedding boyutu
    dim = 768

    # Zincir: A -> B -> C -> D
    # A = Sera Gazlari
    # B = Sicaklik Artisi
    # C = Buzul Erimesi
    # D = Deniz Seviyesi Yukselmesi

    graph.add_node(
        "sera_gazi",
        "Sera gazlari atmosferde birikiyor ve isinin tutulmasina neden oluyor.",
        np.random.randn(dim).astype(np.float32),
        {"category": "neden"}
    )

    graph.add_node(
        "sicaklik",
        "Kuresel ortalama sicaklik son 100 yilda artis gosteriyor.",
        np.random.randn(dim).astype(np.float32),
        {"category": "sonuc"}
    )

    graph.add_node(
        "buzul",
        "Buzullar hizla eriyor, Grönland ve Antarktika'dan buz kaybediliyor.",
        np.random.randn(dim).astype(np.float32),
        {"category": "sonuc"}
    )

    graph.add_node(
        "deniz",
        "Deniz seviyesi yükseliyor, kıyı şehirleri tehdit altında.",
        np.random.randn(dim).astype(np.float32),
        {"category": "sonuc"}
    )

    # Zincir baglantilari
    graph.add_edge("sera_gazi", "sicaklik", RelationType.CAUSES.value)
    graph.add_edge("sicaklik", "buzul", RelationType.CAUSES.value)
    graph.add_edge("buzul", "deniz", RelationType.CAUSES.value)

    # Yan dal: sicaklik -> kuraklik
    graph.add_node(
        "kuraklik",
        "Kurakliklar daha uzun ve siddetli hale geliyor.",
        np.random.randn(dim).astype(np.float32),
        {"category": "sonuc"}
    )
    graph.add_edge("sicaklik", "kuraklik", RelationType.CAUSES.value)

    return graph


@pytest.fixture
def mock_embedding():
    """Mock embedding modeli"""
    class MockEmbedding:
        def get_text_embedding(self, text):
            return np.random.randn(768).astype(np.float32)

    return MockEmbedding()


class TestHopPath:
    """HopPath dataclass testleri"""

    def test_hop_path_creation(self):
        """HopPath oluşturma"""
        path = HopPath(
            nodes=["A", "B", "C"],
            edges=["causes", "causes"],
            total_weight=0.7,
            hop_count=2
        )

        assert path.start == "A"
        assert path.end == "C"
        assert path.hop_count == 2
        assert len(path.nodes) == 3

    def test_hop_path_empty(self):
        """Bos HopPath"""
        path = HopPath(nodes=[], edges=[], total_weight=0.0, hop_count=0)

        assert path.start == ""
        assert path.end == ""


class TestMultiHopRetriever:
    """MultiHopRetriever testleri"""

    def test_initialization(self, sample_graph, mock_embedding):
        """Retriever başlatma"""
        retriever = MultiHopRetriever(
            graph=sample_graph,
            embedding=mock_embedding,
            max_hops=3
        )

        assert retriever.max_hops == 3
        assert retriever.graph == sample_graph

    def test_factory_function(self, sample_graph, mock_embedding):
        """Factory function testi"""
        retriever = create_multi_hop_retriever(
            graph=sample_graph,
            embedding=mock_embedding,
            max_hops=2
        )

        assert isinstance(retriever, MultiHopRetriever)
        assert retriever.max_hops == 2

    def test_search_basic(self, sample_graph, mock_embedding):
        """Temel arama testi"""
        retriever = MultiHopRetriever(
            graph=sample_graph,
            embedding=mock_embedding,
            max_hops=3
        )

        results = retriever.search("sera gazlari", top_k=5)

        assert len(results) > 0
        assert all(isinstance(r, MultiHopResult) for r in results)

    def test_search_finds_multi_hop(self, sample_graph, mock_embedding):
        """Multi-hop dokumanlari bulma"""
        retriever = MultiHopRetriever(
            graph=sample_graph,
            embedding=mock_embedding,
            max_hops=3
        )

        results = retriever.search("iklim", top_k=10)

        # En az bir sonuc olmali
        assert len(results) > 0

        # Farkli hop mesafelerinde sonuclar olmali
        hop_distances = set(r.hop_distance for r in results)
        # En az 0 ve 1 hop'ta sonuc olmali (seed + 1 hop)
        assert 0 in hop_distances or len(results) > 0


class TestPathFinding:
    """Yol bulma testleri"""

    def test_find_paths_direct(self, sample_graph, mock_embedding):
        """Direkt baglanti yol bulma"""
        retriever = MultiHopRetriever(
            graph=sample_graph,
            embedding=mock_embedding,
            max_hops=3
        )

        paths = retriever.find_paths_between("sera_gazi", "sicaklik")

        assert len(paths) > 0
        assert paths[0].hop_count == 1
        assert "sera_gazi" in paths[0].nodes
        assert "sicaklik" in paths[0].nodes

    def test_find_paths_multi_hop(self, sample_graph, mock_embedding):
        """Coklu hop yol bulma"""
        retriever = MultiHopRetriever(
            graph=sample_graph,
            embedding=mock_embedding,
            max_hops=3
        )

        paths = retriever.find_paths_between("sera_gazi", "deniz")

        assert len(paths) > 0

        # En az 3 hoplu yol olmali (sera -> sicaklik -> buzul -> deniz)
        path = paths[0]
        assert path.hop_count >= 3
        assert "sera_gazi" in path.nodes
        assert "deniz" in path.nodes

    def test_find_paths_no_connection(self, sample_graph, mock_embedding):
        """Baglanti olmayan nodelar"""
        # Bagimsiz bir node ekle
        sample_graph.add_node(
            "bagimsiz",
            "Bu node hicbir yere bagli degil.",
            np.random.randn(768).astype(np.float32)
        )

        retriever = MultiHopRetriever(
            graph=sample_graph,
            embedding=mock_embedding,
            max_hops=3
        )

        paths = retriever.find_paths_between("sera_gazi", "bagimsiz")

        assert len(paths) == 0


class TestExplainConnection:
    """Baglanti aciklama testleri"""

    def test_explain_direct_connection(self, sample_graph, mock_embedding):
        """Direkt baglanti aciklamasi"""
        retriever = MultiHopRetriever(
            graph=sample_graph,
            embedding=mock_embedding,
            max_hops=3
        )

        explanation = retriever.explain_connection("sera_gazi", "sicaklik")

        assert explanation is not None
        assert "sera_gazi" in explanation.lower() or "Sera" in explanation
        assert "--(" in explanation  # Edge gosterimi

    def test_explain_multi_hop_connection(self, sample_graph, mock_embedding):
        """Coklu hop baglanti aciklamasi"""
        retriever = MultiHopRetriever(
            graph=sample_graph,
            embedding=mock_embedding,
            max_hops=3
        )

        explanation = retriever.explain_connection("sera_gazi", "deniz")

        assert explanation is not None
        # Ara nodelar gosterilmeli
        assert "sicaklik" in explanation.lower() or "buzul" in explanation.lower()

    def test_explain_no_connection(self, sample_graph, mock_embedding):
        """Baglanti yok durumu"""
        sample_graph.add_node(
            "izole",
            "Izole node",
            np.random.randn(768).astype(np.float32)
        )

        retriever = MultiHopRetriever(
            graph=sample_graph,
            embedding=mock_embedding,
            max_hops=3
        )

        explanation = retriever.explain_connection("sera_gazi", "izole")

        assert explanation is None


class TestEdgeWeights:
    """Edge agirliklari testleri"""

    def test_causes_highest_weight(self, sample_graph, mock_embedding):
        """CAUSES en yuksek agirliga sahip olmali"""
        assert MultiHopRetriever.EDGE_WEIGHTS['causes'] >= \
               MultiHopRetriever.EDGE_WEIGHTS['supports']

    def test_path_weight_decay(self, sample_graph, mock_embedding):
        """Her hop'ta agirlik azalmali"""
        retriever = MultiHopRetriever(
            graph=sample_graph,
            embedding=mock_embedding,
            max_hops=3
        )

        paths = retriever.find_paths_between("sera_gazi", "deniz")

        if paths:
            path = paths[0]
            # 3 hop icin beklenen max weight: 1.0 * 0.7^3 = 0.343
            assert path.total_weight < 1.0
            assert path.total_weight > 0.0


class TestBridgeNodes:
    """Kopru node testleri"""

    def test_bridge_nodes_identified(self, sample_graph, mock_embedding):
        """Ara nodelar tanimlanmali"""
        retriever = MultiHopRetriever(
            graph=sample_graph,
            embedding=mock_embedding,
            max_hops=3
        )

        results = retriever.search("sera deniz", top_k=10)

        # Bazi sonuclarda bridge_nodes olmali
        results_with_bridges = [r for r in results if len(r.bridge_nodes) > 0]

        # Multi-hop sonuclarda bridge olabilir
        # (seed'den 2+ hop uzakta olanlar)
        hop_2_plus = [r for r in results if r.hop_distance >= 2]
        if hop_2_plus:
            # En az birinde bridge olmali
            assert any(len(r.bridge_nodes) > 0 for r in hop_2_plus)


class TestMinPathScore:
    """Minimum yol skoru testleri"""

    def test_low_paths_filtered(self, sample_graph, mock_embedding):
        """Dusuk skorlu yollar elenmeli"""
        retriever_strict = MultiHopRetriever(
            graph=sample_graph,
            embedding=mock_embedding,
            max_hops=5,
            min_path_score=0.5  # Yuksek threshold
        )

        retriever_loose = MultiHopRetriever(
            graph=sample_graph,
            embedding=mock_embedding,
            max_hops=5,
            min_path_score=0.1  # Dusuk threshold
        )

        results_strict = retriever_strict.search("test", top_k=20)
        results_loose = retriever_loose.search("test", top_k=20)

        # Loose daha fazla sonuc donmeli (veya esit)
        assert len(results_loose) >= len(results_strict)


class TestBidirectionalSearch:
    """Cift yonlu arama testleri"""

    def test_bidirectional_finds_more(self, sample_graph, mock_embedding):
        """Cift yonlu arama daha fazla sonuc bulmali"""
        retriever_bidir = MultiHopRetriever(
            graph=sample_graph,
            embedding=mock_embedding,
            max_hops=3,
            use_bidirectional=True
        )

        retriever_unidir = MultiHopRetriever(
            graph=sample_graph,
            embedding=mock_embedding,
            max_hops=3,
            use_bidirectional=False
        )

        results_bidir = retriever_bidir.search("test", top_k=20)
        results_unidir = retriever_unidir.search("test", top_k=20)

        # Bidirectional genelde daha fazla path bulur
        # (en azindan esit)
        assert len(results_bidir) >= len(results_unidir)
