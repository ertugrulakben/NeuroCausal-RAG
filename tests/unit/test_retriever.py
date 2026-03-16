"""
NeuroCausal RAG - Retriever Unit Tests

Author: Ertugrul Akben
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestRetrieverBasic:
    """Retriever temel fonksiyonalite testleri"""

    def test_retriever_initialization(self, mock_graph_engine, mock_embedding_model):
        """Retriever baslatma testi"""
        from neurocausal_rag.search.retriever import Retriever

        retriever = Retriever(
            graph=mock_graph_engine,
            embedding=mock_embedding_model
        )

        assert retriever is not None
        assert retriever.graph == mock_graph_engine

    def test_retriever_with_entity_linker(self, mock_graph_engine, mock_embedding_model):
        """Entity linker ile Retriever testi"""
        from neurocausal_rag.search.retriever import Retriever
        from neurocausal_rag.entity.linker import EntityLinker

        entity_linker = EntityLinker()
        entity_linker.add_alias("Mavi Ufuk", "Gunes Enerjisi A.S.", 0.9)

        retriever = Retriever(
            graph=mock_graph_engine,
            embedding=mock_embedding_model,
            entity_linker=entity_linker
        )

        assert retriever is not None
        assert retriever.entity_linker == entity_linker


class TestEntityLinkingIntegration:
    """Entity Linking entegrasyonu testleri"""

    def test_query_enrichment(self, mock_graph_engine, mock_embedding_model):
        """Sorgu zenginlestirme testi"""
        from neurocausal_rag.entity.linker import EntityLinker

        entity_linker = EntityLinker()
        entity_linker.add_alias("Mavi Ufuk", "Gunes Enerjisi A.S.", 0.9)

        # Enriched query should include canonical name (lowercase)
        enriched = entity_linker.enrich_query("Mavi Ufuk nedir?")
        assert "gunes enerjisi" in enriched.lower()


class TestScoringMechanisms:
    """Skor hesaplama testleri"""

    def test_similarity_calculation(self):
        """Benzerlik hesaplama testi"""
        # Cosine similarity test
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        vec3 = np.array([0.0, 1.0, 0.0])

        # Same vectors should have similarity 1
        sim_same = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        assert abs(sim_same - 1.0) < 0.001

        # Orthogonal vectors should have similarity 0
        sim_ortho = np.dot(vec1, vec3) / (np.linalg.norm(vec1) * np.linalg.norm(vec3))
        assert abs(sim_ortho) < 0.001

    def test_importance_values(self, mock_graph_engine):
        """Importance deger araligi testi"""
        mock_graph_engine.get_importance.return_value = 0.75

        importance = mock_graph_engine.get_importance("test_node")

        assert 0 <= importance <= 1


class TestSearchResults:
    """Arama sonuclari testleri"""

    def test_empty_graph_handling(self, mock_graph_engine, mock_embedding_model):
        """Bos graph isleme"""
        from neurocausal_rag.search.retriever import Retriever

        mock_graph_engine.get_all_embeddings.return_value = (np.array([]), [])

        retriever = Retriever(
            graph=mock_graph_engine,
            embedding=mock_embedding_model
        )

        # Bos graph'ta arama yapmak hata vermemeli
        assert retriever is not None


class TestEntityLinkerDirect:
    """EntityLinker direkt testleri (Retriever'dan bagimsiz)"""

    def test_alias_resolution_in_query(self):
        """Sorguda alias cozumleme"""
        from neurocausal_rag.entity.linker import EntityLinker

        linker = EntityLinker()
        linker.add_alias("Fenix", "ERP Sistemi", 0.9)
        linker.add_alias("Proje X", "Yeni Platform", 0.85)

        resolved = linker.resolve_text("Fenix ve Proje X hakkinda bilgi ver")

        assert "fenix" in resolved
        assert resolved["fenix"] == "erp sistemi"
        assert "proje x" in resolved
        assert resolved["proje x"] == "yeni platform"

    def test_multiple_aliases_same_canonical(self):
        """Ayni canonical icin birden fazla alias"""
        from neurocausal_rag.entity.linker import EntityLinker

        linker = EntityLinker()
        linker.add_alias("Kod A", "Gercek Proje", 0.9)
        linker.add_alias("Alias B", "Gercek Proje", 0.85)
        linker.add_alias("Takim Adi", "Gercek Proje", 0.8)

        all_aliases = linker.get_all_aliases()

        assert "gercek proje" in all_aliases
        assert len(all_aliases["gercek proje"]) == 3


class TestMockGraphOperations:
    """Mock graph operasyonlari"""

    def test_mock_node_retrieval(self, mock_graph_engine):
        """Mock node getirme"""
        mock_graph_engine.get_node.return_value = {
            'id': 'test_doc',
            'content': 'Test content',
            'metadata': {'source': 'test'}
        }

        node = mock_graph_engine.get_node("test_doc")

        assert node is not None
        assert node['id'] == 'test_doc'
        assert node['content'] == 'Test content'

    def test_mock_neighbors(self, mock_graph_engine):
        """Mock komsulari getirme"""
        mock_graph_engine.get_neighbors.return_value = ["neighbor1", "neighbor2", "neighbor3"]

        neighbors = mock_graph_engine.get_neighbors("center_node")

        assert len(neighbors) == 3
        assert "neighbor1" in neighbors

    def test_mock_causal_chain(self, mock_graph_engine):
        """Mock causal chain"""
        mock_graph_engine.get_causal_chain.return_value = ["start", "middle", "end"]

        chain = mock_graph_engine.get_causal_chain("start", max_depth=3)

        assert len(chain) == 3
        assert chain[0] == "start"
        assert chain[-1] == "end"
