"""
NeuroCausal RAG - GraphEngine Unit Tests
v5.1 - FAZ 1.4

Yazar: Ertugrul Akben
"""

import pytest
import numpy as np
from neurocausal_rag.core.graph import GraphEngine
from neurocausal_rag.core.edge import RelationType


def test_graph_add_nodes_and_edges():
    """Node ve edge ekleme testi"""
    engine = GraphEngine()

    emb_a = np.random.randn(768).astype(np.float32)
    emb_b = np.random.randn(768).astype(np.float32)

    engine.add_node("A", "Content A", emb_a)
    engine.add_node("B", "Content B", emb_b)
    engine.add_edge("A", "B", RelationType.CAUSES.value)

    assert engine.graph.has_node("A")
    assert engine.graph.has_node("B")
    assert engine.graph.has_edge("A", "B")


def test_causal_chain_prioritization():
    """Causal chain prioritization testi"""
    engine = GraphEngine()

    emb_start = np.random.randn(768).astype(np.float32)
    emb_weak = np.random.randn(768).astype(np.float32)
    emb_strong = np.random.randn(768).astype(np.float32)

    engine.add_node("Start", "Start node", emb_start)
    engine.add_node("WeakPath", "Weak path node", emb_weak)
    engine.add_node("StrongPath", "Strong path node", emb_strong)

    # Start -> WeakPath (RELATED, weight 0.5)
    engine.add_edge("Start", "WeakPath", RelationType.RELATED.value)

    # Start -> StrongPath (CAUSES, weight 1.0)
    engine.add_edge("Start", "StrongPath", RelationType.CAUSES.value)

    chain = engine.get_causal_chain("Start", max_depth=1)

    # Should choose StrongPath (CAUSES has higher weight)
    assert "Start" in chain
    assert "StrongPath" in chain


class TestGraphEngineCore:
    """GraphEngine core functionality tests"""

    def test_node_count(self):
        """Node sayisi testi"""
        engine = GraphEngine()
        assert engine.node_count == 0

        emb = np.random.randn(768).astype(np.float32)
        engine.add_node("node1", "Content 1", emb)

        assert engine.node_count == 1

    def test_edge_count(self):
        """Edge sayisi testi"""
        engine = GraphEngine()

        emb1 = np.random.randn(768).astype(np.float32)
        emb2 = np.random.randn(768).astype(np.float32)

        engine.add_node("n1", "Node 1", emb1)
        engine.add_node("n2", "Node 2", emb2)

        assert engine.edge_count == 0

        engine.add_edge("n1", "n2", RelationType.CAUSES.value)

        assert engine.edge_count == 1

    def test_get_node(self):
        """Node getirme testi"""
        engine = GraphEngine()

        emb = np.random.randn(768).astype(np.float32)
        engine.add_node("test_node", "Test content", emb, {"key": "value"})

        node = engine.get_node("test_node")

        assert node is not None
        assert node["id"] == "test_node"
        assert node["content"] == "Test content"
        assert node["metadata"]["key"] == "value"

    def test_get_nonexistent_node(self):
        """Var olmayan node testi"""
        engine = GraphEngine()

        node = engine.get_node("nonexistent")
        assert node is None

    def test_get_neighbors(self):
        """Komsu node'lari getir testi"""
        engine = GraphEngine()

        emb1 = np.random.randn(768).astype(np.float32)
        emb2 = np.random.randn(768).astype(np.float32)
        emb3 = np.random.randn(768).astype(np.float32)

        engine.add_node("center", "Center node", emb1)
        engine.add_node("neighbor1", "Neighbor 1", emb2)
        engine.add_node("neighbor2", "Neighbor 2", emb3)

        engine.add_edge("center", "neighbor1", RelationType.CAUSES.value)
        engine.add_edge("center", "neighbor2", RelationType.SUPPORTS.value)

        neighbors = engine.get_neighbors("center")

        assert len(neighbors) == 2
        assert "neighbor1" in neighbors
        assert "neighbor2" in neighbors

    def test_get_predecessors(self):
        """Predecessor node'lari getir testi"""
        engine = GraphEngine()

        emb1 = np.random.randn(768).astype(np.float32)
        emb2 = np.random.randn(768).astype(np.float32)

        engine.add_node("cause", "Cause node", emb1)
        engine.add_node("effect", "Effect node", emb2)

        engine.add_edge("cause", "effect", RelationType.CAUSES.value)

        predecessors = engine.get_predecessors("effect")

        assert "cause" in predecessors

    def test_get_importance(self):
        """PageRank importance testi"""
        engine = GraphEngine()

        emb = np.random.randn(768).astype(np.float32)
        engine.add_node("solo_node", "Solo node", emb)

        importance = engine.get_importance("solo_node")

        assert importance >= 0

    def test_get_all_embeddings(self):
        """Tum embedding'leri getir testi"""
        engine = GraphEngine()

        for i in range(3):
            emb = np.random.randn(768).astype(np.float32)
            engine.add_node(f"node_{i}", f"Content {i}", emb)

        embeddings, ids = engine.get_all_embeddings()

        assert len(ids) == 3
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == 768


class TestCausalChain:
    """Causal chain tests"""

    def test_simple_chain(self):
        """Basit causal chain"""
        engine = GraphEngine()

        embs = [np.random.randn(768).astype(np.float32) for _ in range(3)]

        engine.add_node("A", "Node A", embs[0])
        engine.add_node("B", "Node B", embs[1])
        engine.add_node("C", "Node C", embs[2])

        engine.add_edge("A", "B", RelationType.CAUSES.value)
        engine.add_edge("B", "C", RelationType.CAUSES.value)

        chain = engine.get_causal_chain("A", max_depth=3)

        assert "A" in chain
        assert "B" in chain

    def test_backward_chain(self):
        """Geriye dogru causal chain"""
        engine = GraphEngine()

        embs = [np.random.randn(768).astype(np.float32) for _ in range(3)]

        engine.add_node("A", "Node A", embs[0])
        engine.add_node("B", "Node B", embs[1])
        engine.add_node("C", "Node C", embs[2])

        engine.add_edge("A", "B", RelationType.CAUSES.value)
        engine.add_edge("B", "C", RelationType.CAUSES.value)

        chain = engine.get_causal_chain("C", max_depth=3, direction='backward')

        assert "C" in chain

    def test_find_causal_path(self):
        """Causal path bulma testi"""
        engine = GraphEngine()

        embs = [np.random.randn(768).astype(np.float32) for _ in range(3)]

        engine.add_node("start", "Start", embs[0])
        engine.add_node("middle", "Middle", embs[1])
        engine.add_node("end", "End", embs[2])

        engine.add_edge("start", "middle", RelationType.CAUSES.value)
        engine.add_edge("middle", "end", RelationType.CAUSES.value)

        path, score = engine.find_causal_path("start", "end")

        assert len(path) == 3
        assert path[0] == "start"
        assert path[-1] == "end"


class TestGraphStats:
    """Graph statistics tests"""

    def test_get_stats(self):
        """Graph istatistikleri testi"""
        engine = GraphEngine()

        emb1 = np.random.randn(768).astype(np.float32)
        emb2 = np.random.randn(768).astype(np.float32)

        engine.add_node("n1", "Node 1", emb1)
        engine.add_node("n2", "Node 2", emb2)
        engine.add_edge("n1", "n2", RelationType.CAUSES.value)

        stats = engine.get_stats()

        assert stats["total_nodes"] == 2
        assert stats["total_edges"] == 1


class TestInvalidOperations:
    """Invalid operation tests"""

    def test_add_edge_missing_nodes(self):
        """Olmayan node'lar arasinda edge ekleme"""
        engine = GraphEngine()

        with pytest.raises(ValueError):
            engine.add_edge("nonexistent1", "nonexistent2", RelationType.CAUSES.value)

    def test_empty_chain(self):
        """Bos graph'ta causal chain"""
        engine = GraphEngine()

        chain = engine.get_causal_chain("nonexistent")

        assert chain == []
