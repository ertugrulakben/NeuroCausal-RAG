"""
NeuroCausal RAG - Pipeline Integration Tests
v5.1 - FAZ 1.4

Yazar: Ertugrul Akben
"""

import pytest
import numpy as np
from neurocausal_rag.core.graph import GraphEngine
from neurocausal_rag.reasoning.contradiction import ContradictionDetector
from neurocausal_rag.reasoning.temporal import TemporalEngine


def test_retrieval_pipeline_temporal_filtering():
    """Temporal filtering pipeline test"""
    engine = TemporalEngine()

    # Query: Event in 2021
    # Doc: Event in 2020
    # If Query causes Doc, 2021 cannot cause 2020.
    query = "Event A happened in 2021."
    doc = "Event B happened in 2020."

    is_valid = engine.validate_causal_order(query, doc)
    assert is_valid is False  # 2021 cannot cause 2020


def test_retrieval_pipeline_contradiction_warning():
    """Contradiction detection in pipeline"""
    detector = ContradictionDetector()

    query = "The company reported a record profit."
    doc = "The company is facing bankruptcy due to losses."

    score = detector.detect_conflict(query, doc)
    # Should detect conflict between profit and loss
    assert score > 0.5


def test_retrieval_pipeline_success():
    """Successful causal ordering"""
    engine = TemporalEngine()

    query = "Event A happened in 2019."
    doc = "Event B happened in 2020."

    is_valid = engine.validate_causal_order(query, doc)
    assert is_valid is True  # 2019 can cause 2020


def test_graph_integration_with_reasoning():
    """Graph + Reasoning integration"""
    from neurocausal_rag.core.edge import RelationType

    graph = GraphEngine()

    # Add nodes
    emb1 = np.random.randn(768).astype(np.float32)
    emb2 = np.random.randn(768).astype(np.float32)
    emb3 = np.random.randn(768).astype(np.float32)

    graph.add_node("cause", "Event in 2020", emb1, {"date": "2020-01-01"})
    graph.add_node("effect", "Result in 2021", emb2, {"date": "2021-01-01"})
    graph.add_node("contradiction", "Opposite result", emb3, {})

    graph.add_edge("cause", "effect", RelationType.CAUSES.value)

    # Verify structure
    assert graph.node_count == 3
    assert graph.edge_count == 1

    # Temporal validation
    engine = TemporalEngine()
    cause_node = graph.get_node("cause")
    effect_node = graph.get_node("effect")

    is_valid = engine.validate_causal_order(
        cause_node["content"],
        effect_node["content"]
    )
    assert is_valid is True


def test_full_reasoning_pipeline():
    """Full reasoning pipeline with all components"""
    engine = TemporalEngine()
    detector = ContradictionDetector()

    # Create a document chain
    docs = [
        {"id": "d1", "content": "Research started in 2019-01-01", "date": "2019-01-01"},
        {"id": "d2", "content": "Development began in 2019-06-01", "date": "2019-06-01"},
        {"id": "d3", "content": "Product launched in 2020-01-01", "date": "2020-01-01"},
        {"id": "d4", "content": "Sales increased after launch", "date": "2020-03-01"},
    ]

    # Validate temporal chain
    all_valid = True
    for i in range(len(docs) - 1):
        is_valid = engine.validate_causal_order(docs[i]["content"], docs[i+1]["content"])
        if not is_valid:
            all_valid = False
            break

    assert all_valid is True

    # Check for contradictions
    contradictions = []
    for i, d1 in enumerate(docs):
        for d2 in docs[i+1:]:
            score = detector.detect_conflict(d1["content"], d2["content"])
            if score > 0.5:
                contradictions.append((d1["id"], d2["id"]))

    # No contradictions in this chain
    assert len(contradictions) == 0
