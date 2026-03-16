"""
NeuroCausal RAG - API Routes Unit Tests

Comprehensive tests for all FastAPI endpoints using FastAPI TestClient.
All RAG dependencies are mocked -- no real embeddings or models needed.

Tested endpoints:
    GET  /                         (root info)
    GET  /api/v1/health            (health check)
    POST /api/v1/search            (semantic + causal search)
    POST /api/v1/documents         (create document)
    POST /api/v1/documents/batch   (batch create)
    GET  /api/v1/documents/{id}    (get document)
    DELETE /api/v1/documents/{id}  (delete document)
    POST /api/v1/documents/links   (create causal link)
    GET  /api/v1/graph/stats       (graph statistics)
    POST /api/v1/graph/chain       (causal chain traversal)
    POST /api/v1/feedback          (submit feedback)
    GET  /api/v1/metrics           (system metrics)

Run:
    pytest tests/unit/test_api_routes.py -v
"""

import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace

from fastapi.testclient import TestClient

from neurocausal_rag.api.app import create_app, _valid_api_keys
from neurocausal_rag.api import routes as routes_module


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def _reset_global_state():
    """
    Reset all mutable global state in the routes module AND the
    api_keys set between every test so they stay isolated.
    """
    routes_module._rag_instance = None
    routes_module._start_time = None
    routes_module._request_count = 0
    routes_module._total_response_time = 0
    routes_module._error_count = 0
    _valid_api_keys.clear()
    yield
    routes_module._rag_instance = None
    routes_module._start_time = None
    routes_module._request_count = 0
    routes_module._total_response_time = 0
    routes_module._error_count = 0
    _valid_api_keys.clear()


@pytest.fixture
def mock_rag():
    """
    A fully-mocked NeuroCausalRAG object with sensible defaults
    for every attribute and method the route handlers touch.
    """
    rag = MagicMock()

    # config.version -- used by /health
    rag.config = MagicMock()
    rag.config.version = "6.1.0-test"

    # _retriever -- health check inspects it
    rag._retriever = MagicMock()

    # _llm -- health check inspects it
    rag._llm = MagicMock()

    # _graph -- document CRUD calls methods on it
    rag._graph = MagicMock()

    # get_stats -- used by /graph/stats, /health, /metrics, /discovery
    rag.get_stats.return_value = {
        "version": "6.1.0",
        "node_count": 42,
        "edge_count": 15,
        "avg_degree": 0.71,
        "is_connected": True,
        "num_relation_types": 3,
    }

    return rag


@pytest.fixture
def client(mock_rag):
    """Synchronous TestClient wired to an app with auth disabled."""
    app = create_app(rag_instance=mock_rag, enable_auth=False)
    return TestClient(app, raise_server_exceptions=False)


# =============================================================================
# HELPER
# =============================================================================

def _make_search_result(node_id="doc_1", content="Test content", score=0.95):
    """Return a namespace that quacks like a SearchResult."""
    return SimpleNamespace(
        node_id=node_id,
        content=content,
        score=score,
        similarity_score=score,
        causal_score=0.3,
        importance_score=0.2,
        causal_chain=["doc_1", "doc_2"],
        metadata={"source": "unit-test"},
    )


# =============================================================================
# 1. ROOT ENDPOINT  GET /
# =============================================================================

class TestRootEndpoint:

    def test_root_returns_api_info(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "NeuroCausal RAG API"
        assert "version" in body
        assert body["status"] == "running"
        assert body["docs"] == "/docs"
        assert body["health"] == "/health"


# =============================================================================
# 2. HEALTH ENDPOINT  GET /api/v1/health
# =============================================================================

class TestHealthEndpoint:

    def test_health_returns_200_when_all_components_healthy(self, client, mock_rag):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("healthy", "degraded")
        assert body["version"] == "6.1.0-test"
        assert "components" in body
        assert "timestamp" in body

    def test_health_includes_all_component_keys(self, client, mock_rag):
        resp = client.get("/api/v1/health")
        components = resp.json()["components"]
        assert "graph" in components
        assert "retriever" in components
        assert "llm" in components

    def test_health_degraded_when_graph_fails(self, client, mock_rag):
        mock_rag.get_stats.side_effect = RuntimeError("graph down")
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["components"]["graph"] == "unhealthy"
        assert body["status"] == "degraded"

    def test_health_reports_retriever_not_initialized(self, client, mock_rag):
        mock_rag._retriever = None
        resp = client.get("/api/v1/health")
        assert resp.json()["components"]["retriever"] == "not_initialized"

    def test_health_reports_llm_not_configured(self, client, mock_rag):
        mock_rag._llm = None
        resp = client.get("/api/v1/health")
        assert resp.json()["components"]["llm"] == "not_configured"


# =============================================================================
# 3. SEARCH ENDPOINT  POST /api/v1/search
# =============================================================================

class TestSearchEndpoint:

    def test_search_happy_path(self, client, mock_rag):
        mock_rag.search.return_value = [
            _make_search_result("doc_1", "Climate change overview", 0.95),
            _make_search_result("doc_2", "Greenhouse gases", 0.88),
        ]

        resp = client.post("/api/v1/search", json={
            "query": "What causes climate change?",
            "top_k": 5,
            "mode": "balanced",
        })

        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "What causes climate change?"
        assert body["total"] == 2
        assert len(body["results"]) == 2
        assert body["results"][0]["id"] == "doc_1"
        assert body["results"][0]["score"] == 0.95
        assert body["mode"] == "balanced"
        assert "weights" in body
        assert body["search_time_ms"] > 0

    def test_search_returns_empty_list_for_no_matches(self, client, mock_rag):
        mock_rag.search.return_value = []
        resp = client.post("/api/v1/search", json={"query": "nonexistent topic"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["results"] == []

    def test_search_missing_query_returns_422(self, client):
        resp = client.post("/api/v1/search", json={})
        assert resp.status_code == 422

    def test_search_empty_query_returns_422(self, client):
        resp = client.post("/api/v1/search", json={"query": ""})
        assert resp.status_code == 422

    def test_search_top_k_below_minimum_returns_422(self, client):
        resp = client.post("/api/v1/search", json={"query": "test", "top_k": 0})
        assert resp.status_code == 422

    def test_search_top_k_above_maximum_returns_422(self, client):
        resp = client.post("/api/v1/search", json={"query": "test", "top_k": 100})
        assert resp.status_code == 422

    def test_search_custom_mode_uses_provided_weights(self, client, mock_rag):
        mock_rag.search.return_value = [_make_search_result()]
        resp = client.post("/api/v1/search", json={
            "query": "test custom",
            "mode": "custom",
            "alpha": 0.8,
            "beta": 0.1,
            "gamma": 0.1,
        })
        assert resp.status_code == 200
        weights = resp.json()["weights"]
        assert weights["alpha"] == 0.8
        assert weights["beta"] == 0.1
        assert weights["gamma"] == 0.1

    def test_search_detective_mode_emphasizes_causal(self, client, mock_rag):
        mock_rag.search.return_value = [_make_search_result()]
        resp = client.post("/api/v1/search", json={
            "query": "trace root cause",
            "mode": "detective",
        })
        assert resp.status_code == 200
        assert resp.json()["weights"]["beta"] == 0.5

    def test_search_encyclopedia_mode_emphasizes_similarity(self, client, mock_rag):
        mock_rag.search.return_value = [_make_search_result()]
        resp = client.post("/api/v1/search", json={
            "query": "lookup definition",
            "mode": "encyclopedia",
        })
        assert resp.status_code == 200
        assert resp.json()["weights"]["alpha"] == 0.7

    def test_search_internal_error_returns_500(self, client, mock_rag):
        mock_rag.search.side_effect = RuntimeError("embedding service down")
        resp = client.post("/api/v1/search", json={"query": "will fail"})
        assert resp.status_code == 500

    def test_search_excludes_chains_when_disabled(self, client, mock_rag):
        mock_rag.search.return_value = [_make_search_result()]
        resp = client.post("/api/v1/search", json={
            "query": "test",
            "include_chains": False,
        })
        assert resp.status_code == 200
        assert resp.json()["results"][0]["causal_chain"] is None

    def test_search_includes_chains_by_default(self, client, mock_rag):
        mock_rag.search.return_value = [_make_search_result()]
        resp = client.post("/api/v1/search", json={"query": "test"})
        assert resp.status_code == 200
        assert resp.json()["results"][0]["causal_chain"] == ["doc_1", "doc_2"]


# =============================================================================
# 4. DOCUMENT ENDPOINTS  /api/v1/documents
# =============================================================================

class TestDocumentsEndpoint:

    # -- CREATE --

    def test_create_document_returns_201(self, client, mock_rag):
        mock_rag._graph.get_node.return_value = {"importance": 0.42}
        resp = client.post("/api/v1/documents", json={
            "id": "doc_new",
            "content": "The Earth orbits the Sun.",
            "metadata": {"source": "astronomy"},
        })
        assert resp.status_code == 201
        body = resp.json()
        assert body["id"] == "doc_new"
        assert body["content"] == "The Earth orbits the Sun."
        assert body["importance"] == 0.42
        mock_rag.add_document.assert_called_once_with(
            doc_id="doc_new",
            content="The Earth orbits the Sun.",
            metadata={"source": "astronomy"},
        )

    def test_create_document_missing_content_returns_422(self, client):
        resp = client.post("/api/v1/documents", json={"id": "doc_bad"})
        assert resp.status_code == 422

    def test_create_document_missing_id_returns_422(self, client):
        resp = client.post("/api/v1/documents", json={"content": "text"})
        assert resp.status_code == 422

    # -- GET --

    def test_get_document_found(self, client, mock_rag):
        mock_rag._graph.get_node.return_value = {
            "id": "doc_1",
            "content": "Existing doc",
            "importance": 0.75,
            "metadata": {"tag": "test"},
        }
        mock_rag._graph.get_neighbors.return_value = ["doc_2", "doc_3"]

        resp = client.get("/api/v1/documents/doc_1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == "doc_1"
        assert body["content"] == "Existing doc"
        assert body["importance"] == 0.75
        assert body["neighbors"] == ["doc_2", "doc_3"]

    def test_get_document_not_found_returns_404(self, client, mock_rag):
        mock_rag._graph.get_node.return_value = None
        resp = client.get("/api/v1/documents/nonexistent")
        assert resp.status_code == 404

    # -- DELETE --

    def test_delete_document_success_returns_204(self, client, mock_rag):
        mock_rag._graph.remove_node.return_value = True
        resp = client.delete("/api/v1/documents/doc_1")
        assert resp.status_code == 204
        mock_rag._graph.remove_node.assert_called_once_with("doc_1")
        mock_rag._retriever.rebuild_index.assert_called_once()

    def test_delete_document_not_found_returns_404(self, client, mock_rag):
        mock_rag._graph.remove_node.return_value = False
        resp = client.delete("/api/v1/documents/ghost")
        assert resp.status_code == 404

    # -- BATCH --

    def test_batch_create_documents(self, client, mock_rag):
        mock_rag.add_documents.return_value = 3
        resp = client.post("/api/v1/documents/batch", json={
            "documents": [
                {"id": "d1", "content": "First document"},
                {"id": "d2", "content": "Second document"},
                {"id": "d3", "content": "Third document"},
            ]
        })
        assert resp.status_code == 201
        body = resp.json()
        assert body["added"] == 3
        assert body["status"] == "success"

    def test_batch_create_empty_list_returns_422(self, client):
        resp = client.post("/api/v1/documents/batch", json={"documents": []})
        assert resp.status_code == 422

    # -- LINKS --

    def test_create_link_success(self, client, mock_rag):
        resp = client.post("/api/v1/documents/links", json={
            "source_id": "doc_a",
            "target_id": "doc_b",
            "relation_type": "causes",
            "strength": 0.9,
        })
        assert resp.status_code == 201
        body = resp.json()
        assert body["source"] == "doc_a"
        assert body["target"] == "doc_b"
        assert body["relation"] == "causes"
        assert body["status"] == "created"
        mock_rag.add_causal_link.assert_called_once_with(
            source_id="doc_a",
            target_id="doc_b",
            relation_type="causes",
            strength=0.9,
        )

    def test_create_link_all_relation_types_accepted(self, client, mock_rag):
        """All four RelationType enum values should be accepted."""
        for rtype in ("causes", "supports", "requires", "related"):
            resp = client.post("/api/v1/documents/links", json={
                "source_id": "a",
                "target_id": "b",
                "relation_type": rtype,
            })
            assert resp.status_code == 201, f"Failed for relation_type={rtype}"

    def test_create_link_invalid_relation_type_returns_422(self, client):
        resp = client.post("/api/v1/documents/links", json={
            "source_id": "doc_a",
            "target_id": "doc_b",
            "relation_type": "invalid_type",
        })
        assert resp.status_code == 422

    def test_create_link_strength_out_of_range_returns_422(self, client):
        resp = client.post("/api/v1/documents/links", json={
            "source_id": "doc_a",
            "target_id": "doc_b",
            "relation_type": "supports",
            "strength": 1.5,
        })
        assert resp.status_code == 422


# =============================================================================
# 5. GRAPH ENDPOINTS  /api/v1/graph
# =============================================================================

class TestGraphEndpoint:

    def test_graph_stats_happy_path(self, client, mock_rag):
        resp = client.get("/api/v1/graph/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_nodes"] == 42
        assert body["total_edges"] == 15
        assert body["is_connected"] is True
        assert body["num_relation_types"] == 3
        assert isinstance(body["avg_degree"], float)

    def test_graph_stats_calculates_avg_degree_when_missing(self, client, mock_rag):
        mock_rag.get_stats.return_value = {
            "node_count": 10,
            "edge_count": 20,
            "is_connected": False,
            "num_relation_types": 2,
        }
        resp = client.get("/api/v1/graph/stats")
        # avg_degree = 2 * 20 / 10 = 4.0
        assert resp.json()["avg_degree"] == 4.0

    def test_graph_stats_empty_graph(self, client, mock_rag):
        mock_rag.get_stats.return_value = {
            "node_count": 0,
            "edge_count": 0,
            "is_connected": False,
            "num_relation_types": 0,
        }
        resp = client.get("/api/v1/graph/stats")
        body = resp.json()
        assert body["total_nodes"] == 0
        assert body["total_edges"] == 0

    def test_graph_stats_error_returns_500(self, client, mock_rag):
        mock_rag.get_stats.side_effect = RuntimeError("engine crashed")
        resp = client.get("/api/v1/graph/stats")
        assert resp.status_code == 500

    def test_causal_chain_happy_path(self, client, mock_rag):
        mock_rag.get_causal_chain.return_value = ["n_a", "n_b", "n_c"]
        mock_rag._graph.get_node.side_effect = lambda nid: {
            "n_a": {"content": "First node content", "importance": 0.9},
            "n_b": {"content": "Second node content", "importance": 0.7},
            "n_c": {"content": "Third node content", "importance": 0.5},
        }.get(nid)

        resp = client.post("/api/v1/graph/chain", json={
            "node_id": "n_a",
            "max_depth": 3,
            "direction": "forward",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["start_node"] == "n_a"
        assert body["direction"] == "forward"
        assert body["chain"] == ["n_a", "n_b", "n_c"]
        assert body["length"] == 3
        assert len(body["chain_details"]) == 3

    def test_causal_chain_backward_direction(self, client, mock_rag):
        mock_rag.get_causal_chain.return_value = ["n_x"]
        mock_rag._graph.get_node.return_value = {"content": "node", "importance": 0.1}
        resp = client.post("/api/v1/graph/chain", json={
            "node_id": "n_x",
            "direction": "backward",
        })
        assert resp.status_code == 200
        assert resp.json()["direction"] == "backward"

    def test_causal_chain_invalid_direction_returns_422(self, client):
        resp = client.post("/api/v1/graph/chain", json={
            "node_id": "n_a",
            "direction": "sideways",
        })
        assert resp.status_code == 422

    def test_causal_chain_max_depth_validation(self, client):
        resp = client.post("/api/v1/graph/chain", json={
            "node_id": "n_a",
            "max_depth": 0,
            "direction": "forward",
        })
        assert resp.status_code == 422


# =============================================================================
# 6. FEEDBACK ENDPOINT  POST /api/v1/feedback
# =============================================================================

class TestFeedbackEndpoint:

    def test_submit_feedback_returns_201(self, client, mock_rag):
        resp = client.post("/api/v1/feedback", json={
            "query": "test query",
            "result_ids": ["doc_1", "doc_2"],
            "rating": 0.8,
            "comment": "Good results",
        })
        assert resp.status_code == 201
        body = resp.json()
        assert body["query"] == "test query"
        assert body["rating"] == 0.8
        assert body["status"] == "recorded"
        assert "id" in body
        assert "received_at" in body
        mock_rag.record_feedback.assert_called_once_with(
            query="test query",
            result_ids=["doc_1", "doc_2"],
            rating=0.8,
            comment="Good results",
        )

    def test_submit_feedback_without_comment(self, client, mock_rag):
        resp = client.post("/api/v1/feedback", json={
            "query": "minimal",
            "result_ids": ["doc_1"],
            "rating": 0.5,
        })
        assert resp.status_code == 201

    def test_submit_feedback_missing_rating_returns_422(self, client):
        resp = client.post("/api/v1/feedback", json={
            "query": "test",
            "result_ids": ["doc_1"],
        })
        assert resp.status_code == 422

    def test_submit_feedback_rating_above_1_returns_422(self, client):
        resp = client.post("/api/v1/feedback", json={
            "query": "test",
            "result_ids": ["doc_1"],
            "rating": 1.5,
        })
        assert resp.status_code == 422

    def test_submit_feedback_empty_result_ids_returns_422(self, client):
        resp = client.post("/api/v1/feedback", json={
            "query": "test",
            "result_ids": [],
            "rating": 0.5,
        })
        assert resp.status_code == 422


# =============================================================================
# 7. METRICS ENDPOINT  GET /api/v1/metrics
# =============================================================================

class TestMetricsEndpoint:

    def test_metrics_returns_all_fields(self, client, mock_rag):
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert "requests_total" in body
        assert "requests_per_minute" in body
        assert "avg_response_time_ms" in body
        assert "error_rate" in body
        assert "graph_stats" in body
        assert "memory_usage_mb" in body

    def test_metrics_graph_stats_embedded(self, client, mock_rag):
        resp = client.get("/api/v1/metrics")
        gs = resp.json()["graph_stats"]
        assert "total_nodes" in gs
        assert "total_edges" in gs


# =============================================================================
# 8. RAG NOT INITIALIZED  --  503 Service Unavailable
# =============================================================================

class TestRagNotInitialized:
    """
    When create_app is called without a rag_instance AND the lifespan
    init also fails, every guarded endpoint must return 503.
    """

    @pytest.fixture
    def no_rag_client(self):
        app = create_app(rag_instance=None, enable_auth=False)
        return TestClient(app, raise_server_exceptions=False)

    def test_search_returns_503(self, no_rag_client):
        resp = no_rag_client.post("/api/v1/search", json={"query": "test"})
        assert resp.status_code == 503

    def test_health_returns_503(self, no_rag_client):
        resp = no_rag_client.get("/api/v1/health")
        assert resp.status_code == 503

    def test_graph_stats_returns_503(self, no_rag_client):
        resp = no_rag_client.get("/api/v1/graph/stats")
        assert resp.status_code == 503

    def test_create_document_returns_503(self, no_rag_client):
        resp = no_rag_client.post("/api/v1/documents", json={
            "id": "x", "content": "y",
        })
        assert resp.status_code == 503

    def test_feedback_returns_503(self, no_rag_client):
        resp = no_rag_client.post("/api/v1/feedback", json={
            "query": "q", "result_ids": ["a"], "rating": 0.5,
        })
        assert resp.status_code == 503


# =============================================================================
# 9. AUTHENTICATION  --  API key header checks
# =============================================================================

class TestAuthentication:

    @pytest.fixture
    def auth_client(self, mock_rag):
        app = create_app(
            rag_instance=mock_rag,
            enable_auth=True,
            api_keys=["test-secret-key-123"],
        )
        return TestClient(app, raise_server_exceptions=False)

    def test_request_without_key_returns_401(self, auth_client):
        resp = auth_client.get("/api/v1/graph/stats")
        assert resp.status_code == 401

    def test_request_with_invalid_key_returns_403(self, auth_client):
        resp = auth_client.get(
            "/api/v1/graph/stats",
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 403

    def test_request_with_valid_key_succeeds(self, auth_client):
        resp = auth_client.get(
            "/api/v1/graph/stats",
            headers={"X-API-Key": "test-secret-key-123"},
        )
        assert resp.status_code == 200

    def test_root_does_not_require_auth(self, auth_client):
        """Root / is not behind /api/v1 prefix, so no auth."""
        resp = auth_client.get("/")
        assert resp.status_code == 200


# =============================================================================
# 10. INTERNAL METRICS TRACKING
# =============================================================================

class TestMetricsTracking:
    """Verify route handlers update the module-level counters."""

    def test_successful_search_increments_request_count(self, client, mock_rag):
        mock_rag.search.return_value = []
        assert routes_module._request_count == 0

        client.post("/api/v1/search", json={"query": "a"})
        assert routes_module._request_count == 1

        client.post("/api/v1/search", json={"query": "b"})
        assert routes_module._request_count == 2

    def test_failed_search_increments_error_count(self, client, mock_rag):
        mock_rag.search.side_effect = RuntimeError("boom")
        assert routes_module._error_count == 0

        client.post("/api/v1/search", json={"query": "fail"})
        assert routes_module._error_count == 1

    def test_total_response_time_accumulates(self, client, mock_rag):
        mock_rag.search.return_value = []
        assert routes_module._total_response_time == 0

        client.post("/api/v1/search", json={"query": "t"})
        assert routes_module._total_response_time > 0
