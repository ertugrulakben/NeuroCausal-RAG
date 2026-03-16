"""
NeuroCausal RAG - API Routes
FastAPI route definitions for all endpoints

Endpoints:
- /search - Semantic + causal search
- /documents - Document management
- /agent - Agentic RAG queries
- /feedback - User feedback
- /discovery - Causal discovery
- /graph - Graph operations
- /health - Health check

Author: Ertugrul Akben
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from datetime import datetime
import logging
import time
import uuid

from .models import (
    # Search
    SearchRequest, SearchResponse, SearchResultItem, SearchMode,
    # Documents
    DocumentRequest, DocumentBatchRequest, DocumentResponse, LinkRequest,
    # Agent
    AgentRequest, AgentResponse,
    # Feedback
    FeedbackRequest, FeedbackResponse,
    # Discovery
    DiscoveryRequest, DiscoveryResponse,
    # Graph
    GraphStatsResponse, CausalChainRequest, CausalChainResponse,
    # System
    HealthResponse, MetricsResponse, ErrorResponse
)

logger = logging.getLogger(__name__)

# =============================================================================
# ROUTER INSTANCES
# =============================================================================

search_router = APIRouter(prefix="/search", tags=["Search"])
documents_router = APIRouter(prefix="/documents", tags=["Documents"])
agent_router = APIRouter(prefix="/agent", tags=["Agent"])
feedback_router = APIRouter(prefix="/feedback", tags=["Feedback"])
discovery_router = APIRouter(prefix="/discovery", tags=["Discovery"])
graph_router = APIRouter(prefix="/graph", tags=["Graph"])
system_router = APIRouter(tags=["System"])


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================

# Global RAG instance (set by app.py)
_rag_instance = None
_start_time = None
_request_count = 0
_total_response_time = 0
_error_count = 0


def get_rag():
    """Dependency to get RAG instance"""
    if _rag_instance is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return _rag_instance


def set_rag_instance(rag):
    """Set the RAG instance (called from app.py)"""
    global _rag_instance, _start_time
    _rag_instance = rag
    _start_time = datetime.utcnow()


def track_request(response_time_ms: float, is_error: bool = False):
    """Track request metrics"""
    global _request_count, _total_response_time, _error_count
    _request_count += 1
    _total_response_time += response_time_ms
    if is_error:
        _error_count += 1


# =============================================================================
# SEARCH ENDPOINTS
# =============================================================================

@search_router.post("", response_model=SearchResponse)
async def search(request: SearchRequest, rag=Depends(get_rag)):
    """
    Perform semantic + causal search.

    Returns ranked documents with causal chains based on:
    - Semantic similarity (alpha weight)
    - Causal relevance (beta weight)
    - Importance score (gamma weight)
    """
    start = time.time()

    try:
        # Get weights based on mode or custom
        weights = _get_search_weights(request)

        # Execute search
        results = rag.search(request.query, top_k=request.top_k)

        # Format results
        result_items = []
        for r in results:
            item = SearchResultItem(
                id=r.node_id,
                content=r.content,
                score=r.score,
                similarity_score=getattr(r, 'similarity_score', r.score),
                causal_score=getattr(r, 'causal_score', 0.0),
                importance_score=getattr(r, 'importance_score', 0.0),
                causal_chain=r.causal_chain if request.include_chains else None,
                is_injected=r.metadata.get('injected_from') is not None if r.metadata else False,
                metadata=r.metadata
            )
            result_items.append(item)

        elapsed = (time.time() - start) * 1000
        track_request(elapsed)

        return SearchResponse(
            query=request.query,
            results=result_items,
            total=len(result_items),
            search_time_ms=elapsed,
            mode=request.mode.value,
            weights=weights
        )

    except Exception as e:
        elapsed = (time.time() - start) * 1000
        track_request(elapsed, is_error=True)
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_search_weights(request: SearchRequest) -> dict:
    """Get search weights based on mode"""
    mode_weights = {
        SearchMode.BALANCED: {"alpha": 0.5, "beta": 0.3, "gamma": 0.2},
        SearchMode.ENCYCLOPEDIA: {"alpha": 0.7, "beta": 0.1, "gamma": 0.2},
        SearchMode.DETECTIVE: {"alpha": 0.3, "beta": 0.5, "gamma": 0.2},
        SearchMode.HUB_FOCUSED: {"alpha": 0.3, "beta": 0.2, "gamma": 0.5},
    }

    if request.mode == SearchMode.CUSTOM:
        return {
            "alpha": request.alpha or 0.5,
            "beta": request.beta or 0.3,
            "gamma": request.gamma or 0.2
        }

    return mode_weights.get(request.mode, mode_weights[SearchMode.BALANCED])


# =============================================================================
# DOCUMENT ENDPOINTS
# =============================================================================

@documents_router.post("", response_model=DocumentResponse, status_code=201)
async def create_document(request: DocumentRequest, rag=Depends(get_rag)):
    """Add a new document to the knowledge base."""
    try:
        rag.add_document(
            doc_id=request.id,
            content=request.content,
            metadata=request.metadata
        )

        # Get the added node info
        node = rag._graph.get_node(request.id)

        return DocumentResponse(
            id=request.id,
            content=request.content,
            importance=node.get('importance', 0.0) if node else 0.0,
            metadata=request.metadata,
            neighbors=None
        )

    except Exception as e:
        logger.error(f"Create document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@documents_router.post("/batch", status_code=201)
async def create_documents_batch(request: DocumentBatchRequest, rag=Depends(get_rag)):
    """Add multiple documents at once."""
    try:
        documents = [
            {
                'id': doc.id,
                'content': doc.content,
                'metadata': doc.metadata
            }
            for doc in request.documents
        ]

        count = rag.add_documents(documents)

        return {"added": count, "status": "success"}

    except Exception as e:
        logger.error(f"Batch create error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@documents_router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str, rag=Depends(get_rag)):
    """Get a document by ID."""
    try:
        node = rag._graph.get_node(doc_id)

        if not node:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

        neighbors = rag._graph.get_neighbors(doc_id)

        return DocumentResponse(
            id=node['id'],
            content=node['content'],
            importance=node.get('importance', 0.0),
            metadata=node.get('metadata'),
            neighbors=neighbors
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@documents_router.delete("/{doc_id}", status_code=204)
async def delete_document(doc_id: str, rag=Depends(get_rag)):
    """Delete a document."""
    try:
        success = rag._graph.remove_node(doc_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
        rag._retriever.rebuild_index()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@documents_router.post("/links", status_code=201)
async def create_link(request: LinkRequest, rag=Depends(get_rag)):
    """Create a causal link between documents."""
    try:
        rag.add_causal_link(
            source_id=request.source_id,
            target_id=request.target_id,
            relation_type=request.relation_type.value,
            strength=request.strength
        )

        return {
            "source": request.source_id,
            "target": request.target_id,
            "relation": request.relation_type.value,
            "status": "created"
        }

    except Exception as e:
        logger.error(f"Create link error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# AGENT ENDPOINTS
# =============================================================================

@agent_router.post("/query", response_model=AgentResponse)
async def agent_query(request: AgentRequest, rag=Depends(get_rag)):
    """
    Run agentic RAG query with self-correction.

    The agent will:
    1. Analyze the query
    2. Search for relevant documents
    3. Explore causal chains
    4. Generate and verify answer
    5. Self-correct if confidence is low
    """
    start = time.time()

    try:
        # Import agent
        from ..agents import create_agent

        # Create agent with RAG components
        agent = create_agent(
            retriever=rag._retriever,
            graph_engine=rag._graph,
            llm_client=rag._llm if hasattr(rag, '_llm') else None,
            min_confidence=request.min_confidence,
            max_iterations=request.max_iterations
        )

        # Run agent
        result = agent.run(request.query, context=request.context)

        elapsed = (time.time() - start) * 1000
        track_request(elapsed)

        return AgentResponse(
            query=request.query,
            answer=result.get('answer'),
            confidence=result.get('confidence', 0.0),
            verified=result.get('verified', False),
            iterations=result.get('iterations', 0),
            corrections=result.get('corrections', 0),
            reasoning_chain=result.get('reasoning_chain', []),
            sources=result.get('search_results', []),
            execution_time_ms=elapsed
        )

    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Agent module not available. Install langgraph for full agent support."
        )
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        track_request(elapsed, is_error=True)
        logger.error(f"Agent query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# FEEDBACK ENDPOINTS
# =============================================================================

@feedback_router.post("", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(request: FeedbackRequest, rag=Depends(get_rag)):
    """
    Submit user feedback for learning.

    Feedback is used to:
    - Improve search relevance
    - Adjust causal link weights
    - Train the system over time
    """
    try:
        # Record feedback
        rag.record_feedback(
            query=request.query,
            result_ids=request.result_ids,
            rating=request.rating,
            comment=request.comment
        )

        feedback_id = str(uuid.uuid4())[:8]

        return FeedbackResponse(
            id=feedback_id,
            received_at=datetime.utcnow(),
            query=request.query,
            rating=request.rating,
            status="recorded"
        )

    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# DISCOVERY ENDPOINTS
# =============================================================================

@discovery_router.post("", response_model=DiscoveryResponse)
async def run_discovery(request: DiscoveryRequest, rag=Depends(get_rag)):
    """
    Run causal discovery pipeline.

    Discovers potential causal relationships using:
    - Semantic similarity filtering
    - NLI verification
    - Optional LLM confirmation
    """
    start = time.time()

    try:
        # Use learning engine if available
        relations = rag.discover_links(min_confidence=request.min_confidence)

        # Limit results
        relations = relations[:request.max_relations]

        elapsed = (time.time() - start) * 1000
        track_request(elapsed)

        stats = rag.get_stats()

        return DiscoveryResponse(
            relations=relations,
            stats={
                "node_count": stats.get("node_count", 0),
                "discovered_count": len(relations)
            },
            execution_time_ms=elapsed,
            mode=request.mode.value
        )

    except Exception as e:
        elapsed = (time.time() - start) * 1000
        track_request(elapsed, is_error=True)
        logger.error(f"Discovery error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# GRAPH ENDPOINTS
# =============================================================================

@graph_router.get("/stats", response_model=GraphStatsResponse)
async def get_graph_stats(rag=Depends(get_rag)):
    """Get graph statistics."""
    try:
        stats = rag.get_stats()

        return GraphStatsResponse(
            total_nodes=stats.get("node_count", 0),
            total_edges=stats.get("edge_count", 0),
            avg_degree=stats.get("avg_degree", 0.0) if "avg_degree" in stats else
                       (2 * stats.get("edge_count", 0) / max(1, stats.get("node_count", 1))),
            is_connected=stats.get("is_connected", False),
            num_relation_types=stats.get("num_relation_types", 0)
        )

    except Exception as e:
        logger.error(f"Graph stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@graph_router.post("/chain", response_model=CausalChainResponse)
async def get_causal_chain(request: CausalChainRequest, rag=Depends(get_rag)):
    """Get causal chain from a node."""
    try:
        chain = rag.get_causal_chain(
            doc_id=request.node_id,
            max_depth=request.max_depth
        )

        # Get chain details
        chain_details = []
        for node_id in chain:
            node = rag._graph.get_node(node_id)
            if node:
                chain_details.append({
                    "id": node_id,
                    "content": node['content'][:200],
                    "importance": node.get('importance', 0.0)
                })

        return CausalChainResponse(
            start_node=request.node_id,
            direction=request.direction,
            chain=chain,
            chain_details=chain_details,
            length=len(chain)
        )

    except Exception as e:
        logger.error(f"Causal chain error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SYSTEM ENDPOINTS
# =============================================================================

@system_router.get("/health", response_model=HealthResponse)
async def health_check(rag=Depends(get_rag)):
    """Health check endpoint."""
    try:
        # Check components
        components = {}

        # Graph engine
        try:
            stats = rag.get_stats()
            components["graph"] = "healthy"
        except Exception:
            components["graph"] = "unhealthy"

        # Retriever
        try:
            if rag._retriever:
                components["retriever"] = "healthy"
            else:
                components["retriever"] = "not_initialized"
        except Exception:
            components["retriever"] = "unhealthy"

        # LLM
        try:
            if hasattr(rag, '_llm') and rag._llm:
                components["llm"] = "healthy"
            else:
                components["llm"] = "not_configured"
        except Exception:
            components["llm"] = "unhealthy"

        # Calculate uptime
        uptime = (datetime.utcnow() - _start_time).total_seconds() if _start_time else 0

        # Overall status
        status = "healthy" if all(v == "healthy" for v in components.values() if v != "not_configured") else "degraded"

        return HealthResponse(
            status=status,
            version=rag.config.version if hasattr(rag, 'config') else "unknown",
            uptime_seconds=uptime,
            components=components,
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@system_router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(rag=Depends(get_rag)):
    """Get system metrics."""
    try:
        stats = rag.get_stats()

        # Calculate metrics
        avg_response_time = _total_response_time / max(1, _request_count)
        error_rate = _error_count / max(1, _request_count)
        uptime = (datetime.utcnow() - _start_time).total_seconds() if _start_time else 0
        requests_per_minute = (_request_count / max(1, uptime)) * 60

        # Memory usage (approximate)
        import sys
        memory_mb = sys.getsizeof(rag) / (1024 * 1024)

        return MetricsResponse(
            requests_total=_request_count,
            requests_per_minute=requests_per_minute,
            avg_response_time_ms=avg_response_time,
            error_rate=error_rate,
            cache_hit_rate=0.0,  # TODO: Implement cache tracking
            graph_stats=GraphStatsResponse(
                total_nodes=stats.get("node_count", 0),
                total_edges=stats.get("edge_count", 0),
                avg_degree=(2 * stats.get("edge_count", 0) / max(1, stats.get("node_count", 1))),
                is_connected=False,
                num_relation_types=0
            ),
            memory_usage_mb=memory_mb
        )

    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ROUTER AGGREGATION
# =============================================================================

def get_all_routers():
    """Get all API routers."""
    return [
        search_router,
        documents_router,
        agent_router,
        feedback_router,
        discovery_router,
        graph_router,
        system_router
    ]
