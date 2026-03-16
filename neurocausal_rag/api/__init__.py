"""
NeuroCausal RAG - REST API Module
FastAPI-based production API

v5.0 Features:
- RESTful endpoints for all operations
- API key authentication
- Rate limiting
- Health checks and metrics
- OpenAPI documentation

Usage:
    # Option 1: Run directly
    from neurocausal_rag.api import run_server
    run_server(port=8000)

    # Option 2: Create custom app
    from neurocausal_rag.api import create_app
    from neurocausal_rag import NeuroCausalRAG

    rag = NeuroCausalRAG()
    app = create_app(rag_instance=rag, api_keys=["my-key"])

    # Option 3: Use CLI
    # uvicorn neurocausal_rag.api.app:app --reload
"""

from .app import create_app, app, run_server, add_api_key, remove_api_key
from .routes import get_all_routers, set_rag_instance
from .models import (
    # Enums
    SearchMode,
    RelationType,
    PipelineMode,
    # Search
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    # Documents
    DocumentRequest,
    DocumentBatchRequest,
    DocumentResponse,
    LinkRequest,
    # Agent
    AgentRequest,
    AgentResponse,
    # Feedback
    FeedbackRequest,
    FeedbackResponse,
    # Discovery
    DiscoveryRequest,
    DiscoveryResponse,
    # Graph
    GraphStatsResponse,
    CausalChainRequest,
    CausalChainResponse,
    # System
    HealthResponse,
    MetricsResponse,
    ErrorResponse
)

__all__ = [
    # App
    "create_app",
    "app",
    "run_server",
    "add_api_key",
    "remove_api_key",
    # Routes
    "get_all_routers",
    "set_rag_instance",
    # Models - Enums
    "SearchMode",
    "RelationType",
    "PipelineMode",
    # Models - Search
    "SearchRequest",
    "SearchResponse",
    "SearchResultItem",
    # Models - Documents
    "DocumentRequest",
    "DocumentBatchRequest",
    "DocumentResponse",
    "LinkRequest",
    # Models - Agent
    "AgentRequest",
    "AgentResponse",
    # Models - Feedback
    "FeedbackRequest",
    "FeedbackResponse",
    # Models - Discovery
    "DiscoveryRequest",
    "DiscoveryResponse",
    # Models - Graph
    "GraphStatsResponse",
    "CausalChainRequest",
    "CausalChainResponse",
    # Models - System
    "HealthResponse",
    "MetricsResponse",
    "ErrorResponse",
]
