"""
NeuroCausal RAG - FastAPI Application
Production-ready REST API for NeuroCausal RAG system

Features:
- RESTful API endpoints
- API key authentication
- CORS support
- Request/response logging
- Error handling
- OpenAPI documentation

Usage:
    uvicorn neurocausal_rag.api.app:create_app --reload

    # Or with factory
    from neurocausal_rag.api import create_app
    app = create_app()

Yazar: Ertuğrul Akben
"""

from typing import Optional, Callable
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
import logging
import time
import os

from .routes import get_all_routers, set_rag_instance
from .models import ErrorResponse

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

API_TITLE = "NeuroCausal RAG API"
API_DESCRIPTION = """
## NeuroCausal RAG - Intelligent Information Retrieval System

A production-ready API for semantic + causal graph-based retrieval.

### Features
- **Hybrid Search**: Combines semantic similarity with causal graph structure
- **Agentic RAG**: Self-correcting multi-step reasoning with LangGraph
- **Causal Discovery**: Automatic relationship extraction
- **Feedback Learning**: Continuous improvement from user feedback

### Authentication
All endpoints require an API key passed via `X-API-Key` header.

### Rate Limits
- Standard: 100 requests/minute
- Premium: 1000 requests/minute
"""
from neurocausal_rag import __version__ as API_VERSION


# =============================================================================
# AUTHENTICATION
# =============================================================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# API keys storage (in production, use database or secret manager)
_valid_api_keys = set()


def add_api_key(key: str):
    """Add a valid API key."""
    _valid_api_keys.add(key)


def remove_api_key(key: str):
    """Remove an API key."""
    _valid_api_keys.discard(key)


async def verify_api_key(api_key: str = Depends(api_key_header)) -> str:
    """Verify API key dependency."""
    # Skip auth if no keys configured (development mode)
    if not _valid_api_keys:
        return "dev-mode"

    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key header."
        )

    if api_key not in _valid_api_keys:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key."
        )

    return api_key


# =============================================================================
# MIDDLEWARE
# =============================================================================

class RequestLoggingMiddleware:
    """Log all requests with timing."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.time()
        path = scope.get("path", "")
        method = scope.get("method", "")

        # Process request
        await self.app(scope, receive, send)

        # Log timing
        elapsed = (time.time() - start) * 1000
        logger.info(f"{method} {path} - {elapsed:.2f}ms")


# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("NeuroCausal RAG API starting up...")

    # Load API keys from environment
    env_keys = os.getenv("NEUROCAUSAL_API_KEYS", "")
    if env_keys:
        for key in env_keys.split(","):
            if key.strip():
                add_api_key(key.strip())
        logger.info(f"Loaded {len(_valid_api_keys)} API keys from environment")

    # Initialize RAG if not already done
    if not hasattr(app.state, 'rag') or app.state.rag is None:
        try:
            from .. import NeuroCausalRAG
            app.state.rag = NeuroCausalRAG()
            set_rag_instance(app.state.rag)
            logger.info("NeuroCausal RAG initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")

    yield

    # Shutdown
    logger.info("NeuroCausal RAG API shutting down...")

    # Cleanup
    if hasattr(app.state, 'rag') and app.state.rag:
        # Save state if needed
        pass


# =============================================================================
# APP FACTORY
# =============================================================================

def create_app(
    rag_instance=None,
    enable_auth: bool = True,
    allowed_origins: list = None,
    api_keys: list = None
) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        rag_instance: Pre-configured NeuroCausalRAG instance
        enable_auth: Enable API key authentication
        allowed_origins: CORS allowed origins (default: all)
        api_keys: List of valid API keys

    Returns:
        Configured FastAPI application

    Example:
        >>> from neurocausal_rag.api import create_app
        >>> from neurocausal_rag import NeuroCausalRAG
        >>>
        >>> rag = NeuroCausalRAG()
        >>> app = create_app(rag_instance=rag, api_keys=["my-secret-key"])
        >>>
        >>> # Run with: uvicorn module:app
    """
    # Create app
    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )

    # Store RAG instance
    if rag_instance:
        app.state.rag = rag_instance
        set_rag_instance(rag_instance)

    # Add API keys
    if api_keys:
        for key in api_keys:
            add_api_key(key)

    # ==========================================================================
    # MIDDLEWARE
    # ==========================================================================

    # CORS
    origins = allowed_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging
    app.add_middleware(RequestLoggingMiddleware)

    # ==========================================================================
    # ERROR HANDLERS
    # ==========================================================================

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "code": f"HTTP_{exc.status_code}",
                "path": str(request.url.path)
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled error: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "code": "INTERNAL_ERROR",
                "detail": str(exc) if os.getenv("DEBUG") else None
            }
        )

    # ==========================================================================
    # ROUTES
    # ==========================================================================

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """API root - returns basic info."""
        return {
            "name": API_TITLE,
            "version": API_VERSION,
            "status": "running",
            "docs": "/docs",
            "health": "/health"
        }

    # Register all routers
    for router in get_all_routers():
        # Add auth dependency if enabled
        if enable_auth:
            app.include_router(
                router,
                prefix="/api/v1",
                dependencies=[Depends(verify_api_key)]
            )
        else:
            app.include_router(router, prefix="/api/v1")

    return app


# =============================================================================
# STANDALONE APP
# =============================================================================

# Create default app for uvicorn
app = create_app(enable_auth=False)


# =============================================================================
# CLI RUNNER
# =============================================================================

def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1,
    **kwargs
):
    """
    Run the API server.

    Args:
        host: Bind host
        port: Bind port
        reload: Enable auto-reload
        workers: Number of workers

    Example:
        >>> from neurocausal_rag.api import run_server
        >>> run_server(port=8080, reload=True)
    """
    import uvicorn

    uvicorn.run(
        "neurocausal_rag.api.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        **kwargs
    )


if __name__ == "__main__":
    run_server(reload=True)
