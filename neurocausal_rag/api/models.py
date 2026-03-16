"""
NeuroCausal RAG - API Models
Pydantic models for request/response validation

Author: Ertugrul Akben
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class SearchMode(str, Enum):
    """Search mode presets"""
    BALANCED = "balanced"
    ENCYCLOPEDIA = "encyclopedia"
    DETECTIVE = "detective"
    HUB_FOCUSED = "hub_focused"
    CUSTOM = "custom"


class RelationType(str, Enum):
    """Causal relation types"""
    CAUSES = "causes"
    SUPPORTS = "supports"
    REQUIRES = "requires"
    RELATED = "related"


class PipelineMode(str, Enum):
    """Discovery pipeline modes"""
    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"
    FULL = "full"


# =============================================================================
# SEARCH
# =============================================================================

class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results")
    mode: SearchMode = Field(default=SearchMode.BALANCED, description="Search mode")
    alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Similarity weight")
    beta: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Causal weight")
    gamma: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Importance weight")
    include_chains: bool = Field(default=True, description="Include causal chains")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What causes global warming?",
                "top_k": 5,
                "mode": "balanced",
                "include_chains": True
            }
        }


class SearchResultItem(BaseModel):
    """Single search result"""
    id: str
    content: str
    score: float
    similarity_score: float
    causal_score: float
    importance_score: float
    causal_chain: Optional[List[str]] = None
    is_injected: bool = False
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """Search response model"""
    query: str
    results: List[SearchResultItem]
    total: int
    search_time_ms: float
    mode: str
    weights: Dict[str, float]


# =============================================================================
# DOCUMENTS
# =============================================================================

class DocumentRequest(BaseModel):
    """Document creation request"""
    id: str = Field(..., min_length=1, max_length=256, description="Document ID")
    content: str = Field(..., min_length=1, max_length=50000, description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Document metadata")
    category: Optional[str] = Field(default=None, description="Document category")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "climate_001",
                "content": "Global warming is caused by greenhouse gases...",
                "metadata": {"source": "IPCC", "year": 2023},
                "category": "climate"
            }
        }


class DocumentBatchRequest(BaseModel):
    """Batch document creation"""
    documents: List[DocumentRequest] = Field(..., min_length=1, max_length=100)


class DocumentResponse(BaseModel):
    """Document response"""
    id: str
    content: str
    importance: float
    metadata: Optional[Dict[str, Any]] = None
    neighbors: Optional[List[str]] = None


class LinkRequest(BaseModel):
    """Causal link creation request"""
    source_id: str = Field(..., description="Source document ID")
    target_id: str = Field(..., description="Target document ID")
    relation_type: RelationType = Field(default=RelationType.CAUSES, description="Relation type")
    strength: float = Field(default=1.0, ge=0.0, le=1.0, description="Relation strength")


# =============================================================================
# AGENT
# =============================================================================

class AgentRequest(BaseModel):
    """Agent query request"""
    query: str = Field(..., min_length=1, max_length=1000, description="User question")
    max_iterations: int = Field(default=3, ge=1, le=5, description="Max correction iterations")
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum confidence")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "How does CO2 contribute to climate change?",
                "max_iterations": 3,
                "min_confidence": 0.7
            }
        }


class AgentResponse(BaseModel):
    """Agent query response"""
    query: str
    answer: Optional[str]
    confidence: float
    verified: bool
    iterations: int
    corrections: int
    reasoning_chain: List[str]
    sources: List[Dict[str, Any]]
    execution_time_ms: float


# =============================================================================
# FEEDBACK
# =============================================================================

class FeedbackRequest(BaseModel):
    """User feedback request"""
    query: str = Field(..., description="Original query")
    result_ids: List[str] = Field(..., min_length=1, description="Result document IDs")
    rating: float = Field(..., ge=0.0, le=1.0, description="Rating (0-1)")
    comment: Optional[str] = Field(default=None, max_length=1000, description="Optional comment")
    correct_answer: Optional[str] = Field(default=None, description="User-provided correct answer")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What causes global warming?",
                "result_ids": ["climate_001", "climate_002"],
                "rating": 0.8,
                "comment": "Good results but missing solar factors"
            }
        }


class FeedbackResponse(BaseModel):
    """Feedback response"""
    id: str
    received_at: datetime
    query: str
    rating: float
    status: str = "recorded"


# =============================================================================
# DISCOVERY
# =============================================================================

class DiscoveryRequest(BaseModel):
    """Causal discovery request"""
    mode: PipelineMode = Field(default=PipelineMode.BALANCED, description="Discovery mode")
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum confidence")
    max_relations: int = Field(default=100, ge=1, le=1000, description="Maximum relations to return")


class DiscoveryResponse(BaseModel):
    """Discovery response"""
    relations: List[Dict[str, Any]]
    stats: Dict[str, Any]
    execution_time_ms: float
    mode: str


# =============================================================================
# GRAPH
# =============================================================================

class GraphStatsResponse(BaseModel):
    """Graph statistics response"""
    total_nodes: int
    total_edges: int
    avg_degree: float
    is_connected: bool
    num_relation_types: int


class CausalChainRequest(BaseModel):
    """Causal chain request"""
    node_id: str = Field(..., description="Starting node ID")
    max_depth: int = Field(default=3, ge=1, le=10, description="Maximum chain depth")
    direction: str = Field(default="forward", pattern="^(forward|backward)$")


class CausalChainResponse(BaseModel):
    """Causal chain response"""
    start_node: str
    direction: str
    chain: List[str]
    chain_details: List[Dict[str, Any]]
    length: int


# =============================================================================
# SYSTEM
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    version: str
    uptime_seconds: float
    components: Dict[str, str]
    timestamp: datetime


class MetricsResponse(BaseModel):
    """System metrics response"""
    requests_total: int
    requests_per_minute: float
    avg_response_time_ms: float
    error_rate: float
    cache_hit_rate: float
    graph_stats: GraphStatsResponse
    memory_usage_mb: float


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    code: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
