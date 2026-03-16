"""
NeuroCausal RAG - Learning Module
Auto-discovery, feedback, and continuous learning

v5.0 Features:
- FunnelDiscovery: O(N²) → O(50) optimization
- AsyncFunnelDiscovery: Async NLI processing
- EntityExtractor: NER-based entity extraction
- DiscoveryPipeline: Unified pipeline with modes
- FeedbackLoop: Continuous learning from user feedback
"""

from .discovery import AutoCausalDiscovery
from .learner import LearningEngine
from .semantic_discovery import SemanticCausalDiscovery, enhanced_causal_discovery
from .deep_discovery import DeepCausalDiscovery, deep_causal_discovery
from .funnel_discovery import FunnelDiscovery, AsyncFunnelDiscovery, funnel_causal_discovery
from .entity_extraction import EntityExtractor, EntityRelationDiscovery, extract_entities_and_relations
from .pipeline import DiscoveryPipeline, PipelineMode, run_discovery_pipeline
from .feedback import (
    FeedbackLoop,
    FeedbackStore,
    FeedbackRecord,
    FeedbackType,
    WeightAdjuster,
    create_feedback_loop
)

__all__ = [
    # Original
    "AutoCausalDiscovery",
    "LearningEngine",
    # Semantic
    "SemanticCausalDiscovery",
    "enhanced_causal_discovery",
    # Deep/NLI
    "DeepCausalDiscovery",
    "deep_causal_discovery",
    # Funnel (v5.0)
    "FunnelDiscovery",
    "AsyncFunnelDiscovery",
    "funnel_causal_discovery",
    # Entity (v5.0)
    "EntityExtractor",
    "EntityRelationDiscovery",
    "extract_entities_and_relations",
    # Pipeline (v5.0)
    "DiscoveryPipeline",
    "PipelineMode",
    "run_discovery_pipeline",
    # Feedback (v5.0)
    "FeedbackLoop",
    "FeedbackStore",
    "FeedbackRecord",
    "FeedbackType",
    "WeightAdjuster",
    "create_feedback_loop",
]
