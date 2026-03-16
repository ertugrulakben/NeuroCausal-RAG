"""
NeuroCausal RAG - Discovery Pipeline
Unified pipeline combining all discovery methods

Pipeline Modes:
    - FAST: Semantic discovery only (recommended for production)
    - BALANCED: Semantic + Funnel (good balance)
    - DEEP: Semantic + Funnel + Entity (most comprehensive)
    - FULL: All methods (for development/research)

Usage:
    >>> pipeline = DiscoveryPipeline(mode="balanced")
    >>> relations = pipeline.run(documents, embeddings)

Author: Ertugrul Akben
"""

import time
import numpy as np
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass
from enum import Enum
import logging

from .semantic_discovery import enhanced_causal_discovery
from .funnel_discovery import funnel_causal_discovery
from .entity_extraction import extract_entities_and_relations

logger = logging.getLogger(__name__)


class PipelineMode(str, Enum):
    """Discovery pipeline modes"""
    FAST = "fast"  # Semantic only - ~100ms
    BALANCED = "balanced"  # Semantic + Funnel - ~500ms
    DEEP = "deep"  # + Entity extraction - ~1s
    FULL = "full"  # All methods - ~2-5s


@dataclass
class PipelineResult:
    """Pipeline execution result"""
    relations: List[Dict]
    stats: Dict
    execution_time_ms: float
    mode: str


class DiscoveryPipeline:
    """
    Unified discovery pipeline that combines all methods.

    Features:
    - Mode-based execution (fast, balanced, deep, full)
    - Automatic deduplication
    - Confidence score fusion
    - Execution statistics

    Example:
        >>> pipeline = DiscoveryPipeline(mode="balanced")
        >>> result = pipeline.run(documents, embeddings)
        >>> print(f"Found {len(result.relations)} relations in {result.execution_time_ms:.0f}ms")
    """

    def __init__(
        self,
        mode: Literal["fast", "balanced", "deep", "full"] = "balanced",
        semantic_threshold: float = 0.5,
        funnel_top_k: int = 50,
        entity_min_frequency: int = 2,
        entity_domain: str = "climate"
    ):
        """
        Args:
            mode: Pipeline mode
            semantic_threshold: Minimum confidence for semantic discovery
            funnel_top_k: Top-K for funnel stage 1
            entity_min_frequency: Minimum entity frequency
            entity_domain: Domain for entity patterns
        """
        self.mode = PipelineMode(mode)
        self.semantic_threshold = semantic_threshold
        self.funnel_top_k = funnel_top_k
        self.entity_min_frequency = entity_min_frequency
        self.entity_domain = entity_domain

    def run(
        self,
        documents: List[Dict],
        embeddings: np.ndarray,
        llm_callback: Optional[callable] = None
    ) -> PipelineResult:
        """
        Run discovery pipeline.

        Args:
            documents: [{'id': str, 'content': str, 'category': str}, ...]
            embeddings: (n_docs, dim) embedding matrix
            llm_callback: Optional LLM function for Stage 3

        Returns:
            PipelineResult with relations and stats
        """
        start_time = time.time()
        all_relations = []
        stats = {
            "n_documents": len(documents),
            "mode": self.mode.value,
            "stages_run": []
        }

        logger.info(f"DiscoveryPipeline: mode={self.mode.value}, {len(documents)} documents")

        # ================================================================
        # STAGE: Semantic Discovery (always runs)
        # ================================================================
        stage_start = time.time()
        semantic_relations = enhanced_causal_discovery(
            documents=documents,
            embeddings=embeddings,
            similarity_threshold=self.semantic_threshold,
            min_confidence=self.semantic_threshold
        )
        stats["semantic"] = {
            "relations": len(semantic_relations),
            "time_ms": (time.time() - stage_start) * 1000
        }
        stats["stages_run"].append("semantic")
        all_relations.extend(semantic_relations)
        logger.info(f"  Semantic: {len(semantic_relations)} relations")

        # ================================================================
        # STAGE: Funnel Discovery (balanced, deep, full)
        # ================================================================
        if self.mode in [PipelineMode.BALANCED, PipelineMode.DEEP, PipelineMode.FULL]:
            stage_start = time.time()
            try:
                funnel_relations = funnel_causal_discovery(
                    documents=documents,
                    embeddings=embeddings,
                    top_k_semantic=self.funnel_top_k,
                    top_k_nli=20,
                    use_async=True
                )
                stats["funnel"] = {
                    "relations": len(funnel_relations),
                    "time_ms": (time.time() - stage_start) * 1000
                }
                stats["stages_run"].append("funnel")
                all_relations.extend(funnel_relations)
                logger.info(f"  Funnel: {len(funnel_relations)} relations")
            except Exception as e:
                logger.warning(f"  Funnel failed: {e}")
                stats["funnel"] = {"error": str(e)}

        # ================================================================
        # STAGE: Entity-based Discovery (deep, full)
        # ================================================================
        if self.mode in [PipelineMode.DEEP, PipelineMode.FULL]:
            stage_start = time.time()
            try:
                entities, entity_relations = extract_entities_and_relations(
                    documents=documents,
                    domain=self.entity_domain,
                    min_frequency=self.entity_min_frequency
                )

                # Convert entity relations to standard format
                entity_rels_formatted = []
                for er in entity_relations:
                    entity_rels_formatted.append({
                        'source': er.source_entity,
                        'target': er.target_entity,
                        'relation_type': er.relation_type,
                        'confidence': er.strength,
                        'method': 'entity_extraction',
                        'evidence': er.evidence
                    })

                stats["entity"] = {
                    "entities": len(entities),
                    "relations": len(entity_rels_formatted),
                    "time_ms": (time.time() - stage_start) * 1000
                }
                stats["stages_run"].append("entity")
                all_relations.extend(entity_rels_formatted)
                logger.info(f"  Entity: {len(entities)} entities, {len(entity_rels_formatted)} relations")
            except Exception as e:
                logger.warning(f"  Entity extraction failed: {e}")
                stats["entity"] = {"error": str(e)}

        # ================================================================
        # STAGE: Deep NLI (full mode only, with LLM callback)
        # ================================================================
        if self.mode == PipelineMode.FULL and llm_callback:
            stage_start = time.time()
            try:
                from .deep_discovery import deep_causal_discovery
                deep_relations = deep_causal_discovery(
                    documents=documents,
                    embeddings=embeddings,
                    use_nli=True,
                    max_pairs=500
                )
                stats["deep"] = {
                    "relations": len(deep_relations),
                    "time_ms": (time.time() - stage_start) * 1000
                }
                stats["stages_run"].append("deep")
                all_relations.extend(deep_relations)
                logger.info(f"  Deep: {len(deep_relations)} relations")
            except Exception as e:
                logger.warning(f"  Deep discovery failed: {e}")
                stats["deep"] = {"error": str(e)}

        # ================================================================
        # POST-PROCESSING: Deduplicate and fuse scores
        # ================================================================
        final_relations = self._deduplicate_and_fuse(all_relations)

        execution_time = (time.time() - start_time) * 1000
        stats["total_raw_relations"] = len(all_relations)
        stats["final_relations"] = len(final_relations)

        logger.info(f"Pipeline complete: {len(final_relations)} relations in {execution_time:.0f}ms")

        return PipelineResult(
            relations=final_relations,
            stats=stats,
            execution_time_ms=execution_time,
            mode=self.mode.value
        )

    def _deduplicate_and_fuse(
        self,
        relations: List[Dict]
    ) -> List[Dict]:
        """
        Deduplicate relations and fuse confidence scores.

        When same (source, target) pair found by multiple methods,
        combine their confidence scores.
        """
        pair_relations: Dict[tuple, List[Dict]] = {}

        for rel in relations:
            key = (rel['source'], rel['target'])

            if key not in pair_relations:
                pair_relations[key] = []
            pair_relations[key].append(rel)

        # Fuse scores
        final = []
        for (source, target), rels in pair_relations.items():
            if len(rels) == 1:
                final.append(rels[0])
                continue

            # Multiple methods found this relation - boost confidence
            confidences = [r['confidence'] for r in rels]
            methods = list(set(r.get('method', 'unknown') for r in rels))

            # Weighted average with diversity bonus
            avg_confidence = sum(confidences) / len(confidences)
            diversity_bonus = min(0.2, len(methods) * 0.05)
            fused_confidence = min(1.0, avg_confidence + diversity_bonus)

            # Combine evidence
            all_evidence = []
            for r in rels:
                if 'evidence' in r:
                    if isinstance(r['evidence'], list):
                        all_evidence.extend(r['evidence'][:2])
                    else:
                        all_evidence.append(str(r['evidence']))

            # Determine relation type (prefer 'causes' over others)
            relation_types = [r['relation_type'] for r in rels]
            if 'causes' in relation_types:
                rel_type = 'causes'
            elif 'supports' in relation_types:
                rel_type = 'supports'
            else:
                rel_type = relation_types[0]

            final.append({
                'source': source,
                'target': target,
                'relation_type': rel_type,
                'confidence': fused_confidence,
                'method': 'fused',
                'methods_used': methods,
                'evidence': all_evidence[:5],
                'fusion_count': len(rels)
            })

        # Sort by confidence
        final.sort(key=lambda x: x['confidence'], reverse=True)
        return final


def run_discovery_pipeline(
    documents: List[Dict],
    embeddings: np.ndarray,
    mode: str = "balanced"
) -> List[Dict]:
    """
    Run discovery pipeline - simple API.

    Args:
        documents: [{'id': str, 'content': str}, ...]
        embeddings: (n_docs, dim) embedding matrix
        mode: "fast", "balanced", "deep", or "full"

    Returns:
        List of discovered relations

    Example:
        >>> relations = run_discovery_pipeline(docs, embeddings, mode="balanced")
    """
    pipeline = DiscoveryPipeline(mode=mode)
    result = pipeline.run(documents, embeddings)
    return result.relations
