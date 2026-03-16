"""
NeuroCausal RAG - Deep Causal Discovery
NLI (Natural Language Inference) based deep causal discovery

This module uses transformer models instead of simple regex patterns
to discover semantic causal relationships.

Core idea:
- "Greenhouse gases accumulate in atmosphere" -> "Earth warms"
- If ENTAILMENT exists between two sentences, causality may exist

Author: Ertugrul Akben
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CausalPair:
    """Causal pair"""
    source_id: str
    target_id: str
    source_text: str
    target_text: str
    entailment_score: float
    causal_score: float
    evidence: str


class DeepCausalDiscovery:
    """
    NLI (Natural Language Inference) based deep causal discovery.

    How it works:
    1. Test the "X causes Y" hypothesis for each document pair
    2. Compute entailment score using NLI model
    3. High entailment = potential causality

    NOTE: This method is much more powerful than regex because:
    - It uses logical inference, not semantic similarity
    - It can capture logical connections even without word overlap
      between "greenhouse gas" and "warming"
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        device: str = "cpu",
        threshold: float = 0.6
    ):
        """
        Args:
            model_name: NLI model name (HuggingFace)
            device: "cpu" or "cuda"
            threshold: Entailment threshold
        """
        self.model_name = model_name
        self.device = device
        self.threshold = threshold
        self._model = None
        self._use_simple = False

    def _load_model(self):
        """Lazy load the model"""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info(f"DeepCausalDiscovery: {self.model_name} loaded")
        except ImportError:
            logger.warning("sentence-transformers not installed, using simple mode")
            self._use_simple = True
        except Exception as e:
            logger.warning(f"Model load failed: {e}, using simple mode")
            self._use_simple = True

    def discover(
        self,
        documents: List[Dict],
        max_pairs: int = 500
    ) -> List[Dict]:
        """
        Discover causal relationships between documents.

        Args:
            documents: [{'id': str, 'content': str}, ...]
            max_pairs: Maximum number of pairs (for performance)

        Returns:
            Discovered causal relations
        """
        self._load_model()

        if self._use_simple:
            return self._simple_discovery(documents)

        n = len(documents)
        pairs = []

        # Generate all pairs (limited)
        all_pairs = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    all_pairs.append((i, j))

        # Random sampling
        if len(all_pairs) > max_pairs:
            import random
            random.shuffle(all_pairs)
            all_pairs = all_pairs[:max_pairs]

        logger.info(f"DeepCausalDiscovery: analyzing {len(all_pairs)} pairs...")

        # Prepare for batch processing
        batch_size = 32
        results = []

        for batch_start in range(0, len(all_pairs), batch_size):
            batch = all_pairs[batch_start:batch_start + batch_size]

            # NLI format pairs
            nli_pairs = []
            for i, j in batch:
                # Premise: source content (summary)
                premise = documents[i]['content'][:200]
                # Hypothesis: "Therefore {target} occurs"
                target_text = documents[j]['content'][:100]
                hypothesis = f"Bu nedenle {target_text}"
                nli_pairs.append((premise, hypothesis))

            try:
                # NLI skorları (entailment, neutral, contradiction)
                scores = self._model.predict(nli_pairs)

                for idx, (i, j) in enumerate(batch):
                    # CrossEncoder returns logits [contradiction, entailment, neutral]
                    if isinstance(scores[idx], (list, np.ndarray)):
                        entailment = float(scores[idx][1])  # entailment score
                    else:
                        entailment = float(scores[idx])

                    if entailment > self.threshold:
                        results.append({
                            'source': documents[i]['id'],
                            'target': documents[j]['id'],
                            'relation_type': 'causes',
                            'confidence': entailment,
                            'method': 'nli_deep',
                            'evidence': f"NLI Entailment: {entailment:.3f}"
                        })
            except Exception as e:
                logger.error(f"NLI batch error: {e}")
                continue

        logger.info(f"DeepCausalDiscovery: {len(results)} relations found")
        return sorted(results, key=lambda x: x['confidence'], reverse=True)

    def _simple_discovery(self, documents: List[Dict]) -> List[Dict]:
        """
        Simple discovery mode (when model is unavailable).
        Uses embedding similarity + keyword matching.
        """
        results = []
        n = len(documents)

        # Causality keywords
        cause_words = {'neden', 'sebep', 'kaynak', 'etken', 'tetikler',
                       'cause', 'source', 'trigger', 'leads'}
        effect_words = {'sonuç', 'etki', 'netice', 'oluşur', 'meydana',
                        'effect', 'result', 'outcome', 'leads to'}

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                text_i = documents[i]['content'].lower()
                text_j = documents[j]['content'].lower()

                # if i contains cause words and j contains effect words
                has_cause = any(w in text_i for w in cause_words)
                has_effect = any(w in text_j for w in effect_words)

                if has_cause and has_effect:
                    # Simple score
                    score = 0.6 + 0.1 * (sum(1 for w in cause_words if w in text_i) +
                                         sum(1 for w in effect_words if w in text_j))
                    score = min(0.9, score)

                    results.append({
                        'source': documents[i]['id'],
                        'target': documents[j]['id'],
                        'relation_type': 'causes',
                        'confidence': score,
                        'method': 'keyword_simple',
                        'evidence': f"Cause keywords in source, effect keywords in target"
                    })

        return sorted(results, key=lambda x: x['confidence'], reverse=True)[:100]


class CausalStrengthEstimator:
    """
    Embedding-based causal strength estimation.

    Estimates the causal relationship strength between two documents.
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self._projection = None

    def estimate_strength(
        self,
        source_emb: np.ndarray,
        target_emb: np.ndarray
    ) -> float:
        """
        Causal strength estimation between two embeddings.

        Method:
        1. Cosine similarity (base similarity)
        2. Asymmetry score (direction info)
        3. Projection score (causal space projection)
        """
        # 1. Cosine similarity
        cos_sim = np.dot(source_emb, target_emb) / (
            np.linalg.norm(source_emb) * np.linalg.norm(target_emb) + 1e-8
        )

        # 2. Asymmetry: if source -> target is strong, causality may exist
        # Direction of the vector difference
        diff = target_emb - source_emb
        diff_norm = np.linalg.norm(diff)

        # Projection of source in the diff direction
        if diff_norm > 0:
            projection = np.dot(source_emb, diff) / diff_norm
            asymmetry = max(0, projection / (np.linalg.norm(source_emb) + 1e-8))
        else:
            asymmetry = 0

        # 3. Final score
        strength = 0.5 * cos_sim + 0.5 * asymmetry
        return float(np.clip(strength, 0, 1))


def deep_causal_discovery(
    documents: List[Dict],
    embeddings: Optional[np.ndarray] = None,
    use_nli: bool = True,
    max_pairs: int = 500
) -> List[Dict]:
    """
    Deep causal discovery - single function API.

    Args:
        documents: [{'id': str, 'content': str}, ...]
        embeddings: Optional embedding matrix
        use_nli: Use NLI model?
        max_pairs: Maximum number of pairs

    Returns:
        Discovered causal relations
    """
    all_relations = []

    # 1. NLI-based discovery (varsa)
    if use_nli:
        try:
            nli_discovery = DeepCausalDiscovery()
            nli_relations = nli_discovery.discover(documents, max_pairs)
            all_relations.extend(nli_relations)
        except Exception as e:
            logger.warning(f"NLI discovery skipped: {e}")

    # 2. Embedding-based strength estimation (varsa)
    if embeddings is not None:
        estimator = CausalStrengthEstimator()
        n = len(documents)

        for i in range(min(n, 50)):  # For the first 50 documents
            for j in range(min(n, 50)):
                if i == j:
                    continue

                strength = estimator.estimate_strength(embeddings[i], embeddings[j])
                if strength > 0.6:
                    all_relations.append({
                        'source': documents[i]['id'],
                        'target': documents[j]['id'],
                        'relation_type': 'supports',
                        'confidence': strength,
                        'method': 'embedding_strength',
                        'evidence': f"Embedding strength: {strength:.3f}"
                    })

    # Deduplicate
    seen = set()
    unique = []
    for rel in sorted(all_relations, key=lambda x: x['confidence'], reverse=True):
        key = (rel['source'], rel['target'])
        if key not in seen:
            seen.add(key)
            unique.append(rel)

    return unique
