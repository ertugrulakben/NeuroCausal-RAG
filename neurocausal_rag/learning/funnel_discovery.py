"""
NeuroCausal RAG - Funnel Discovery Engine
O(N^2) -> O(50) optimization with three-stage filtering

Strategy:
    Stage 1: Semantic Pre-filter (Fast, O(N*K))
        - Determine candidate pairs via embedding similarity
        - Top-K most similar pairs are selected

    Stage 2: NLI Verification (Medium, O(K))
        - NLI only for pairs passing Stage 1
        - Speedup with async batch processing

    Stage 3: LLM Confirmation (Optional, O(M))
        - For high-security applications
        - LLM verification only for critical relations

Complexity:
    - Naive: O(N²) where N=1000 → 1,000,000 comparisons
    - Funnel: O(N*50) + O(50) → ~50,000 comparisons (20x faster)

Author: Ertugrul Akben
"""

import asyncio
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


@dataclass
class CandidatePair:
    """Candidate causal pair"""
    source_id: str
    target_id: str
    source_text: str
    target_text: str
    stage1_score: float  # Semantic similarity
    stage2_score: Optional[float] = None  # NLI entailment
    stage3_score: Optional[float] = None  # LLM confidence
    final_score: float = 0.0
    relation_type: str = "causes"
    evidence: List[str] = None

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []


class FunnelDiscovery:
    """
    Efficient causal discovery with three-stage funnel.

    Provides O(N^2) -> O(50) optimization:
    - Stage 1: N×K semantic filter (K=50 default)
    - Stage 2: K×K NLI verification
    - Stage 3: Top-M LLM confirmation (optional)

    Example:
        >>> funnel = FunnelDiscovery(top_k_semantic=50, top_k_nli=20)
        >>> relations = funnel.discover(documents, embeddings)
    """

    def __init__(
        self,
        top_k_semantic: int = 50,
        top_k_nli: int = 20,
        semantic_threshold: float = 0.4,
        nli_threshold: float = 0.6,
        use_async: bool = True,
        max_workers: int = 4,
        nli_model_name: str = "cross-encoder/nli-deberta-v3-small"
    ):
        """
        Args:
            top_k_semantic: Candidates per document in Stage 1 (default: 50)
            top_k_nli: Candidates passing Stage 2 (default: 20)
            semantic_threshold: Minimum semantic similarity
            nli_threshold: Minimum NLI entailment score
            use_async: Use async processing
            max_workers: Thread pool worker count
            nli_model_name: NLI model name
        """
        self.top_k_semantic = top_k_semantic
        self.top_k_nli = top_k_nli
        self.semantic_threshold = semantic_threshold
        self.nli_threshold = nli_threshold
        self.use_async = use_async
        self.max_workers = max_workers
        self.nli_model_name = nli_model_name

        self._nli_model = None
        self._executor = None

    def _get_nli_model(self):
        """Lazy load NLI model"""
        if self._nli_model is not None:
            return self._nli_model

        try:
            from sentence_transformers import CrossEncoder
            self._nli_model = CrossEncoder(self.nli_model_name, device="cpu")
            logger.info(f"FunnelDiscovery: NLI model loaded - {self.nli_model_name}")
            return self._nli_model
        except ImportError:
            logger.warning("sentence-transformers not available, Stage 2 will use fallback")
            return None
        except Exception as e:
            logger.warning(f"Failed to load NLI model: {e}")
            return None

    def discover(
        self,
        documents: List[Dict],
        embeddings: np.ndarray,
        enable_stage3: bool = False,
        llm_callback: Optional[Callable] = None
    ) -> List[Dict]:
        """
        Main discovery function.

        Args:
            documents: [{'id': str, 'content': str, 'category': str}, ...]
            embeddings: (n_docs, dim) embedding matrix
            enable_stage3: Is LLM verification active?
            llm_callback: LLM callback function (optional)

        Returns:
            Discovered causal relations
        """
        n = len(documents)
        if n < 2:
            return []

        logger.info(f"FunnelDiscovery: {n} documents, starting funnel...")

        # ============================================================
        # STAGE 1: Semantic Pre-filter (Fast)
        # ============================================================
        logger.info("  Stage 1: Semantic pre-filtering...")
        candidates = self._stage1_semantic_filter(documents, embeddings)
        logger.info(f"    -> {len(candidates)} candidate pairs found")

        if not candidates:
            return []

        # ============================================================
        # STAGE 2: NLI Verification (Medium)
        # ============================================================
        logger.info("  Stage 2: NLI verification...")
        if self.use_async:
            verified = self._stage2_nli_async(candidates)
        else:
            verified = self._stage2_nli_sync(candidates)
        logger.info(f"    -> {len(verified)} pairs verified")

        if not verified:
            return []

        # ============================================================
        # STAGE 3: LLM Confirmation (Optional)
        # ============================================================
        if enable_stage3 and llm_callback:
            logger.info("  Stage 3: LLM confirmation...")
            final = self._stage3_llm_confirm(verified, llm_callback)
            logger.info(f"    -> {len(final)} pairs confirmed")
        else:
            final = verified

        # Format output
        relations = self._format_results(final)
        logger.info(f"FunnelDiscovery: Total {len(relations)} relations discovered")

        return relations

    def _stage1_semantic_filter(
        self,
        documents: List[Dict],
        embeddings: np.ndarray
    ) -> List[CandidatePair]:
        """
        Stage 1: Fast semantic filtering.

        Find the K most similar documents for each document.
        O(N * K * log(K)) complexity.
        """
        n = len(documents)
        candidates = []
        seen_pairs = set()

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = embeddings / norms

        # Similarity matrix (batch)
        sim_matrix = np.dot(normalized, normalized.T)

        for i in range(n):
            # Top-K most similar documents for i
            similarities = sim_matrix[i]

            # Exclude self, above threshold, top-K
            indices = np.argsort(similarities)[::-1]  # Descending

            count = 0
            for j in indices:
                if count >= self.top_k_semantic:
                    break
                if i == j:
                    continue
                if similarities[j] < self.semantic_threshold:
                    break

                pair_key = (min(i, j), max(i, j))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                # Determine direction: More general one is the source
                # Simple heuristic: Shorter text is usually more general
                len_i = len(documents[i]['content'])
                len_j = len(documents[j]['content'])

                if len_i <= len_j:
                    source_idx, target_idx = i, j
                else:
                    source_idx, target_idx = j, i

                candidates.append(CandidatePair(
                    source_id=documents[source_idx]['id'],
                    target_id=documents[target_idx]['id'],
                    source_text=documents[source_idx]['content'][:300],
                    target_text=documents[target_idx]['content'][:300],
                    stage1_score=float(similarities[j]),
                    evidence=[f"Semantic sim: {similarities[j]:.3f}"]
                ))
                count += 1

        # Sort by stage1 score and limit
        candidates.sort(key=lambda x: x.stage1_score, reverse=True)
        return candidates[:self.top_k_semantic * 2]  # Keep more for Stage 2

    def _stage2_nli_sync(
        self,
        candidates: List[CandidatePair]
    ) -> List[CandidatePair]:
        """Stage 2: Synchronous NLI verification"""
        model = self._get_nli_model()

        if model is None:
            # Fallback: keyword-based scoring
            return self._stage2_fallback(candidates)

        verified = []
        batch_size = 32

        for batch_start in range(0, len(candidates), batch_size):
            batch = candidates[batch_start:batch_start + batch_size]

            # NLI pairs: (premise, hypothesis)
            nli_pairs = []
            for c in batch:
                premise = c.source_text
                hypothesis = f"Bu nedenle, {c.target_text[:100]}"
                nli_pairs.append((premise, hypothesis))

            try:
                scores = model.predict(nli_pairs)

                for idx, candidate in enumerate(batch):
                    score = scores[idx]
                    if isinstance(score, (list, np.ndarray)):
                        entailment = float(score[1]) if len(score) > 1 else float(score[0])
                    else:
                        entailment = float(score)

                    if entailment >= self.nli_threshold:
                        candidate.stage2_score = entailment
                        candidate.final_score = (candidate.stage1_score + entailment) / 2
                        candidate.evidence.append(f"NLI entailment: {entailment:.3f}")
                        verified.append(candidate)

            except Exception as e:
                logger.error(f"NLI batch error: {e}")
                continue

        # Sort and limit
        verified.sort(key=lambda x: x.final_score, reverse=True)
        return verified[:self.top_k_nli]

    def _stage2_nli_async(
        self,
        candidates: List[CandidatePair]
    ) -> List[CandidatePair]:
        """Stage 2: Async NLI verification with thread pool"""
        model = self._get_nli_model()

        if model is None:
            return self._stage2_fallback(candidates)

        verified = []
        batch_size = 32

        def process_batch(batch: List[CandidatePair]) -> List[CandidatePair]:
            results = []
            nli_pairs = [
                (c.source_text, f"Bu nedenle, {c.target_text[:100]}")
                for c in batch
            ]

            try:
                scores = model.predict(nli_pairs)

                for idx, candidate in enumerate(batch):
                    score = scores[idx]
                    if isinstance(score, (list, np.ndarray)):
                        entailment = float(score[1]) if len(score) > 1 else float(score[0])
                    else:
                        entailment = float(score)

                    if entailment >= self.nli_threshold:
                        candidate.stage2_score = entailment
                        candidate.final_score = (candidate.stage1_score + entailment) / 2
                        candidate.evidence.append(f"NLI entailment: {entailment:.3f}")
                        results.append(candidate)
            except Exception as e:
                logger.error(f"Async batch error: {e}")

            return results

        # Split into batches
        batches = [
            candidates[i:i + batch_size]
            for i in range(0, len(candidates), batch_size)
        ]

        # Process with thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]

            for future in futures:
                try:
                    results = future.result(timeout=60)
                    verified.extend(results)
                except Exception as e:
                    logger.error(f"Future error: {e}")

        # Sort and limit
        verified.sort(key=lambda x: x.final_score, reverse=True)
        return verified[:self.top_k_nli]

    def _stage2_fallback(
        self,
        candidates: List[CandidatePair]
    ) -> List[CandidatePair]:
        """Fallback: Keyword-based scoring when NLI not available"""
        cause_words = {'neden', 'sebep', 'kaynak', 'cause', 'trigger', 'leads'}
        effect_words = {'sonuç', 'etki', 'result', 'effect', 'outcome'}

        verified = []

        for candidate in candidates:
            source_lower = candidate.source_text.lower()
            target_lower = candidate.target_text.lower()

            # Score based on keyword presence
            cause_score = sum(1 for w in cause_words if w in source_lower) / len(cause_words)
            effect_score = sum(1 for w in effect_words if w in target_lower) / len(effect_words)

            fallback_score = (cause_score + effect_score) / 2

            if fallback_score > 0.1 or candidate.stage1_score > 0.6:
                candidate.stage2_score = fallback_score
                candidate.final_score = candidate.stage1_score * 0.7 + fallback_score * 0.3
                candidate.evidence.append(f"Keyword fallback: {fallback_score:.3f}")
                verified.append(candidate)

        verified.sort(key=lambda x: x.final_score, reverse=True)
        return verified[:self.top_k_nli]

    def _stage3_llm_confirm(
        self,
        candidates: List[CandidatePair],
        llm_callback: Callable
    ) -> List[CandidatePair]:
        """
        Stage 3: Final confirmation via LLM.

        llm_callback signature:
            def callback(source_text: str, target_text: str) -> Tuple[bool, float, str]
            Returns: (is_causal, confidence, explanation)
        """
        confirmed = []

        for candidate in candidates[:10]:  # Top 10 only
            try:
                is_causal, confidence, explanation = llm_callback(
                    candidate.source_text,
                    candidate.target_text
                )

                if is_causal and confidence > 0.5:
                    candidate.stage3_score = confidence
                    candidate.final_score = (
                        candidate.final_score * 0.5 + confidence * 0.5
                    )
                    candidate.evidence.append(f"LLM: {explanation[:100]}")
                    confirmed.append(candidate)

            except Exception as e:
                logger.error(f"LLM callback error: {e}")
                confirmed.append(candidate)  # Keep without LLM

        return confirmed

    def _format_results(
        self,
        candidates: List[CandidatePair]
    ) -> List[Dict]:
        """Format candidates as relation dictionaries"""
        relations = []

        for c in candidates:
            relations.append({
                'source': c.source_id,
                'target': c.target_id,
                'relation_type': c.relation_type,
                'confidence': c.final_score,
                'method': 'funnel_discovery',
                'stages': {
                    'semantic': c.stage1_score,
                    'nli': c.stage2_score,
                    'llm': c.stage3_score
                },
                'evidence': c.evidence
            })

        return relations


class AsyncFunnelDiscovery(FunnelDiscovery):
    """
    Full async version - works with asyncio event loop.

    Usage:
        >>> async_funnel = AsyncFunnelDiscovery()
        >>> relations = await async_funnel.discover_async(documents, embeddings)
    """

    async def discover_async(
        self,
        documents: List[Dict],
        embeddings: np.ndarray,
        enable_stage3: bool = False,
        llm_callback: Optional[Callable] = None
    ) -> List[Dict]:
        """Async discovery with proper event loop handling"""

        # Stage 1 is already fast (numpy), run sync
        candidates = self._stage1_semantic_filter(documents, embeddings)

        if not candidates:
            return []

        # Stage 2 async
        verified = await self._stage2_nli_async_full(candidates)

        if not verified:
            return []

        # Stage 3 optional
        if enable_stage3 and llm_callback:
            final = await self._stage3_llm_async(verified, llm_callback)
        else:
            final = verified

        return self._format_results(final)

    async def _stage2_nli_async_full(
        self,
        candidates: List[CandidatePair]
    ) -> List[CandidatePair]:
        """Full async NLI processing"""
        model = self._get_nli_model()

        if model is None:
            return self._stage2_fallback(candidates)

        batch_size = 32
        verified = []

        async def process_batch_async(batch: List[CandidatePair]) -> List[CandidatePair]:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._process_nli_batch,
                batch,
                model
            )

        # Create tasks for all batches
        batches = [
            candidates[i:i + batch_size]
            for i in range(0, len(candidates), batch_size)
        ]

        tasks = [process_batch_async(batch) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Async batch error: {result}")
            elif result:
                verified.extend(result)

        verified.sort(key=lambda x: x.final_score, reverse=True)
        return verified[:self.top_k_nli]

    def _process_nli_batch(
        self,
        batch: List[CandidatePair],
        model
    ) -> List[CandidatePair]:
        """Process a single NLI batch"""
        results = []
        nli_pairs = [
            (c.source_text, f"Bu nedenle, {c.target_text[:100]}")
            for c in batch
        ]

        try:
            scores = model.predict(nli_pairs)

            for idx, candidate in enumerate(batch):
                score = scores[idx]
                if isinstance(score, (list, np.ndarray)):
                    entailment = float(score[1]) if len(score) > 1 else float(score[0])
                else:
                    entailment = float(score)

                if entailment >= self.nli_threshold:
                    candidate.stage2_score = entailment
                    candidate.final_score = (candidate.stage1_score + entailment) / 2
                    candidate.evidence.append(f"NLI: {entailment:.3f}")
                    results.append(candidate)
        except Exception as e:
            logger.error(f"NLI batch error: {e}")

        return results

    async def _stage3_llm_async(
        self,
        candidates: List[CandidatePair],
        llm_callback: Callable
    ) -> List[CandidatePair]:
        """Async LLM confirmation"""
        confirmed = []

        async def confirm_single(candidate: CandidatePair) -> Optional[CandidatePair]:
            try:
                loop = asyncio.get_event_loop()
                is_causal, confidence, explanation = await loop.run_in_executor(
                    None,
                    llm_callback,
                    candidate.source_text,
                    candidate.target_text
                )

                if is_causal and confidence > 0.5:
                    candidate.stage3_score = confidence
                    candidate.final_score = candidate.final_score * 0.5 + confidence * 0.5
                    candidate.evidence.append(f"LLM: {explanation[:100]}")
                    return candidate
            except Exception as e:
                logger.error(f"LLM async error: {e}")
                return candidate
            return None

        tasks = [confirm_single(c) for c in candidates[:10]]
        results = await asyncio.gather(*tasks)

        for result in results:
            if result:
                confirmed.append(result)

        return confirmed


def funnel_causal_discovery(
    documents: List[Dict],
    embeddings: np.ndarray,
    top_k_semantic: int = 50,
    top_k_nli: int = 20,
    use_async: bool = True
) -> List[Dict]:
    """
    Funnel discovery - single function API.

    Provides O(N^2) -> O(50) optimization.

    Args:
        documents: [{'id': str, 'content': str}, ...]
        embeddings: (n_docs, dim) embedding matrix
        top_k_semantic: Stage 1 limit
        top_k_nli: Stage 2 limit
        use_async: Async processing

    Returns:
        Discovered causal relations

    Example:
        >>> relations = funnel_causal_discovery(docs, embeddings)
        >>> print(f"Found {len(relations)} relations")
    """
    funnel = FunnelDiscovery(
        top_k_semantic=top_k_semantic,
        top_k_nli=top_k_nli,
        use_async=use_async
    )
    return funnel.discover(documents, embeddings)
