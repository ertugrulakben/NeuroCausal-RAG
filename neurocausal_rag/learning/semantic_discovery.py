"""
NeuroCausal RAG - Semantic Causal Discovery Engine
Advanced causal discovery system that goes beyond regex limitations

This module uses instead of simple regex patterns:
1. Asymmetric Embedding Analysis - A->B vs B->A asymmetry
2. Cluster-based Discovery - Structure within semantic clusters
3. Graph Propagation - Transitive relation inference
4. Multi-Signal Fusion - Multiple signal combination

Author: Ertugrul Akben
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CausalSignal:
    """Single causal signal"""
    source_id: str
    target_id: str
    signal_type: str  # 'asymmetric', 'cluster', 'lexical', 'structural'
    strength: float  # 0-1
    evidence: str  # Why this signal was detected


class SemanticCausalDiscovery:
    """
    Semantic causal discovery beyond regex.

    Core principle: Causality is not a single signal,
    but a combination of multiple weak signals.

    Signals:
    1. Asymmetric Similarity: If A->B similarity differs from B->A
    2. Lexical Causality: Causal keywords in text
    3. Structural Position: Position in graph structure
    4. Category Coherence: Documents in the same category
    5. Temporal Markers: Time indicators
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        asymmetry_threshold: float = 0.1,
        min_confidence: float = 0.6
    ):
        self.similarity_threshold = similarity_threshold
        self.asymmetry_threshold = asymmetry_threshold
        self.min_confidence = min_confidence

        # Causality keywords (language-independent scoring)
        self.cause_markers = {
            # Turkish
            'neden': 0.9, 'sebep': 0.9, 'kaynak': 0.7, 'etken': 0.8,
            'faktör': 0.8, 'tetikler': 0.9, 'başlatır': 0.8, 'oluşturur': 0.7,
            # English
            'cause': 0.9, 'source': 0.7, 'factor': 0.8, 'trigger': 0.9,
            'origin': 0.7, 'driver': 0.8, 'leads': 0.8, 'produces': 0.7,
        }

        self.effect_markers = {
            # Turkish
            'sonuç': 0.9, 'etki': 0.9, 'netice': 0.8, 'çıktı': 0.7,
            'oluşur': 0.7, 'meydana': 0.8, 'gerçekleşir': 0.7,
            # English
            'effect': 0.9, 'result': 0.9, 'outcome': 0.8, 'impact': 0.9,
            'consequence': 0.9, 'occurs': 0.7, 'happens': 0.6,
        }

        self.temporal_cause_markers = {
            'önce': 0.7, 'başlangıç': 0.8, 'ilk': 0.6,
            'before': 0.7, 'initial': 0.8, 'first': 0.6, 'primary': 0.7,
        }

        self.temporal_effect_markers = {
            'sonra': 0.7, 'ardından': 0.8, 'sonuç': 0.7,
            'after': 0.7, 'then': 0.6, 'subsequently': 0.8, 'resulting': 0.8,
        }

    def discover(
        self,
        documents: List[Dict],
        embeddings: np.ndarray
    ) -> List[Dict]:
        """
        Main discovery function.

        Args:
            documents: [{'id': str, 'content': str, 'category': str}, ...]
            embeddings: (n_docs, dim) embedding matrix

        Returns:
            Discovered causal relations
        """
        n = len(documents)
        if n < 2:
            return []

        logger.info(f"SemanticCausalDiscovery: analyzing {n} documents...")

        # Collect all signals
        all_signals: List[CausalSignal] = []

        # 1. Asymmetric Similarity Analysis
        logger.info("  1. Asymmetric similarity analysis...")
        asymmetric_signals = self._analyze_asymmetric_similarity(documents, embeddings)
        all_signals.extend(asymmetric_signals)

        # 2. Lexical Causality Signals
        logger.info("  2. Lexical causality analysis...")
        lexical_signals = self._analyze_lexical_causality(documents)
        all_signals.extend(lexical_signals)

        # 3. Category-based Signals
        logger.info("  3. Category-based analysis...")
        category_signals = self._analyze_category_structure(documents, embeddings)
        all_signals.extend(category_signals)

        # 4. Cluster-based Discovery
        logger.info("  4. Cluster-based analysis...")
        cluster_signals = self._analyze_clusters(documents, embeddings)
        all_signals.extend(cluster_signals)

        # Multi-Signal Fusion
        logger.info("  5. Signal fusion...")
        fused_relations = self._fuse_signals(all_signals)

        # Filter by confidence
        confident_relations = [
            r for r in fused_relations
            if r['confidence'] >= self.min_confidence
        ]

        logger.info(f"  Total {len(confident_relations)} relations discovered")
        return confident_relations

    def _analyze_asymmetric_similarity(
        self,
        documents: List[Dict],
        embeddings: np.ndarray
    ) -> List[CausalSignal]:
        """
        Asymmetric similarity analysis.

        Idea: If A's similarity to B differs from B's similarity to A,
        there may be a directional relationship (causality is typically directional).

        Example: In "Greenhouse gases" -> "Global warming",
        greenhouse gases describe global warming but not vice versa.
        """
        signals = []
        n = len(documents)

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = embeddings / norms

        # Compute similarity matrix
        sim_matrix = np.dot(normalized, normalized.T)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                sim_ij = sim_matrix[i, j]
                sim_ji = sim_matrix[j, i]  # Should be the same actually but...

                # Unidirectional "context" analysis
                # A's coverage of B vs B's coverage of A
                content_i = documents[i]['content'].lower()
                content_j = documents[j]['content'].lower()

                # Word intersection analysis (for asymmetry detection)
                words_i = set(content_i.split())
                words_j = set(content_j.split())

                # i's coverage of j
                if len(words_j) > 0:
                    coverage_i_to_j = len(words_i & words_j) / len(words_j)
                else:
                    coverage_i_to_j = 0

                # j's coverage of i
                if len(words_i) > 0:
                    coverage_j_to_i = len(words_i & words_j) / len(words_i)
                else:
                    coverage_j_to_i = 0

                # Asymmetry: if i covers j more, there may be an i->j relation
                asymmetry = coverage_i_to_j - coverage_j_to_i

                if sim_ij > self.similarity_threshold and asymmetry > self.asymmetry_threshold:
                    signals.append(CausalSignal(
                        source_id=documents[i]['id'],
                        target_id=documents[j]['id'],
                        signal_type='asymmetric',
                        strength=min(1.0, (sim_ij + asymmetry) / 2),
                        evidence=f"Sim={sim_ij:.2f}, Asymmetry={asymmetry:.2f}"
                    ))

        return signals

    def _analyze_lexical_causality(
        self,
        documents: List[Dict]
    ) -> List[CausalSignal]:
        """
        Lexical causality analysis.

        Classify documents as "cause" or "effect" based on
        the presence of cause/effect keywords in the text.
        """
        signals = []

        # Compute cause/effect score for each document
        cause_scores = {}
        effect_scores = {}

        for doc in documents:
            content = doc['content'].lower()

            cause_score = 0.0
            for marker, weight in self.cause_markers.items():
                if marker in content:
                    cause_score += weight
            cause_score += sum(
                w for m, w in self.temporal_cause_markers.items() if m in content
            )

            effect_score = 0.0
            for marker, weight in self.effect_markers.items():
                if marker in content:
                    effect_score += weight
            effect_score += sum(
                w for m, w in self.temporal_effect_markers.items() if m in content
            )

            cause_scores[doc['id']] = cause_score
            effect_scores[doc['id']] = effect_score

        # Match cause-scored documents with effect-scored documents
        for doc_i in documents:
            for doc_j in documents:
                if doc_i['id'] == doc_j['id']:
                    continue

                cause_i = cause_scores[doc_i['id']]
                effect_j = effect_scores[doc_j['id']]

                # if i is cause, j is effect
                if cause_i > 0.5 and effect_j > 0.5:
                    strength = min(1.0, (cause_i + effect_j) / 4)  # Normalize
                    signals.append(CausalSignal(
                        source_id=doc_i['id'],
                        target_id=doc_j['id'],
                        signal_type='lexical',
                        strength=strength,
                        evidence=f"Cause={cause_i:.1f}, Effect={effect_j:.1f}"
                    ))

        return signals

    def _analyze_category_structure(
        self,
        documents: List[Dict],
        embeddings: np.ndarray
    ) -> List[CausalSignal]:
        """
        Category-based structural analysis.

        Documents in the same category may have structural causality.
        Example: In "climate" category: greenhouse->warming->glacier->sea chain.
        """
        signals = []

        # Group by categories
        categories = defaultdict(list)
        for i, doc in enumerate(documents):
            cat = doc.get('category', 'unknown')
            categories[cat].append((i, doc))

        # Analyze within each category
        for cat, cat_docs in categories.items():
            if len(cat_docs) < 2:
                continue

            # Intra-category similarity matrix
            indices = [i for i, _ in cat_docs]
            cat_embeddings = embeddings[indices]

            norms = np.linalg.norm(cat_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized = cat_embeddings / norms
            sim_matrix = np.dot(normalized, normalized.T)

            # Find highest similarity pairs
            for i in range(len(cat_docs)):
                for j in range(len(cat_docs)):
                    if i >= j:
                        continue

                    sim = sim_matrix[i, j]
                    if sim > self.similarity_threshold:
                        doc_i = cat_docs[i][1]
                        doc_j = cat_docs[j][1]

                        # Which comes first? (alphabetical order heuristic - improvable)
                        if doc_i['id'] < doc_j['id']:
                            source, target = doc_i, doc_j
                        else:
                            source, target = doc_j, doc_i

                        signals.append(CausalSignal(
                            source_id=source['id'],
                            target_id=target['id'],
                            signal_type='category',
                            strength=sim * 0.7,  # Category signal is weaker
                            evidence=f"Category={cat}, Sim={sim:.2f}"
                        ))

        return signals

    def _analyze_clusters(
        self,
        documents: List[Dict],
        embeddings: np.ndarray,
        n_clusters: int = 5
    ) -> List[CausalSignal]:
        """
        Cluster-based causal analysis.

        Cluster documents in embedding space.
        Proximity to cluster centers -> importance
        Inter-cluster transitions -> causal chain
        """
        signals = []
        n = len(documents)

        if n < n_clusters:
            return signals

        # Simple k-means clustering
        # (In production, sklearn can be used)
        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=min(n_clusters, n//2), random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            # Cluster centers
            centers = kmeans.cluster_centers_

            # Each document's distance to center (importance score)
            importance = []
            for i, emb in enumerate(embeddings):
                center = centers[cluster_labels[i]]
                dist = np.linalg.norm(emb - center)
                importance.append(1.0 / (1.0 + dist))

            # Signal from important documents to less important ones
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue

                    # If in the same cluster and i is more important
                    if cluster_labels[i] == cluster_labels[j]:
                        if importance[i] > importance[j] + 0.1:
                            signals.append(CausalSignal(
                                source_id=documents[i]['id'],
                                target_id=documents[j]['id'],
                                signal_type='cluster',
                                strength=(importance[i] - importance[j]) * 0.8,
                                evidence=f"Cluster={cluster_labels[i]}, Imp_diff={importance[i]-importance[j]:.2f}"
                            ))
        except ImportError:
            # Simple clustering if sklearn not available
            pass

        return signals

    def _fuse_signals(
        self,
        signals: List[CausalSignal]
    ) -> List[Dict]:
        """
        Multi-Signal Fusion.

        Multiple weak signals -> strong relation

        Fusion strategy:
        - If multiple signals for the same (source, target), boost
        - Different signal types are more valuable
        - Use weighted sum, not maximum
        """
        # Group signals by (source, target) pairs
        pair_signals: Dict[Tuple[str, str], List[CausalSignal]] = defaultdict(list)

        for signal in signals:
            key = (signal.source_id, signal.target_id)
            pair_signals[key].append(signal)

        # Fusion for each pair
        relations = []
        for (source, target), signals_list in pair_signals.items():
            if not signals_list:
                continue

            # Count signal types
            signal_types = set(s.signal_type for s in signals_list)
            type_diversity = len(signal_types) / 4  # 4 different types exist

            # Weighted sum
            total_strength = sum(s.strength for s in signals_list)
            avg_strength = total_strength / len(signals_list)

            # Diversity bonus
            confidence = min(1.0, avg_strength * (1 + type_diversity))

            # Determine relation type
            relation_type = self._infer_relation_type(signals_list)

            relations.append({
                'source': source,
                'target': target,
                'relation_type': relation_type,
                'confidence': confidence,
                'signal_count': len(signals_list),
                'signal_types': list(signal_types),
                'evidence': [s.evidence for s in signals_list[:3]],  # First 3 evidence items
                'method': 'semantic_fusion'
            })

        return sorted(relations, key=lambda x: x['confidence'], reverse=True)

    def _infer_relation_type(self, signals: List[CausalSignal]) -> str:
        """Infer relation type from signal types"""
        # If lexical signal exists and is strong -> causes
        lexical_signals = [s for s in signals if s.signal_type == 'lexical']
        if lexical_signals and max(s.strength for s in lexical_signals) > 0.6:
            return 'causes'

        # If asymmetric signal exists -> supports
        asymmetric_signals = [s for s in signals if s.signal_type == 'asymmetric']
        if asymmetric_signals:
            return 'supports'

        # Category or cluster signal -> related
        return 'related'


class GraphPropagation:
    """
    Transitive relation inference on the graph.

    If A->B and B->C exist, A->C can also be inferred (with decay).
    """

    def __init__(self, decay_factor: float = 0.7):
        self.decay_factor = decay_factor

    def propagate(
        self,
        relations: List[Dict],
        max_depth: int = 3
    ) -> List[Dict]:
        """
        Infer transitive relations from existing relations.
        """
        # Build adjacency list
        adj = defaultdict(list)
        edge_weights = {}

        for rel in relations:
            src, tgt = rel['source'], rel['target']
            conf = rel['confidence']
            adj[src].append(tgt)
            edge_weights[(src, tgt)] = conf

        # New relations
        new_relations = []
        existing_pairs = set((r['source'], r['target']) for r in relations)

        # BFS from each node
        all_nodes = list(adj.keys())  # Copy to avoid mutation issues
        for start_node in all_nodes:
            visited = {start_node: 1.0}  # node -> strength
            queue = [(start_node, 1.0, 0)]  # (node, strength, depth)

            while queue:
                current, strength, depth = queue.pop(0)

                if depth >= max_depth:
                    continue

                for neighbor in adj[current]:
                    edge_strength = edge_weights.get((current, neighbor), 0.5)
                    new_strength = strength * edge_strength * self.decay_factor

                    if neighbor not in visited or visited[neighbor] < new_strength:
                        visited[neighbor] = new_strength
                        queue.append((neighbor, new_strength, depth + 1))

                        # New transitive relation
                        if (start_node, neighbor) not in existing_pairs:
                            existing_pairs.add((start_node, neighbor))
                            new_relations.append({
                                'source': start_node,
                                'target': neighbor,
                                'relation_type': 'causes',  # Transitive → causes
                                'confidence': new_strength,
                                'depth': depth + 1,
                                'method': 'propagation'
                            })

        return new_relations


def enhanced_causal_discovery(
    documents: List[Dict],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.5,
    min_confidence: float = 0.55
) -> List[Dict]:
    """
    Enhanced causal discovery - single function API.

    Args:
        documents: [{'id': str, 'content': str, 'category': str}, ...]
        embeddings: (n_docs, dim) embedding matrix
        similarity_threshold: Similarity threshold
        min_confidence: Minimum confidence score

    Returns:
        List of discovered causal relations
    """
    # 1. Semantic Discovery
    semantic = SemanticCausalDiscovery(
        similarity_threshold=similarity_threshold,
        min_confidence=min_confidence
    )
    initial_relations = semantic.discover(documents, embeddings)

    # 2. Graph Propagation
    propagator = GraphPropagation(decay_factor=0.7)
    transitive_relations = propagator.propagate(initial_relations, max_depth=2)

    # 3. Merge and sort
    all_relations = initial_relations + transitive_relations

    # Remove duplicates
    seen = set()
    unique_relations = []
    for rel in sorted(all_relations, key=lambda x: x['confidence'], reverse=True):
        key = (rel['source'], rel['target'])
        if key not in seen:
            seen.add(key)
            unique_relations.append(rel)

    return unique_relations
