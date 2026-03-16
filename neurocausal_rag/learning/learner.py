"""
NeuroCausal RAG - Learning Engine
Feedback collection and continuous learning
"""

from typing import List, Dict, Optional
from collections import defaultdict
import logging
from datetime import datetime

from ..config import LearningConfig
from ..interfaces import ILearningEngine

logger = logging.getLogger(__name__)


class LearningEngine(ILearningEngine):
    """
    Continuous Learning Engine.

    Collects user feedback and updates node weights.
    Discovers new links based on usage patterns.
    """

    def __init__(self, graph, config: Optional[LearningConfig] = None):
        self.graph = graph
        self.config = config or LearningConfig()

        # Feedback storage
        self.feedback_history: List[Dict] = []
        self.node_feedback_scores: Dict[str, List[float]] = defaultdict(list)
        self.query_patterns: Dict[str, int] = defaultdict(int)

    def record_feedback(
        self,
        query: str,
        result_ids: List[str],
        rating: float,
        comment: Optional[str] = None
    ) -> None:
        """Record user feedback for learning"""
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'result_ids': result_ids,
            'rating': rating,
            'comment': comment
        }
        self.feedback_history.append(feedback)

        # Update node scores
        for node_id in result_ids:
            self.node_feedback_scores[node_id].append(rating)

        # Track query patterns
        self.query_patterns[query.lower().strip()] += 1

        # Update weights if enough feedback
        if len(self.feedback_history) % 10 == 0:
            self.update_weights()

        logger.debug(f"Recorded feedback: rating={rating}, results={len(result_ids)}")

    def discover_links(self, min_confidence: float = 0.7) -> List[Dict]:
        """
        Discover potential new causal links based on feedback patterns.

        If documents are frequently retrieved together and get positive
        feedback, they might be causally related.
        """
        potential_links = []

        # Analyze co-occurrence patterns
        cooccurrence = defaultdict(int)
        for feedback in self.feedback_history:
            if feedback['rating'] > 0.6:  # Positive feedback
                ids = feedback['result_ids']
                for i in range(len(ids)):
                    for j in range(i+1, len(ids)):
                        pair = tuple(sorted([ids[i], ids[j]]))
                        cooccurrence[pair] += 1

        # Find strong co-occurrences
        for (id1, id2), count in cooccurrence.items():
            if count >= 3:  # At least 3 co-occurrences
                confidence = min(1.0, count / 10.0)
                if confidence >= min_confidence:
                    potential_links.append({
                        'source': id1,
                        'target': id2,
                        'relation_type': 'related',
                        'confidence': confidence,
                        'evidence': f'Co-occurred {count} times with positive feedback',
                        'method': 'feedback_cooccurrence'
                    })

        return potential_links

    def update_weights(self) -> Dict[str, float]:
        """
        Update node weights based on accumulated feedback.

        Nodes with consistently good feedback get higher importance.
        """
        updated = {}

        for node_id, scores in self.node_feedback_scores.items():
            if len(scores) >= 3:  # Minimum feedback threshold
                avg_score = sum(scores) / len(scores)

                # Calculate adjustment (bounded)
                current_importance = self.graph.get_importance(node_id)
                adjustment = (avg_score - 0.5) * self.config.learning_rate

                new_importance = max(0.1, min(0.9, current_importance + adjustment))

                # Update in graph
                if node_id in self.graph.nodes:
                    self.graph.nodes[node_id].importance = new_importance
                    updated[node_id] = new_importance

        logger.info(f"Updated weights for {len(updated)} nodes")
        return updated

    def get_statistics(self) -> Dict:
        """Get learning statistics"""
        total_feedback = len(self.feedback_history)
        avg_rating = 0.0
        if total_feedback > 0:
            avg_rating = sum(f['rating'] for f in self.feedback_history) / total_feedback

        return {
            'total_feedback': total_feedback,
            'average_rating': avg_rating,
            'unique_queries': len(self.query_patterns),
            'nodes_with_feedback': len(self.node_feedback_scores),
            'top_queries': sorted(
                self.query_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

    def export_learning_data(self) -> Dict:
        """Export all learning data for persistence"""
        return {
            'feedback_history': self.feedback_history,
            'node_scores': dict(self.node_feedback_scores),
            'query_patterns': dict(self.query_patterns)
        }

    def import_learning_data(self, data: Dict) -> None:
        """Import previously saved learning data"""
        self.feedback_history = data.get('feedback_history', [])
        self.node_feedback_scores = defaultdict(list, data.get('node_scores', {}))
        self.query_patterns = defaultdict(int, data.get('query_patterns', {}))
