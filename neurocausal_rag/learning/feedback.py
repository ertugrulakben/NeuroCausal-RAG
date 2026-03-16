"""
NeuroCausal RAG - Feedback Loop System
Continuous learning from user feedback

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Feedback Loop System                          │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │   User Feedback          FeedbackStore          Weight Adjuster  │
    │   ┌──────────┐          ┌───────────┐          ┌───────────┐    │
    │   │ Rating   │ -------> │ SQLite/   │ -------> │ Edge      │    │
    │   │ Query    │          │ JSON      │          │ Weights   │    │
    │   │ Results  │          │ Storage   │          │ Update    │    │
    │   └──────────┘          └───────────┘          └───────────┘    │
    │                               │                      │           │
    │                               v                      v           │
    │                        ┌───────────┐          ┌───────────┐     │
    │                        │ Analytics │          │ Graph     │     │
    │                        │ Dashboard │          │ Evolution │     │
    │                        └───────────┘          └───────────┘     │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘

Author: Ertugrul Akben
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import json
import logging
import threading
import sqlite3
import uuid

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of feedback"""
    EXPLICIT = "explicit"       # Direct user rating
    IMPLICIT = "implicit"       # Click-through, dwell time
    CORRECTION = "correction"   # User-provided correct answer


@dataclass
class FeedbackRecord:
    """Single feedback record"""
    id: str
    timestamp: datetime
    query: str
    result_ids: List[str]
    rating: float  # 0.0 to 1.0
    feedback_type: FeedbackType = FeedbackType.EXPLICIT
    comment: Optional[str] = None
    correct_answer: Optional[str] = None
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'query': self.query,
            'result_ids': self.result_ids,
            'rating': self.rating,
            'feedback_type': self.feedback_type.value,
            'comment': self.comment,
            'correct_answer': self.correct_answer,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'FeedbackRecord':
        return cls(
            id=data['id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            query=data['query'],
            result_ids=data['result_ids'],
            rating=data['rating'],
            feedback_type=FeedbackType(data.get('feedback_type', 'explicit')),
            comment=data.get('comment'),
            correct_answer=data.get('correct_answer'),
            metadata=data.get('metadata')
        )


class FeedbackStore:
    """
    Persistent feedback storage.

    Supports SQLite for production and JSON for development.
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        use_sqlite: bool = True
    ):
        self.storage_path = storage_path or "feedback_store"
        self.use_sqlite = use_sqlite
        self._lock = threading.Lock()

        if use_sqlite:
            self._init_sqlite()
        else:
            self._init_json()

    def _init_sqlite(self):
        """Initialize SQLite database"""
        db_path = f"{self.storage_path}.db"
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                result_ids TEXT NOT NULL,
                rating REAL NOT NULL,
                feedback_type TEXT DEFAULT 'explicit',
                comment TEXT,
                correct_answer TEXT,
                metadata TEXT
            )
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON feedback(timestamp)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_rating ON feedback(rating)
        """)

        self.conn.commit()
        logger.info(f"SQLite feedback store initialized: {db_path}")

    def _init_json(self):
        """Initialize JSON file storage"""
        self.json_path = Path(f"{self.storage_path}.json")
        if not self.json_path.exists():
            self.json_path.write_text("[]")
        logger.info(f"JSON feedback store initialized: {self.json_path}")

    def add(self, record: FeedbackRecord) -> str:
        """Add feedback record"""
        with self._lock:
            if self.use_sqlite:
                return self._add_sqlite(record)
            return self._add_json(record)

    def _add_sqlite(self, record: FeedbackRecord) -> str:
        self.conn.execute("""
            INSERT INTO feedback
            (id, timestamp, query, result_ids, rating, feedback_type, comment, correct_answer, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.id,
            record.timestamp.isoformat(),
            record.query,
            json.dumps(record.result_ids),
            record.rating,
            record.feedback_type.value,
            record.comment,
            record.correct_answer,
            json.dumps(record.metadata) if record.metadata else None
        ))
        self.conn.commit()
        return record.id

    def _add_json(self, record: FeedbackRecord) -> str:
        data = json.loads(self.json_path.read_text())
        data.append(record.to_dict())
        self.json_path.write_text(json.dumps(data, indent=2))
        return record.id

    def get_recent(
        self,
        limit: int = 100,
        min_rating: Optional[float] = None,
        max_rating: Optional[float] = None
    ) -> List[FeedbackRecord]:
        """Get recent feedback records"""
        with self._lock:
            if self.use_sqlite:
                return self._get_recent_sqlite(limit, min_rating, max_rating)
            return self._get_recent_json(limit, min_rating, max_rating)

    def _get_recent_sqlite(
        self,
        limit: int,
        min_rating: Optional[float],
        max_rating: Optional[float]
    ) -> List[FeedbackRecord]:
        query = "SELECT * FROM feedback WHERE 1=1"
        params = []

        if min_rating is not None:
            query += " AND rating >= ?"
            params.append(min_rating)
        if max_rating is not None:
            query += " AND rating <= ?"
            params.append(max_rating)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.execute(query, params)
        rows = cursor.fetchall()

        return [self._row_to_record(row) for row in rows]

    def _get_recent_json(
        self,
        limit: int,
        min_rating: Optional[float],
        max_rating: Optional[float]
    ) -> List[FeedbackRecord]:
        data = json.loads(self.json_path.read_text())

        # Filter
        if min_rating is not None:
            data = [d for d in data if d['rating'] >= min_rating]
        if max_rating is not None:
            data = [d for d in data if d['rating'] <= max_rating]

        # Sort and limit
        data.sort(key=lambda x: x['timestamp'], reverse=True)
        data = data[:limit]

        return [FeedbackRecord.from_dict(d) for d in data]

    def _row_to_record(self, row: sqlite3.Row) -> FeedbackRecord:
        return FeedbackRecord(
            id=row['id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            query=row['query'],
            result_ids=json.loads(row['result_ids']),
            rating=row['rating'],
            feedback_type=FeedbackType(row['feedback_type']),
            comment=row['comment'],
            correct_answer=row['correct_answer'],
            metadata=json.loads(row['metadata']) if row['metadata'] else None
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        with self._lock:
            if self.use_sqlite:
                return self._get_stats_sqlite()
            return self._get_stats_json()

    def _get_stats_sqlite(self) -> Dict[str, Any]:
        cursor = self.conn.execute("""
            SELECT
                COUNT(*) as total,
                AVG(rating) as avg_rating,
                MIN(rating) as min_rating,
                MAX(rating) as max_rating,
                SUM(CASE WHEN rating >= 0.7 THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN rating < 0.3 THEN 1 ELSE 0 END) as negative
            FROM feedback
        """)
        row = cursor.fetchone()

        return {
            'total_feedback': row['total'] or 0,
            'average_rating': row['avg_rating'] or 0.0,
            'min_rating': row['min_rating'] or 0.0,
            'max_rating': row['max_rating'] or 0.0,
            'positive_count': row['positive'] or 0,
            'negative_count': row['negative'] or 0
        }

    def _get_stats_json(self) -> Dict[str, Any]:
        data = json.loads(self.json_path.read_text())

        if not data:
            return {
                'total_feedback': 0,
                'average_rating': 0.0,
                'min_rating': 0.0,
                'max_rating': 0.0,
                'positive_count': 0,
                'negative_count': 0
            }

        ratings = [d['rating'] for d in data]
        return {
            'total_feedback': len(data),
            'average_rating': sum(ratings) / len(ratings),
            'min_rating': min(ratings),
            'max_rating': max(ratings),
            'positive_count': sum(1 for r in ratings if r >= 0.7),
            'negative_count': sum(1 for r in ratings if r < 0.3)
        }

    def get_by_result_id(self, result_id: str) -> List[FeedbackRecord]:
        """Get all feedback for a specific result"""
        with self._lock:
            if self.use_sqlite:
                cursor = self.conn.execute(
                    "SELECT * FROM feedback WHERE result_ids LIKE ?",
                    (f'%"{result_id}"%',)
                )
                return [self._row_to_record(row) for row in cursor.fetchall()]
            else:
                data = json.loads(self.json_path.read_text())
                return [
                    FeedbackRecord.from_dict(d)
                    for d in data
                    if result_id in d['result_ids']
                ]


class WeightAdjuster:
    """
    Adjusts graph edge weights based on feedback.

    Uses exponential moving average to smooth weight updates.
    """

    def __init__(
        self,
        graph_engine,
        learning_rate: float = 0.1,
        min_feedback_count: int = 3,
        decay_factor: float = 0.95
    ):
        self.graph = graph_engine
        self.learning_rate = learning_rate
        self.min_feedback_count = min_feedback_count
        self.decay_factor = decay_factor

        # Track feedback counts per edge
        self._edge_feedback: Dict[Tuple[str, str], List[float]] = {}
        self._lock = threading.Lock()

    def process_feedback(self, feedback: FeedbackRecord) -> Dict[str, Any]:
        """
        Process feedback and update edge weights.

        Returns update statistics.
        """
        updates = []

        # Update weights between consecutive results
        for i in range(len(feedback.result_ids) - 1):
            source = feedback.result_ids[i]
            target = feedback.result_ids[i + 1]

            with self._lock:
                edge_key = (source, target)

                # Track feedback
                if edge_key not in self._edge_feedback:
                    self._edge_feedback[edge_key] = []
                self._edge_feedback[edge_key].append(feedback.rating)

                # Update if enough feedback
                if len(self._edge_feedback[edge_key]) >= self.min_feedback_count:
                    update = self._update_edge_weight(source, target)
                    if update:
                        updates.append(update)

        return {
            'feedback_id': feedback.id,
            'edges_updated': len(updates),
            'updates': updates
        }

    def _update_edge_weight(self, source: str, target: str) -> Optional[Dict]:
        """Update single edge weight"""
        edge_key = (source, target)
        ratings = self._edge_feedback[edge_key]

        # Calculate new weight using EMA
        avg_rating = sum(ratings) / len(ratings)

        # Get current weight
        try:
            current = self.graph.get_edge_weight(source, target)
            if current is None:
                current = 0.5
        except (AttributeError, TypeError):
            current = 0.5

        # Calculate new weight
        new_weight = (
            self.decay_factor * current +
            (1 - self.decay_factor) * avg_rating
        )

        # Clamp to [0, 1]
        new_weight = max(0.0, min(1.0, new_weight))

        # Update graph
        try:
            self.graph.update_edge_weight(source, target, new_weight)

            return {
                'source': source,
                'target': target,
                'old_weight': current,
                'new_weight': new_weight,
                'feedback_count': len(ratings)
            }
        except (AttributeError, TypeError) as e:
            logger.warning(f"Could not update edge weight: {e}")
            return None


class FeedbackLoop:
    """
    Main feedback loop controller.

    Coordinates feedback collection, storage, and learning.
    """

    def __init__(
        self,
        graph_engine=None,
        storage_path: Optional[str] = None,
        use_sqlite: bool = True,
        auto_adjust: bool = True,
        learning_rate: float = 0.1
    ):
        self.graph = graph_engine
        self.store = FeedbackStore(storage_path, use_sqlite)
        self.auto_adjust = auto_adjust

        if graph_engine and auto_adjust:
            self.adjuster = WeightAdjuster(
                graph_engine,
                learning_rate=learning_rate
            )
        else:
            self.adjuster = None

        logger.info("FeedbackLoop initialized")

    def record(
        self,
        query: str,
        result_ids: List[str],
        rating: float,
        comment: Optional[str] = None,
        correct_answer: Optional[str] = None,
        feedback_type: FeedbackType = FeedbackType.EXPLICIT,
        metadata: Optional[Dict] = None
    ) -> FeedbackRecord:
        """
        Record user feedback.

        Args:
            query: Original search query
            result_ids: List of result document IDs
            rating: Rating from 0.0 to 1.0
            comment: Optional user comment
            correct_answer: User-provided correct answer
            feedback_type: Type of feedback
            metadata: Additional metadata

        Returns:
            Created FeedbackRecord
        """
        # Create record
        record = FeedbackRecord(
            id=str(uuid.uuid4())[:8],
            timestamp=datetime.utcnow(),
            query=query,
            result_ids=result_ids,
            rating=rating,
            feedback_type=feedback_type,
            comment=comment,
            correct_answer=correct_answer,
            metadata=metadata
        )

        # Store
        self.store.add(record)

        # Auto-adjust weights
        if self.auto_adjust and self.adjuster:
            self.adjuster.process_feedback(record)

        logger.info(f"Recorded feedback: {record.id}, rating={rating}")
        return record

    def get_document_score(self, doc_id: str) -> Dict[str, Any]:
        """
        Get aggregate feedback score for a document.
        """
        feedback_list = self.store.get_by_result_id(doc_id)

        if not feedback_list:
            return {
                'doc_id': doc_id,
                'feedback_count': 0,
                'average_rating': None,
                'trend': None
            }

        ratings = [f.rating for f in feedback_list]

        # Calculate trend (last 5 vs previous 5)
        if len(ratings) >= 10:
            recent = sum(ratings[-5:]) / 5
            previous = sum(ratings[-10:-5]) / 5
            trend = "improving" if recent > previous else "declining" if recent < previous else "stable"
        else:
            trend = "insufficient_data"

        return {
            'doc_id': doc_id,
            'feedback_count': len(ratings),
            'average_rating': sum(ratings) / len(ratings),
            'min_rating': min(ratings),
            'max_rating': max(ratings),
            'trend': trend
        }

    def get_low_quality_results(
        self,
        threshold: float = 0.3,
        min_feedback: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Identify consistently low-rated results.

        Useful for finding documents that need improvement.
        """
        # Get all negative feedback
        negative = self.store.get_recent(limit=1000, max_rating=threshold)

        # Count per document
        doc_counts: Dict[str, List[float]] = {}
        for fb in negative:
            for doc_id in fb.result_ids:
                if doc_id not in doc_counts:
                    doc_counts[doc_id] = []
                doc_counts[doc_id].append(fb.rating)

        # Filter and sort
        low_quality = []
        for doc_id, ratings in doc_counts.items():
            if len(ratings) >= min_feedback:
                low_quality.append({
                    'doc_id': doc_id,
                    'feedback_count': len(ratings),
                    'average_rating': sum(ratings) / len(ratings)
                })

        low_quality.sort(key=lambda x: x['average_rating'])
        return low_quality

    def get_analytics(self) -> Dict[str, Any]:
        """Get feedback analytics dashboard data"""
        stats = self.store.get_stats()
        recent = self.store.get_recent(limit=100)

        # Time-based analytics
        now = datetime.utcnow()
        last_24h = [f for f in recent if f.timestamp > now - timedelta(hours=24)]
        last_7d = [f for f in recent if f.timestamp > now - timedelta(days=7)]

        return {
            'overall': stats,
            'last_24h': {
                'count': len(last_24h),
                'avg_rating': sum(f.rating for f in last_24h) / max(1, len(last_24h))
            },
            'last_7d': {
                'count': len(last_7d),
                'avg_rating': sum(f.rating for f in last_7d) / max(1, len(last_7d))
            },
            'low_quality_docs': self.get_low_quality_results()[:10]
        }


def create_feedback_loop(
    graph_engine=None,
    storage_path: Optional[str] = None,
    **kwargs
) -> FeedbackLoop:
    """
    Factory function to create FeedbackLoop.

    Example:
        >>> from neurocausal_rag.learning import create_feedback_loop
        >>> loop = create_feedback_loop(graph_engine=rag._graph)
        >>> loop.record("climate change", ["doc1", "doc2"], 0.8)
    """
    return FeedbackLoop(
        graph_engine=graph_engine,
        storage_path=storage_path,
        **kwargs
    )
