"""
NeuroCausal RAG - Node Data Structure
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass
class NeuroCausalNode:
    """
    NeuroCausal node structure.

    Carries three different embeddings:
    - text_embedding: Semantic vector of text (sentence-transformers)
    - structure_embedding: Graph structural info (from GNN)
    - final_embedding: Fusion of both (used for retrieval)
    """
    id: str
    content: str
    text_embedding: np.ndarray
    structure_embedding: Optional[np.ndarray] = None
    final_embedding: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)
    importance: float = 0.5

    def to_dict(self) -> Dict:
        """Convert node to dictionary"""
        return {
            'id': self.id,
            'content': self.content,
            'text_embedding': self.text_embedding.tolist() if self.text_embedding is not None else None,
            'structure_embedding': self.structure_embedding.tolist() if self.structure_embedding is not None else None,
            'final_embedding': self.final_embedding.tolist() if self.final_embedding is not None else None,
            'metadata': self.metadata,
            'importance': self.importance
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "NeuroCausalNode":
        """Create node from dictionary"""
        return cls(
            id=data['id'],
            content=data['content'],
            text_embedding=np.array(data['text_embedding']) if data.get('text_embedding') else None,
            structure_embedding=np.array(data['structure_embedding']) if data.get('structure_embedding') else None,
            final_embedding=np.array(data['final_embedding']) if data.get('final_embedding') else None,
            metadata=data.get('metadata', {}),
            importance=data.get('importance', 0.5)
        )

    def compute_final_embedding(self) -> np.ndarray:
        """Compute final embedding (text + structure fusion)"""
        if self.structure_embedding is not None:
            self.final_embedding = np.concatenate([self.text_embedding, self.structure_embedding])
        else:
            self.final_embedding = self.text_embedding
        return self.final_embedding
