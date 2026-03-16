"""
NeuroCausal RAG - Edge Data Structure
"""

from dataclasses import dataclass
from typing import Dict
from enum import Enum


class RelationType(Enum):
    """Causal relationship types"""
    CAUSES = "causes"              # A -> B (cause-effect)
    REQUIRES = "requires"          # A |- B (prerequisite)
    CONTRADICTS = "contradicts"    # A _|_ B (contradiction)
    SUPPORTS = "supports"          # A => B (supporting)
    RELATED = "related"            # A <-> B (related)


RELATION_TYPE_TO_IDX = {
    RelationType.CAUSES: 0,
    RelationType.REQUIRES: 1,
    RelationType.CONTRADICTS: 2,
    RelationType.SUPPORTS: 3,
    RelationType.RELATED: 4
}


@dataclass
class NeuroCausalEdge:
    """Causal graph edge"""
    source: str
    target: str
    relation_type: RelationType
    strength: float = 1.0
    evidence: str = ""

    def to_dict(self) -> Dict:
        """Convert edge to dictionary"""
        return {
            'source': self.source,
            'target': self.target,
            'relation_type': self.relation_type.value,
            'strength': self.strength,
            'evidence': self.evidence
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "NeuroCausalEdge":
        """Create edge from dictionary"""
        rel_type = data['relation_type']
        if isinstance(rel_type, str):
            rel_type = RelationType(rel_type)
        return cls(
            source=data['source'],
            target=data['target'],
            relation_type=rel_type,
            strength=data.get('strength', 1.0),
            evidence=data.get('evidence', '')
        )
