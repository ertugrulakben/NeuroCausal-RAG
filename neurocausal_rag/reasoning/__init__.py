"""
NeuroCausal RAG - Reasoning Module
Contradiction Detection and Temporal Reasoning

Author: Ertugrul Akben
"""

from .contradiction import ContradictionDetector
from .temporal import TemporalEngine

__all__ = [
    'ContradictionDetector',
    'TemporalEngine'
]
