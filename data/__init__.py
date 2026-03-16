"""
NeuroCausal RAG - Data Module
Built-in datasets for testing and demonstration
"""

from .climate_knowledge_base import (
    get_documents,
    get_documents_by_category,
    get_document_count,
    DOCUMENTS
)

__all__ = [
    "get_documents",
    "get_documents_by_category",
    "get_document_count",
    "DOCUMENTS"
]
