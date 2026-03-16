"""
NeuroCausal RAG - Index Backends
Strategy Pattern implementation for vector search
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import pickle
import logging

from ..config import IndexConfig
from ..interfaces import IIndexBackend

logger = logging.getLogger(__name__)


class BruteForceIndex(IIndexBackend):
    """
    Brute Force Index Backend.
    Simple but accurate - computes all pairwise similarities.
    Best for small datasets (< 10K vectors).
    """

    def __init__(self):
        self.embeddings: Optional[np.ndarray] = None
        self.ids: List[str] = []
        self._id_to_idx: Dict[str, int] = {}

    def build(self, embeddings: np.ndarray, ids: List[str]) -> None:
        """Build index from embeddings"""
        self.embeddings = embeddings
        self.ids = ids
        self._id_to_idx = {id_: idx for idx, id_ in enumerate(ids)}
        logger.info(f"BruteForce index built with {len(ids)} vectors")

    def add(self, embedding: np.ndarray, node_id: str) -> None:
        """Add single embedding to index"""
        if self.embeddings is None:
            self.embeddings = embedding.reshape(1, -1)
            self.ids = [node_id]
        else:
            self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)])
            self.ids.append(node_id)
        self._id_to_idx[node_id] = len(self.ids) - 1

    def search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors"""
        if self.embeddings is None or len(self.ids) == 0:
            return []

        # Compute all similarities
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        emb_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-9)
        similarities = np.dot(emb_norms, query_norm)

        # Get top-k indices
        k = min(k, len(self.ids))
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append((self.ids[idx], float(similarities[idx])))

        return results

    def remove(self, node_id: str) -> bool:
        """Remove node from index"""
        if node_id not in self._id_to_idx:
            return False

        idx = self._id_to_idx[node_id]
        self.embeddings = np.delete(self.embeddings, idx, axis=0)
        self.ids.pop(idx)

        # Rebuild id mapping
        self._id_to_idx = {id_: i for i, id_ in enumerate(self.ids)}
        return True

    def save(self, path: str) -> None:
        """Save index to disk"""
        data = {
            'embeddings': self.embeddings,
            'ids': self.ids
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Index saved to {path}")

    def load(self, path: str) -> None:
        """Load index from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.embeddings = data['embeddings']
        self.ids = data['ids']
        self._id_to_idx = {id_: idx for idx, id_ in enumerate(self.ids)}
        logger.info(f"Index loaded from {path}: {len(self.ids)} vectors")

    @property
    def size(self) -> int:
        """Return number of vectors in index"""
        return len(self.ids)


class FAISSIndex(IIndexBackend):
    """
    FAISS Index Backend.
    Fast approximate nearest neighbor search.
    Supports Flat, IVF, and HNSW index types.
    """

    def __init__(self, config: IndexConfig):
        self.config = config
        self.index = None
        self.ids: List[str] = []
        self._id_to_idx: Dict[str, int] = {}
        self._dimension: Optional[int] = None

    def _create_index(self, dimension: int) -> None:
        """Create FAISS index based on config"""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu or faiss-gpu required for FAISSIndex")

        self._dimension = dimension

        if self.config.faiss_index_type == "flat":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine for normalized)
        elif self.config.faiss_index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, dimension,
                self.config.faiss_nlist,
                faiss.METRIC_INNER_PRODUCT
            )
        elif self.config.faiss_index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(dimension)

        logger.info(f"Created FAISS {self.config.faiss_index_type} index")

    def build(self, embeddings: np.ndarray, ids: List[str]) -> None:
        """Build index from embeddings"""
        if len(embeddings) == 0:
            return

        self._create_index(embeddings.shape[1])
        self.ids = ids
        self._id_to_idx = {id_: idx for idx, id_ in enumerate(ids)}

        # Normalize for cosine similarity
        faiss_embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        faiss_embeddings = faiss_embeddings.astype(np.float32)

        # Train if needed (IVF)
        if hasattr(self.index, 'train') and not self.index.is_trained:
            self.index.train(faiss_embeddings)

        self.index.add(faiss_embeddings)
        logger.info(f"FAISS index built with {len(ids)} vectors")

    def add(self, embedding: np.ndarray, node_id: str) -> None:
        """Add single embedding to index"""
        if self.index is None:
            self._create_index(len(embedding))

        # Normalize
        normalized = embedding / (np.linalg.norm(embedding) + 1e-9)
        normalized = normalized.reshape(1, -1).astype(np.float32)

        self.index.add(normalized)
        self.ids.append(node_id)
        self._id_to_idx[node_id] = len(self.ids) - 1

    def search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors"""
        if self.index is None or self.index.ntotal == 0:
            return []

        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        query_norm = query_norm.reshape(1, -1).astype(np.float32)

        # Set nprobe for IVF
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.config.faiss_nprobe

        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_norm, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.ids):
                results.append((self.ids[idx], float(score)))

        return results

    def remove(self, node_id: str) -> bool:
        """Remove node from index (rebuild required for FAISS)"""
        if node_id not in self._id_to_idx:
            return False

        # FAISS doesn't support efficient removal
        # Mark as removed and rebuild periodically
        logger.warning("FAISS removal requires index rebuild")
        return False

    def save(self, path: str) -> None:
        """Save index to disk"""
        import faiss
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.ids", 'wb') as f:
            pickle.dump(self.ids, f)
        logger.info(f"FAISS index saved to {path}")

    def load(self, path: str) -> None:
        """Load index from disk"""
        import faiss
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.ids", 'rb') as f:
            self.ids = pickle.load(f)
        self._id_to_idx = {id_: idx for idx, id_ in enumerate(self.ids)}
        logger.info(f"FAISS index loaded from {path}: {len(self.ids)} vectors")

    @property
    def size(self) -> int:
        """Return number of vectors in index"""
        return len(self.ids)


class MilvusIndex(IIndexBackend):
    """
    Milvus Index Backend.
    Distributed vector database for production-scale deployments.
    Supports billions of vectors with automatic sharding.

    Requires: pymilvus>=2.3.0
    Docker: docker-compose up milvus
    """

    def __init__(self, config: IndexConfig):
        self.config = config
        self.collection = None
        self.ids: List[str] = []
        self._id_to_idx: Dict[str, int] = {}
        self._connected = False
        self._collection_name = config.milvus_collection or "neurocausal_vectors"
        self._dimension: Optional[int] = None

    def _connect(self) -> None:
        """Connect to Milvus server"""
        if self._connected:
            return

        try:
            from pymilvus import connections, utility
        except ImportError:
            raise ImportError("pymilvus>=2.3.0 required for MilvusIndex. Install: pip install pymilvus")

        host = self.config.milvus_host or "localhost"
        port = self.config.milvus_port or 19530

        connections.connect(
            alias="default",
            host=host,
            port=port
        )
        self._connected = True
        logger.info(f"Connected to Milvus at {host}:{port}")

    def _create_collection(self, dimension: int) -> None:
        """Create Milvus collection with schema"""
        from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility

        self._dimension = dimension

        # Check if collection exists
        if utility.has_collection(self._collection_name):
            self.collection = Collection(self._collection_name)
            logger.info(f"Using existing Milvus collection: {self._collection_name}")
            return

        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
        ]
        schema = CollectionSchema(fields=fields, description="NeuroCausal RAG vectors")

        # Create collection
        self.collection = Collection(
            name=self._collection_name,
            schema=schema
        )

        # Create index for fast search
        index_params = {
            "metric_type": "IP",  # Inner Product (cosine for normalized)
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        logger.info(f"Created Milvus collection: {self._collection_name} with dimension {dimension}")

    def build(self, embeddings: np.ndarray, ids: List[str]) -> None:
        """Build index from embeddings"""
        if len(embeddings) == 0:
            return

        self._connect()
        self._create_collection(embeddings.shape[1])

        # Normalize embeddings for cosine similarity
        normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        normalized = normalized.astype(np.float32)

        # Insert data
        entities = [
            ids,  # node_id field
            normalized.tolist()  # embedding field
        ]

        self.collection.insert([entities[0], entities[1]])
        self.collection.flush()
        self.collection.load()

        self.ids = ids
        self._id_to_idx = {id_: idx for idx, id_ in enumerate(ids)}
        logger.info(f"Milvus index built with {len(ids)} vectors")

    def add(self, embedding: np.ndarray, node_id: str) -> None:
        """Add single embedding to index"""
        self._connect()

        if self.collection is None:
            self._create_collection(len(embedding))

        # Normalize
        normalized = embedding / (np.linalg.norm(embedding) + 1e-9)
        normalized = normalized.astype(np.float32)

        # Insert
        self.collection.insert([[node_id], [normalized.tolist()]])
        self.collection.flush()

        self.ids.append(node_id)
        self._id_to_idx[node_id] = len(self.ids) - 1

    def search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors"""
        if not self._connected or self.collection is None:
            return []

        # Ensure collection is loaded
        self.collection.load()

        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        query_norm = query_norm.reshape(1, -1).astype(np.float32)

        # Search
        search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
        results = self.collection.search(
            data=query_norm.tolist(),
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["node_id"]
        )

        # Format results
        output = []
        for hits in results:
            for hit in hits:
                node_id = hit.entity.get("node_id")
                score = float(hit.distance)
                output.append((node_id, score))

        return output

    def remove(self, node_id: str) -> bool:
        """Remove node from index"""
        if not self._connected or self.collection is None:
            return False

        try:
            # Delete by node_id
            expr = f'node_id == "{node_id}"'
            self.collection.delete(expr)
            self.collection.flush()

            if node_id in self._id_to_idx:
                del self._id_to_idx[node_id]
                self.ids.remove(node_id)

            return True
        except Exception as e:
            logger.error(f"Failed to remove from Milvus: {e}")
            return False

    def save(self, path: str) -> None:
        """Save ID mapping to disk (Milvus persists automatically)"""
        data = {'ids': self.ids, 'collection': self._collection_name}
        with open(f"{path}.milvus", 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Milvus ID mapping saved to {path}")

    def load(self, path: str) -> None:
        """Load ID mapping and connect to existing collection"""
        with open(f"{path}.milvus", 'rb') as f:
            data = pickle.load(f)

        self.ids = data['ids']
        self._collection_name = data['collection']
        self._id_to_idx = {id_: idx for idx, id_ in enumerate(self.ids)}

        self._connect()
        from pymilvus import Collection
        self.collection = Collection(self._collection_name)
        self.collection.load()
        logger.info(f"Milvus collection loaded: {self._collection_name} with {len(self.ids)} vectors")

    @property
    def size(self) -> int:
        """Return number of vectors in index"""
        return len(self.ids)

    def drop_collection(self) -> None:
        """Drop the entire collection (for cleanup)"""
        if self.collection:
            from pymilvus import utility
            utility.drop_collection(self._collection_name)
            self.collection = None
            self.ids = []
            self._id_to_idx = {}
            logger.info(f"Dropped Milvus collection: {self._collection_name}")


def create_index_backend(config: IndexConfig) -> IIndexBackend:
    """Factory function to create index backend based on config"""
    if config.backend == "faiss":
        return FAISSIndex(config)
    elif config.backend == "milvus":
        return MilvusIndex(config)
    else:
        return BruteForceIndex()
