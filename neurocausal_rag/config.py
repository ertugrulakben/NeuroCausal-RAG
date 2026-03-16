"""
NeuroCausal RAG - Configuration Management
Pydantic-based type-safe configuration

Author: Ertugrul Akben
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from pathlib import Path
import yaml
import os


class EmbeddingConfig(BaseModel):
    """Embedding model configuration"""
    model_name: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Sentence transformer model name"
    )
    dimension: int = Field(default=384, ge=64, le=4096)
    normalize: bool = Field(default=True)
    cache_enabled: bool = Field(default=True)
    cache_ttl_hours: int = Field(default=1, ge=1, le=24)
    max_cache_size: int = Field(default=1000, ge=100, le=10000)


class GraphConfig(BaseModel):
    """Graph engine configuration"""
    backend: Literal["networkx", "neo4j"] = Field(
        default="networkx",
        description="Graph backend type"
    )
    pagerank_alpha: float = Field(default=0.85, ge=0.0, le=1.0)
    max_causal_depth: int = Field(default=3, ge=1, le=10)
    importance_default: float = Field(default=0.5, ge=0.0, le=1.0)


class Neo4jConfig(BaseModel):
    """Neo4j graph database configuration"""
    uri: str = Field(
        default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        description="Neo4j connection URI"
    )
    user: str = Field(
        default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"),
        description="Neo4j username"
    )
    password: str = Field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "neurocausal123"),
        description="Neo4j password"
    )
    database: str = Field(default="neo4j", description="Neo4j database name")
    max_connection_pool_size: int = Field(default=50, ge=1, le=100)
    connection_timeout: int = Field(default=30, ge=5, le=120)


class SearchConfig(BaseModel):
    """Search and retrieval configuration"""
    top_k: int = Field(default=5, ge=1, le=100)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Similarity weight")
    beta: float = Field(default=0.3, ge=0.0, le=1.0, description="Causal weight")
    gamma: float = Field(default=0.2, ge=0.0, le=1.0, description="Importance weight")

    @field_validator('gamma')
    @classmethod
    def validate_weights_sum(cls, v, info):
        """Ensure alpha + beta + gamma = 1.0"""
        alpha = info.data.get('alpha', 0.5)
        beta = info.data.get('beta', 0.3)
        total = alpha + beta + v
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"alpha + beta + gamma must equal 1.0, got {total}")
        return v


class IndexConfig(BaseModel):
    """Index backend configuration"""
    backend: Literal["brute_force", "faiss", "milvus"] = Field(
        default="brute_force",
        description="Index backend type"
    )
    faiss_index_type: Literal["flat", "ivf", "hnsw"] = Field(
        default="flat",
        description="FAISS index type"
    )
    faiss_nlist: int = Field(default=100, ge=1, description="Number of clusters for IVF")
    faiss_nprobe: int = Field(default=10, ge=1, description="Number of clusters to search")
    milvus_host: str = Field(default="localhost")
    milvus_port: int = Field(default=19530)
    milvus_collection: str = Field(default="neurocausal")


class LLMConfig(BaseModel):
    """LLM configuration"""
    provider: Literal["openai", "anthropic", "ollama", "local"] = Field(
        default="openai"
    )
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=100, le=4096)
    api_key_env: str = Field(
        default="OPENAI_API_KEY",
        description="Environment variable name for API key"
    )


class LearningConfig(BaseModel):
    """Self-learning engine configuration"""
    enabled: bool = Field(default=True)
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    learning_rate: float = Field(default=0.1, ge=0.01, le=1.0)
    discovery_threshold: float = Field(default=0.75, ge=0.5, le=1.0)
    auto_approve: bool = Field(default=False)


class NeuroCausalConfig(BaseModel):
    """Main configuration container"""
    version: str = Field(default="6.1.0")
    debug: bool = Field(default=False)
    data_dir: Path = Field(default=Path("./data"))

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NeuroCausalConfig":
        """Load configuration from YAML file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def default(cls) -> "NeuroCausalConfig":
        """Create default configuration"""
        return cls()


# Global config singleton
_config: Optional[NeuroCausalConfig] = None


def get_config() -> NeuroCausalConfig:
    """Get global configuration (lazy initialization)"""
    global _config
    if _config is None:
        _config = NeuroCausalConfig.default()
    return _config


def set_config(config: NeuroCausalConfig) -> None:
    """Set global configuration"""
    global _config
    _config = config


def load_config(path: str | Path) -> NeuroCausalConfig:
    """Load and set configuration from file"""
    config = NeuroCausalConfig.from_yaml(path)
    set_config(config)
    return config
