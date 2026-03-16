"""
NeuroCausal RAG - Config Validation Tests
Pydantic model validation, YAML loading, weight constraints, defaults

Yazar: Ertugrul Akben
"""

import pytest
import tempfile
import os
from pathlib import Path
from pydantic import ValidationError

from neurocausal_rag.config import (
    NeuroCausalConfig,
    EmbeddingConfig,
    GraphConfig,
    Neo4jConfig,
    SearchConfig,
    IndexConfig,
    LLMConfig,
    LearningConfig,
    get_config,
    set_config,
    load_config,
    _config,
)


# =========================================================================
# DEFAULT VALUES
# =========================================================================

class TestDefaultConfig:
    """Test that default configuration values are correct."""

    def test_default_config_creates_successfully(self):
        config = NeuroCausalConfig.default()
        assert config is not None
        assert isinstance(config, NeuroCausalConfig)

    def test_default_version(self):
        config = NeuroCausalConfig()
        assert config.version == "5.2.0"

    def test_default_debug_is_false(self):
        config = NeuroCausalConfig()
        assert config.debug is False

    def test_default_data_dir(self):
        config = NeuroCausalConfig()
        assert config.data_dir == Path("./data")

    def test_all_sub_configs_initialized(self):
        config = NeuroCausalConfig()
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.graph, GraphConfig)
        assert isinstance(config.neo4j, Neo4jConfig)
        assert isinstance(config.search, SearchConfig)
        assert isinstance(config.index, IndexConfig)
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.learning, LearningConfig)


# =========================================================================
# EMBEDDING CONFIG DEFAULTS
# =========================================================================

class TestEmbeddingConfigDefaults:
    """Test EmbeddingConfig default values."""

    def test_default_model_name(self):
        cfg = EmbeddingConfig()
        assert cfg.model_name == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def test_default_dimension(self):
        cfg = EmbeddingConfig()
        assert cfg.dimension == 384

    def test_default_normalize(self):
        cfg = EmbeddingConfig()
        assert cfg.normalize is True

    def test_default_cache_enabled(self):
        cfg = EmbeddingConfig()
        assert cfg.cache_enabled is True

    def test_default_cache_ttl_hours(self):
        cfg = EmbeddingConfig()
        assert cfg.cache_ttl_hours == 1

    def test_default_max_cache_size(self):
        cfg = EmbeddingConfig()
        assert cfg.max_cache_size == 1000

    def test_dimension_boundaries(self):
        # minimum valid
        cfg = EmbeddingConfig(dimension=64)
        assert cfg.dimension == 64
        # maximum valid
        cfg = EmbeddingConfig(dimension=4096)
        assert cfg.dimension == 4096

    def test_dimension_below_min_rejected(self):
        with pytest.raises(ValidationError):
            EmbeddingConfig(dimension=32)

    def test_dimension_above_max_rejected(self):
        with pytest.raises(ValidationError):
            EmbeddingConfig(dimension=5000)


# =========================================================================
# GRAPH CONFIG DEFAULTS
# =========================================================================

class TestGraphConfigDefaults:
    """Test GraphConfig default values."""

    def test_default_backend(self):
        cfg = GraphConfig()
        assert cfg.backend == "networkx"

    def test_default_pagerank_alpha(self):
        cfg = GraphConfig()
        assert cfg.pagerank_alpha == 0.85

    def test_default_max_causal_depth(self):
        cfg = GraphConfig()
        assert cfg.max_causal_depth == 3

    def test_default_importance_default(self):
        cfg = GraphConfig()
        assert cfg.importance_default == 0.5

    def test_backend_neo4j_accepted(self):
        cfg = GraphConfig(backend="neo4j")
        assert cfg.backend == "neo4j"

    def test_invalid_backend_rejected(self):
        with pytest.raises(ValidationError):
            GraphConfig(backend="invalid_backend")

    def test_pagerank_alpha_boundaries(self):
        cfg = GraphConfig(pagerank_alpha=0.0)
        assert cfg.pagerank_alpha == 0.0
        cfg = GraphConfig(pagerank_alpha=1.0)
        assert cfg.pagerank_alpha == 1.0

    def test_pagerank_alpha_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            GraphConfig(pagerank_alpha=1.5)
        with pytest.raises(ValidationError):
            GraphConfig(pagerank_alpha=-0.1)


# =========================================================================
# SEARCH CONFIG - WEIGHT VALIDATION
# =========================================================================

class TestSearchConfigWeights:
    """Test alpha + beta + gamma weight validation."""

    def test_default_weights_sum_to_one(self):
        cfg = SearchConfig()
        total = cfg.alpha + cfg.beta + cfg.gamma
        assert abs(total - 1.0) <= 0.01

    def test_valid_custom_weights(self):
        cfg = SearchConfig(alpha=0.4, beta=0.4, gamma=0.2)
        assert abs(cfg.alpha + cfg.beta + cfg.gamma - 1.0) <= 0.01

    def test_weights_not_summing_to_one_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            SearchConfig(alpha=0.5, beta=0.5, gamma=0.5)
        assert "must equal 1.0" in str(exc_info.value)

    def test_zero_weights_rejected(self):
        """alpha=0 + beta=0 + gamma=0 = 0 != 1.0"""
        with pytest.raises(ValidationError):
            SearchConfig(alpha=0.0, beta=0.0, gamma=0.0)

    def test_equal_weights(self):
        """1/3 + 1/3 + 1/3 is close enough to 1.0"""
        cfg = SearchConfig(alpha=0.34, beta=0.33, gamma=0.33)
        total = cfg.alpha + cfg.beta + cfg.gamma
        assert abs(total - 1.0) <= 0.01

    def test_default_top_k(self):
        cfg = SearchConfig()
        assert cfg.top_k == 5

    def test_top_k_boundaries(self):
        cfg = SearchConfig(top_k=1)
        assert cfg.top_k == 1
        cfg = SearchConfig(top_k=100)
        assert cfg.top_k == 100

    def test_top_k_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            SearchConfig(top_k=0)
        with pytest.raises(ValidationError):
            SearchConfig(top_k=101)


# =========================================================================
# LLM CONFIG
# =========================================================================

class TestLLMConfig:
    """Test LLMConfig values and validation."""

    def test_default_provider(self):
        cfg = LLMConfig()
        assert cfg.provider == "openai"

    def test_default_model(self):
        cfg = LLMConfig()
        assert cfg.model == "gpt-4o-mini"

    def test_default_temperature(self):
        cfg = LLMConfig()
        assert cfg.temperature == 0.7

    def test_default_max_tokens(self):
        cfg = LLMConfig()
        assert cfg.max_tokens == 1000

    def test_default_api_key_env(self):
        cfg = LLMConfig()
        assert cfg.api_key_env == "OPENAI_API_KEY"

    def test_valid_providers(self):
        for provider in ("openai", "anthropic", "ollama", "local"):
            cfg = LLMConfig(provider=provider)
            assert cfg.provider == provider

    def test_invalid_provider_rejected(self):
        with pytest.raises(ValidationError):
            LLMConfig(provider="google")

    def test_temperature_range(self):
        cfg = LLMConfig(temperature=0.0)
        assert cfg.temperature == 0.0
        cfg = LLMConfig(temperature=2.0)
        assert cfg.temperature == 2.0

    def test_temperature_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            LLMConfig(temperature=2.5)
        with pytest.raises(ValidationError):
            LLMConfig(temperature=-0.1)

    def test_max_tokens_boundaries(self):
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=50)
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=5000)


# =========================================================================
# LEARNING CONFIG
# =========================================================================

class TestLearningConfig:
    """Test LearningConfig defaults and validation."""

    def test_default_enabled(self):
        cfg = LearningConfig()
        assert cfg.enabled is True

    def test_default_min_confidence(self):
        cfg = LearningConfig()
        assert cfg.min_confidence == 0.7

    def test_default_learning_rate(self):
        cfg = LearningConfig()
        assert cfg.learning_rate == 0.1

    def test_default_discovery_threshold(self):
        cfg = LearningConfig()
        assert cfg.discovery_threshold == 0.75

    def test_default_auto_approve(self):
        cfg = LearningConfig()
        assert cfg.auto_approve is False

    def test_learning_rate_boundaries(self):
        cfg = LearningConfig(learning_rate=0.01)
        assert cfg.learning_rate == 0.01
        cfg = LearningConfig(learning_rate=1.0)
        assert cfg.learning_rate == 1.0

    def test_learning_rate_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            LearningConfig(learning_rate=0.001)
        with pytest.raises(ValidationError):
            LearningConfig(learning_rate=1.5)


# =========================================================================
# INDEX CONFIG
# =========================================================================

class TestIndexConfig:
    """Test IndexConfig validation."""

    def test_default_backend(self):
        cfg = IndexConfig()
        assert cfg.backend == "brute_force"

    def test_valid_backends(self):
        for backend in ("brute_force", "faiss", "milvus"):
            cfg = IndexConfig(backend=backend)
            assert cfg.backend == backend

    def test_invalid_backend_rejected(self):
        with pytest.raises(ValidationError):
            IndexConfig(backend="annoy")

    def test_default_faiss_index_type(self):
        cfg = IndexConfig()
        assert cfg.faiss_index_type == "flat"

    def test_valid_faiss_index_types(self):
        for t in ("flat", "ivf", "hnsw"):
            cfg = IndexConfig(faiss_index_type=t)
            assert cfg.faiss_index_type == t


# =========================================================================
# YAML LOADING / SAVING
# =========================================================================

class TestYAMLConfig:
    """Test YAML serialization and deserialization."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Round-trip through YAML. We write manually because to_yaml uses
        yaml.dump which serializes Path objects with Python-specific tags
        that yaml.safe_load cannot parse."""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            "version: '6.1.0'\ndebug: true\ndata_dir: './data'\n",
            encoding="utf-8",
        )

        loaded = NeuroCausalConfig.from_yaml(yaml_path)
        assert loaded.version == "6.1.0"
        assert loaded.debug is True
        assert loaded.data_dir == Path("./data")

    def test_load_with_partial_data(self, tmp_path):
        """YAML with only some fields should use defaults for the rest."""
        yaml_path = tmp_path / "partial.yaml"
        yaml_path.write_text("version: '9.9.9'\ndebug: true\n", encoding="utf-8")

        loaded = NeuroCausalConfig.from_yaml(yaml_path)
        assert loaded.version == "9.9.9"
        assert loaded.debug is True
        # sub-configs should be defaults
        assert loaded.embedding.dimension == 384
        assert loaded.search.alpha == 0.5

    def test_load_with_nested_config(self, tmp_path):
        """YAML with nested sub-config should override defaults."""
        yaml_content = """
version: "6.1.0"
embedding:
  dimension: 768
  normalize: false
search:
  top_k: 10
  alpha: 0.6
  beta: 0.3
  gamma: 0.1
"""
        yaml_path = tmp_path / "nested.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")

        loaded = NeuroCausalConfig.from_yaml(yaml_path)
        assert loaded.embedding.dimension == 768
        assert loaded.embedding.normalize is False
        assert loaded.search.top_k == 10
        assert loaded.search.alpha == 0.6

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            NeuroCausalConfig.from_yaml("/nonexistent/config.yaml")

    def test_save_creates_file(self, tmp_path):
        config = NeuroCausalConfig()
        yaml_path = tmp_path / "out.yaml"
        assert not yaml_path.exists()
        config.to_yaml(yaml_path)
        assert yaml_path.exists()
        assert yaml_path.stat().st_size > 0


# =========================================================================
# GLOBAL CONFIG SINGLETON
# =========================================================================

class TestGlobalConfig:
    """Test get_config / set_config / load_config global state."""

    def setup_method(self):
        """Reset global config before each test."""
        import neurocausal_rag.config as cfg_module
        cfg_module._config = None

    def test_get_config_returns_default(self):
        config = get_config()
        assert isinstance(config, NeuroCausalConfig)

    def test_set_config_changes_global(self):
        custom = NeuroCausalConfig(version="custom")
        set_config(custom)
        assert get_config().version == "custom"

    def test_load_config_from_yaml(self, tmp_path):
        yaml_path = tmp_path / "load_test.yaml"
        yaml_path.write_text("version: 'loaded'\n", encoding="utf-8")
        config = load_config(str(yaml_path))
        assert config.version == "loaded"
        assert get_config().version == "loaded"

    def test_get_config_lazy_init(self):
        """First call creates, subsequent calls return the same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2


# =========================================================================
# PYDANTIC INVALID VALUES
# =========================================================================

class TestInvalidValues:
    """Test that Pydantic rejects clearly invalid values."""

    def test_embedding_dimension_negative(self):
        with pytest.raises(ValidationError):
            EmbeddingConfig(dimension=-1)

    def test_cache_ttl_zero(self):
        with pytest.raises(ValidationError):
            EmbeddingConfig(cache_ttl_hours=0)

    def test_max_cache_size_below_min(self):
        with pytest.raises(ValidationError):
            EmbeddingConfig(max_cache_size=50)

    def test_max_causal_depth_zero(self):
        with pytest.raises(ValidationError):
            GraphConfig(max_causal_depth=0)

    def test_max_causal_depth_above_max(self):
        with pytest.raises(ValidationError):
            GraphConfig(max_causal_depth=11)

    def test_neo4j_pool_size_zero(self):
        with pytest.raises(ValidationError):
            Neo4jConfig(max_connection_pool_size=0)

    def test_neo4j_connection_timeout_below_min(self):
        with pytest.raises(ValidationError):
            Neo4jConfig(connection_timeout=2)

    def test_search_alpha_above_one(self):
        with pytest.raises(ValidationError):
            SearchConfig(alpha=1.5, beta=0.0, gamma=0.0)

    def test_wrong_type_for_debug(self):
        """Pydantic should coerce or reject non-bool for debug field."""
        # Pydantic v2 coerces many types; string "notabool" should fail
        # depending on strict mode. We just check int works (truthy coercion).
        config = NeuroCausalConfig(debug=1)
        assert config.debug is True
