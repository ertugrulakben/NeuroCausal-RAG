"""
NeuroCausal RAG - Import and Circular Dependency Tests
Verifies all major imports work, __version__ is accessible, __all__ is correct,
and submodule imports do not cause circular dependency errors.

Author: Ertugrul Akben
"""

import pytest
import importlib
import sys


# =========================================================================
# TOP-LEVEL IMPORTS
# =========================================================================

class TestTopLevelImports:
    """Test that the main package imports without errors."""

    def test_import_neurocausal_rag(self):
        import neurocausal_rag
        assert neurocausal_rag is not None

    def test_version_accessible(self):
        from neurocausal_rag import __version__
        assert isinstance(__version__, str)
        assert len(__version__) > 0
        # Should follow semver-like pattern (X.Y.Z)
        parts = __version__.split(".")
        assert len(parts) >= 2

    def test_version_value(self):
        from neurocausal_rag import __version__
        assert __version__ == "6.1.0"

    def test_author_accessible(self):
        import neurocausal_rag
        assert hasattr(neurocausal_rag, "__author__")
        assert neurocausal_rag.__author__ == "Ertugrul Akben"

    def test_email_accessible(self):
        import neurocausal_rag
        assert hasattr(neurocausal_rag, "__email__")
        assert neurocausal_rag.__email__ == "i@ertugrulakben.com"


# =========================================================================
# __all__ EXPORTS
# =========================================================================

class TestAllExports:
    """Test __all__ contains expected public API exports."""

    def test_all_is_defined(self):
        import neurocausal_rag
        assert hasattr(neurocausal_rag, "__all__")
        assert isinstance(neurocausal_rag.__all__, list)

    def test_all_contains_main_class(self):
        from neurocausal_rag import __all__
        assert "NeuroCausalRAG" in __all__

    def test_all_contains_config_exports(self):
        from neurocausal_rag import __all__
        expected_config_exports = [
            "NeuroCausalConfig",
            "get_config",
            "set_config",
            "load_config",
        ]
        for name in expected_config_exports:
            assert name in __all__, f"{name} missing from __all__"

    def test_all_contains_data_classes(self):
        from neurocausal_rag import __all__
        expected_data_classes = [
            "SearchResult",
            "EvaluationResult",
            "EntityResolution",
        ]
        for name in expected_data_classes:
            assert name in __all__, f"{name} missing from __all__"

    def test_all_contains_version(self):
        from neurocausal_rag import __all__
        assert "__version__" in __all__

    def test_all_exports_are_importable(self):
        """Every name in __all__ should be importable from the package."""
        import neurocausal_rag
        for name in neurocausal_rag.__all__:
            obj = getattr(neurocausal_rag, name, None)
            assert obj is not None, f"__all__ lists '{name}' but it is not accessible via getattr"

    def test_all_count(self):
        """__all__ should have a reasonable number of exports."""
        from neurocausal_rag import __all__
        # Currently 9 items, allow some room for growth
        assert len(__all__) >= 8
        assert len(__all__) <= 20


# =========================================================================
# MAIN CLASS IMPORT
# =========================================================================

class TestMainClassImport:
    """Test that the main facade class is importable and constructable."""

    def test_import_neurocausal_rag_class(self):
        from neurocausal_rag import NeuroCausalRAG
        assert NeuroCausalRAG is not None

    def test_neurocausal_rag_class_is_class(self):
        from neurocausal_rag import NeuroCausalRAG
        assert isinstance(NeuroCausalRAG, type)

    def test_neurocausal_rag_instantiation(self):
        """NeuroCausalRAG() should work with default config (lazy init)."""
        from neurocausal_rag import NeuroCausalRAG
        rag = NeuroCausalRAG()
        assert rag is not None
        assert rag.config is not None
        # Should not be fully initialized yet (lazy)
        assert rag._initialized is False


# =========================================================================
# CONFIG MODULE IMPORTS
# =========================================================================

class TestConfigImports:
    """Test config module imports without circular dependency."""

    def test_import_config_module(self):
        from neurocausal_rag import config
        assert config is not None

    def test_import_all_config_classes(self):
        from neurocausal_rag.config import (
            NeuroCausalConfig,
            EmbeddingConfig,
            GraphConfig,
            Neo4jConfig,
            SearchConfig,
            IndexConfig,
            LLMConfig,
            LearningConfig,
        )
        assert NeuroCausalConfig is not None
        assert EmbeddingConfig is not None
        assert GraphConfig is not None
        assert Neo4jConfig is not None
        assert SearchConfig is not None
        assert IndexConfig is not None
        assert LLMConfig is not None
        assert LearningConfig is not None

    def test_import_config_functions(self):
        from neurocausal_rag.config import get_config, set_config, load_config
        assert callable(get_config)
        assert callable(set_config)
        assert callable(load_config)


# =========================================================================
# INTERFACES MODULE IMPORTS
# =========================================================================

class TestInterfacesImports:
    """Test interfaces module imports."""

    def test_import_interfaces_module(self):
        from neurocausal_rag import interfaces
        assert interfaces is not None

    def test_import_data_classes(self):
        from neurocausal_rag.interfaces import SearchResult, EvaluationResult, EntityResolution
        assert SearchResult is not None
        assert EvaluationResult is not None
        assert EntityResolution is not None

    def test_import_abstract_interfaces(self):
        from neurocausal_rag.interfaces import (
            IEmbeddingEngine,
            IGraphEngine,
            IIndexBackend,
            IRetriever,
            ILLMClient,
            ILearningEngine,
        )
        assert IEmbeddingEngine is not None
        assert IGraphEngine is not None
        assert IIndexBackend is not None
        assert IRetriever is not None
        assert ILLMClient is not None
        assert ILearningEngine is not None


# =========================================================================
# SUBMODULE IMPORTS (NO CIRCULAR DEPENDENCY)
# =========================================================================

class TestSubmoduleImports:
    """Test that submodule imports work without circular dependency errors."""

    def test_import_llm_module(self):
        from neurocausal_rag.llm import LLMClient
        assert LLMClient is not None

    def test_import_llm_client_directly(self):
        from neurocausal_rag.llm.client import LLMClient
        assert LLMClient is not None

    def test_import_core_submodule(self):
        """Core submodule should be importable."""
        mod = importlib.import_module("neurocausal_rag.core")
        assert mod is not None

    def test_import_search_submodule(self):
        """Search submodule should be importable."""
        mod = importlib.import_module("neurocausal_rag.search")
        assert mod is not None

    def test_import_embedding_submodule(self):
        """Embedding submodule should be importable."""
        mod = importlib.import_module("neurocausal_rag.embedding")
        assert mod is not None

    def test_import_learning_submodule(self):
        """Learning submodule should be importable."""
        mod = importlib.import_module("neurocausal_rag.learning")
        assert mod is not None

    def test_import_entity_submodule(self):
        """Entity submodule should be importable."""
        mod = importlib.import_module("neurocausal_rag.entity")
        assert mod is not None

    def test_import_reasoning_submodule(self):
        """Reasoning submodule should be importable."""
        mod = importlib.import_module("neurocausal_rag.reasoning")
        assert mod is not None

    def test_import_api_submodule(self):
        """API submodule should be importable."""
        mod = importlib.import_module("neurocausal_rag.api")
        assert mod is not None

    def test_import_agents_submodule(self):
        """Agents submodule should be importable."""
        mod = importlib.import_module("neurocausal_rag.agents")
        assert mod is not None


# =========================================================================
# RE-IMPORT / MODULE RELOAD
# =========================================================================

class TestReimportSafety:
    """Test that re-importing does not cause issues."""

    def test_reimport_main_package(self):
        import neurocausal_rag
        mod = importlib.reload(neurocausal_rag)
        assert mod.__version__ == "6.1.0"

    def test_reimport_config_preserves_classes(self):
        from neurocausal_rag import config
        importlib.reload(config)
        assert hasattr(config, "NeuroCausalConfig")
        assert hasattr(config, "get_config")

    def test_multiple_imports_same_module(self):
        """Importing the same thing twice should return the same objects."""
        from neurocausal_rag import NeuroCausalRAG as RAG1
        from neurocausal_rag import NeuroCausalRAG as RAG2
        assert RAG1 is RAG2
