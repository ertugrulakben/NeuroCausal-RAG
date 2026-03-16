import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_embedding_model():
    model = MagicMock()
    # Mock encode to return a fixed vector
    model.encode.return_value = [0.1] * 768
    return model

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    # Mock generate/predict response
    llm.generate.return_value = "This is a mocked LLM response."
    return llm

@pytest.fixture
def mock_graph_engine():
    engine = MagicMock()
    # Mock graph operations if needed
    return engine
