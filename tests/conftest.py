import pytest
from unittest.mock import patch

@pytest.fixture(autouse=True)
def mock_model_trained_global():
    with patch('src.microstructure._model_trained', True):
        yield
