import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys

# Mock faiss
mock_faiss = MagicMock()
mock_faiss.IndexFlatIP = MagicMock()
sys.modules['faiss'] = mock_faiss

from index.vector_store import VectorStore

@pytest.fixture
def vs():
    return VectorStore()

def test_init_index_success(vs):
    vs._init_index(768)
    assert vs.dim == 768
    mock_faiss.IndexFlatIP.assert_called_with(768)

def test_init_index_failure(vs):
    with pytest.raises(ValueError, match="Only nomic-embed-text"):
        vs._init_index(512)

def test_coerce_shape(vs):
    # 1D
    v1 = [0.1] * 768
    c1 = vs._coerce_shape(v1)
    assert c1.shape == (1, 768)
    
    # 2D
    v2 = [[0.1] * 768]
    c2 = vs._coerce_shape(v2)
    assert c2.shape == (1, 768)
    
    # 3D
    v3 = [[[0.1] * 768]]
    c3 = vs._coerce_shape(v3)
    assert c3.shape == (1, 768)
    
    # Invalid
    with pytest.raises(ValueError, match="Cannot coerce"):
        vs._coerce_shape(np.zeros((1, 1, 1, 768)))

def test_add_success(vs):
    mock_index = MagicMock()
    mock_faiss.IndexFlatIP.return_value = mock_index
    
    vs.add([[0.1] * 768], [{"text": "t1"}])
    assert vs.dim == 768
    mock_index.add.assert_called_once()
    assert len(vs.metadata) == 1

def test_add_dim_mismatch(vs):
    vs.dim = 768
    vs.index = MagicMock()
    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        vs.add([[0.1] * 512], [])

def test_search_empty(vs):
    assert vs.search([0.1]*768) == []

def test_search_success(vs):
    vs.dim = 768
    mock_index = MagicMock()
    mock_index.ntotal = 1
    mock_index.search.return_value = (np.array([[0.9]]), np.array([[0]]))
    vs.index = mock_index
    vs.metadata = [{"text": "t1", "source": "s1"}]
    
    res = vs.search([0.1]*768)
    assert len(res) == 1
    assert res[0]["text"] == "t1"

def test_search_dim_mismatch(vs):
    vs.dim = 768
    vs.index = MagicMock()
    vs.index.ntotal = 1
    with pytest.raises(ValueError, match="Query dimension mismatch"):
        vs.search([0.1]*512)

def test_search_invalid_indices(vs):
    vs.dim = 768
    mock_index = MagicMock()
    mock_index.ntotal = 1
    mock_index.search.return_value = (np.array([[0.9]]), np.array([[-1]]))
    vs.index = mock_index
    vs.metadata = []
    
    assert vs.search([0.1]*768) == []
