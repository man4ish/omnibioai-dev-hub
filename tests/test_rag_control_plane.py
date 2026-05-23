import pytest
import time
from unittest.mock import MagicMock, patch
from rag.control_plane import ControlPlane, ControlPlaneState

@pytest.fixture
def cp():
    return ControlPlane()

def test_control_plane_init_state(cp):
    assert cp.state.status == "INIT"
    assert not cp._initialized
    assert cp.engine is None

def test_control_plane_init_success(cp):
    mock_vs = MagicMock()
    mock_gs = MagicMock()
    mock_pi = MagicMock()
    mock_emb = MagicMock()
    
    # Patch at the source since it's a local import
    with patch("rag.engine.RAGEngine") as mock_engine_cls:
        cp.init(mock_vs, mock_gs, mock_pi, mock_emb)
        
        assert cp.state.status == "READY"
        assert cp._initialized
        assert cp.vector_store == mock_vs
        assert cp.graph_store == mock_gs
        assert cp.plugin_index == mock_pi
        assert cp.embedder == mock_emb
        assert cp.engine is not None
        mock_engine_cls.assert_called_once_with(mock_vs)

def test_control_plane_reinit(cp):
    mock_vs = MagicMock()
    with patch("rag.engine.RAGEngine"):
        cp.init(mock_vs, None, None, None)
        status1 = cp.state.status
        assert status1 == "READY"
        
        # Re-init should return early
        cp.init(MagicMock(), None, None, None)
        assert cp.vector_store == mock_vs # Still the same

def test_control_plane_init_failure(cp):
    mock_vs = MagicMock()
    with patch("rag.engine.RAGEngine") as mock_engine_cls:
        mock_engine_cls.side_effect = Exception("Engine failed")
        
        with pytest.raises(Exception, match="Engine failed"):
            cp.init(mock_vs, None, None, None)
        
        assert cp.state.status == "FAILED"
        assert cp.state.last_error == "Engine failed"

def test_build_graph_seed(cp):
    mock_gs = MagicMock()
    cp.graph_store = mock_gs
    cp._build_graph_seed()
    assert mock_gs.add_edge.call_count == 3

def test_build_plugin_seed(cp):
    mock_pi = MagicMock()
    mock_pi.docs = None
    cp.plugin_index = mock_pi
    cp._build_plugin_seed()
    assert len(mock_pi.docs) == 3

def test_build_plugin_seed_existing(cp):
    mock_pi = MagicMock()
    mock_pi.docs = [{"text": "existing"}]
    cp.plugin_index = mock_pi
    cp._build_plugin_seed()
    assert len(mock_pi.docs) == 4

def test_status(cp):
    status = cp.status()
    assert status["status"] == "INIT"
    assert status["initialized"] is False
    assert "uptime_sec" in status

def test_ensure_ready(cp):
    with pytest.raises(RuntimeError, match="Control plane not ready"):
        cp.ensure_ready()
    
    cp.state.status = "READY"
    cp.ensure_ready() # Should not raise

def test_get_engine(cp):
    with pytest.raises(RuntimeError, match="RAG engine not initialized"):
        cp.get_engine()
    
    mock_engine = MagicMock()
    cp.engine = mock_engine
    assert cp.get_engine() == mock_engine
