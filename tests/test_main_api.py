import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
import sys

# Mock heavy dependencies properly
mock_faiss = MagicMock()
mock_faiss.__spec__ = MagicMock()
sys.modules['faiss'] = mock_faiss

mock_st = MagicMock()
mock_st.__spec__ = MagicMock()
sys.modules['sentence_transformers'] = mock_st

# Mock dependencies at their source to avoid issues during import of api.main
with patch("index.vector_store.VectorStore"),      patch("index.graph_store.GraphStore"),      patch("index.plugin_index.PluginIndex"):
    from api.main import app

client = TestClient(app)

def test_health_endpoint():
    with patch("api.main.CONTROL_PLANE.status", return_value={"status": "READY"}):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

def test_status_endpoint():
    with patch("api.main.CONTROL_PLANE.status", return_value={"status": "READY"}):
        with patch("api.main.graph_store") as mock_gs:
            mock_gs.size.return_value = {"nodes": 3, "edges": 4}
            response = client.get("/status")
            assert response.status_code == 200
            assert response.json()["graph_edges"] == 4

@pytest.mark.asyncio
async def test_wait_for_ollama_success():
    from api.main import wait_for_ollama
    with patch("api.main.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        res = await wait_for_ollama(retries=1)
        assert res is True

@pytest.mark.asyncio
async def test_wait_for_ollama_failure():
    from api.main import wait_for_ollama
    with patch("api.main.requests.get") as mock_get:
        mock_get.side_effect = Exception("Fail")
        res = await wait_for_ollama(retries=1, delay=0)
        assert res is False

@pytest.mark.asyncio
async def test_build_index_success():
    from api.main import build_index
    with patch("api.main.wait_for_ollama", return_value=True),          patch("api.main.load_documents", return_value=[{"text": "t1", "source": "s1"}]),          patch("api.main.chunk_text", return_value=["c1"]),          patch("api.main.ollama_embed", return_value=[0.1]*768),          patch("api.main.vector_store.add") as mock_add:
        
        await build_index()
        mock_add.assert_called_once()

@pytest.mark.asyncio
async def test_build_index_no_ollama():
    from api.main import build_index
    with patch("api.main.wait_for_ollama", return_value=False):
        await build_index()
        from api.main import indexing_status
        assert indexing_status["errors"] == -1

def test_guard_requests_middleware_ready():
    with patch("api.main.CONTROL_PLANE.status", return_value={"status": "READY"}):
        with patch("api.routes.rag.get_engine"):
            response = client.post("/rag/query", json={"query": "q"})
            assert response.status_code != 503

def test_guard_requests_middleware_not_ready():
    with patch("api.main.CONTROL_PLANE.status", return_value={"status": "INIT"}):
        response = client.post("/rag/query", json={"query": "q"})
        assert response.status_code == 503
        assert response.json()["detail"] == "Control plane not ready"

def test_build_graph_seed():
    from api.main import build_graph_seed, graph_store
    mock_gs = MagicMock()
    with patch("api.main.graph_store", mock_gs):
        build_graph_seed()
        assert mock_gs.add_edge.call_count == 4

def test_build_plugin_index():
    from api.main import build_plugin_index, plugin_index
    mock_pi = MagicMock()
    with patch("api.main.plugin_index", mock_pi):
        build_plugin_index()
        assert len(mock_pi.docs) == 3

@pytest.mark.asyncio
async def test_init_control_plane():
    from api.main import init_control_plane
    with patch("api.main.build_graph_seed"),          patch("api.main.build_plugin_index"),          patch("api.main.CONTROL_PLANE.init"),          patch("api.main.asyncio.create_task"):
        await init_control_plane()

@pytest.mark.asyncio
async def test_index_in_background_error():
    from api.main import _index_in_background
    with patch("api.main.build_index", side_effect=Exception("bg error")):
        await _index_in_background() # Should catch error and log it

@pytest.mark.asyncio
async def test_build_index_empty_doc():
    from api.main import build_index, indexing_status
    with patch("api.main.wait_for_ollama", return_value=True),          patch("api.main.load_documents", return_value=[{"text": "", "source": "s1"}]):
        await build_index()
        assert indexing_status["docs"] == 0

@pytest.mark.asyncio
async def test_build_index_empty_chunk():
    from api.main import build_index, indexing_status
    with patch("api.main.wait_for_ollama", return_value=True),          patch("api.main.load_documents", return_value=[{"text": "t1", "source": "s1"}]),          patch("api.main.chunk_text", return_value=[" "]),          patch("api.main.ollama_embed", return_value=[0.1]*768):
        await build_index()
        assert indexing_status["chunks"] == 1 # it was one chunk " "
        # but line 135: if not chunk.strip(): continue

@pytest.mark.asyncio
async def test_build_index_exception():
    from api.main import build_index, indexing_status
    with patch("api.main.wait_for_ollama", return_value=True),          patch("api.main.load_documents", return_value=[{"text": "t1", "source": "s1"}]),          patch("api.main.chunk_text", side_effect=Exception("Error")):
        await build_index()
        assert indexing_status["errors"] == 1

@pytest.mark.asyncio
async def test_build_index_no_vectors():
    from api.main import build_index
    with patch("api.main.wait_for_ollama", return_value=True),          patch("api.main.load_documents", return_value=[]),          patch("api.main.logger.warning") as mock_warn:
        await build_index()
        mock_warn.assert_any_call("build_index: no vectors collected — skipping VectorStore.add()")

@pytest.mark.asyncio
async def test_startup_event():
    from api.main import startup_event
    with patch("api.main.init_control_plane", new_callable=AsyncMock) as mock_init:
        await startup_event()
        mock_init.assert_called_once()
