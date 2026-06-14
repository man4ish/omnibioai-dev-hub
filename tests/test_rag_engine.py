import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from rag.engine import RAGEngine, ollama_embed, ollama_generate, cosine

# =========================================================
# UNIT TESTS FOR ollama_embed
# =========================================================

@patch("rag.engine.requests.post")
def test_ollama_embed_success(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1] * 768}
    mock_post.return_value = mock_response
    
    vec = ollama_embed("test text")
    assert vec.shape == (768,)
    assert vec.dtype == np.float32
    mock_post.assert_called_once()

@patch("rag.engine.requests.post")
def test_ollama_embed_batch_dim(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [[0.1] * 768]}
    mock_post.return_value = mock_response
    
    vec = ollama_embed("test text")
    assert vec.shape == (768,)

@patch("rag.engine.requests.post")
def test_ollama_embed_dim_mismatch(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1] * 512}
    mock_post.return_value = mock_response
    
    with pytest.raises(ValueError, match="Embedding dim mismatch"):
        ollama_embed("test text")

# =========================================================
# UNIT TESTS FOR ollama_generate
# =========================================================

@patch("rag.engine.requests.post")
def test_ollama_generate_success(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "generated answer"}
    mock_post.return_value = mock_response
    
    resp = ollama_generate("prompt")
    assert resp == "generated answer"

# =========================================================
# UNIT TESTS FOR cosine
# =========================================================

def test_cosine():
    a = [1, 0]
    b = [1, 0]
    assert cosine(a, b) == 1.0
    
    c = [0, 1]
    assert cosine(a, c) == 0.0
    
    d = [0, 0]
    assert cosine(a, d) == 0.0

# =========================================================
# RAGEngine TESTS
# =========================================================

@pytest.fixture
def mock_vector_store():
    return MagicMock()

@pytest.fixture
def engine(mock_vector_store):
    return RAGEngine(mock_vector_store)

def test_engine_init(engine, mock_vector_store):
    assert engine.vector_store == mock_vector_store
    assert engine.embed_model == "nomic-embed-text"

@patch("rag.engine.ollama_embed")
def test_engine_embed_success(mock_embed, engine):
    mock_embed.return_value = np.array([0.1] * 768, dtype=np.float32)
    vec = engine._embed("query")
    assert vec.shape == (768,)

@patch("rag.engine.ollama_embed")
def test_engine_embed_invalid_type(mock_embed, engine):
    with pytest.raises(ValueError, match="Query must be a string"):
        engine._embed(123)

@patch("rag.engine.ollama_embed")
def test_engine_embed_dim_mismatch(mock_embed, engine):
    mock_embed.return_value = np.array([0.1] * 512, dtype=np.float32)
    with pytest.raises(ValueError, match="Query embedding mismatch"):
        engine._embed("query")

@patch("rag.engine.ollama_embed")
def test_engine_retrieve_empty(mock_embed, engine, mock_vector_store):
    mock_embed.return_value = np.array([0.1] * 768, dtype=np.float32)
    mock_vector_store.index = None
    
    results = engine.retrieve("query")
    assert results == []

@patch("rag.engine.ollama_embed")
def test_engine_retrieve_success(mock_embed, engine, mock_vector_store):
    mock_embed.return_value = np.array([0.1] * 768, dtype=np.float32)
    
    mock_index = MagicMock()
    mock_index.ntotal = 10
    mock_index.search.return_value = (
        np.array([[0.9, 0.8]]), # scores
        np.array([[1, 2]])      # indices
    )
    mock_vector_store.index = mock_index
    mock_vector_store.metadata = [
        {}, # 0
        {"text": "text1", "source": "src1"}, # 1
        {"text": "text2", "source": "src2"}  # 2
    ]
    
    results = engine.retrieve("query", top_k=2)
    assert len(results) == 2
    assert results[0]["text"] == "text1"
    assert results[1]["text"] == "text2"

@patch("rag.engine.ollama_embed")
def test_engine_retrieve_invalid_indices(mock_embed, engine, mock_vector_store):
    mock_embed.return_value = np.array([0.1] * 768, dtype=np.float32)
    
    mock_index = MagicMock()
    mock_index.ntotal = 1
    mock_index.search.return_value = (
        np.array([[0.9]]), # scores
        np.array([[-1]])   # invalid index
    )
    mock_vector_store.index = mock_index
    mock_vector_store.metadata = [{"text": "text1"}]
    
    results = engine.retrieve("query")
    assert results == []

def test_build_context(engine):
    docs = [
        {"text": "t1", "source": "s1"},
        {"text": "t2", "source": "s2"}
    ]
    ctx = engine.build_context(docs)
    assert "[s1]\nt1" in ctx
    assert "[s2]\nt2" in ctx

def test_build_context_empty(engine):
    assert engine.build_context([]) == "No relevant context found."

def test_build_prompt(engine):
    prompt = engine.build_prompt("q", "ctx")
    assert "ctx" in prompt
    assert "q" in prompt

@patch("rag.engine.ollama_generate")
def test_engine_answer_success(mock_gen, engine, mock_vector_store):
    mock_gen.return_value = "final answer"
    
    # Mock retrieve to return something
    with patch.object(engine, "retrieve", return_value=[{"source": "s1"}]) as mock_retrieve:
        res = engine.answer("query")
        assert res["answer"] == "final answer"
        assert res["sources"] == ["s1"]
        assert res["version"] == "v6-faiss"

@patch("rag.engine.ollama_generate")
def test_engine_answer_failure(mock_gen, engine, mock_vector_store):
    mock_gen.side_effect = Exception("Gen failed")
    
    with patch.object(engine, "retrieve", return_value=[]):
        res = engine.answer("query")
        assert "[LLM_ERROR] Gen failed" in res["answer"]

def test_engine_query(engine):
    with patch.object(engine, "answer", return_value={"ok": True}) as mock_answer:
        res = engine.query("q")
        assert res["ok"] is True
        mock_answer.assert_called_once_with("q", repo=None, bundle=None)


def test_engine_query_passes_scope(engine):
    with patch.object(engine, "answer", return_value={"ok": True}) as mock_answer:
        engine.query("q", repo="my-repo", bundle="my-bundle")
        mock_answer.assert_called_once_with("q", repo="my-repo", bundle="my-bundle")


@patch("rag.engine.ollama_embed")
def test_engine_retrieve_with_bundle_filter(mock_embed, engine, mock_vector_store):
    mock_embed.return_value = np.array([0.1] * 768, dtype=np.float32)

    mock_index = MagicMock()
    mock_index.ntotal = 2
    mock_vector_store.index = mock_index
    mock_vector_store.metadata = []

    # filter_search is present on the mock
    mock_vector_store.filter_search.return_value = [
        {"score": 0.9, "text": "filtered", "source": "s1", "repo": "r", "bundle": "b"}
    ]

    results = engine.retrieve("query", top_k=5, bundle="b")
    mock_vector_store.filter_search.assert_called_once()
    assert results[0]["text"] == "filtered"


@patch("rag.engine.ollama_embed")
def test_engine_retrieve_with_repo_filter(mock_embed, engine, mock_vector_store):
    mock_embed.return_value = np.array([0.1] * 768, dtype=np.float32)

    mock_index = MagicMock()
    mock_index.ntotal = 1
    mock_vector_store.index = mock_index
    mock_vector_store.metadata = []
    mock_vector_store.filter_search.return_value = [
        {"score": 0.8, "text": "repo-result", "source": "s2", "repo": "my-repo", "bundle": None}
    ]

    results = engine.retrieve("query", repo="my-repo")
    # Should call filter_search with field="repo"
    call_kwargs = mock_vector_store.filter_search.call_args
    assert call_kwargs[1].get("field") == "repo" or call_kwargs[0][2] == "repo"
    assert results[0]["text"] == "repo-result"


@patch("rag.engine.ollama_embed")
def test_engine_retrieve_bundle_takes_priority_over_repo(mock_embed, engine, mock_vector_store):
    mock_embed.return_value = np.array([0.1] * 768, dtype=np.float32)

    mock_index = MagicMock()
    mock_index.ntotal = 1
    mock_vector_store.index = mock_index
    mock_vector_store.metadata = []
    mock_vector_store.filter_search.return_value = []

    engine.retrieve("query", repo="r", bundle="b")
    call_kwargs = mock_vector_store.filter_search.call_args
    # bundle takes priority — field should be "bundle"
    args = call_kwargs[0]
    kwargs = call_kwargs[1]
    field_used = kwargs.get("field") or (args[2] if len(args) > 2 else None)
    assert field_used == "bundle"


@patch("rag.engine.ollama_embed")
def test_engine_answer_passes_scope(mock_embed, engine, mock_vector_store):
    mock_embed.return_value = np.array([0.1] * 768, dtype=np.float32)

    with patch.object(engine, "retrieve", return_value=[]) as mock_retrieve:
        with patch("rag.engine.ollama_generate", return_value="ans"):
            engine.answer("q", repo="r", bundle="b")
            mock_retrieve.assert_called_once_with("q", repo="r", bundle="b")
