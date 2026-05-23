import pytest
from unittest.mock import patch, mock_open
from ingestion.doc_loader import load_documents

def test_load_documents():
    with patch("os.path.exists", return_value=True),          patch("builtins.open", mock_open(read_data="content")):
        docs = load_documents(["/repo"])
        assert len(docs) == 1
        assert docs[0]["text"] == "content"
        assert docs[0]["source"] == "/repo/README.md"

def test_load_documents_not_found():
    with patch("os.path.exists", return_value=False):
        docs = load_documents(["/repo"])
        assert len(docs) == 0
