from processing.chunker import chunk_text

def test_chunk_text():
    text = "a b c d e f"
    chunks = chunk_text(text, chunk_size=2)
    assert len(chunks) == 3
    assert chunks[0] == "a b"
    assert chunks[1] == "c d"
    assert chunks[2] == "e f"
