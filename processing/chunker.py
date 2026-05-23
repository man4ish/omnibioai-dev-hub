MAX_CHARS = 2000


def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        # Hard cap: split oversized chunks at MAX_CHARS boundaries
        while len(chunk) > MAX_CHARS:
            chunks.append(chunk[:MAX_CHARS])
            chunk = chunk[MAX_CHARS:]
        if chunk:
            chunks.append(chunk)

    return chunks