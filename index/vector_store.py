import pickle
import os
import numpy as np


class VectorStore:
    def __init__(self):
        self.vectors = []
        self.metadata = []

    # -----------------------------
    # ADD DATA
    # -----------------------------
    def add(self, vectors, metadata):
        self.vectors.extend(vectors)
        self.metadata.extend(metadata)

    # -----------------------------
    # SEARCH (optional use later)
    # -----------------------------
    def search(self, query_vector, top_k=5):
        scores = []

        for i, vec in enumerate(self.vectors):
            score = self.cosine_similarity(query_vector, vec)
            scores.append((score, self.metadata[i]))

        scores.sort(reverse=True, key=lambda x: x[0])
        return scores[:top_k]

    # -----------------------------
    # COSINE SIMILARITY
    # -----------------------------
    def cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # -----------------------------
    # SAVE INDEX
    # -----------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({
                "vectors": self.vectors,
                "metadata": self.metadata
            }, f)

    # -----------------------------
    # LOAD INDEX
    # -----------------------------
    def load(self, path: str):
        if not os.path.exists(path):
            return False

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.vectors = data["vectors"]
        self.metadata = data["metadata"]

        return True