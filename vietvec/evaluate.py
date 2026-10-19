"""Query and evaluate trained embeddings: nearest neighbours, analogies, word-similarity."""

from typing import List, Tuple

import numpy as np
import torch


class Embeddings:
    """Read-only word vectors loaded from a file written by `trainer.save_embeddings`."""

    def __init__(self, path: str):
        data = torch.load(path, map_location="cpu")
        self.words: List[str] = data["words"]
        self.vectors: np.ndarray = data["embeddings"].numpy()
        self.word2idx = {w: i for i, w in enumerate(self.words)}
        self.unit = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True).clip(1e-8)

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx

    def __getitem__(self, word: str) -> np.ndarray:
        return self.vectors[self.word2idx[word]]

    def similarity(self, w1: str, w2: str) -> float:
        return float(self.unit[self.word2idx[w1]] @ self.unit[self.word2idx[w2]])

    def _nearest(self, query: np.ndarray, exclude: List[str], k: int) -> List[Tuple[str, float]]:
        scores = self.unit @ (query / np.linalg.norm(query))
        for w in exclude:
            scores[self.word2idx[w]] = -np.inf
        top = np.argpartition(-scores, k)[:k]
        return [(self.words[i], float(scores[i])) for i in top[np.argsort(-scores[top])]]

    def most_similar(self, word: str, k: int = 10) -> List[Tuple[str, float]]:
        return self._nearest(self.unit[self.word2idx[word]], [word], k)

    def analogy(self, a: str, b: str, c: str, k: int = 5) -> List[Tuple[str, float]]:
        """a : b = c : ?  (vector b - a + c), e.g. ông : bà = vua : nữ_hoàng."""
        u = self.unit
        query = u[self.word2idx[b]] - u[self.word2idx[a]] + u[self.word2idx[c]]
        return self._nearest(query, [a, b, c], k)

    def evaluate_analogies(self, quads: List[Tuple[str, str, str, str]]) -> float:
        """Top-1 accuracy over (a, b, c, d) quadruples whose words are all in vocab."""
        valid = [q for q in quads if all(w in self for w in q)]
        hits = sum(self.analogy(a, b, c, k=1)[0][0] == d for a, b, c, d in valid)
        return hits / max(len(valid), 1)

    def evaluate_similarity(self, pairs: List[Tuple[str, str, float]]) -> Tuple[float, int]:
        """Spearman correlation between cosine similarity and human scores."""
        from scipy.stats import spearmanr

        valid = [(a, b, s) for a, b, s in pairs if a in self and b in self]
        rho = spearmanr([self.similarity(a, b) for a, b, _ in valid], [s for *_, s in valid]).correlation
        return float(rho), len(valid)

    def save_word2vec_format(self, path: str) -> None:
        """Export to the text .vec format (Gensim KeyedVectors.load_word2vec_format)."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{len(self.words)} {self.vectors.shape[1]}\n")
            for w, v in zip(self.words, self.vectors):
                f.write(w + " " + " ".join(f"{x:.5f}" for x in v) + "\n")
