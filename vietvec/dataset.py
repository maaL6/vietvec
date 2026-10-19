"""Vectorized Skip-gram data pipeline: corpus encoding, pair generation, negative sampling.

The whole corpus lives in memory (GPU if available) as a flat array of word IDs plus
one sentence ID per token. Training pairs are generated chunk by chunk with tensor ops,
so there is no Python loop over tokens.
"""

from typing import Iterator, Tuple

import numpy as np
import torch

from vietvec.preprocessing import Vocabulary


def encode_corpus(corpus_path: str, vocab: Vocabulary, out_path: str) -> None:
    """Encode a tokenized text file (one sentence per line) into word IDs.

    Saves an .npz with `ids` (int32, all tokens concatenated, OOV words dropped)
    and `lengths` (int32, number of kept tokens per sentence; sentences with < 2 in-vocab
    tokens are dropped).
    """
    from array import array

    w2i = vocab.word2idx
    ids, lengths = array("i"), array("i")
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            sent = [w2i[w] for w in line.split() if w in w2i]
            if len(sent) >= 2:
                ids.extend(sent)
                lengths.append(len(sent))
    np.savez(out_path, ids=np.frombuffer(ids, dtype=np.int32), lengths=np.frombuffer(lengths, dtype=np.int32))


class NegativeSampler:
    """Unigram^0.75 sampler using Mikolov's precomputed lookup table."""

    def __init__(self, word_counts, table_size: int = 10_000_000, power: float = 0.75, device="cpu"):
        probs = np.power(np.asarray(word_counts, dtype=np.float64), power)
        cum_probs = np.cumsum(probs / probs.sum())
        cum_probs[-1] = 1.0
        table = np.searchsorted(cum_probs, np.linspace(0, 1, table_size, endpoint=False))
        self.table = torch.from_numpy(table).to(device)

    def sample(self, num_samples: int, num_negatives: int = 5) -> torch.Tensor:
        """Return a LongTensor of shape (num_samples, num_negatives)."""
        idx = torch.randint(0, len(self.table), (num_samples, num_negatives), device=self.table.device)
        return self.table[idx]


class SkipGramBatcher:
    """Generates shuffled (center, context, negatives) mini-batches for one epoch.

    Per epoch: Mikolov subsampling on all tokens, then for each chunk of centers a random
    window b ~ U{1..max_window}; every neighbour at distance <= b inside the same sentence
    becomes a positive pair.
    """

    def __init__(
        self,
        ids: np.ndarray,
        lengths: np.ndarray,
        vocab: Vocabulary,
        batch_size: int = 32768,
        max_window: int = 5,
        num_negatives: int = 5,
        chunk_tokens: int = 2_000_000,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.ids = torch.from_numpy(ids).to(self.device)
        lengths_t = torch.from_numpy(lengths).long().to(self.device)
        self.sent = torch.repeat_interleave(torch.arange(len(lengths_t), device=self.device, dtype=torch.int32), lengths_t)
        self.keep_probs = torch.from_numpy(vocab.keep_probs).float().to(self.device)
        self.sampler = NegativeSampler(vocab.word_counts, device=self.device)
        self.batch_size = batch_size
        self.max_window = max_window
        self.num_negatives = num_negatives
        self.chunk_tokens = chunk_tokens

    def estimate_batches(self) -> int:
        """Expected batches per epoch (for the LR schedule), from one subsampling draw."""
        keep = torch.rand(len(self.ids), device=self.device) < self.keep_probs[self.ids.long()]
        sent, W = self.sent[keep], self.max_window
        # Neighbours at distance d in the same sentence, each used with P(b >= d) = (W - d + 1) / W
        pairs = sum(2 * (sent[d:] == sent[:-d]).sum().item() * (W - d + 1) / W for d in range(1, W + 1))
        n_chunks = (len(sent) + self.chunk_tokens - 1) // self.chunk_tokens
        return int(pairs / self.batch_size) + n_chunks

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        dev = self.device
        keep = torch.rand(len(self.ids), device=dev) < self.keep_probs[self.ids.long()]
        ids, sent = self.ids[keep].long(), self.sent[keep]
        n = len(ids)

        starts = torch.arange(0, n, self.chunk_tokens)
        for s in starts[torch.randperm(len(starts))].tolist():
            pos = torch.arange(s, min(s + self.chunk_tokens, n), device=dev)
            b = torch.randint(1, self.max_window + 1, (len(pos),), device=dev)
            centers, contexts = [], []
            for d in range(-self.max_window, self.max_window + 1):
                if d == 0:
                    continue
                raw = pos + d
                ctx = raw.clamp(0, n - 1)  # safe index; out-of-range positions are masked below
                ok = (b >= abs(d)) & (raw >= 0) & (raw < n) & (sent[ctx] == sent[pos])
                centers.append(ids[pos[ok]])
                contexts.append(ids[ctx[ok]])
            centers, contexts = torch.cat(centers), torch.cat(contexts)

            perm = torch.randperm(len(centers), device=dev)
            for i in range(0, len(perm), self.batch_size):
                j = perm[i : i + self.batch_size]
                yield centers[j], contexts[j], self.sampler.sample(len(j), self.num_negatives)
