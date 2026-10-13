"""Data pipeline and Negative Sampling for Word2Vec (Skip-gram)."""

import os
import random
from typing import Iterator, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

from vietvec.preprocessing import Vocabulary


class NegativeSampler:
    """Fast unigram table negative sampler following Mikolov's paper.

    Draws negative samples from the distribution:
        P_n(w) proportional to count(w)^0.75

    Uses a precomputed flat integer lookup table (default size 10,000,000 entries)
    to achieve sub-millisecond batch sampling.
    """

    def __init__(
        self,
        word_counts: Union[List[int], np.ndarray],
        table_size: int = 10_000_000,
        power: float = 0.75,
    ):
        self.vocab_size = len(word_counts)
        self.table_size = table_size

        counts = np.array(word_counts, dtype=np.float64)
        powered_counts = np.power(counts, power)
        norm_constant = np.sum(powered_counts)
        probs = powered_counts / norm_constant

        # Precompute cumulative probabilities
        cum_probs = np.cumsum(probs)
        cum_probs[-1] = 1.0  # Safeguard numerical rounding

        # Vectorized lookup table creation using searchsorted
        points = np.linspace(0, 1, table_size, endpoint=False)
        self.table = np.searchsorted(cum_probs, points).astype(np.int64)

    def sample(self, num_samples: int, num_negatives: int = 5) -> torch.Tensor:
        """Sample negative word IDs for a batch.

        Args:
            num_samples: Number of items in batch (B).
            num_negatives: Number of negative samples per item (K).

        Returns:
            torch.Tensor: LongTensor of shape (B, K).
        """
        random_indices = np.random.randint(0, self.table_size, size=(num_samples, num_negatives))
        sampled_ids = self.table[random_indices]
        return torch.from_numpy(sampled_ids)


class Word2VecStreamDataset(IterableDataset):
    """Streaming IterableDataset for large tokenized corpora.

    Reads the tokenized text file line-by-line without loading the entire
    corpus into RAM. Applies:
        1. Token encoding via Vocabulary
        2. Mikolov subsampling of frequent words
        3. Dynamic context window (random c from 1 to max_window)
        4. Generation of (center, context) pairs
        5. Negative sampling on-the-fly for each batch
    """

    def __init__(
        self,
        corpus_path: str,
        vocab: Vocabulary,
        negative_sampler: NegativeSampler,
        batch_size: int = 4096,
        max_window: int = 5,
        num_negatives: int = 5,
        apply_subsampling: bool = True,
        max_lines: Optional[int] = None,
    ):
        super().__init__()
        self.corpus_path = corpus_path
        self.vocab = vocab
        self.negative_sampler = negative_sampler
        self.batch_size = batch_size
        self.max_window = max_window
        self.num_negatives = num_negatives
        self.apply_subsampling = apply_subsampling
        self.max_lines = max_lines

    def _line_iterator(self) -> Iterator[str]:
        """Iterate through the lines of the corpus file, handling worker splitting."""
        worker_info = torch.utils.data.get_worker_info()

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if self.max_lines is not None and line_idx >= self.max_lines:
                    break

                # If multi-worker DataLoader, shard lines across workers
                if worker_info is not None:
                    if line_idx % worker_info.num_workers != worker_info.id:
                        continue

                line = line.strip()
                if line:
                    yield line

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield mini-batches of (center_words, context_words, negative_words)."""
        center_buffer: List[int] = []
        context_buffer: List[int] = []

        for line in self._line_iterator():
            words = line.split()
            if len(words) < 2:
                continue

            # 1. Map to IDs
            token_ids = self.vocab.encode(words)

            # 2. Subsample frequent words
            if self.apply_subsampling:
                token_ids = self.vocab.subsample(token_ids)

            if len(token_ids) < 2:
                continue

            # 3. Dynamic context window & pair generation
            seq_len = len(token_ids)
            for i, center_id in enumerate(token_ids):
                # Mikolov: random window size from 1 to max_window
                window_size = random.randint(1, self.max_window)

                start_idx = max(0, i - window_size)
                end_idx = min(seq_len, i + window_size + 1)

                for j in range(start_idx, end_idx):
                    if i == j:
                        continue
                    context_id = token_ids[j]

                    center_buffer.append(center_id)
                    context_buffer.append(context_id)

                    # When buffer reaches batch_size, emit batch
                    if len(center_buffer) >= self.batch_size:
                        center_tensor = torch.tensor(center_buffer, dtype=torch.long)
                        context_tensor = torch.tensor(context_buffer, dtype=torch.long)
                        neg_tensor = self.negative_sampler.sample(
                            num_samples=len(center_buffer),
                            num_negatives=self.num_negatives,
                        )

                        yield center_tensor, context_tensor, neg_tensor

                        center_buffer = []
                        context_buffer = []

        # Emit remaining items if any
        if center_buffer:
            center_tensor = torch.tensor(center_buffer, dtype=torch.long)
            context_tensor = torch.tensor(context_buffer, dtype=torch.long)
            neg_tensor = self.negative_sampler.sample(
                num_samples=len(center_buffer),
                num_negatives=self.num_negatives,
            )
            yield center_tensor, context_tensor, neg_tensor


def get_dataloader(
    corpus_path: str,
    vocab: Vocabulary,
    negative_sampler: Optional[NegativeSampler] = None,
    batch_size: int = 4096,
    max_window: int = 5,
    num_negatives: int = 5,
    apply_subsampling: bool = True,
    num_workers: int = 0,
    max_lines: Optional[int] = None,
) -> DataLoader:
    """Factory helper to instantiate a DataLoader for Word2Vec training.

    Args:
        corpus_path: Path to tokenized text file.
        vocab: Vocabulary instance.
        negative_sampler: Optional NegativeSampler (created automatically if None).
        batch_size: Number of (center, context) pairs per batch.
        max_window: Maximum context window radius.
        num_negatives: Number of negative samples K.
        apply_subsampling: Whether to apply Mikolov subsampling.
        num_workers: Number of DataLoader worker processes.
        max_lines: Optional limit on lines read (useful for testing).

    Returns:
        DataLoader yielding (center_words, context_words, negative_words).
    """
    if negative_sampler is None:
        negative_sampler = NegativeSampler(word_counts=vocab.word_counts)

    dataset = Word2VecStreamDataset(
        corpus_path=corpus_path,
        vocab=vocab,
        negative_sampler=negative_sampler,
        batch_size=batch_size,
        max_window=max_window,
        num_negatives=num_negatives,
        apply_subsampling=apply_subsampling,
        max_lines=max_lines,
    )

    # Note: batch_size=None because dataset already yields pre-batched tensors
    return DataLoader(dataset, batch_size=None, num_workers=num_workers)


if __name__ == "__main__":
    import time
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    vocab_file = "data/processed/vocab.json"
    corpus_file = "data/processed/wiki_vi_tokenized.txt"

    if not os.path.exists(vocab_file) or not os.path.exists(corpus_file):
        print("Vui lòng đảm bảo vocab.json và wiki_vi_tokenized.txt đã được tạo.")
        exit(1)

    print("--- 1. Tải Vocabulary và Khởi tạo Negative Sampler ---")
    t0 = time.time()
    vocab = Vocabulary.load(vocab_file)
    print(f"Đã nạp vocab: {len(vocab):,} từ ({time.time() - t0:.2f}s)")

    t0 = time.time()
    sampler = NegativeSampler(word_counts=vocab.word_counts)
    print(f"Đã tạo NegativeSampler Unigram Table: {sampler.table_size:,} ô ({time.time() - t0:.2f}s)")

    print("\n--- 2. Kiểm tra lấy mẫu âm (Negative Sampling) ---")
    neg_test = sampler.sample(num_samples=3, num_negatives=5)
    print(f"Tensor shape (B=3, K=5): {neg_test.shape}")
    for i in range(3):
        neg_words = [vocab.idx2word[idx.item()] for idx in neg_test[i]]
        print(f"Mẫu âm dòng {i+1}: {neg_words}")

    print("\n--- 3. Kiểm tra Streaming DataLoader (Thử 3 batch đầu) ---")
    loader = get_dataloader(
        corpus_path=corpus_file,
        vocab=vocab,
        negative_sampler=sampler,
        batch_size=2048,
        max_window=5,
        num_negatives=5,
        max_lines=1000,
    )

    for b_idx, (center, context, neg) in enumerate(loader):
        print(f"\n[Batch {b_idx + 1}]")
        print(f"- Center words shape:  {center.shape} (dtype: {center.dtype})")
        print(f"- Context words shape: {context.shape} (dtype: {context.dtype})")
        print(f"- Negative words shape: {neg.shape} (dtype: {neg.dtype})")

        # In 3 cặp mẫu đầu tiên để kiểm chứng
        print("  Ví dụ 3 cặp huấn luyện trong batch:")
        for k in range(3):
            c_word = vocab.idx2word[center[k].item()]
            ctx_word = vocab.idx2word[context[k].item()]
            n_words = [vocab.idx2word[idx.item()] for idx in neg[k]]
            print(f"    ({k+1}) Center: '{c_word}'  <->  Target Context: '{ctx_word}'")
            print(f"        Negative samples: {n_words}")

        if b_idx >= 2:
            break

    print("\n🎉 Kiểm tra Bước 2.1 & 2.2 THÀNH CÔNG RỰC RỠ!")
