"""Preprocessing module for Vietnamese text using underthesea."""

import json
import os
import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from underthesea import word_tokenize


URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_REGEX = re.compile(r"<.*?>")
TOKEN_REGEX = re.compile(r"^\w+$")


def clean_and_tokenize(
    sentence: str,
    replace_numbers: bool = False,
    min_word_len: int = 1,
) -> List[str]:
    """Normalize and tokenize a Vietnamese sentence using Underthesea.

    Steps:
        1. NFC unicode normalization
        2. Lowercasing
        3. Remove URLs and HTML tags
        4. Word segmentation with underthesea (format="text" joins compounds with '_')
        5. Filter out punctuation, symbols, and keep valid word tokens

    Args:
        sentence: Raw input sentence.
        replace_numbers: If True, replace pure number tokens with '<num>'.
        min_word_len: Minimum character length for a token to be kept.

    Returns:
        List[str]: List of segmented word tokens (e.g., ['hà_nội', 'là', 'thủ_đô']).
    """
    if not sentence or not isinstance(sentence, str):
        return []

    # 1. Unicode NFC & lowercase
    text = unicodedata.normalize("NFC", sentence.strip().lower())

    # 2. Remove URLs & HTML tags
    text = URL_REGEX.sub(" ", text)
    text = HTML_REGEX.sub(" ", text)

    # 3. Word tokenize with underthesea
    try:
        tokenized_text = word_tokenize(text, format="text")
    except Exception:
        # Fallback to whitespace split if tokenizer fails on malformed input
        tokenized_text = text

    # 4. Filter and clean individual tokens
    raw_tokens = tokenized_text.split()
    clean_tokens: List[str] = []

    for token in raw_tokens:
        # Strip remaining punctuation characters attached to words
        token = token.strip(".,;:!?\"'()[]{}«»“”„…-–—/")
        if not token or len(token) < min_word_len:
            continue

        if replace_numbers and token.replace(".", "").replace(",", "").isdigit():
            clean_tokens.append("<num>")
            continue

        # Keep valid alphanumeric and compound word tokens
        if TOKEN_REGEX.match(token):
            clean_tokens.append(token)

    return clean_tokens


class Vocabulary:
    """Vocabulary manager with Subsampling support for Word2Vec."""

    def __init__(self, min_count: int = 5, subsample_threshold: float = 1e-4):
        self.min_count = min_count
        self.subsample_threshold = subsample_threshold

        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_counts: List[int] = []
        self.total_words: int = 0
        self.keep_probs: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.word2idx)

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx

    def build_vocab(self, sentences_tokens: Iterable[List[str]]) -> None:
        """Count token frequencies and build vocabulary filtering by min_count.

        Args:
            sentences_tokens: Iterable of tokenized sentences.
        """
        counter: Counter = Counter()
        for tokens in sentences_tokens:
            counter.update(tokens)

        # Filter words with frequency >= min_count
        valid_words = [
            (word, count)
            for word, count in counter.most_common()
            if count >= self.min_count
        ]

        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = []

        for idx, (word, count) in enumerate(valid_words):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.word_counts.append(count)

        self.total_words = sum(self.word_counts)
        self._calculate_subsampling_probs()

    def _calculate_subsampling_probs(self) -> None:
        """Precompute discard/keep probabilities for Mikolov's subsampling technique.

        Formula:
            P(keep w) = (sqrt(z(w) / t) + 1) * (t / z(w))
            where z(w) = count(w) / total_words, t = subsample_threshold
        """
        if self.total_words == 0 or len(self.word_counts) == 0:
            self.keep_probs = None
            return

        counts = np.array(self.word_counts, dtype=np.float64)
        z = counts / self.total_words
        t = self.subsample_threshold

        # Keep probability formula
        keep_prob = (np.sqrt(z / t) + 1.0) * (t / z)
        # Probabilities are capped at 1.0
        self.keep_probs = np.clip(keep_prob, 0.0, 1.0)

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert a list of word tokens to word IDs (skipping out-of-vocab words)."""
        return [self.word2idx[w] for w in tokens if w in self.word2idx]

    def decode(self, indices: List[int]) -> List[str]:
        """Convert a list of word IDs back to word tokens."""
        return [self.idx2word[i] for i in indices if i in self.idx2word]

    def subsample(self, token_ids: List[int]) -> List[int]:
        """Apply Mikolov subsampling to a sequence of token IDs.

        Tokens with higher frequency are dropped with higher probability.
        """
        if self.keep_probs is None or not token_ids:
            return token_ids

        sampled: List[int] = []
        rand_vals = np.random.random(len(token_ids))

        for idx, word_id in enumerate(token_ids):
            if word_id < len(self.keep_probs):
                if rand_vals[idx] < self.keep_probs[word_id]:
                    sampled.append(word_id)
            else:
                sampled.append(word_id)

        return sampled

    def save(self, filepath: str) -> None:
        """Save vocabulary to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "min_count": self.min_count,
            "subsample_threshold": self.subsample_threshold,
            "total_words": self.total_words,
            "word2idx": self.word2idx,
            "word_counts": self.word_counts,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Vocabulary":
        """Load vocabulary from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls(
            min_count=data["min_count"],
            subsample_threshold=data["subsample_threshold"],
        )
        vocab.total_words = data["total_words"]
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(idx): word for word, idx in vocab.word2idx.items()}
        vocab.word_counts = data["word_counts"]
        vocab._calculate_subsampling_probs()
        return vocab


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    sample_sentences = [
        "Hà Nội là thủ đô của Việt Nam.",
        "Trí tuệ nhân tạo đang làm thay đổi thế giới công nghệ.",
        "Thủ đô Hà Nội có bề dày lịch sử ngàn năm văn hiến.",
        "Việt Nam là một quốc gia phát triển nhanh chóng.",
    ]

    print("--- 1. Kiểm tra Tokenize với Underthesea ---")
    tokenized = []
    for s in sample_sentences:
        tokens = clean_and_tokenize(s)
        tokenized.append(tokens)
        print("Gốc:", s)
        print("Tách:", tokens)
        print()

    print("--- 2. Kiểm tra Xây dựng Từ điển (Vocabulary) ---")
    vocab = Vocabulary(min_count=1)
    vocab.build_vocab(tokenized)
    print(f"Kích thước từ điển: {len(vocab)} từ")
    print(f"Top 5 từ xuất hiện nhiều nhất: {list(vocab.word2idx.keys())[:5]}")

    print("\n--- 3. Kiểm tra Encode & Subsample ---")
    first_sentence_ids = vocab.encode(tokenized[0])
    print(f"Câu: {tokenized[0]}")
    print(f"Token IDs: {first_sentence_ids}")
    print(f"Sau Subsample: {vocab.subsample(first_sentence_ids)}")
