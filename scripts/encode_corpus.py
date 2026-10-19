"""Encode the tokenized corpus into word IDs (data/processed/corpus_ids.npz) for training."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vietvec.dataset import encode_corpus
from vietvec.preprocessing import Vocabulary

if __name__ == "__main__":
    t0 = time.time()
    vocab = Vocabulary.load("data/processed/vocab.json")
    encode_corpus("data/processed/wiki_vi_tokenized.txt", vocab, "data/processed/corpus_ids.npz")
    print(f"Done in {time.time() - t0:.0f}s -> data/processed/corpus_ids.npz")
