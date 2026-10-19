"""Train VietVec (Skip-gram + Negative Sampling).

Prerequisites: data/processed/vocab.json and data/processed/corpus_ids.npz
(built by scripts/encode_corpus.py).

    python train.py                          # full run (GPU recommended)
    python train.py --max_tokens 2000000 --epochs 1 --dim 100   # quick smoke test
"""

import argparse
import os

import numpy as np
import torch

from vietvec.dataset import SkipGramBatcher
from vietvec.evaluate import Embeddings
from vietvec.model import SkipGramModel
from vietvec.preprocessing import Vocabulary
from vietvec.trainer import train

PROBE_WORDS = ["hà_nội", "vua", "phụ_nữ", "bóng_đá", "máy_tính", "sông"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--out_dir", default="models")
    p.add_argument("--dim", type=int, default=300)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--negatives", type=int, default=5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32768)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--max_tokens", type=int, default=None, help="Use only the first N tokens (testing)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | {vars(args)}")

    vocab = Vocabulary.load(os.path.join(args.data_dir, "vocab.json"))
    corpus = np.load(os.path.join(args.data_dir, "corpus_ids.npz"))
    ids, lengths = corpus["ids"], corpus["lengths"]
    if args.max_tokens:
        n_sent = int(np.searchsorted(np.cumsum(lengths), args.max_tokens))
        lengths = lengths[:n_sent]
        ids = ids[: int(lengths.sum())]
    print(f"Vocab {len(vocab):,} words | corpus {len(ids):,} tokens, {len(lengths):,} sentences")

    words = [vocab.idx2word[i] for i in range(len(vocab))]
    batcher = SkipGramBatcher(
        ids, lengths, vocab, args.batch_size, args.window, args.negatives, device=device
    )
    model = SkipGramModel(len(vocab), args.dim).to(device)
    train(model, batcher, words, args.epochs, args.lr, out_dir=args.out_dir)

    final_path = os.path.join(args.out_dir, "vietvec.pt")
    emb = Embeddings(final_path)
    for w in PROBE_WORDS:
        if w in emb:
            print(w, "->", ", ".join(f"{n} ({s:.2f})" for n, s in emb.most_similar(w, 8)))
    print("Saved", final_path)


if __name__ == "__main__":
    main()
