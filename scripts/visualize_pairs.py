"""Visualize 10 word pairs from the trained embeddings, reduced to 2D and 3D with PCA.

Pairs cover two relations (gender, country -> capital): if the embeddings are good,
arrows of the same relation point in roughly the same direction.

    python scripts/visualize_pairs.py [--model models/vietvec.pt] [--out results]
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vietvec.evaluate import Embeddings

PAIRS = {
    "Giới tính (nam → nữ)": [
        ("vua", "nữ_hoàng"),
        ("ông", "bà"),
        ("bố", "mẹ"),
        ("anh_trai", "chị_gái"),
        ("đàn_ông", "phụ_nữ"),
    ],
    "Quốc gia → Thủ đô": [
        ("việt_nam", "hà_nội"),
        ("nhật_bản", "tokyo"),
        ("pháp", "paris"),
        ("hàn_quốc", "seoul"),
        ("nga", "moskva"),
    ],
}
COLORS = ["#2563eb", "#dc2626"]


def label(word: str) -> str:
    return word.replace("_", " ")


def plot(points: dict, dims: int, path: str) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d" if dims == 3 else None)
    for (relation, pairs), color in zip(PAIRS.items(), COLORS):
        for i, (a, b) in enumerate(pairs):
            pa, pb = points[a], points[b]
            ax.scatter(*pa, color=color, s=40, marker="o", label=relation if i == 0 else None)
            ax.scatter(*pb, color=color, s=60, marker="^")
            if dims == 2:
                ax.annotate("", xy=pb, xytext=pa, arrowprops=dict(arrowstyle="->", color=color, alpha=0.6))
            else:
                ax.plot(*np.stack([pa, pb]).T, color=color, alpha=0.6)
            for w, pt in ((a, pa), (b, pb)):
                ax.text(*pt, " " + label(w), fontsize=10)
    ax.set_title(f"VietVec: 10 cặp từ (PCA {dims}D) — ● từ đầu, ▲ từ cuối")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/vietvec.pt")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    emb = Embeddings(args.model)
    words = [w for pairs in PAIRS.values() for pair in pairs for w in pair]
    vectors = np.stack([emb.unit[emb.word2idx[w]] for w in words])

    for dims in (2, 3):
        pca = PCA(n_components=dims).fit(vectors)
        reduced = pca.transform(vectors)
        path = os.path.join(args.out, f"pairs_{dims}d.png")
        plot(dict(zip(words, reduced)), dims, path)
        print(f"{path} (explained variance {pca.explained_variance_ratio_.sum():.0%})")

    print("\nCosine similarity của từng cặp:")
    for pairs in PAIRS.values():
        for a, b in pairs:
            print(f"  {label(a):>10} – {label(b):<10} {emb.similarity(a, b):.3f}")


if __name__ == "__main__":
    main()
