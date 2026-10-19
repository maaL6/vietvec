"""Visualize 20 word pairs from the trained embeddings, reduced to 2D and 3D with PCA.

Four groups share a real relation (country -> capital, male -> female, ...): if the
embeddings work, the offset vectors b - a of a group point in the same direction.
Two control groups show what "not working" looks like: antonyms sit close together
but their offsets point anywhere, and random pairs are neither close nor parallel.

Writes results/pairs_{2,3}d.png and results/pairs.json (read by web/index.html).

    python scripts/visualize_pairs.py [--model models/vietvec.pt] [--out results]
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vietvec.evaluate import Embeddings

# (key, name, is_control, pairs) — keys are stable ids used by the web viewer for colors.
GROUPS = [
    ("capital", "Quốc gia → thủ đô", False, [
        ("việt_nam", "hà_nội"),
        ("hàn_quốc", "seoul"),
        ("nga", "moskva"),
        ("thái_lan", "bangkok"),
    ]),
    ("gender", "Nam → nữ", False, [
        ("ông", "bà"),
        ("anh_trai", "chị_gái"),
        ("vua", "nữ_hoàng"),
        ("đàn_ông", "phụ_nữ"),
    ]),
    ("language", "Quốc gia → ngôn ngữ", False, [
        ("pháp", "tiếng_pháp"),
        ("đức", "tiếng_đức"),
        ("ý", "tiếng_ý"),
    ]),
    ("province", "Thành phố → tỉnh", False, [
        ("nha_trang", "khánh_hòa"),
        ("đà_lạt", "lâm_đồng"),
        ("quy_nhơn", "bình_định"),
    ]),
    ("antonym", "Trái nghĩa (đối chứng)", True, [
        ("cao", "thấp"),
        ("nhanh", "chậm"),
        ("nóng", "lạnh"),
    ]),
    ("random", "Ngẫu nhiên (đối chứng)", True, [
        ("sông", "bác_sĩ"),
        ("âm_nhạc", "xi_măng"),
        ("bánh_mì", "vũ_trụ"),
    ]),
]
COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#6b7280"]


def label(word: str) -> str:
    return word.replace("_", " ")


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v).clip(1e-8)


def plot(points: dict, dims: int, path: str) -> None:
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection="3d" if dims == 3 else None)
    for (_, name, _, pairs), color in zip(GROUPS, COLORS):
        for i, (a, b) in enumerate(pairs):
            pa, pb = points[a], points[b]
            ax.scatter(*pa, color=color, s=40, marker="o", label=name if i == 0 else None)
            ax.scatter(*pb, color=color, s=60, marker="^")
            if dims == 2:
                ax.annotate("", xy=pb, xytext=pa, arrowprops=dict(arrowstyle="->", color=color, alpha=0.6))
            else:
                ax.plot(*np.stack([pa, pb]).T, color=color, alpha=0.6)
            for w, pt in ((a, pa), (b, pb)):
                ax.text(*pt, " " + label(w), fontsize=9)
    ax.set_title(f"VietVec: 20 cặp từ (PCA {dims}D) — ● từ đầu, ▲ từ cuối")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def export_json(emb: Embeddings, words: list, path: str) -> None:
    """Coordinates for the web viewer plus the 300D numbers that PCA can't show faithfully."""
    vectors = np.stack([emb.unit[emb.word2idx[w]] for w in words])
    coords = {}
    explained = {}
    for dims in (2, 3):
        pca = PCA(n_components=dims).fit(vectors)
        reduced = pca.transform(vectors)
        reduced /= np.abs(reduced).max()
        coords[dims] = dict(zip(words, reduced.round(4).tolist()))
        explained[dims] = round(float(pca.explained_variance_ratio_.sum()), 4)

    offsets = {(a, b): unit(emb.unit[emb.word2idx[b]] - emb.unit[emb.word2idx[a]])
               for _, _, _, pairs in GROUPS for a, b in pairs}
    pair_list = [p for _, _, _, pairs in GROUPS for p in pairs]
    matrix = np.array([[offsets[p] @ offsets[q] for q in pair_list] for p in pair_list])

    # Offsets drawn from a shared origin get their own PCA: the word PCA keeps too little
    # variance to show which offsets are parallel. No centering, so the origin stays 0.
    raw_offsets = np.stack([emb.unit[emb.word2idx[b]] - emb.unit[emb.word2idx[a]] for a, b in pair_list])
    offset_coords = {}
    for dims in (2, 3):
        _, s, vt = np.linalg.svd(raw_offsets, full_matrices=False)
        reduced = raw_offsets @ vt[:dims].T
        reduced /= np.linalg.norm(reduced, axis=1).max()
        offset_coords[dims] = reduced.round(4).tolist()
        explained[f"offset_{dims}"] = round(float((s[:dims] ** 2).sum() / (s ** 2).sum()), 4)

    groups = []
    for key, name, control, pairs in GROUPS:
        out_pairs = []
        for a, b in pairs:
            # Predict b from a using the other pairs' average offset: a + mean(b' - a').
            others = [q for q in pairs if q != (a, b)]
            shift = np.mean([emb.unit[emb.word2idx[d]] - emb.unit[emb.word2idx[c]] for c, d in others], axis=0)
            query = emb.unit[emb.word2idx[a]] + shift
            exclude = [a] + [w for q in others for w in q]
            guesses = emb._nearest(query, exclude, 3)
            out_pairs.append({
                "a": a,
                "b": b,
                "cosine": round(emb.similarity(a, b), 4),
                "prediction": [{"word": w, "score": round(s, 4)} for w, s in guesses],
            })
        idx = [pair_list.index(p) for p in pairs]
        block = matrix[np.ix_(idx, idx)]
        consistency = (block.sum() - len(idx)) / (len(idx) * (len(idx) - 1))
        groups.append({"key": key, "name": name, "control": control,
                       "consistency": round(float(consistency), 4), "pairs": out_pairs})

    data = {
        "dim": int(emb.vectors.shape[1]),
        "vocab": len(emb.words),
        "explained_variance": explained,
        "coords": coords,
        "offset_coords": offset_coords,
        "groups": groups,
        "offset_dot": matrix.round(4).tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    # Same data as a script, so web/index.html also works when opened straight from disk.
    web_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
    os.makedirs(web_dir, exist_ok=True)
    with open(os.path.join(web_dir, "pairs.js"), "w", encoding="utf-8") as f:
        f.write("window.VIETVEC_PAIRS = " + json.dumps(data, ensure_ascii=False) + ";\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/vietvec.pt")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    emb = Embeddings(args.model)
    words = list(dict.fromkeys(w for _, _, _, pairs in GROUPS for pair in pairs for w in pair))
    missing = [w for w in words if w not in emb]
    if missing:
        sys.exit(f"Không có trong vocab: {', '.join(missing)}")
    vectors = np.stack([emb.unit[emb.word2idx[w]] for w in words])

    for dims in (2, 3):
        pca = PCA(n_components=dims).fit(vectors)
        reduced = pca.transform(vectors)
        path = os.path.join(args.out, f"pairs_{dims}d.png")
        plot(dict(zip(words, reduced)), dims, path)
        print(f"{path} (explained variance {pca.explained_variance_ratio_.sum():.0%})")

    json_path = os.path.join(args.out, "pairs.json")
    export_json(emb, words, json_path)
    print(json_path)

    print("\nCosine của từng cặp | độ song song của vector hiệu trong nhóm:")
    with open(json_path, encoding="utf-8") as f:
        groups = json.load(f)["groups"]
    for g in groups:
        print(f"  {g['name']}  (song song trung bình {g['consistency']:.2f})")
        for pr in g["pairs"]:
            guess = pr["prediction"][0]["word"]
            mark = "✓" if guess == pr["b"] else "✗"
            print(f"    {label(pr['a']):>10} – {label(pr['b']):<12} {pr['cosine']:.3f}   đoán: {label(guess)} {mark}")


if __name__ == "__main__":
    main()
