"""Qualitative + quantitative check of trained embeddings, and optional .vec export.

    python scripts/evaluate_model.py [--model models/vietvec.pt] [--export models/vietvec.vec]
"""

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vietvec.evaluate import Embeddings

NEIGHBOUR_WORDS = ["hà_nội", "vua", "phụ_nữ", "bóng_đá", "máy_tính", "sông", "đẹp", "chạy", "bác_sĩ", "tiếng_việt"]

# Each group holds word pairs sharing one relation; every ordered combination of two
# pairs (a, b), (c, d) gives one analogy a : b = c : d.
ANALOGY_GROUPS = {
    "quốc gia → thủ đô": [
        ("việt_nam", "hà_nội"), ("nhật_bản", "tokyo"), ("pháp", "paris"), ("hàn_quốc", "seoul"),
        ("nga", "moskva"), ("đức", "berlin"), ("thái_lan", "bangkok"), ("ý", "roma"),
        ("ai_cập", "cairo"), ("anh", "luân_đôn"), ("hà_lan", "amsterdam"), ("thụy_điển", "stockholm"),
        ("hy_lạp", "athena"),
    ],
    "nam → nữ": [
        ("vua", "nữ_hoàng"), ("ông", "bà"), ("bố", "mẹ"), ("anh_trai", "chị_gái"),
        ("đàn_ông", "phụ_nữ"), ("hoàng_tử", "công_chúa"), ("chồng", "vợ"), ("chú", "cô"),
        ("em_trai", "em_gái"), ("nam_giới", "nữ_giới"), ("hoàng_đế", "hoàng_hậu"),
    ],
    "quốc gia → ngôn ngữ": [
        ("pháp", "tiếng_pháp"), ("đức", "tiếng_đức"), ("nhật_bản", "tiếng_nhật"),
        ("việt_nam", "tiếng_việt"), ("ý", "tiếng_ý"), ("thái_lan", "tiếng_thái"),
    ],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/vietvec.pt")
    p.add_argument("--export", default=None, help="Optional path to write a .vec file")
    args = p.parse_args()
    emb = Embeddings(args.model)
    print(f"{len(emb.words):,} words x {emb.vectors.shape[1]} dims\n")

    print("== Từ gần nghĩa nhất")
    for w in NEIGHBOUR_WORDS:
        if w in emb:
            print(f"{w:>12}: " + ", ".join(n for n, _ in emb.most_similar(w, 8)))

    print("\n== Phép loại suy: ông : bà = vua : ?")
    print("  ", emb.analogy("ông", "bà", "vua"))
    print("== việt_nam : hà_nội = nhật_bản : ?")
    print("  ", emb.analogy("việt_nam", "hà_nội", "nhật_bản"))

    print("\n== Độ chính xác loại suy (top-1)")
    for name, pairs in ANALOGY_GROUPS.items():
        pairs = [pr for pr in pairs if all(w in emb for w in pr)]
        quads = [(a, b, c, d) for (a, b), (c, d) in itertools.permutations(pairs, 2)]
        print(f"  {name:<22} {emb.evaluate_analogies(quads):6.1%}  ({len(quads)} câu hỏi)")

    if args.export:
        emb.save_word2vec_format(args.export)
        print("\nExported", args.export)


if __name__ == "__main__":
    main()
