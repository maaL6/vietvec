"""Post-processing script to filter vocabulary:
1. Filter by min_count >= 20
2. Remove non-Vietnamese characters (Chinese/CJK, Russian/Cyrillic, symbols)
3. Recompute indices and Mikolov subsampling probabilities
"""

import json
import os
import re
import shutil
import sys
import numpy as np

# Ensure vietvec package is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vietvec.preprocessing import Vocabulary

# Match valid Vietnamese characters (lowercase letters, digits, underscore)
VIETNAMESE_VALID_TOKEN_REGEX = re.compile(r"^[a-zà-ỹ0-9_]+$")


def clean_and_filter_vocabulary(
    input_vocab_path: str = "data/processed/vocab.json",
    output_vocab_path: str = "data/processed/vocab.json",
    backup_path: str = "data/processed/vocab_raw_487k.json",
    min_count: int = 20,
    subsample_threshold: float = 1e-4,
):
    print("=" * 60)
    print("🧹 BẮT ĐẦU LÀM SẠCH VÀ LỌC TỪ ĐIỂN CHUYÊN SÂU")
    print(f"- File từ điển gốc: {input_vocab_path}")
    print(f"- Ngưỡng tần suất tối thiểu: min_count >= {min_count}")
    print("=" * 60)

    # 1. Sao lưu file gốc nếu chưa có
    if not os.path.exists(backup_path):
        print(f"Đang sao lưu từ điển gốc vào: {backup_path}...")
        shutil.copyfile(input_vocab_path, backup_path)

    # 2. Đọc từ điển hiện tại
    with open(backup_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_word2idx = data["word2idx"]
    raw_counts = data["word_counts"]
    raw_total_words = data["total_words"]

    print(f"\nTừ điển ban đầu: {len(raw_word2idx):,} từ ({raw_total_words:,} lượt xuất hiện)")

    # 3. Lọc từ: kiểm tra regex và min_count
    valid_words = []
    dropped_rare = 0
    dropped_non_vi = 0

    for word, idx in raw_word2idx.items():
        count = raw_counts[idx]

        # Kiểm tra tần suất
        if count < min_count:
            dropped_rare += 1
            continue

        # Kiểm tra ký tự thuần tiếng Việt / Latinh / Số / Dấu _
        if not VIETNAMESE_VALID_TOKEN_REGEX.match(word) or word.strip("_") == "":
            dropped_non_vi += 1
            continue

        valid_words.append((word, count))

    # Sắp xếp theo tần suất giảm dần
    valid_words.sort(key=lambda x: x[1], reverse=True)

    # 4. Xây dựng lại cấu trúc Vocabulary
    new_vocab = Vocabulary(min_count=min_count, subsample_threshold=subsample_threshold)
    new_vocab.word2idx = {}
    new_vocab.idx2word = {}
    new_vocab.word_counts = []

    for new_idx, (word, count) in enumerate(valid_words):
        new_vocab.word2idx[word] = new_idx
        new_vocab.idx2word[new_idx] = word
        new_vocab.word_counts.append(count)

    new_vocab.total_words = sum(new_vocab.word_counts)
    new_vocab._calculate_subsampling_probs()

    # 5. Lưu từ điển đã làm sạch
    new_vocab.save(output_vocab_path)

    print("\n" + "=" * 60)
    print("🎉 LÀM SẠCH VÀ LỌC TỪ ĐIỂN HOÀN TẤT!")
    print(f"- Số từ bị loại do tần suất < {min_count}: {dropped_rare:,} từ")
    print(f"- Số từ bị loại do chứa chữ Hán/Nga/ký tự lạ: {dropped_non_vi:,} từ")
    print(f"- Kích thước từ điển MỚI (Chuẩn sạch): {len(new_vocab):,} từ")
    print(f"- Tổng số lượt từ trong từ điển mới: {new_vocab.total_words:,} ({new_vocab.total_words / raw_total_words * 100:.2f}% tổng từ gốc)")
    print(f"- File lưu: {output_vocab_path}")
    print("=" * 60)


if __name__ == "__main__":
    clean_and_filter_vocabulary()
