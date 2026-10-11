"""Script to preprocess Wikipedia corpus: stream -> clean & tokenize -> save -> build vocab."""

import argparse
import os
import sys
from tqdm import tqdm

# Ensure vietvec package can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vietvec.data_reader import stream_sentences, get_parquet_files
from vietvec.preprocessing import clean_and_tokenize, Vocabulary


def preprocess_corpus(
    max_sentences: int = 100000,
    min_count: int = 5,
    output_text_path: str = "data/processed/wiki_vi_tokenized.txt",
    output_vocab_path: str = "data/processed/vocab.json",
):
    os.makedirs(os.path.dirname(output_text_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_vocab_path), exist_ok=True)

    print(f"Bắt đầu tiền xử lý văn bản...")
    print(f"- Số câu tối đa: {max_sentences:,}")
    print(f"- Min count cho từ điển: {min_count}")
    print(f"- File text output: {output_text_path}")
    print(f"- File vocab output: {output_vocab_path}\n")

    vocab = Vocabulary(min_count=min_count)
    tokenized_sentences_buffer = []

    print("Đang đọc và tách từ bằng Underthesea...")
    sentence_stream = stream_sentences(max_sentences=max_sentences)

    total_tokens_count = 0
    with open(output_text_path, "w", encoding="utf-8") as f_out:
        for sentence in tqdm(sentence_stream, total=max_sentences, desc="Tokenizing"):
            tokens = clean_and_tokenize(sentence)
            if len(tokens) >= 3:  # Bỏ qua câu quá ngắn (< 3 từ)
                tokenized_sentences_buffer.append(tokens)
                f_out.write(" ".join(tokens) + "\n")
                total_tokens_count += len(tokens)

    print(f"\nĐã ghi {len(tokenized_sentences_buffer):,} câu ({total_tokens_count:,} từ) vào {output_text_path}")

    print("\nĐang xây dựng từ điển (Vocabulary)...")
    vocab.build_vocab(tokenized_sentences_buffer)
    vocab.save(output_vocab_path)

    print(f"Xây dựng từ điển hoàn tất:")
    print(f"- Kích thước từ điển (sau lọc min_count={min_count}): {len(vocab):,} từ vựng")
    print(f"- Tổng số lượt xuất hiện của từ vựng: {vocab.total_words:,}")
    print(f"- Đã lưu từ điển tại: {output_vocab_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Vietnamese Wikipedia corpus")
    parser.add_argument("--max_sentences", type=int, default=10000, help="Số câu tối đa để tiền xử lý")
    parser.add_argument("--min_count", type=int, default=5, help="Tần suất tối thiểu của từ")
    args = parser.parse_args()

    preprocess_corpus(max_sentences=args.max_sentences, min_count=args.min_count)
