"""Multiprocessing pipeline to tokenize the entire Wikipedia corpus."""

import argparse
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from typing import Generator, List
from tqdm import tqdm

# Ensure vietvec package is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vietvec.data_reader import stream_articles, get_parquet_files
from vietvec.preprocessing import process_articles_batch, Vocabulary


def batch_generator(
    article_stream: Generator[str, None, None], batch_size: int = 200
) -> Generator[List[str], None, None]:
    """Group incoming streamed articles into fixed batches for worker processes."""
    batch: List[str] = []
    for art in article_stream:
        batch.append(art)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_preprocessing(
    num_workers: int = 8,
    batch_size: int = 200,
    min_count: int = 5,
    max_articles: int = None,
    output_text_path: str = "data/processed/wiki_vi_tokenized.txt",
    output_vocab_path: str = "data/processed/vocab.json",
):
    start_time = time.time()
    os.makedirs(os.path.dirname(output_text_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_vocab_path), exist_ok=True)

    print("=" * 60)
    print("🚀 BẮT ĐẦU TIỀN XỬ LÝ TOÀN BỘ WIKIPEDIA TIẾNG VIỆT")
    print(f"- Số tiến trình (Workers): {num_workers} processes")
    print(f"- Kích thước batch mỗi task: {batch_size} bài viết")
    print(f"- Min count cho từ điển: {min_count}")
    print(f"- Tối đa bài viết: {max_articles if max_articles else 'Toàn bộ (~1.28 triệu)'}")
    print(f"- File output: {output_text_path}")
    print("=" * 60)

    # 1. Khởi tạo generator bài viết
    article_stream = stream_articles(max_articles=max_articles)
    batches = batch_generator(article_stream, batch_size=batch_size)

    total_articles_approx = max_articles if max_articles else 1_288_680
    total_batches = (total_articles_approx + batch_size - 1) // batch_size

    total_sentences = 0
    total_articles_processed = 0

    print("\n[Bước 1/2] Đang tách từ và làm sạch văn bản đa tiến trình...")
    with open(output_text_path, "w", encoding="utf-8") as f_out:
        with Pool(processes=num_workers) as pool:
            # imap_unordered yields results as workers finish them
            pbar = tqdm(total=total_articles_approx, desc="Articles", unit="art")
            for sentence_lines in pool.imap_unordered(process_articles_batch, batches, chunksize=2):
                if sentence_lines:
                    f_out.write("\n".join(sentence_lines) + "\n")
                    total_sentences += len(sentence_lines)
                total_articles_processed += batch_size
                pbar.update(batch_size)
            pbar.close()

    step1_time = time.time() - start_time
    print(f"\n✅ Đã ghi {total_sentences:,} câu từ {total_articles_processed:,} bài viết.")
    print(f"⏱️ Thời gian tách từ: {step1_time:.1f}s ({total_articles_processed / max(step1_time, 1):.1f} bài/s)")

    # 2. Xây dựng từ điển trực tiếp từ file tokenized (tiết kiệm RAM tuyệt đối)
    print("\n[Bước 2/2] Đang xây dựng từ điển (Vocabulary) theo luồng...")
    t_v0 = time.time()
    vocab = Vocabulary(min_count=min_count)
    vocab.build_vocab_from_file(output_text_path)
    vocab.save(output_vocab_path)
    t_v1 = time.time() - t_v0

    total_time = time.time() - start_time
    print("=" * 60)
    print(f"🎉 HOÀN THÀNH TIỀN XỬ LÝ TOÀN BỘ!")
    print(f"- Kích thước từ điển (Vocab size): {len(vocab):,} từ vựng")
    print(f"- Tổng số lượt từ (Total tokens): {vocab.total_words:,}")
    print(f"- Thời gian dựng từ điển: {t_v1:.1f}s")
    print(f"- Tổng thời gian xử lý: {total_time / 60:.1f} phút")
    print(f"- File từ điển: {output_vocab_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(cpu_count() - 2, 8))
    parser.add_argument("--batch_size", type=int, default=150)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--max_articles", type=int, default=None)
    parser.add_argument("--output_text_path", type=str, default="data/processed/wiki_vi_tokenized.txt")
    parser.add_argument("--output_vocab_path", type=str, default="data/processed/vocab.json")
    args = parser.parse_args()

    run_preprocessing(
        num_workers=args.workers,
        batch_size=args.batch_size,
        min_count=args.min_count,
        max_articles=args.max_articles,
        output_text_path=args.output_text_path,
        output_vocab_path=args.output_vocab_path,
    )
