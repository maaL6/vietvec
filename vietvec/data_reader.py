"""Data reader module for streaming text from Parquet files."""

import glob
import os
import re
from typing import Generator, List, Optional
import pyarrow.parquet as pq


def get_parquet_files(raw_dir: str = "data/raw") -> List[str]:
    """Find all parquet files in the raw data directory."""
    pattern = os.path.join(raw_dir, "*.parquet")
    files = sorted(glob.glob(pattern))
    return files


def stream_articles(
    parquet_paths: Optional[List[str]] = None,
    batch_size: int = 2048,
    max_articles: Optional[int] = None,
) -> Generator[str, None, None]:
    """Stream article texts from parquet files one by one.

    Uses iter_batches from PyArrow to prevent high memory usage.

    Args:
        parquet_paths: List of parquet file paths. Defaults to all in data/raw.
        batch_size: Batch size for reading chunks from disk.
        max_articles: Optional maximum number of articles to yield.

    Yields:
        str: Raw text of each Wikipedia article.
    """
    if parquet_paths is None:
        parquet_paths = get_parquet_files()

    count = 0
    for file_path in parquet_paths:
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["text"]):
            texts = batch.column("text").to_pylist()
            for text in texts:
                if not text or not isinstance(text, str):
                    continue
                text = text.strip()
                if not text:
                    continue

                yield text
                count += 1
                if max_articles is not None and count >= max_articles:
                    return


def split_into_sentences(text: str, min_chars: int = 20) -> List[str]:
    """Split article text into clean sentences/paragraphs.

    Filters out standard Wikipedia section headers like 'Chú thích', 'Tham khảo'.
    """
    # Exclude common boilerplate Wikipedia headers
    boilerplate_patterns = [
        r"^chú thích$",
        r"^tham khảo$",
        r"^liên kết ngoài$",
        r"^xem thêm$",
        r"^thư mục$",
        r"^nguồn$",
    ]
    boilerplate_re = re.compile("|".join(boilerplate_patterns), re.IGNORECASE)

    # First split by line breaks to get paragraphs
    raw_lines = text.split("\n")
    sentences = []

    for line in raw_lines:
        line = line.strip()
        if not line or len(line) < min_chars:
            continue
        if boilerplate_re.match(line):
            continue

        # Split line by sentence terminators (. ! ?)
        parts = re.split(r"(?<=[.!?])\s+", line)
        for part in parts:
            part = part.strip()
            if len(part) >= min_chars:
                sentences.append(part)

    return sentences


def stream_sentences(
    parquet_paths: Optional[List[str]] = None,
    batch_size: int = 2048,
    max_sentences: Optional[int] = None,
) -> Generator[str, None, None]:
    """Stream individual cleaned sentences directly from parquet files.

    Yields:
        str: Clean sentence string ready for tokenization.
    """
    count = 0
    for article in stream_articles(parquet_paths=parquet_paths, batch_size=batch_size):
        sentences = split_into_sentences(article)
        for sentence in sentences:
            yield sentence
            count += 1
            if max_sentences is not None and count >= max_sentences:
                return


if __name__ == "__main__":
    # Test reading sample articles
    print("Finding parquet files...")
    files = get_parquet_files()
    print(f"Found {len(files)} parquet file(s): {[os.path.basename(f) for f in files]}")

    print("\nReading first 5 sample sentences from parquet streaming:")
    for i, s in enumerate(stream_sentences(max_sentences=5)):
        print(f"[{i+1}] {s}")
