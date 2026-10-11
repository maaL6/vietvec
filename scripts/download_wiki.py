import os
import urllib.request
from tqdm import tqdm

PARQUET_URLS = [
    "https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.vi/train/0.parquet",
    "https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.vi/train/1.parquet",
    "https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.vi/train/2.parquet",
    "https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.vi/train/3.parquet",
]

DEST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")


def download_file(url: str, dest_path: str):
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        block_size = 1024 * 1024  # 1 MB

        desc = os.path.basename(dest_path)
        with open(dest_path, "wb") as f, tqdm(
            desc=desc,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                f.write(buffer)
                bar.update(len(buffer))


def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"Target directory: {DEST_DIR}")

    for idx, url in enumerate(PARQUET_URLS):
        dest_filename = f"wiki_vi_train_{idx}.parquet"
        dest_path = os.path.join(DEST_DIR, dest_filename)

        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            print(f"[{idx+1}/{len(PARQUET_URLS)}] {dest_filename} already exists, skipping.")
            continue

        print(f"[{idx+1}/{len(PARQUET_URLS)}] Downloading {dest_filename}...")
        download_file(url, dest_path)

    print("\nDownload completed successfully!")


if __name__ == "__main__":
    main()
