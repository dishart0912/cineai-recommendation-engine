"""
data/download_data.py
─────────────────────────────────────────────────────────────────────────────
Downloads the MovieLens dataset (1M by default, switchable to 25M).
Extracts movies.csv and ratings.csv into data/raw/.

Usage:
    python data/download_data.py --size 1m          # ~6 MB  (default)
    python data/download_data.py --size 25m         # ~250 MB
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import zipfile
import requests
from tqdm import tqdm

DATASETS = {
    "1m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "zip_name": "ml-1m.zip",
        "folder": "ml-1m",
        "ratings_file": "ratings.dat",
        "movies_file": "movies.dat",
        "sep": "::",
        "encoding": "latin-1",
    },
    "25m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
        "zip_name": "ml-25m.zip",
        "folder": "ml-25m",
        "ratings_file": "ratings.csv",
        "movies_file": "movies.csv",
        "sep": ",",
        "encoding": "utf-8",
    },
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")


def download_file(url: str, dest: str) -> None:
    """Stream-download a file with a progress bar."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        desc=os.path.basename(dest),
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def convert_dat_to_csv(src: str, dest: str, sep: str, encoding: str, columns: list) -> None:
    """Convert .dat (double-colon-separated) to .csv."""
    import pandas as pd
    df = pd.read_csv(src, sep=sep, names=columns, encoding=encoding, engine="python")
    df.to_csv(dest, index=False)
    print(f"  ✔  Converted {os.path.basename(src)} → {os.path.basename(dest)}")


def main(size: str = "1m") -> None:
    cfg = DATASETS[size]
    os.makedirs(RAW_DIR, exist_ok=True)

    zip_path = os.path.join(RAW_DIR, cfg["zip_name"])

    # ── 1. Download ─────────────────────────────────────────────────────────
    if os.path.exists(zip_path):
        print(f"Zip already exists: {zip_path} — skipping download.")
    else:
        print(f"\n📥  Downloading MovieLens {size.upper()} from:\n    {cfg['url']}\n")
        download_file(cfg["url"], zip_path)

    # ── 2. Extract ──────────────────────────────────────────────────────────
    extract_dir = os.path.join(RAW_DIR, cfg["folder"])
    if os.path.exists(extract_dir):
        print(f"Already extracted: {extract_dir}")
    else:
        print(f"\n📦  Extracting {cfg['zip_name']} …")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(RAW_DIR)
        print("  ✔  Extraction complete.")

    # ── 3. Normalise to CSV ─────────────────────────────────────────────────
    out_ratings = os.path.join(RAW_DIR, "ratings.csv")
    out_movies = os.path.join(RAW_DIR, "movies.csv")

    if size == "1m":
        ratings_src = os.path.join(extract_dir, "ratings.dat")
        movies_src = os.path.join(extract_dir, "movies.dat")

        if not os.path.exists(out_ratings):
            convert_dat_to_csv(
                ratings_src, out_ratings, cfg["sep"], cfg["encoding"],
                ["userId", "movieId", "rating", "timestamp"],
            )
        if not os.path.exists(out_movies):
            convert_dat_to_csv(
                movies_src, out_movies, cfg["sep"], cfg["encoding"],
                ["movieId", "title", "genres"],
            )
    else:
        # 25M already ships as CSV
        import shutil
        for src, dst in [(os.path.join(extract_dir, "ratings.csv"), out_ratings),
                         (os.path.join(extract_dir, "movies.csv"), out_movies)]:
            if not os.path.exists(dst):
                shutil.copy(src, dst)
                print(f"  ✔  Copied {os.path.basename(src)}")

    print(f"\n✅  Dataset ready in: {RAW_DIR}")
    print(f"   ratings.csv : {out_ratings}")
    print(f"   movies.csv  : {out_movies}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MovieLens dataset")
    parser.add_argument("--size", choices=["1m", "25m"], default="1m",
                        help="Dataset size: '1m' (6 MB) or '25m' (250 MB)")
    args = parser.parse_args()
    main(args.size)
