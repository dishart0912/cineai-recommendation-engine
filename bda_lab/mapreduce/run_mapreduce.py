"""
bda_lab/mapreduce/run_mapreduce.py
─────────────────────────────────────────────────────────────────────────────
Experiment 4: Hadoop MapReduce Runner & Benchmark.

Simulates the complete Hadoop MapReduce pipeline:
  1. Input Phase: Read raw movie dataset (movies.csv)
  2. Map Phase: mapper.py emits (key, 1) pairs
  3. Shuffle & Sort Phase: group intermediate pairs by key
  4. Reduce Phase: reducer.py aggregates counts per key
  5. Output Phase: save top keywords and genre distributions
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import time
import json
import subprocess
from typing import Dict, List, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAPPER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mapper.py")
REDUCER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reducer.py")
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "movies.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "logs", "mapreduce_results.json")


def run_simulated_mapreduce(input_csv: str = DATA_PATH) -> Dict[str, Any]:
    """Execute Python MapReduce streaming simulation."""
    if not os.path.exists(input_csv):
        return {"error": f"File {input_csv} does not exist."}

    start_time = time.time()

    # Step 1: Map Phase
    # Read file and pass to mapper
    with open(input_csv, "r", encoding="utf-8", errors="replace") as f:
        input_data = f.read()

    map_proc = subprocess.Popen(
        [sys.executable, MAPPER_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8"
    )
    map_stdout, _ = map_proc.communicate(input=input_data)
    map_lines = [line.strip() for line in map_stdout.strip().split("\n") if line.strip()]

    # Step 2: Shuffle & Sort Phase (Standard MapReduce key sort)
    sorted_map_lines = sorted(map_lines)

    # Step 3: Reduce Phase
    reduce_proc = subprocess.Popen(
        [sys.executable, REDUCER_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8"
    )
    reduce_stdout, _ = reduce_proc.communicate(input="\n".join(sorted_map_lines))
    reduce_lines = [line.strip() for line in reduce_stdout.strip().split("\n") if line.strip()]

    duration_ms = round((time.time() - start_time) * 1000, 2)

    # Separate genres and title words
    genres = {}
    words = {}
    for line in reduce_lines:
        parts = line.split("\t")
        if len(parts) == 2:
            key, count = parts[0], int(parts[1])
            if key.startswith("GENRE_"):
                genres[key.replace("GENRE_", "")] = count
            elif key.startswith("WORD_"):
                words[key.replace("WORD_", "")] = count

    # Sort descending
    sorted_genres = sorted(genres.items(), key=lambda x: -x[1])
    sorted_words = sorted(words.items(), key=lambda x: -x[1])[:30]

    results = {
        "dataset": os.path.basename(input_csv),
        "total_map_emissions": len(map_lines),
        "unique_reduced_keys": len(reduce_lines),
        "execution_time_ms": duration_ms,
        "genre_frequencies": dict(sorted_genres),
        "top_title_keywords": dict(sorted_words),
    }

    # Save to logs
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    print("=" * 70)
    print("🐘  BDA Experiment 4: Hadoop MapReduce Word Count Demonstration")
    print("=" * 70)
    res = run_simulated_mapreduce()

    print(f"Dataset Input: {res.get('dataset')}")
    print(f"Total Map (Key, 1) Pairs Emitted: {res.get('total_map_emissions'):,}")
    print(f"Total Unique Keys Reduced: {res.get('unique_reduced_keys'):,}")
    print(f"Pipeline Execution Time: {res.get('execution_time_ms')} ms")

    print("\n📊 Top Movie Genres (MapReduce Aggregation):")
    for g, c in list(res.get("genre_frequencies", {}).items())[:8]:
        print(f"   • {g.ljust(15)} : {c:,} movies")

    print("\n🔤 Top Title Words (WordCount):")
    for w, c in list(res.get("top_title_keywords", {}).items())[:8]:
        print(f"   • {w.ljust(15)} : {c:,} occurrences")
    print("=" * 70)


if __name__ == "__main__":
    main()
