#!/usr/bin/env python
"""
bda_lab/mapreduce/mapper.py
─────────────────────────────────────────────────────────────────────────────
Experiment 4: Hadoop MapReduce Word Counting Mapper.

Reads raw CSV lines from sys.stdin.
Extracts title keywords and genre tags, and emits (token, 1) pairs to sys.stdout.
Compatible with standard Hadoop Streaming:
  hadoop jar hadoop-streaming.jar -mapper mapper.py -reducer reducer.py ...
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import re

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Stop words to ignore during title tokenization
STOP_WORDS = {
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "it", "this",
    "that", "no", "genres", "listed"
}

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("movieId,") or line.startswith("userId,"):
            continue

        # MovieLens format: movieId,title,genres OR movieId::title::genres
        parts = line.split("::") if "::" in line else line.split(",")

        if len(parts) >= 3:
            # Genres are in the last column, separated by '|'
            genres_raw = parts[-1]
            genres = genres_raw.split("|")
            for g in genres:
                g = g.strip()
                if g and g.lower() not in STOP_WORDS and g != "(no genres listed)":
                    print(f"GENRE_{g}\t1")

            # Title tokens
            title = " ".join(parts[1:-1])
            words = re.findall(r"\b[A-Za-z]{3,}\b", title.lower())
            for w in words:
                if w not in STOP_WORDS:
                    print(f"WORD_{w}\t1")

if __name__ == "__main__":
    main()
