#!/usr/bin/env python
"""
bda_lab/mapreduce/reducer.py
─────────────────────────────────────────────────────────────────────────────
Experiment 4: Hadoop MapReduce Word Counting Reducer.

Reads key-sorted lines formatted as: "<key>\t<count>" from sys.stdin.
Aggregates frequencies and emits "<key>\t<total_count>" to sys.stdout.
Compatible with standard Hadoop Streaming:
  hadoop jar hadoop-streaming.jar -mapper mapper.py -reducer reducer.py ...
─────────────────────────────────────────────────────────────────────────────
"""

import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def main():
    current_key = None
    current_count = 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            key, count_str = line.split("\t", 1)
            count = int(count_str)
        except ValueError:
            continue

        if current_key == key:
            current_count += count
        else:
            if current_key is not None:
                print(f"{current_key}\t{current_count}")
            current_key = key
            current_count = count

    # Output final key
    if current_key is not None:
        print(f"{current_key}\t{current_count}")

if __name__ == "__main__":
    main()
