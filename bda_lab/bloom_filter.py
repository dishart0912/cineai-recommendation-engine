"""
bda_lab/bloom_filter.py
─────────────────────────────────────────────────────────────────────────────
Experiment 8: Bloom Filter for Constant-Time Recommendation Deduplication.

In modern recommendation platforms (Netflix, Medium, Spotify), checking whether
a user has already seen or rated an item across billions of historical interactions
cannot be done by loading raw sets into memory.

A Bloom Filter is a space-efficient probabilistic data structure that provides:
  1. O(k) membership testing (where k is number of hash functions).
  2. Zero False Negatives (if item was added, contains() always returns True).
  3. Bounded False Positive Rate (p ≈ 0.01) using 10x less RAM than raw hash sets.
─────────────────────────────────────────────────────────────────────────────
"""

import math
import sys
import hashlib
from typing import List, Any, Dict, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class BloomFilter:
    """
    Probabilistic Bloom Filter using Kirsch-Mitzenmacher double-hashing technique:
        hash_i(x) = (h1(x) + i * h2(x)) % m
    """

    def __init__(self, expected_items: int = 5000, false_positive_rate: float = 0.01):
        """
        Initialize Bloom Filter with optimal bit array size (m) and hash functions (k).

        Args:
            expected_items (n): Expected number of elements to store.
            false_positive_rate (p): Desired false positive probability (e.g. 0.01 = 1%).
        """
        self.n = max(1, expected_items)
        self.p = min(0.5, max(1e-6, false_positive_rate))

        # Optimal bit array size: m = -(n * ln(p)) / (ln(2)^2)
        self.m = int(- (self.n * math.log(self.p)) / (math.log(2) ** 2))
        self.m = max(64, self.m)

        # Optimal number of hash functions: k = (m / n) * ln(2)
        self.k = int(round((self.m / self.n) * math.log(2)))
        self.k = max(1, min(self.k, 30))

        # Bit array implemented as integer or bytearray
        self._bit_array = bytearray((self.m + 7) // 8)
        self.count = 0

    def _hashes(self, item: Any) -> List[int]:
        """Generate k hash indices for a given item using double hashing."""
        item_bytes = str(item).encode("utf-8")
        h1 = int(hashlib.sha256(item_bytes).hexdigest()[:16], 16)
        h2 = int(hashlib.md5(item_bytes).hexdigest()[:16], 16)

        indices = []
        for i in range(self.k):
            idx = (h1 + i * h2) % self.m
            indices.append(idx)
        return indices

    def add(self, item: Any):
        """Add an item to the Bloom Filter."""
        for idx in self._hashes(item):
            byte_idx = idx // 8
            bit_offset = idx % 8
            self._bit_array[byte_idx] |= (1 << bit_offset)
        self.count += 1

    def contains(self, item: Any) -> bool:
        """
        Check if item might be in the set.
        Returns:
            False: Item is DEFINITIVELY NOT in the set (0% False Negatives).
            True:  Item is PROBABLY in the set (with False Positive probability <= p).
        """
        for idx in self._hashes(item):
            byte_idx = idx // 8
            bit_offset = idx % 8
            if not (self._bit_array[byte_idx] & (1 << bit_offset)):
                return False
        return True

    def __contains__(self, item: Any) -> bool:
        return self.contains(item)

    def filter_unseen(self, candidates: List[Dict[str, Any]], key: str = "movieId") -> List[Dict[str, Any]]:
        """Filter out candidates that have already been seen/rated."""
        unseen = []
        for item in candidates:
            val = item.get(key)
            if val is not None and not self.contains(val):
                unseen.append(item)
        return unseen

    def get_stats(self) -> Dict[str, Any]:
        """Return memory footprint and statistical diagnostics."""
        bits_set = sum(bin(byte).count("1") for byte in self._bit_array)
        fraction_set = bits_set / self.m
        # Current empirical false positive probability: (1 - e^(-k*n / m))^k
        current_fp_rate = (1.0 - math.exp(-self.k * self.count / self.m)) ** self.k if self.m > 0 else 0.0

        memory_bytes = len(self._bit_array)
        # Compare with Python raw set: ~200-300 bytes per element in set
        python_set_bytes_est = self.count * 64

        return {
            "capacity_n": self.n,
            "target_fp_rate": self.p,
            "bit_array_size_m": self.m,
            "num_hash_functions_k": self.k,
            "elements_inserted": self.count,
            "bits_set": bits_set,
            "bit_density": round(fraction_set, 4),
            "estimated_current_fp_rate": round(current_fp_rate, 5),
            "bloom_memory_bytes": memory_bytes,
            "bloom_memory_kb": round(memory_bytes / 1024, 2),
            "equivalent_python_set_kb": round(python_set_bytes_est / 1024, 2),
            "memory_reduction_factor": round(max(1.0, python_set_bytes_est / max(1, memory_bytes)), 1),
        }


def demo_bloom_filter():
    """Interactive demo testing on MovieLens movie IDs."""
    print("=" * 70)
    print("🎬  BDA Experiment 8: Bloom Filter Demonstration")
    print("=" * 70)

    # 1. Initialize Bloom Filter for 500 items with 1% false positive rate
    bf = BloomFilter(expected_items=500, false_positive_rate=0.01)

    # 2. Insert 300 movie IDs (representing user's watched/rated history)
    watched_movies = [f"movie_{i}" for i in range(1, 301)]
    for m in watched_movies:
        bf.add(m)

    # 3. Test False Negatives (MUST BE 0)
    false_negatives = sum(1 for m in watched_movies if not bf.contains(m))
    print(f"✅ Items Inserted: {len(watched_movies)}")
    print(f"✅ False Negatives (Known Watched Movies missed): {false_negatives} (Always 0.0%)")

    # 4. Test False Positives on 10,000 unseen movies
    unseen_movies = [f"movie_{i}" for i in range(1000, 11000)]
    false_positives = sum(1 for m in unseen_movies if bf.contains(m))
    fp_rate = (false_positives / len(unseen_movies)) * 100
    print(f"📊 Unseen Items Tested: {len(unseen_movies):,}")
    print(f"📊 False Positives detected: {false_positives} ({fp_rate:.2f}% vs target {bf.p*100:.1f}%)")

    stats = bf.get_stats()
    print("\n📦 Memory Diagnostics:")
    print(f"   Bit Array Size (m): {stats['bit_array_size_m']} bits ({stats['bloom_memory_kb']} KB)")
    print(f"   Number of Hashes (k): {stats['num_hash_functions_k']}")
    print(f"   Raw Hash Set Size: ~{stats['equivalent_python_set_kb']} KB")
    print(f"   Memory Savings: {stats['memory_reduction_factor']}x less RAM!")
    print("=" * 70)


if __name__ == "__main__":
    demo_bloom_filter()
