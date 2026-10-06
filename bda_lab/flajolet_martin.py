"""
bda_lab/flajolet_martin.py
─────────────────────────────────────────────────────────────────────────────
Experiment 9: Flajolet-Martin (FM) Algorithm for Streaming Cardinality Estimation.

In Big Data interaction streams (e.g. tracking unique active users viewing movies),
storing every unique user ID in a Set or Hash Table requires O(N) memory, which
exhausts RAM when N reaches hundreds of millions.

The Flajolet-Martin algorithm computes an approximate count of distinct elements
using O(log(N)) memory by hashing elements and observing the maximum number
of trailing zeroes (R) in their binary bit representations.

Formula:
  Estimate = (2^R) / phi
  where phi ≈ 0.77351 (Flajolet-Martin correction factor)

To reduce variance, we use multiple independent hash functions grouped into
buckets, computing the average within each bucket and the median across buckets.
─────────────────────────────────────────────────────────────────────────────
"""

import math
import sys
import hashlib
from typing import List, Any, Dict, Iterable, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class FlajoletMartin:
    """
    Flajolet-Martin streaming distinct elements estimator.
    Uses multi-hash grouping (median of averages) for variance reduction.
    """

    PHI = 0.77351  # Correction constant for FM algorithm

    def __init__(self, num_hashes: int = 32, num_groups: int = 4):
        """
        Initialize FM estimator.

        Args:
            num_hashes: Total number of independent hash functions (e.g. 32).
            num_groups: Number of buckets to divide hash functions into for median averaging.
        """
        self.num_hashes = max(4, num_hashes)
        self.num_groups = max(1, min(num_groups, self.num_hashes))
        self.hashes_per_group = self.num_hashes // self.num_groups

        # Track the maximum trailing zeroes observed for each hash function
        self.max_trailing_zeros = [0] * self.num_hashes
        self.total_stream_items = 0

    @staticmethod
    def _count_trailing_zeros(val: int) -> int:
        """Count number of trailing zero bits in binary representation."""
        if val == 0:
            return 32
        count = 0
        while (val & 1) == 0:
            count += 1
            val >>= 1
        return count

    def _hash(self, item: Any, seed: int) -> int:
        """32-bit deterministic hash for item and seed."""
        data = f"{seed}:{item}".encode("utf-8")
        digest = hashlib.md5(data).digest()
        # Extract 32-bit integer
        return int.from_bytes(digest[:4], byteorder="little")

    def add(self, item: Any):
        """Process one incoming element from the continuous stream."""
        self.total_stream_items += 1
        for i in range(self.num_hashes):
            hashed_val = self._hash(item, seed=i * 10007 + 3)
            trailing = self._count_trailing_zeros(hashed_val)
            if trailing > self.max_trailing_zeros[i]:
                self.max_trailing_zeros[i] = trailing

    def update_stream(self, items: Iterable[Any]):
        """Stream multiple items into the estimator."""
        for item in items:
            self.add(item)

    def estimate(self) -> int:
        """
        Compute cardinality estimate using textbook MMDS Flajolet-Martin:
        Group hash functions, calculate 2^(mean(R)) per group, and take the median.
        """
        group_estimates = []
        for g in range(self.num_groups):
            start = g * self.hashes_per_group
            end = start + self.hashes_per_group
            group_r = self.max_trailing_zeros[start:end]

            # Mean of R within group, then 2^mean / PHI
            mean_r = sum(group_r) / len(group_r)
            est = (2 ** mean_r) / self.PHI
            group_estimates.append(est)

        # Median of groups eliminates outlier distortion
        group_estimates.sort()
        mid = len(group_estimates) // 2
        if len(group_estimates) % 2 == 1:
            median_est = group_estimates[mid]
        else:
            median_est = (group_estimates[mid - 1] + group_estimates[mid]) / 2.0

        return max(0, int(round(median_est)))

    def get_diagnostics(self, exact_distinct_count: Optional[int] = None) -> Dict[str, Any]:
        """Return memory footprint and estimation accuracy diagnostics."""
        estimated = self.estimate()
        memory_bytes = self.num_hashes * 4  # 32-bit integer per hash register

        stats = {
            "total_stream_events": self.total_stream_items,
            "estimated_distinct_elements": estimated,
            "fm_memory_bytes": memory_bytes,
            "num_hash_functions": self.num_hashes,
            "num_groups": self.num_groups,
            "max_trailing_zeros_registers": self.max_trailing_zeros,
        }

        if exact_distinct_count is not None and exact_distinct_count > 0:
            abs_error = abs(estimated - exact_distinct_count)
            rel_error_pct = (abs_error / exact_distinct_count) * 100
            stats["exact_distinct_count"] = exact_distinct_count
            stats["absolute_error"] = abs_error
            stats["relative_error_pct"] = round(rel_error_pct, 2)
            stats["accuracy_pct"] = round(max(0.0, 100.0 - rel_error_pct), 2)
            # Memory comparison
            python_set_bytes = exact_distinct_count * 56  # Set memory overhead
            stats["python_set_bytes"] = python_set_bytes
            stats["memory_savings_factor"] = round(python_set_bytes / max(1, memory_bytes), 1)

        return stats


def demo_flajolet_martin():
    """Interactive demo on MovieLens user rating stream."""
    import os
    import csv

    print("=" * 70)
    print("⚡  BDA Experiment 9: Flajolet-Martin (FM) Cardinality Estimation")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ratings_path = os.path.join(base_dir, "data", "raw", "ratings.csv")

    fm = FlajoletMartin(num_hashes=32, num_groups=4)
    exact_users = set()

    if os.path.exists(ratings_path):
        print(f"Streaming from real MovieLens data: {ratings_path}")
        with open(ratings_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            count = 0
            for row in reader:
                if row:
                    user_id = row[0]
                    fm.add(user_id)
                    exact_users.add(user_id)
                    count += 1
                    if count >= 200000:  # Sample 200k interactions for rapid demonstration
                        break
    else:
        # Fallback synthetic stream
        print("Simulating streaming log with 5,000 distinct users across 100,000 events...")
        import random
        for _ in range(100000):
            uid = f"user_{random.randint(1, 5000)}"
            fm.add(uid)
            exact_users.add(uid)

    exact_count = len(exact_users)
    diag = fm.get_diagnostics(exact_distinct_count=exact_count)

    print(f"\n📊 Total Stream Events Processed: {diag['total_stream_events']:,}")
    print(f"🎯 Exact Distinct Users: {exact_count:,}")
    print(f"🔮 FM Estimated Distinct Users: {diag['estimated_distinct_elements']:,}")
    print(f"📈 Accuracy: {diag['accuracy_pct']}% (Relative Error: {diag['relative_error_pct']}%)")
    print(f"💾 Memory Used by FM: {diag['fm_memory_bytes']} bytes (vs Set: ~{diag['python_set_bytes']:,} bytes)")
    print(f"🚀 Memory Compression: {diag['memory_savings_factor']}x less RAM!")
    print("=" * 70)


if __name__ == "__main__":
    demo_flajolet_martin()
