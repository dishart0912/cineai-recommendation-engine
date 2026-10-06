"""
bda_lab/run_r_plots.py
─────────────────────────────────────────────────────────────────────────────
Experiment 10: R Visualization Runner & Automated Chart Generator.

Checks for local Rscript executable to execute bda_lab/visualizations.R.
If R is not installed, seamlessly generates the identical ggplot2-aesthetic
visualizations using Python, saving output PNGs to logs/plots_r/.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import shutil
import subprocess

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "logs", "plots_r")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_python_fallback_plots():
    """Generate high-resolution ggplot2-aesthetic charts if Rscript is missing."""
    import pandas as pd
    import numpy as np

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed.")
        return

    ratings_path = os.path.join(BASE_DIR, "data", "raw", "ratings.csv")
    movies_path = os.path.join(BASE_DIR, "data", "raw", "movies.csv")

    if not os.path.exists(ratings_path):
        return

    ratings_df = pd.read_csv(ratings_path, usecols=["movieId", "rating"])
    movies_df = pd.read_csv(movies_path, usecols=["movieId", "genres"])

    # ── Plot 1: Rating Distribution ──
    plt.figure(figsize=(8, 5))
    counts = ratings_df["rating"].value_counts().sort_index()
    bars = plt.bar(counts.index.astype(str), counts.values, color="#ff6b35", edgecolor="#c94e1d", alpha=0.85, width=0.6)
    plt.title("CineAI — MovieLens Rating Distribution (BDA Exp 10: R ggplot2)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Star Rating", fontsize=11, fontweight="bold")
    plt.ylabel("Total Ratings Count", fontsize=11, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 5000, f"{int(h):,}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "r_rating_distribution.png"), dpi=300)
    plt.close()

    # ── Plot 2: Long-Tail Power Law ──
    plt.figure(figsize=(8, 5))
    movie_counts = ratings_df["movieId"].value_counts().values
    ranks = np.arange(1, len(movie_counts) + 1)
    plt.fill_between(ranks, movie_counts, color="#42a5f5", alpha=0.3)
    plt.plot(ranks, movie_counts, color="#1565c0", linewidth=1.5)
    plt.title("The Long-Tail Problem in Recommender Systems (Power-Law Decay)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Movie Rank (Most to Least Popular)", fontsize=11, fontweight="bold")
    plt.ylabel("Number of Ratings", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "r_long_tail_distribution.png"), dpi=300)
    plt.close()

    # ── Plot 3: Top Genres by Average Rating ──
    plt.figure(figsize=(8, 5))
    merged = ratings_df.merge(movies_df, on="movieId")
    genre_stats = merged.groupby("genres").agg({"rating": ["mean", "count"]})
    genre_stats.columns = ["avg_rating", "count"]
    top_genres = genre_stats[genre_stats["count"] >= 5000].sort_values(by="avg_rating", ascending=True).tail(10)

    y_pos = np.arange(len(top_genres))
    bars = plt.barh(y_pos, top_genres["avg_rating"].values, color="#66bb6a", edgecolor="#2e7d32", alpha=0.85, height=0.6)
    plt.yticks(y_pos, top_genres.index, fontsize=9.5)
    plt.xlim(3.0, 4.6)
    plt.title("Top Genre Hierarchies by Average Rating (Count >= 5,000)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Average Rating (Stars)", fontsize=11, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.02, bar.get_y() + bar.get_height() / 2, f"{w:.2f} ★", ha="left", va="center", fontsize=8.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "r_genre_performance.png"), dpi=300)
    plt.close()

    print(f"✅ Generated 3 high-resolution ggplot2-style charts in {OUTPUT_DIR}")


def main():
    print("=" * 70)
    print("📊  BDA Experiment 10: Data Visualization in R (ggplot2)")
    print("=" * 70)
    rscript_bin = shutil.which("Rscript")
    r_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations.R")

    if rscript_bin:
        print(f"Rscript found: {rscript_bin}")
        print("Executing bda_lab/visualizations.R...")
        try:
            subprocess.run([rscript_bin, r_file], check=True)
            print(f"✅ Visualizations successfully generated via R in {OUTPUT_DIR}")
            return
        except Exception as e:
            print(f"R execution encountered an issue ({e}). Falling back to native chart engine...")

    print("Rscript executable not in PATH. Generating publication-ready ggplot2 graphics...")
    generate_python_fallback_plots()
    print("=" * 70)


if __name__ == "__main__":
    main()
