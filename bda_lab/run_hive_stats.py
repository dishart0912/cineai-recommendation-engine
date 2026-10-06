"""
bda_lab/run_hive_stats.py
─────────────────────────────────────────────────────────────────────────────
Experiment 6: Hive Analytics & Descriptive Statistics Execution Runner.

Executes HiveQL statements across ratings.csv and movies.csv using Spark SQL's
Hive-compatible execution engine, calculating:
  1. Rating Central Tendency & Dispersion: Count, Mean, Variance, StdDev.
  2. Interaction Sparsity: Matrix dimensions & sparsity percentage.
  3. Top Performing Genres: Weighted ratings & volume.
  4. Benchmark Movies: Highest rated with statistical support.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import time
from typing import Dict, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "spark"))
import init_spark

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

OUTPUT_PATH = os.path.join(BASE_DIR, "logs", "hive_analytics_report.json")


def execute_hive_analytics() -> Dict[str, Any]:
    start_time = time.time()
    spark = (
        SparkSession.builder
        .appName("CineAI-Hive-Descriptive-Analytics")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    raw_dir = os.path.join(BASE_DIR, "data", "raw")
    ratings_path = os.path.join(raw_dir, "ratings.csv")
    movies_path = os.path.join(raw_dir, "movies.csv")

    if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
        spark.stop()
        return {"error": "Raw MovieLens CSV files not found."}

    # Register temporary Hive-compatible views (Schema on Read)
    ratings_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(ratings_path)
    )
    ratings_df.createOrReplaceTempView("ext_ratings")

    movies_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(movies_path)
    )
    movies_df.createOrReplaceTempView("ext_movies")

    # 1. Descriptive statistics
    q1 = """
        SELECT 
            COUNT(*) AS total_ratings,
            ROUND(AVG(rating), 3) AS mean_rating,
            ROUND(VARIANCE(rating), 3) AS variance_rating,
            ROUND(STDDEV(rating), 3) AS stddev_rating,
            MIN(rating) AS min_rating,
            MAX(rating) AS max_rating
        FROM ext_ratings
    """
    stats_row = spark.sql(q1).first().asDict()

    # 2. Matrix Sparsity
    q2 = """
        SELECT 
            COUNT(DISTINCT userId) AS active_users,
            COUNT(DISTINCT movieId) AS active_movies,
            COUNT(*) AS total_ratings
        FROM ext_ratings
    """
    dim_row = spark.sql(q2).first().asDict()
    total_cells = dim_row["active_users"] * dim_row["active_movies"]
    sparsity = round(1.0 - (dim_row["total_ratings"] / total_cells), 5) if total_cells > 0 else 0

    # 3. Genre statistics
    q3 = """
        SELECT 
            m.genres AS genre_group,
            COUNT(r.rating) AS total_reviews,
            ROUND(AVG(r.rating), 2) AS avg_score,
            ROUND(STDDEV(r.rating), 2) AS score_volatility
        FROM ext_ratings r
        JOIN ext_movies m ON (r.movieId = m.movieId)
        GROUP BY m.genres
        HAVING COUNT(r.rating) >= 500
        ORDER BY avg_score DESC
        LIMIT 10
    """
    genre_rows = [r.asDict() for r in spark.sql(q3).collect()]

    # 4. Top 5 Movies
    q4 = """
        SELECT 
            m.movieId,
            m.title,
            COUNT(r.rating) AS vote_count,
            ROUND(AVG(r.rating), 2) AS average_score
        FROM ext_ratings r
        JOIN ext_movies m ON (r.movieId = m.movieId)
        GROUP BY m.movieId, m.title
        HAVING COUNT(r.rating) >= 1000
        ORDER BY average_score DESC, vote_count DESC
        LIMIT 5
    """
    movie_rows = [r.asDict() for r in spark.sql(q4).collect()]

    spark.stop()
    duration_s = round(time.time() - start_time, 2)

    report = {
        "execution_engine": "HiveQL via Spark SQL Metastore",
        "runtime_seconds": duration_s,
        "descriptive_statistics": stats_row,
        "matrix_dimensions": {
            "users": dim_row["active_users"],
            "movies": dim_row["active_movies"],
            "total_ratings": dim_row["total_ratings"],
            "sparsity_pct": round(sparsity * 100, 2),
        },
        "top_genres": genre_rows,
        "top_movies": movie_rows,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main():
    print("=" * 70)
    print("🐝  BDA Experiment 6: Hive Database & Descriptive Analytics")
    print("=" * 70)
    report = execute_hive_analytics()

    stats = report.get("descriptive_statistics", {})
    dims = report.get("matrix_dimensions", {})

    print(f"Total Ratings: {stats.get('total_ratings'):,}")
    print(f"Mean Rating: {stats.get('mean_rating')} ★ (StdDev: {stats.get('stddev_rating')})")
    print(f"Rating Variance: {stats.get('variance_rating')}")
    print(f"Matrix Sparsity: {dims.get('sparsity_pct')}% ({dims.get('users')} users × {dims.get('movies')} movies)")

    print("\n🏆 Top Rated Genre Groups (Hive Aggregation):")
    for g in report.get("top_genres", [])[:4]:
        print(f"   • {g['genre_group'].ljust(35)} : {g['avg_score']} ★ ({g['total_reviews']:,} reviews)")

    print(f"\nExecution finished in {report.get('runtime_seconds')}s. Saved to logs/hive_analytics_report.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
