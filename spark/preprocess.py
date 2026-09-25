"""
spark/preprocess.py
─────────────────────────────────────────────────────────────────────────────
PySpark preprocessing pipeline:
  1. Load ratings.csv and movies.csv from data/raw/
  2. Type-cast and clean data (remove nulls, filter low-count users/movies)
  3. Re-index movieId to consecutive integers (required by ALS)
  4. 80/20 stratified train/test split
  5. Write processed Parquet files to data/processed/

Run:
    python spark/preprocess.py
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import init_spark

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, FloatType, LongType
from pyspark.ml.feature import StringIndexer

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def create_spark_session() -> SparkSession:
    """Create a local Spark session (YARN/K8s-ready via config change)."""
    return (
        SparkSession.builder
        .appName("MovieRecommender-Preprocess")
        .master("local[*]")                          # ← change to yarn / k8s for cluster
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .config("spark.sql.shuffle.partitions", "50")
        .config("spark.ui.showConsoleProgress", "true")
        .getOrCreate()
    )


def load_and_clean_ratings(spark: SparkSession) -> "DataFrame":
    """Load ratings CSV and apply quality filters."""
    print("\n📂  Loading ratings …")
    df = (
        spark.read
        .option("header", "true")
        .csv(os.path.join(RAW_DIR, "ratings.csv"))
        .select(
            F.col("userId").cast(IntegerType()),
            F.col("movieId").cast(IntegerType()),
            F.col("rating").cast(FloatType()),
            F.col("timestamp").cast(LongType()),
        )
        .dropna()
    )

    raw_count = df.count()
    print(f"  Raw ratings: {raw_count:,}")

    # Filter: keep users with ≥20 ratings, movies with ≥5 ratings
    user_counts = df.groupBy("userId").count().filter(F.col("count") >= 20)
    movie_counts = df.groupBy("movieId").count().filter(F.col("count") >= 5)

    df = (
        df.join(user_counts.select("userId"), "userId")
          .join(movie_counts.select("movieId"), "movieId")
    )

    clean_count = df.count()
    print(f"  Clean ratings (≥20 per user, ≥5 per movie): {clean_count:,}")
    print(f"  Retention rate: {clean_count/raw_count*100:.1f}%")

    return df


def load_movies(spark: SparkSession) -> "DataFrame":
    """Load and clean the movies metadata."""
    print("\n🎬  Loading movies …")
    df = (
        spark.read
        .option("header", "true")
        .csv(os.path.join(RAW_DIR, "movies.csv"))
        .select(
            F.col("movieId").cast(IntegerType()),
            F.col("title"),
            F.col("genres"),
        )
        .dropna(subset=["movieId", "title"])
    )
    print(f"  Movies loaded: {df.count():,}")
    return df


def reindex_movies(ratings_df: "DataFrame", movies_df: "DataFrame"):
    """
    ALS needs consecutive integer IDs.
    Map original movieId → sequential movieIndex (0-based).
    """
    indexer = StringIndexer(inputCol="movieId_str", outputCol="movieIndex")

    ratings_with_str = ratings_df.withColumn("movieId_str", F.col("movieId").cast("string"))
    model = indexer.fit(ratings_with_str)
    ratings_indexed = model.transform(ratings_with_str).withColumn(
        "movieIndex", F.col("movieIndex").cast(IntegerType())
    )

    # Build mapping table: movieIndex ↔ movieId
    mapping = (
        ratings_indexed.select("movieId", "movieIndex")
        .distinct()
        .join(movies_df, "movieId", "left")
    )

    return ratings_indexed, mapping, model


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    # ── Load & clean ─────────────────────────────────────────────────────────
    ratings = load_and_clean_ratings(spark)
    movies = load_movies(spark)

    # ── Re-index for ALS ─────────────────────────────────────────────────────
    print("\n🔢  Re-indexing movie IDs for ALS …")
    ratings_indexed, mapping, _ = reindex_movies(ratings, movies)

    # ── Train / Test split (80/20) ───────────────────────────────────────────
    print("\n✂️   Splitting train/test (80/20) …")
    train, test = ratings_indexed.randomSplit([0.8, 0.2], seed=42)
    train_count = train.count()
    test_count = test.count()
    print(f"  Train: {train_count:,}  |  Test: {test_count:,}")

    # ── Write Parquet ─────────────────────────────────────────────────────────
    print("\n💾  Writing processed data …")
    train_path = os.path.join(PROCESSED_DIR, "train.parquet")
    test_path = os.path.join(PROCESSED_DIR, "test.parquet")
    mapping_path = os.path.join(PROCESSED_DIR, "movie_mapping.parquet")

    train.select("userId", "movieIndex", "rating").write.mode("overwrite").parquet(train_path)
    test.select("userId", "movieIndex", "rating").write.mode("overwrite").parquet(test_path)
    mapping.write.mode("overwrite").parquet(mapping_path)

    # ── Save dataset stats ────────────────────────────────────────────────────
    stats = {
        "num_users": ratings.select("userId").distinct().count(),
        "num_movies": ratings.select("movieId").distinct().count(),
        "num_ratings_total": ratings.count(),
        "num_ratings_train": train_count,
        "num_ratings_test": test_count,
        "avg_rating": float(ratings.agg(F.avg("rating")).first()[0]),
        "sparsity": 1 - (ratings.count() / (
            ratings.select("userId").distinct().count() *
            ratings.select("movieId").distinct().count()
        )),
    }
    stats_path = os.path.join(LOGS_DIR, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✅  Preprocessing complete!")
    print(f"   Dataset stats saved to: {stats_path}")
    for k, v in stats.items():
        print(f"   {k}: {v}")

    spark.stop()


if __name__ == "__main__":
    main()
