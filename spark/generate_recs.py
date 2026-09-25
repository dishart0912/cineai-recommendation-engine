"""
spark/generate_recs.py
─────────────────────────────────────────────────────────────────────────────
Batch recommendation generation for ALL users using the trained ALS model.
Output saved as Parquet for fast API lookups.

Also exports a lightweight JSON cache for the Streamlit UI's cold-start demo.

Run:
    python spark/generate_recs.py
    python spark/generate_recs.py --top-n 20
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import init_spark

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALSModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")


def create_spark_session():
    return (
        SparkSession.builder
        .appName("MovieRecommender-BatchRecs")
        .master("local[*]")
        .config("spark.driver.memory", "3g")
        .getOrCreate()
    )


def main(top_n: int = 10):
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, "als_model")
    if not os.path.exists(model_path):
        print("❌  Model not found. Run spark/train_als.py first.")
        sys.exit(1)

    print(f"\n📦  Loading ALS model …")
    model = ALSModel.load(model_path)

    # ── Load movie mapping ────────────────────────────────────────────────────
    print("📂  Loading movie mapping …")
    mapping = spark.read.parquet(os.path.join(PROCESSED_DIR, "movie_mapping.parquet"))
    train = spark.read.parquet(os.path.join(PROCESSED_DIR, "train.parquet"))

    # ── Generate top-N recs for all users ─────────────────────────────────────
    all_users = train.select("userId").distinct()
    user_count = all_users.count()
    print(f"\n🔮  Generating top-{top_n} recommendations for {user_count:,} users …")

    recs = model.recommendForUserSubset(all_users, top_n)

    # ── Explode and join with movie titles ────────────────────────────────────
    recs_flat = (
        recs
        .select("userId", F.explode("recommendations").alias("rec"))
        .select(
            "userId",
            F.col("rec.movieIndex").alias("movieIndex"),
            F.col("rec.rating").alias("predicted_rating"),
        )
        .join(mapping.select("movieIndex", "movieId", "title", "genres"), "movieIndex", "left")
    )

    # ── Save as Parquet (fast API lookups) ────────────────────────────────────
    recs_path = os.path.join(DATA_DIR, "recommendations.parquet")
    print(f"\n💾  Saving recommendations to: {recs_path}")
    recs_flat.write.mode("overwrite").parquet(recs_path)

    # ── Export lightweight JSON for UI / API cache (sample: 1000 users) ───────
    print("\n📝  Exporting JSON cache (sample 1000 users) …")
    sample = (
        recs_flat
        .filter(F.col("userId") <= 1000)
        .orderBy("userId", F.desc("predicted_rating"))
        .select(
            "userId",
            "movieId",
            F.coalesce(F.col("title"), F.lit("Unknown")).alias("title"),
            F.coalesce(F.col("genres"), F.lit("Unknown")).alias("genres"),
            F.round("predicted_rating", 3).alias("predicted_rating"),
        )
        .rdd
        .map(lambda r: r.asDict())
        .collect()
    )

    # Group by user
    user_recs = {}
    for row in sample:
        uid = str(row["userId"])
        if uid not in user_recs:
            user_recs[uid] = []
        user_recs[uid].append({
            "movieId": row["movieId"],
            "title": row["title"],
            "genres": row["genres"],
            "predicted_rating": row["predicted_rating"],
        })

    cache_path = os.path.join(DATA_DIR, "recs_cache.json")
    with open(cache_path, "w") as f:
        json.dump(user_recs, f, indent=2)

    print(f"   Cached {len(user_recs)} users")

    # ── Genre-based popularity for cold-start ─────────────────────────────────
    print("\n❄️   Building cold-start genre popularity index …")
    genre_popular = (
        train
        .join(mapping.select("movieIndex", "movieId", "title", "genres"), "movieIndex", "left")
        .groupBy("movieIndex", "movieId", "title", "genres")
        .agg(
            F.avg("rating").alias("avg_rating"),
            F.count("rating").alias("num_ratings"),
        )
        .orderBy(F.desc("num_ratings"))
        .limit(500)
        .rdd.map(lambda r: r.asDict())
        .collect()
    )

    cold_start_path = os.path.join(DATA_DIR, "cold_start_popular.json")
    with open(cold_start_path, "w") as f:
        json.dump(genre_popular, f, indent=2)

    print(f"\n✅  Batch generation complete!")
    print(f"   Recommendations parquet : {recs_path}")
    print(f"   JSON cache              : {cache_path}")
    print(f"   Cold-start index        : {cold_start_path}")

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=10, help="Number of recs per user")
    args = parser.parse_args()
    main(top_n=args.top_n)
