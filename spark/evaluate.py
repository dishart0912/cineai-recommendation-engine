"""
spark/evaluate.py
─────────────────────────────────────────────────────────────────────────────
Load saved ALS model and run comprehensive evaluation on the test set.
Produces an evaluation_report.json with:
  - RMSE, MAE
  - Precision@K, Recall@K  (ranking quality)
  - Coverage (catalog / user)
  - Rating distribution analysis

Run:
    python spark/evaluate.py
    python spark/evaluate.py --k 10
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
from pyspark.ml.evaluation import RegressionEvaluator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")


def create_spark_session():
    return (
        SparkSession.builder
        .appName("MovieRecommender-Evaluate")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def precision_at_k(recommendations_df, test_df, k: int = 10) -> dict:
    """
    Compute Precision@K and Recall@K.
    recommendations_df: userId, recommendations (array of (movieIndex, rating))
    test_df: userId, movieIndex, rating
    """
    # Relevant = movies rated ≥ 4.0 in test set
    relevant = (
        test_df.filter(F.col("rating") >= 4.0)
        .groupBy("userId")
        .agg(F.collect_set("movieIndex").alias("relevant_items"))
    )

    # Top-K predicted items per user
    top_k = (
        recommendations_df
        .select("userId", F.slice("recommendations.movieIndex", 1, k).alias("predicted_items"))
    )

    joined = top_k.join(relevant, "userId", "inner")

    # Precision@K = |relevant ∩ predicted| / K
    # Recall@K    = |relevant ∩ predicted| / |relevant|
    def intersect_size(pred, rel):
        return len(set(pred) & set(rel))

    from pyspark.sql.types import IntegerType
    from pyspark.sql.functions import udf
    intersect_udf = udf(lambda pred, rel: intersect_size(pred, rel), IntegerType())

    metrics_df = joined.withColumn(
        "hits", intersect_udf(F.col("predicted_items"), F.col("relevant_items"))
    ).withColumn(
        "precision_k", F.col("hits") / k
    ).withColumn(
        "recall_k", F.col("hits") / F.size("relevant_items")
    )

    agg = metrics_df.agg(
        F.avg("precision_k").alias("precision_at_k"),
        F.avg("recall_k").alias("recall_at_k"),
    ).first()

    return {
        f"precision@{k}": round(float(agg["precision_at_k"]), 4),
        f"recall@{k}": round(float(agg["recall_at_k"]), 4),
    }


def main(k: int = 10):
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, "als_model")
    if not os.path.exists(model_path):
        print("❌  Model not found. Run spark/train_als.py first.")
        sys.exit(1)

    print(f"\n📦  Loading model from: {model_path}")
    model = ALSModel.load(model_path)

    # ── Load test data ────────────────────────────────────────────────────────
    print("📂  Loading test data …")
    test = spark.read.parquet(os.path.join(PROCESSED_DIR, "test.parquet"))

    # ── Rating-level metrics ──────────────────────────────────────────────────
    print("\n📊  Computing rating-level metrics …")
    predictions = model.transform(test).dropna(subset=["prediction"])

    rmse_eval = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    mae_eval = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")
    r2_eval = RegressionEvaluator(metricName="r2", labelCol="rating", predictionCol="prediction")

    rmse = rmse_eval.evaluate(predictions)
    mae = mae_eval.evaluate(predictions)
    r2 = r2_eval.evaluate(predictions)

    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")

    # ── Top-N recommendation metrics ──────────────────────────────────────────
    print(f"\n🎯  Computing Precision@{k} and Recall@{k} …")
    # Generate top-K recs for all test users
    test_users = test.select("userId").distinct()
    recs_df = model.recommendForUserSubset(test_users, k)
    ranking_metrics = precision_at_k(recs_df, test, k)
    print(f"  Precision@{k}: {ranking_metrics[f'precision@{k}']}")
    print(f"  Recall@{k}   : {ranking_metrics[f'recall@{k}']}")

    # ── Coverage metrics ──────────────────────────────────────────────────────
    print("\n📈  Computing coverage …")
    total_users = test.select("userId").distinct().count()
    covered_users = predictions.select("userId").distinct().count()
    user_coverage = covered_users / total_users * 100

    total_items = test.select("movieIndex").distinct().count()
    recommended_items = recs_df.select(F.explode("recommendations.movieIndex").alias("movieIndex")).distinct().count()
    catalog_coverage = recommended_items / total_items * 100

    print(f"  User coverage: {user_coverage:.1f}%")
    print(f"  Catalog coverage: {catalog_coverage:.1f}%")

    # ── Rating distribution ───────────────────────────────────────────────────
    rating_dist = (
        test.groupBy("rating")
        .count()
        .orderBy("rating")
        .rdd.map(lambda r: {"rating": float(r["rating"]), "count": int(r["count"])})
        .collect()
    )

    # ── Save report ───────────────────────────────────────────────────────────
    report = {
        "rating_metrics": {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
        },
        "ranking_metrics": ranking_metrics,
        "coverage": {
            "user_coverage_pct": round(user_coverage, 2),
            "catalog_coverage_pct": round(catalog_coverage, 2),
        },
        "rating_distribution": rating_dist,
    }

    report_path = os.path.join(LOGS_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅  Evaluation report saved to: {report_path}")
    spark.stop()


if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10, help="K for Precision@K / Recall@K")
    args = parser.parse_args()
    main(k=args.k)
