"""
spark/train_als.py
─────────────────────────────────────────────────────────────────────────────
Distributed ALS (Alternating Least Squares) training with hyperparameter
tuning via CrossValidator — all running on Spark distributed executors.

Pipeline:
  1. Load preprocessed train/test Parquet from data/processed/
  2. Define ALS model and hyperparameter grid
  3. Run 3-fold CrossValidator (distributed across Spark workers)
  4. Evaluate best model on held-out test set
  5. Save model to models/als_model/
  6. Export training metrics to logs/training_metrics.json

Run:
    python spark/train_als.py
    python spark/train_als.py --fast        # quick 1-param grid for demo
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import init_spark

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("MovieRecommender-ALS-Training")
        .master("local[*]")
        .config("spark.driver.memory", "3g")
        .config("spark.executor.memory", "3g")
        .config("spark.sql.shuffle.partitions", "50")
        .getOrCreate()
    )


def build_param_grid(als: ALS, fast: bool = False):
    """
    Construct the hyperparameter search grid.
    fast=True → minimal grid for quick demo runs
    fast=False → full grid for thorough tuning
    """
    if fast:
        grid = (
            ParamGridBuilder()
            .addGrid(als.rank, [10, 50])
            .addGrid(als.regParam, [0.1])
            .addGrid(als.maxIter, [10])
            .build()
        )
    else:
        grid = (
            ParamGridBuilder()
            .addGrid(als.rank, [10, 50, 100])           # Latent factors
            .addGrid(als.regParam, [0.01, 0.1, 1.0])    # L2 regularisation
            .addGrid(als.maxIter, [10, 20])              # ALS iterations
            .build()
        )
    return grid


def main(fast: bool = False):
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n📂  Loading preprocessed data …")
    train = spark.read.parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    test = spark.read.parquet(os.path.join(PROCESSED_DIR, "test.parquet"))
    print(f"  Train samples: {train.count():,}")
    print(f"  Test  samples: {test.count():,}")

    # ── 2. ALS Model ──────────────────────────────────────────────────────────
    print("\n🔧  Configuring ALS model …")
    als = ALS(
        userCol="userId",
        itemCol="movieIndex",
        ratingCol="rating",
        coldStartStrategy="drop",   # Handles unknown users/items at inference
        nonnegative=True,           # Enforce non-negative factors
        implicitPrefs=False,        # Explicit ratings (0–5 scale)
    )

    # ── 3. Evaluator ──────────────────────────────────────────────────────────
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction",
    )

    # ── 4. Param Grid ─────────────────────────────────────────────────────────
    param_grid = build_param_grid(als, fast=fast)
    total_combos = len(param_grid)
    print(f"  Hyperparameter combinations: {total_combos}")
    print(f"  Cross-validation folds: 3")
    print(f"  Total Spark jobs: {total_combos * 3}")

    # ── 5. CrossValidator (runs distributed across Spark workers) ─────────────
    cv = CrossValidator(
        estimator=als,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=4,              # Parallel param evaluation
        seed=42,
    )

    print(f"\n🚀  Starting distributed hyperparameter tuning …")
    print(f"   (This may take 3–10 minutes depending on hardware)")
    t_start = time.time()
    cv_model = cv.fit(train)
    elapsed = time.time() - t_start
    print(f"\n⏱️   Training completed in {elapsed:.1f}s")

    # ── 6. Best model ─────────────────────────────────────────────────────────
    best_model = cv_model.bestModel
    best_rank = best_model.rank
    best_reg = best_model._java_obj.parent().getRegParam()
    best_max_iter = best_model._java_obj.parent().getMaxIter()

    print(f"\n🏆  Best hyperparameters:")
    print(f"   rank     = {best_rank}")
    print(f"   regParam = {best_reg}")
    print(f"   maxIter  = {best_max_iter}")

    # ── 7. Evaluate on test set ───────────────────────────────────────────────
    print("\n📊  Evaluating on test set …")
    test_predictions = best_model.transform(test).dropna(subset=["prediction"])

    rmse = evaluator.evaluate(test_predictions)
    mae_evaluator = RegressionEvaluator(
        metricName="mae", labelCol="rating", predictionCol="prediction"
    )
    mae = mae_evaluator.evaluate(test_predictions)

    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")

    # Coverage: % of test users that received predictions
    total_test_users = test.select("userId").distinct().count()
    covered_users = test_predictions.select("userId").distinct().count()
    coverage = covered_users / total_test_users * 100
    print(f"  Coverage: {coverage:.1f}%")

    # ── 8. All CV results ─────────────────────────────────────────────────────
    avg_metrics = cv_model.avgMetrics
    cv_results = []
    for params, metric in zip(param_grid, avg_metrics):
        cv_results.append({
            "rank": params[als.rank],
            "regParam": params[als.regParam],
            "maxIter": params[als.maxIter],
            "cv_rmse": round(metric, 4),
        })
    cv_results.sort(key=lambda x: x["cv_rmse"])

    # ── 9. Save model ─────────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, "als_model")
    print(f"\n💾  Saving best model to: {model_path}")
    best_model.write().overwrite().save(model_path)

    # ── 10. Save metrics ──────────────────────────────────────────────────────
    metrics = {
        "best_hyperparams": {
            "rank": best_rank,
            "regParam": best_reg,
            "maxIter": best_max_iter,
        },
        "test_metrics": {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "coverage_pct": round(coverage, 2),
        },
        "training_time_seconds": round(elapsed, 1),
        "total_param_combinations": total_combos,
        "cv_folds": 3,
        "cv_results": cv_results,
    }

    metrics_path = os.path.join(LOGS_DIR, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅  All done!")
    print(f"   Model    : {model_path}")
    print(f"   Metrics  : {metrics_path}")
    print(f"   RMSE={rmse:.4f}  MAE={mae:.4f}  Coverage={coverage:.1f}%")

    spark.stop()
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ALS recommendation model")
    parser.add_argument("--fast", action="store_true",
                        help="Use minimal param grid for quick demo (2 combos vs 18)")
    args = parser.parse_args()
    main(fast=args.fast)
