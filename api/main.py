"""
api/main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI REST backend for the Movie Recommendation Engine.

Endpoints:
  GET  /                            → API info
  GET  /health                      → Health check
  GET  /movies/search?q=&limit=     → Fuzzy movie search
  POST /recommendations             → Get recs from user ratings
  GET  /recommendations/{user_id}   → Pre-computed recs for known user
  GET  /model/metrics               → ALS training metrics
  GET  /dataset/stats               → Dataset statistics

Run:
    uvicorn api.main:app --reload --port 8000
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.models import (
    RecommendationRequest, RecommendationResponse, RecommendedMovie,
    MovieSearchResponse, MovieSearchResult,
    ModelMetrics, HealthResponse,
)
from api.recommender import store, search_movies, recommend, get_precomputed_recs, filter_rated_candidates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Recommendation Engine API …")
    store.load()
    logger.info("✅ API ready")
    yield
    logger.info("API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="🎬 Movie Recommendation Engine",
    description=(
        "Large-scale distributed recommendation system powered by "
        "**PySpark MLlib ALS** (Alternating Least Squares).\n\n"
        "Built for BDA (Big Data Analytics) demonstration."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    """API information and quick links."""
    return {
        "name": "Movie Recommendation Engine API",
        "version": "1.0.0",
        "framework": "PySpark MLlib ALS",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "search_movies": "GET /movies/search?q=inception",
            "get_recommendations": "POST /recommendations",
            "user_recommendations": "GET /recommendations/{user_id}",
            "model_metrics": "GET /model/metrics",
            "dataset_stats": "GET /dataset/stats",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health():
    """Health check — verifies model and data are loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=store.is_model_available,
        dataset_loaded=store._loaded,
        total_users=store.total_users,
        total_movies=store.total_movies,
    )


@app.get("/movies/search", response_model=MovieSearchResponse, tags=["Movies"])
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
):
    """
    Fuzzy search for movies by title.
    Returns movie metadata including average rating and rating count.
    """
    results_raw = search_movies(q, limit=limit)
    results = [MovieSearchResult(**r) for r in results_raw]
    return MovieSearchResponse(results=results, total=len(results), query=q)


@app.post("/recommendations", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(request: RecommendationRequest):
    """
    **Core endpoint**: Given a list of movie ratings, return personalised recommendations.

    - **≥5 ratings**: Uses genre-similarity scoring (ALS-approximation, fast)
    - **<5 ratings**: Cold-start genre-popularity fallback
    - **Known user**: Pre-computed ALS batch recommendations (O(1) lookup)
    """
    if not request.ratings:
        raise HTTPException(status_code=400, detail="At least 1 rating required")

    rated_movies = [
        {
            "movie_id": r.movie_id,
            "title": r.title,
            "rating": r.rating,
            "genres": r.genres or "",
        }
        for r in request.ratings
    ]

    raw_recs, is_cold_start = recommend(rated_movies, top_n=request.top_n)

    if not raw_recs:
        raise HTTPException(status_code=404, detail="No recommendations found")

    # Exclude already-rated movies
    if request.exclude_rated:
        raw_recs = filter_rated_candidates(raw_recs, rated_movies)

    recommendations = [RecommendedMovie(**r) for r in raw_recs[: request.top_n]]

    return RecommendationResponse(
        recommendations=recommendations,
        is_cold_start=is_cold_start,
        total_returned=len(recommendations),
    )


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_user_recommendations(
    user_id: int,
    top_n: int = Query(10, ge=1, le=50, description="Number of recommendations"),
):
    """
    Retrieve pre-computed ALS recommendations for a known user ID.
    Users are from the MovieLens dataset (valid range: 1–6040 for 1M).
    """
    recs = get_precomputed_recs(user_id, top_n)
    if not recs:
        raise HTTPException(
            status_code=404,
            detail=f"No pre-computed recommendations for user {user_id}. "
                   f"Run spark/generate_recs.py first, or use POST /recommendations."
        )

    recommendations = [RecommendedMovie(**r) for r in recs]
    return RecommendationResponse(
        user_id=user_id,
        recommendations=recommendations,
        is_cold_start=False,
        total_returned=len(recommendations),
    )


@app.get("/model/metrics", response_model=ModelMetrics, tags=["Model"])
async def model_metrics():
    """
    Returns ALS model training metrics:
    - Best hyperparameters (rank, regParam, maxIter)
    - Test set RMSE, MAE
    - All cross-validation results
    """
    if not store.training_metrics:
        raise HTTPException(
            status_code=404,
            detail="Training metrics not found. Run spark/train_als.py first."
        )
    metrics = store.training_metrics
    metrics["dataset_stats"] = store.dataset_stats
    return ModelMetrics(**metrics)


@app.get("/dataset/stats", tags=["Model"])
async def dataset_stats():
    """Returns dataset statistics from preprocessing."""
    if not store.dataset_stats:
        raise HTTPException(
            status_code=404,
            detail="Dataset stats not found. Run spark/preprocess.py first."
        )
    return store.dataset_stats


@app.get("/model/cv-results", tags=["Model"])
async def cv_results():
    """Returns all cross-validation results sorted by RMSE (for dashboard charts)."""
    if not store.training_metrics:
        raise HTTPException(status_code=404, detail="Training metrics not found.")
    return {
        "cv_results": store.training_metrics.get("cv_results", []),
        "best": store.training_metrics.get("best_hyperparams", {}),
    }


# ── BDA Lab Experiment Endpoints ─────────────────────────────────────────────

@app.get("/bda/experiments", tags=["BDA Experiments"])
async def list_bda_experiments():
    """List all BDA experiments implemented across CineAI."""
    return {
        "status": "success",
        "experiments_covered": [
            {"id": "Exp-1", "name": "HDFS Ingestion & Command Suite", "module": "bda_lab/hdfs_ingest.bat"},
            {"id": "Exp-4", "name": "Hadoop MapReduce WordCount & Genre Frequencies", "module": "bda_lab/mapreduce/"},
            {"id": "Exp-6", "name": "Hive Database & Descriptive Statistics", "module": "bda_lab/hive_analytics.hql"},
            {"id": "Exp-7", "name": "MongoDB NoSQL Document Store", "module": "bda_lab/mongo_manager.py"},
            {"id": "Exp-8", "name": "Bloom Filter O(1) Candidate Deduplication", "module": "bda_lab/bloom_filter.py"},
            {"id": "Exp-9", "name": "Flajolet-Martin Streaming Cardinality Estimation", "module": "bda_lab/flajolet_martin.py"},
            {"id": "Exp-10", "name": "Data Visualization using R ggplot2", "module": "bda_lab/visualizations.R"},
            {"id": "Exp-12", "name": "Big Data 5Vs & Distributed Architecture", "module": "ui/app.py"},
            {"id": "Exp-13", "name": "Social Graph Mining: Girvan-Newman & CPM", "module": "bda_lab/graph_mining.py"},
        ]
    }


@app.get("/bda/bloom-filter/demo", tags=["BDA Experiments"])
async def bda_bloom_filter_demo(items: int = Query(300, ge=10, le=2000)):
    """Run live Bloom Filter membership & false positive evaluation."""
    from bda_lab.bloom_filter import BloomFilter
    bf = BloomFilter(expected_items=items, false_positive_rate=0.01)
    for i in range(1, items + 1):
        bf.add(f"movie_{i}")

    false_negatives = sum(1 for i in range(1, items + 1) if not bf.contains(f"movie_{i}"))
    test_unseen = 5000
    false_positives = sum(1 for i in range(items + 1000, items + 1000 + test_unseen) if bf.contains(f"movie_{i}"))
    fp_rate = (false_positives / test_unseen) * 100

    stats = bf.get_stats()
    stats["false_negatives"] = false_negatives
    stats["tested_unseen_items"] = test_unseen
    stats["false_positives_detected"] = false_positives
    stats["observed_fp_rate_pct"] = round(fp_rate, 3)
    return stats


@app.get("/bda/flajolet-martin/stats", tags=["BDA Experiments"])
async def bda_flajolet_martin_stats():
    """Run streaming Flajolet-Martin distinct user cardinality estimation."""
    from bda_lab.flajolet_martin import FlajoletMartin
    ratings_path = os.path.join(BASE_DIR, "data", "raw", "ratings.csv")

    fm = FlajoletMartin(num_hashes=32, num_groups=4)
    exact_users = set()

    if os.path.exists(ratings_path):
        import csv
        with open(ratings_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            count = 0
            for r in reader:
                if r:
                    fm.add(r[0])
                    exact_users.add(r[0])
                    count += 1
                    if count >= 100000:
                        break

    return fm.get_diagnostics(exact_distinct_count=len(exact_users))


@app.get("/bda/mongodb/status", tags=["BDA Experiments"])
async def bda_mongodb_status():
    """Retrieve MongoDB NoSQL connection status and collection statistics."""
    from bda_lab.mongo_manager import mongo_manager
    return mongo_manager.get_stats()


@app.get("/bda/graph-mining/summary", tags=["BDA Experiments"])
async def bda_graph_mining_summary():
    """Retrieve Girvan-Newman communities and CPM overlapping cluster analysis."""
    graph_log = os.path.join(LOGS_DIR, "graph_communities.json")
    if os.path.exists(graph_log):
        with open(graph_log, "r", encoding="utf-8") as f:
            return json.load(f)

    from bda_lab.graph_mining import execute_graph_analysis
    return execute_graph_analysis()


@app.get("/bda/hive-stats", tags=["BDA Experiments"])
async def bda_hive_stats():
    """Retrieve Hive descriptive analytics report."""
    hive_log = os.path.join(LOGS_DIR, "hive_analytics_report.json")
    if os.path.exists(hive_log):
        with open(hive_log, "r", encoding="utf-8") as f:
            return json.load(f)

    return {"message": "Run 'python bda_lab/run_hive_stats.py' to generate Hive statistics."}


@app.get("/bda/mapreduce/results", tags=["BDA Experiments"])
async def bda_mapreduce_results():
    """Retrieve Hadoop MapReduce word count and genre frequency outputs."""
    mr_log = os.path.join(LOGS_DIR, "mapreduce_results.json")
    if os.path.exists(mr_log):
        with open(mr_log, "r", encoding="utf-8") as f:
            return json.load(f)

    from bda_lab.mapreduce.run_mapreduce import run_simulated_mapreduce
    return run_simulated_mapreduce()

