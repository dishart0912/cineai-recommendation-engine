"""
api/recommender.py
─────────────────────────────────────────────────────────────────────────────
Recommendation inference engine used by the FastAPI app.

Strategy:
  1. Pre-computed recs (recs_cache.json) for known users → O(1) lookup
  2. Real-time ALS inference using PySpark (for new user ratings)
  3. Cold-start fallback: genre-similarity + popularity for <5 ratings
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import math
import logging
from typing import List, Optional, Dict
from functools import lru_cache

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

try:
    from bda_lab.bloom_filter import BloomFilter
except ImportError:
    BloomFilter = None

try:
    from bda_lab.mongo_manager import mongo_manager
except ImportError:
    mongo_manager = None


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (cached in-memory at startup)
# ─────────────────────────────────────────────────────────────────────────────

class DataStore:
    """Lightweight in-memory data store loaded at startup."""

    def __init__(self):
        self.recs_cache: Dict[str, list] = {}
        self.cold_start: List[dict] = []
        self.movies: Dict[int, dict] = {}          # movieId → {title, genres, ...}
        self.training_metrics: dict = {}
        self.dataset_stats: dict = {}
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        logger.info("Loading data store …")

        # Recommendations cache
        cache_path = os.path.join(DATA_DIR, "recs_cache.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                self.recs_cache = json.load(f)
            logger.info(f"Loaded recs cache for {len(self.recs_cache)} users")

        # Movie catalog (load all_movies.json if exists, else cold_start_popular)
        all_path = os.path.join(DATA_DIR, "all_movies.json")
        cold_path = os.path.join(DATA_DIR, "cold_start_popular.json")

        if os.path.exists(all_path):
            with open(all_path, encoding="utf-8") as f:
                all_movies = json.load(f)
            self.cold_start = all_movies
            for m in all_movies:
                mid = m.get("movieId")
                if mid:
                    self.movies[mid] = m
            logger.info(f"Loaded full catalog of {len(self.movies)} movies into API")
        elif os.path.exists(cold_path):
            with open(cold_path, encoding="utf-8") as f:
                self.cold_start = json.load(f)
            for m in self.cold_start:
                mid = m.get("movieId")
                if mid:
                    self.movies[mid] = m
            logger.info(f"Loaded {len(self.cold_start)} cold-start movies")

        # Training metrics
        metrics_path = os.path.join(LOGS_DIR, "training_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                self.training_metrics = json.load(f)

        # Dataset stats
        stats_path = os.path.join(LOGS_DIR, "dataset_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                self.dataset_stats = json.load(f)

        self._loaded = True
        logger.info("Data store ready.")

    @property
    def is_model_available(self) -> bool:
        model_path = os.path.join(MODELS_DIR, "als_model")
        return os.path.exists(model_path)

    @property
    def total_users(self) -> int:
        return int(self.dataset_stats.get("num_users", len(self.recs_cache)))

    @property
    def total_movies(self) -> int:
        return int(self.dataset_stats.get("num_movies", len(self.movies)))


# Global singleton
store = DataStore()


# ─────────────────────────────────────────────────────────────────────────────
# Movie search (fuzzy, in-memory)
# ─────────────────────────────────────────────────────────────────────────────

def search_movies(query: str, limit: int = 20) -> List[dict]:
    """Fuzzy title search over the in-memory movie catalog."""
    store.load()
    q = query.lower().strip()
    results = []

    for movie in store.cold_start:
        title = (movie.get("title") or "").lower()
        if q in title:
            results.append({
                "movie_id": movie.get("movieId"),
                "title": movie.get("title", ""),
                "genres": movie.get("genres", ""),
                "avg_rating": round(float(movie.get("avg_rating", 0)), 2),
                "num_ratings": int(movie.get("num_ratings", 0)),
                "year": _extract_year(movie.get("title", "")),
            })

    # Sort by relevance (exact prefix match first, then popularity)
    results.sort(key=lambda x: (
        not (x["title"].lower().startswith(q)),
        -x["num_ratings"]
    ))
    return results[:limit]


def _extract_year(title: str) -> Optional[int]:
    import re
    m = re.search(r"\((\d{4})\)", title)
    return int(m.group(1)) if m else None


# ─────────────────────────────────────────────────────────────────────────────
# Filtering and Recommendation Algorithms
# ─────────────────────────────────────────────────────────────────────────────

def filter_rated_candidates(candidates: List[dict], rated_movies: List[dict]) -> List[dict]:
    """
    Bulletproof filtering of already-rated movies using BDA Bloom Filter (Exp 8).
    Checks ID (int and str) and Title (exact lower and year-stripped lower).
    """
    import re
    rated_ids = set()
    rated_titles = set()

    # Initialize Bloom Filter for O(1) probabilistic deduplication
    bf = None
    if BloomFilter is not None and rated_movies:
        bf = BloomFilter(expected_items=max(20, len(rated_movies) * 3), false_positive_rate=0.005)

    for m in rated_movies:
        mid = m.get("movie_id") if m.get("movie_id") is not None else m.get("movieId")
        if mid is not None:
            rated_ids.add(mid)
            rated_ids.add(str(mid).strip())
            if bf:
                bf.add(f"id:{mid}")
                bf.add(f"id:{str(mid).strip()}")
            try:
                rated_ids.add(int(mid))
            except (ValueError, TypeError):
                pass

        t = str(m.get("title", "")).strip().lower()
        if t:
            rated_titles.add(t)
            if bf:
                bf.add(f"title:{t}")
            clean_t = re.sub(r"\s*\(\d{4}\).*", "", t).strip()
            if clean_t:
                rated_titles.add(clean_t)
                if bf:
                    bf.add(f"title:{clean_t}")

    unrated = []
    for m in candidates:
        mid = m.get("movieId") if m.get("movieId") is not None else m.get("movie_id")
        if mid is not None:
            # Check Bloom Filter membership first (O(k))
            if bf and bf.contains(f"id:{mid}"):
                continue
            if mid in rated_ids or str(mid).strip() in rated_ids:
                continue
            try:
                if int(mid) in rated_ids:
                    continue
            except (ValueError, TypeError):
                pass

        t = str(m.get("title", "")).strip().lower()
        if bf and bf.contains(f"title:{t}"):
            continue
        if t in rated_titles:
            continue
        clean_t = re.sub(r"\s*\(\d{4}\).*", "", t).strip()
        if clean_t and clean_t in rated_titles:
            continue

        unrated.append(m)

    return unrated


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cold-start recommendations (< 5 ratings)
# ─────────────────────────────────────────────────────────────────────────────

def get_cold_start_recs(rated_movies: List[dict], top_n: int = 10) -> List[dict]:
    """
    Cold-start strategy for < 5 ratings:
    Scores candidate movies by initial genre overlap with popularity-weighted fallback.
    """
    store.load()

    # Build liked-genre profile
    liked_genres: Dict[str, float] = {}
    for movie in rated_movies:
        r = float(movie.get("rating", 3.5))
        if r >= 3.0:
            genres = (movie.get("genres") or "").split("|")
            weight = r / 5.0
            for g in genres:
                if g and g != "(no genres listed)":
                    liked_genres[g] = liked_genres.get(g, 0.0) + weight

    candidates = filter_rated_candidates(store.cold_start, rated_movies)

    scored = []
    for movie in candidates:
        genres = (movie.get("genres") or "").split("|")
        genre_score = sum(liked_genres.get(g, 0.0) for g in genres)
        popularity = math.log1p(movie.get("num_ratings", 0))
        quality = float(movie.get("avg_rating", 3.0)) / 5.0
        combined = (genre_score * 0.45) + (popularity * 0.35) + (quality * 0.20)
        scored.append((combined, movie))

    scored.sort(key=lambda x: -x[0])
    top_movies = scored[:top_n]

    results = []
    max_score = scored[0][0] if scored else 1.0
    for score, movie in top_movies:
        norm_score = min(99.0, max(50.0, (score / max_score) * 98.0)) if max_score > 0 else 75.0
        results.append({
            "movie_id": movie.get("movieId"),
            "title": movie.get("title", "Unknown"),
            "genres": movie.get("genres", ""),
            "predicted_rating": round(float(movie.get("avg_rating", 3.5)), 2),
            "match_score": round(norm_score, 1),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Collaborative Filtering recommendations (≥ 5 ratings)
# ─────────────────────────────────────────────────────────────────────────────

def get_collaborative_recs(rated_movies: List[dict], top_n: int = 10) -> List[dict]:
    """
    Personalized Collaborative Filtering for ≥ 5 ratings:
    Constructs a centered multi-dimensional user taste vector (positive boost for 4-5★,
    penalties for 1-2★), computes cosine similarity against candidate genre profiles,
    and dampens popularity bias so genuine taste drives recommendations.
    """
    store.load()

    user_taste: Dict[str, float] = {}
    ratings = [float(m.get("rating", 3.5)) for m in rated_movies]
    user_mean = sum(ratings) / len(ratings) if ratings else 3.8

    # Rating-centered weights: (rating - 2.5) gives [-2.0 .. +2.5]
    for m in rated_movies:
        r = float(m.get("rating", 3.5))
        weight = r - 2.5
        for g in (m.get("genres") or "").split("|"):
            if g and g != "(no genres listed)":
                user_taste[g] = user_taste.get(g, 0.0) + weight

    taste_norm = math.sqrt(sum(v * v for v in user_taste.values())) if user_taste else 1.0

    candidates = filter_rated_candidates(store.cold_start, rated_movies)

    scored = []
    for movie in candidates:
        genres = [g for g in (movie.get("genres") or "").split("|") if g and g != "(no genres listed)"]
        if not genres:
            continue

        g_dot = sum(user_taste.get(g, 0.0) for g in genres)
        g_norm = math.sqrt(len(genres))
        sim = g_dot / (taste_norm * g_norm) if (taste_norm * g_norm) > 0 else 0.0

        quality = float(movie.get("avg_rating", 3.0)) / 5.0
        pop = math.log1p(movie.get("num_ratings", 0)) / 10.0

        # Highly personalized: 80% taste vector similarity, 15% quality, 5% popularity
        combined = (sim * 0.80) + (quality * 0.15) + (pop * 0.05)
        scored.append((combined, movie, sim))

    scored.sort(key=lambda x: -x[0])
    top_movies = scored[:top_n]

    results = []
    for combined, movie, sim in top_movies:
        match_score = min(99.0, max(52.0, 52.0 + (sim * 46.0)))
        pred_rating = min(5.0, max(3.5, (user_mean * 0.6) + (float(movie.get("avg_rating", 3.5)) * 0.4)))
        results.append({
            "movie_id": movie.get("movieId"),
            "title": movie.get("title", "Unknown"),
            "genres": movie.get("genres", ""),
            "predicted_rating": round(pred_rating, 2),
            "match_score": round(match_score, 1),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pre-computed recommendation lookup
# ─────────────────────────────────────────────────────────────────────────────

def get_precomputed_recs(user_id: int, top_n: int = 10) -> Optional[List[dict]]:
    """Return pre-computed recommendations for a known user (MongoDB NoSQL with local JSON fallback)."""
    # 1. Try querying MongoDB if online (Exp 7)
    if mongo_manager and mongo_manager.is_connected:
        m_recs = mongo_manager.get_user_recommendations(user_id, top_n)
        if m_recs:
            return m_recs

    # 2. In-memory cache fallback
    store.load()
    cached = store.recs_cache.get(str(user_id))
    if not cached:
        return None

    results = []
    for i, item in enumerate(cached[:top_n]):
        max_rating = 5.0
        norm_score = min(100.0, (item["predicted_rating"] / max_rating) * 100)
        results.append({
            "movie_id": item.get("movieId"),
            "title": item.get("title", "Unknown"),
            "genres": item.get("genres", ""),
            "predicted_rating": round(float(item["predicted_rating"]), 2),
            "match_score": round(norm_score, 1),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point used by FastAPI
# ─────────────────────────────────────────────────────────────────────────────

def recommend(rated_movies: List[dict], top_n: int = 10, user_id: Optional[int] = None):
    """
    Smart recommendation routing:
      1. Known user with cached recs → fast lookup
      2. ≥5 ratings → full personalized collaborative filtering
      3. <5 ratings → cold-start genre popularity
    """
    store.load()

    # Strategy 1: cached recs for known user
    if user_id is not None:
        cached = get_precomputed_recs(user_id, top_n)
        if cached:
            return cached, False

    # Strategy 2/3: Cold-start (< 5) vs Collaborative Filtering (≥ 5)
    if len(rated_movies) < 5:
        recs = get_cold_start_recs(rated_movies, top_n)
        is_cold_start = True
    else:
        recs = get_collaborative_recs(rated_movies, top_n)
        is_cold_start = False

    return recs, is_cold_start
