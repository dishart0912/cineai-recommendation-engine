"""
api/models.py
─────────────────────────────────────────────────────────────────────────────
Pydantic request/response schemas for the FastAPI recommendation API.
─────────────────────────────────────────────────────────────────────────────
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Request Models ────────────────────────────────────────────────────────────

class MovieRating(BaseModel):
    """A single user-provided movie rating."""
    movie_id: int = Field(..., description="MovieLens movieId", example=1)
    title: str = Field(..., description="Movie title", example="Toy Story (1995)")
    rating: float = Field(..., ge=0.5, le=5.0, description="Rating (0.5–5.0)", example=4.5)
    genres: Optional[str] = Field(None, description="Pipe-separated genres", example="Animation|Children's|Comedy")


class RecommendationRequest(BaseModel):
    """Request body for getting recommendations based on rated movies."""
    ratings: List[MovieRating] = Field(
        ..., min_length=1, description="List of movies the user has rated"
    )
    top_n: int = Field(10, ge=1, le=50, description="Number of recommendations to return")
    exclude_rated: bool = Field(True, description="Exclude already-rated movies from results")


# ── Response Models ───────────────────────────────────────────────────────────

class RecommendedMovie(BaseModel):
    """A single recommended movie with metadata."""
    movie_id: int
    title: str
    genres: str
    predicted_rating: float
    match_score: float = Field(..., description="Normalized match score 0–100")
    poster_url: Optional[str] = Field(None, description="TMDB poster URL")
    tmdb_id: Optional[int] = None
    overview: Optional[str] = None


class RecommendationResponse(BaseModel):
    """Response for recommendation requests."""
    user_id: Optional[int] = None
    recommendations: List[RecommendedMovie]
    model_type: str = "ALS-Collaborative-Filtering"
    is_cold_start: bool = False
    total_returned: int


class MovieSearchResult(BaseModel):
    """A single movie search result."""
    movie_id: int
    title: str
    genres: str
    year: Optional[int] = None
    avg_rating: Optional[float] = None
    num_ratings: Optional[int] = None


class MovieSearchResponse(BaseModel):
    """Response for movie search."""
    results: List[MovieSearchResult]
    total: int
    query: str


class ModelMetrics(BaseModel):
    """Training and evaluation metrics for the ALS model."""
    best_hyperparams: dict
    test_metrics: dict
    training_time_seconds: float
    total_param_combinations: int
    cv_folds: int
    cv_results: List[dict]
    dataset_stats: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    dataset_loaded: bool
    total_users: int
    total_movies: int
    version: str = "1.0.0"
