-- ══════════════════════════════════════════════════════════════════════════════
-- bda_lab/hive_analytics.hql
-- ─────────────────────────────────────────────────────────────────────────────
-- Experiment 6: Hive Database & Descriptive Statistics / Analytics
--
-- Demonstrates schema-on-read external tables, partitioned analytics,
-- and descriptive statistics (mean, variance, counts, sparsity) on MovieLens.
--
-- Execute via Apache Hive CLI or Beeline:
--   hive -f bda_lab/hive_analytics.hql
-- Or execute via PySpark Hive context:
--   python bda_lab/run_hive_stats.py
-- ══════════════════════════════════════════════════════════════════════════════

-- 1. Create Database
CREATE DATABASE IF NOT EXISTS cineai_dw;
USE cineai_dw;

-- 2. Define External Table for Raw Ratings (Schema on Read)
DROP TABLE IF EXISTS ext_ratings;
CREATE EXTERNAL TABLE ext_ratings (
    userId INT,
    movieId INT,
    rating FLOAT,
    rating_timestamp BIGINT
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/cineai/raw/ratings';

-- 3. Define External Table for Movies Catalog
DROP TABLE IF EXISTS ext_movies;
CREATE EXTERNAL TABLE ext_movies (
    movieId INT,
    title STRING,
    genres STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/cineai/raw/movies';

-- 4. Analytical Query 1: Descriptive Statistics of User Ratings
-- Computes Count, Mean, Variance, StdDev, Min, and Max
SELECT 
    COUNT(*) AS total_ratings,
    ROUND(AVG(rating), 3) AS mean_rating,
    ROUND(VARIANCE(rating), 3) AS variance_rating,
    ROUND(STDDEV(rating), 3) AS stddev_rating,
    MIN(rating) AS min_rating,
    MAX(rating) AS max_rating
FROM ext_ratings;

-- 5. Analytical Query 2: Rating Sparsity and User Engagement
SELECT 
    COUNT(DISTINCT userId) AS active_users,
    COUNT(DISTINCT movieId) AS active_movies,
    COUNT(*) AS interaction_count,
    ROUND(1.0 - (COUNT(*) / (COUNT(DISTINCT userId) * COUNT(DISTINCT movieId))), 4) AS matrix_sparsity
FROM ext_ratings;

-- 6. Analytical Query 3: Genre Performance & Popularity
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
LIMIT 15;

-- 7. Analytical Query 4: Top 10 Most Influential Benchmark Movies
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
LIMIT 10;
