# Machine Learning Architecture & Methodology: CineAI

This document provides a comprehensive technical overview of the Machine Learning models, algorithms, training workflows, and inference strategies implemented in the **CineAI Distributed Movie Recommendation Engine**.

---

## 1. Executive Summary

CineAI uses a **hybrid recommendation architecture** centered around **Distributed Alternating Least Squares (ALS) Matrix Factorization** from **Apache Spark MLlib (`pyspark.ml.recommendation.ALS`)**. 

To deliver responsive recommendations while overcoming the classic cold-start limitations of collaborative filtering, the system pairs offline batch ALS matrix factorization with a real-time, 3-tier hybrid inference engine:
- **Offline / Batch**: Spark MLlib ALS Matrix Factorization tuned via 3-Fold Cross-Validation.
- **Online / Real-time**: Tiered serving strategy with $O(1)$ pre-computed cache lookups, dynamic genre-weighted taste vector cosine similarity, and Bayesian popularity fallbacks.

---

## 2. Core Model: Alternating Least Squares (ALS)

### 2.1 Mathematical Formulation

Collaborative filtering via Matrix Factorization decomposes a high-dimensional, sparse user-item interaction matrix $R \in \mathbb{R}^{m \times n}$ ($m$ users, $n$ movies) into two lower-dimensional dense matrices:

$$R \approx U \times V^T$$

Where:
- $U \in \mathbb{R}^{m \times k}$ represents the user latent preference matrix (each row $u_u \in \mathbb{R}^k$ captures user $u$'s latent tastes).
- $V \in \mathbb{R}^{n \times k}$ represents the movie latent feature matrix (each row $v_i \in \mathbb{R}^k$ captures movie $i$'s latent attributes).
- $k$ is the latent factor rank (`rank = 100`).

### 2.2 Optimization Objective

The objective function minimizes the regularized squared error across all observed ratings $\mathcal{K}$:

$$\min_{U, V} \sum_{(u, i) \in \mathcal{K}} \left( r_{ui} - u_u v_i^T \right)^2 + \lambda \left( \|u_u\|_2^2 + \|v_i\|_2^2 \right)$$

Where:
- $r_{ui}$ is the observed rating given by user $u$ to movie $i$ (explicit feedback on a 1.0–5.0 scale).
- $\hat{r}_{ui} = u_u v_i^T$ is the predicted rating.
- $\lambda$ is the $L_2$ regularization parameter (`regParam = 0.1`) preventing overfitting on sparse entries.

### 2.3 The "Alternating" Optimization Process

Because optimizing both $U$ and $V$ simultaneously is non-convex, the ALS algorithm alternates:
1. **Fix $V$, solve for $U$**: The objective becomes quadratic and decomposes into independent ridge regression problems for each user, solved in parallel across Spark cluster nodes.
2. **Fix $U$, solve for $V$**: The objective decomposes into independent ridge regression problems for each movie, solved in parallel.
3. **Repeat**: Alternate for `maxIter = 20` iterations until the loss converges.

---

## 3. Training & Hyperparameter Tuning Pipeline

The model training pipeline is implemented in [`spark/train_als.py`](../spark/train_als.py).

### 3.1 Model Configuration
```python
als = ALS(
    userCol="userId",
    itemCol="movieIndex",
    ratingCol="rating",
    coldStartStrategy="drop",   # Drops unobserved users/items during evaluation
    nonnegative=True,           # Enforces non-negative factors for interpretability
    implicitPrefs=False,        # Uses explicit rating feedback
)
```

### 3.2 Distributed Hyperparameter Search Grid

Hyperparameter optimization uses Spark's `ParamGridBuilder` with 3-Fold `CrossValidator`:

| Hyperparameter | Search Values | Selected Best Value | Description |
| :--- | :--- | :--- | :--- |
| `rank` | `[10, 50, 100]` | **100** | Dimension of latent feature space |
| `regParam` ($\lambda$) | `[0.01, 0.1, 1.0]` | **0.1** | $L_2$ regularization penalty |
| `maxIter` | `[10, 20]` | **20** | Alternating iterations |

- **Total Combinations**: $3 \times 3 \times 2 = 18$ parameter sets.
- **Cross-Validation**: 3 folds $\implies$ **54 distributed Spark jobs** evaluated concurrently with parallelism = 4.

---

## 4. Evaluation & Benchmarks

Model performance was evaluated on the held-out test split (20% held-out interactions from MovieLens 1M):

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **RMSE (Root Mean Squared Error)** | **0.8678** | Typical error is under 0.87 stars on a 1–5 scale |
| **MAE (Mean Absolute Error)** | **0.6959** | Average absolute rating deviation |
| **$R^2$ Score** | **0.3942** | Explains ~39.4% of rating variance |
| **User Coverage** | **100.0%** | Zero cold-start dropouts via fallback system |
| **Catalog Coverage** | **22.76%** | Balances popular items with long-tail discovery |

---

## 5. Multi-Tier Hybrid Inference & Serving Architecture

To address the limitations of ALS (namely latency for ad-hoc requests and inability to recommend to users without interaction history), the serving engine in [`api/recommender.py`](../api/recommender.py) applies a 3-tier strategy:

```
                            User Request
                                 │
                ┌────────────────┴────────────────┐
                ▼                                 ▼
      Existing User ID?                   Interactive Ratings?
                │                                 │
         ┌──────┴──────┐                   ┌──────┴──────┐
         ▼             ▼                   ▼             ▼
      In Cache?     Not Found           < 5 ratings   >= 5 ratings
         │             │                   │             │
      Tier 1        Tier 3              Tier 3        Tier 2
     (ALS O(1)    (Popularity         (Cold-Start   (Taste Vector
      Lookup)      Fallback)           Fallback)     Cosine Sim)
```

### Tier 1: Active / Known Users ($>20$ ratings)
- **Engine**: Batch ALS offline pre-computation ([`spark/generate_recs.py`](../spark/generate_recs.py)).
- **Mechanism**: `model.recommendForUserSubset(all_users, top_n)` generates top-20 predictions per user, cached in `data/recs_cache.json` or MongoDB.
- **Latency**: $< 2\text{ ms}$ ($O(1)$ key-value lookup).

### Tier 2: Warm Users ($5 - 20$ ratings)
- **Engine**: Dynamic Centered Taste Vector & Cosine Similarity (`get_collaborative_recs`).
- **Mechanism**:
  1. Computes mean-centered rating weight $w_i = r_i - 2.5$ (rewards 4–5★, penalizes 1–2★).
  2. Builds normalized user taste vector across genres: $\vec{u}_{\text{taste}}$.
  3. Calculates cosine similarity against unrated movie candidate genre vectors:
     $$\text{sim}(\vec{u}_{\text{taste}}, \vec{g}_{\text{movie}}) = \frac{\vec{u}_{\text{taste}} \cdot \vec{g}_{\text{movie}}}{\|\vec{u}_{\text{taste}}\| \|\vec{g}_{\text{movie}}\|}$$
  4. Blends score: $80\%$ taste similarity $+ 15\%$ movie average quality $+ 5\%$ popularity dampening.
- **Latency**: $< 25\text{ ms}$.

### Tier 3: Cold-Start Users ($< 5$ ratings)
- **Engine**: Hybrid Genre-Weighted Bayesian Popularity (`get_cold_start_recs`).
- **Mechanism**: 
  - Prevents cold-start failures when interaction history is insufficient for matrix decomposition.
  - Scores candidates via:
    $$\text{Score} = (0.45 \times \text{Genre Overlap}) + (0.35 \times \log(1 + \text{Popularity})) + (0.20 \times \text{Avg Rating})$$
- **Latency**: $< 5\text{ ms}$.

---

## 6. Deduplication & Big Data Optimizations

- **Probabilistic Deduplication**: Utilizes an in-memory **Bloom Filter** (`bda_lab/bloom_filter.py`) with an expected false positive rate of 0.5% to verify and filter out already-rated movies in $O(k)$ hash operations.
- **Data Persistence**: Train/Test splits stored in compressed columnar **Apache Parquet** format for efficient distributed I/O.
- **Caching Layer**: Decoupled FastAPI and Streamlit architecture querying in-memory JSON caches and optional MongoDB NoSQL storage for sub-millisecond response times.
