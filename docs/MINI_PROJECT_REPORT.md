# REPORT ON MINI PROJECT
**Subject:** Big Data Analytics (CSC702)  
**Academic Year:** 2026–27  
**Project Title:** Scalable Distributed Movie Recommendation Engine Using PySpark MLlib ALS  

---

### Team Members
- **Disha Takawale** — 2413195
- **Prerana Patankar** — 2413196
- **Neha Maniyar** — 2413210
- **Saloni Patil** — 2413211

### Guided By
**Ms. Nidhi Jha**

---

# CHAPTER 1: INTRODUCTION

### 1.1 Project Description
**CineAI** is an enterprise-grade distributed movie recommendation system engineered to provide personalized movie recommendations based on user rating behavior while handling big-data scale interactions.

Modern streaming platforms such as Netflix, Amazon Prime Video, and Disney+ offer catalogues containing tens of thousands of media titles. As catalogue size grows, users experience choice paralysis and struggle to find movies aligned with their tastes. Recommendation systems resolve this bottleneck by analyzing historical user-item interactions and uncovering latent preference structures.

CineAI is built on the **MovieLens 1M** benchmark dataset, consisting of approximately 1,000,209 interactions across 6,040 users and 3,913 movies. The project utilizes **Apache Spark** and **PySpark MLlib** to run distributed extract-transform-load (ETL) pipelines and train an **Alternating Least Squares (ALS)** matrix factorization collaborative filtering model.

The end-to-end data and learning flow operates as:
$$\text{User Rates Movies} \longrightarrow \text{Distributed Spark ETL} \longrightarrow \text{ALS Learns Latent Vectors} \longrightarrow \text{Predicts Unseen Ratings} \longrightarrow \text{Personalized Top-N Delivered}$$

To combat the classic **Cold-Start Problem** (where sparse or non-existent history prevents matrix decomposition from computing reliable recommendations), CineAI implements a dedicated multi-tier recommendation engine:
- **0 ratings**: Displays a cold-start prompt encouraging the user to rate titles from the curated catalog.
- **1–4 ratings (Cold-Start Mode)**: Uses a hybrid genre-affinity and Bayesian popularity-weighted fallback algorithm.
- **5 or more ratings (Full AI Mode)**: Switches dynamically to the personalized recommendation pipeline using centered taste vectors, cosine similarity, and ALS latent projections.

The model is deployed via a high-performance **FastAPI REST backend** ($<35\text{ ms}$ latency), supported by probabilistic **Bloom Filter** deduplication, while an interactive glassmorphic **Streamlit** dashboard serves as the user-facing interface.

---

### 1.2 Problem Statement
With thousands of titles hosted on streaming platforms, users need accurate, real-time discovery mechanisms. CineAI directly resolves four foundational big-data challenges:
1. **Problem 1 — Information Overload**: Finding relevant content manually in massive catalogues is inefficient.
2. **Problem 2 — Large-Scale Distributed Data Processing**: Efficiently handling ~1M interaction records requires distributed cluster computing primitives rather than in-memory single-node computation.
3. **Problem 3 — Severe Matrix Sparsity**: With 95.16% of user-movie pairs unrated, standard statistical methods fail without dimensionality reduction and latent factor decomposition.
4. **Problem 4 — Cold-Start Dilemma**: New users lack sufficient interaction histories, preventing matrix-factorization inference from generating valid vectors.

---

### 1.3 Objectives
- Ingest and process large-scale movie rating datasets with distributed PySpark DataFrames.
- Perform exploratory data analytics on user-item interaction distributions and matrix sparsity.
- Design and train an ALS Matrix Factorization model using Spark MLlib with 3-fold cross-validation.
- Optimize hyperparameters (`rank`, `regParam`, `maxIter`) distributed across Spark executor cores.
- Formulate a 3-tier cold-start fallback strategy using genre-weighted Bayesian popularity.
- Implement an in-memory **Bloom Filter** for $O(k)$ constant-time deduplication of previously watched/rated movies.
- Expose recommendation, search, and performance endpoints through an asynchronous FastAPI REST API.
- Develop an interactive, reactive web UI using Streamlit with real-time rating updates and visual analytics.
- Evaluate model quality systematically using RMSE, MAE, $R^2$, and catalog coverage metrics.
- Demonstrate BDA curriculum alignment (HDFS Ingestion, MapReduce, Hive DB, MongoDB NoSQL, Bloom Filters, Flajolet-Martin streaming cardinality, R ggplot2 visualizations, and Social Graph Mining).

---

### 1.4 Questions Addressed Through the Project
- How can low-rank matrix factorization predict unseen ratings across a **95.16% sparse** interaction matrix?
- How does PySpark distribute hyperparameter grid search across multiple worker partitions to accelerate model selection?
- How can collaborative filtering extract hidden, latent taste vectors without requiring manual metadata labeling?
- How can an intelligent fallback system prevent user dropout during cold-start ($<5$ ratings)?
- How can probabilistic data structures like Bloom Filters prevent duplicate recommendations in constant time?
- How can an offline distributed Spark ML model be integrated with a low-latency ($<35\text{ ms}$) online REST API?
- How do RMSE (0.8678) and MAE (0.6959) quantify the practical accuracy of the recommendation engine?

---

### 1.5 Market Betterment
- **Digital Streaming Platforms**: Enhances customer retention and session watch-time through hyper-personalized content feeds.
- **Content Studios & Distributors**: Discovers audience sub-communities and genre affinities to inform production investments.
- **Big Data Engineering**: Provides a reusable architecture combining batch distributed training with lightweight, sub-second online serving.
- **Low-Latency Scalability**: Proves how decoupled microservices (FastAPI + Streamlit + Spark + MongoDB) scale horizontally on production clusters.

---

### 1.6 Tools and Technologies Used

| Technology | Category | Purpose in CineAI |
| :--- | :--- | :--- |
| **Python 3.10+** | Core Language | Primary implementation language across backend, ML scripts, and UI |
| **Apache Spark 3.5.1** | Distributed Engine | Large-scale parallel data processing, matrix operations, and partitioning |
| **PySpark MLlib** | Machine Learning | Distributed ALS implementation, `CrossValidator`, `ParamGridBuilder` |
| **Parquet** | Storage Format | Compressed columnar disk storage for optimized schema reading |
| **FastAPI 0.111** | Backend Framework | Asynchronous high-throughput REST API serving endpoints |
| **Uvicorn** | ASGI Web Server | Production-ready asynchronous web server hosting FastAPI |
| **Pydantic** | Validation | Schema validation and serialization for API requests/responses |
| **Streamlit 1.35** | Frontend Interface | Reactive dashboard featuring rating studio, recommendations, and analytics |
| **Bloom Filter** | Data Structure | $O(k)$ constant-time probabilistic deduplication of rated movies |
| **MongoDB** | NoSQL Database | Distributed document store for caching user recommendations and logs |
| **Hadoop HDFS / Hive** | Big Data Ecosystem | Cluster data ingestion, schema-on-read querying, and analytics |
| **R (ggplot2)** | Statistical Graphics | Publication-grade exploratory data visualization and decay curves |

---

### 1.7 User Interface and Visualization
- **Explore & Rate Studio**: Users search titles using fuzzy in-memory search, browse by genres, assign 0.5–5.0 star ratings, and track their active interaction counter.
- **Dynamic Recommendations Cockpit**:
  - *0 ratings*: Informative cold-start prompt instructing the user to rate initial titles.
  - *1–4 ratings*: Cold-Start fallback mode with genre affinity and Bayesian popularity scores.
  - *$\ge 5$ ratings*: **Full AI Mode** transition displaying Top-10 personalized movie picks with predicted rating stars and percentage match scores.
- **Analytics & Model Cockpit**: Renders dataset statistics, matrix sparsity metrics, cross-validation tuning curves, error distributions, and BDA architecture diagrams.

---

# CHAPTER 2: DATA DESCRIPTION AND ANALYSIS

### 2.1 Dataset Overview
CineAI utilizes the benchmark **MovieLens 1M** dataset provided by GroupLens Research.

#### Verified Dataset Statistics
| Metric | Value |
| :--- | :--- |
| **Raw Ratings Ingested** | 1,000,209 |
| **Clean Processed Ratings** | 999,611 |
| **Training Set Size (80%)** | 799,562 |
| **Testing Set Size (20%)** | 200,049 |
| **Unique Users** | 6,040 |
| **Unique Movies (Processed)** | 3,416 |
| **Extended Catalog (Master)** | 3,913 |
| **Rating Scale** | 0.5 – 5.0 (step size 0.5) |
| **Average Dataset Rating** | **3.58★** |
| **Interaction Matrix Sparsity** | **95.16%** |

---

### 2.2 Dataset Structure
Data is preprocessed and persisted in compressed columnar **Parquet** partitions:

#### Ratings Dataset (`train.parquet` & `test.parquet`)
| Column | PySpark Data Type | Description |
| :--- | :--- | :--- |
| `userId` | `IntegerType` | Unique user identification number (1 to 6,040) |
| `movieIndex` | `IntegerType` | Contiguous 0-indexed integer assigned to movie for ALS matrix math |
| `movieId` | `IntegerType` | Original MovieLens dataset movie identifier |
| `rating` | `FloatType` | User-assigned rating on a 0.5–5.0 scale |
| `timestamp` | `LongType` | Unix epoch timestamp of interaction |

#### Movie Metadata (`movie_mapping.parquet` & `all_movies.json`)
| Column | Type | Description |
| :--- | :--- | :--- |
| `movieId` | `Integer` | Original identifier |
| `movieIndex` | `Integer` | Contiguous ALS matrix index |
| `title` | `String` | Movie title with release year |
| `genres` | `String` | Pipe-delimited genre categories (`Action\|Sci-Fi`) |
| `avg_rating` | `Float` | Historical mean rating across all users |
| `num_ratings` | `Integer` | Total interaction count (popularity weight) |

---

### 2.3 Data Preprocessing Pipeline
Implemented in `spark/preprocess.py`:
1. **Ingestion & Schema Enforcement**: Raw `.dat`/`.csv` files are parsed with explicit PySpark schemas.
2. **Integrity & Null Filtering**: Drops corrupt records and isolates valid rating bounds ($0.5 \le r \le 5.0$).
3. **Contiguous Indexing**: Generates `movieIndex` using Spark's `StringIndexer` to guarantee dense coordinate matrices for ALS executors.
4. **Deterministic Partitioning**: Performs an 80/20 train-test split seeded at `seed=42`.
5. **Columnar Export**: Writes output to `data/processed/train.parquet`, `test.parquet`, and `movie_mapping.parquet`.

---

### 2.4 Matrix Sparsity Analysis
The user-item interaction space represents a matrix $R$ of dimensions $6,040 \times 3,416 = 20,632,640$ possible entries.
With only $999,611$ clean interactions observed:
$$\text{Sparsity} = 1 - \left( \frac{999,611}{6,040 \times 3,416} \right) = 1 - \frac{999,611}{20,632,640} = 1 - 0.04845 = \mathbf{95.16\%}$$

Because over 95% of the matrix entries are missing, simple neighbourhood heuristics (such as User-User k-NN) suffer from severe memory bottlenecks and the curse of dimensionality. ALS overcomes this via low-rank latent factorization.

---

### 2.5 Analytics Insights
1. **Long-Tail Distribution**: A small fraction of blockbuster films (e.g., *Star Wars*, *The Matrix*) command the majority of ratings, while long-tail movies receive sparse ratings.
2. **User Activity Skewness**: User interaction counts follow a power-law curve, where core users provide hundreds of ratings, but casual users require cold-start handling.
3. **Optimistic Rating Bias**: The dataset mean rating is $3.58\star$, with 4-star and 5-star ratings accounting for over 57% of all recorded interactions.

---

# CHAPTER 3: DESIGN OF DATA PIPELINE

### 3.1 System Architecture

```
                                  MovieLens 1M Dataset
                                (1,000,209 Raw Ratings)
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              Distributed PySpark Pipeline                              │
│                                                                                        │
│   ratings.dat / movies.dat                                                             │
│              │                                                                         │
│              ▼ [spark/preprocess.py]                                                   │
│      train.parquet (799,562)  │  test.parquet (200,049)                                │
│              │                                                                         │
│              ▼ [spark/train_als.py]                                                    │
│      Spark MLlib ALS + 3-Fold CrossValidator (ParamGrid: rank, regParam, maxIter)      │
│              │                                                                         │
│              ▼ [spark/generate_recs.py & spark/evaluate.py]                            │
│      Trained ALS Model  ──►  Batch Recs Cache (JSON / MongoDB)  ──► Evaluation Report  │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                      ┌────────────────────┴────────────────────┐
                      ▼                                         ▼
        ┌───────────────────────────┐             ┌───────────────────────────┐
        │     FastAPI REST API      │             │       Streamlit UI        │
        │      (Port: 8000)         │ ◄────────── │       (Port: 8501)        │
        │  • /recommendations       │             │  • Interactive Rating     │
        │  • /movies/search         │             │  • Dynamic Mode Switch    │
        │  • /model/metrics         │             │  • Analytics Cockpit      │
        │  • Bloom Filter Dedupl.   │             │  • Real-Time Recs Feed    │
        └───────────────────────────┘             └───────────────────────────┘
```

---

### 3.2 Detailed Pipeline Components

#### Component 1 — MovieLens Dataset Source
Provides raw input records containing user IDs, movie IDs, rating scores, timestamps, and pipe-delimited genre metadata.

#### Component 2 — PySpark Preprocessing Engine
Cleans, indexes, casts data types, generates contiguous matrix coordinates, and enforces train/test partitioning.

#### Component 3 — Columnar Parquet Storage Layer
Stores processed splits efficiently with Snappy compression, enabling high-throughput predicate pushdown during model training.

#### Component 4 — Spark MLlib ALS Matrix Factorization Model
Decomposes the interaction matrix into latent factor matrices:
$$R \approx U \times V^T$$
Where:
- $U \in \mathbb{R}^{m \times k}$: User latent vector matrix.
- $V \in \mathbb{R}^{n \times k}$: Movie latent feature matrix.
- $k$: Latent factor rank ($k = 50$).

Loss function minimized with $L_2$ regularization:
$$\min_{U, V} \sum_{(u, i) \in \mathcal{K}} \left( r_{ui} - u_u v_i^T \right)^2 + \lambda \left( \|u_u\|_2^2 + \|v_i\|_2^2 \right)$$

#### Component 5 — Predictions and Latent Factor Outputs
Extracts user-factor and item-factor Parquet matrices used for offline batch inference and test set scoring.

#### Component 6 — 3-Tier Multi-Strategy Recommendation Engine
- **Tier 1 (Active/Pre-computed Users, $>20$ ratings)**: Offline batch Top-20 matrix predictions queried from pre-computed cache in $O(1)$ constant time ($<2\text{ ms}$).
- **Tier 2 (Warm Users, $\ge 5$ ratings)**: Computes a rating-centered taste vector ($w_i = r_i - 2.5$), measures cosine similarity against candidate genre vectors, and blends quality and popularity scores ($<25\text{ ms}$).
- **Tier 3 (Cold-Start Users, $1 - 4$ ratings)**: Bayesian popularity-weighted genre overlap fallback scoring ($<5\text{ ms}$).

#### Component 7 — Probabilistic Filtering via Bloom Filter
Prevents previously rated movies from reappearing in recommendation feeds. Uses an in-memory **Bloom Filter** (`bda_lab/bloom_filter.py`) configured with an expected false-positive rate of 0.5% for $O(k)$ constant-time membership checking.

#### Component 8 — High-Throughput FastAPI REST Backend
Exposes endpoints (`/recommendations/{user_id}`, `/movies/search`, `/health`, `/model/metrics`) with automatic OpenAPI Swagger documentation.

#### Component 9 — Interactive Streamlit Frontend
A dashboard allowing users to search, rate, track thresholds, inspect recommendation feeds, and analyze system benchmarks.

#### Component 10 — Personalized Movie Picks
The final output delivered to the user, displaying movie poster cards, release years, predicted star ratings, and match percentages.

---

### 3.3 Recommendation Workflow

```
                              User Interaction
                                     │
                                     ▼
                      Count User Ratings History (N)
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
       N = 0                      1 ≤ N < 5                    N ≥ 5
   [Zero State]               [Cold-Start Mode]           [Full AI Mode]
   Display message:           Genre Affinity +            Centered Taste Vector
   "Rate movies to            Bayesian Popularity         + Cosine Sim + ALS
   learn taste"               Weighted Fallback           Inference
         │                           │                           │
         └───────────────────────────┴─────────────┬─────────────┘
                                                   │
                                                   ▼
                                     Bloom Filter Deduplication
                                    (Filter Already-Rated Films)
                                                   │
                                                   ▼
                                        Rank & Return Top-10 Picks
```

---

# CHAPTER 4: RESULT ANALYSIS

### 4.1 Overall Model Performance

| Metric | Measured Value | Benchmark Interpretation |
| :--- | :--- | :--- |
| **RMSE (Root Mean Squared Error)** | **0.8678** | High accuracy (deviation < 0.87 stars on a 1–5 scale) |
| **MAE (Mean Absolute Error)** | **0.6959** | Typical rating error is ~0.70 stars |
| **$R^2$ Score (Coefficient of Determination)** | **0.3942** | Explains 39.42% of unseen variance |
| **User Prediction Coverage** | **100.0%** | Zero dropouts achieved via multi-tier fallback |
| **Catalog Coverage** | **22.76%** | Balances popular discovery with long-tail items |
| **Training Duration** | **60.8 seconds** | Fast distributed convergence on Spark local[*] |
| **API Inference Latency** | **< 35 ms** | Sub-second real-time responsiveness |

---

### 4.2 RMSE Analysis
Root Mean Squared Error penalizes larger prediction discrepancies more heavily.
$$\text{RMSE} = \sqrt{\frac{1}{|\mathcal{T}|} \sum_{(u, i) \in \mathcal{T}} (r_{ui} - \hat{r}_{ui})^2} = \mathbf{0.8678}$$
On a 5-star rating scale, an RMSE of 0.8678 indicates strong predictive accuracy competitive with published MovieLens 1M benchmarks.

---

### 4.3 MAE Analysis
Mean Absolute Error measures the average magnitude of absolute prediction errors:
$$\text{MAE} = \frac{1}{|\mathcal{T}|} \sum_{(u, i) \in \mathcal{T}} |r_{ui} - \hat{r}_{ui}| = \mathbf{0.6959}$$
This confirms that the model's rating predictions typically deviate by less than 0.70 rating points from the user's actual rating.

---

### 4.4 Hyperparameter Optimization Results

Hyperparameter tuning was conducted via Spark's distributed 3-Fold Cross-Validation:

| Latent Rank ($k$) | Regularization ($\lambda$) | Max Iterations | 3-Fold CV RMSE | Selection Status |
| :---: | :---: | :---: | :---: | :---: |
| 10 | 0.1 | 10 | 0.8816 | Baseline |
| **50** | **0.1** | **10** | **0.8768** | **Selected Optimal Model** |

- **Rank 50** reduced the CV RMSE to **0.8768**, offering the best balance between latent feature expressiveness and training time (60.8s).
- Increasing rank from 10 to 50 enabled the model to capture more subtle sub-genre affinities without overfitting, controlled by $\lambda = 0.1$.

---

### 4.5 Rating Error Distribution
Evaluation on the 200,049 held-out test ratings demonstrates consistent prediction stability across all rating levels:
- **1-Star Ratings**: 11,016 instances
- **2-Star Ratings**: 21,478 instances
- **3-Star Ratings**: 52,202 instances
- **4-Star Ratings**: 69,800 instances
- **5-Star Ratings**: 45,553 instances

---

### 4.6 Cold-Start vs Full AI Demonstration Walkthrough

| Step | User Action | System State | Strategy Employed | Output |
| :---: | :--- | :---: | :--- | :--- |
| **1** | User opens app, 0 ratings | **Zero State** | N/A | Prompts user to rate initial titles |
| **2** | Rates 2 movies (e.g., *Toy Story*, *Finding Nemo*) | **Cold-Start Mode** | Genre affinity + Bayesian popularity | Recommends popular Animation/Family titles |
| **3** | Rates 3 more movies (e.g., *The Matrix*, *Terminator 2*, *Inception*) | **Threshold Reached** | Dynamic UI banner switches to **✓ Full AI Mode** | Confirms model initialization |
| **4** | Requests recommendations | **Full AI Mode** | Centered taste vector + Cosine Similarity + ALS | Tailored Top-10 sci-fi/action list with match scores |

---

# CHAPTER 5: CONCLUSION AND FUTURE SCOPE

### 5.1 Conclusion
The **CineAI** project implements an end-to-end distributed movie recommendation engine using **Apache Spark**, **PySpark MLlib ALS**, **FastAPI**, and **Streamlit**.

Key achievements include:
- Scalable distributed processing of **~1 million interactions** across 6,040 users and 3,913 movies.
- Matrix factorization handling **95.16% matrix sparsity** while achieving an **RMSE of 0.8678** and **MAE of 0.6959**.
- Elimination of cold-start dropouts through a 3-tier hybrid inference strategy achieving **100% user coverage**.
- Integration of Big Data techniques including **Bloom Filter deduplication**, **Parquet columnar storage**, and **MongoDB NoSQL caching**.
- Sub-35ms inference latency via a production-grade FastAPI REST service.

The project demonstrates how modern Big Data frameworks and machine learning algorithms work together to process large interaction datasets and power practical, interactive web applications.

---

### 5.2 Future Scope
1. **Real-Time Streaming Pipelines**: Integrate **Apache Kafka** and **Spark Structured Streaming** to update user latent vectors incrementally upon every rating submission.
2. **Deep Learning Hybridization**: Implement Two-Tower Neural Collaborative Filtering (NCF) or Transformer-based sequential recommenders (e.g., SASRec).
3. **Multimodal Content Features**: Incorporate text embeddings from movie plots (via BERT/Sentence-Transformers) and visual embeddings from movie posters.
4. **Context-Aware Recommendations**: Factor in temporal dynamics (time of day, day of week) and viewing device context.
5. **Distributed Cloud Scaling**: Deploy the Spark pipeline onto cloud infrastructure such as AWS EMR, Databricks, or Google Cloud Dataproc.
6. **Implicit Feedback Modeling**: Expand interaction logging to include dwell time, search clicks, and trailer watch duration using Spark ALS implicit feedback mode (`implicitPrefs=True`).

---

# REFERENCES
1. F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19.
2. Apache Spark MLlib Collaborative Filtering Documentation: https://spark.apache.org/docs/latest/ml-collaborative-filtering.html
3. Y. Koren, R. Bell, and C. Volinsky. 2009. *Matrix Factorization Techniques for Recommender Systems.* Computer, vol. 42, no. 8, pp. 30–37.
4. FastAPI Documentation: https://fastapi.tiangolo.com/
5. Streamlit Documentation: https://docs.streamlit.io/
6. B. H. Bloom. 1970. *Space/time trade-offs in hash coding with allowable errors.* Communications of the ACM, 13(7):422–426.
