# 🎬 CineAI — Distributed Movie Recommendation Engine

> **Large-Scale Collaborative Filtering System using PySpark MLlib ALS, FastAPI, and Streamlit**  
> *Production-ready Big Data Analytics pipeline with hybrid inference and cold-start mitigation.*

[![PySpark](https://img.shields.io/badge/PySpark-3.5.1-E25A1C?logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-MovieLens%201M%20%2F%2025M-yellow)](https://grouplens.org/datasets/movielens/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
  - [1. Quick Start (Pre-computed / Demo Mode)](#1-quick-start-demo-mode)
  - [2. Full Big Data Pipeline from Scratch](#2-full-big-data-pipeline-from-scratch)
- [Machine Learning & Algorithm Details](#-machine-learning--algorithm-details)
  - [ALS Matrix Factorization](#als-matrix-factorization)
  - [Hyperparameter Tuning & Cross-Validation](#hyperparameter-tuning)
  - [3-Tier Cold-Start Strategy](#3-tier-cold-start-strategy)
- [Performance & Evaluation Benchmarks](#-performance--evaluation-benchmarks)
- [API Reference](#-api-reference)
- [Cluster & Cloud Scaling (HDFS / Cloud)](#-cluster--cloud-scaling)
- [BDA Experiments Mapping](#-bda-experiments-mapping)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)
- [Pushing to GitHub](#-pushing-to-github)

---

## 📌 Overview

**CineAI** is an enterprise-grade distributed recommendation system designed to handle large-scale user-item interactions using the **MovieLens** benchmark dataset.

The system utilizes **Apache Spark MLlib's Alternating Least Squares (ALS)** matrix factorization to model latent user preferences and movie features. It combines pre-computed batch recommendations with real-time inference and a content/popularity fallback engine to solve the classic cold-start challenge.

---

## ✨ Key Features

- **⚡ Distributed Data Processing**: PySpark ETL pipeline handling ingestion, schema validation, deduplication, and Parquet persistence.
- **🤖 ALS Matrix Factorization**: Spark MLlib collaborative filtering tuned across 18 parameter combinations using parallel k-fold cross-validation.
- **❄️ Multi-Tier Cold Start Engine**:
  - *New Users (< 5 ratings)*: High-rated genre popularity fallback.
  - *Warm Users (5–20 ratings)*: Dynamic genre-weighted ALS approximation.
  - *Active Users (> 20 ratings)*: Ultra-fast $O(1)$ batch recommendations lookup with real-time vector inference.
- **🚀 High-Throughput REST API**: Asynchronous FastAPI service exposing 7 endpoints including fuzzy title search, real-time recommendation generation, and metric reporting.
- **🎨 Interactive Streamlit UI**: Sleek, glassmorphic dashboard featuring:
  - Personalized movie recommendation feed with poster previews.
  - Interactive rating studio to rate movies and generate instant recommendations.
  - Analytics & performance cockpit showing RMSE, loss curves, and rating distributions.

---

## 🏗️ System Architecture

```
                             MovieLens Dataset
                           (1M / 25M Interactions)
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                     PySpark Distributed Pipeline                          │
│                                                                           │
│   ratings.csv / movies.csv                                                │
│              │                                                            │
│              ▼ [spark/preprocess.py]                                      │
│      train.parquet / test.parquet                                         │
│              │                                                            │
│              ▼ [spark/train_als.py]                                       │
│      ParamGridBuilder + 3-Fold CrossValidator                             │
│      (rank=[10, 50, 100], regParam=[0.01, 0.1, 1.0], maxIter=[10, 20])   │
│              │                                                            │
│              ▼ [spark/generate_recs.py & spark/evaluate.py]               │
│      Trained ALS Model  ──►  Batch Recs Cache  ──► Evaluation Report       │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                 ┌────────────────────┴───────────────────┐
                 ▼                                        ▼
   ┌───────────────────────────┐            ┌───────────────────────────┐
   │     FastAPI REST API      │            │       Streamlit UI        │
   │      (Port: 8000)         │ ◄───────── │       (Port: 8501)        │
   │  • /recommendations       │            │  • Interactive Dashboard  │
   │  • /movies/search         │            │  • Real-Time Rating Studio│
   │  • /model/metrics         │            │  • Analytics Cockpit      │
   └───────────────────────────┘            └───────────────────────────┘
```

---

## 📁 Repository Structure

```
recommendation-engine/
├── .streamlit/
│   └── config.toml           # Streamlit custom dark theme configuration
├── api/
│   ├── __init__.py
│   ├── main.py               # FastAPI application with REST endpoints
│   ├── models.py             # Pydantic schemas for requests and responses
│   └── recommender.py        # Recommendation inference & cold-start engine
├── data/
│   ├── all_movies.json       # Master catalog with genres & average ratings
│   ├── cold_start_popular.json# Pre-computed popular fallback items
│   ├── recs_cache.json       # High-speed pre-computed recommendations cache
│   ├── download_data.py      # Automated MovieLens dataset downloader
│   ├── raw/                  # Downloaded raw CSV / DAT files (git-ignored)
│   └── processed/            # Spark-generated train/test Parquet datasets
├── hadoop/                   # Windows winutils & hadoop.dll for native Spark
├── logs/
│   ├── dataset_stats.json    # Summary metrics (users, ratings, sparsity)
│   ├── evaluation_report.json# RMSE, MAE, R2, and ranking metrics
│   └── training_metrics.json # Hyperparameter grid search logs
├── models/
│   └── als_model/            # Serialized Spark MLlib ALS trained model
├── notebooks/                # Exploratory data analysis notebooks
├── spark/
│   ├── preprocess.py         # PySpark distributed data preprocessing
│   ├── train_als.py          # ALS model training & hyperparameter search
│   ├── evaluate.py           # Evaluation pipeline (RMSE, MAE, Coverage)
│   └── generate_recs.py      # Batch top-K recommendation generator
├── .gitignore                # Git ignore rules for venv, raw data, caches
├── requirements.txt          # Python dependencies
├── setup.bat                 # 1-Click setup script for Windows
├── setup.sh                  # 1-Click setup script for Linux/macOS
└── README.md                 # Complete project documentation
```

---

## ⚙️ Prerequisites

1. **Python**: Version `3.9`, `3.10`, or `3.11` recommended.
2. **Java (JDK)**: **Java 8, 11, or 17** is required by PySpark.
   - [Download Eclipse Adoptium Temurin JDK](https://adoptium.net/)
   - Verify installation: `java -version`
3. **Git**: Installed on your system.

---

## 🚀 Installation & Setup

### Option A: Automated Setup

#### On Windows:
Double-click `setup.bat` or run:
```cmd
setup.bat
```

#### On Linux / macOS:
```bash
chmod +x setup.sh
./setup.sh
```

---

### Option B: Manual Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/recommendation-engine.git
   cd recommendation-engine
   ```

2. **Create and activate a virtual environment:**
   - **Windows:**
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 💻 How to Run

### 1. Quick Start (Demo Mode)

The repository comes packaged with pre-processed catalogs, evaluation logs, and recommendation caches, allowing you to launch both services immediately without waiting for model re-training.

#### Step 1: Start the FastAPI Backend
```bash
uvicorn api.main:app --reload --port 8000
```
- API Base URL: `http://localhost:8000`
- Interactive Swagger Docs: `http://localhost:8000/docs`

#### Step 2: Start the Streamlit Dashboard (In a new terminal)
```bash
streamlit run ui/app.py
```
- Access the web interface at: `http://localhost:8501`

---

### 2. Full Big Data Pipeline from Scratch

If you wish to download the raw dataset and re-run the entire distributed PySpark pipeline from source:

#### Step 1: Download MovieLens Data
```bash
# MovieLens 1M (~6 MB, recommended for local machines)
python data/download_data.py --size 1m

# OR MovieLens 25M (~250 MB, for high-memory / cluster setups)
python data/download_data.py --size 25m
```

#### Step 2: Preprocess Data with PySpark
```bash
python spark/preprocess.py
```
*Casts schemas, filters rating records, creates unified user/item integer indices, and outputs `train.parquet` and `test.parquet`.*

#### Step 3: Train ALS Matrix Factorization Model
```bash
# Fast mode (2 hyperparameter combinations for quick verification):
python spark/train_als.py --fast

# Full hyperparameter tuning (18 combinations with 3-fold cross validation):
python spark/train_als.py
```

#### Step 4: Evaluate the Model
```bash
python spark/evaluate.py
```
*Computes RMSE, MAE, R², ranking precision@10, catalog coverage, and saves metrics to `logs/evaluation_report.json`.*

#### Step 5: Generate Batch Recommendations Cache
```bash
python spark/generate_recs.py
```
*Generates Top-20 recommendations for all users and updates `data/recs_cache.json`.*

---

## 🤖 Machine Learning & Algorithm Details

### ALS Matrix Factorization

The sparse user-movie interaction matrix $R \in \mathbb{R}^{m \times n}$ is decomposed into low-rank factor matrices:

$$R \approx U \times V^T$$

Where:
- $U \in \mathbb{R}^{m \times k}$ represents user latent preference vectors.
- $V \in \mathbb{R}^{n \times k}$ represents movie latent feature vectors.
- $k$ is the latent factor dimension (`rank`).

The alternating optimization objective minimizes the regularized squared loss:

$$\min_{U, V} \sum_{(u, i) \in \mathcal{K}} \left( r_{ui} - u_u v_i^T \right)^2 + \lambda \left( \|u_u\|_2^2 + \|v_i\|_2^2 \right)$$

### Hyperparameter Tuning

| Hyperparameter | Search Grid | Best Value | Description |
|:---|:---|:---|:---|
| `rank` | `[10, 50, 100]` | **100** | Dimension of latent vector representation |
| `regParam` ($\lambda$) | `[0.01, 0.1, 1.0]` | **0.1** | L2 regularization coefficient to prevent overfitting |
| `maxIter` | `[10, 20]` | **20** | Alternating iterations before convergence |

*A total of 18 combinations were evaluated across 3 folds (54 parallel Spark jobs).*

### 3-Tier Cold-Start Strategy

| User State | Ratings Count | Strategy Employed | Latency |
|:---|:---|:---|:---|
| **Cold Start** | $< 5$ ratings | Genre-weighted Bayesian average rating popularity fallback | $< 5\text{ ms}$ |
| **Warm Start** | $5 - 20$ ratings | Cosine genre-similarity combined with ALS latent profile estimation | $< 25\text{ ms}$ |
| **Active User** | $> 20$ ratings | Pre-computed ALS batch recommendations with real-time filtering | $< 2\text{ ms}$ ($O(1)$) |

---

## 📊 Performance & Evaluation Benchmarks

Evaluated on the **MovieLens 1M** test split (20% held-out interactions):

| Metric | Score | Note |
|:---|:---|:---|
| **RMSE (Root Mean Squared Error)** | **0.8678** | Consistent with state-of-the-art ALS on MovieLens 1M |
| **MAE (Mean Absolute Error)** | **0.6959** | Average deviation on a 1–5 scale |
| **$R^2$ Score** | **0.3942** | Variance explained across unseen ratings |
| **User Coverage** | **100.0%** | Zero cold-start dropouts via fallback strategy |
| **Catalog Coverage** | **22.76%** | Diverse item discovery across long-tail titles |

---

## 🌐 API Reference

Interactive Swagger documentation is available at `http://localhost:8000/docs`.

### Available Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/` | API metadata and system status |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/movies/search?q={query}` | In-memory fuzzy search across movie catalog |
| `GET` | `/recommendations/{user_id}?n=10` | Retrieve pre-computed recommendations for existing user |
| `POST`| `/recommendations` | Generate dynamic recommendations from a list of user ratings |
| `GET` | `/model/metrics` | Retrieve model training parameters & RMSE metrics |
| `GET` | `/dataset/stats` | Retrieve dataset distribution and interaction statistics |

#### Example: POST `/recommendations` Request

```json
{
  "ratings": [
    {"movie_id": 1, "rating": 5.0},
    {"movie_id": 260, "rating": 4.5},
    {"movie_id": 1196, "rating": 5.0}
  ],
  "num_recs": 5
}
```

---

## ☁️ Cluster & Cloud Scaling

To transition from standalone local mode to a production distributed compute cluster:

```python
# In spark/preprocess.py, spark/train_als.py:

# 1. Local standalone (Default)
SparkSession.builder.master("local[*]")

# 2. Apache Hadoop YARN
SparkSession.builder.master("yarn")

# 3. Kubernetes (k8s)
SparkSession.builder.master("k8s://https://<kubernetes-api-server>:6443")
```

Data sources can be directly switched from local filesystem to **HDFS**, **Amazon S3** (`s3a://bucket/data`), or **Google Cloud Storage** (`gs://bucket/data`).

---

## 🛠️ Troubleshooting & FAQs

### 1. `java.lang.NoClassDefFoundError` or `Java not found`
- **Cause**: Spark requires a compatible Java JDK (Java 8, 11, or 17).
- **Fix**: Install [Adoptium Temurin JDK 11](https://adoptium.net/) and ensure `JAVA_HOME` environment variable points to your JDK directory:
  ```powershell
  [System.Environment]::SetEnvironmentVariable('JAVA_HOME', 'C:\Program Files\Eclipse Adoptium\jdk-11.x.x', [System.EnvironmentVariableTarget]::User)
  ```

### 2. Windows `HADOOP_HOME` / `winutils.exe` Warning
- **Cause**: Windows requires native Hadoop binaries (`winutils.exe` and `hadoop.dll`) for filesystem operations.
- **Fix**: This repository already includes the necessary binaries in the `hadoop/` directory. `setup.bat` and Spark scripts automatically point `HADOOP_HOME` to this folder.

### 3. Port Already in Use (Port 8000 or 8501)
- If port 8000 is occupied: `uvicorn api.main:app --port 8001`
- If port 8501 is occupied: `streamlit run ui/app.py --server.port 8502`

---

## 📤 Pushing to GitHub

Follow these steps to publish this repository to your GitHub profile:

### Step 1: Initialize Git and Check Status
```bash
git init
git status
```

### Step 2: Add Files and Make Initial Commit
```bash
git add .
git commit -m "Initial commit: CineAI distributed movie recommendation engine"
```

### Step 3: Link to Your GitHub Repository
1. Go to [GitHub](https://github.com/new) and create a new repository (e.g., `cineai-recommendation-engine`).
2. *Leave "Initialize this repository with a README" unchecked.*
3. Run the following commands in your project directory:

```bash
# Rename branch to main
git branch -M main

# Add your GitHub repository remote
git remote add origin https://github.com/<your-username>/<your-repo-name>.git

# Push code to GitHub
git push -u origin main
```

---

---

## 🔬 BDA Experiments Mapping

CineAI is designed to demonstrate key Big Data Analytics (BDA) syllabus concepts in an end-to-end production architecture:

| Exp # | Official Experiment Title | CineAI Implementation | Demonstrable Script / File |
| :---: | :--- | :--- | :--- |
| **Exp 1** | Hadoop & HDFS Command Suite | HDFS distributed namespace creation, block uploads, replication inspection | `bda_lab/hdfs_ingest.bat` |
| **Exp 4** | Hadoop MapReduce WordCount | Mapper and Reducer streaming for Movie Genre frequencies and Title keywords | `bda_lab/mapreduce/run_mapreduce.py` |
| **Exp 6** | Hive DB & Descriptive Analytics | Schema-on-Read external tables, rating mean, variance, stddev, matrix sparsity | `bda_lab/run_hive_stats.py` (`hive_analytics.hql`) |
| **Exp 7** | MongoDB NoSQL Database | Nested document store for user recommendations and real-time rating event logs | `bda_lab/mongo_manager.py` |
| **Exp 8** | Bloom Filter | Constant-time $O(k)$ memory-efficient deduplication of already-watched movies | `bda_lab/bloom_filter.py` |
| **Exp 9** | Flajolet-Martin (FM) Algorithm | Streaming cardinality estimation of distinct active users via trailing zeros | `bda_lab/flajolet_martin.py` |
| **Exp 10**| Data Visualization using R | Publication-quality EDA charts using R `ggplot2` (rating decay, genre spread) | `bda_lab/run_r_plots.py` (`visualizations.R`) |
| **Exp 12**| Big Data 5Vs & Distributed ML | PySpark distributed ALS matrix factorisation ($R \approx U \times V^T$) on workers | `spark/train_als.py`, `ui/app.py` |
| **Exp 13**| Social Graph Mining (GN & CPM) | Girvan-Newman edge-betweenness clusters and Clique Percolation overlapping films | `bda_lab/graph_mining.py` |

All experiments are interactively demonstrable in the **Streamlit UI** under the **"🔬 BDA Lab Cockpit"** page!

---

## 📜 License & Acknowledgments

- **Dataset**: Provided by [GroupLens Research](https://grouplens.org/datasets/movielens/) (MovieLens 1M / 25M).
- **Matrix Factorization Reference**: *Matrix Factorization Techniques for Recommender Systems* (Yehuda Koren, Robert Bell, Chris Volinsky, 2009).
- **License**: MIT License. Open for educational and research use.

