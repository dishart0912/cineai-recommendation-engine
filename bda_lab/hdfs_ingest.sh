#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# bda_lab/hdfs_ingest.sh
# ─────────────────────────────────────────────────────────────────────────────
# BDA Experiment 1: Installation of Hadoop & HDFS Command Suite
# Demonstrates distributed storage management on Hadoop Distributed File System.
# ══════════════════════════════════════════════════════════════════════════════

set -e

echo "=== CineAI: BDA Experiment 1 - HDFS Commands ==="

if ! command -v hdfs &> /dev/null; then
    echo "[INFO] HDFS CLI not found in PATH."
    echo "Simulating command execution flow for BDA Viva presentation:"
    echo "  1. hdfs dfs -mkdir -p /cineai/raw"
    echo "  2. hdfs dfs -put data/raw/ratings.csv /cineai/raw/"
    echo "  3. hdfs dfs -ls -R /cineai"
    echo "  4. hdfs dfs -du -h /cineai"
    exit 0
fi

echo "[1/4] Creating HDFS directories..."
hdfs dfs -mkdir -p /cineai/raw/ratings /cineai/raw/movies /cineai/processed

echo "[2/4] Uploading raw datasets to HDFS..."
hdfs dfs -put -f data/raw/ratings.csv /cineai/raw/ratings/
hdfs dfs -put -f data/raw/movies.csv /cineai/raw/movies/

echo "[3/4] Listing HDFS files with block replication..."
hdfs dfs -ls -R /cineai

echo "[4/4] Checking distributed storage allocation..."
hdfs dfs -du -h /cineai

echo "[SUCCESS] HDFS pipeline executed."
