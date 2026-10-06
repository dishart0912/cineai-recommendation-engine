@echo off
REM ══════════════════════════════════════════════════════════════════════════
REM  bda_lab/hdfs_ingest.bat
REM ─────────────────────────────────────────────────────────────────────────
REM  BDA Experiment 1: Installation of Hadoop & HDFS Command Suite
REM  Demonstrates distributed storage management on Hadoop Distributed File System.
REM ══════════════════════════════════════════════════════════════════════════

echo.
echo  ╔═══════════════════════════════════════════════════════════╗
echo  ║   CineAI — HDFS Ingestion & Command Suite (BDA Exp 1)     ║
echo  ╚═══════════════════════════════════════════════════════════╝
echo.

WHERE hdfs >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [NOTICE] 'hdfs' command not found in current PATH.
    echo.
    echo [BDA ARCHITECTURE EXPLANATION]:
    echo In a production Hadoop cluster, dataset ingestion executes via:
    echo.
    echo 1. Create HDFS Directory Structure:
    echo    hdfs dfs -mkdir -p /cineai/raw/ratings /cineai/raw/movies /cineai/models
    echo.
    echo 2. Upload MovieLens dataset to HDFS (blocks split across DataNodes):
    echo    hdfs dfs -put -f data\raw\ratings.csv /cineai/raw/ratings/
    echo    hdfs dfs -put -f data\raw\movies.csv  /cineai/raw/movies/
    echo.
    echo 3. Verify directory contents and replication:
    echo    hdfs dfs -ls -R /cineai
    echo.
    echo 4. Check disk usage and distributed block allocation:
    echo    hdfs dfs -du -h /cineai
    echo.
    echo 5. Stream head of file from HDFS:
    echo    hdfs dfs -cat /cineai/raw/movies/movies.csv ^| more
    echo.
    echo Note: Winutils and Hadoop.dll are pre-configured in hadoop/bin for Spark.
    echo ══════════════════════════════════════════════════════════════════════════
    goto :end
)

echo [1/5] Creating distributed directories in HDFS ...
hdfs dfs -mkdir -p /cineai/raw/ratings /cineai/raw/movies /cineai/models

echo [2/5] Uploading ratings.csv and movies.csv to HDFS DataNodes ...
hdfs dfs -put -f data\raw\ratings.csv /cineai/raw/ratings/
hdfs dfs -put -f data\raw\movies.csv  /cineai/raw/movies/

echo [3/5] Listing distributed namespace (/cineai) ...
hdfs dfs -ls -R /cineai

echo [4/5] Checking block distribution and storage utilization ...
hdfs dfs -du -h /cineai

echo [5/5] Sample read from HDFS DataNode ...
hdfs dfs -cat /cineai/raw/movies/movies.csv | head -n 5

echo.
echo [OK] HDFS Experiment 1 execution complete!

:end
