# ══════════════════════════════════════════════════════════════════════════════
# bda_lab/visualizations.R
# ─────────────────────────────────────────────────────────────────────────────
# Experiment 10: Data Visualization using R (ggplot2)
#
# Generates publication-quality Exploratory Data Analysis (EDA) visualizations:
#   1. Rating Distribution Histogram & Density Curve
#   2. Long-Tail Power Law Distribution of Movie Popularity
#   3. Average Ratings by Major Genres with Confidence Intervals
#
# Run from terminal:
#   Rscript bda_lab/visualizations.R
# ══════════════════════════════════════════════════════════════════════════════

# Install packages if missing
required_packages <- c("ggplot2", "dplyr", "readr", "scales")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages, repos="https://cloud.r-project.org")

library(ggplot2)
library(dplyr)
library(readr)
library(scales)

# Paths
base_dir <- getwd()
data_dir <- file.path(base_dir, "data", "raw")
output_dir <- file.path(base_dir, "logs", "plots_r")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

ratings_path <- file.path(data_dir, "ratings.csv")
movies_path <- file.path(data_dir, "movies.csv")

if (!file.exists(ratings_path) || !file.exists(movies_path)) {
  stop("MovieLens raw CSV files not found in data/raw/")
}

message("Reading MovieLens dataset...")
ratings <- read_csv(ratings_path, col_types = cols_only(
  userId = col_integer(),
  movieId = col_integer(),
  rating = col_double()
))

movies <- read_csv(movies_path, col_types = cols_only(
  movieId = col_integer(),
  title = col_character(),
  genres = col_character()
))

# ── 1. Rating Distribution Plot ───────────────────────────────────────────────
message("Generating Plot 1: Rating Distribution...")
p1 <- ggplot(ratings, aes(x = factor(rating))) +
  geom_bar(fill = "#ff6b35", color = "#c94e1d", alpha = 0.85, width = 0.6) +
  geom_text(stat='count', aes(label=scales::comma(..count..)), vjust = -0.4, fontface = "bold", size = 3.5) +
  theme_minimal(base_size = 13) +
  labs(
    title = "CineAI - MovieLens 1M Rating Frequency Distribution",
    subtitle = "BDA Experiment 10: R ggplot2 Exploratory Analysis",
    x = "Star Rating (1.0 to 5.0)",
    y = "Total Interaction Count"
  ) +
  scale_y_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.1))) +
  theme(
    plot.title = element_text(face = "bold", color = "#1a1a1a"),
    plot.subtitle = element_text(color = "#666666"),
    panel.grid.minor = element_blank()
  )

ggsave(file.path(output_dir, "r_rating_distribution.png"), plot = p1, width = 8, height = 5, dpi = 300)

# ── 2. Long-Tail Power Law Distribution ──────────────────────────────────────
message("Generating Plot 2: Long-Tail Distribution...")
movie_counts <- ratings %>%
  group_by(movieId) %>%
  summarise(n_ratings = n()) %>%
  arrange(desc(n_ratings)) %>%
  mutate(rank = row_number())

p2 <- ggplot(movie_counts, aes(x = rank, y = n_ratings)) +
  geom_area(fill = "#42a5f5", alpha = 0.35) +
  geom_line(color = "#1565c0", size = 1) +
  theme_minimal(base_size = 13) +
  labs(
    title = "The Long-Tail Problem in Recommender Systems",
    subtitle = "Rank vs Rating Frequency (Classic Power-Law Decay)",
    x = "Movie Rank (Most to Least Popular)",
    y = "Number of Ratings"
  ) +
  scale_x_continuous(labels = scales::comma) +
  scale_y_continuous(labels = scales::comma) +
  theme(
    plot.title = element_text(face = "bold", color = "#1a1a1a"),
    plot.subtitle = element_text(color = "#666666")
  )

ggsave(file.path(output_dir, "r_long_tail_distribution.png"), plot = p2, width = 8, height = 5, dpi = 300)

# ── 3. Genre Ratings Comparison ──────────────────────────────────────────────
message("Generating Plot 3: Top Genres...")
merged <- ratings %>%
  inner_join(movies, by = "movieId") %>%
  group_by(genres) %>%
  summarise(
    avg_rating = mean(rating),
    count = n()
  ) %>%
  filter(count >= 5000) %>%
  arrange(desc(avg_rating)) %>%
  head(10)

p3 <- ggplot(merged, aes(x = reorder(genres, avg_rating), y = avg_rating)) +
  geom_col(fill = "#66bb6a", color = "#2e7d32", alpha = 0.85, width = 0.65) +
  coord_flip(ylim = c(3.0, 4.5)) +
  theme_minimal(base_size = 13) +
  labs(
    title = "Top Genre Combinations by Mean Rating (n >= 5,000)",
    x = "Genre Hierarchy",
    y = "Average Star Rating"
  ) +
  theme(
    plot.title = element_text(face = "bold", color = "#1a1a1a")
  )

ggsave(file.path(output_dir, "r_genre_performance.png"), plot = p3, width = 8, height = 5, dpi = 300)

message("All R plots saved to logs/plots_r/")
