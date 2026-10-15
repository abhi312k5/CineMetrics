# CineMetrics: Movie Recommendation System (Big Data Analytics with R)

An interactive R Shiny dashboard that analyses movie ratings and recommends movies using collaborative filtering.

## 1. Abstract
Streaming platforms hold millions of user ratings. This project turns a large, sparse user-movie rating matrix into (a) analytics about audience behaviour and (b) personalised recommendations. It compares four recommendation algorithms and serves everything through a professional dashboard.

## 2. Objectives
- Explore and visualise rating behaviour (distribution, genres, trends, popularity).
- Build and compare recommenders: Popularity, User-Based CF, Item-Based CF, SVD.
- Evaluate them with RMSE, MAE, ROC and Precision-Recall.
- Deliver a usable UI for per-user recommendations and similar-movie search.

## 3. Dataset
[MovieLens](https://grouplens.org/datasets/movielens/) (GroupLens). Default: `ml-latest-small` (about 100k ratings, 600 users, 9k movies). For a real big-data run use `ml-25m` (25M ratings).
If no data is found, the app generates a synthetic dataset so it always runs.

## 4. Methodology
| Stage | Technique | R package |
|---|---|---|
| Ingestion | Fast multi-threaded CSV read | `data.table` |
| Cleaning | Drop movies/users with < 10 ratings | `data.table` |
| Representation | Sparse user x movie matrix | `Matrix`, `recommenderlab` |
| Statistics | Bayesian weighted rating, genre explode, decade bins | `data.table` |
| Modelling | POPULAR, UBCF, IBCF, SVD | `recommenderlab` |
| Content similarity | Cosine on genre vectors | `Matrix` |
| Evaluation | 80/20 split, "given-N" protocol | `recommenderlab` |
| Visualisation / UI | Interactive charts, dashboard | `ggplot2`, `plotly`, `DT`, `shinydashboard` |

**Weighted rating** = (v / (v + m)) * R + (m / (v + m)) * C, where R = movie mean, v = its rating count, m = a minimum-votes threshold, C = global mean.

## 5. Algorithms
- **Popularity baseline**: recommends what is rated highest overall; the benchmark to beat.
- **User-Based CF**: finds the 30 most similar users (cosine) and averages their ratings.
- **Item-Based CF**: recommends movies similar to those the user already liked.
- **SVD**: low-rank matrix factorisation that learns latent taste factors.

## 6. Dashboard pages
Overview, Genre Analysis (incl. genre x decade heatmap), Movie Rankings, Trends, Recommendations (CSV download), Similar Movies, Model Evaluation, About.

## 7. How to run
```r
install.packages(c("shiny", "shinydashboard", "data.table", "ggplot2",
                   "plotly", "DT", "Matrix", "recommenderlab"))
setwd("movie-recommender")
source("download_data.R")      # once, needs internet
shiny::runApp()
```
Startup trains the models (about 20-60 s on ml-latest-small).

## 8. Scaling to big data
- `data.table` and sparse matrices keep memory low.
- For 25M+ ratings, use `sparklyr` with Spark MLlib ALS (`ml_als`) for distributed training; the dashboard layer stays the same.

## 9. Limitations and future work
- Cold-start: new users/movies need content-based or hybrid methods.
- Add hybrid model (CF + genres/tags), time-aware weighting, and a "rate movies yourself" onboarding page.

## 10. References
Harper & Konstan (2015), *The MovieLens Datasets*; Hahsler, *recommenderlab* package documentation.
