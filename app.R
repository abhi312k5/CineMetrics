# ==============================================================================
#  CineMetrics - Movie Recommendation System & Analytics Dashboard
#  Big Data Analytics with R
#
#  Run:  shiny::runApp("movie-recommender")
#  Data: MovieLens (run download_data.R once). If no data is found the app
#        generates a realistic synthetic dataset so it always works.
# ==============================================================================

suppressPackageStartupMessages({
  library(shiny)
  library(shinydashboard)
  library(data.table)
  library(ggplot2)
  library(plotly)
  library(DT)
  library(Matrix)
  library(recommenderlab)
})

# ------------------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------------------
DATA_DIR           <- "data/ml-latest-small"
MIN_MOVIE_RATINGS  <- 10   # drop movies rated fewer times than this
MIN_USER_RATINGS   <- 10   # drop users who rated fewer movies than this

COL <- list(blue = "#2563eb", navy = "#0f172a", orange = "#f59e0b",
            green = "#10b981", red = "#ef4444", purple = "#8b5cf6",
            grey = "#64748b")

theme_cine <- function() {
  theme_minimal(base_size = 12) +
    theme(panel.grid.minor = element_blank(),
          plot.title = element_text(face = "bold"),
          legend.position = "right")
}
pl <- function(p, tooltip = "all") {
  ggplotly(p, tooltip = tooltip) %>% config(displayModeBar = FALSE)
}

# ------------------------------------------------------------------------------
# 2. DATA INGESTION  (data.table = fast, memory-efficient, scales to big data)
# ------------------------------------------------------------------------------
make_synthetic <- function(n_users = 600, n_movies = 1500, seed = 42) {
  set.seed(seed)
  genres <- c("Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
              "Documentary", "Drama", "Fantasy", "Horror", "Mystery", "Romance",
              "Sci-Fi", "Thriller")
  adj  <- c("Silent", "Crimson", "Last", "Hidden", "Broken", "Midnight", "Golden",
            "Lost", "Wild", "Eternal", "Dark", "Electric", "Frozen", "Secret",
            "Burning", "Distant")
  noun <- c("Horizon", "Empire", "Garden", "Echo", "Voyage", "Promise", "Shadow",
            "Harvest", "Frontier", "Letter", "Machine", "Kingdom", "River",
            "Signal", "Mirror", "Season")
  base  <- paste(sample(adj, n_movies, TRUE), sample(noun, n_movies, TRUE))
  nth   <- ave(seq_along(base), base, FUN = seq_along)
  base  <- ifelse(nth == 1, base, paste(base, nth))
  years <- sample(1950:2018, n_movies, replace = TRUE)
  movies <- data.table(
    movieId = seq_len(n_movies),
    title   = paste0(base, " (", years, ")"),
    genres  = vapply(seq_len(n_movies), function(i)
      paste(sample(genres, sample(1:3, 1)), collapse = "|"), character(1))
  )

  k  <- 5
  U  <- matrix(rnorm(n_users * k), n_users)
  V  <- matrix(rnorm(n_movies * k), n_movies)
  ub <- rnorm(n_users, 0, 0.3)
  mb <- rnorm(n_movies, 0, 0.4)
  pop <- rlnorm(n_movies, 0, 1.2)
  n_per_user <- pmin(n_movies, pmax(20, round(rlnorm(n_users, 4.2, 0.7))))
  r <- rbindlist(lapply(seq_len(n_users), function(u)
    data.table(userId = u, movieId = sample.int(n_movies, n_per_user[u], prob = pop))))
  score <- 3.5 + ub[r$userId] + mb[r$movieId] +
    0.35 * rowSums(U[r$userId, , drop = FALSE] * V[r$movieId, , drop = FALSE]) +
    rnorm(nrow(r), 0, 0.5)
  r[, rating := pmin(5, pmax(0.5, round(score * 2) / 2))]
  r[, timestamp := as.integer(as.POSIXct("2012-01-01", tz = "UTC")) +
       sample.int(6L * 365L * 86400L, .N, replace = TRUE)]
  list(ratings = r, movies = movies)
}

load_data <- function() {
  rf <- file.path(DATA_DIR, "ratings.csv")
  mf <- file.path(DATA_DIR, "movies.csv")
  if (file.exists(rf) && file.exists(mf)) {
    list(ratings = fread(rf), movies = fread(mf),
         source = "MovieLens (ml-latest-small)")
  } else {
    s <- make_synthetic()
    list(ratings = s$ratings, movies = s$movies,
         source = "Synthetic demo data (run download_data.R for MovieLens)")
  }
}

dat      <- load_data()
ratings  <- dat$ratings[, .(userId, movieId, rating = as.numeric(rating), timestamp)]
movies   <- dat$movies[, .(movieId, title, genres)]
DATA_SRC <- dat$source
RAW_N    <- nrow(ratings)

# ------------------------------------------------------------------------------
# 3. CLEANING & FEATURE ENGINEERING
# ------------------------------------------------------------------------------
ok_m    <- ratings[, .N, by = movieId][N >= MIN_MOVIE_RATINGS, movieId]
ratings <- ratings[movieId %in% ok_m]
ok_u    <- ratings[, .N, by = userId][N >= MIN_USER_RATINGS, userId]
ratings <- ratings[userId %in% ok_u]
movies  <- movies[movieId %in% ratings$movieId]

ratings[, date := as.Date(as.POSIXct(timestamp, origin = "1970-01-01", tz = "UTC"))]

# Sparse user x movie matrix (the core "big data" structure)
u_f <- factor(ratings$userId)
m_f <- factor(ratings$movieId)
sm  <- sparseMatrix(i = as.integer(u_f), j = as.integer(m_f),
                    x = ratings$rating,
                    dimnames = list(levels(u_f), levels(m_f)))
rmat <- as(sm, "realRatingMatrix")

# keep movie table aligned with matrix columns
movies <- movies[match(as.integer(colnames(sm)), movieId)]
movies[, year := suppressWarnings(as.integer(
  ifelse(grepl("\\(\\d{4}\\)\\s*$", title),
         sub(".*\\((\\d{4})\\)\\s*$", "\\1", title), NA)))]

# Movie-level statistics + Bayesian weighted rating (IMDB-style)
mstats <- ratings[, .(n_ratings = .N, avg_rating = mean(rating)), by = movieId]
C      <- mean(ratings$rating)
m_thr  <- as.numeric(quantile(mstats$n_ratings, 0.60))
mstats[, wr := (n_ratings / (n_ratings + m_thr)) * avg_rating +
         (m_thr / (n_ratings + m_thr)) * C]
movies[mstats, `:=`(n_ratings = i.n_ratings, avg_rating = i.avg_rating, wr = i.wr),
       on = "movieId"]

# Genre long table + rating/genre join
genre_dt <- movies[, .(genre = unlist(strsplit(genres, "|", fixed = TRUE))),
                   by = movieId][genre != "(no genres listed)"]
rg <- merge(ratings[, .(movieId, rating)], genre_dt, by = "movieId",
            allow.cartesian = TRUE)

# Content matrix (genres) for content-based similarity
all_genres <- sort(unique(genre_dt$genre))
G <- sparseMatrix(i = match(genre_dt$movieId, movies$movieId),
                  j = match(genre_dt$genre, all_genres), x = 1,
                  dims = c(nrow(movies), length(all_genres)))
Gn <- Diagonal(x = 1 / sqrt(pmax(rowSums(G), 1))) %*% G
cnorm <- sqrt(colSums(sm^2))

# ------------------------------------------------------------------------------
# 4. MODEL TRAINING (collaborative filtering with recommenderlab)
# ------------------------------------------------------------------------------
train_safe <- function(method, param = NULL) {
  tryCatch(Recommender(rmat, method, parameter = param), error = function(e) NULL)
}
models <- Filter(Negate(is.null), list(
  "Popularity Baseline" = train_safe("POPULAR"),
  "User-Based CF"       = train_safe("UBCF", list(method = "Cosine", nn = 30)),
  "Item-Based CF"       = train_safe("IBCF", list(method = "Cosine", k = 30)),
  "SVD Matrix Factorization" = train_safe("SVD", list(k = 20))
))

recommend_for <- function(user, algo, n) {
  r  <- models[[algo]]
  pr <- predict(r, rmat[user, ], type = "ratings")
  v  <- as(pr, "matrix")[1, ]
  v  <- sort(v[!is.na(v)], decreasing = TRUE)
  v  <- head(v, n)
  if (!length(v)) return(data.table())
  r  <- data.table(movieId = as.integer(names(v)),
                   pred = pmin(5, pmax(0.5, round(unname(v), 2))))
  out <- movies[r, on = "movieId", nomatch = 0]
  out[, rank := seq_len(.N)]
  out[]
}

similar_to <- function(mid, type, n) {
  j <- match(mid, movies$movieId)
  if (type == "Genre (content-based)") {
    s <- drop(as.matrix(Gn %*% t(Gn[j, , drop = FALSE])))
  } else {
    s <- drop(as.matrix(crossprod(sm, sm[, j]))) / (cnorm * cnorm[j])
  }
  s[is.na(s)] <- 0
  s[j] <- -Inf
  # tiny quality tie-break so equal-similarity movies are ordered by rating
  top <- order(s + 0.001 * movies$wr, decreasing = TRUE)[seq_len(n)]
  out <- movies[top]
  out[, similarity := round(s[top], 3)]
  out[]
}

# ------------------------------------------------------------------------------
# 5. USER INTERFACE
# ------------------------------------------------------------------------------
css <- "
.skin-blue .main-header .logo{background:#0f172a;font-weight:700;letter-spacing:.5px}
.skin-blue .main-header .logo:hover{background:#0f172a}
.skin-blue .main-header .navbar{background:#1e293b}
.skin-blue .main-sidebar{background:#0f172a}
.skin-blue .sidebar-menu>li.active>a,.skin-blue .sidebar-menu>li:hover>a{
  background:#1e293b;border-left-color:#f59e0b}
.content-wrapper{background:#f1f5f9}
.box{border-top:3px solid #2563eb;border-radius:8px;box-shadow:0 1px 4px rgba(15,23,42,.08)}
.box.box-solid.box-primary>.box-header{background:#2563eb}
.small-box{border-radius:8px}
.small-box h3{font-size:26px}
.small-box .icon{font-size:55px}
.hero{background:linear-gradient(135deg,#0f172a,#2563eb);color:#fff;border-radius:10px;
  padding:22px 26px;margin-bottom:18px}
.hero h2{margin:0 0 6px 0;font-weight:700}
.hero p{margin:0;opacity:.85}
.about p,.about li{font-size:15px;line-height:1.6}
"

ui <- dashboardPage(
  title = "CineMetrics",
  dashboardHeader(title = "CineMetrics"),
  dashboardSidebar(
    sidebarMenu(
      id = "tabs",
      menuItem("Overview",        tabName = "overview",  icon = icon("tachometer-alt")),
      menuItem("Genre Analysis",  tabName = "genres",    icon = icon("theater-masks")),
      menuItem("Movie Rankings",  tabName = "rankings",  icon = icon("trophy")),
      menuItem("Trends",          tabName = "trends",    icon = icon("chart-line")),
      menuItem("Recommendations", tabName = "reco",      icon = icon("magic")),
      menuItem("Similar Movies",  tabName = "similar",   icon = icon("film")),
      menuItem("Model Evaluation",tabName = "eval",      icon = icon("flask")),
      menuItem("About Project",   tabName = "about",     icon = icon("info-circle"))
    )
  ),
  dashboardBody(
    tags$head(tags$style(HTML(css))),
    tabItems(
      # ---------------- Overview ----------------
      tabItem("overview",
        div(class = "hero",
            h2("Movie Recommendation System"),
            p("Big data analytics with R: sparse matrices, collaborative filtering and interactive analytics.")),
        fluidRow(valueBoxOutput("vb_ratings", 2), valueBoxOutput("vb_users", 2),
                 valueBoxOutput("vb_movies", 2),  valueBoxOutput("vb_avg", 2),
                 valueBoxOutput("vb_sparse", 2),  valueBoxOutput("vb_models", 2)),
        fluidRow(
          box(title = "Rating Distribution", width = 6, plotlyOutput("p_rating_dist", height = 300)),
          box(title = "Average Rating per User", width = 6, plotlyOutput("p_user_avg", height = 300))
        ),
        fluidRow(
          box(title = "Ratings per User (log scale)", width = 6, plotlyOutput("p_user_activity", height = 300)),
          box(title = "Movie Popularity Long Tail", width = 6, plotlyOutput("p_long_tail", height = 300))
        )
      ),
      # ---------------- Genres ----------------
      tabItem("genres",
        fluidRow(
          box(title = "Number of Ratings by Genre", width = 6, plotlyOutput("p_genre_count", height = 420)),
          box(title = "Average Rating by Genre", width = 6, plotlyOutput("p_genre_avg", height = 420))
        ),
        fluidRow(
          box(title = "Genre x Decade: Average Rating Heatmap", width = 12,
              plotlyOutput("p_genre_heat", height = 420))
        )
      ),
      # ---------------- Rankings ----------------
      tabItem("rankings",
        fluidRow(
          box(width = 4, title = "Controls",
              sliderInput("min_n", "Minimum number of ratings",
                          min = 1, max = max(50, as.integer(quantile(movies$n_ratings, .95))),
                          value = min(20, max(movies$n_ratings))),
              radioButtons("rank_by", "Rank movies by",
                           c("Weighted rating" = "wr", "Average rating" = "avg_rating",
                             "Number of ratings" = "n_ratings")),
              helpText("Weighted rating shrinks scores of rarely-rated movies toward the global mean.")),
          box(width = 8, title = "Top 15 Movies", plotlyOutput("p_top", height = 380))
        ),
        fluidRow(
          box(width = 6, title = "Rating vs Popularity", plotlyOutput("p_scatter", height = 380)),
          box(width = 6, title = "Ranking Table", DTOutput("t_top"))
        )
      ),
      # ---------------- Trends ----------------
      tabItem("trends",
        fluidRow(box(title = "Ratings Activity Over Time (monthly)", width = 12,
                     plotlyOutput("p_time", height = 300))),
        fluidRow(
          box(title = "Movies in Catalogue by Release Year", width = 6, plotlyOutput("p_year_count", height = 320)),
          box(title = "Average Rating by Release Year", width = 6, plotlyOutput("p_year_avg", height = 320))
        )
      ),
      # ---------------- Recommendations ----------------
      tabItem("reco",
        fluidRow(
          box(width = 3, title = "Personalised Recommendations",
              selectizeInput("user", "Select user ID", choices = rownames(sm),
                             selected = rownames(sm)[1]),
              selectInput("algo", "Algorithm", choices = names(models),
                          selected = names(models)[min(2, length(models))]),
              sliderInput("topn", "Number of recommendations", 5, 20, 10, step = 1),
              downloadButton("dl_reco", "Download CSV", class = "btn-primary")),
          valueBoxOutput("vb_u_n", 3), valueBoxOutput("vb_u_avg", 3), valueBoxOutput("vb_u_fav", 3)
        ),
        fluidRow(
          box(title = "Top Recommended Movies", width = 7, DTOutput("t_reco")),
          box(title = "Predicted Ratings", width = 5, plotlyOutput("p_reco", height = 420))
        ),
        fluidRow(
          box(title = "This User's Favourite Movies", width = 7, DTOutput("t_hist")),
          box(title = "User Genre Profile", width = 5, plotlyOutput("p_profile", height = 330))
        )
      ),
      # ---------------- Similar ----------------
      tabItem("similar",
        fluidRow(
          box(width = 4, title = "Find Similar Movies",
              selectizeInput("sim_movie", "Pick a movie",
                             choices = setNames(movies$movieId, movies$title),
                             selected = movies[order(-n_ratings)]$movieId[1]),
              radioButtons("sim_type", "Similarity based on",
                           c("Genre (content-based)", "User ratings (collaborative)")),
              sliderInput("sim_n", "How many", 5, 20, 10)),
          box(width = 8, title = "Most Similar Movies", plotlyOutput("p_sim", height = 360))
        ),
        fluidRow(box(width = 12, title = "Details", DTOutput("t_sim")))
      ),
      # ---------------- Evaluation ----------------
      tabItem("eval",
        fluidRow(
          box(width = 4, title = "Evaluation Setup",
              p("80/20 user split. For each test user, the algorithm sees only a few known ratings and must predict the rest."),
              sliderInput("given", "Ratings revealed per test user", 3, 8, 5, step = 1),
              sliderInput("good", "A 'good' movie has rating >=", 3, 5, 4, step = 0.5),
              actionButton("run_eval", "Run evaluation", icon = icon("play"), class = "btn-primary"),
              helpText("May take a minute on larger datasets.")),
          box(width = 8, title = "Accuracy Summary", DTOutput("t_eval"))
        ),
        fluidRow(
          box(width = 6, title = "ROC Curve", plotlyOutput("p_roc", height = 360)),
          box(width = 6, title = "Precision-Recall Curve", plotlyOutput("p_pr", height = 360))
        )
      ),
      # ---------------- About ----------------
      tabItem("about",
        box(width = 12, title = "About this Project", class = "about",
          HTML(paste0("
          <p><b>Objective.</b> Build a scalable movie recommendation system in R and analyse
          user rating behaviour with interactive visualisations.</p>
          <p><b>Data source in use:</b> ", DATA_SRC, ".<br>
          <b>Raw ratings:</b> ", format(RAW_N, big.mark = ","), " &rarr;
          <b>after cleaning:</b> ", format(nrow(ratings), big.mark = ","), ".</p>
          <p><b>Pipeline.</b></p>
          <ol>
            <li><b>Ingest</b> with <code>data.table::fread</code> (fast, multi-threaded).</li>
            <li><b>Clean</b>: remove movies and users with too few ratings.</li>
            <li><b>Represent</b> ratings as a sparse user&times;movie matrix (<code>Matrix</code>, <code>recommenderlab</code>).</li>
            <li><b>Model</b>: Popularity baseline, User-Based CF, Item-Based CF, SVD.</li>
            <li><b>Evaluate</b>: RMSE, MAE, ROC and Precision-Recall on a held-out split.</li>
            <li><b>Serve</b> through a Shiny dashboard.</li>
          </ol>
          <p><b>Scaling to truly big data.</b> The same pipeline runs on the full MovieLens 25M
          dataset by swapping the data file and using ALS in <code>sparklyr</code> (Spark MLlib).</p>")))
      )
    )
  )
)

# ------------------------------------------------------------------------------
# 6. SERVER
# ------------------------------------------------------------------------------
server <- function(input, output, session) {

  # ----- Overview -----
  vb <- function(value, subtitle, icon_name, color)
    valueBox(value, subtitle, icon = icon(icon_name), color = color)

  output$vb_ratings <- renderValueBox(vb(format(nrow(ratings), big.mark = ","), "Ratings", "star", "blue"))
  output$vb_users   <- renderValueBox(vb(format(nrow(sm), big.mark = ","), "Users", "users", "purple"))
  output$vb_movies  <- renderValueBox(vb(format(ncol(sm), big.mark = ","), "Movies", "film", "green"))
  output$vb_avg     <- renderValueBox(vb(sprintf("%.2f", C), "Average rating", "thumbs-up", "yellow"))
  output$vb_sparse  <- renderValueBox(vb(sprintf("%.1f%%", 100 * (1 - nrow(ratings) / (nrow(sm) * ncol(sm)))),
                                         "Matrix sparsity", "th", "red"))
  output$vb_models  <- renderValueBox(vb(length(models), "Algorithms trained", "robot", "navy"))

  output$p_rating_dist <- renderPlotly({
    d <- ratings[, .N, by = rating]
    pl(ggplot(d, aes(factor(rating), N, text = paste0("Rating ", rating, ": ", format(N, big.mark = ",")))) +
         geom_col(fill = COL$blue) + labs(x = "Rating", y = "Count") + theme_cine(), "text")
  })
  output$p_user_avg <- renderPlotly({
    d <- ratings[, .(avg = mean(rating)), by = userId]
    pl(ggplot(d, aes(avg)) + geom_histogram(bins = 30, fill = COL$purple, colour = "white") +
         labs(x = "Mean rating given by user", y = "Users") + theme_cine())
  })
  output$p_user_activity <- renderPlotly({
    d <- ratings[, .N, by = userId]
    pl(ggplot(d, aes(N)) + geom_histogram(bins = 35, fill = COL$green, colour = "white") +
         scale_x_log10() + labs(x = "Ratings per user (log)", y = "Users") + theme_cine())
  })
  output$p_long_tail <- renderPlotly({
    d <- movies[order(-n_ratings)][, rank := .I]
    pl(ggplot(d, aes(rank, n_ratings)) +
         geom_line(colour = COL$orange, linewidth = 1) +
         geom_point(aes(text = paste0(title, "<br>Rank: ", rank,
                                      "<br>Ratings: ", n_ratings)),
                    colour = COL$orange, alpha = .35, size = 1) +
         scale_y_log10() +
         labs(x = "Movie rank", y = "Ratings (log)") + theme_cine(), "text")
  })

  # ----- Genres -----
  genre_stats <- rg[, .(n = .N, avg = mean(rating), movies = uniqueN(movieId)), by = genre]

  output$p_genre_count <- renderPlotly({
    pl(ggplot(genre_stats, aes(reorder(genre, n), n, fill = avg,
                               text = paste0(genre, "<br>Ratings: ", format(n, big.mark = ","),
                                             "<br>Avg: ", round(avg, 2)))) +
         geom_col() + coord_flip() +
         scale_fill_gradient(low = "#bfdbfe", high = COL$blue, name = "Avg") +
         labs(x = NULL, y = "Ratings") + theme_cine(), "text")
  })
  output$p_genre_avg <- renderPlotly({
    pl(ggplot(genre_stats, aes(reorder(genre, avg), avg,
                               text = paste0(genre, "<br>Avg: ", round(avg, 2)))) +
         geom_col(fill = COL$orange) +
         coord_flip(ylim = c(min(genre_stats$avg) - .3, max(genre_stats$avg) + .1)) +
         labs(x = NULL, y = "Average rating") + theme_cine(), "text")
  })
  output$p_genre_heat <- renderPlotly({
    hm <- merge(rg, movies[, .(movieId, year)], by = "movieId")
    hm <- hm[!is.na(year) & year >= 1950]
    hm[, decade := paste0(floor(year / 10) * 10, "s")]
    hm <- hm[, .(avg = mean(rating), n = .N), by = .(genre, decade)][n >= 30]
    pl(ggplot(hm, aes(decade, genre, fill = avg,
                      text = paste0(genre, " / ", decade, "<br>Avg: ", round(avg, 2), "<br>n = ", n))) +
         geom_tile(colour = "white") +
         scale_fill_gradient2(low = COL$red, mid = "#fef3c7", high = COL$green,
                              midpoint = C, name = "Avg") +
         labs(x = NULL, y = NULL) + theme_cine(), "text")
  })

  # ----- Rankings -----
  top_movies <- reactive({
    d <- movies[n_ratings >= input$min_n]
    d <- d[order(-get(input$rank_by))]
    head(d, 15)
  })
  output$p_top <- renderPlotly({
    d <- top_movies(); req(nrow(d) > 0)
    d[, short := substr(title, 1, 38)]
    metric <- input$rank_by
    pl(ggplot(d, aes(reorder(short, get(metric)), get(metric),
                     text = paste0(title, "<br>Avg: ", round(avg_rating, 2),
                                   "<br>Ratings: ", n_ratings))) +
         geom_col(fill = COL$blue) + coord_flip() +
         labs(x = NULL, y = names(which(c("Weighted rating" = "wr", "Average rating" = "avg_rating",
                                          "Number of ratings" = "n_ratings") == metric))) +
         theme_cine(), "text")
  })
  output$p_scatter <- renderPlotly({
    d <- movies[n_ratings >= input$min_n]
    if (nrow(d) > 2500) d <- d[sample(.N, 2500)]
    pl(ggplot(d, aes(n_ratings, avg_rating, colour = wr,
                     text = paste0(title, "<br>Avg: ", round(avg_rating, 2), "<br>Ratings: ", n_ratings))) +
         geom_point(alpha = .6) + scale_x_log10() +
         scale_colour_gradient(low = "#93c5fd", high = COL$navy, name = "WR") +
         labs(x = "Number of ratings (log)", y = "Average rating") + theme_cine(), "text")
  })
  output$t_top <- renderDT({
    d <- movies[n_ratings >= input$min_n][order(-get(input$rank_by))]
    d <- d[, .(Title = title, Genres = genres, Ratings = n_ratings,
               Avg = round(avg_rating, 2), Weighted = round(wr, 2))]
    datatable(d, rownames = FALSE, options = list(pageLength = 8, dom = "tp", scrollX = TRUE))
  })

  # ----- Trends -----
  output$p_time <- renderPlotly({
    d <- ratings[, .N, by = .(month = as.Date(format(date, "%Y-%m-01")))][order(month)]
    pl(ggplot(d, aes(month, N)) + geom_area(fill = "#bfdbfe") + geom_line(colour = COL$blue, linewidth = .8) +
         labs(x = NULL, y = "Ratings") + theme_cine())
  })
  output$p_year_count <- renderPlotly({
    d <- movies[!is.na(year) & year >= 1930, .N, by = year]
    pl(ggplot(d, aes(year, N)) + geom_col(fill = COL$purple) +
         labs(x = "Release year", y = "Movies") + theme_cine())
  })
  output$p_year_avg <- renderPlotly({
    d <- merge(ratings[, .(movieId, rating)], movies[, .(movieId, year)], by = "movieId")
    d <- d[!is.na(year) & year >= 1930, .(avg = mean(rating), n = .N), by = year][n >= 30]
    pl(ggplot(d, aes(year, avg)) +
         geom_line(colour = COL$orange, linewidth = 1) +
         geom_point(aes(text = paste0(year, ": ", round(avg, 2), " (n = ", n, ")")),
                    colour = COL$orange, size = 1.5) +
         geom_smooth(method = "loess", formula = y ~ x, se = FALSE,
                     colour = COL$navy, linewidth = 1) +
         labs(x = "Release year", y = "Average rating") + theme_cine(), "text")
  })

  # ----- Recommendations -----
  reco <- reactive({
    req(input$user, input$algo)
    withProgress(message = "Computing recommendations...", {
      recommend_for(input$user, input$algo, input$topn)
    })
  })
  user_ratings <- reactive(ratings[userId == as.integer(input$user)])

  output$vb_u_n   <- renderValueBox(vb(nrow(user_ratings()), "Movies rated", "list", "blue"))
  output$vb_u_avg <- renderValueBox(vb(sprintf("%.2f", mean(user_ratings()$rating)), "User's avg rating", "star", "yellow"))
  output$vb_u_fav <- renderValueBox({
    ug <- merge(user_ratings()[, .(movieId, rating)], genre_dt, by = "movieId")[, .N, by = genre][order(-N)]
    vb(if (nrow(ug)) ug$genre[1] else "-", "Favourite genre", "heart", "purple")
  })

  output$t_reco <- renderDT({
    d <- reco(); req(nrow(d) > 0)
    d <- d[, .(`#` = rank, Title = title, Genres = genres, `Predicted` = pred,
               `Avg rating` = round(avg_rating, 2), Ratings = n_ratings)]
    datatable(d, rownames = FALSE, options = list(pageLength = 10, dom = "tp", scrollX = TRUE)) %>%
      formatStyle("Predicted", background = styleColorBar(c(0, 5), "#93c5fd"),
                  backgroundSize = "100% 70%", backgroundRepeat = "no-repeat",
                  backgroundPosition = "center")
  })
  output$p_reco <- renderPlotly({
    d <- reco(); req(nrow(d) > 0)
    d[, short := substr(title, 1, 34)]
    pl(ggplot(d, aes(reorder(short, pred), pred, text = paste0(title, "<br>Predicted: ", pred))) +
         geom_col(fill = COL$blue) + coord_flip(ylim = c(0, 5)) +
         labs(x = NULL, y = "Predicted rating") + theme_cine(), "text")
  })
  output$t_hist <- renderDT({
    h <- head(user_ratings()[order(-rating, -timestamp)], 10)
    d <- movies[h, on = "movieId"][, .(Title = title, Genres = genres, `User rating` = rating)]
    datatable(d, rownames = FALSE, options = list(dom = "t", pageLength = 10, scrollX = TRUE))
  })
  output$p_profile <- renderPlotly({
    ug <- merge(user_ratings()[, .(movieId, rating)], genre_dt, by = "movieId")[
      , .(n = .N, avg = mean(rating)), by = genre]
    req(nrow(ug) > 0)
    pl(ggplot(ug, aes(reorder(genre, n), n, text = paste0(genre, "<br>Movies: ", n, "<br>Avg: ", round(avg, 2)))) +
         geom_col(fill = COL$purple) + coord_flip() + labs(x = NULL, y = "Movies rated") + theme_cine(), "text")
  })
  output$dl_reco <- downloadHandler(
    filename = function() paste0("recommendations_user", input$user, ".csv"),
    content = function(file) {
      d <- reco()
      fwrite(d[, .(rank, movieId, title, genres, predicted_rating = pred)], file)
    })

  # ----- Similar movies -----
  sim <- reactive({
    req(input$sim_movie)
    similar_to(as.integer(input$sim_movie), input$sim_type, input$sim_n)
  })
  output$p_sim <- renderPlotly({
    d <- sim()
    d[, short := substr(title, 1, 38)]
    pl(ggplot(d, aes(reorder(short, similarity), similarity,
                     text = paste0(title, "<br>Similarity: ", similarity))) +
         geom_col(fill = COL$green) + coord_flip() +
         labs(x = NULL, y = "Similarity score") + theme_cine(), "text")
  })
  output$t_sim <- renderDT({
    d <- sim()[, .(Title = title, Genres = genres, Similarity = similarity,
                   `Avg rating` = round(avg_rating, 2), Ratings = n_ratings)]
    datatable(d, rownames = FALSE, options = list(pageLength = 10, dom = "tp", scrollX = TRUE))
  })

  # ----- Evaluation -----
  eval_res <- eventReactive(input$run_eval, {
    withProgress(message = "Evaluating algorithms...", value = 0, {
      set.seed(123)
      n_vals <- c(1, 3, 5, 10, 15, 20)
      sch <- evaluationScheme(rmat, method = "split", train = 0.8,
                              given = input$given, goodRating = input$good)
      methods <- list(
        "Random"              = list(name = "RANDOM",  param = NULL),
        "Popularity Baseline" = list(name = "POPULAR", param = NULL),
        "User-Based CF"       = list(name = "UBCF", param = list(method = "Cosine", nn = 30)),
        "Item-Based CF"       = list(name = "IBCF", param = list(method = "Cosine", k = 30))
      )
      incProgress(0.1, detail = "Top-N accuracy")
      topn <- evaluate(sch, methods, type = "topNList", n = n_vals)
      av   <- avg(topn)
      tn   <- rbindlist(lapply(names(av), function(nm) {
        d <- as.data.table(as.data.frame(av[[nm]]))
        d[, n := n_vals]
        d[, Algorithm := nm]
        d
      }), fill = TRUE)

      incProgress(0.5, detail = "Rating prediction error")
      acc <- rbindlist(lapply(setdiff(names(methods), "Random"), function(nm) {
        r <- Recommender(getData(sch, "train"), methods[[nm]]$name, parameter = methods[[nm]]$param)
        p <- predict(r, getData(sch, "known"), type = "ratings")
        a <- calcPredictionAccuracy(p, getData(sch, "unknown"))
        data.table(Algorithm = nm, RMSE = unname(a["RMSE"]), MAE = unname(a["MAE"]))
      }))
      list(topn = tn, acc = acc)
    })
  })

  output$t_eval <- renderDT({
    ev <- eval_res()
    p10 <- ev$topn[n == 10, .(Algorithm, `Precision@10` = round(precision, 4), `Recall@10` = round(recall, 4))]
    d <- merge(p10, ev$acc[, .(Algorithm, RMSE = round(RMSE, 3), MAE = round(MAE, 3))],
               by = "Algorithm", all.x = TRUE)
    datatable(d, rownames = FALSE, options = list(dom = "t"))
  })
  output$p_roc <- renderPlotly({
    d <- eval_res()$topn
    pl(ggplot(d, aes(FPR, TPR, colour = Algorithm, group = Algorithm,
                     text = paste0(Algorithm, " (top-", n, ")"))) +
         geom_line() + geom_point() + geom_abline(linetype = 2, colour = "grey60") +
         labs(x = "False positive rate", y = "True positive rate") + theme_cine(), "text")
  })
  output$p_pr <- renderPlotly({
    d <- eval_res()$topn
    pl(ggplot(d, aes(recall, precision, colour = Algorithm, group = Algorithm,
                     text = paste0(Algorithm, " (top-", n, ")"))) +
         geom_line() + geom_point() +
         labs(x = "Recall", y = "Precision") + theme_cine(), "text")
  })
}

shinyApp(ui, server)
