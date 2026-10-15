# Run once:  source("download_data.R")
# Downloads the MovieLens "latest-small" dataset (100k ratings) into ./data
# For a bigger dataset use ml-25m.zip (25 million ratings) and change DATA_DIR in app.R.

dir.create("data", showWarnings = FALSE)
url  <- "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
dest <- file.path("data", "ml-latest-small.zip")
options(timeout = 300)
download.file(url, dest, mode = "wb")
unzip(dest, exdir = "data")
file.remove(dest)
message("Done. Files in: ", normalizePath("data/ml-latest-small"))
