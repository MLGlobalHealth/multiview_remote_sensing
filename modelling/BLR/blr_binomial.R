# Binomial Bayesian linear regression of cluster-level child-poverty deprivation
# on satellite-image embeddings, with a regularised-horseshoe prior for sparsity.
#
# This replaces the old approach in BLRfit.R, which fabricated the binomial
# denominator with `n = rpois(., 10)` because the survey-processing pipeline only
# emitted cluster *prevalences* (p = k/n), discarding the real counts. With the
# patched survey_processing/main.py we now carry the true k (# deprived children)
# and n (# children surveyed) per cluster, so the likelihood can use real sample
# sizes — a prevalence of 0.5 from n=2 is treated very differently from n=200.
#
# Inputs (produced by modelling/BLR/prep_blr_data.py):
#   <data_dir>/X_train.csv, X_test.csv   feature matrix, one column per embedding dim
#   <data_dir>/Y_train.csv, Y_test.csv   CENTROID_ID, k, n, prevalence (the target)
#
# Usage:
#   Rscript modelling/BLR/blr_binomial.R <data_dir> <out_dir> [target_prefix] [n_iter]
#   target_prefix defaults to "deprived_sev"; columns expected: <prefix>_k, <prefix>_n.

suppressPackageStartupMessages({
  library(rstanarm)
  library(readr)
})
options(mc.cores = parallel::detectCores())

args <- commandArgs(trailingOnly = TRUE)
data_dir <- ifelse(length(args) >= 1, args[[1]], "modelling/BLR/data/multispectral")
out_dir  <- ifelse(length(args) >= 2, args[[2]], "modelling/BLR/output")
prefix   <- ifelse(length(args) >= 3, args[[3]], "deprived_sev")
n_iter   <- ifelse(length(args) >= 4, as.integer(args[[4]]), 2000L)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

k_col <- paste0(prefix, "_k")
n_col <- paste0(prefix, "_n")

# Tolerate either the SatMAE prep output (X_train.csv / Y_train.csv) or the DINO
# evaluate.py output (X_train.csv / y_train.csv, lowercase). The feature matrix
# width is read from the file, so 768-dim DINO and 1024-dim SatMAE both work.
read_first <- function(dir, candidates) {
  for (f in candidates) if (file.exists(file.path(dir, f)))
    return(read_csv(file.path(dir, f), show_col_types = FALSE))
  stop(sprintf("none of {%s} found in %s", paste(candidates, collapse=", "), dir))
}
Xtr <- as.matrix(read_first(data_dir, c("X_train.csv")))
Xts <- as.matrix(read_first(data_dir, c("X_test.csv")))
Ytr <- read_first(data_dir, c("Y_train.csv", "y_train.csv"))
Yts <- read_first(data_dir, c("Y_test.csv",  "y_test.csv"))

# Drop clusters with no valid denominator (n missing or 0) — these are exactly the
# rows the old code papered over with a fake Poisson n.
keep_tr <- is.finite(Ytr[[n_col]]) & Ytr[[n_col]] > 0
keep_ts <- is.finite(Yts[[n_col]]) & Yts[[n_col]] > 0
Xtr <- Xtr[keep_tr, , drop = FALSE]; Ytr <- Ytr[keep_tr, ]
Xts <- Xts[keep_ts, , drop = FALSE]; Yts <- Yts[keep_ts, ]

# Standardise features on the TRAIN statistics (centre/scale) — the horseshoe
# prior assumes comparable coefficient scales, and we must not leak test stats.
mu <- colMeans(Xtr); sdv <- apply(Xtr, 2, sd); sdv[sdv == 0] <- 1
Xtr <- scale(Xtr, center = mu, scale = sdv)
Xts <- scale(Xts, center = mu, scale = sdv)

D <- ncol(Xtr); N <- nrow(Xtr)
k <- as.integer(round(Ytr[[k_col]]))
n <- as.integer(round(Ytr[[n_col]]))
k <- pmin(k, n)  # guard against rounding making k > n

# Regularised horseshoe (Piironen & Vehtari 2017). global_scale encodes the prior
# guess at the number of relevant features p0; rstanarm's hs() is the regularised
# horseshoe, with slab_scale/slab_df taming the unregularised tails.
p0 <- max(1, floor(0.05 * D))                     # ~5% of embedding dims relevant
global_scale <- (p0 / (D - p0)) / sqrt(N)
prior_coef <- hs(df = 1, global_df = 1, global_scale = global_scale,
                 slab_df = 4, slab_scale = 2.5)

df_tr <- data.frame(k = k, n = n, Xtr)
cat(sprintf("Fitting binomial hs() model: N=%d clusters, D=%d features, p0=%d, global_scale=%.3g\n",
            N, D, p0, global_scale))

fit <- stan_glm(
  cbind(k, n - k) ~ .,
  data    = df_tr[, c("k", "n", colnames(Xtr))],
  family  = binomial(link = "logit"),
  prior   = prior_coef,
  prior_intercept = normal(0, 5),
  chains  = 4,
  iter    = n_iter,
  seed    = 1234,
  refresh = max(1L, n_iter %/% 20L)
)
saveRDS(fit, file.path(out_dir, paste0("fit_binomial_", prefix, ".rds")))

# Posterior predictive on test: expected prevalence (probability scale) per cluster.
df_ts <- data.frame(Xts); colnames(df_ts) <- colnames(Xtr)
epred <- posterior_epred(fit, newdata = df_ts)   # draws x N_test, in [0,1]
Yts$pred_prevalence <- apply(epred, 2, median)
Yts$pred_lower      <- apply(epred, 2, quantile, 0.025)
Yts$pred_upper      <- apply(epred, 2, quantile, 0.975)
obs_prev <- Yts[[k_col]] / Yts[[n_col]]
Yts$obs_prevalence <- obs_prev

# Count-scale posterior predictive (binomial draws) for proper-scoring / coverage.
ppred <- posterior_predict(fit, newdata = df_ts)  # uses n from the fit? supply offset via n
write_csv(Yts, file.path(out_dir, paste0("Y_test_pred_", prefix, ".csv")))

mae <- mean(abs(Yts$pred_prevalence - obs_prev))
cover <- mean(obs_prev >= Yts$pred_lower & obs_prev <= Yts$pred_upper)
cat(sprintf("Test prevalence MAE: %.4f | 95%% CI empirical coverage: %.3f\n", mae, cover))
cat(sprintf("Wrote predictions and fit to %s\n", out_dir))
