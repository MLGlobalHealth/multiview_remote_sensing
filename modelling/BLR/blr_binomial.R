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
#   Rscript modelling/BLR/blr_binomial.R <data_dir> <out_dir> [target_prefix] [n_iter] [n_pc]
#   target_prefix defaults to "deprived_sev"; columns expected: <prefix>_k, <prefix>_n.
#   n_pc (default 100): reduce the embeddings to this many principal components
#     before the horseshoe. A horseshoe over hundreds of dense, entangled
#     embedding dimensions is both extremely slow (NUTS funnel: ~days for a few
#     hundred iters) and ill-posed (it assumes a sparse set of meaningful
#     predictors). PCA first makes the fit tractable and the shrinkage sensible.
#     Set n_pc=0 to keep all features (faithful to a raw-feature horseshoe, but
#     expect a multi-day run / consider cmdstanr within-chain threading).

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
n_pc     <- ifelse(length(args) >= 5, as.integer(args[[5]]), 100L)
prior_nm <- ifelse(length(args) >= 6, args[[6]], "hs")  # "hs" | "normal"
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

# Optional PCA reduction (fit on train, applied to test — no leakage). Turns the
# horseshoe into a tractable model over a handful of orthogonal components.
if (n_pc > 0 && n_pc < ncol(Xtr)) {
  pca <- prcomp(Xtr, center = FALSE, scale. = FALSE)  # Xtr already standardised
  varexp <- 100 * sum(pca$sdev[1:n_pc]^2) / sum(pca$sdev^2)
  Xtr <- pca$x[, 1:n_pc, drop = FALSE]
  Xts <- predict(pca, Xts)[, 1:n_pc, drop = FALSE]
  cat(sprintf("PCA: %d features -> %d PCs (%.1f%% variance retained)\n",
              length(pca$sdev), n_pc, varexp))
}

# Give features syntactically valid, stable names. The DINO export writes integer
# column headers (0..767); data.frame() would otherwise rename them to X0..X767
# and break explicit column references.
colnames(Xtr) <- colnames(Xts) <- paste0("f", seq_len(ncol(Xtr)))

D <- ncol(Xtr); N <- nrow(Xtr)
k <- as.integer(round(Ytr[[k_col]]))
n <- as.integer(round(Ytr[[n_col]]))
k <- pmin(k, n)  # guard against rounding making k > n

# Coefficient prior. "normal" is a weakly-informative Gaussian (ridge-like) that
# samples in minutes; "hs" is the regularised horseshoe (Piironen & Vehtari 2017),
# faithful to sparse shrinkage but slow here due to the funnel geometry.
if (prior_nm == "normal") {
  prior_coef <- normal(0, 2.5, autoscale = TRUE)
  cat(sprintf("Prior: normal(0, 2.5) autoscaled\n"))
} else {
  p0 <- max(1, floor(0.05 * D))                   # ~5% of components relevant
  global_scale <- (p0 / (D - p0)) / sqrt(N)
  prior_coef <- hs(df = 1, global_df = 1, global_scale = global_scale,
                   slab_df = 4, slab_scale = 2.5)
  cat(sprintf("Prior: regularised horseshoe, p0=%d, global_scale=%.3g\n", p0, global_scale))
}

df_tr <- data.frame(k = k, n = n, Xtr, check.names = FALSE)
cat(sprintf("Fitting binomial model: N=%d clusters, D=%d features, prior=%s\n",
            N, D, prior_nm))

# In cbind(k, n-k) ~ ., the '.' expands to every column except those named in the
# response (k, n), i.e. the features only — so n is the trial count, not a predictor.
fit <- stan_glm(
  cbind(k, n - k) ~ .,
  data    = df_tr,
  family  = binomial(link = "logit"),
  prior   = prior_coef,
  prior_intercept = normal(0, 5),
  # QR=TRUE orthogonalises the (highly correlated) embedding dimensions internally
  # for sampling and reports coefficients back on the original scale. Same model,
  # same features — it just stops NUTS from hitting max tree-depth on the
  # correlated raw-feature posterior. This is a reparameterisation, NOT PCA/feature
  # reduction. Most useful with the normal prior; harmless otherwise.
  QR      = TRUE,
  chains  = 4,
  iter    = n_iter,
  seed    = 1234,
  refresh = max(1L, n_iter %/% 20L)
)
saveRDS(fit, file.path(out_dir, paste0("fit_binomial_", prefix, ".rds")))

# Expected prevalence (mean probability) per test cluster + its credible interval.
df_ts <- data.frame(Xts, check.names = FALSE)
epred <- posterior_epred(fit, newdata = df_ts)   # draws x N_test, in [0,1]
Yts$pred_prevalence <- apply(epred, 2, median)
Yts$mean_lower      <- apply(epred, 2, quantile, 0.025)
Yts$mean_upper      <- apply(epred, 2, quantile, 0.975)
obs_prev <- Yts[[k_col]] / Yts[[n_col]]
Yts$obs_prevalence <- obs_prev

# Posterior PREDICTIVE interval for the OBSERVED prevalence k/n: fold the binomial
# sampling at each cluster's own n into the draws of the mean probability. This is
# the interval that should cover k/n ~95% of the time; the mean-probability CI
# above is far too narrow for that (it omits sampling noise) and is the reason a
# naive coverage check looks broken. Carrying real n is exactly what makes this
# possible.
n_test <- as.integer(Yts[[n_col]])
ppred <- vapply(seq_len(ncol(epred)),
                function(j) rbinom(nrow(epred), size = n_test[j], prob = epred[, j]) / n_test[j],
                numeric(nrow(epred)))
Yts$pred_lower <- apply(ppred, 2, quantile, 0.025)
Yts$pred_upper <- apply(ppred, 2, quantile, 0.975)

write_csv(Yts, file.path(out_dir, paste0("Y_test_pred_", prefix, ".csv")))

mae         <- mean(abs(Yts$pred_prevalence - obs_prev))
cover_mean  <- mean(obs_prev >= Yts$mean_lower  & obs_prev <= Yts$mean_upper)
cover_pred  <- mean(obs_prev >= Yts$pred_lower  & obs_prev <= Yts$pred_upper)
cat(sprintf("Test prevalence MAE: %.4f\n", mae))
cat(sprintf("95%% coverage of k/n: mean-prob CI %.3f (expected low) | posterior-predictive %.3f (target 0.95)\n",
            cover_mean, cover_pred))
cat(sprintf("Wrote predictions and fit to %s\n", out_dir))
