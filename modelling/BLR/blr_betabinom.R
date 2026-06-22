# Beta-binomial Bayesian regression of cluster-level child-poverty deprivation on
# satellite-image embeddings, with an optional regularised-horseshoe prior.
#
# WHY this exists alongside blr_binomial.R:
#   The plain binomial model (blr_binomial.R) fits the systematic signal well
#   (corr(pred, obs) ~= 0.57) but is badly OVERCONFIDENT: its 95% posterior-
#   predictive interval for the observed prevalence k/n covers only ~0.34 of test
#   clusters (target 0.95). A binomial likelihood assumes the only noise around
#   X*beta is sampling at the cluster's own n; real cluster deprivation has extra
#   variation (unobserved spatial/contextual heterogeneity) that a binomial cannot
#   absorb. The result is intervals that are far too narrow.
#
#   The beta-binomial fixes this with ONE extra parameter: it lets the per-cluster
#   success probability vary around X*beta as Beta(.), so Var(k) exceeds the
#   binomial n*p*(1-p) by a factor that grows with 1/(phi+1). This is the standard
#   overdispersed-count model and is far more parsimonious here than an observation-
#   level random effect, which would add one latent parameter per cluster
#   (N ~= 10.7k). rstanarm has no beta-binomial family, so we use brms.
#
# Inputs (produced by modelling/BLR/prep_blr_data.py, or the DINO evaluate.py export):
#   <data_dir>/X_train.csv, X_test.csv   feature matrix, one column per embedding dim
#   <data_dir>/Y_train.csv, Y_test.csv   CENTROID_ID, k, n, prevalence (the target)
#
# Usage:
#   Rscript modelling/BLR/blr_betabinom.R <data_dir> <out_dir> [target_prefix] [n_iter] [n_pc] [prior]
#   target_prefix defaults to "deprived_sev"; columns expected: <prefix>_k, <prefix>_n.
#   prior: "normal" (default, weakly-informative Gaussian, fast) | "hs" (regularised
#     horseshoe). n_pc (default 0): keep all features and rely on the internal QR
#     reparameterisation (decomp="QR") for sampling. Set n_pc>0 to PCA-reduce first
#     (see notes in blr_binomial.R) if a full-feature fit is too slow.

suppressPackageStartupMessages({
  library(brms)
  library(readr)
})
options(mc.cores = parallel::detectCores())

args <- commandArgs(trailingOnly = TRUE)
data_dir <- ifelse(length(args) >= 1, args[[1]], "modelling/BLR/data/multispectral")
out_dir  <- ifelse(length(args) >= 2, args[[2]], "modelling/BLR/output_bb")
prefix   <- ifelse(length(args) >= 3, args[[3]], "deprived_sev")
n_iter   <- ifelse(length(args) >= 4, as.integer(args[[4]]), 1000L)
n_pc     <- ifelse(length(args) >= 5, as.integer(args[[5]]), 0L)
prior_nm <- ifelse(length(args) >= 6, args[[6]], "normal")  # "normal" | "hs"
n_chains <- ifelse(length(args) >= 7, as.integer(args[[7]]), 4L)  # set 1 to debug (errors surface instead of being masked by mclapply)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

k_col <- paste0(prefix, "_k")
n_col <- paste0(prefix, "_n")

# Tolerate either the SatMAE prep output (Y_train.csv) or the DINO evaluate.py
# output (y_train.csv, lowercase). Feature width is read from the file.
read_first <- function(dir, candidates) {
  for (f in candidates) if (file.exists(file.path(dir, f)))
    return(read_csv(file.path(dir, f), show_col_types = FALSE))
  stop(sprintf("none of {%s} found in %s", paste(candidates, collapse=", "), dir))
}
Xtr <- as.matrix(read_first(data_dir, c("X_train.csv")))
Xts <- as.matrix(read_first(data_dir, c("X_test.csv")))
Ytr <- read_first(data_dir, c("Y_train.csv", "y_train.csv"))
Yts <- read_first(data_dir, c("Y_test.csv",  "y_test.csv"))

# Drop clusters with no valid denominator (n missing or 0).
keep_tr <- is.finite(Ytr[[n_col]]) & Ytr[[n_col]] > 0
keep_ts <- is.finite(Yts[[n_col]]) & Yts[[n_col]] > 0
Xtr <- Xtr[keep_tr, , drop = FALSE]; Ytr <- Ytr[keep_tr, ]
Xts <- Xts[keep_ts, , drop = FALSE]; Yts <- Yts[keep_ts, ]

# Standardise features on the TRAIN statistics (no test leakage).
mu <- colMeans(Xtr); sdv <- apply(Xtr, 2, sd); sdv[sdv == 0] <- 1
Xtr <- scale(Xtr, center = mu, scale = sdv)
Xts <- scale(Xts, center = mu, scale = sdv)

# Optional PCA reduction (fit on train, applied to test — no leakage).
if (n_pc > 0 && n_pc < ncol(Xtr)) {
  pca <- prcomp(Xtr, center = FALSE, scale. = FALSE)  # Xtr already standardised
  varexp <- 100 * sum(pca$sdev[1:n_pc]^2) / sum(pca$sdev^2)
  Xtr <- pca$x[, 1:n_pc, drop = FALSE]
  Xts <- predict(pca, Xts)[, 1:n_pc, drop = FALSE]
  cat(sprintf("PCA: %d features -> %d PCs (%.1f%% variance retained)\n",
              length(pca$sdev), n_pc, varexp))
}

# Stable, syntactically valid feature names (DINO writes integer headers 0..767).
feat <- paste0("f", seq_len(ncol(Xtr)))
colnames(Xtr) <- colnames(Xts) <- feat

D <- ncol(Xtr); N <- nrow(Xtr)
k <- as.integer(round(Ytr[[k_col]]))
n <- as.integer(round(Ytr[[n_col]]))
k <- pmin(k, n)  # guard against rounding making k > n

# Coefficient prior on the (standardised) features.
if (prior_nm == "hs") {
  p0 <- max(1, floor(0.05 * D))                   # ~5% of components relevant
  global_scale <- (p0 / (D - p0)) / sqrt(N)
  prior_b <- set_prior(horseshoe(df = 1, scale_global = global_scale, df_global = 1,
                                 scale_slab = 2.5, df_slab = 4, autoscale = FALSE),
                       class = "b")
  cat(sprintf("Prior: regularised horseshoe, p0=%d, global_scale=%.3g\n", p0, global_scale))
} else {
  prior_b <- set_prior("normal(0, 2.5)", class = "b")
  cat("Prior: normal(0, 2.5) on standardised features\n")
}
priors <- c(prior_b, set_prior("normal(0, 5)", class = "Intercept"))

# Explicit feature formula (avoid '.' so the trials variable n is never swept in as
# a predictor). 'k | trials(n)' is brms' binomial/beta-binomial response syntax.
# decomp = "QR" (a brmsformula option, NOT a brm() argument) orthogonalises the
# highly correlated embedding columns internally for sampling and reports
# coefficients back on the original scale -- the brms equivalent of rstanarm's
# QR=TRUE. This is what keeps a full-768-feature fit from stalling NUTS on max
# tree-depth, so we can avoid PCA entirely (n_pc=0). Harmless when n_pc>0.
form <- bf(sprintf("k | trials(n) ~ %s", paste(feat, collapse = " + ")), decomp = "QR")
df_tr <- data.frame(k = k, n = n, Xtr, check.names = FALSE)
cat(sprintf("Fitting beta-binomial model: N=%d clusters, D=%d features, prior=%s\n",
            N, D, prior_nm))

fit <- brm(
  formula = form,
  data    = df_tr,
  family  = beta_binomial(),            # logit link on mu; one overdispersion phi
  prior   = priors,
  chains  = n_chains,
  iter    = n_iter,
  cores   = n_chains,
  seed    = 1234,
  backend = "rstan",
  # init = 0 (all params at 0 on the unconstrained scale -> mu = 0.5 everywhere)
  # is essential for beta_binomial: brms parameterises it as Beta(mu*phi,(1-mu)*phi),
  # and the default random inits (init_r = 2) over many coefficients saturate the
  # linear predictor so mu hits 0/1, making a Beta shape parameter exactly 0. Stan
  # then rejects EVERY initial value and the chain never starts ("does not contain
  # samples"). Starting at mu = 0.5 keeps both shape parameters strictly positive.
  init    = 0,
  refresh = max(1L, n_iter %/% 20L),
  control = list(adapt_delta = 0.95, max_treedepth = 12)
)
saveRDS(fit, file.path(out_dir, paste0("fit_betabinom_", prefix, ".rds")))

phi_summary <- posterior_summary(fit, variable = "phi")
cat(sprintf("Overdispersion phi: mean=%.3g [%.3g, %.3g] (larger phi -> closer to binomial)\n",
            phi_summary[1, "Estimate"], phi_summary[1, "Q2.5"], phi_summary[1, "Q97.5"]))

# Mean probability per test cluster (the systematic embedding signal, no trials
# needed): posterior_linpred with the inverse link applied.
df_ts <- data.frame(Xts, n = as.integer(Yts[[n_col]]), check.names = FALSE)
muprob <- posterior_linpred(fit, newdata = df_ts, transform = TRUE)  # draws x N_test, [0,1]
Yts$pred_prevalence <- apply(muprob, 2, median)
Yts$mean_lower      <- apply(muprob, 2, quantile, 0.025)
Yts$mean_upper      <- apply(muprob, 2, quantile, 0.975)
obs_prev <- Yts[[k_col]] / Yts[[n_col]]
Yts$obs_prevalence <- obs_prev

# Posterior PREDICTIVE interval for the OBSERVED prevalence k/n. Unlike the binomial
# model's manual rbinom fold, the beta-binomial draws here ALREADY carry the extra-
# binomial variance (phi), so this is the interval that should cover k/n ~95%.
n_test <- as.integer(Yts[[n_col]])
ppc   <- posterior_predict(fit, newdata = df_ts)            # draws x N_test, COUNTS
ppred <- sweep(ppc, 2, n_test, "/")                         # -> prevalence draws
Yts$pred_lower <- apply(ppred, 2, quantile, 0.025)
Yts$pred_upper <- apply(ppred, 2, quantile, 0.975)

write_csv(Yts, file.path(out_dir, paste0("Y_test_pred_", prefix, ".csv")))

mae        <- mean(abs(Yts$pred_prevalence - obs_prev))
cover_mean <- mean(obs_prev >= Yts$mean_lower & obs_prev <= Yts$mean_upper)
cover_pred <- mean(obs_prev >= Yts$pred_lower & obs_prev <= Yts$pred_upper)
cat(sprintf("Test prevalence MAE: %.4f\n", mae))
cat(sprintf("95%% coverage of k/n: mean-prob CI %.3f (expected low) | posterior-predictive %.3f (target 0.95)\n",
            cover_mean, cover_pred))
cat(sprintf("Wrote predictions and fit to %s\n", out_dir))
