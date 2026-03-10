# install required packages if not already installed
if (!require("tidyverse")) install.packages("tidyverse", version ='2.0.0')
if (!require("ggplot2")) install.packages("ggplot2", version ='3.5.1')
if (!require("here")) install.packages("here", version = '1.0.1')
# For cmdstanr visit https://mc-stan.org/cmdstanr/articles/cmdstanr.html and 
# install cmdstanr version:0.8.1, CmdStan version: 2.34.1

library(ggplot2)
library(tidyverse) 
library(here)
library(cmdstanr) 

# Change here 
data_path = 'data/multispectral/' 
output_path = 'output/'
model_name = 'Hierarchical_Shrinkage' # another option'Reguralized_Horseshoe'

# Read data
Xtr = read.csv(paste0(data_path,'X_train.csv'))
Ytr = read.csv(paste0(data_path,'Y_train.csv'))
Xts = read.csv(paste0(data_path,'X_test.csv'))
Yts = read.csv(paste0(data_path,'Y_test.csv'))


Ytr = read_csv("~/Downloads/train_df.csv")
nrow(Xtr)
Xtr = read_csv("~/Downloads/train_features.csv")
nrow(Ytr)

sum(is.na(Ytr$deprived_sev))

library(rstanarm)
options(mc.cores = parallel::detectCores())

n = rpois(length(Ytr$deprived_sev),10)
k = as.integer(Ytr$deprived_sev * n)

fit <- stan_glm(
  cbind(k, n - k) ~ .,
  data = Xtr,
  prior = normal(0, .1),
  family = binomial(link = "logit"),
  chains = 4,
  iter = 200
)
saveRDS(fit,"/tmp/test.rds")
a=predict(Xtr[sample(nrow(Xtr)),])

# Set-up stan
stanfile = here(paste0(model_name,'.stan'))
mod = cmdstan_model(stanfile)

data_list = list(
  N = nrow(Xtr),
  B = ncol(Xtr),
  Y = Ytr$target,
  X = Xtr,
  nu = 3
  # For reguralized_horseshoe
  # nu_local = 1,
  # nu_global = 1,
  # scale_icept = 5,
  # scale_global = p0/((ncol(Xtr)-p0)*sqrt(nrow(Xtr))), 
  ## where p0 =  floor(ncol(Xtr)*0.05)  #prior belief of how many non-zero coef. e.g. say 5% 
  # slab_scale = 1,
  # slab_df = 1
)

fit =  mod$sample(
  data = data_list,
  seed = 1234,
  iter_warmup = 500,
  iter_sampling = 1000,
  save_warmup = TRUE,
  chains = 4,
  parallel_chains = 4,
  refresh = 50
)

df_post_w = fit$draws(format = "df", variables=c('w'),  inc_warmup = F)[,1:ncol(Xtr)]
post_sigma = fit$draws(format = "df", variables=c('sigma'),  inc_warmup = F)$sigma
post_w0 = fit$draws(format = "df", variables=c('w0'),  inc_warmup = F)$w0

## Posteior predictive for test data
Xw_tes = as.matrix(Xts)%*%t(as.matrix(df_post_w)) 
y_new_post = matrix(NA,nrow(Xts),nrow(df_post_w))
for (i in 1:nrow(df_post_w)){
  mu_vec = post_w0[i] + c(Xw_tes[,i])
  v = post_sigma[i]
  y_new_post[,i] = rnorm(nrow(Xts),mu_vec,v)
}
Yts$post_mean <-rowMeans(y_new_post) |> unname()
Yts$post_sd <- sqrt(apply(y_new_post, 1, var))
write.csv(Yts, paste0(output_path,"Y_test_result_",model_name,".csv"))

y_new_truncated = y_new_post
y_new_truncated[y_new_post > 1] <- 1 #pi/2
y_new_truncated[y_new_post < 0] <- 0
write.csv(y_new_truncated, paste0(output_path,"post_pred_sample_truncated_",model_name,".csv"))

## Posteior predictive for training data
Xw_tr = as.matrix(Xtr)%*%t(as.matrix(df_post_w)) 
y_tr_post = matrix(NA,nrow(Xtr),nrow(df_post_w))
for (i in 1:nrow(df_post_w)){
  mu_vec = post_w0[i] + c(Xw_tr[,i])
  v = post_sigma[i]
  y_tr_post[,i] = rnorm(nrow(Xtr),mu_vec,v)
}
Ytr$post_mean <-rowMeans(y_tr_post) |> unname()
Ytr$post_sd <- sqrt(apply(y_tr_post, 1, var))
write.csv(Ytr, paste0(output_path,"Y_trainig_result_",model_name,".csv"))

y_tr_truncated = y_tr_post
y_tr_truncated[y_tr_post > 1] <- 1 #pi/2
y_tr_truncated[y_tr_post < 0] <- 0
write.csv(y_tr_truncated, paste0(output_path,"post_trainig_sample_truncated_",model_name,".csv"))

print(mean(abs(Yts$post_mean-Yts$target)))
print(mean(abs(Ytr$post_mean-Ytr$target)))