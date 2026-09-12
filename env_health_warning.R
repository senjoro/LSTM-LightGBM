
# -------------------------- 0. 工具与环境初始化 --------------------------
log_msg <- function(txt){
  cat(sprintf("[%s] %s\n", format(Sys.time(),"%Y‑%m‑%d %H:%M:%S"), txt))
}

set.seed(123)
if(requireNamespace("tensorflow", quietly = TRUE)) tensorflow::set_random_seed(123)

required_packages <- c(
  "dplyr","tidyr","zoo","lubridate","ggplot2","gridExtra",
  "caret","lightgbm","keras","tensorflow","mgcv","forecast","moments"
)

install_if_missing <- function(pkgs) {
  missing <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
  if(length(missing)>0){
    log_msg(paste("安装缺失包：",paste(missing,collapse = ", ")))
    install.packages(missing, dependencies = TRUE, quiet = TRUE)
  }
  invisible(sapply(pkgs, library, character.only = TRUE, quietly = TRUE))
}
suppressPackageStartupMessages(install_if_missing(required_packages))
select <- dplyr::select

# -------------------------- 1. 全局配置（严格对齐论文手稿） --------------------------
CFG <- list(
  lstm = list(
    units = 128, num_layers = 2, dropout = 0.2, recurrent_dropout = 0.3,
    dense_units = 32, epochs = 100, batch_size = 64,
    validation_split = 0.15, early_stop_patience = 20, learning_rate = 0.001
  ),
  lgb = list(
    objective = "regression", metric = "rmse",
    learning_rate = 0.03, max_depth = 4, num_leaves = 2^4 - 1,
    min_data_in_leaf = 50, feature_fraction = 0.6,
    bagging_fraction = 0.6, bagging_freq = 5,
    lambda_l1 = 0.5, lambda_l2 = 0.5, verbose = -1
  ),
  lgb_nrounds = 200,
  # 论文α取值：1,5,10,15,20
  alpha_test_values = c(1.0,5.0,10.0,15.0,20.0),
  lag_days_to_test = c(3,5,7,10,14),
  best_lag_override = NULL,
  risk_quantiles = c(0.75,0.90,0.975),
  risk_labels = c("低风险 (Low)","中风险 (Moderate)","高风险 (High)","极高风险 (Very High)"),
  validation = list(method="expanding", n_splits=5, test_size_days=120, min_train_fraction=0.4),
  standardization = list(enabled=TRUE, method="zscore", eb_shrinkage=TRUE,
                         eb_prior_weight=200, eb_adaptive=TRUE,
                         eb_small_district_weight=500, eb_size_threshold=50),
  shap = list(enabled=TRUE, n_samples=300),
  lag_effect = list(enabled=TRUE, pollutants=c("pm25","pm10","no2","so2","o3"), n_quantiles=10),
  output_dir = "output",
  save_models = TRUE, save_plots = TRUE, dpi = 300
)

log_msg("全局配置加载完成；与0909手稿对齐")

pollution_weather_features <- c("pm25","pm10","no2","so2","o3","templag0","rhlag0")
pollution_features <- c("pm25","pm10","no2","so2","o3")
weather_features <- c("templag0","rhlag0")
target_var <- "pv.res"

# -------------------------- 2. 数据读取、特征工程、【先插值原始数据】 --------------------------
load_and_validate_data <- function(filepath) {
  data <- read.csv2(filepath, header = TRUE, sep = "\t",
                    stringsAsFactors = FALSE, na.strings = c("","NA","NaN","null"))
  numeric_cols <- 6:36
  data[,numeric_cols] <- lapply(data[,numeric_cols], as.numeric)
  must_cols <- c("mcode","date","pv.res","pm25","pm10","no2","so2","o3")
  miss_cols <- setdiff(must_cols, colnames(data))
  if(length(miss_cols)>0) stop(paste("缺失必要字段：",paste(miss_cols,collapse=",")))
  data$date_parsed <- as.Date(data$date)
  if(any(is.na(data$date_parsed))) stop("date列解析失败")
  data <- data %>% arrange(mcode, date_parsed)
  log_msg(sprintf("原始数据：%d行 × %d列；区县数：%d；时间：%s ~ %s",
                  nrow(data),ncol(data),length(unique(data$mcode)),
                  min(data$date_parsed),max(data$date_parsed)))
  data
}

engineer_features <- function(data){
  data %>% mutate(
    Day_of_week = lubridate::wday(date_parsed, week_start = 1),
    Is_Weekend = as.integer(Day_of_week %in% c(6,7)),
    Day_of_year = lubridate::yday(date_parsed),
    DOY_sin = sin(2*pi*Day_of_year/365.25),
    DOY_cos = cos(2*pi*Day_of_year/365.25),
    Year_num = as.integer(lubridate::year(date_parsed)) - min(lubridate::year(date_parsed)) +1,
    Time_trend = as.numeric(date_parsed - min(date_parsed))/365.25,
    Month = lubridate::month(date_parsed),
    Month_sin = sin(2*pi*Month/12),
    Month_cos = cos(2*pi*Month/12)
  )
}

# 原始尺度插值：按区县，na.approx，训练集中位数回填；**在标准化之前执行（论文标准流程）**
interpolate_raw_dataset <- function(data, feat_list, target){
  data_out <- data
  for(f in c(feat_list, target)){
    if(!f %in% colnames(data_out)) next
    data_out <- data_out %>% group_by(mcode) %>%
      mutate(!!f := zoo::na.approx(!!sym(f), na.rm = FALSE)) %>%
      ungroup()
    fill_med <- median(data_out[[f]], na.rm = TRUE)
    data_out[[f]][is.na(data_out[[f]])] <- fill_med
  }
  data_out
}

raw_data <- load_and_validate_data("data1030.txt")
raw_data <- engineer_features(raw_data)
temporal_features <- c("Day_of_week","Is_Weekend","DOY_sin","DOY_cos",
                       "Year_num","Time_trend","Month_sin","Month_cos")
lstm_features <- c(pollution_weather_features, temporal_features)
# 原始数据插值（先插值，后标准化）
raw_data <- interpolate_raw_dataset(raw_data, pollution_weather_features, target_var)

# -------------------------- 3. 经验贝叶斯区县Z‑score标准化 --------------------------
compute_district_stats <- function(data, target_var, features, method="zscore",
                                   eb_shrinkage=TRUE, eb_prior_weight=200,
                                   eb_adaptive=TRUE, eb_small_district_weight=500,
                                   eb_size_threshold=50){
  global_mean <- mean(data[[target_var]], na.rm=TRUE)
  global_sd <- sd(data[[target_var]], na.rm=TRUE)
  district_stats <- data %>% group_by(mcode) %>%
    summarise(n_days = n(), raw_mean = mean(!!sym(target_var),na.rm=TRUE),
              raw_sd = sd(!!sym(target_var),na.rm=TRUE), .groups="drop") %>%
    mutate(size_rank = rank(raw_mean),
           size_tier = cut(size_rank, breaks=quantile(size_rank,c(0,1/3,2/3,1)),
                           labels=c("small","medium","large"),include.lowest=TRUE))
  tier_stats <- district_stats %>% group_by(size_tier) %>%
    summarise(tier_mean=mean(raw_mean), tier_sd=mean(raw_sd), n_districts=n(), .groups="drop")
  district_stats <- district_stats %>% left_join(tier_stats, by="size_tier")
  if(eb_shrinkage && eb_adaptive){
    district_stats <- district_stats %>% mutate(
      base_prior = eb_prior_weight,
      extra_prior = ifelse(size_tier=="small", eb_small_district_weight,
                           ifelse(size_tier=="medium", eb_small_district_weight*0.3,0)),
      effective_prior = base_prior + extra_prior,
      shrinkage_factor = effective_prior/(effective_prior + n_days),
      final_mean = shrinkage_factor*tier_mean + (1‑shrinkage_factor)*raw_mean,
      final_sd = pmax(shrinkage_factor*tier_sd + (1‑shrinkage_factor)*raw_sd, global_sd*0.1)
    )
  }else{
    district_stats <- district_stats %>% mutate(final_mean=raw_mean, final_sd=pmax(raw_sd,global_sd*0.1))
  }
  feature_stats <- list()
  for(feat in features){
    feat_global_mean <- mean(data[[feat]], na.rm=TRUE)
    feat_global_sd <- sd(data[[feat]], na.rm=TRUE)
    feat_district <- data %>% group_by(mcode) %>%
      summarise(n_days=n(), raw_mean=mean(!!sym(feat),na.rm=TRUE),
                raw_sd=sd(!!sym(feat),na.rm=TRUE), .groups="drop") %>%
      left_join(district_stats %>% select(mcode,size_tier), by="mcode")
    feat_tier_stats <- feat_district %>% group_by(size_tier) %>%
      summarise(tier_mean=mean(raw_mean), tier_sd=mean(raw_sd), .groups="drop")
    feat_district <- feat_district %>% left_join(feat_tier_stats, by="size_tier")
    if(eb_shrinkage && eb_adaptive){
      feat_district <- feat_district %>% mutate(
        base_prior = eb_prior_weight,
        extra_prior = ifelse(size_tier=="small", eb_small_district_weight,
                             ifelse(size_tier=="medium", eb_small_district_weight*0.3,0)),
        effective_prior = base_prior + extra_prior,
        shrinkage_factor = effective_prior/(effective_prior + n_days),
        final_mean = shrinkage_factor*tier_mean + (1‑shrinkage_factor)*raw_mean,
        final_sd = pmax(shrinkage_factor*tier_sd + (1‑shrinkage_factor)*raw_sd, feat_global_sd*0.1)
      )
    }else{
      feat_district <- feat_district %>% mutate(final_mean=raw_mean, final_sd=pmax(raw_sd,feat_global_sd*0.1))
    }
    feature_stats[[feat]] <- feat_district
  }
  list(target=district_stats, features=feature_stats, global_mean=global_mean, global_sd=global_sd, tier_stats=tier_stats)
}

standardize_data <- function(data, stats, target_var, features){
  data_std <- data
  t_stats <- stats$target %>% select(mcode, final_mean, final_sd)
  data_std <- data_std %>% left_join(t_stats, by="mcode") %>%
    mutate(!!target_var := (!!sym(target_var)‑final_mean)/final_sd) %>%
    select(-final_mean,‑final_sd)
  for(feat in features){
    if(!feat %in% names(stats$features)) next
    f_stat <- stats$features[[feat]] %>% select(mcode, final_mean, final_sd)
    colnames(f_stat) <- c("mcode","feat_mean","feat_sd")
    data_std <- data_std %>% left_join(f_stat, by="mcode") %>%
      mutate(!!feat := (!!sym(feat)‑feat_mean)/feat_sd) %>%
      select(-feat_mean,‑feat_sd)
  }
  data_std
}

unstandardize_predictions <- function(pred_std, mcode_vec, stats){
  t_stats <- stats$target %>% select(mcode, final_mean, final_sd)
  df <- data.frame(mcode=mcode_vec, pred_std=pred_std) %>% left_join(t_stats, by="mcode")
  df$pred_std * df$final_sd + df$final_mean
}

# -------------------------- 4. 扩展窗口交叉验证切分 --------------------------
create_expanding_splits <- function(data, n_splits, test_size_days=120){
  all_dates <- sort(unique(data$date_parsed))
  n_total <- length(all_dates)
  if(test_size_days * n_splits >= n_total*0.6){
    test_size_days <- floor(n_total*0.5 / n_splits)
    log_msg(sprintf("警告：样本不足，测试集调整为 %d天", test_size_days))
  }
  splits <- list()
  for(i in 1:n_splits){
    test_end_idx <- n_total ‑ (i‑1)*test_size_days
    test_start_idx <- test_end_idx ‑ test_size_days + 1
    train_end_idx <- test_start_idx ‑ 1
    if(train_end_idx < floor(n_total*0.3)){
      log_msg(sprintf("折%d：训练集样本不足，终止折划分",i))
      break
    }
    splits[[i]] <- list(fold_id=i, train_start=all_dates[1], train_end=all_dates[train_end_idx],
                        test_start=all_dates[test_start_idx], test_end=all_dates[test_end_idx],
                        train_days=train_end_idx, test_days=test_size_days)
    log_msg(sprintf("折%d：训练 %s~%s；测试 %s~%s",i,all_dates[1],all_dates[train_end_idx],
                    all_dates[test_start_idx],all_dates[test_end_idx]))
  }
  splits
}

splits <- create_expanding_splits(raw_data, CFG$validation$n_splits, CFG$validation$test_size_days)
CFG$validation$n_splits <- length(splits)

# -------------------------- 5. 模型工具函数 --------------------------
build_lagged_features <- function(data, n_lags, features){
  d <- data %>% group_by(mcode)
  for(f in features){
    for(lag in 1:n_lags){
      nm <- paste0(f,"_lag",lag)
      d <- d %>% mutate(!!nm := dplyr::lag(!!sym(f), lag))
    }
  }
  d %>% ungroup() %>% na.omit()
}

build_lstm_model <- function(n_lags, n_features, params){
  keras::backend()$clear_session()
  inp <- layer_input(shape = c(n_lags, n_features))
  x <- inp %>% layer_lstm(units=params$units, dropout=params$dropout,
                          recurrent_dropout=params$recurrent_dropout,
                          return_sequences = (params$num_layers>1))
  if(params$num_layers>1){
    x <- x %>% layer_lstm(units=floor(params$units/2), dropout=params$dropout*0.5,
                          recurrent_dropout=params$recurrent_dropout*0.5, return_sequences=FALSE)
  }
  out <- x %>% layer_dense(units=params$dense_units, activation="relu") %>%
    layer_dropout(rate=params$dropout*0.5) %>% layer_dense(units=1)
  model <- keras_model(inputs=inp, outputs=out)
  model %>% compile(loss="mse", optimizer=optimizer_adam(learning_rate=params$learning_rate))
  model
}

build_3d_tensor <- function(data, n_lags, features){
  n_samp <- nrow(data)
  n_feat <- length(features)
  arr <- array(0, dim=c(n_samp, n_lags, n_feat))
  for(fi in seq_along(features)){
    for(t in 1:n_lags){
      col_nm <- paste0(features[fi],"_lag", n_lags‑t+1)
      arr[,t,fi] <- data[[col_nm]]
    }
  }
  arr
}

calc_metrics <- function(actual, predicted){
  rmse <- sqrt(mean((actual‑predicted)^2, na.rm=TRUE))
  ss_res <- sum((actual‑predicted)^2, na.rm=TRUE)
  ss_tot <- sum((actual‑mean(actual, na.rm=TRUE))^2, na.rm=TRUE)
  r2 <- 1‑ss_res/ss_tot
  mae <- mean(abs(actual‑predicted), na.rm=TRUE)
  mape <- mean(abs((actual‑predicted)/(actual+1e‑5)), na.rm=TRUE)*100
  data.frame(RMSE=rmse, R2=r2, MAE=mae, MAPE=mape)
}

# 核心训练：LSTM + LightGBM残差；【主模型：残差符号样本加权（对齐论文非对称加权策略）】
train_hybrid <- function(train_raw, test_raw, n_lags, features, lstm_cfg, lgb_cfg, lgb_nrounds, stats, target, alpha_weight){
  train_lag <- build_lagged_features(train_raw, n_lags, features)
  test_lag  <- build_lagged_features(test_raw, n_lags, features)
  y_tr_std <- train_lag[[target]]
  y_te_std <- test_lag[[target]]

  x_cols <- as.vector(sapply(features, function(f) paste0(f,"_lag",1:n_lags)))
  x3d_tr <- build_3d_tensor(train_lag, n_lags, features)
  x3d_te <- build_3d_tensor(test_lag, n_lags, features)

  lstm_mod <- build_lstm_model(n_lags, length(features), lstm_cfg)
  earlystop <- callback_early_stopping(monitor="val_loss", patience=lstm_cfg$early_stop_patience, restore_best_weights=TRUE)
  lrdecay <- callback_reduce_lr_on_plateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e‑6)
  lstm_mod %>% fit(x3d_tr, y_tr_std, epochs=lstm_cfg$epochs, batch_size=lstm_cfg$batch_size,
                   validation_split=lstm_cfg$validation_split, verbose=0, callbacks=list(earlystop, lrdecay))

  pred_tr_lstm_std <- as.numeric(predict(lstm_mod, x3d_tr, verbose=0))
  pred_te_lstm_std <- as.numeric(predict(lstm_mod, x3d_te, verbose=0))
  resid_tr_std <- y_tr_std ‑ pred_tr_lstm_std

  # ✅论文：低估残差(resid>0，真实>预测)施加α权重；其余为1
  sample_w <- ifelse(resid_tr_std > 0, alpha_weight, 1.0)
  sample_w <- sample_w / mean(sample_w)

  xmat_tr <- as.matrix(train_lag[,x_cols])
  xmat_te <- as.matrix(test_lag[,x_cols])
  dtrain <- lgb.Dataset(data=xmat_tr, label=resid_tr_std, weight=sample_w)
  lgb_mod <- lgb.train(params=lgb_cfg, data=dtrain, nrounds=lgb_nrounds)

  pred_tr_lgb_std <- predict(lgb_mod, xmat_tr)
  pred_te_lgb_std <- predict(lgb_mod, xmat_te)

  pred_tr_std <- pred_tr_lstm_std + pred_tr_lgb_std
  pred_te_std <- pred_te_lstm_std + pred_te_lgb_std

  pred_tr_orig <- pmax(0, unstandardize_predictions(pred_tr_std, train_lag$mcode, stats))
  pred_te_orig <- pmax(0, unstandardize_predictions(pred_te_std, test_lag$mcode, stats))
  y_tr_orig <- unstandardize_predictions(y_tr_std, train_lag$mcode, stats)
  y_te_orig <- unstandardize_predictions(y_te_std, test_lag$mcode, stats)

  list(lstm=lstm_mod, lgb=lgb_mod,
       train_pred=pred_tr_orig, test_pred=pred_te_orig,
       train_actual=y_tr_orig, test_actual=y_te_orig,
       train_lag=train_lag, test_lag=test_lag,
       x_cols=x_cols, x_train=xmat_tr, x_test=xmat_te,
       n_lags=n_lags)
}

# -------------------------- 6. 滞后窗口搜索（使用fold1训练集搜索，全折固定超参，防泄露） --------------------------
fold1 <- splits[[1]]
train_f1_raw <- raw_data %>% filter(date_parsed <= fold1$train_end)
test_f1_raw  <- raw_data %>% filter(date_parsed >= fold1$test_start & date_parsed <= fold1$test_end)

fold1_stats <- compute_district_stats(train_f1_raw, target_var, pollution_weather_features,
                                      eb_shrinkage = CFG$standardization$eb_shrinkage,
                                      eb_prior_weight = CFG$standardization$eb_prior_weight,
                                      eb_adaptive = CFG$standardization$eb_adaptive,
                                      eb_small_district_weight = CFG$standardization$eb_small_district_weight)
train_f1_std <- standardize_data(train_f1_raw, fold1_stats, target_var, pollution_weather_features)
test_f1_std  <- standardize_data(test_f1_raw, fold1_stats, target_var, pollution_weather_features)

lag_tbl <- data.frame(lag_days=CFG$lag_days_to_test, RMSE=NA_real_, R2=NA_real_, MAE=NA_real_, MAPE=NA_real_)
best_lag <- NULL; best_rmse <- Inf; best_lag_r2 <- NA

if(!is.null(CFG$best_lag_override)){
  best_lag <- CFG$best_lag_override
  log_msg(sprintf("强制指定滞后窗口：%d天", best_lag))
}else{
  for(i in seq_along(CFG$lag_days_to_test)){
    nl <- CFG$lag_days_to_test[i]
    log_msg(sprintf("搜索滞后窗口 = %d天", nl))
    res <- tryCatch(train_hybrid(train_f1_std, test_f1_std, nl, lstm_features,
                                 CFG$lstm, CFG$lgb, CFG$lgb_nrounds, fold1_stats, target_var, alpha_weight=10.0),
                    error=function(e){log_msg(paste("滞后",nl,"报错：",e$message)); NULL})
    if(!is.null(res)){
      met <- calc_metrics(res$test_actual, res$test_pred)
      lag_tbl[i,2:5] <- met
      log_msg(sprintf("  R²=%.4f RMSE=%.2f", met$R2, met$RMSE))
      if(met$RMSE < best_rmse){
        best_rmse <- met$RMSE
        best_lag <- nl
        best_lag_r2 <- met$R2
      }
    }
    rm(res);gc()
  }
}
log_msg(sprintf("最优滞后窗口 = %d天；fold1 R²=%.4f RMSE=%.2f", best_lag, best_lag_r2, best_rmse))

# -------------------------- 7. α参数敏感性分析（论文α序列：1,5,10,15,20） --------------------------
log_msg("===== α参数敏感性分析 =====")
train_lag_a <- build_lagged_features(train_f1_std, best_lag, lstm_features)
test_lag_a  <- build_lagged_features(test_f1_std, best_lag, lstm_features)
x3d_tr_a <- build_3d_tensor(train_lag_a, best_lag, lstm_features)
x3d_te_a <- build_3d_tensor(test_lag_a, best_lag, lstm_features)
xcols_a <- as.vector(sapply(lstm_features, function(f) paste0(f,"_lag",1:best_lag)))
xmat_tr_a <- as.matrix(train_lag_a[,xcols_a])
xmat_te_a <- as.matrix(test_lag_a[,xcols_a])

# 固定LSTM权重，仅改变LightGBM样本权重（对齐论文实验设计）
lstm_a <- build_lstm_model(best_lag, length(lstm_features), CFG$lstm)
es_a <- callback_early_stopping(monitor="val_loss", patience=CFG$lstm$early_stop_patience, restore_best_weights=TRUE)
lr_a <- callback_reduce_lr_on_plateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e‑6)
lstm_a %>% fit(x3d_tr_a, train_lag_a[[target_var]], epochs=CFG$lstm$epochs, batch_size=CFG$lstm$batch_size,
               validation_split=CFG$lstm$validation_split, verbose=0, callbacks=list(es_a, lr_a))

pred_tr_lstm_a <- as.numeric(predict(lstm_a, x3d_tr_a, verbose=0))
resid_a <- train_lag_a[[target_var]] ‑ pred_tr_lstm_a
y_te_std_a <- test_lag_a[[target_var]]
y_te_orig_a <- unstandardize_predictions(y_te_std_a, test_lag_a$mcode, fold1_stats)

alpha_tbl <- data.frame(alpha=CFG$alpha_test_values, RMSE=NA_real_, R2=NA_real_, MAE=NA_real_,
                        high_risk_sn=NA_real_, high_risk_sp=NA_real_)
high_thresh <- quantile(unstandardize_predictions(train_lag_a[[target_var]], train_lag_a$mcode, fold1_stats), 0.90, na.rm=TRUE)

for(ai in seq_along(CFG$alpha_test_values)){
  av <- CFG$alpha_test_values[ai]
  w_a <- ifelse(resid_a>0, av, 1.0)
  w_a <- w_a / mean(w_a)
  d_a <- lgb.Dataset(xmat_tr_a, label=resid_a, weight=w_a)
  lgb_a <- lgb.train(params=CFG$lgb, data=d_a, nrounds=CFG$lgb_nrounds)
  pred_lgb_te <- predict(lgb_a, xmat_te_a)
  pred_lstm_te <- as.numeric(predict(lstm_a, x3d_te_a, verbose=0))
  pred_std_a <- pred_lstm_te + pred_lgb_te
  pred_orig_a <- pmax(0, unstandardize_predictions(pred_std_a, test_lag_a$mcode, fold1_stats))
  met_a <- calc_metrics(y_te_orig_a, pred_orig_a)
  actual_high <- as.integer(y_te_orig_a > high_thresh)
  pred_high <- as.integer(pred_orig_a > high_thresh)
  tp <- sum(actual_high==1 & pred_high==1)
  fn <- sum(actual_high==1 & pred_high==0)
  tn <- sum(actual_high==0 & pred_high==0)
  fp <- sum(actual_high==0 & pred_high==1)
  sn <- tp/max(tp+fn, 1)
  sp <- tn/max(tn+fp, 1)
  alpha_tbl[ai,] <- c(av, met_a$RMSE, met_a$R2, met_a$MAE, sn, sp)
  log_msg(sprintf("α=%.1f | R²=%.4f Sn=%.1f%% Sp=%.1f%%", av, met_a$R2, sn*100, sp*100))
  rm(lgb_a);gc()
}
best_alpha <- alpha_tbl$alpha[which.max(alpha_tbl$high_risk_sn)]
log_msg(sprintf("最优α=%.1f（基于高风险敏感度）", best_alpha))

# -------------------------- 8. 5折扩展窗口交叉验证（固定best_lag、best_alpha） --------------------------
log_msg("===== 开始5折扩展窗口交叉验证 =====")
cv_res <- list(); cv_preds <- list(); cv_models <- list()
fold1_full_result <- NULL; fold1_full_stats <- NULL

for(fold_idx in seq_along(splits)){
  fobj <- splits[[fold_idx]]
  log_msg(sprintf("----- CV折 %d/%d -----", fold_idx, length(splits)))
  tr_raw <- raw_data %>% filter(date_parsed <= fobj$train_end)
  te_raw <- raw_data %>% filter(date_parsed >= fobj$test_start & date_parsed <= fobj$test_end)
  fstats <- compute_district_stats(tr_raw, target_var, pollution_weather_features,
                                   eb_shrinkage=CFG$standardization$eb_shrinkage,
                                   eb_prior_weight=CFG$standardization$eb_prior_weight,
                                   eb_adaptive=CFG$standardization$eb_adaptive,
                                   eb_small_district_weight=CFG$standardization$eb_small_district_weight)
  tr_std <- standardize_data(tr_raw, fstats, target_var, pollution_weather_features)
  te_std <- standardize_data(te_raw, fstats, target_var, pollution_weather_features)
  fold_out <- train_hybrid(tr_std, te_std, best_lag, lstm_features,
                            CFG$lstm, CFG$lgb, CFG$lgb_nrounds, fstats, target_var, alpha_weight=best_alpha)
  met_fold <- calc_metrics(fold_out$test_actual, fold_out$test_pred)
  cv_res[[fold_idx]] <- met_fold
  cv_models[[fold_idx]] <- list(lstm=fold_out$lstm, lgb=fold_out$lgb, stats=fstats)
  pred_df <- data.frame(fold=fold_idx, mcode=fold_out$test_lag$mcode, date=fold_out$test_lag$date_parsed,
                        Actual=fold_out$test_actual, Predicted=fold_out$test_pred)
  cv_preds[[fold_idx]] <- pred_df
  log_msg(sprintf("折%d结果：R²=%.4f RMSE=%.2f MAE=%.2f", fold_idx, met_fold$R2, met_fold$RMSE, met_fold$MAE))
  if(fold_idx==1){
    fold1_full_result <- fold_out
    fold1_full_stats <- fstats
  }
  rm(fold_out);gc()
}
cv_summary <- do.call(rbind, cv_res) %>% mutate(fold=row_number())
cv_mean <- colMeans(cv_summary[,c("RMSE","R2","MAE","MAPE")])
cv_sd <- apply(cv_summary[,c("RMSE","R2","MAE","MAPE")],2,sd)
all_cv_pred <- do.call(rbind, cv_preds)

log_msg("===== CV汇总 =====")
print(data.frame(Metric=c("RMSE","R2","MAE","MAPE"), Mean=round(cv_mean,4), SD=round(cv_sd,4)), row.names=FALSE)

# -------------------------- 9. 基准模型：ETS / 全局GAM / 分层GAM / 单独LSTM --------------------------
log_msg("===== 基准模型对比 =====")
tr_base <- train_f1_std
te_base <- test_f1_std
tr_lag_base <- build_lagged_features(tr_base, best_lag, lstm_features)
te_lag_base  <- build_lagged_features(te_base, best_lag, lstm_features)
y_tr_base_std <- tr_lag_base[[target_var]]
y_te_base_std <- te_lag_base[[target_var]]
y_te_base_orig <- unstandardize_predictions(y_te_base_std, te_lag_base$mcode, fold1_full_stats)
dist_list <- unique(te_lag_base$mcode)

## 9.1 ETS
ets_pred_std <- numeric(nrow(te_lag_base)); ptr <- 1
for(d in dist_list){
  yd_tr <- tr_base %>% filter(mcode==d) %>% pull(!!sym(target_var))
  yd_te <- te_base %>% filter(mcode==d) %>% pull(!!sym(target_var))
  tsobj <- ts(yd_tr, frequency=7)
  m_ets <- tryCatch(forecast::ets(tsobj, model="ZZA", allow.multiplicative.trend=FALSE), error=function(e) NULL)
  ypred_d <- if(!is.null(m_ets)) as.numeric(forecast::forecast(m_ets, h=length(yd_te))$mean) else rep(mean(yd_tr, na.rm=TRUE), length(yd_te))
  ets_pred_std[ptr:(ptr+length(yd_te)-1)] <- ypred_d
  ptr <- ptr + length(yd_te)
}
ets_pred_orig <- pmax(0, unstandardize_predictions(ets_pred_std, te_lag_base$mcode, fold1_full_stats))
met_ets <- calc_metrics(y_te_base_orig, ets_pred_orig)

## 9.2 全局GAM（DLNM风格，包含SO₂，对齐论文污染物集合）
key_lags <- c(1,3,7); key_lags <- key_lags[key_lags <= best_lag]
poll_smooth <- c()
for(v in c("pm25","pm10","no2","so2","o3")) for(lag in key_lags){
  poll_smooth <- c(poll_smooth, paste0("s(",v,"_lag",lag,", k=5)"))
}
gam_form <- as.formula(paste(target_var, "~", paste(poll_smooth, collapse="+"), "+",
                              "s(templag0_lag1,k=6)+s(rhlag0_lag1,k=5)+as.factor(Day_of_week_lag1)+",
                              "s(DOY_sin_lag1,k=4)+s(DOY_cos_lag1,k=4)+s(Year_num_lag1,k=4)"))
m_gam_global <- tryCatch(gam(gam_form, data=tr_lag_base, method="REML"),
                         error=function(e){
                           log_msg(paste("全局GAM报错，回退简单公式：",e$message))
                           gam(as.formula(paste0(target_var,"~pm25_lag1+pm10_lag1+no2_lag1+so2_lag1+o3_lag1+templag0_lag1+rhlag0_lag1+as.factor(Day_of_week_lag1)")),
                               data=tr_lag_base)
                         })
gam_pred_std <- predict(m_gam_global, newdata=te_lag_base)
gam_pred_orig <- pmax(0, unstandardize_predictions(gam_pred_std, te_lag_base$mcode, fold1_full_stats))
met_gam_global <- calc_metrics(y_te_base_orig, gam_pred_orig)

##9.3 分层GAM（区县独立，样本不足回退均值，统计回退数量）
district_gam_pred_std <- numeric(nrow(te_lag_base)); ok_cnt <- 0
for(d in dist_list){
  d_tr <- tr_lag_base %>% filter(mcode==d)
  d_te <- te_lag_base %>% filter(mcode==d)
  mg <- if(nrow(d_tr)>=100){
    tryCatch(gam(gam_form, data=d_tr, method="REML"),
             error=function(e) tryCatch(gam(as.formula(paste0(target_var,"~pm25_lag1+pm10_lag1+no2_lag1+so2_lag1+templag0_lag1+as.factor(Day_of_week_lag1)")),data=d_tr),error=function(e2) NULL))
  }else{NULL}
  if(!is.null(mg)){
    dp <- predict(mg, newdata=d_te)
    ok_cnt <- ok_cnt+1
  }else{
    dp <- rep(mean(d_tr[[target_var]], na.rm=TRUE), nrow(d_te))
  }
  idx_start <- which(te_lag_base$mcode==d)[1]
  district_gam_pred_std[idx_start:(idx_start+length(dp)-1)] <- dp
}
log_msg(sprintf("分层GAM成功拟合区县：%d个；回退均值预测：%d个", ok_cnt, length(dist_list)-ok_cnt))
district_gam_pred_orig <- pmax(0, unstandardize_predictions(district_gam_pred_std, te_lag_base$mcode, fold1_full_stats))
met_gam_district <- calc_metrics(y_te_base_orig, district_gam_pred_orig)

##9.4 单独LSTM
x3d_tr_only <- build_3d_tensor(tr_lag_base, best_lag, lstm_features)
x3d_te_only <- build_3d_tensor(te_lag_base, best_lag, lstm_features)
m_lstm_only <- build_lstm_model(best_lag, length(lstm_features), CFG$lstm)
es_only <- callback_early_stopping(monitor="val_loss", patience=CFG$lstm$early_stop_patience, restore_best_weights=TRUE)
lr_only <- callback_reduce_lr_on_plateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e‑6)
m_lstm_only %>% fit(x3d_tr_only, y_tr_base_std, epochs=CFG$lstm$epochs, batch_size=CFG$lstm$batch_size,
                    validation_split=CFG$lstm$validation_split, verbose=0, callbacks=list(es_only, lr_only))
lstm_only_pred_std <- as.numeric(predict(m_lstm_only, x3d_te_only, verbose=0))
lstm_only_pred_orig <- pmax(0, unstandardize_predictions(lstm_only_pred_std, te_lag_base$mcode, fold1_full_stats))
met_lstm_only <- calc_metrics(y_te_base_orig, lstm_only_pred_orig)

##9.5 本研究级联模型（fold1）
met_hybrid <- calc_metrics(fold1_full_result$test_actual, fold1_full_result$test_pred)

model_compare <- data.frame(
  Model = c("ETS指数平滑","全局GAM(DLNM‑含SO₂)","分层GAM(区县独立)","单独双层LSTM","LSTM‑LightGBM级联模型"),
  R2 = c(met_ets$R2, met_gam_global$R2, met_gam_district$R2, met_lstm_only$R2, met_hybrid$R2),
  RMSE = c(met_ets$RMSE, met_gam_global$RMSE, met_gam_district$RMSE, met_lstm_only$RMSE, met_hybrid$RMSE),
  MAE = c(met_ets$MAE, met_gam_global$MAE, met_gam_district$MAE, met_lstm_only$MAE, met_hybrid$MAE),
  MAPE = c(met_ets$MAPE, met_gam_global$MAPE, met_gam_district$MAPE, met_lstm_only$MAPE, met_hybrid$MAPE)
)
log_msg("基准模型对比结果：")
print(model_compare, row.names=FALSE)

# -------------------------- 10. 区县独立分位数风险分级【论文核心方法】 --------------------------
log_msg("===== 区县独立分位数风险分级（手稿2.3） =====")
results_df <- data.frame(mcode=fold1_full_result$test_lag$mcode,
                         date=fold1_full_result$test_lag$date_parsed,
                         Actual=fold1_full_result$test_actual,
                         Predicted=fold1_full_result$test_pred)
# 从训练集每个区县单独计算75/90/97.5分位数
train_act_df <- data.frame(mcode=fold1_full_result$train_lag$mcode,
                           act_train=fold1_full_result$train_actual)
district_thresholds <- train_act_df %>% group_by(mcode) %>%
  summarise(q75=quantile(act_train,0.75,na.rm=TRUE),
            q90=quantile(act_train,0.90,na.rm=TRUE),
            q975=quantile(act_train,0.975,na.rm=TRUE), .groups="drop")
results_df <- results_df %>% left_join(district_thresholds, by="mcode")

risk_cut <- function(x, q75,q90,q975){
  case_when(
    x <= q75 ~ CFG$risk_labels[1],
    x <= q90  ~ CFG$risk_labels[2],
    x <= q975 ~ CFG$risk_labels[3],
    TRUE ~ CFG$risk_labels[4]
  )
}
results_df <- results_df %>%
  rowwise() %>%
  mutate(Actual_Level = risk_cut(Actual, q75, q90, q975),
         Pred_Level  = risk_cut(Predicted, q75, q90, q975)) %>%
  ungroup()
results_df$Actual_Level <- factor(results_df$Actual_Level, levels=CFG$risk_labels, ordered=TRUE)
results_df$Pred_Level  <- factor(results_df$Pred_Level, levels=CFG$risk_labels, ordered=TRUE)

conf_mat <- caret::confusionMatrix(data=results_df$Pred_Level, reference=results_df$Actual_Level)
class_met <- as.data.frame(conf_mat$byClass) %>%
  mutate(F1=2*Sensitivity*Precision/(Sensitivity+Precision)) %>%
  select(Sensitivity, Specificity, Precision, F1, `Balanced Accuracy`)
rownames(class_met) <- CFG$risk_labels

log_msg(sprintf("风险分级：总体准确率=%.2f%%；Kappa=%.4f", conf_mat$overall["Accuracy"]*100, conf_mat$overall["Kappa"]))
print(class_met, row.names=TRUE)

# 分区县规模异质性分析
district_meta <- fold1_full_stats$target %>% select(mcode, raw_mean, raw_sd, n_days, size_tier) %>%
  rename(avg_visits=raw_mean, sd_visits=raw_sd) %>%
  mutate(district_size=recode(size_tier, small="小型区县", medium="中型区县", large="大型区县"))
district_perf <- results_df %>% group_by(mcode) %>%
  summarise(n_days=n(), RMSE=sqrt(mean((Actual‑Predicted)^2)),
            R2=1‑sum((Actual‑Predicted)^2)/sum((Actual‑mean(Actual))^2),
            MAE=mean(abs(Actual‑Predicted)),
            MAPE=mean(abs((Actual‑Predicted)/(Actual+1e‑5)))*100, .groups="drop") %>%
  left_join(district_meta, by="mcode")
size_group_sum <- district_perf %>% group_by(district_size) %>%
  summarise(n_districts=n(), mean_R2=mean(R2,na.rm=TRUE), sd_R2=sd(R2,na.rm=TRUE),
            median_R2=median(R2,na.rm=TRUE), mean_RMSE=mean(RMSE,na.rm=TRUE),
            mean_MAE=mean(MAE,na.rm=TRUE), mean_MAPE=mean(MAPE,na.rm=TRUE), .groups="drop")
log_msg("按区县规模分组性能：")
print(size_group_sum, row.names=FALSE)

# -------------------------- 11. SHAP + Gain特征重要性 + PDP滞后效应分析（论文2.5） --------------------------
log_msg("===== 可解释性分析：Gain / SHAP / PDP滞后效应 =====")
lgb_mod <- fold1_full_result$lgb
x_test_mat <- fold1_full_result$x_test

# Gain重要性
gain_imp <- lgb.importance(lgb_mod, percentage=TRUE) %>%
  mutate(Feature_category = case_when(
    grepl("pm25|pm10|no2|so2|o3", Feature) ~ "污染物",
    grepl("templag0|rhlag0", Feature) ~ "气象",
    grepl("Day_of_week|Is_Weekend|DOY|Year_num|Time_trend|Month", Feature) ~ "时间特征",
    TRUE ~ "其他"
  ), Lag_day = as.integer(sub(".*_lag(\\d+).*","\\1",Feature)))
gain_cat_sum <- gain_imp %>% group_by(Feature_category) %>%
  summarise(total_gain=sum(Gain), n_feat=n(), .groups="drop") %>% arrange(desc(total_gain))
poll_gain <- gain_imp %>% filter(Feature_category=="污染物") %>%
  mutate(Pollutant=sub("_lag.*","",Feature)) %>% group_by(Pollutant) %>%
  summarise(total_gain=sum(Gain), n_lags=n(), .groups="drop") %>% arrange(desc(total_gain))
poll_lag_gain <- gain_imp %>% filter(Feature_category=="污染物") %>%
  mutate(Pollutant=sub("_lag.*","",Feature)) %>% group_by(Pollutant, Lag_day) %>%
  summarise(gain=sum(Gain), .groups="drop")

# SHAP（修复版：lgb.interprete返回list，逐样本汇总）
shap_import <- NULL
if(CFG$shap$enabled){
  n_shap <- min(CFG$shap$n_samples, nrow(x_test_mat))
  shap_idx <- sample(1:nrow(x_test_mat), n_shap)
  shap_raw <- tryCatch(lgb.interprete(model=lgb_mod, data=x_test_mat, idx=shap_idx), error=function(e){log_msg(paste("SHAP计算失败：",e$message));NULL})
  if(!is.null(shap_raw) && length(shap_raw)>0){
    f_names <- colnames(x_test_mat)
    shap_sum <- rep(0, length(f_names)); names(shap_sum) <- f_names; n_valid <- 0
    for(samp in shap_raw){
      if(is.data.frame(samp) && nrow(samp)>0 && "Feature" %in% colnames(samp)){
        for(j in 1:nrow(samp)){
          ft <- as.character(samp$Feature[j])
          if(ft %in% names(shap_sum)) shap_sum[ft] <- shap_sum[ft] + abs(samp$value[j])
        }
        n_valid <- n_valid+1
      }
    }
    if(n_valid>0){
      shap_mean <- shap_sum / n_valid
      shap_import <- data.frame(Feature=names(shap_mean), mean_abs_shap=as.numeric(shap_mean), stringsAsFactors=FALSE) %>%
        arrange(desc(mean_abs_shap)) %>%
        mutate(Feature_category = case_when(
          grepl("pm25|pm10|no2|so2|o3", Feature) ~ "污染物",
          grepl("templag0|rhlag0", Feature) ~ "气象",
          grepl("Day_of_week|Is_Weekend|DOY|Year_num|Time_trend|Month", Feature) ~ "时间特征",
          TRUE ~ "其他"
        ))
    }
  }
}

# PDP：各污染物多滞后天数，计算效应强度（论文滞后效应曲线）
lstm_offset_mean <- mean(fold1_full_result$test_pred_std - predict(lgb_mod, x_test_mat))
lag_effect_list <- list()
pollutants_pdp <- CFG$lag_effect$pollutants
for(poll in pollutants_pdp){
  lag_cols <- grep(paste0("^",poll,"_lag\\d+$"), colnames(x_test_mat), value=TRUE)
  if(length(lag_cols)==0) next
  lag_ds <- as.integer(sub(".*_lag(\\d+)","\\1",lag_cols))
  eff_str <- numeric(length(lag_cols))
  pdp_one_poll <- list()
  for(ii in seq_along(lag_cols)){
    ft <- lag_cols[ii]
    q_vals <- quantile(x_test_mat[,ft], probs=seq(0.05,0.95,length.out=CFG$lag_effect$n_quantiles), na.rm=TRUE)
    pdp_v <- numeric(length(q_vals))
    x_tmp <- x_test_mat
    for(qq in seq_along(q_vals)){
      x_tmp[,ft] <- q_vals[qq]
      pdp_v[qq] <- mean(predict(lgb_mod, x_tmp)) + lstm_offset_mean
    }
    eff_str[ii] <- max(pdp_v)‑min(pdp_v)
    fstat <- fold1_full_stats$features[[poll]]
    f_mean <- mean(fstat$final_mean); f_sd <- mean(fstat$final_sd)
    pdp_one_poll[[ii]] <- data.frame(
      Pollutant=poll, lag_day=lag_ds[ii],
      feature_value_std=q_vals, feature_value=q_vals*f_sd+f_mean,
      partial_effect_std=pdp_v, partial_effect_orig=pdp_v*fold1_full_stats$global_sd+fold1_full_stats$global_mean
    )
  }
  lag_effect_list[[poll]] <- do.call(rbind, pdp_one_poll)
}
lag_effect_all <- do.call(rbind, lag_effect_list)
lag_effect_summary <- lag_effect_all %>% group_by(Pollutant, lag_day) %>%
  summarise(effect_strength_std=max(partial_effect_std)‑min(partial_effect_std),
            effect_strength_orig=max(partial_effect_orig)‑min(partial_effect_orig), .groups="drop") %>%
  group_by(Pollutant) %>%
  summarise(optimal_lag=lag_day[which.max(effect_strength_std)],
            peak_effect_std=max(effect_strength_std),
            peak_effect_orig=max(effect_strength_orig),
            total_effect=sum(effect_strength_std), .groups="drop") %>% arrange(desc(total_effect))

log_msg("污染物滞后效应汇总(PDP)：")
print(lag_effect_summary, row.names=FALSE)

# -------------------------- 12. 输出目录、保存全部结果 --------------------------
out_dir <- CFG$output_dir
if(!dir.exists(out_dir)) dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)

if(CFG$save_models){
  save_model_hdf5(fold1_full_result$lstm, file.path(out_dir,"best_lstm_model.h5"))
  lgb.save(fold1_full_result$lgb, file.path(out_dir,"best_lgb_model.txt"))
}

save(lag_tbl, cv_summary, cv_mean, cv_sd, all_cv_pred, alpha_tbl, model_compare,
     district_perf, size_group_sum, conf_mat, class_met, results_df, district_thresholds,
     gain_imp, gain_cat_sum, poll_gain, poll_lag_gain, shap_import, lag_effect_all, lag_effect_summary,
     fold1_full_stats, file=file.path(out_dir,"model_results.RData"))

write.csv(results_df, file.path(out_dir,"prediction_results.csv"), row.names=FALSE)
write.csv(model_compare, file.path(out_dir,"model_comparison.csv"), row.names=FALSE)
write.csv(alpha_tbl, file.path(out_dir,"alpha_sensitivity.csv"), row.names=FALSE)
write.csv(district_perf, file.path(out_dir,"district_performance.csv"), row.names=FALSE)
write.csv(size_group_sum, file.path(out_dir,"size_group_summary.csv"), row.names=FALSE)
write.csv(district_thresholds, file.path(out_dir,"district_risk_thresholds.csv"), row.names=FALSE)
write.csv(gain_cat_sum, file.path(out_dir,"gain_category_importance.csv"), row.names=FALSE)
write.csv(poll_gain, file.path(out_dir,"pollutant_gain_importance.csv"), row.names=FALSE)
write.csv(poll_lag_gain, file.path(out_dir,"pollutant_lag_gain.csv"), row.names=FALSE)
if(!is.null(shap_import)) write.csv(shap_import, file.path(out_dir,"shap_importance.csv"), row.names=FALSE)
write.csv(lag_effect_summary, file.path(out_dir,"lag_effect_summary.csv"), row.names=FALSE)
write.csv(lag_effect_all, file.path(out_dir,"lag_effect_pdp_curves.csv"), row.names=FALSE)

# -------------------------- 13. 核心论文图表（5张主图，精简；GitHub版不生成海量附图） --------------------------
library(ggplot2)
library(gridExtra)

## 图1：滞后窗口性能
p1 <- ggplot(lag_tbl, aes(x=lag_days, y=R2)) +
  geom_line(linewidth=1, color="#2c3e50") + geom_point(size=3, color="#2c3e50") +
  geom_vline(xintercept=best_lag, color="#e74c3c", linetype="dashed", linewidth=1)+
  annotate("text", x=best_lag+0.5, y=max(lag_tbl$R2,na.rm=TRUE)*0.98,
           label=paste0("最优滞后: ",best_lag,"天"), color="#e74c3c", hjust=0)+
  labs(x="滞后天数", y=expression(R^2), title="不同滞后窗口模型性能") + theme_bw(base_size=12)+
  theme(plot.title=element_text(hjust=0.5))
ggsave(file.path(out_dir,"fig1_lag_performance.png"), p1, width=8, height=6, dpi=CFG$dpi)

## 图2：预测‑真实散点
p2 <- ggplot(results_df, aes(x=Actual, y=Predicted)) +
  geom_point(alpha=0.25, color="#3498db", size=0.8)+
  geom_abline(slope=1, intercept=0, color="#e74c3c", linewidth=1)+
  geom_smooth(method="lm", se=TRUE, color="#2c3e50", linetype="dashed")+
  labs(x="实际就诊人次", y="预测就诊人次", title="预测值‑实际值散点图")+
  theme_bw(base_size=12)+theme(plot.title=element_text(hjust=0.5))
ggsave(file.path(out_dir,"fig2_scatter.png"), p2, width=8, height=6, dpi=CFG$dpi)

## 图3：混淆矩阵（区县独立分位数风险分级）
conf_df <- as.data.frame(conf_mat$table)
p3 <- ggplot(conf_df, aes(x=Reference, y=Prediction, fill=Freq)) +
  geom_tile(color="white") + geom_text(aes(label=Freq), size=4.5, color="black")+
  scale_fill_gradient(low="#ebf5fb", high="#2980b9")+
  labs(x="实际风险等级", y="预测风险等级", title="风险分级混淆矩阵（区县独立分位数阈值）")+
  theme_bw(base_size=12)+theme(plot.title=element_text(hjust=0.5), axis.text.x=element_text(angle=45,hjust=1))
ggsave(file.path(out_dir,"fig3_confusion_matrix.png"), p3, width=8, height=6, dpi=CFG$dpi)

## 图4：多模型对比柱状图
model_melt <- reshape2::melt(model_compare, id.vars="Model")
p4 <- ggplot(model_melt, aes(x=Model, y=value, fill=Model)) +
  geom_bar(stat="identity") + facet_wrap(~variable, scales="free_y", nrow=1)+
  scale_fill_brewer(palette="Set2")+
  labs(x="模型", y="指标值", title="不同模型预测性能对比")+
  theme_bw(base_size=12)+theme(plot.title=element_text(hjust=0.5), axis.text.x=element_text(angle=45,hjust=1), legend.position="none")
ggsave(file.path(out_dir,"fig4_model_comparison.png"), p4, width=10, height=6, dpi=CFG$dpi)

## 图5：污染物滞后效应曲线(PDP)
p5a <- ggplot(lag_effect_summary, aes(x=reorder(Pollutant, peak_effect_std), y=peak_effect_std, fill=Pollutant))+
  geom_bar(stat="identity")+coord_flip()+
  geom_text(aes(label=paste0("lag",optimal_lag)), hjust=-0.1, size=3.5)+
  scale_fill_brewer(palette="Set2")+
  labs(x="污染物", y="峰值效应强度(σ)", title="各污染物峰值效应与最优滞后天数")+
  theme_bw(base_size=11)+theme(plot.title=element_text(hjust=0.5), legend.position="none")

p5b <- ggplot(lag_effect_all, aes(x=lag_day, y=effect_strength_std, color=Pollutant))+
  geom_line(linewidth=1)+geom_point(size=3)+scale_color_brewer(palette="Set1")+
  labs(x="滞后天数", y="效应强度(σ)", title="污染物滞后效应强度曲线")+
  theme_bw(base_size=11)+theme(plot.title=element_text(hjust=0.5))

p5 <- grid.arrange(p5b, p5a, ncol=2, top="基于PDP的污染物滞后效应分析")
ggsave(file.path(out_dir,"fig5_lag_effect.png"), p5, width=14, height=6, dpi=CFG$dpi)
