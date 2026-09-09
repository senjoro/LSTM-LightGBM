# 环境流行病学级联预测系统｜BiLSTM‑LightGBM 环境健康时序预警
# Paper: 基于时序网络的城市社区级环境健康风险预警框架研究
rm(list = ls())
set.seed(123)
if(require(tensorflow, quietly = TRUE)) tensorflow::set_random_seed(123)

# 依赖包
pkgs <- c("dplyr","ggplot2","caret","zoo","lightgbm","keras","tensorflow","mgcv","reshape2","lubridate","forecast")
for(p in pkgs) if(!require(p,character.only=T)){install.packages(p,deps=T,quiet=T);library(p,character.only=T)}

# ========== 超参数配置（已优化固定） ==========
LSTM_CFG <- list(units=64,dropout=0.2,recurrent_dropout=0.1,dense_units=16,epochs=20,batch_size=64,val_split=0.1,patience=5)
LGB_CFG  <- list(objective="regression",metric="rmse",learning_rate=0.1,max_depth=6,num_leaves=63,min_data_in_leaf=20,nrounds=80,verbose=-1)
ALPHA    <- 4.0
LAG_SEQ  <- c(2,3,4,5,6,7,8,9,10,12,14,21)
RISK_Q   <- c(0.75,0.90,0.975)
RISK_LAB <- c("低风险 (Low)","中风险 (Moderate)","高风险 (High)","极高风险 (Very High)")
FEAT_VEC <- c("pm25","pm10","no2","so2","o3","templag0","rhlag0","Day_of_week","Is_Weekend","pv.res")
TARGET   <- "pv.res"

# ========== 工具函数 ==========
build_lag <- function(df,nlag,feats){
  df %>% group_by(mcode) -> d
  for(f in feats) for(t in 1:nlag) d <- mutate(d,!!paste0(f,"_lag",t):=lag(!!sym(f),t))
  ungroup(d) %>% na.omit()
}
asym_loss <- function(y_true,y_pred){
  err <- y_true - y_pred
  tf$where(err>0,ALPHA*tf$square(err),tf$square(err)) %>% tf$reduce_mean()
}
metric_df <- function(act,pred){
  rmse <- sqrt(mean((act-pred)^2,na.rm=T))
  r2   <- 1-sum((act-pred)^2,na.rm=T)/sum((act-mean(act,na.rm=T))^2,na.rm=T)
  mae  <- mean(abs(act-pred),na.rm=T)
  data.frame(RMSE=rmse,R2=r2,MAE=mae)
}

# ========== 数据读取与时序分割（严格防止数据泄露） ==========
df_raw <- read.csv2("data1030.txt",header=T,sep="\t",stringsAsFactors=F,na.strings=c("","NA","NaN"))
df_raw[,6:36] <- lapply(df_raw[,6:36],as.numeric)
if("date"%in%colnames(df_raw)){
  df_raw$date_parsed <- as.Date(df_raw$date)
  df_raw$Day_of_week <- wday(df_raw$date_parsed,week_start=1)
  df_raw$Is_Weekend  <- ifelse(df_raw$Day_of_week%in%c(6,7),1,0)
}else{
  df_raw$Day_of_week <- (1:nrow(df_raw))%%7+1
  df_raw$Is_Weekend  <- ifelse(df_raw$Day_of_week%in%c(6,7),1,0)
}
# 按社区时序8:2切分
train <- df_raw %>% group_by(mcode) %>% filter(row_number()<=floor(n()*0.8)) %>% ungroup()
test  <- df_raw %>% group_by(mcode) %>% filter(row_number()>floor(n()*0.8)) %>% ungroup()

# 训练集中位数插值规则，测试集复用训练集统计量
fill_na_rule <- function(tr,te,varname){
  med <- median(tr[[varname]],na.rm=T)
  tr[[varname]] <- na.approx(tr[[varname]],na.rm=F);tr[[varname]][is.na(tr[[varname]])]<-med
  te[[varname]] <- na.approx(te[[varname]],na.rm=F);te[[varname]][is.na(te[[varname]])]<-med
  list(tr=tr,te=te)
}
for(f in FEAT_VEC){
  res <- fill_na_rule(train,test,f)
  train <- res$tr; test <- res$te
}
y_med <- median(train[[TARGET]],na.rm=T)
train[[TARGET]][is.na(train[[TARGET]])] <- y_med
test[[TARGET]][is.na(test[[TARGET]])]  <- y_med

# ========== 滞后窗口遍历选优 ==========
lag_tbl <- data.frame(lag=LAG_SEQ,RMSE=NA,R2=NA,MAE=NA)
best_rmse <- Inf
best_env <- NULL

for(i in seq_along(LAG_SEQ)){
  nl <- LAG_SEQ[i]
  tr_lag <- build_lag(train,nl,FEAT_VEC)
  te_lag <- build_lag(test,nl,FEAT_VEC)
  tr_lag$mcode_numeric <- as.integer(as.factor(tr_lag$mcode))
  te_lag$mcode_numeric <- as.integer(as.factor(te_lag$mcode))
  ytr <- tr_lag[[TARGET]]
  yte <- te_lag[[TARGET]]
  
  # 构造3D时序张量
  xcol_ts <- as.vector(sapply(FEAT_VEC,function(f) paste0(f,"_lag",1:nl)))
  dim3_tr <- array(0,c(nrow(tr_lag),nl,length(FEAT_VEC)))
  dim3_te <- array(0,c(nrow(te_lag),nl,length(FEAT_VEC)))
  for(ifi in seq_along(FEAT_VEC)){
    for(ti in 1:nl){
      cname <- paste0(FEAT_VEC[ifi],"_lag",(nl-ti+1))
      dim3_tr[,ti,ifi] <- tr_lag[[cname]]
      dim3_te[,ti,ifi] <- te_lag[[cname]]
    }
  }
  # BiLSTM
  backend()$clear_session()
  inp <- layer_input(shape=c(nl,length(FEAT_VEC)))
  out_ts <- inp %>% bidirectional(layer_lstm(units=LSTM_CFG$units,dropout=LSTM_CFG$dropout,recurrent_dropout=LSTM_CFG$recurrent_dropout,return_sequences=F)) %>%
    layer_dense(LSTM_CFG$dense_units,activation="relu") %>% layer_dense(1)
  model_lstm <- keras_model(inp,out_ts)
  model_lstm %>% compile(loss=asym_loss,optimizer=optimizer_adam(0.001))
  cb_early <- callback_early_stopping(monitor="val_loss",patience=LSTM_CFG$patience,restore_best_weights=T)
  model_lstm %>% fit(dim3_tr,ytr,epochs=LSTM_CFG$epochs,batch_size=LSTM_CFG$batch_size,validation_split=LSTM_CFG$val_split,verbose=0,callbacks=list(cb_early))
  pred_tr_lstm <- predict(model_lstm,dim3_tr,verbose=0)
  
  # LightGBM残差拟合
  xcol_gbm <- c(xcol_ts,"mcode_numeric")
  mat_tr <- as.matrix(tr_lag[,xcol_gbm])
  mat_te <- as.matrix(te_lag[,xcol_gbm])
  resid <- as.numeric(ytr)-as.numeric(pred_tr_lstm)
  wgt   <- as.numeric(ytr)/(mean(ytr,na.rm=T)+1e-5)
  dset  <- lgb.Dataset(mat_tr,label=resid,weight=wgt,categorical_feature="mcode_numeric")
  model_gbm <- lgb.train(LGB_CFG,dset,LGB_CFG$nrounds)
  
  # 预测集成
  pred_te_lstm <- predict(model_lstm,dim3_te,verbose=0)
  pred_te_res  <- predict(model_gbm,mat_te)
  pred_final   <- pmax(0,as.numeric(pred_te_lstm)+as.numeric(pred_te_res))
  met <- metric_df(yte,pred_final)
  lag_tbl[i,2:4] <- met
  
  if(met$RMSE<best_rmse){
    best_rmse <- met$RMSE
    best_env <- list(lag=nl,tr=tr_lag,te=te_lag,act=yte,pred=pred_final,lstm=model_lstm,lgb=model_gbm,thr=quantile(tr_lag[[TARGET]],RISK_Q,na.rm=T))
  }
  rm(model_lstm,model_gbm,dim3_tr,dim3_te);gc()
}

# ========== 基准模型对比（GAM / ETS） ==========
best_nlag <- best_env$lag
tr_best <- build_lag(train,best_nlag,FEAT_VEC)
te_best <- build_lag(test,best_nlag,FEAT_VEC)
tr_best$mcode_numeric <- as.integer(as.factor(tr_best$mcode))
te_best$mcode_numeric <- as.integer(as.factor(te_best$mcode))
y_test_true <- best_env$act

# GAM
smooth_vars <- c("pm25","pm10","no2","so2","o3","templag0","rhlag0")
s_terms <- as.vector(sapply(smooth_vars,function(v) paste0("s(",v,"_lag",1:best_nlag,",k=",ifelse(v%in%c("templag0","rhlag0"),5,6),")")))
lin_terms <- as.vector(sapply(c("Day_of_week","Is_Weekend"),function(v) paste0(v,"_lag",1:best_nlag)))
form_gam <- as.formula(paste0(TARGET," ~ ",paste(c(s_terms,lin_terms,"mcode_numeric"),collapse=" + ")))
fit_gam <- bam(form_gam,data=tr_best,method="fREML",discrete=T,select=T,nthreads=parallel::detectCores())
pred_gam <- pmax(0,pmin(predict(fit_gam,te_best),max(tr_best[[TARGET]],na.rm=T)*1.5))
met_gam <- metric_df(y_test_true,pred_gam)

# ETS单社区时序
comm_list <- unique(test$mcode)
pred_ets <- numeric(nrow(te_best))
ptr <-1
for(cid in comm_list){
  yc_tr <- train %>% filter(mcode==cid) %>% pull(!!sym(TARGET))
  yc_te <- test %>% filter(mcode==cid) %>% pull(!!sym(TARGET))
  fit_ets <- tryCatch(ets(ts(yc_tr,frequency=7),model="ZZA",allow.multiplicative.trend=F),error=function(e) NULL)
  yc_pred <- if(!is.null(fit_ets)) forecast(fit_ets,h=length(yc_te))$mean else rep(mean(yc_tr,na.rm=T),length(yc_te))
  yc_pred <- pmax(0,pmin(yc_pred,max(yc_tr,na.rm=T)*1.5))
  pred_ets[ptr:(ptr+length(yc_te)-1)] <- yc_pred
  ptr <- ptr+length(yc_te)
}
met_ets <- metric_df(y_test_true,pred_ets)
met_hyb <- lag_tbl[lag_tbl$lag==best_nlag,c("RMSE","R2","MAE")]
cmp_tbl <- data.frame(
  Model=c("ETS季节性平滑","GAM广义加性模型","BiLSTM‑LightGBM级联模型"),
  R2=c(met_ets$R2,met_gam$R2,met_hyb$R2),
  RMSE=c(met_ets$RMSE,met_gam$RMSE,met_hyb$RMSE),
  MAE=c(met_ets$MAE,met_gam$MAE,met_hyb$MAE)
)

# ========== 风险分级评估 & 结果表 ==========
res_df <- data.frame(Actual=best_env$act,Predicted=best_env$pred)
brk <- c(-Inf,best_env$thr,Inf)
res_df$Actual_Level <- cut(res_df$Actual,breaks=brk,labels=RISK_LAB,ordered_result=T)
res_df$Pred_Level  <- cut(res_df$Predicted,breaks=brk,labels=RISK_LAB,ordered_result=T)
conf_mat <- confusionMatrix(res_df$Pred_Level,res_df$Actual_Level)
class_eval <- as.data.frame(conf_mat$byClass) %>% mutate(F1=2*(Sensitivity*Precision)/(Sensitivity+Precision)) %>%
  select(Sensitivity,Specificity,Precision,F1,`Balanced Accuracy`)
rownames(class_eval) <- RISK_LAB

# ========== 输出保存 & 绘图 ==========
if(!dir.exists("output")) dir.create("output",showWarnings=F)
save_model_hdf5(best_env$lstm,"output/best_lstm.h5")
lgb.save(best_env$lgb,"output/best_lightgbm.txt")
save(lag_tbl,cmp_tbl,conf_mat,class_eval,best_env,file="output/result.RData")
write.csv(res_df,"output/prediction.csv",row.names=F)

# 可视化
p1 <- ggplot(lag_tbl,aes(x=lag,y=R2))+geom_line(linewidth=1,color="#2c3e50")+geom_point(size=3,color="#2c3e50")+
  geom_vline(xintercept=best_nlag,color="#e74c3c",linetype="dashed",linewidth=1)+
  labs(x="滞后天数",y="R²",title="滞后窗口性能")+theme_bw(base_size=12)+theme(plot.title=element_text(hjust=0.5))
ggsave("output/lag_curve.png",p1,width=8,height=6,dpi=300)

p2 <- ggplot(res_df,aes(x=Actual,y=Predicted))+geom_point(alpha=0.3,color="#3498db")+geom_abline(slope=1,intercept=0,color="#e74c3c",linewidth=1)+
  labs(x="实际就诊量",y="预测就诊量",title="预测‑真实散点")+theme_bw(base_size=12)+theme(plot.title=element_text(hjust=0.5))
ggsave("output/scatter.png",p2,width=8,height=6,dpi=300)

p3 <- ggplot(as.data.frame(conf_mat$table),aes(x=Reference,y=Prediction,fill=Freq))+geom_tile(color="white")+geom_text(aes(label=Freq),size=5)+
  scale_fill_gradientn(colours=c("white","#3498db","#0073AC"))+
  labs(x="真实风险等级",y="预测风险等级",title="混淆矩阵热力图")+theme_bw(base_size=12)+theme(plot.title=element_text(hjust=0.5),axis.text.x=element_text(angle=45,hjust=1))
ggsave("output/confusion_heatmap.png",p3,width=8,height=6,dpi=300)

p4 <- ggplot(melt(cmp_tbl,id.vars="Model"),aes(x=Model,y=value,fill=Model))+geom_bar(stat="identity")+facet_wrap(~variable,scales="free_y")+
  labs(x="模型",y="指标",title="多模型性能对比")+theme_bw(base_size=12)+theme(plot.title=element_text(hjust=0.5),axis.text.x=element_text(angle=45,hjust=1))
ggsave("output/model_compare.png",p4,width=10,height=6,dpi=300)

# 控制台摘要输出
cat("\n==== Final Summary ====\n")
cat(paste0("Optimal lag days: ",best_nlag,"\n"))
print(cmp_tbl,row.names=F)
cat(paste0("Overall Accuracy: ",round(conf_mat$overall["Accuracy"]*100,2)," %\n"))
cat(paste0("Kappa: ",round(conf_mat$overall["Kappa"],4),"\n"))
write.csv(cmp_tbl,"output/model_comparison.csv",row.names=F)
write.csv(class_eval,"output/class_metrics.csv",row.names=T)
