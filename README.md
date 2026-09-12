# A LSTM‑LightGBM Cascade Framework for District‑Level Respiratory Disease Outpatient Visits Prediction

> Reproducible code for manuscript:
> 基于LSTM‑LightGBM级联框架的城市区县级呼吸系统疾病就诊预测研究
>
> This repository contains the official implementation for district‑level environmental‑health risk early‑warning in Beijing (2016‑2018).

Environment & Dependencies
> R ≥ **4.2.0**
> Python backend for TensorFlow/Keras is required for deep‑learning part.

Install required packages inside R console:
required_packages <- c(
  "dplyr","tidyr","zoo","lubridate","ggplot2","gridExtra",
  "caret","lightgbm","keras","tensorflow","mgcv","forecast","moments"
)
install.packages(required_packages, dependencies = TRUE)


## 📌 Abstract Summary
Air‑pollution‑related respiratory disease burden exhibits strong within‑city spatial heterogeneity across urban districts.
We propose a **two‑stage cascade model (Bi‑LSTM + LightGBM residual correction)** for district‑daily outpatient visit prediction:
1. Bidirectional‑LSTM captures long‑term seasonal & temporal patterns of environment‑health time‑series.
2. LightGBM fits residuals from LSTM; **sample‑weighted asymmetric strategy** penalizes under‑prediction for high‑risk days.
3. District‑wise quantile‑based risk stratification (75% / 90% / 97.5% percentiles from training data of each district).
4. Expanding‑window 5‑fold time‑series cross‑validation (120‑day held‑out test per fold).
5. Benchmarks: ETS, global‑GAM, district‑wise stratified‑GAM, stand‑alone two‑layer LSTM.
6. Model interpretability: LightGBM Gain importance, SHAP, Partial‑Dependence‑Plot(PDP) for pollutant lag‑response curves.

## Pipeline Workflow

1. Data cleaning & gap‑filling (training‑set‑only median imputation)
2. Time‑series lag‑feature construction grouped by community ID (`mcode`)
3. Grid search for optimal lag window (minimize test‑set RMSE)
4. Cascade model: BiLSTM main prediction → LightGBM residual calibration
5. Baseline model training (ETS, GAM) for comparison
6. Global absolute risk threshold classification (4‑level risk warning)
7. Regression metrics (R², RMSE, MAE) & classification metrics (Sensitivity, Precision, F1, Kappa)
8. Visualization & file export

## Hyper‑parameters (Optimized by cross‑validation)

- LSTM: units=64, dropout=0.2, batch=64
- LightGBM: lr=0.1, max_depth=6, nrounds=80
- Asymmetric loss penalty factor α = 4
- Risk quantile cut‑points: 0.75 / 0.90 / 0.975

## Citation

If you use this code for academic research, please cite the original paper.

## License

MIT License
