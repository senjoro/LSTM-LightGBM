Repository Structure
├── env_health_warning.R   # Core modeling script
├── data1030.txt           # Raw tab‑separated input dataset (user‑provided)
├── output/                # Auto‑generated outputs
│   ├── *.png              # Performance figures
│   ├── *.csv              # prediction & evaluation table
│   ├── *.h5 / *.txt       # saved model weights
│   └── result.RData       # full workspace object for reproduction
└── README.md

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
