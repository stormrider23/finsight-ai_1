# Financial AI Portfolio 🧠🛡️
### FinSight AI + FraudShield AI — Complete Financial AI System

**Author:** Suman Das — Senior Applied Scientist, Financial AI
**Stack:** Python · PyTorch · XGBoost · Reinforcement Learning · Streamlit · FastAPI
**Domain:** 11 yrs Banking (PNB) + MTech IAR (Jadavpur University, CGPA 9.79)

## 🚀 Live Demo
👉 **https://finsight-ai-9qwq8uhnwckfp2vxx9ch4b.streamlit.app/**

---

## Project 1 — FinSight AI 📈
### Financial Time Series Forecasting + RL Trading Agent

End-to-end Financial AI system combining deep learning forecasting
with reinforcement learning portfolio optimization.

### FinSight Progress
- [x] Day 1 — Data pipeline, EDA, stationarity tests, ARIMA baseline
- [x] Day 2 — LSTM forecaster, walk-forward validation, scaling pipeline
- [x] Day 3 — Temporal Fusion Transformer + attention heatmap
- [x] Day 4 — RL Environment, Gymnasium TradingEnv, random agent baseline
- [x] Day 5 — PPO V1+V2+V3 trained, BalancedTradingEnv, bear market analysis
- [x] Day 6 — Streamlit dashboard live, 3 tabs, permanent public URL
- [x] Day 7 — Upwork + Fiverr profile launched, LinkedIn published

### Assets Covered
NIFTY50 · Reliance · TCS · Bitcoin (2019-2024)

### Forecasting Results — Reliance 2024
| Model          | MAE      | RMSE     | Directional Acc |
|----------------|----------|----------|-----------------|
| ARIMA baseline | 0.011502 | 0.013421 | ~50%            |
| LSTM V2        | 0.015139 | 0.018931 | 49.7%           |
| TFT            | 0.012382 | 0.018418 | 53.8%           |

### RL Trading Results — Reliance 2024 (Bear Market)
| Agent          | Total Return | Sharpe | Max Drawdown | Behaviour          |
|----------------|-------------|--------|--------------|--------------------|
| PPO V1         | 0.00%       | 0.000  | 0.00%        | Cash preservation  |
| PPO V2         | 0.00%       | 0.000  | 0.00%        | Cash preservation  |
| PPO V3 ⭐      | -5.47%      | -0.745 | -7.79%       | Active trading     |
| Buy & Hold     | -18.08%     | -0.963 | -24.46%      | Market benchmark   |
| Random Agent   | -14.37%     | -1.456 | -16.07%      | Random baseline    |

PPO V3 outperformed Buy & Hold by +12.61% in bear market.
Max drawdown reduced from 24.46% to 7.79% — 16.67% improvement.

### RL Agent Evolution Story
- PPO V1: Learned cash preservation — rational in bear market
- PPO V2: Deeper network [128,128,64] — same cash preservation
- PPO V3: BalancedTradingEnv with inactivity penalty (-0.0002)
  broke cash-preservation bias — active trading achieved
  (38 buys, 205 sells across 243 trading days)

### TFT Attention Insight
TFT independently discovered three financially meaningful patterns:
- Days 0-25: Near zero attention — old data is noise
- Day 30 spike: Monthly options expiry cycle detected
- Days 50-60: Highest attention — recency matters most

---

## Project 2 — FraudShield AI 🛡️
### Bank Fraud Detection with Explainable AI

End-to-end fraud detection system handling extreme class imbalance
(598:1) with SHAP explainability and real-time API deployment.

### FraudShield Progress
- [x] Day 1-2 — EDA, SMOTE, 5 models trained and compared
- [ ] Day 3   — SHAP Explainability
- [ ] Day 4   — Autoencoder Anomaly Detection
- [ ] Day 5   — FastAPI Real-Time Scoring Endpoint
- [ ] Day 6   — Streamlit Fraud Analyst Dashboard
- [ ] Day 7   — GitHub Polish + New Fiverr Gig

### Dataset
Credit Card Fraud Detection (Kaggle)
284,807 transactions · 492 frauds · 598:1 imbalance ratio

### FraudShield Results
| Model               | Precision | Recall | F1     | False Alarms |
|---------------------|-----------|--------|--------|--------------|
| Logistic Regression | 5.21%     | 87.37% | 0.0983 | 1,510        |
| Random Forest       | 59.38%    | 80.00% | 0.6816 | 52           |
| XGBoost Base ⭐     | 94.94%    | 78.95% | 0.8621 | 4            |
| XGBoost Tuned V1    | 93.51%    | 75.79% | 0.8372 | 5            |
| XGBoost Tuned V2    | 92.50%    | 77.89% | 0.8457 | 6            |

XGBoost achieved 94.94% precision with only 4 false alarms
per 56,651 legitimate transactions — production-grade performance.

### Key Fraud Insights
- Peak fraud hours: 2AM-4AM (30x higher rate than 10AM)
- Fraud median amount lower than legitimate — threshold evasion
- scale_pos_weight more effective than SMOTE for XGBoost
- Base model outperformed all tuned variants

---

## Key Technical Decisions

### FinSight AI
- Walk-forward validation — no data leakage
- Log returns instead of raw prices — ensures stationarity
- ADF test — statistically confirmed stationarity
- StandardScaler with inverse transform — fair model comparison
- Dropout 0.3 + gradient clipping — overfitting control
- Sharpe-adjusted reward — penalises volatility not just losses
- Transaction cost 0.1% — realistic trading simulation
- Stop loss at 70% — prevents catastrophic drawdown
- PPO clip_range=0.2 — stable policy updates
- Inactivity penalty -0.0002 — forces active trading behaviour

### FraudShield AI
- SMOTE oversampling — 598:1 imbalance handled
- scale_pos_weight=599 — XGBoost native imbalance handling
- Stratified train/test split — preserves fraud ratio
- Log transform on Amount — reduces skewness
- Hour of day feature — temporal fraud pattern signal
- 5-model comparison — rigorous evaluation framework

---

## Project Structure
finsight-ai/
  notebooks/
    FinSight_Day1.ipynb                — EDA + ARIMA baseline
    FinSight_Day2_LSTM.ipynb           — LSTM training pipeline
    FinSight_Day3_TFT.ipynb            — TFT + attention heatmap
    FinSight_Day4_RL_Environment.ipynb — Custom Gymnasium environment
    FinSight_Day5_PPO_Agent.ipynb      — PPO V1+V2+V3 training
    FinSight_Day6_Dashboard.ipynb      — Streamlit dashboard

fraudshield-ai/
  notebooks/
    FraudShield_Day1_Day2_EDA_Models.ipynb — EDA + 5 models

dashboard/
  app.py                             — FinSight Streamlit app
  requirements.txt                   — Dependencies

models/
  ppo_trading_agent.zip              — PPO V1 weights
  ppo_finsight_v2_optimized.zip      — PPO V2 weights
  ppo_balanced_agent_v3.zip          — PPO V3 weights (best)
  fraudshield_xgb_model.pkl          — FraudShield XGBoost model

visuals/
  finsight/
    tft_attention.png                — TFT attention heatmap
    final_equity_comparison.png      — All agents comparison
    ppo_backtest_results.png         — PPO vs baselines
  fraudshield/
    class_distribution.png           — Class imbalance chart
    fraud_by_hour.png                — Temporal fraud patterns
    complete_model_comparison.png    — All models comparison

---

## Contact
📧 suman.ju.ai@gmail.com
🔗 linkedin.com/in/suman-das-6b0749276
💻 github.com/suman-ju-ai
🎯 Upwork: upwork.com/freelancers/sumandas
