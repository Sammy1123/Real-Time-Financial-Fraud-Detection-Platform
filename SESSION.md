# Project Session Notes — Fraud Detection Platform

## What's Built (all working)
- `data/eda.py` — EDA, 8 plots + summary report to data/eda_outputs/
- `features/feature_engineering.py` — 54 features, outputs train/test parquets + freq_maps.pkl + cat_maps.pkl
- `models/train_xgboost.py` — XGBoost, ROC-AUC 0.94, PR-AUC 0.63, threshold 0.83, saved to models/
- `models/train_gnn.py` — GraphSAGE, ROC-AUC 0.68, PR-AUC 0.51, saved to models/gnn_model.pt
- `api/scoring_service.py` — FastAPI, POST /score + GET /health, fast startup via pre-serialized maps
- `streaming/kafka_producer.py` — streams test transactions to 'transactions' Kafka topic
- `streaming/kafka_consumer.py` — scores transactions via API, publishes to 'fraud-alerts' topic
- `dashboard/app.py` — Streamlit, 4 pages: Overview, Transactions, Model Performance, Live Alerts
- `monitoring/drift_detection.py` — PSI + KS drift detection, MLflow logging, drift_report.json

## Environment
- Python 3.10 (C:\Users\HP\AppData\Local\Programs\Python\Python310)
- Run scripts with: py -3.10 or python main.py <command>
- All deps installed in Python 3.10 env

## What Needs Doing (see TODO.md for full detail)
1. HIGH: Threshold tuning — build models/tune_threshold.py
2. ✅ DONE: API startup speed — freq/cat maps serialized as pickles
3. ✅ DONE: Kafka + Docker — installed and running
4. MEDIUM: GNN improvements — ensemble with XGBoost, GAT, StandardScaler, full dataset
5. MEDIUM: XGBoost hyperparameter tuning via Optuna
6. MEDIUM: Velocity features in feature engineering
7. MEDIUM: Redis caching in API
8. MEDIUM: Drift dashboard page + alerting
9. LOW: Integration tests, requirements.txt pinning, .env config

## How to Run
- Dashboard : python main.py dashboard
- API       : python main.py api
- Producer  : python main.py producer  (needs Kafka)
- Consumer  : python main.py consumer  (needs Kafka)
- Monitor   : python main.py monitor
- Train     : python main.py train

## MLflow
- 3 experiments logged: fraud-detection-xgboost, fraud-detection-gnn, fraud-detection-monitoring
- View UI: py -3.10 -m mlflow ui  → http://localhost:5000

## Next Session — Suggested Order
1. Threshold tuning script (models/tune_threshold.py)
2. Test full streaming pipeline (API + consumer + producer)
3. Optuna hyperparameter tuning
4. Velocity features in feature engineering
