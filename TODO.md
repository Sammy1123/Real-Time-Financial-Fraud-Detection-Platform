# Fraud Detection Platform — TODO / Pending Improvements

## 🔧 Model Tuning

### GNN Improvements (Priority: MEDIUM)
- ROC-AUC 0.68 / PR-AUC 0.51 — weaker than XGBoost standalone (0.94 / 0.63)
- GNN's real value is as an ensemble signal — combine XGBoost prob + GNN prob as input to a meta-learner
- Graph currently only uses card1 edges — add addr1 and P_emaildomain edges for richer fraud ring detection
- Train on full dataset with NeighborLoader (mini-batch) instead of full-graph on subsample
- Replace GraphSAGE with GAT (Graph Attention Network) — learns which neighbours matter more
- Loss is high (9–22) — scale node features with StandardScaler before training

### Hyperparameter Tuning (Priority: MEDIUM)
- Current params are reasonable defaults, not tuned
- Try Optuna on: max_depth (4–8), learning_rate (0.01–0.1), subsample, colsample_bytree, min_child_weight
- Use PR-AUC as the optimization metric (not ROC-AUC — misleading on imbalanced data)

### Feature Improvements (Priority: MEDIUM)
- Add freq encoding per (card1 + addr1) pair
- Add velocity features: # transactions per card1 in last 1h / 24h (needs time-sorted data)
- Consider target encoding for high-cardinality categoricals instead of label encoding

---

## 🌐 API

### Caching (Priority: MEDIUM)
- Add Redis caching for repeated card1/addr1 lookups
- Cache model predictions for identical feature vectors (TTL: 60s)

### Lifespan Shutdown Hook (Priority: LOW)
- Add cleanup after `yield` in lifespan: close Redis connections, flush any buffers

### Batch Scoring Endpoint (Priority: MEDIUM)
- Consumer currently scores one-by-one — add POST /score/batch for higher throughput

---

## 📡 Streaming

### Consumer Resilience (Priority: MEDIUM)
- Add retry logic with exponential backoff on scoring API failures
- Add dead-letter topic for transactions that fail scoring after N retries

### Producer (Priority: LOW)
- PRODUCER_DELAY_MS=100 simulates ~10 tx/s — tune down to 0 for load testing
- Add schema validation (Avro/Protobuf) before publishing to topic

---

## 📈 Monitoring

### Drift Detection Improvements (Priority: MEDIUM)
- M1/M3/M7/M8/M9 features show significant PSI drift (>0.2) — investigate root cause
- Add scheduled runs (cron / Airflow) to run drift_detection.py daily
- Add alerting: send email/Slack webhook when data_drift_alert or model_drift_alert is True
- Integrate drift report into the Streamlit dashboard as a 5th page

---

## 📋 General

- Pin all dependencies to exact versions matching Python 3.10 env in requirements.txt
- Add integration tests in tests/ that hit the /score endpoint with known fraud/legit samples
