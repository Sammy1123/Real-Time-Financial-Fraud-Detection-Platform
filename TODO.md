# Fraud Detection Platform — TODO / Pending Improvements

## 🔧 Model Tuning

### GNN Improvements (Priority: MEDIUM)
- ROC-AUC 0.68 / PR-AUC 0.51 — weaker than XGBoost standalone (0.95 / 0.74)
- GNN's real value is as an ensemble signal — combine XGBoost prob + GNN prob as input to a meta-learner
- Graph currently only uses card1 edges — add addr1 and P_emaildomain edges for richer fraud ring detection
- Train on full dataset with NeighborLoader (mini-batch) instead of full-graph on subsample
- Replace GraphSAGE with GAT (Graph Attention Network) — learns which neighbours matter more
- Loss is high (9–22) — scale node features with StandardScaler before training

---

## 🌐 API

### Batch Scoring Endpoint (Priority: MEDIUM)
- Consumer currently scores one-by-one — add POST /score/batch for higher throughput

### Caching (Priority: MEDIUM)
- Add Redis caching for repeated card1/addr1 lookups
- Cache model predictions for identical feature vectors (TTL: 60s)

### Lifespan Shutdown Hook (Priority: LOW)
- Add cleanup after `yield` in lifespan: close Redis connections, flush any buffers

---

## 📡 Streaming

### Producer (Priority: LOW)
- PRODUCER_DELAY_MS=100 simulates ~10 tx/s — tune down to 0 for load testing
- Add schema validation (Avro/Protobuf) before publishing to topic

---

## 📈 Monitoring

### Drift Detection Improvements (Priority: LOW)
- M1/M3/M7/M8/M9 features show significant PSI drift (>0.2) — investigate root cause
- Add scheduled runs (cron / Airflow) to run drift_detection.py daily
- Add alerting: send email/Slack webhook when data_drift_alert or model_drift_alert is True

---

## 📋 General

- Pin all dependencies to exact versions matching Python 3.10 env in requirements.txt
- Add integration tests in tests/ that hit the /score endpoint with known fraud/legit samples
