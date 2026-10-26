# Fraud Detection Platform — TODO / Pending Improvements

## 🔧 Model Tuning (XGBoost)

### Threshold Tuning (Priority: HIGH)
- ✅ Done — `models/tune_threshold.py` built with PR curve, cost analysis, --target-recall CLI arg
- Results at different strategies:
  - Best F1 threshold : 0.83 (current) — F1=0.59, Recall=51%
  - Min cost threshold: 0.60 (FN=10x, FP=1x) — better for production
  - 80% recall threshold: 0.52 — catches 80% of fraud, precision drops to 27%
- To apply a new threshold: `py -3.10 models/tune_threshold.py --target-recall 0.8 --save`

### GNN Improvements (Priority: MEDIUM)
- ROC-AUC 0.68 / PR-AUC 0.51 — weaker than XGBoost standalone (0.94 / 0.63)
- GNN's real value is as an ensemble signal, not standalone — combine XGBoost prob + GNN prob as input to a meta-learner
- Graph currently only uses card1 edges — add addr1 and P_emaildomain edges for richer fraud ring detection
- Train on full dataset with NeighborLoader (mini-batch) instead of full-graph on subsample
- Replace GraphSAGE with GAT (Graph Attention Network) — learns which neighbours matter more
- Loss is high (9–22) — scale node features with StandardScaler before training

### Hyperparameter Tuning (Priority: MEDIUM)
- Current params are reasonable defaults, not tuned
- Try Optuna or sklearn BayesSearchCV on:
  - max_depth (4–8), learning_rate (0.01–0.1), subsample, colsample_bytree
  - min_child_weight (helps with imbalance)
- Use PR-AUC as the optimization metric (not ROC-AUC — misleading on imbalanced data)

### Feature Improvements (Priority: MEDIUM)
- Add more aggregation features: freq encoding per (card1 + addr1) pair
- Add velocity features: # transactions per card1 in last 1h / 24h (needs time-sorted data)
- Consider target encoding for high-cardinality categoricals instead of label encoding

---

## 🌐 API (scoring_service.py)

### Startup Performance (Priority: HIGH)
- ✅ Fixed — freq maps + cat maps now serialized as pickles during feature engineering
- API loads them instantly via joblib.load on startup

### Caching (Priority: MEDIUM)
- Add Redis caching for repeated card1/addr1 lookups
- Cache model predictions for identical feature vectors (TTL: 60s)

### Lifespan Shutdown Hook (Priority: LOW)
- Add cleanup after `yield` in lifespan: close Redis connections, flush any buffers

---

## 🔄 Pipeline

- GNN model (models/train_gnn.py) — ✅ built (GraphSAGE, ROC-AUC 0.68, PR-AUC 0.51)
- Kafka producer/consumer (streaming/) — ✅ built
- Dashboard (dashboard/app.py) — ✅ built (4 pages: Overview, Transactions, Model Performance, Live Alerts)
- Drift monitoring (monitoring/drift_detection.py) — ✅ built (PSI + KS test, MLflow logging)

## 📡 Streaming (kafka_producer/consumer)

### Needs Kafka Running (Priority: HIGH)
- Both producer and consumer require a live Kafka broker
- Quick local start: `docker run -p 9092:9092 apache/kafka:3.7.0`
- For production: use Amazon MSK (Managed Streaming for Kafka)

### Consumer Resilience (Priority: MEDIUM)
- Add retry logic with exponential backoff on scoring API failures
- Add dead-letter topic for transactions that fail scoring after N retries
- Consumer currently scores one-by-one — batch scoring endpoint on the API would improve throughput

### Producer (Priority: LOW)
- PRODUCER_DELAY_MS=100 simulates ~10 tx/s — tune down to 0 for load testing
- Add schema validation (Avro/Protobuf) before publishing to topic

---

## 📋 General

- Pin all dependencies to exact versions matching Python 3.10 env in requirements.txt
- Add .env file for config (model path, threshold override, Redis URL, Kafka brokers)
- Add integration tests in tests/ that hit the /score endpoint with known fraud/legit samples

## 📈 Monitoring Improvements

### Drift Detection (Priority: MEDIUM)
- Currently splits train data into ref/current — in production, reference = training data, current = live scored transactions window
- M1/M3/M7/M8/M9 features show significant PSI drift (>0.2) — investigate if these are label-encoded categoricals shifting or real distribution change
- KS test flagging all 54 features is expected with 400k vs 177k samples — KS is very sensitive at large N, PSI is more actionable
- Add scheduled runs (cron / Airflow) to run drift_detection.py daily
- Add alerting: send email/Slack webhook when data_drift_alert or model_drift_alert is True
- Integrate drift report into the Streamlit dashboard as a 5th page
