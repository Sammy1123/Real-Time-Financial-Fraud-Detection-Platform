# Fraud Detection Platform — TODO / Pending Improvements

## 🔧 Model Tuning

### GNN Improvements (Priority: MEDIUM)
- Current GAT: ROC-AUC 0.71 / PR-AUC 0.47 (up from 0.68 / 0.51 baseline)
- Train on full dataset with NeighborLoader (mini-batch) instead of 67k subsample
- GNN covers only ~3% of validation rows — ensemble PR-AUC gain (+0.8pp) is on covered rows only; expand coverage before trusting the ensemble

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
