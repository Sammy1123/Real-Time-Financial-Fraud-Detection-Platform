# Fraud Detection Platform — TODO / Pending Improvements

## 🔧 Model Tuning

### GNN Improvements (Priority: MEDIUM)
- ✅ DONE: Full-dataset training (590k nodes, 12.7M edges) — ROC-AUC 0.68→~0.79, PR-AUC improving across epochs
- ✅ DONE: Ensemble updated for full-graph inference (100% validation coverage, was 3%)
- Remaining: multi-hop neighbors (currently 1-hop, k=10) — torch-sparse unavailable (no C++ build tools)
- Remaining: real-time graph serving (API uses prior-day graph / base-rate fallback)

---

## 📡 Streaming

### Producer (Priority: LOW)
- ✅ DONE: Schema validation before publish (validate_message; rejects to DLQ)
- PRODUCER_DELAY_MS=100 simulates ~10 tx/s — tune down to 0 for load testing

---

## 📈 Monitoring

### Drift Detection Improvements (Priority: LOW)
- M1/M3/M7/M8/M9 features show significant PSI drift (>0.2) — investigate root cause
- Add scheduled runs (cron / Airflow) to run drift_detection.py daily
- Add alerting: send email/Slack webhook when data_drift_alert or model_drift_alert is True
