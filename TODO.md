# Fraud Detection Platform — TODO / Pending Improvements

## 🔧 Model Tuning

### GNN Improvements (Priority: MEDIUM)
- Remaining: multi-hop neighbors (currently 1-hop, k=10) — torch-sparse unavailable (no C++ build tools)
- Remaining: real-time graph serving (API uses prior-day graph / base-rate fallback)

---

## 📡 Streaming

### Producer (Priority: LOW)
- PRODUCER_DELAY_MS=100 simulates ~10 tx/s — tune down to 0 for load testing

---

## 📈 Monitoring

### Drift Detection Improvements (Priority: LOW)
- M1/M3/M7/M8/M9 features show significant PSI drift (>0.2) — investigate root cause
- Add scheduled runs (cron / Airflow) to run drift_detection.py daily
- Add alerting: send email/Slack webhook when data_drift_alert or model_drift_alert is True
