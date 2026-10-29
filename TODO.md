# Fraud Detection Platform — TODO / Pending Improvements

## 🔧 Model Tuning

### GNN Improvements (Priority: MEDIUM)
- Retrain GNN with 2-hop sampling, then rebuild ensemble meta-learner (current checkpoint is 1-hop)
- Extend graph cache to streaming test transactions (currently training-graph nodes only)

---

## 📡 Streaming

No open items.

---

## 📈 Monitoring

### Drift Detection (Priority: LOW)
- Wire scheduler.py into Windows Task Scheduler / cron for real daily runs
- Populate ALERT_SLACK_WEBHOOK / ALERT_EMAIL_* in .env for real alert delivery
