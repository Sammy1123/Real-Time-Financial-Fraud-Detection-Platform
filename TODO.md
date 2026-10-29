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
- Email alerts (alerting.py) are dormant until ALERT_EMAIL_* is set in .env — optional

---

## 🎓 Presentability (internship review recommendations)

1. ~~Unit tests for feature engineering & monitoring~~ ✅ DONE — tests/test_pipeline_logic.py, 29 tests covering chain_edges, PSI, schema validation, preprocessing
2. ~~Commit the small artifacts~~ ✅ DONE — train_sample.parquet + model_meta.json committed, dashboard falls back to the sample
3. ~~requirements.txt dev/prod split~~ ✅ DONE — requirements-dev.txt for pytest/httpx/playwright
4. ~~Type hints + docstrings on streaming modules~~ ✅ DONE — both Kafka modules fully annotated
5. ~~Architecture diagram image~~ ✅ DONE — mermaid flowchart (GitHub renders natively)
6. ~~Dashboard walkthrough~~ — SKIPPED per decision; capture_demo.py remains available as an optional tool
