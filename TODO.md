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

High-impact items an interviewer is likely to probe:

1. **Unit tests for feature engineering & monitoring** — the API has integration tests, but the PSI computation, chain-linking edge builder, and preprocessing have no direct tests. A pytest file per module (~10 tests each) would let you say "the pipeline logic is tested", not just the HTTP layer.
2. **Commit the small artifacts the dashboard needs** — `models/model_meta.json` and a downsampled `train_sample.parquet` (~150k rows, a few MB) would let anyone clone the repo and run the dashboard without downloading 590k rows from Kaggle.
3. **requirements.txt dev/prod split** — separate out pytest, httpx etc. into requirements-dev.txt so production installs stay minimal.
4. **Type hints + docstring pass on streaming modules** — `kafka_producer.py` and `kafka_consumer.py` are the least documented files; a 15-minute pass brings them up to the same standard as `api/`.
5. **One architecture diagram image** (even a screenshot of the mermaid diagram in README.md rendered via mermaid.live) — pinned at the top of the README it reads better in a GitHub preview than ASCII art.
6. **Record a 60-second dashboard walkthrough** (OBS/PowerPoint screen record) — a GIF or mp4 linked from the README shows the product working without making anyone run it.
