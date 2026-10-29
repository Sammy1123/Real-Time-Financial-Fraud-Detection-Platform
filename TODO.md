# Fraud Detection Platform — TODO / Pending Improvements

## 🔧 Model Tuning

### GNN Improvements
- ✅ DONE: Full-dataset training (590k nodes, 12.7M edges)
- ✅ DONE: 2-hop neighbour sampling (bounded k=10/5 per level)
- ✅ DONE: Nightly graph cache (api/build_graph_cache.py) feeds live GNN signal to scoring API
- Remaining: upgrade scheduler.py one-shot runs to Windows Task Scheduler / cron in production
- Remaining: graph cache currently scores training-graph nodes; extend to cover streaming test transactions

---

## 📡 Streaming

### Producer
- ✅ DONE: Schema validation before publish (validate_message; rejects to DLQ)
- ✅ DONE: Load-test mode (`--delay-ms 0`, `--limit N`) with live throughput reporting

---

## 📈 Monitoring

### Drift Detection
- ✅ DONE: PSI drift on M1/M3/M7/M8/M9 diagnosed — identity-table coverage skew (41% → ~19% over time), excluded from alerting with rationale
- ✅ DONE: scheduler.py — daily pipeline (graph cache → drift → alerts)
- ✅ DONE: alerting.py — Slack webhook + email (SMTP) on drift alerts
- Remaining: populate ALERT_SLACK_WEBHOOK / ALERT_EMAIL_* in .env for real delivery
