# Fraud Detection Platform — Roadmap

All items from the previous TODO are complete. Remaining suggestions below,
ordered by impact.

## 🚀 Hosting & Deployment
- [ ] Streamlit Cloud / Hugging Face Spaces deployment for dashboard (free, git-backed)
- [ ] Containerize API + dashboard (Dockerfile + docker-compose with Kafka/Redis)
- [ ] CI pipeline (GitHub Actions): lint + pytest + API smoke test on push

## 🔧 Model
- [ ] Retrain GNN with the new 2-hop sampling and rebuild ensemble meta-learner
- [ ] Extend graph cache to streaming test transactions (currently training-graph only)

## 📈 Ops
- [ ] Wire scheduler.py into Windows Task Scheduler / cron for real daily runs
- [ ] Populate ALERT_SLACK_WEBHOOK / ALERT_EMAIL_* in .env for alert delivery
