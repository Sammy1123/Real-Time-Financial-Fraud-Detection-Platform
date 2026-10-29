"""
Fraud Detection Platform — Streamlit Dashboard
Pages: Overview | Transactions | Model Performance | Live Alerts | Drift Monitor

Hosted via Streamlit Community Cloud (git-backed) or locally:
    python main.py dashboard
"""
import json
import time
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
)

# ── Config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent.parent
PROC        = ROOT / "data" / "processed"
MODELS      = ROOT / "models"
ALERTS_FILE = ROOT / "streaming" / "fraud_alerts.jsonl"
DRIFT_FILE  = ROOT / "monitoring" / "drift_report.json"

# ── Cached loaders ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_train():
    """Downsample for cloud-friendly memory footprint; full data locally."""
    df = pd.read_parquet(PROC / "train.parquet")
    if len(df) > 150_000:
        # keep every fraud row, subsample legit rows
        fraud = df[df.isFraud == 1]
        legit = df[df.isFraud == 0].sample(
            n=min(140_000, (df.isFraud == 0).sum()), random_state=42)
        df = pd.concat([fraud, legit]).sample(frac=1, random_state=42)
    return df

@st.cache_data(show_spinner=False)
def load_meta():
    return json.load(open(MODELS / "model_meta.json"))

@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODELS / "xgboost_model.pkl")

def load_alerts() -> pd.DataFrame:
    if not ALERTS_FILE.exists() or ALERTS_FILE.stat().st_size == 0:
        return pd.DataFrame()
    rows = [json.loads(l) for l in ALERTS_FILE.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df

# ── Sidebar nav ────────────────────────────────────────────────────────────
st.sidebar.title("🛡️ Fraud Detection")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Transactions", "Model Performance", "Live Alerts", "Drift Monitor"],
)
st.sidebar.markdown("---")
st.sidebar.caption("IEEE-CIS Fraud Detection Platform")

# ════════════════════════
# PAGE 1 — Overview
# ════════════════════════
if page == "Overview":
    st.title("📊 Overview")

    df   = load_train()
    meta = load_meta()

    total      = len(df)
    fraud_n    = int(df["isFraud"].sum())
    legit_n    = total - fraud_n
    fraud_rate = fraud_n / total

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Fraud Cases",        f"{fraud_n:,}", delta=f"{fraud_rate:.2%}", delta_color="inverse")
    c3.metric("Model ROC-AUC",      f"{meta['roc_auc']:.4f}")
    c4.metric("Model PR-AUC",       f"{meta['pr_auc']:.4f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    # Class distribution
    with col1:
        fig = px.bar(
            x=["Legit", "Fraud"], y=[legit_n, fraud_n],
            color=["Legit", "Fraud"],
            color_discrete_map={"Legit": "steelblue", "Fraud": "crimson"},
            title="Class Distribution", labels={"x": "", "y": "Count"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Fraud rate by ProductCD
    with col2:
        prod = df.groupby("ProductCD")["isFraud"].mean().reset_index()
        prod.columns = ["ProductCD", "FraudRate"]
        prod = prod.sort_values("FraudRate", ascending=False)
        fig = px.bar(prod, x="ProductCD", y="FraudRate",
                     title="Fraud Rate by ProductCD",
                     color="FraudRate", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    # Transaction amount distribution
    with col3:
        sample = df.sample(min(10000, len(df)), random_state=42)
        fig = px.histogram(
            sample, x="log_amt", color=sample["isFraud"].map({0: "Legit", 1: "Fraud"}),
            barmode="overlay", nbins=60, opacity=0.7,
            title="Log Transaction Amount (Legit vs Fraud)",
            color_discrete_map={"Legit": "steelblue", "Fraud": "crimson"},
            labels={"log_amt": "log1p(TransactionAmt)", "color": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Fraud by hour
    with col4:
        hourly = df.groupby("hour")["isFraud"].mean().reset_index()
        fig = px.line(hourly, x="hour", y="isFraud", markers=True,
                      title="Fraud Rate by Hour of Day",
                      labels={"hour": "Hour", "isFraud": "Fraud Rate"})
        fig.update_traces(line_color="crimson")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════
# PAGE 2 — Transactions
# ════════════════════════
elif page == "Transactions":
    st.title("💳 Transactions")

    df = load_train()

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        label_filter = st.selectbox("Label", ["All", "Fraud", "Legit"])
    with fc2:
        amt_min, amt_max = float(df["TransactionAmt"].min()), float(df["TransactionAmt"].max())
        amt_range = st.slider("Transaction Amount", amt_min, min(amt_max, 5000.0), (amt_min, 1000.0))
    with fc3:
        hour_range = st.slider("Hour of Day", 0, 23, (0, 23))

    filtered = df.copy()
    if label_filter == "Fraud":
        filtered = filtered[filtered["isFraud"] == 1]
    elif label_filter == "Legit":
        filtered = filtered[filtered["isFraud"] == 0]
    filtered = filtered[
        filtered["TransactionAmt"].between(*amt_range) &
        filtered["hour"].between(*hour_range)
    ]

    st.markdown(f"Showing **{len(filtered):,}** transactions")

    display_cols = ["TransactionID", "TransactionAmt", "log_amt", "hour", "day", "isFraud"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[display_cols].head(500),
        use_container_width=True,
        height=400,
    )

    # Amount distribution for filtered set
    if len(filtered) > 0:
        sample_n = min(5000, len(filtered))
        sample = filtered.sample(sample_n, random_state=42)
        fig = px.histogram(
            sample, x="TransactionAmt", nbins=50,
            color=sample["isFraud"].map({0:"Legit",1:"Fraud"}),
            barmode="overlay", opacity=0.7,
            title="Transaction Amount Distribution (filtered)",
            color_discrete_map={"Legit": "steelblue", "Fraud": "crimson"},
            labels={"color": ""},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No transactions match the filters.")


# ══════════════
# PAGE 3 — Model Performance
# ════════════════════════
elif page == "Model Performance":
    st.title("🤖 Model Performance")

    meta  = load_meta()
    model = load_model()
    df    = load_train()

    FEATURES = meta["features"]

    # Sample for curves (full set is slow)
    sample = df.sample(min(30000, len(df)), random_state=42)
    X      = sample[FEATURES].values
    y      = sample["isFraud"].values
    proba  = model.predict_proba(X)[:, 1]

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("ROC-AUC",       f"{roc_auc_score(y, proba):.4f}")
    mc2.metric("PR-AUC",        f"{average_precision_score(y, proba):.4f}")
    mc3.metric("Threshold",     f"{meta['threshold']:.2f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    # ROC curve
    with col1:
        fpr, tpr, _ = roc_curve(y, proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC",
                                 line=dict(color="steelblue", width=2)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                 line=dict(dash="dash", color="gray"), name="Random"))
        fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig, use_container_width=True)

    # PR curve
    with col2:
        prec, rec, thresholds = precision_recall_curve(y, proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR",
                                 line=dict(color="crimson", width=2)))
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray",
                      annotation_text="50% Recall")
        fig.update_layout(title="Precision-Recall Curve",
                          xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, use_container_width=True)

    # Threshold tradeoff
    st.subheader("Threshold Tradeoff")
    from sklearn.metrics import f1_score
    thresh_range = np.arange(0.1, 0.95, 0.01)
    f1s   = [f1_score(y, proba >= t, zero_division=0) for t in thresh_range]
    precs = [pd.Series((proba >= t)).pipe(lambda s: (y[s.values] == 1).mean() if s.sum() > 0 else 0) for t in thresh_range]
    recs  = [((proba >= t) & (y == 1)).sum() / max((y == 1).sum(), 1) for t in thresh_range]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresh_range, y=f1s,   name="F1",        line=dict(color="steelblue")))
    fig.add_trace(go.Scatter(x=thresh_range, y=precs,  name="Precision", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=thresh_range, y=recs,   name="Recall",    line=dict(color="crimson")))
    fig.add_vline(x=meta["threshold"], line_dash="dash", line_color="orange",
                  annotation_text=f"Current ({meta['threshold']:.2f})")
    fig.update_layout(title="F1 / Precision / Recall vs Threshold",
                      xaxis_title="Threshold", yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.subheader("Top 20 Feature Importances")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"feature": FEATURES, "importance": importances})
    feat_df = feat_df.nlargest(20, "importance").sort_values("importance")
    fig = px.bar(feat_df, x="importance", y="feature", orientation="h",
                 color="importance", color_continuous_scale="Blues",
                 title="XGBoost Feature Importances (gain)")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════
# PAGE 4 — Live Alerts
# ══════════════
elif page == "Live Alerts":
    st.title("🚨 Live Fraud Alerts")

    auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=True)

    alerts = load_alerts()

    if alerts.empty:
        st.info("No fraud alerts yet. Start the Kafka producer and consumer to see live alerts.")
        st.code("python main.py producer\npython main.py consumer")
    else:
        a1, a2, a3 = st.columns(3)
        a1.metric("Total Alerts",    f"{len(alerts):,}")
        a2.metric("Avg Fraud Prob",  f"{alerts['fraud_probability'].mean():.4f}")
        a3.metric("Total Amount",    f"${alerts['TransactionAmt'].sum():,.2f}")

        st.markdown("---")

        # Risk level breakdown
        col1, col2 = st.columns(2)
        with col1:
            risk_counts = alerts["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["risk_level", "count"]
            fig = px.pie(risk_counts, names="risk_level", values="count",
                         title="Alerts by Risk Level",
                         color="risk_level",
                         color_discrete_map={"HIGH":"crimson","MEDIUM":"orange","LOW":"steelblue"})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(alerts, x="fraud_probability", nbins=30,
                               title="Fraud Probability Distribution",
                               color_discrete_sequence=["crimson"])
            st.plotly_chart(fig, use_container_width=True)

        # Alerts over time
        alerts_sorted = alerts.sort_values("timestamp")
        alerts_sorted["cumulative"] = range(1, len(alerts_sorted) + 1)
        fig = px.line(alerts_sorted, x="timestamp", y="cumulative",
                      title="Cumulative Fraud Alerts Over Time",
                      labels={"cumulative": "Alerts", "timestamp": "Time"})
        fig.update_traces(line_color="crimson")
        st.plotly_chart(fig, use_container_width=True)

        # Raw alerts table
        st.subheader("Recent Alerts")
        st.dataframe(
            alerts.sort_values("timestamp", ascending=False).head(100),
            use_container_width=True,
            height=350,
        )

    if auto_refresh:
        time.sleep(5)
        st.rerun()


# ══════════════
# PAGE 5 — Drift Monitor
# ══════════════
elif page == "Drift Monitor":
    st.title("📡 Drift Monitor")

    if not DRIFT_FILE.exists():
        st.warning("No drift report found. Run the monitor first.")
        st.code("python main.py monitor")
        st.stop()

    report = json.load(open(DRIFT_FILE))

    data_alert  = report["alerts"]["data_drift_alert"]
    model_alert = report["alerts"]["model_drift_alert"]

    # ── KPI row ────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Data Drift Alert",  "YES" if data_alert  else "NO",
              delta=None, delta_color="inverse" if data_alert else "normal")
    k2.metric("Model Drift Alert", "YES" if model_alert else "NO")
    k3.metric("Features Drifted (PSI)",
              report["data_drift"]["features_drifted"])
    k4.metric("Features Failing KS",
              report["ks_test"]["features_failing"])

    if data_alert:
        st.error("Data drift detected — feature distributions have shifted significantly.")
    else:
        st.success("No significant data drift detected.")

    excluded = report["data_drift"].get("excluded_identity_only", [])
    if excluded:
        st.caption(f"ℹ️ Identity-only columns excluded from alerting "
                   f"(coverage skew, not real drift): {', '.join(excluded)}")

    st.caption(f"Report generated: {report['run_timestamp']}  |  "
               f"Reference: {report['reference_size']:,} rows  |  "
               f"Current: {report['current_size']:,} rows")

    st.markdown("---")

    # ── PSI chart ──────────────────────────────────────────────────────────
    st.subheader("PSI by Feature (top 20)")
    psi_data = report["data_drift"]["all_psi"]
    psi_df   = (pd.DataFrame(list(psi_data.items()), columns=["feature", "psi"])
                  .sort_values("psi", ascending=False)
                  .head(20))

    psi_df["status"] = psi_df["psi"].apply(
        lambda v: "Alert (>0.2)" if v >= 0.2 else ("Warn (>0.1)" if v >= 0.1 else "OK"))

    fig = px.bar(psi_df.sort_values("psi"), x="psi", y="feature",
                 orientation="h", color="status",
                 color_discrete_map={"Alert (>0.2)": "crimson",
                                     "Warn (>0.1)": "orange",
                                     "OK": "steelblue"},
                 title="Population Stability Index (PSI) — Top 20 Features",
                 labels={"psi": "PSI", "feature": ""})
    fig.add_vline(x=0.1, line_dash="dash", line_color="orange",
                  annotation_text="Warn 0.1")
    fig.add_vline(x=0.2, line_dash="dash", line_color="crimson",
                  annotation_text="Alert 0.2")
    st.plotly_chart(fig, use_container_width=True)

    # ── KS chart ───────────────────────────────────────────────────────────
    st.subheader("KS Statistic by Feature (top 20)")
    ks_data = report["ks_test"]["results"]
    ks_df   = (pd.DataFrame(
                   [{"feature": k, "ks_stat": v["statistic"], "p_value": v["p_value"]}
                    for k, v in ks_data.items()])
                 .sort_values("ks_stat", ascending=False)
                 .head(20))

    ks_df["significant"] = ks_df["p_value"].apply(
        lambda p: "Significant (p<0.05)" if p < 0.05 else "Not Significant")

    fig = px.bar(ks_df.sort_values("ks_stat"), x="ks_stat", y="feature",
                 orientation="h", color="significant",
                 color_discrete_map={"Significant (p<0.05)": "crimson",
                                     "Not Significant": "steelblue"},
                 title="KS Test Statistic — Top 20 Features",
                 labels={"ks_stat": "KS Statistic", "feature": ""})
    st.plotly_chart(fig, use_container_width=True)

    # ── Model score drift ──────────────────────────────────────────────────
    st.subheader("Model Score Drift")
    md = report["model_drift"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Score PSI",         f"{md['score_psi']:.4f}")
    m2.metric("Score KS Statistic",f"{md['score_ks_statistic']:.4f}")
    m3.metric("Ref Fraud Rate",    f"{md['ref_fraud_rate']:.4%}")
    m4.metric("Cur Fraud Rate",    f"{md['cur_fraud_rate']:.4%}",
              delta=f"{md['fraud_rate_delta']:+.4%}")
