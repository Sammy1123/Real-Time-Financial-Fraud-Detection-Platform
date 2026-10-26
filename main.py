"""
Real-Time Financial Fraud Detection Platform
Main entry point - run individual components via CLI.

Usage:
    python main.py eda
    python main.py features
    python main.py train
    python main.py api
    python main.py producer
    python main.py dashboard
    python main.py monitor
"""
import argparse
import subprocess
import sys


COMMANDS = {
    "download":  ["python", "data/download_data.py"],
    "eda":       ["python", "data/eda.py"],
    "features":  ["python", "features/feature_engineering.py"],
    "train":     ["python", "models/train_xgboost.py"],
    "gnn":       ["python", "models/train_gnn.py"],
    "api":       ["python", "-m", "uvicorn", "api.scoring_service:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
    "producer":  ["python", "streaming/kafka_producer.py"],
    "consumer":  ["python", "streaming/kafka_consumer.py"],
    "dashboard": ["python", "-m", "streamlit", "run", "dashboard/app.py"],
    "monitor":   ["python", "monitoring/drift_detection.py"],
    "tune":      ["python", "models/tune_xgboost_optuna.py"],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection Platform CLI")
    parser.add_argument("command", choices=list(COMMANDS.keys()), help="Component to run")
    args = parser.parse_args()
    subprocess.run(COMMANDS[args.command], check=True)
