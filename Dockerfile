FROM python:3.10-slim

WORKDIR /app

# Install system deps for parquet/xgboost wheels (none needed beyond defaults)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY api/ api/
COPY models/ models/
COPY features/feature_store_data/ features/feature_store_data/
COPY dashboard/ dashboard/
COPY monitoring/ monitoring/
COPY data/processed/ data/processed/

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
