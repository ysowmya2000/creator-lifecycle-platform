FROM python:3.12-slim

WORKDIR /app

COPY dashboard/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard/ ./dashboard/

COPY data/processed/features.parquet ./data/processed/features.parquet
COPY data/processed/rd_dataset.parquet ./data/processed/rd_dataset.parquet
COPY data/processed/competing_risk_labels.parquet ./data/processed/competing_risk_labels.parquet

COPY outputs/module1_metrics.json ./outputs/module1_metrics.json
COPY outputs/module2_metrics.json ./outputs/module2_metrics.json
COPY outputs/module3_metrics.json ./outputs/module3_metrics.json
COPY outputs/module2_did_results.csv ./outputs/module2_did_results.csv
COPY outputs/module2_ltv_by_tier.csv ./outputs/module2_ltv_by_tier.csv
COPY outputs/models/module1_xgb.pkl ./outputs/models/module1_xgb.pkl

EXPOSE 8501

CMD streamlit run dashboard/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true
