# 📦 Smart Inventory Manager

Smart Inventory Manager is an Applied Machine Learning project that analyzes inventory and sales data and provides intelligent product insights.

## Features

- Inventory CSV/Excel upload
- Data preprocessing
- Feature engineering
- K-Means clustering
- Fast-Moving / Normal-Moving / Slow-Moving classification
- Isolation Forest anomaly detection
- Product analysis
- Category analysis
- Interactive Plotly charts
- Smart insights
- Search and filters
- Downloadable reports

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- OpenPyXL

## Machine Learning

### K-Means Clustering

Groups products according to:

- Stock
- Units Sold
- Revenue
- Sales Frequency
- Stock Utilization
- Stock Turnover

### Isolation Forest

Detects unusual inventory and sales behavior.

## Workflow

Inventory Data
→ Preprocessing
→ Feature Engineering
→ K-Means
→ Product Movement Analysis
→ Isolation Forest
→ Anomaly Detection
→ Smart Insights
→ Dashboard

## Run Project

Install dependencies:

```bash
python -m pip install streamlit pandas numpy scikit-learn plotly openpyxl joblib