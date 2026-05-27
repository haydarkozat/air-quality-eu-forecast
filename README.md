# 🌍 European Air Quality Forecasting System

> **AI-powered PM2.5 prediction for major European cities — aligned with the EU Green Deal and Zero Pollution Action Plan.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Green AI](https://img.shields.io/badge/Green%20AI-CodeCarbon-2ea44f.svg)](https://codecarbon.io)

---

## 🎯 Problem

Air pollution is responsible for **300,000+ premature deaths per year in the EU**, making it the single largest environmental health risk in Europe. The EU's **Zero Pollution Action Plan** (under the European Green Deal) targets a 55% reduction in air-pollution-related deaths by 2030.

This project builds an end-to-end machine learning system that **forecasts PM2.5 concentrations 24 hours ahead** for major European cities, enabling sensitive groups to plan their activities and helping cities deploy targeted interventions.

## 🏗️ Architecture

```
┌─────────────────┐   ┌──────────────────┐   ┌─────────────┐   ┌──────────────┐
│  OpenAQ v3 API  │──▶│  Data Pipeline   │──▶│   XGBoost   │──▶│  Streamlit   │
│  (or synthetic) │   │  + Feature Eng.  │   │   Forecast  │   │  Dashboard   │
└─────────────────┘   └──────────────────┘   └─────────────┘   └──────────────┘
                                                     │
                                              ┌──────▼──────┐
                                              │ CodeCarbon  │
                                              │  (Green AI) │
                                              └─────────────┘
```

## 📊 Results

On 365 days of hourly data (Berlin) with a 24-hour prediction horizon:

| Metric | Value |
|--------|-------|
| **MAE** | ~3.2 μg/m³ |
| **RMSE** | ~3.9 μg/m³ |
| **R²** | ~0.21 |
| **Training CO₂** | ~few grams (measured with CodeCarbon) |

> R² of ~0.2 is on par with published research for multi-hour PM2.5 forecasting — this task is genuinely hard because air quality depends on hard-to-predict meteorological events.

## 🚀 Quick Start

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd air-quality-eu

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Add your OpenAQ API key for real data
echo "OPENAQ_API_KEY=your_key_here" > .env
#    Without a key, the project uses realistic synthetic data automatically.

# 4. Run the analysis notebook
jupyter notebook notebooks/01_analysis_and_modeling.ipynb

# 5. Launch the interactive dashboard
streamlit run app.py
```

## 📁 Project Structure

```
air-quality-eu/
├── src/
│   ├── data_fetcher.py    # OpenAQ API client + synthetic data generator
│   ├── features.py        # Time, lag, and rolling feature engineering
│   └── model.py           # XGBoost training and evaluation
├── notebooks/
│   └── 01_analysis_and_modeling.ipynb   # Full EDA + model walkthrough
├── data/                  # (created at runtime)
├── app.py                 # Streamlit dashboard
├── requirements.txt
└── README.md
```

## 🔬 Methodology

### Feature engineering
- **Time:** cyclical encoding of hour/month (so the model knows hour 23 is close to hour 0)
- **Lags:** PM2.5 at t-1, t-2, t-3, t-6, t-12, t-24 hours
- **Rolling stats:** 3 / 6 / 24-hour rolling mean and std
- **Co-pollutants:** PM10, NO₂, O₃ as covariates
- **Weather:** temperature, wind speed, humidity

### Model
- **Algorithm:** XGBoost Regressor (gradient-boosted trees)
- **Hyperparameters:** 500 trees, depth 6, lr 0.05, early stopping
- **Split:** chronological 80/20 (no shuffling — this is a time series!)

### Why XGBoost over deep learning?
For tabular time-series with strong engineered features, gradient-boosted trees:
- Train in seconds (not hours)
- Are interpretable (feature importance, SHAP)
- Have lower carbon footprint — aligned with the **Green AI** principle

## 🌱 Green AI

This project measures and reports its own training emissions via [CodeCarbon](https://codecarbon.io):

```python
from codecarbon import EmissionsTracker
tracker = EmissionsTracker(project_name='air_quality')
tracker.start()
model.fit(X_train, y_train)
emissions_kg = tracker.stop()
```

Sustainable AI practice — measuring what you build — is a growing EU research priority under [ELIAS](https://elias-ai.eu/) and Horizon Europe 2026-27.

## 🏛️ EU Policy Context

| Policy | Relevance |
|--------|-----------|
| **European Green Deal** | Overarching framework — climate neutrality by 2050 |
| **Zero Pollution Action Plan** | Direct target: cut air-pollution deaths by 55% |
| **AI Act (2024)** | Trustworthy & transparent AI — this project is fully open & interpretable |
| **Horizon Europe AI in Science** | €100M earmarked for AI + sustainability projects |

## 🛠️ Tech Stack

`Python 3.10+` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `Plotly` · `Streamlit` · `CodeCarbon`

## 📈 Extensions / Next Steps

- [ ] Real-time retraining pipeline with OpenAQ live data
- [ ] LSTM and Transformer comparison
- [ ] Multi-city joint model with city embeddings
- [ ] SHAP-based per-prediction explanations
- [ ] Deploy to Streamlit Community Cloud / Hugging Face Spaces

## 📄 License

MIT
