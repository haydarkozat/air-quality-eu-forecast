"""
Air Quality Forecasting Dashboard
==================================
Interactive Streamlit dashboard for European city air quality predictions.

Run with:  streamlit run app.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_fetcher import get_data, EU_CITIES
from src.features import build_features, available_features
from src.model import time_split, train_model, evaluate

# ----------------------------------------------------------------------
# Page config & styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="EU Air Quality Forecast",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem; font-weight: 700;
        background: linear-gradient(90deg, #2E8B57, #4682B4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle { color: #5a6c7d; font-size: 1.1rem; margin-top: 0; }
    .metric-card {
        background: white; padding: 1.2rem; border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-left: 4px solid #4682B4;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🌍 European Air Quality Forecast</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered PM2.5 predictions for major European cities · Aligned with the EU Green Deal</p>',
            unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Sidebar — configuration
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    city = st.selectbox("City", list(EU_CITIES.keys()), index=0)
    days = st.slider("Days of historical data", 30, 365, 180)
    horizon = st.select_slider("Prediction horizon (hours)",
                                options=[1, 3, 6, 12, 24, 48], value=24)

    st.divider()
    st.markdown("**Data source**")
    use_real = st.checkbox("Try OpenAQ live data", value=True,
                            help="Requires OPENAQ_API_KEY env variable. Falls back to synthetic data automatically.")

    st.divider()
    st.markdown("### About")
    st.info("""
    Project for the **EU Zero Pollution Action Plan** context.
    Predicts PM2.5 several hours ahead using XGBoost on weather + lag features.

    Built with Python, XGBoost, Streamlit.
    """)


# ----------------------------------------------------------------------
# Data + model (cached so we don't re-fetch on every interaction)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_data(city: str, days: int, prefer_real: bool):
    return get_data(city, days=days, prefer_real=prefer_real)


@st.cache_resource(show_spinner=False)
def train(city: str, days: int, horizon: int, prefer_real: bool):
    df = load_data(city, days, prefer_real)
    feat_df = build_features(df, target="pm25", horizon=horizon)
    feats = available_features(feat_df)
    train_df, test_df = time_split(feat_df, test_size=0.2)
    model = train_model(train_df[feats], train_df["y"],
                        test_df[feats], test_df["y"])
    metrics = evaluate(model, test_df[feats], test_df["y"])
    return df, feat_df, feats, train_df, test_df, model, metrics


with st.spinner(f"📡 Loading data for {city} and training model..."):
    df, feat_df, feats, train_df, test_df, model, metrics = train(
        city, days, horizon, use_real
    )

st.success(f"✓ Loaded {len(df):,} hourly observations for **{city}** · Model trained on {len(train_df):,} samples.")


# ----------------------------------------------------------------------
# Top metrics row
# ----------------------------------------------------------------------
latest_pm25 = df["pm25"].iloc[-1]
latest_dt = pd.to_datetime(df["datetime"].iloc[-1]).strftime("%Y-%m-%d %H:%M UTC")
who_limit = 15  # μg/m³ annual

# Predict the next horizon
def predict_next():
    last_row = feat_df.iloc[[-1]][feats]
    return float(model.predict(last_row)[0])

next_pred = predict_next()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Current PM2.5", f"{latest_pm25:.1f} μg/m³",
              delta=f"{latest_pm25 - who_limit:+.1f} vs WHO limit",
              delta_color="inverse")
with col2:
    st.metric(f"Forecast (+{horizon}h)", f"{next_pred:.1f} μg/m³",
              delta=f"{next_pred - latest_pm25:+.1f} vs now",
              delta_color="inverse")
with col3:
    st.metric("Model MAE", f"{metrics['MAE']:.2f} μg/m³",
              help="Mean absolute error on the hold-out test set")
with col4:
    st.metric("Model R²", f"{metrics['R2']:.3f}",
              help="Coefficient of determination — higher is better")


# Health advisory based on PM2.5 levels (EU AQI thresholds)
def health_advisory(pm: float):
    if pm < 10:   return "🟢 Good", "Air quality is excellent. No precautions needed."
    if pm < 20:   return "🟡 Fair", "Air quality is acceptable for most people."
    if pm < 25:   return "🟠 Moderate", "Sensitive groups should consider limiting prolonged outdoor exertion."
    if pm < 50:   return "🔴 Poor", "Sensitive groups should avoid outdoor activities. General public should reduce exposure."
    return "🟣 Very Poor", "Everyone should avoid outdoor activities. Use air purifiers indoors."

status, advice = health_advisory(next_pred)
st.info(f"**Health advisory ({horizon}h ahead):** {status} — {advice}")


# ----------------------------------------------------------------------
# Main tabs
# ----------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "🔍 Historical Patterns",
                                    "🧠 Model Insights", "ℹ️ About"])

with tab1:
    st.subheader(f"PM2.5 Forecast vs Actual — {city}")

    # Build prediction series for the test set
    y_pred = metrics["y_pred"]
    test_plot = test_df.copy()
    test_plot["prediction"] = y_pred
    test_plot["target_datetime"] = test_plot["datetime"] + pd.Timedelta(hours=horizon)

    n_show = st.slider("Show last N hours", 48, min(720, len(test_plot)), 168, key="forecast_n")
    plot_df = test_plot.tail(n_show)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["target_datetime"], y=plot_df["y"],
                              mode="lines", name="Actual",
                              line=dict(color="#2c3e50", width=2)))
    fig.add_trace(go.Scatter(x=plot_df["target_datetime"], y=plot_df["prediction"],
                              mode="lines", name="Predicted",
                              line=dict(color="#e74c3c", width=2, dash="dash")))
    fig.add_hline(y=15, line_dash="dot", line_color="orange",
                  annotation_text="WHO annual limit (15)")
    fig.add_hline(y=25, line_dash="dot", line_color="red",
                  annotation_text="EU daily limit (25)")
    fig.update_layout(
        xaxis_title="Date", yaxis_title="PM2.5 (μg/m³)",
        height=450, hovermode="x unified",
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Scatter
    st.subheader("Prediction Accuracy")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=metrics["y_true"], y=metrics["y_pred"],
                               mode="markers",
                               marker=dict(size=4, opacity=0.5, color="#3498db"),
                               name="Predictions"))
    mx = max(float(metrics["y_true"].max()), float(metrics["y_pred"].max()))
    fig2.add_trace(go.Scatter(x=[0, mx], y=[0, mx],
                               mode="lines", line=dict(dash="dash", color="red"),
                               name="Perfect prediction"))
    fig2.update_layout(xaxis_title="Actual PM2.5 (μg/m³)",
                       yaxis_title="Predicted PM2.5 (μg/m³)",
                       height=400, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)


with tab2:
    st.subheader(f"Temporal Patterns — {city}")

    df_h = df.copy()
    df_h["hour"] = pd.to_datetime(df_h["datetime"]).dt.hour
    df_h["dayofweek"] = pd.to_datetime(df_h["datetime"]).dt.day_name()
    df_h["month"] = pd.to_datetime(df_h["datetime"]).dt.month_name()

    c1, c2 = st.columns(2)
    with c1:
        hourly = df_h.groupby("hour")["pm25"].mean().reset_index()
        fig = px.line(hourly, x="hour", y="pm25", markers=True,
                       title="Average PM2.5 by Hour of Day",
                       labels={"pm25": "PM2.5 (μg/m³)", "hour": "Hour"})
        fig.update_traces(line_color="#e74c3c", line_width=3)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        weekly = df_h.groupby("dayofweek")["pm25"].mean().reindex(day_order).reset_index()
        fig = px.bar(weekly, x="dayofweek", y="pm25",
                      title="Average PM2.5 by Day of Week",
                      labels={"pm25": "PM2.5 (μg/m³)", "dayofweek": ""})
        fig.update_traces(marker_color="#3498db")
        st.plotly_chart(fig, use_container_width=True)

    # Correlation
    st.subheader("Variable Correlations")
    available_cols = [c for c in ["pm25", "pm10", "no2", "o3",
                                    "temperature", "wind_speed", "humidity"]
                       if c in df.columns]
    corr = df[available_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                     zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)


with tab3:
    st.subheader("Feature Importance")
    from src.model import feature_importance
    imp = feature_importance(model, feats).head(15)

    fig = px.bar(imp.iloc[::-1], x="importance", y="feature",
                  orientation="h", color="importance",
                  color_continuous_scale="Viridis")
    fig.update_layout(height=500, showlegend=False,
                       coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Interpretation:**
    - Recent lag values of PM2.5 (1-3 hours ago) are usually the strongest predictors —
      air quality has strong temporal autocorrelation.
    - Cyclical hour/month features capture diurnal and seasonal patterns.
    - Weather variables (wind, humidity) play a secondary but meaningful role.
    """)

    with st.expander("📊 Model details"):
        st.json({
            "algorithm": "XGBoost Regressor",
            "n_estimators": int(model.n_estimators),
            "max_depth": int(model.max_depth) if model.max_depth else None,
            "learning_rate": float(model.learning_rate),
            "n_features": len(feats),
            "n_train_samples": len(train_df),
            "n_test_samples": len(test_df),
            "test_MAE": round(metrics["MAE"], 3),
            "test_RMSE": round(metrics["RMSE"], 3),
            "test_R2": round(metrics["R2"], 3),
        })


with tab4:
    st.markdown("""
    ### 🌍 EU Air Quality Forecasting Project

    This dashboard predicts **PM2.5 fine particulate matter** concentrations
    several hours ahead for major European cities using machine learning.

    #### Why this matters
    - Air pollution causes **300,000+ premature deaths** in the EU annually.
    - The **EU Zero Pollution Action Plan** targets a 55% reduction in
      premature deaths from air pollution by 2030.
    - Early warnings let sensitive groups (children, elderly, asthmatics)
      adjust their activities.

    #### How it works
    1. **Data:** OpenAQ v3 API (or realistic synthetic fallback)
    2. **Features:** lag values, rolling statistics, cyclical time features, weather
    3. **Model:** XGBoost regressor with chronological train/test split
    4. **Green AI:** Training emissions tracked with CodeCarbon

    #### Tech stack
    `Python` · `Pandas` · `XGBoost` · `Plotly` · `Streamlit` · `CodeCarbon`

    ---
    Built as a portfolio project demonstrating end-to-end ML engineering
    in the context of the **EU Green Deal**.
    """)
