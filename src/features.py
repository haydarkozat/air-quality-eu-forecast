"""
Feature Engineering for Air Quality Forecasting
================================================
Builds time-based, lag, and rolling features that a tree-based model
(XGBoost) can use to predict PM2.5 several hours ahead.
"""

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, dt_col: str = "datetime") -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[dt_col])
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    # Cyclical encoding so the model knows hour 23 is close to hour 0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_features(df: pd.DataFrame, target: str = "pm25",
                     lags=(1, 2, 3, 6, 12, 24)) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target: str = "pm25",
                         windows=(3, 6, 24)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"{target}_rollmean_{w}"] = df[target].shift(1).rolling(w).mean()
        df[f"{target}_rollstd_{w}"]  = df[target].shift(1).rolling(w).std()
    return df


def build_features(df: pd.DataFrame, target: str = "pm25",
                   horizon: int = 24) -> pd.DataFrame:
    """
    Build the full feature matrix.
    Predicts `target` `horizon` hours into the future.
    """
    df = df.sort_values("datetime").reset_index(drop=True)
    df = add_time_features(df)
    df = add_lag_features(df, target=target)
    df = add_rolling_features(df, target=target)

    # Prediction target: PM2.5 `horizon` hours from now
    df["y"] = df[target].shift(-horizon)

    # Drop rows with NaN created by lags / future shift
    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLS = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "dayofweek", "is_weekend",
    "pm10", "no2", "o3",
    "temperature", "wind_speed", "humidity",
    "pm25_lag_1", "pm25_lag_2", "pm25_lag_3",
    "pm25_lag_6", "pm25_lag_12", "pm25_lag_24",
    "pm25_rollmean_3", "pm25_rollmean_6", "pm25_rollmean_24",
    "pm25_rollstd_3", "pm25_rollstd_6", "pm25_rollstd_24",
]


def available_features(df: pd.DataFrame) -> list:
    """Return only the feature columns present in the dataframe."""
    return [c for c in FEATURE_COLS if c in df.columns]
