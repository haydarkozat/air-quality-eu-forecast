"""
PM2.5 Forecasting Model
=======================
XGBoost regressor with time-series-aware train/test split.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def time_split(df: pd.DataFrame, test_size: float = 0.2):
    """Chronological train/test split (no shuffling — this is a time series)."""
    n = len(df)
    split = int(n * (1 - test_size))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def train_model(X_train, y_train, X_val=None, y_val=None, **kwargs):
    """Train an XGBoost regressor with sensible defaults."""
    params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30 if X_val is not None else None,
    )
    params.update(kwargs)

    model = xgb.XGBRegressor(**params)

    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
    else:
        model.fit(X_train, y_train, verbose=False)

    return model


def evaluate(model, X_test, y_test) -> dict:
    """Return MAE, RMSE, R² and a sample of predictions vs actuals."""
    y_pred = model.predict(X_test)
    return {
        "MAE":  float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2":   float(r2_score(y_test, y_pred)),
        "y_true": np.asarray(y_test),
        "y_pred": y_pred,
    }


def feature_importance(model, feature_names) -> pd.DataFrame:
    """Return a sorted dataframe of feature importances."""
    imp = model.feature_importances_
    return (pd.DataFrame({"feature": feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))
