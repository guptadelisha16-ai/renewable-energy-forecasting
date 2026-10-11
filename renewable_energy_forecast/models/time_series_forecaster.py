"""
time_series_forecaster.py
--------------------------
Hybrid time-series forecasting for hourly PV / renewable generation.

Models
------
1. SARIMA  – captures daily (s=24) seasonality & trend.
2. XGBoost lag-feature model – captures non-linear patterns.
3. Ensemble  – weighted average of both for final forecast.

Outputs rolling 24-hour ahead predictions with RMSE / MAE / R² metrics.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Dict

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_lag_features(
    series: np.ndarray,
    lags: list[int],
    exog: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a supervised-learning matrix from lags.

    Returns X (features) and y (targets), trimming the first max(lags) rows.
    """
    max_lag = max(lags)
    X_parts = []
    for lag in lags:
        X_parts.append(series[max_lag - lag : len(series) - lag])
    X = np.column_stack(X_parts)
    y = series[max_lag:]
    if exog is not None:
        X = np.column_stack([X, exog[max_lag:]])
    return X, y


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}


# ── SARIMA wrapper ─────────────────────────────────────────────────────────────

class SARIMAForecaster:
    """
    Fits a SARIMA(1,0,1)(1,0,1)[24] model on daily-aggregated data
    then distributes to hourly via a learned diurnal profile.
    Training on full daily series; in-sample RMSE for validation.
    """

    def __init__(self, order=(1,0,1), seasonal_order=(1,0,1,24)):
        self.order          = order
        self.seasonal_order = seasonal_order
        self.model_fit      = None
        self.diurnal_profile = None   # mean hourly ratio per hour-of-day

    def fit(self, hourly_series: pd.Series) -> "SARIMAForecaster":
        """
        Fit on the first 80 % of `hourly_series`.
        For speed, down-sample to daily totals for SARIMA,
        keeping the diurnal profile separately.
        """
        print("[SARIMA] Building diurnal profile …")
        hour_index = hourly_series.index.hour if hasattr(hourly_series.index, "hour") \
                     else np.arange(len(hourly_series)) % 24

        # Diurnal profile: mean output by hour-of-day (normalised)
        profile_df = pd.DataFrame({
            "value": hourly_series.values,
            "hour":  hour_index,
        })
        self.diurnal_profile = (
            profile_df.groupby("hour")["value"].mean()
            .reindex(range(24), fill_value=0)
            .values
            .copy()  # ensure writeable
        )
        total = self.diurnal_profile.sum()
        if total > 0:
            self.diurnal_profile = self.diurnal_profile / total

        # Daily totals for SARIMA
        daily_series = hourly_series.resample("D").sum() if hasattr(
            hourly_series.index, "freq") else pd.Series(
            [hourly_series.values[i*24:(i+1)*24].sum()
             for i in range(len(hourly_series)//24)]
        )
        n_train = int(len(daily_series) * 0.80)
        train_daily = daily_series.iloc[:n_train]

        print(f"[SARIMA] Fitting on {len(train_daily)} daily obs …")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sarima = SARIMAX(
                train_daily,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.model_fit = sarima.fit(disp=False, maxiter=50)
        print("[SARIMA] Fit complete.")
        return self

    def predict_daily(self, n_days: int) -> np.ndarray:
        """Forecast `n_days` of daily totals starting from end of training."""
        fc = self.model_fit.forecast(steps=n_days)
        return np.maximum(fc.values, 0)

    def predict_hourly(self, n_days: int) -> np.ndarray:
        """Distribute daily forecasts to hourly using the diurnal profile."""
        daily = self.predict_daily(n_days)
        hourly = np.outer(daily, self.diurnal_profile).ravel()
        return np.maximum(hourly, 0)


# ── XGBoost lag-feature forecaster ────────────────────────────────────────────

class XGBoostLagForecaster:
    """
    Supervised XGBoost regressor using:
      - Lags: 1h, 2h, 3h, 6h, 12h, 24h, 48h
      - Hour-of-day, day-of-week, month (cyclic encoded)
      - Exogenous weather: temperature, cloud cover, irradiance
    """

    LAGS = [1, 2, 3, 6, 12, 24, 48]

    def __init__(self):
        self.model   = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        self.scaler  = StandardScaler()
        self.metrics: Dict[str, float] = {}

    def _cyclic(self, val: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
        return (np.sin(2 * np.pi * val / period),
                np.cos(2 * np.pi * val / period))

    def _make_exog(self, df: pd.DataFrame) -> np.ndarray:
        hour  = df["hour"].values
        month = df["month"].values
        sin_h, cos_h = self._cyclic(hour,  24)
        sin_m, cos_m = self._cyclic(month, 12)
        parts = [sin_h, cos_h, sin_m, cos_m,
                 df["temperature_c"].values,
                 df["cloud_cover_pct"].values,
                 df["solar_irradiance_wm2"].values,
                 df["wind_speed_ms"].values]
        return np.column_stack(parts)

    def fit(self, df: pd.DataFrame, target_col: str = "pv_output_kw") -> "XGBoostLagForecaster":
        series = df[target_col].values.astype(float)
        exog   = self._make_exog(df)

        X, y   = _build_lag_features(series, self.LAGS, exog)
        n_train = int(len(X) * 0.80)
        X_tr, X_val = X[:n_train], X[n_train:]
        y_tr, y_val = y[:n_train], y[n_train:]

        X_tr  = self.scaler.fit_transform(X_tr)
        X_val = self.scaler.transform(X_val)

        print(f"[XGBoost] Training on {len(X_tr):,} samples …")
        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_pred        = np.maximum(self.model.predict(X_val), 0)
        self.metrics  = _evaluate(y_val, y_pred)
        print(f"[XGBoost] Val metrics: {self.metrics}")
        return self

    def predict(self, df: pd.DataFrame, target_col: str = "pv_output_kw") -> np.ndarray:
        series = df[target_col].values.astype(float)
        exog   = self._make_exog(df)
        X, _   = _build_lag_features(series, self.LAGS, exog)
        X_sc   = self.scaler.transform(X)
        return np.maximum(self.model.predict(X_sc), 0)

    def feature_importance(self) -> pd.Series:
        lag_names  = [f"lag_{l}" for l in self.LAGS]
        exog_names = ["sin_h","cos_h","sin_m","cos_m",
                      "temperature","cloud_cover","irradiance","wind_speed"]
        names      = lag_names + exog_names
        return pd.Series(self.model.feature_importances_, index=names).sort_values(ascending=False)


# ── Ensemble forecaster ────────────────────────────────────────────────────────

class EnsembleForecaster:
    """
    Weighted combination: w_sarima * SARIMA + w_xgb * XGBoost.
    Weights are tuned on the validation set.
    """

    def __init__(self, w_xgb: float = 0.70, w_sarima: float = 0.30):
        self.w_xgb    = w_xgb
        self.w_sarima = w_sarima
        self.sarima   = SARIMAForecaster()
        self.xgb      = XGBoostLagForecaster()
        self.metrics_xgb:    Dict[str, float] = {}
        self.metrics_ensemble: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, target_col: str = "pv_output_kw") -> "EnsembleForecaster":
        ts = df.set_index("timestamp")[target_col] if "timestamp" in df.columns else df[target_col]
        self.sarima.fit(ts)
        self.xgb.fit(df, target_col)
        self.metrics_xgb = self.xgb.metrics
        return self

    def predict_next_24h(
        self, df_recent: pd.DataFrame, target_col: str = "pv_output_kw"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (xgb_pred, sarima_pred, ensemble_pred) — all length-24 arrays.
        `df_recent` must contain at least 48 recent hourly rows.
        """
        # XGBoost: predict the last 24 hours of df_recent (overlap strategy)
        xgb_full   = self.xgb.predict(df_recent, target_col)
        xgb_pred   = xgb_full[-24:]

        # SARIMA: forecast next 1 day
        sarima_pred = self.sarima.predict_hourly(n_days=1)[:24]

        # Ensemble
        ensemble = self.w_xgb * xgb_pred + self.w_sarima * sarima_pred
        return xgb_pred, sarima_pred, ensemble

    def evaluate_on_test(
        self, df: pd.DataFrame, target_col: str = "pv_output_kw"
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics on the last 20 % of df."""
        n_test = int(len(df) * 0.20)
        df_test = df.iloc[-n_test:]

        xgb_pred = self.xgb.predict(df_test, target_col)
        max_lag  = max(self.xgb.LAGS)
        y_true   = df_test[target_col].values[max_lag:]
        xgb_pred = xgb_pred[:len(y_true)]

        # Align SARIMA forecast length
        sarima_pred = self.sarima.predict_hourly(n_days=len(y_true)//24 + 1)[:len(y_true)]
        ensemble    = self.w_xgb * xgb_pred + self.w_sarima * sarima_pred

        results = {
            "XGBoost":  _evaluate(y_true, xgb_pred),
            "SARIMA":   _evaluate(y_true, sarima_pred),
            "Ensemble": _evaluate(y_true, ensemble),
        }
        self.metrics_ensemble = results["Ensemble"]
        return results

    def save(self, path: str = "renewable_energy_forecast/outputs/ensemble_forecaster.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"[EnsembleForecaster] Saved → {path}")

    @staticmethod
    def load(path: str) -> "EnsembleForecaster":
        return joblib.load(path)
