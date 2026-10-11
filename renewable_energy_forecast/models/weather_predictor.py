"""
weather_predictor.py
---------------------
Weather-based renewable energy generation prediction.

Uses a two-stage pipeline:
  Stage 1 – Random Forest predicts solar irradiance from weather inputs.
  Stage 2 – Gradient Boosted Trees (XGBoost) maps irradiance + weather → PV output.

Also includes:
  - A wind power predictor (XGBoost).
  - A WeatherGenerationPredictor façade that combines both.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Tuple

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


# ── Helpers ────────────────────────────────────────────────────────────────────

def _evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    metrics = {"model": name, "RMSE": round(rmse,4), "MAE": round(mae,4), "R2": round(r2,4)}
    print(f"  [{name}]  RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.4f}")
    return metrics


def _cyclic_encode(val: np.ndarray, period: float) -> Tuple[np.ndarray, np.ndarray]:
    return np.sin(2 * np.pi * val / period), np.cos(2 * np.pi * val / period)


# ── Feature engineering ────────────────────────────────────────────────────────

IRRADIANCE_FEATURES = [
    "hour", "doy", "month",
    "temperature_c", "humidity_pct", "cloud_cover_pct",
    "wind_speed_ms",
]

PV_FEATURES = [
    "solar_irradiance_wm2", "temperature_c", "cloud_cover_pct",
    "humidity_pct", "hour", "doy", "month",
]

WIND_FEATURES = [
    "wind_speed_ms", "temperature_c", "humidity_pct",
    "hour", "doy", "month",
]


def _build_features(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Build feature matrix with cyclic hour / doy / month encoding."""
    parts = []
    for col in feature_cols:
        if col == "hour":
            s, c = _cyclic_encode(df["hour"].values, 24)
            parts.extend([s, c])
        elif col == "doy":
            s, c = _cyclic_encode(df["doy"].values, 365)
            parts.extend([s, c])
        elif col == "month":
            s, c = _cyclic_encode(df["month"].values, 12)
            parts.extend([s, c])
        else:
            parts.append(df[col].values)
    return np.column_stack(parts)


# ── Stage-1: Solar Irradiance Predictor ───────────────────────────────────────

class IrradiancePredictor:
    """
    Random Forest that predicts solar irradiance (W/m²) from weather inputs.
    Captures non-linear cloud / humidity / temperature interactions.
    """

    def __init__(self, n_estimators: int = 200):
        self.pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=12,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=42,
            )),
        ])
        self.metrics: Dict = {}

    def fit(self, df: pd.DataFrame) -> "IrradiancePredictor":
        X = _build_features(df, IRRADIANCE_FEATURES)
        y = df["solar_irradiance_wm2"].values

        n_train = int(len(X) * 0.80)
        X_tr, X_val = X[:n_train], X[n_train:]
        y_tr, y_val = y[:n_train], y[n_train:]

        print("[IrradiancePredictor] Fitting Random Forest …")
        self.pipe.fit(X_tr, y_tr)
        y_pred = np.maximum(self.pipe.predict(X_val), 0)
        self.metrics = _evaluate("IrradianceRF", y_val, y_pred)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = _build_features(df, IRRADIANCE_FEATURES)
        return np.maximum(self.pipe.predict(X), 0)

    def feature_importances(self) -> pd.Series:
        rf    = self.pipe.named_steps["rf"]
        names = []
        for col in IRRADIANCE_FEATURES:
            if col in ("hour", "doy", "month"):
                names.extend([f"{col}_sin", f"{col}_cos"])
            else:
                names.append(col)
        return pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)


# ── Stage-2: PV Output Predictor ──────────────────────────────────────────────

class PVOutputPredictor:
    """
    XGBoost model predicting PV power (kW) from irradiance + weather.
    Temperature de-rating and cloud shading are implicitly learned.
    """

    def __init__(self):
        self.pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )),
        ])
        self.metrics: Dict = {}

    def fit(self, df: pd.DataFrame) -> "PVOutputPredictor":
        X = _build_features(df, PV_FEATURES)
        y = df["pv_output_kw"].values

        n_train = int(len(X) * 0.80)
        X_tr, X_val = X[:n_train], X[n_train:]
        y_tr, y_val = y[:n_train], y[n_train:]

        print("[PVOutputPredictor] Fitting XGBoost …")
        self.pipe.fit(X_tr, y_tr)
        y_pred = np.maximum(self.pipe.predict(X_val), 0)
        self.metrics = _evaluate("PV_XGBoost", y_val, y_pred)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = _build_features(df, PV_FEATURES)
        return np.maximum(self.pipe.predict(X), 0)

    def feature_importances(self) -> pd.Series:
        xgb   = self.pipe.named_steps["xgb"]
        names = []
        for col in PV_FEATURES:
            if col in ("hour", "doy", "month"):
                names.extend([f"{col}_sin", f"{col}_cos"])
            else:
                names.append(col)
        return pd.Series(xgb.feature_importances_, index=names).sort_values(ascending=False)


# ── Wind Power Predictor ───────────────────────────────────────────────────────

class WindOutputPredictor:
    """
    Gradient Boosted Trees predicting wind turbine output (kW).
    Learns the cut-in / rated / cut-out power curve implicitly.
    """

    def __init__(self):
        self.pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("gb", GradientBoostingRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )),
        ])
        self.metrics: Dict = {}

    def fit(self, df: pd.DataFrame) -> "WindOutputPredictor":
        X = _build_features(df, WIND_FEATURES)
        y = df["wind_output_kw"].values

        n_train = int(len(X) * 0.80)
        X_tr, X_val = X[:n_train], X[n_train:]
        y_tr, y_val = y[:n_train], y[n_train:]

        print("[WindOutputPredictor] Fitting Gradient Boosting …")
        self.pipe.fit(X_tr, y_tr)
        y_pred = np.maximum(self.pipe.predict(X_val), 0)
        self.metrics = _evaluate("Wind_GB", y_val, y_pred)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = _build_features(df, WIND_FEATURES)
        return np.maximum(self.pipe.predict(X), 0)


# ── Façade: WeatherGenerationPredictor ────────────────────────────────────────

class WeatherGenerationPredictor:
    """
    Full two-stage pipeline:
      weather → irradiance (RF) → PV output (XGBoost)
      weather → wind output (GBT)

    Public API
    ----------
    .fit(df)          : train all three sub-models
    .predict(df)      : return DataFrame with predicted columns
    .all_metrics()    : dict of all sub-model metrics
    .save / .load     : joblib persistence
    """

    def __init__(self):
        self.irradiance_model = IrradiancePredictor()
        self.pv_model         = PVOutputPredictor()
        self.wind_model       = WindOutputPredictor()

    def fit(self, df: pd.DataFrame) -> "WeatherGenerationPredictor":
        print("\n── Stage 1: Irradiance Prediction ─────────────────────")
        self.irradiance_model.fit(df)

        # Enrich df with predicted irradiance for Stage 2
        df = df.copy()
        df["solar_irradiance_wm2"] = self.irradiance_model.predict(df)

        print("\n── Stage 2: PV Output Prediction ──────────────────────")
        self.pv_model.fit(df)

        print("\n── Stage 3: Wind Output Prediction ────────────────────")
        self.wind_model.fit(df)

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : DataFrame with weather columns (no PV/wind actuals needed).

        Returns
        -------
        DataFrame with added columns:
          pred_irradiance_wm2, pred_pv_kw, pred_wind_kw, pred_total_kw
        """
        out = df.copy()
        out["pred_irradiance_wm2"] = self.irradiance_model.predict(df)

        df_enr = df.copy()
        df_enr["solar_irradiance_wm2"] = out["pred_irradiance_wm2"]
        out["pred_pv_kw"]   = self.pv_model.predict(df_enr)
        out["pred_wind_kw"] = self.wind_model.predict(df)
        out["pred_total_kw"] = out["pred_pv_kw"] + out["pred_wind_kw"]
        return out

    def all_metrics(self) -> Dict:
        return {
            "irradiance_RF": self.irradiance_model.metrics,
            "pv_XGBoost":    self.pv_model.metrics,
            "wind_GB":       self.wind_model.metrics,
        }

    def cross_validate_pv(self, df: pd.DataFrame, cv: int = 3) -> Dict:
        """5-fold CV on PV output prediction."""
        X = _build_features(df, PV_FEATURES)
        y = df["pv_output_kw"].values
        xgb = self.pv_model.pipe
        kf  = KFold(n_splits=cv, shuffle=False)
        scores = cross_val_score(xgb, X, y,
                                 cv=kf, scoring="r2", n_jobs=-1)
        return {"cv_r2_mean": round(scores.mean(), 4),
                "cv_r2_std":  round(scores.std(),  4)}

    def save(self, path: str = "renewable_energy_forecast/outputs/weather_predictor.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"[WeatherGenerationPredictor] Saved → {path}")

    @staticmethod
    def load(path: str) -> "WeatherGenerationPredictor":
        return joblib.load(path)
