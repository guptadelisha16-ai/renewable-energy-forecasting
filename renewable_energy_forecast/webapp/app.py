"""
app.py  —  Flask web application for Renewable Energy Forecasting & Optimization
"""

import sys, io, os, json, warnings
warnings.filterwarnings("ignore")

from pathlib import Path

# ── Path resolution ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent   # c:\...\bharat
sys.path.insert(0, str(ROOT))

from flask import Flask, render_template, request, jsonify, send_from_directory

import numpy as np
import pandas as pd
import joblib

from renewable_energy_forecast.data.data_generator import generate_dataset
from renewable_energy_forecast.models.weather_predictor import WeatherGenerationPredictor
from renewable_energy_forecast.models.time_series_forecaster import EnsembleForecaster
from renewable_energy_forecast.optimization.battery_optimizer import (
    BatteryOptimizer, BatteryConfig, OptimizationWeights,
)
from renewable_energy_forecast.optimization.energy_allocator import EnergyAllocator

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = ROOT / "renewable_energy_forecast" / "data"
OUTPUTS_DIR = ROOT / "renewable_energy_forecast" / "outputs"
WEBAPP_DIR  = Path(__file__).parent

app = Flask(__name__, template_folder=str(WEBAPP_DIR / "templates"),
            static_folder=str(WEBAPP_DIR / "static"))

# ── Lazy-load models once ─────────────────────────────────────────────────────
_wp: WeatherGenerationPredictor | None = None
_ef: EnsembleForecaster | None = None
_df: pd.DataFrame | None = None

def get_dataset() -> pd.DataFrame:
    global _df
    if _df is None:
        csv = DATA_DIR / "energy_dataset.csv"
        _df = pd.read_csv(csv, parse_dates=["timestamp"])
    return _df

def get_weather_model() -> WeatherGenerationPredictor:
    global _wp
    if _wp is None:
        pkl = OUTPUTS_DIR / "weather_predictor.pkl"
        _wp = joblib.load(pkl)
    return _wp

def get_ts_model() -> EnsembleForecaster:
    global _ef
    if _ef is None:
        pkl = OUTPUTS_DIR / "ensemble_forecaster.pkl"
        _ef = joblib.load(pkl)
    return _ef


# ════════════════════════════════════════════════════════════════════════════════
# PAGE ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.route("/")
def dashboard():
    return render_template("dashboard.html", active="dashboard")

@app.route("/forecast")
def forecast():
    return render_template("forecast.html", active="forecast")

@app.route("/optimizer")
def optimizer():
    return render_template("optimizer.html", active="optimizer")

@app.route("/allocator")
def allocator():
    return render_template("allocator.html", active="allocator")

@app.route("/charts")
def charts():
    chart_files = sorted([f.name for f in OUTPUTS_DIR.glob("*.png")])
    return render_template("charts.html", active="charts", charts=chart_files)

@app.route("/dataset")
def dataset():
    return render_template("dataset.html", active="dataset")

@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(str(OUTPUTS_DIR), filename)


# ════════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ════════════════════════════════════════════════════════════════════════════════

# ── Dashboard KPIs ─────────────────────────────────────────────────────────────
@app.route("/api/dashboard_kpis")
def api_dashboard_kpis():
    df = get_dataset()
    total_pv   = df["pv_output_kw"].sum()
    total_wind = df["wind_output_kw"].sum()
    total_load = df["load_demand_kw"].sum()
    total_ren  = df["total_renewable_kw"].sum()
    avg_irr    = df["solar_irradiance_wm2"].mean()
    avg_temp   = df["temperature_c"].mean()
    avg_soc    = df["battery_soc_kwh"].mean()
    self_suff  = min((total_ren / total_load) * 100, 100)

    # Daily aggregates for sparklines (last 30 days)
    daily = (df.set_index("timestamp")
               .resample("D")[["pv_output_kw","wind_output_kw","load_demand_kw"]]
               .sum(numeric_only=True)
               .tail(30)
               .reset_index())

    return jsonify({
        "kpis": {
            "total_pv_mwh":       round(total_pv / 1000, 1),
            "total_wind_mwh":     round(total_wind / 1000, 1),
            "total_renewable_mwh":round(total_ren / 1000, 1),
            "avg_irradiance":     round(avg_irr, 1),
            "avg_temperature":    round(avg_temp, 1),
            "avg_battery_soc":    round(avg_soc, 1),
            "self_sufficiency":   round(self_suff, 1),
            "co2_avoided_tonnes": round(total_ren * 0.42 / 1000, 1),
        },
        "sparklines": {
            "dates":  daily["timestamp"].dt.strftime("%b %d").tolist(),
            "pv":     daily["pv_output_kw"].round(1).tolist(),
            "wind":   daily["wind_output_kw"].round(1).tolist(),
            "load":   daily["load_demand_kw"].round(1).tolist(),
        }
    })


# ── Dataset stats ──────────────────────────────────────────────────────────────
@app.route("/api/dataset_stats")
def api_dataset_stats():
    df = get_dataset()
    desc = df.describe(include="number").round(3).to_dict()
    preview = df.tail(48)[["timestamp","temperature_c","cloud_cover_pct",
                            "wind_speed_ms","solar_irradiance_wm2",
                            "pv_output_kw","wind_output_kw",
                            "load_demand_kw","battery_soc_kwh"]].copy()
    preview["timestamp"] = preview["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    return jsonify({
        "shape":   list(df.shape),
        "columns": list(df.columns),
        "describe": desc,
        "preview": preview.values.tolist(),
        "preview_cols": list(preview.columns),
    })


# ── Season hourly profile ──────────────────────────────────────────────────────
@app.route("/api/seasonal_profile")
def api_seasonal_profile():
    df = get_dataset()
    result = {}
    for season in ["Winter","Spring","Summer","Autumn"]:
        sub = df[df["season"] == season]
        h   = sub.groupby("hour")["pv_output_kw"].mean().round(3)
        result[season] = h.tolist()
    return jsonify(result)


# ── Weather-based prediction ───────────────────────────────────────────────────
@app.route("/api/predict_weather", methods=["POST"])
def api_predict_weather():
    body = request.get_json()
    hour         = int(body.get("hour", 12))
    month        = int(body.get("month", 6))
    temperature  = float(body.get("temperature", 30))
    cloud_cover  = float(body.get("cloud_cover", 20))
    wind_speed   = float(body.get("wind_speed", 5))
    humidity     = float(body.get("humidity", 50))

    doy = int((month - 1) * 30.4 + 15)
    row = pd.DataFrame([{
        "hour": hour, "month": month, "doy": doy,
        "temperature_c": temperature, "cloud_cover_pct": cloud_cover,
        "wind_speed_ms": wind_speed, "humidity_pct": humidity,
        "solar_irradiance_wm2": 0,   # will be predicted in stage 1
    }])

    wp  = get_weather_model()
    out = wp.predict(row)
    return jsonify({
        "pred_irradiance_wm2": round(float(out["pred_irradiance_wm2"].iloc[0]), 2),
        "pred_pv_kw":          round(float(out["pred_pv_kw"].iloc[0]),          3),
        "pred_wind_kw":        round(float(out["pred_wind_kw"].iloc[0]),        3),
        "pred_total_kw":       round(float(out["pred_total_kw"].iloc[0]),       3),
    })


# ── Time-series forecast (next 24h) ───────────────────────────────────────────
@app.route("/api/forecast_24h")
def api_forecast_24h():
    df = get_dataset()
    ef = get_ts_model()
    df_recent = df.tail(72).copy()

    xgb_p, sarima_p, ens_p = ef.predict_next_24h(df_recent, "pv_output_kw")
    hours = list(range(24))

    # Also get actuals for last 24h to show context
    actual_last = df["pv_output_kw"].tail(24).round(3).tolist()

    return jsonify({
        "hours":       hours,
        "xgb":         [round(float(v),3) for v in xgb_p],
        "sarima":      [round(float(v),3) for v in sarima_p],
        "ensemble":    [round(float(v),3) for v in ens_p],
        "actual_last": actual_last,
    })


# ── Battery optimization ───────────────────────────────────────────────────────
@app.route("/api/optimize_battery", methods=["POST"])
def api_optimize_battery():
    body = request.get_json() or {}
    date_str = body.get("date", "2023-06-15")
    cap      = float(body.get("capacity_kwh", 40))
    max_ch   = float(body.get("max_charge_kw", 10))
    max_dis  = float(body.get("max_discharge_kw", 10))

    df = get_dataset()
    ts = pd.Timestamp(date_str)
    df_day = df[df["timestamp"].dt.date == ts.date()].copy()
    if len(df_day) < 24:
        df_day = df.tail(24).copy()

    pv_fc   = df_day["pv_output_kw"].values[:24]
    wind_fc = df_day["wind_output_kw"].values[:24]
    load_fc = df_day["load_demand_kw"].values[:24]

    bat  = BatteryConfig(capacity_kwh=cap, max_charge_kw=max_ch, max_discharge_kw=max_dis)
    opt  = BatteryOptimizer(battery=bat)
    sched= opt.optimize(pv_fc, wind_fc, load_fc)
    kpis = opt.compute_kpis(sched)

    return jsonify({
        "schedule": sched.to_dict(orient="list"),
        "kpis":     kpis,
    })


# ── Energy allocation ──────────────────────────────────────────────────────────
@app.route("/api/allocate", methods=["POST"])
def api_allocate():
    body    = request.get_json() or {}
    date_str = body.get("date", "2023-06-15")
    cap     = float(body.get("capacity_kwh", 40))

    df = get_dataset()
    ts = pd.Timestamp(date_str)
    df_day = df[df["timestamp"].dt.date == ts.date()].copy()
    if len(df_day) < 24:
        df_day = df.tail(24).copy()

    bat  = BatteryConfig(capacity_kwh=cap)
    opt  = BatteryOptimizer(battery=bat)
    sched= opt.optimize(df_day["pv_output_kw"].values[:24],
                         df_day["wind_output_kw"].values[:24],
                         df_day["load_demand_kw"].values[:24])

    alloc = EnergyAllocator()
    result = alloc.allocate(sched, battery_capacity_kwh=cap)

    return jsonify({
        "kpis":            result["kpi_summary"],
        "recommendations": result["recommendations"],
        "hourly":          result["hourly_allocation"].to_dict(orient="list"),
        "dispatch":        result["priority_dispatch"].to_dict(orient="list"),
    })


# ── Model metrics ──────────────────────────────────────────────────────────────
@app.route("/api/model_metrics")
def api_model_metrics():
    wp = get_weather_model()
    ef = get_ts_model()
    return jsonify({
        "weather": wp.all_metrics(),
        "timeseries": {
            "xgb_val": ef.metrics_xgb,
            "ensemble": ef.metrics_ensemble,
        }
    })


# ── System report text ─────────────────────────────────────────────────────────
@app.route("/api/report")
def api_report():
    rpt = OUTPUTS_DIR / "system_report.txt"
    return jsonify({"report": rpt.read_text(encoding="utf-8") if rpt.exists() else "Report not found."})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
