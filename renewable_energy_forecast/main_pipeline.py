"""
main_pipeline.py
-----------------
End-to-end renewable energy forecasting and optimization pipeline.

Stages
------
1. Data Generation  – simulate 2 years of hourly solar/weather/PV data
2. Weather Model    – train irradiance + PV + wind predictors
3. Time-Series Model– train SARIMA + XGBoost ensemble forecaster
4. Optimization     – run LP battery scheduler on a representative day
5. Allocation       – classify hours and generate recommendations
6. Visualization    – produce all charts
7. Report           – print and save a full text summary

Run:  python renewable_energy_forecast/main_pipeline.py
"""

import sys
import io
import warnings

# Force UTF-8 on Windows console so Unicode chars print safely
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# ── Path resolution ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from renewable_energy_forecast.data.data_generator        import generate_dataset
from renewable_energy_forecast.models.weather_predictor   import WeatherGenerationPredictor
from renewable_energy_forecast.models.time_series_forecaster import EnsembleForecaster
from renewable_energy_forecast.optimization.battery_optimizer import (
    BatteryOptimizer, BatteryConfig, OptimizationWeights,
)
from renewable_energy_forecast.optimization.energy_allocator  import EnergyAllocator
from renewable_energy_forecast.visualization.visualizer        import build_full_dashboard

OUT_DIR = ROOT / "renewable_energy_forecast" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Data Generation
# ══════════════════════════════════════════════════════════════════════════════

def stage_data(regenerate: bool = True) -> pd.DataFrame:
    print("\n" + "═"*60)
    print("  STAGE 1 │ Data Generation")
    print("═"*60)
    csv_path = ROOT / "renewable_energy_forecast" / "data" / "energy_dataset.csv"
    if csv_path.exists() and not regenerate:
        print(f"[Data] Loading existing dataset from {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    else:
        df = generate_dataset(output_path=str(csv_path))
    print(f"[Data] Shape: {df.shape}  |  Columns: {list(df.columns)}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Weather-Based Generation Prediction
# ══════════════════════════════════════════════════════════════════════════════

def stage_weather_model(df: pd.DataFrame) -> tuple[WeatherGenerationPredictor, dict]:
    print("\n" + "═"*60)
    print("  STAGE 2 │ Weather-Based Generation Prediction")
    print("═"*60)

    wp = WeatherGenerationPredictor()
    wp.fit(df)

    metrics = wp.all_metrics()
    print("\n[Weather Model] Metrics:")
    for name, m in metrics.items():
        print(f"  {name:20s} → RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}  R²={m['R2']:.4f}")

    # Predict on test slice
    n_test = int(len(df) * 0.20)
    df_test = df.iloc[-n_test:].copy()
    df_pred = wp.predict(df_test)

    wp.save(str(OUT_DIR / "weather_predictor.pkl"))
    return wp, metrics


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Time-Series Forecasting
# ══════════════════════════════════════════════════════════════════════════════

def stage_ts_model(df: pd.DataFrame) -> tuple[EnsembleForecaster, dict, np.ndarray, np.ndarray, np.ndarray]:
    print("\n" + "═"*60)
    print("  STAGE 3 │ Time-Series Forecasting (SARIMA + XGBoost Ensemble)")
    print("═"*60)

    ef = EnsembleForecaster(w_xgb=0.70, w_sarima=0.30)
    ef.fit(df, target_col="pv_output_kw")

    ts_metrics = ef.evaluate_on_test(df, target_col="pv_output_kw")
    print("\n[Time-Series] Test-set metrics:")
    for model_name, m in ts_metrics.items():
        print(f"  {model_name:12s} → RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}  R²={m['R2']:.4f}")

    # Prepare aligned test arrays for visualization
    n_test   = int(len(df) * 0.20)
    df_test  = df.iloc[-n_test:]
    max_lag  = max(ef.xgb.LAGS)
    y_true   = df_test["pv_output_kw"].values[max_lag:]
    xgb_pred = ef.xgb.predict(df_test)[:len(y_true)]
    sarima_p = ef.sarima.predict_hourly(n_days=len(y_true)//24 + 1)[:len(y_true)]
    ensemble = ef.w_xgb * xgb_pred + ef.w_sarima * sarima_p

    ef.save(str(OUT_DIR / "ensemble_forecaster.pkl"))
    return ef, ts_metrics, y_true, xgb_pred, ensemble


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Battery Optimization (24-hour window)
# ══════════════════════════════════════════════════════════════════════════════

def stage_optimization(
    df: pd.DataFrame,
    ef: EnsembleForecaster,
    target_date_str: str = "2023-06-15",
) -> pd.DataFrame:
    print("\n" + "═"*60)
    print("  STAGE 4 │ Battery Charge/Discharge Optimization")
    print("═"*60)

    # Select a representative day + 48h context for XGBoost lags
    target_date = pd.Timestamp(target_date_str)
    window_start = target_date - pd.Timedelta(hours=48)
    df_window = df[df["timestamp"] >= window_start].head(48 + 24).copy()

    # Get forecasts for the 24-hour target day
    df_day  = df[df["timestamp"].dt.date == target_date.date()].copy()
    if len(df_day) < 24:
        print("[Warning] Target date not fully available; using last 24 hours of dataset.")
        df_day = df.iloc[-24:].copy()

    pv_forecast   = df_day["pv_output_kw"].values[:24]
    wind_forecast = df_day["wind_output_kw"].values[:24]
    load_forecast = df_day["load_demand_kw"].values[:24]

    bat_config = BatteryConfig(
        capacity_kwh=40.0,
        max_charge_kw=10.0,
        max_discharge_kw=10.0,
        charge_eff=0.95,
        discharge_eff=0.95,
    )
    optimizer = BatteryOptimizer(battery=bat_config)
    schedule  = optimizer.optimize(pv_forecast, wind_forecast, load_forecast)

    kpis = optimizer.compute_kpis(schedule)
    print(f"\n[Optimizer] 24h KPIs for {target_date_str}:")
    for k, v in kpis.items():
        print(f"  {k:35s}: {v}")

    return schedule, bat_config


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — Energy Allocation & Recommendations
# ══════════════════════════════════════════════════════════════════════════════

def stage_allocation(
    schedule:  pd.DataFrame,
    bat_config,
) -> dict:
    print("\n" + "═"*60)
    print("  STAGE 5 │ Energy Allocation & Recommendations")
    print("═"*60)

    allocator   = EnergyAllocator()
    alloc_result = allocator.allocate(schedule,
                                      battery_capacity_kwh=bat_config.capacity_kwh)

    print("\n[Allocator] KPI Summary:")
    for k, v in alloc_result["kpi_summary"].items():
        print(f"  {k:40s}: {v}")

    print("\n[Allocator] Recommendations:")
    for rec in alloc_result["recommendations"]:
        print(f"  {rec}")

    return alloc_result


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — Visualization
# ══════════════════════════════════════════════════════════════════════════════

def stage_visualization(
    df:              pd.DataFrame,
    schedule:        pd.DataFrame,
    alloc_result:    dict,
    y_true:          np.ndarray,
    y_pred_xgb:      np.ndarray,
    y_pred_ensemble: np.ndarray,
    wp:              WeatherGenerationPredictor,
    ef:              EnsembleForecaster,
) -> dict:
    print("\n" + "═"*60)
    print("  STAGE 6 │ Visualization")
    print("═"*60)
    xgb_imp = ef.xgb.feature_importance()
    rf_imp  = wp.irradiance_model.feature_importances()

    saved = build_full_dashboard(
        df=df,
        schedule=schedule,
        alloc_result=alloc_result,
        y_true=y_true,
        y_pred_xgb=y_pred_xgb,
        y_pred_ensemble=y_pred_ensemble,
        xgb_importance=xgb_imp,
        rf_importance=rf_imp,
    )
    return saved


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — Text Report
# ══════════════════════════════════════════════════════════════════════════════

def stage_report(
    weather_metrics: dict,
    ts_metrics:      dict,
    alloc_result:    dict,
    saved_charts:    dict,
) -> None:
    print("\n" + "═"*60)
    print("  STAGE 7 │ Final Report")
    print("═"*60)

    lines = [
        "=" * 70,
        "  RENEWABLE ENERGY FORECASTING & OPTIMIZATION — SYSTEM REPORT",
        "=" * 70,
        "",
        "── Model Performance ───────────────────────────────────────────────",
        "",
        "  Weather-Based Prediction:",
    ]
    for name, m in weather_metrics.items():
        lines.append(f"    {name:25s} RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}  R²={m['R2']:.4f}")

    lines += ["", "  Time-Series Forecasting:"]
    for model_name, m in ts_metrics.items():
        lines.append(f"    {model_name:15s} RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}  R²={m['R2']:.4f}")

    kpis = alloc_result["kpi_summary"]
    lines += [
        "",
        "── 24-Hour Energy Allocation KPIs ──────────────────────────────────",
        "",
        f"    Total Load                 : {kpis['total_load_kwh']:.2f} kWh",
        f"    Total Renewable Generated  : {kpis['total_renewable_kwh']:.2f} kWh",
        f"    Grid Import (Fossil)       : {kpis['total_grid_import_kwh']:.2f} kWh",
        f"    Renewable Coverage         : {kpis['renewable_coverage_pct']:.1f} %",
        f"    Fossil Fraction            : {kpis['fossil_fraction_pct']:.1f} %",
        f"    Curtailed Energy           : {kpis['total_curtailed_kwh']:.2f} kWh",
        f"    Avg Self-Sufficiency       : {kpis['avg_self_sufficiency_pct']:.1f} %",
        f"    CO₂ Avoided                : {kpis['co2_avoided_kg']:.1f} kg",
        f"    Estimated Grid Cost        : ${kpis['estimated_cost_usd']:.2f} USD",
        "",
        "── Operator Recommendations ────────────────────────────────────────",
        "",
    ]
    for rec in alloc_result["recommendations"]:
        lines.append(f"    {rec}")

    lines += [
        "",
        "── Output Charts ───────────────────────────────────────────────────",
        "",
    ]
    for key, path in saved_charts.items():
        lines.append(f"    {key:22s} → {path}")

    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    report_path = OUT_DIR / "system_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\n[Report] Saved → {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    print("\n" + "="*62)
    print("   RENEWABLE ENERGY FORECASTING & OPTIMIZATION PIPELINE")
    print("="*62 + "\n")

    df                              = stage_data(regenerate=True)
    wp, weather_metrics             = stage_weather_model(df)
    ef, ts_metrics, y_true, xgb_p, ens_p = stage_ts_model(df)
    schedule, bat_config            = stage_optimization(df, ef)
    alloc_result                    = stage_allocation(schedule, bat_config)
    saved_charts                    = stage_visualization(
                                          df, schedule, alloc_result,
                                          y_true, xgb_p, ens_p, wp, ef)
    stage_report(weather_metrics, ts_metrics, alloc_result, saved_charts)

    print("\n✅ Pipeline complete. All outputs in:",
          OUT_DIR.resolve())


if __name__ == "__main__":
    run_pipeline()
