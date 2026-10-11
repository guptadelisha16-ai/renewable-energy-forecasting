"""
visualizer.py
--------------
Comprehensive visualization dashboard for the renewable energy forecasting system.

Charts produced
---------------
1.  Dataset Overview      – time-series of all key variables (2 years)
2.  Seasonal Decomposition – solar output by season
3.  Weather Correlation    – heatmap of feature correlations
4.  Forecast vs Actual     – time-series comparison (test window)
5.  Feature Importance     – XGBoost & RF bar charts
6.  Battery Schedule       – 24-hour charge/discharge/SOC stacked chart
7.  Energy Allocation      – source breakdown pie + hourly stacked bar
8.  KPI Dashboard          – gauge-style summary panel
9.  Wind Power Curve       – scatter of wind speed vs output
10. Error Distribution     – residual histogram & Q-Q plot
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for all envs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional


# ── Style ──────────────────────────────────────────────────────────────────────
PALETTE    = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
              "#44BBA4", "#E94F37", "#393E41"]
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor": "#F7F8FA",
    "axes.facecolor":   "#FFFFFF",
    "axes.edgecolor":   "#E5E7EB",
    "grid.color":       "#E5E7EB",
    "grid.linewidth":   0.6,
})
OUT_DIR = Path("renewable_energy_forecast/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> str:
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Saved] {path}")
    return str(path)


# ── 1. Dataset Overview ────────────────────────────────────────────────────────

def plot_dataset_overview(df: pd.DataFrame) -> str:
    df_daily = df.set_index("timestamp").resample("D").mean(numeric_only=True).reset_index()

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle("Renewable Energy Dataset Overview  (Daily Averages)", fontsize=14, fontweight="bold", y=1.01)

    axes[0].fill_between(df_daily["timestamp"], df_daily["solar_irradiance_wm2"],
                         alpha=0.7, color=PALETTE[2], label="Solar Irradiance (W/m²)")
    axes[0].set_ylabel("W/m²"); axes[0].legend(loc="upper right")

    axes[1].plot(df_daily["timestamp"], df_daily["pv_output_kw"],
                 color=PALETTE[0], lw=1.2, label="PV Output (kW)")
    axes[1].fill_between(df_daily["timestamp"], df_daily["wind_output_kw"],
                         alpha=0.4, color=PALETTE[3], label="Wind Output (kW)")
    axes[1].set_ylabel("kW"); axes[1].legend(loc="upper right")

    axes[2].plot(df_daily["timestamp"], df_daily["temperature_c"],
                 color=PALETTE[4], lw=1.2, label="Temperature (°C)")
    ax2b = axes[2].twinx()
    ax2b.plot(df_daily["timestamp"], df_daily["cloud_cover_pct"],
              color=PALETTE[5], lw=1.0, linestyle="--", alpha=0.8, label="Cloud Cover (%)")
    axes[2].set_ylabel("°C"); ax2b.set_ylabel("Cloud %")
    lines2  = axes[2].get_lines() + ax2b.get_lines()
    labels2 = [l.get_label() for l in lines2]
    axes[2].legend(lines2, labels2, loc="upper right")

    axes[3].fill_between(df_daily["timestamp"], df_daily["load_demand_kw"],
                         alpha=0.6, color=PALETTE[1], label="Load Demand (kW)")
    axes[3].fill_between(df_daily["timestamp"], df_daily["battery_soc_kwh"],
                         alpha=0.4, color=PALETTE[6], label="Battery SOC (kWh)")
    axes[3].set_ylabel("kW / kWh"); axes[3].legend(loc="upper right")
    axes[3].set_xlabel("Date")

    plt.tight_layout()
    return _save(fig, "01_dataset_overview.png")


# ── 2. Seasonal Solar Profile ─────────────────────────────────────────────────

def plot_seasonal_profile(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Average Hourly PV Output by Season", fontsize=14, fontweight="bold")
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    colors  = [PALETTE[0], PALETTE[5], PALETTE[2], PALETTE[3]]

    for ax, season, color in zip(axes.flat, seasons, colors):
        sub   = df[df["season"] == season]
        hourly= sub.groupby("hour")["pv_output_kw"].agg(["mean","std"]).reset_index()
        ax.fill_between(hourly["hour"],
                        hourly["mean"] - hourly["std"],
                        hourly["mean"] + hourly["std"],
                        alpha=0.25, color=color)
        ax.plot(hourly["hour"], hourly["mean"], color=color, lw=2.5, label=f"{season} avg")
        ax.set_title(season); ax.set_xlabel("Hour of Day"); ax.set_ylabel("PV (kW)")
        ax.set_xlim(0, 23); ax.legend()

    plt.tight_layout()
    return _save(fig, "02_seasonal_pv_profile.png")


# ── 3. Correlation Heatmap ────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    cols = ["solar_irradiance_wm2", "pv_output_kw", "wind_output_kw",
            "temperature_c", "humidity_pct", "cloud_cover_pct",
            "wind_speed_ms", "load_demand_kw", "battery_soc_kwh"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    cmap = LinearSegmentedColormap.from_list("rg", ["#C73E1D", "#FFFFFF", "#2E86AB"])
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap,
                vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                annot_kws={"size": 9})
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "03_correlation_heatmap.png")


# ── 4. Forecast vs Actual ─────────────────────────────────────────────────────

def plot_forecast_vs_actual(
    y_true: np.ndarray,
    y_pred_xgb: np.ndarray,
    y_pred_ensemble: np.ndarray,
    n_display: int = 168,   # 1 week
) -> str:
    n = min(n_display, len(y_true))
    t = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("Forecast vs Actual – PV Output (Test Window)", fontsize=14, fontweight="bold")

    # Time-series comparison
    axes[0].plot(t, y_true[:n],         color=PALETTE[1], lw=2.0, label="Actual")
    axes[0].plot(t, y_pred_xgb[:n],     color=PALETTE[0], lw=1.4, linestyle="--", label="XGBoost")
    axes[0].plot(t, y_pred_ensemble[:n], color=PALETTE[2], lw=1.4, linestyle=":",  label="Ensemble")
    axes[0].set_xlabel("Hour"); axes[0].set_ylabel("PV Output (kW)")
    axes[0].legend(); axes[0].set_title("Time-series Comparison")

    # Scatter
    axes[1].scatter(y_true[:n], y_pred_ensemble[:n],
                    alpha=0.4, s=15, color=PALETTE[0], label="Ensemble")
    axes[1].scatter(y_true[:n], y_pred_xgb[:n],
                    alpha=0.3, s=10, color=PALETTE[3], label="XGBoost")
    lim = max(y_true[:n].max(), y_pred_ensemble[:n].max()) * 1.05
    axes[1].plot([0, lim], [0, lim], "k--", lw=1, label="Perfect fit")
    axes[1].set_xlabel("Actual (kW)"); axes[1].set_ylabel("Predicted (kW)")
    axes[1].legend(); axes[1].set_title("Predicted vs Actual Scatter")

    plt.tight_layout()
    return _save(fig, "04_forecast_vs_actual.png")


# ── 5. Feature Importance ─────────────────────────────────────────────────────

def plot_feature_importance(
    xgb_imp: pd.Series,
    rf_imp:  pd.Series,
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Feature Importance", fontsize=14, fontweight="bold")

    top_xgb = xgb_imp.head(10)
    axes[0].barh(top_xgb.index[::-1], top_xgb.values[::-1], color=PALETTE[0])
    axes[0].set_title("XGBoost (PV Output)"); axes[0].set_xlabel("Importance")

    top_rf = rf_imp.head(10)
    axes[1].barh(top_rf.index[::-1], top_rf.values[::-1], color=PALETTE[5])
    axes[1].set_title("Random Forest (Irradiance)"); axes[1].set_xlabel("Importance")

    plt.tight_layout()
    return _save(fig, "05_feature_importance.png")


# ── 6. Battery Schedule ───────────────────────────────────────────────────────

def plot_battery_schedule(schedule: pd.DataFrame) -> str:
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle("24-Hour Battery Optimization Schedule", fontsize=14, fontweight="bold")

    hours = schedule["hour"]

    # Energy flows
    axes[0].bar(hours, schedule["pv_kw"],          color=PALETTE[2], label="PV")
    axes[0].bar(hours, schedule["wind_kw"],         color=PALETTE[0], bottom=schedule["pv_kw"], label="Wind")
    axes[0].bar(hours, schedule["grid_import_kw"],  color=PALETTE[3],
                bottom=schedule["pv_kw"] + schedule["wind_kw"], label="Grid Import", alpha=0.8)
    axes[0].plot(hours, schedule["load_kw"], "k--", lw=2, label="Load Demand")
    axes[0].set_ylabel("kW"); axes[0].legend(ncol=4); axes[0].set_title("Generation Mix vs Load")

    # Charge / Discharge
    axes[1].bar(hours,  schedule["charge_kw"],    color=PALETTE[5], label="Charge ↑", alpha=0.85)
    axes[1].bar(hours, -schedule["discharge_kw"], color=PALETTE[4], label="Discharge ↓", alpha=0.85)
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_ylabel("kW"); axes[1].legend(); axes[1].set_title("Battery Charge / Discharge")

    # SOC
    axes[2].fill_between(hours, schedule["soc_kwh"],
                         alpha=0.6, color=PALETTE[0], label="Battery SOC")
    axes[2].plot(hours, schedule["soc_kwh"], color=PALETTE[0], lw=2)
    axes[2].set_ylabel("kWh"); axes[2].set_xlabel("Hour of Day")
    axes[2].legend(); axes[2].set_title("State of Charge (kWh)")

    plt.tight_layout()
    return _save(fig, "06_battery_schedule.png")


# ── 7. Energy Allocation ──────────────────────────────────────────────────────

def plot_energy_allocation(alloc: pd.DataFrame) -> str:
    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig)
    fig.suptitle("Energy Allocation & Source Mix", fontsize=14, fontweight="bold")

    # Hourly stacked source bar
    ax1 = fig.add_subplot(gs[0, :])
    ax1.bar(alloc["hour"], alloc["renewable_kw"],   color=PALETTE[2], label="Renewable", alpha=0.85)
    ax1.bar(alloc["hour"], alloc["discharge_kw"],   color=PALETTE[5],
            bottom=alloc["renewable_kw"], label="Battery Discharge", alpha=0.85)
    ax1.bar(alloc["hour"], alloc["grid_import_kw"], color=PALETTE[3],
            bottom=alloc["renewable_kw"] + alloc["discharge_kw"], label="Grid Import", alpha=0.8)
    ax1.plot(alloc["hour"], alloc["renewable_kw"] + alloc["discharge_kw"] + alloc["grid_import_kw"],
             "k--", lw=1.5, label="Total Supply")
    ax1.set_xlabel("Hour"); ax1.set_ylabel("kW"); ax1.legend(ncol=4)
    ax1.set_title("Hourly Energy Source Stack")

    # Pie: total source split
    ax2 = fig.add_subplot(gs[1, 0])
    sizes  = [alloc["renewable_kw"].sum(),
              alloc["discharge_kw"].sum(),
              alloc["grid_import_kw"].sum()]
    labels = ["Renewable", "Battery Discharge", "Grid Import"]
    colors = [PALETTE[2], PALETTE[5], PALETTE[3]]
    ax2.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, pctdistance=0.80)
    ax2.set_title("24h Source Split")

    # Self-sufficiency by hour
    ax3 = fig.add_subplot(gs[1, 1])
    ss_colors = [PALETTE[5] if v >= 80 else PALETTE[2] if v >= 60 else PALETTE[3]
                 for v in alloc["self_suff_pct"]]
    ax3.bar(alloc["hour"], alloc["self_suff_pct"], color=ss_colors, alpha=0.85)
    ax3.axhline(80, color="green",  lw=1.2, linestyle="--", label="80% target")
    ax3.axhline(60, color="orange", lw=1.2, linestyle="--", label="60% warning")
    ax3.set_xlabel("Hour"); ax3.set_ylabel("%")
    ax3.set_title("Self-Sufficiency (%)"); ax3.legend()

    plt.tight_layout()
    return _save(fig, "07_energy_allocation.png")


# ── 8. KPI Dashboard ──────────────────────────────────────────────────────────

def plot_kpi_dashboard(kpis: Dict) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Energy System KPI Dashboard", fontsize=15, fontweight="bold")
    fig.patch.set_facecolor("#F7F8FA")

    def _gauge(ax, value, max_val, title, unit, color, reverse=False):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        frac = value / max_val
        bar_color = color if not reverse else (PALETTE[5] if frac < 0.3 else
                                               PALETTE[2] if frac < 0.6 else PALETTE[3])
        rect = mpatches.FancyBboxPatch((0.05, 0.3), 0.9 * frac, 0.25,
                                       boxstyle="round,pad=0.02",
                                       facecolor=bar_color, edgecolor="none", alpha=0.85)
        bg   = mpatches.FancyBboxPatch((0.05, 0.3), 0.90, 0.25,
                                       boxstyle="round,pad=0.02",
                                       facecolor="#E5E7EB", edgecolor="none")
        ax.add_patch(bg); ax.add_patch(rect)
        ax.text(0.5, 0.72, title, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(0.5, 0.17, f"{value:.1f} {unit}", ha="center", va="center",
                fontsize=13, color=bar_color, fontweight="bold")

    _gauge(axes[0,0], kpis["avg_self_sufficiency_pct"],  100, "Self-Sufficiency",  "%",   PALETTE[5])
    _gauge(axes[0,1], kpis["renewable_coverage_pct"],    100, "Renewable Coverage","%",   PALETTE[2])
    _gauge(axes[0,2], kpis["fossil_fraction_pct"],       100, "Fossil Fraction",   "%",   PALETTE[3], reverse=True)
    _gauge(axes[1,0], kpis["renewable_waste_pct"],       100, "Renewable Waste",   "%",   PALETTE[6], reverse=True)
    _gauge(axes[1,1], kpis["co2_avoided_kg"],            200, "CO₂ Avoided",       "kg",  PALETTE[5])
    _gauge(axes[1,2], kpis["estimated_cost_usd"],        20,  "Grid Cost",         "USD", PALETTE[3], reverse=True)

    plt.tight_layout()
    return _save(fig, "08_kpi_dashboard.png")


# ── 9. Wind Power Curve ───────────────────────────────────────────────────────

def plot_wind_power_curve(df: pd.DataFrame) -> str:
    sample = df.sample(min(5000, len(df)), random_state=42)
    bins   = np.arange(0, 26, 1)
    grp    = sample.groupby(pd.cut(sample["wind_speed_ms"], bins))["wind_output_kw"]
    mn, sd = grp.mean(), grp.std()
    centers = [(b.left + b.right) / 2 for b in mn.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(sample["wind_speed_ms"], sample["wind_output_kw"],
               alpha=0.1, s=8, color=PALETTE[0], label="Observations")
    ax.fill_between(centers, mn - sd, mn + sd, alpha=0.25, color=PALETTE[3])
    ax.plot(centers, mn, color=PALETTE[3], lw=2.5, label="Mean power curve")
    ax.set_xlabel("Wind Speed (m/s)"); ax.set_ylabel("Wind Output (kW)")
    ax.set_title("Wind Power Curve", fontsize=13, fontweight="bold")
    ax.legend(); plt.tight_layout()
    return _save(fig, "09_wind_power_curve.png")


# ── 10. Error Distribution ────────────────────────────────────────────────────

def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Ensemble",
) -> str:
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Prediction Error Distribution – {model_name}", fontsize=13, fontweight="bold")

    # Histogram
    axes[0].hist(residuals, bins=60, color=PALETTE[0], edgecolor="white",
                 linewidth=0.4, alpha=0.85)
    axes[0].axvline(0, color="black", lw=1.5, linestyle="--")
    axes[0].set_xlabel("Residual (kW)"); axes[0].set_ylabel("Count")
    axes[0].set_title("Residual Histogram")

    # Q-Q plot
    from scipy import stats
    (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
    qq_x = np.linspace(osm[0], osm[-1], 200)
    axes[1].scatter(osm, osr, alpha=0.3, s=8, color=PALETTE[0])
    axes[1].plot(qq_x, slope * qq_x + intercept, color=PALETTE[3], lw=2)
    axes[1].set_xlabel("Theoretical Quantiles"); axes[1].set_ylabel("Sample Quantiles")
    axes[1].set_title("Q-Q Plot (Normal)")

    plt.tight_layout()
    return _save(fig, "10_error_distribution.png")


# ── Full Dashboard Builder ─────────────────────────────────────────────────────

def build_full_dashboard(
    df:              pd.DataFrame,
    schedule:        pd.DataFrame,
    alloc_result:    Dict,
    y_true:          np.ndarray,
    y_pred_xgb:      np.ndarray,
    y_pred_ensemble: np.ndarray,
    xgb_importance:  pd.Series,
    rf_importance:   pd.Series,
) -> Dict[str, str]:
    print("\n[Visualizer] Generating charts …")
    saved = {}
    saved["overview"]          = plot_dataset_overview(df)
    saved["seasonal"]          = plot_seasonal_profile(df)
    saved["correlation"]       = plot_correlation_heatmap(df)
    saved["forecast"]          = plot_forecast_vs_actual(y_true, y_pred_xgb, y_pred_ensemble)
    saved["feature_imp"]       = plot_feature_importance(xgb_importance, rf_importance)
    saved["battery_schedule"]  = plot_battery_schedule(schedule)
    saved["allocation"]        = plot_energy_allocation(alloc_result["hourly_allocation"])
    saved["kpi_dashboard"]     = plot_kpi_dashboard(alloc_result["kpi_summary"])
    saved["wind_curve"]        = plot_wind_power_curve(df)
    saved["error_dist"]        = plot_error_distribution(y_true, y_pred_ensemble)
    print(f"[Visualizer] All {len(saved)} charts saved to {OUT_DIR}/")
    return saved
