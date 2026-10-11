# ☀️ SolarWise AI — Renewable Energy Forecasting & Optimization

A complete end-to-end machine learning system that forecasts solar and wind power generation, optimizes battery charge/discharge scheduling using linear programming, and provides actionable energy allocation recommendations — all served through an interactive Flask web application.

---

## 🚀 Live Features

| Module | Technology | What it does |
|---|---|---|
| **Data Simulation** | Physics-based model | 2 years of hourly solar irradiance, weather & PV records |
| **Weather Prediction** | Random Forest → XGBoost (2-stage) | Predicts irradiance then PV output from weather inputs |
| **Time-Series Forecast** | SARIMA + XGBoost Ensemble | 24-hour ahead PV generation forecasting |
| **Battery Optimizer** | Linear Programming (scipy HiGHS) | Optimal 24h charge/discharge schedule |
| **Energy Allocator** | Rule + heuristic engine | Priority dispatch + natural-language recommendations |
| **Web Dashboard** | Flask + Chart.js | Interactive 6-page UI for all functionalities |

---

## 📊 Model Performance

| Model | RMSE | MAE | R² |
|---|---|---|---|
| XGBoost PV Time-Series | 0.054 kW | 0.040 kW | **0.9998** |
| Random Forest Irradiance | 16.9 W/m² | 9.6 W/m² | **0.998** |
| XGBoost PV (weather) | 0.123 kW | 0.082 kW | **0.9989** |
| GBT Wind Output | 0.021 kW | 0.016 kW | **0.9912** |
| SARIMA Seasonal | 1.362 kW | 0.825 kW | 0.861 |
| Ensemble (70/30) | 0.411 kW | 0.257 kW | **0.9873** |

---

## 🗂️ Project Structure

```
renewable_energy_forecast/
├── data/
│   ├── data_generator.py        # Physics-based solar/weather simulation
│   └── energy_dataset.csv       # 17,520 rows — 2 years hourly data
├── models/
│   ├── time_series_forecaster.py  # SARIMA + XGBoost ensemble
│   └── weather_predictor.py       # RF → XGBoost two-stage pipeline
├── optimization/
│   ├── battery_optimizer.py     # LP optimizer (scipy HiGHS)
│   └── energy_allocator.py      # Priority dispatch + recommendations
├── visualization/
│   └── visualizer.py            # 10 production charts (matplotlib)
├── webapp/
│   ├── app.py                   # Flask application & API routes
│   └── templates/               # Jinja2 HTML templates
│       ├── base.html            # Sidebar layout + dark theme
│       ├── dashboard.html       # KPI overview + sparklines
│       ├── forecast.html        # Weather input + 24h forecast
│       ├── optimizer.html       # Battery LP schedule
│       ├── allocator.html       # Energy allocation + recommendations
│       ├── charts.html          # Gallery of all output charts
│       └── dataset.html         # Dataset explorer
├── outputs/                     # Generated charts + reports
│   ├── 01_dataset_overview.png
│   ├── 02_seasonal_pv_profile.png
│   ├── ...
│   └── system_report.txt
├── main_pipeline.py             # End-to-end training pipeline
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/renewable-energy-forecasting.git
cd renewable-energy-forecasting
```

### 2. Install dependencies
```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn xgboost statsmodels pulp joblib flask
```

### 3. Run the full training pipeline (generates data + trains all models)
```bash
python renewable_energy_forecast/main_pipeline.py
```

### 4. Launch the web application
```bash
python renewable_energy_forecast/webapp/app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## 🌐 Web App Pages

| Page | URL | Description |
|---|---|---|
| Dashboard | `/` | KPI cards, 30-day trends, seasonal profiles, model metrics |
| Forecast | `/forecast` | Weather form → instant PV/wind prediction + 24h chart |
| Optimizer | `/optimizer` | LP battery scheduler with schedule table + charts |
| Allocator | `/allocator` | Recommendations, priority dispatch, source-split pie |
| Charts | `/charts` | Gallery of all 10 output charts with lightbox |
| Dataset | `/dataset` | Stats, preview table, variable explorer |

---

## 🧠 Algorithm Details

### Solar Irradiance Model
- Extraterrestrial radiation using solar declination + hour angle
- Air mass via Kasten–Young formula
- Cloud attenuation with empirical Beer-Lambert model
- Temperature de-rating for PV cells (−0.4%/°C above 25°C)

### Battery LP Optimizer
- 5 decision variables per timestep: charge, discharge, grid import, curtailment, SOC
- 2T equality constraints: energy balance + SOC dynamics
- Solved with scipy `linprog` HiGHS backend (no external binary needed)
- Objective: minimize α·grid_import + β·curtailment + γ·peak

### Ensemble Forecaster
- SARIMA(1,0,1)(1,0,1)[24] on daily totals + diurnal profile redistribution
- XGBoost with lags [1h, 2h, 3h, 6h, 12h, 24h, 48h] + cyclic time features
- Final prediction: 70% XGBoost + 30% SARIMA

---

## 📈 Output Charts

1. Dataset Overview (2-year daily averages)
2. Seasonal PV Profile (hourly mean ± std per season)
3. Correlation Heatmap
4. Forecast vs Actual (time-series + scatter)
5. Feature Importance (XGBoost + Random Forest)
6. Battery Schedule (24h charge/discharge/SOC)
7. Energy Allocation (source stack + pie + self-sufficiency)
8. KPI Dashboard (gauge-style)
9. Wind Power Curve
10. Error Distribution (histogram + Q-Q plot)

---

## 🌱 Environmental Impact

On a typical summer day (10 kW PV + 5 kW wind, 40 kWh battery):
- **83% self-sufficiency** from renewable sources
- **~36 kg CO₂ avoided** per day
- **~$3 grid electricity cost** at $0.12/kWh

---

## 📦 Dependencies

```
numpy, pandas, scikit-learn, scipy, matplotlib, seaborn,
xgboost, statsmodels, pulp, joblib, flask
```

---

## 📄 License

MIT License — free to use, modify and distribute.
