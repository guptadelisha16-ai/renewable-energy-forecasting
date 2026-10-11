"""
data_generator.py
-----------------
Simulates two years of hourly solar irradiance, weather conditions,
wind speed, and photovoltaic (PV) output records.

The physics-based solar model uses:
  - Extraterrestrial radiation (declination + hour angle)
  - Atmospheric attenuation via air mass and turbidity
  - Cloud-cover and temperature de-rating for PV cells
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)


def _solar_declination(doy: np.ndarray) -> np.ndarray:
    """Solar declination angle (degrees) by day-of-year."""
    return 23.45 * np.sin(np.radians(360 / 365 * (doy - 81)))


def _hour_angle(hour: np.ndarray) -> np.ndarray:
    """Hour angle (degrees): 0 at solar noon."""
    return 15 * (hour - 12)


def _solar_elevation(lat_deg: float, doy: np.ndarray, hour: np.ndarray) -> np.ndarray:
    """Solar elevation angle (degrees above horizon)."""
    lat  = np.radians(lat_deg)
    dec  = np.radians(_solar_declination(doy))
    ha   = np.radians(_hour_angle(hour))
    sin_elev = (np.sin(lat) * np.sin(dec) +
                np.cos(lat) * np.cos(dec) * np.cos(ha))
    return np.degrees(np.arcsin(np.clip(sin_elev, -1, 1)))


def generate_dataset(
    latitude: float   = 28.6,          # New Delhi, India
    years: int        = 2,
    panel_capacity_kw: float = 10.0,   # installed PV capacity
    battery_capacity_kwh: float = 40.0,
    output_path: str  = "renewable_energy_forecast/data/energy_dataset.csv",
) -> pd.DataFrame:
    """
    Generate an hourly dataset for `years` years and save to CSV.

    Returns
    -------
    pd.DataFrame  with columns:
        timestamp, doy, hour, month, season,
        temperature_c, humidity_pct, wind_speed_ms, cloud_cover_pct,
        solar_irradiance_wm2, ghi_wm2,
        pv_output_kw, wind_output_kw,
        total_renewable_kw, load_demand_kw, net_energy_kw,
        battery_soc_kwh
    """
    periods = 24 * 365 * years
    idx     = pd.date_range("2022-01-01", periods=periods, freq="h")

    doy     = idx.day_of_year.to_numpy().astype(float)
    hour    = idx.hour.to_numpy().astype(float)
    month   = idx.month.to_numpy()

    # ── Weather simulation ────────────────────────────────────────────────────
    # Temperature: seasonal sinusoidal + daily cycle + noise
    temp_seasonal  = 25 + 12 * np.sin(np.radians(360 / 365 * (doy - 120)))
    temp_diurnal   = 5  * np.sin(np.radians(360 / 24  * (hour - 14)))
    temperature    = temp_seasonal + temp_diurnal + rng.normal(0, 1.5, periods)

    # Humidity: anticorrelated with temperature
    humidity       = np.clip(
        70 - 0.8 * (temperature - 25) + rng.normal(0, 8, periods), 10, 100
    )

    # Wind speed: Weibull distribution, slight seasonal variation
    wind_scale     = 4.5 + 1.5 * np.sin(np.radians(360 / 365 * (doy - 60)))
    wind_speed     = rng.weibull(2.0, periods) * wind_scale

    # Cloud cover: higher in monsoon (Jun–Sep), random walk component
    cloud_base     = np.where((month >= 6) & (month <= 9), 60, 25).astype(float)
    cloud_noise    = rng.normal(0, 15, periods)
    cloud_cover    = np.clip(cloud_base + cloud_noise, 0, 100)

    # ── Solar irradiance ──────────────────────────────────────────────────────
    elev = _solar_elevation(latitude, doy, hour)
    elev_clipped = np.clip(elev, 0, 90)

    # Extraterrestrial irradiance
    I0 = 1361 * (1 + 0.033 * np.cos(np.radians(360 / 365 * doy)))

    # Air mass (Kasten–Young formula)
    elev_rad = np.radians(elev_clipped + 1e-6)
    air_mass  = np.where(
        elev > 0,
        1 / (np.sin(elev_rad) + 0.50572 * (elev_clipped + 6.07995) ** -1.6364),
        0,
    )
    air_mass  = np.clip(air_mass, 0, 38)

    # Beam irradiance with turbidity (τ = 0.1)
    tau = 0.1
    beam_clear = I0 * np.exp(-tau * air_mass) * np.sin(elev_rad)
    beam_clear = np.where(elev > 0, beam_clear, 0)

    # Diffuse component (~15% of beam under clear sky)
    diffuse_clear = 0.15 * beam_clear

    # Cloud attenuation
    cloud_factor = 1 - 0.75 * (cloud_cover / 100) ** 3.4
    ghi          = (beam_clear + diffuse_clear) * cloud_factor

    # Add measurement noise
    solar_irradiance = np.clip(
        ghi + rng.normal(0, 10, periods), 0, 1200
    )

    # ── PV output ─────────────────────────────────────────────────────────────
    pv_efficiency   = 0.18                         # 18 % panel efficiency
    temp_coeff      = -0.004                        # −0.4 % / °C above 25 °C
    pv_area         = panel_capacity_kw / (pv_efficiency * 1.0)  # m²·kW proxy
    temp_de_rate    = 1 + temp_coeff * (temperature - 25)
    pv_output       = (solar_irradiance / 1000) * panel_capacity_kw * temp_de_rate
    pv_output       = np.clip(pv_output + rng.normal(0, 0.05, periods), 0, panel_capacity_kw)

    # ── Wind turbine output ───────────────────────────────────────────────────
    rated_wind_kw   = 5.0
    cut_in, rated, cut_out = 3.0, 12.0, 25.0
    wind_output = np.where(
        wind_speed < cut_in, 0,
        np.where(
            wind_speed < rated,
            rated_wind_kw * ((wind_speed - cut_in) / (rated - cut_in)) ** 3,
            np.where(wind_speed < cut_out, rated_wind_kw, 0),
        ),
    )
    wind_output = np.clip(wind_output + rng.normal(0, 0.03, periods), 0, rated_wind_kw)

    # ── Load demand (residential + light industrial) ──────────────────────────
    load_base      = 6.0
    load_diurnal   = 2.5 * np.sin(np.radians(360 / 24 * (hour - 18)))  # peak evening
    load_seasonal  = 1.5 * np.sin(np.radians(360 / 365 * (doy - 200))) # peak summer
    load_demand    = np.clip(
        load_base + load_diurnal + load_seasonal + rng.normal(0, 0.5, periods),
        1.0, 20.0,
    )

    # ── Total renewable and net energy ────────────────────────────────────────
    total_renewable = pv_output + wind_output
    net_energy      = total_renewable - load_demand    # + surplus, − deficit

    # ── Battery state-of-charge simulation (simple rule-based) ───────────────
    soc    = np.zeros(periods)
    soc[0] = battery_capacity_kwh * 0.5               # start at 50 %
    charge_eff, discharge_eff = 0.95, 0.95
    for t in range(1, periods):
        net = net_energy[t]
        if net > 0:                                    # surplus → charge
            soc[t] = min(battery_capacity_kwh,
                         soc[t-1] + net * charge_eff)
        else:                                          # deficit → discharge
            soc[t] = max(0, soc[t-1] + net / discharge_eff)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                  3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Autumn", 10: "Autumn", 11: "Autumn"}
    season = [season_map[m] for m in month]

    df = pd.DataFrame({
        "timestamp":           idx,
        "doy":                 doy.astype(int),
        "hour":                hour.astype(int),
        "month":               month,
        "season":              season,
        "temperature_c":       temperature.round(2),
        "humidity_pct":        humidity.round(2),
        "wind_speed_ms":       wind_speed.round(3),
        "cloud_cover_pct":     cloud_cover.round(2),
        "solar_irradiance_wm2": solar_irradiance.round(2),
        "ghi_wm2":             ghi.round(2),
        "pv_output_kw":        pv_output.round(4),
        "wind_output_kw":      wind_output.round(4),
        "total_renewable_kw":  total_renewable.round(4),
        "load_demand_kw":      load_demand.round(4),
        "net_energy_kw":       net_energy.round(4),
        "battery_soc_kwh":     soc.round(4),
    })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[DataGenerator] Saved {len(df):,} rows → {output_path}")
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(df.describe().round(2))
