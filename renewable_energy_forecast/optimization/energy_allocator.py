"""
energy_allocator.py
--------------------
Rule + heuristic energy allocation engine.

Given a 24-hour optimized schedule, this module:
  1. Classifies each hour into an energy state.
  2. Applies priority-based load allocation (critical → flexible → EV).
  3. Generates plain-language recommendations for the operator.
  4. Produces a daily allocation summary with KPIs.

No external solver needed — this is a post-optimization advisory layer.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ── Load profile categories ────────────────────────────────────────────────────

@dataclass
class LoadProfile:
    """Defines load categories with priority and flexibility."""
    critical_load_pct:  float = 0.40   # % of total load that is non-deferrable
    flexible_load_pct:  float = 0.35   # e.g., HVAC, washing machines
    ev_charging_pct:    float = 0.15   # EV / overnight loads
    lighting_pct:       float = 0.10   # low-priority background


# ── Energy State Classification ────────────────────────────────────────────────

ENERGY_STATES = {
    "SURPLUS_HIGH":   "Renewable surplus > 30 % of load — charge battery + export",
    "SURPLUS_LOW":    "Renewable surplus ≤ 30 % of load — charge battery only",
    "BALANCED":       "Renewable ≈ load — minimal grid draw needed",
    "DEFICIT_LOW":    "Small deficit — discharge battery first",
    "DEFICIT_HIGH":   "Large deficit — battery + grid import required",
    "NIGHT_LOW_WIND": "Night-time, low renewable — rely on battery / grid",
}


def classify_hour(
    renewable_kw: float,
    load_kw:      float,
    soc_kwh:      float,
    battery_capacity_kwh: float = 40.0,
) -> str:
    if load_kw <= 0:
        return "BALANCED"
    ratio  = renewable_kw / load_kw
    soc_pct = soc_kwh / battery_capacity_kwh

    if ratio >= 1.30:
        return "SURPLUS_HIGH"
    if ratio >= 1.00:
        return "SURPLUS_LOW"
    if ratio >= 0.85:
        return "BALANCED"
    if ratio >= 0.50:
        return "DEFICIT_LOW"
    if renewable_kw < 0.5 and soc_pct < 0.2:
        return "NIGHT_LOW_WIND"
    return "DEFICIT_HIGH"


# ── Allocation Engine ──────────────────────────────────────────────────────────

class EnergyAllocator:
    """
    Translates an optimized battery schedule into actionable allocation
    decisions and operator recommendations.

    Usage
    -----
    allocator = EnergyAllocator()
    result    = allocator.allocate(schedule_df, battery_capacity_kwh=40)
    print(result["recommendations"])
    """

    def __init__(self, load_profile: LoadProfile | None = None):
        self.lp = load_profile or LoadProfile()

    # ── Main entry point ────────────────────────────────────────────────────

    def allocate(
        self,
        schedule: pd.DataFrame,
        battery_capacity_kwh: float = 40.0,
    ) -> Dict:
        """
        Parameters
        ----------
        schedule : DataFrame from BatteryOptimizer.optimize()
        battery_capacity_kwh : installed capacity for SOC % calculation

        Returns
        -------
        dict with keys:
          hourly_allocation   : DataFrame
          recommendations     : List[str]
          kpi_summary         : Dict
          priority_dispatch   : DataFrame
        """
        T = len(schedule)

        # ── Per-hour energy state ────────────────────────────────────────────
        states, load_splits = [], []
        for _, row in schedule.iterrows():
            state = classify_hour(
                row["renewable_kw"], row["load_kw"],
                row["soc_kwh"], battery_capacity_kwh,
            )
            states.append(state)

            available = row["renewable_kw"] + row["discharge_kw"] * 0.95 + row["grid_import_kw"]
            load  = row["load_kw"]
            lp    = self.lp

            critical  = load * lp.critical_load_pct
            flexible  = load * lp.flexible_load_pct
            ev        = load * lp.ev_charging_pct
            lighting  = load * lp.lighting_pct

            # Shed flexible / EV if grid import is high
            if state in ("DEFICIT_HIGH", "NIGHT_LOW_WIND"):
                flex_served = min(flexible, available - critical)
                flex_served = max(flex_served, 0)
                ev_served   = 0                     # defer EV charging
            else:
                flex_served = flexible
                ev_served   = ev

            load_splits.append({
                "hour":           row["hour"],
                "state":          state,
                "critical_kw":    round(critical, 3),
                "flexible_kw":    round(flex_served, 3),
                "ev_kw":          round(ev_served, 3),
                "lighting_kw":    round(lighting, 3),
                "deferred_ev_kw": round(ev - ev_served, 3),
            })

        alloc_df = pd.DataFrame(load_splits)
        alloc_df["renewable_kw"]    = schedule["renewable_kw"].values
        alloc_df["charge_kw"]       = schedule["charge_kw"].values
        alloc_df["discharge_kw"]    = schedule["discharge_kw"].values
        alloc_df["grid_import_kw"]  = schedule["grid_import_kw"].values
        alloc_df["curtailment_kw"]  = schedule["curtailment_kw"].values
        alloc_df["soc_kwh"]         = schedule["soc_kwh"].values
        alloc_df["soc_pct"]         = (schedule["soc_kwh"].values / battery_capacity_kwh * 100).round(1)
        alloc_df["self_suff_pct"]   = schedule["self_sufficiency_pct"].values

        # ── Priority dispatch table ──────────────────────────────────────────
        priority_df = self._build_priority_dispatch(schedule, states)

        # ── KPI summary ──────────────────────────────────────────────────────
        kpis = self._compute_kpis(schedule, alloc_df)

        # ── Recommendations ──────────────────────────────────────────────────
        recs = self._generate_recommendations(alloc_df, kpis, battery_capacity_kwh)

        return {
            "hourly_allocation": alloc_df,
            "priority_dispatch": priority_df,
            "recommendations":   recs,
            "kpi_summary":       kpis,
        }

    # ── Priority dispatch ────────────────────────────────────────────────────

    def _build_priority_dispatch(
        self, schedule: pd.DataFrame, states: List[str]
    ) -> pd.DataFrame:
        rows = []
        for i, (_, row) in enumerate(schedule.iterrows()):
            state   = states[i]
            ren     = row["renewable_kw"]
            bat_dis = row["discharge_kw"] * 0.95
            grid    = row["grid_import_kw"]
            total   = ren + bat_dis + grid

            rows.append({
                "hour":            row["hour"],
                "state":           state,
                "src_renewable_%": round(ren  / total * 100, 1) if total > 0 else 0,
                "src_battery_%":   round(bat_dis / total * 100, 1) if total > 0 else 0,
                "src_grid_%":      round(grid / total * 100, 1) if total > 0 else 0,
                "action":          self._hour_action(state, row),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _hour_action(state: str, row: pd.Series) -> str:
        if state == "SURPLUS_HIGH":
            return f"Charge battery +{row['charge_kw']:.1f} kW; curtail {row['curtailment_kw']:.1f} kW"
        if state == "SURPLUS_LOW":
            return f"Charge battery +{row['charge_kw']:.1f} kW"
        if state == "BALANCED":
            return "Hold battery; serve load from renewable"
        if state == "DEFICIT_LOW":
            return f"Discharge battery {row['discharge_kw']:.1f} kW"
        if state == "DEFICIT_HIGH":
            return f"Discharge {row['discharge_kw']:.1f} kW + import {row['grid_import_kw']:.1f} kW"
        return f"Import {row['grid_import_kw']:.1f} kW from grid; conserve battery"

    # ── KPI computation ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_kpis(schedule: pd.DataFrame, alloc: pd.DataFrame) -> Dict:
        total_load        = schedule["load_kw"].sum()
        total_renewable   = schedule["renewable_kw"].sum()
        total_grid        = schedule["grid_import_kw"].sum()
        total_curtailed   = schedule["curtailment_kw"].sum()
        total_deferred_ev = alloc["deferred_ev_kw"].sum()
        avg_soc           = schedule["soc_kwh"].mean()
        avg_self_suff     = schedule["self_sufficiency_pct"].mean()

        renewable_coverage = (total_renewable / total_load * 100) if total_load > 0 else 0
        fossil_fraction    = (total_grid / total_load * 100) if total_load > 0 else 0
        waste_pct          = (total_curtailed / total_renewable * 100) if total_renewable > 0 else 0

        return {
            "total_load_kwh":           round(total_load, 2),
            "total_renewable_kwh":      round(total_renewable, 2),
            "total_grid_import_kwh":    round(total_grid, 2),
            "total_curtailed_kwh":      round(total_curtailed, 2),
            "total_deferred_ev_kwh":    round(total_deferred_ev, 2),
            "avg_battery_soc_kwh":      round(avg_soc, 2),
            "avg_self_sufficiency_pct": round(avg_self_suff, 2),
            "renewable_coverage_pct":   round(renewable_coverage, 2),
            "fossil_fraction_pct":      round(fossil_fraction, 2),
            "renewable_waste_pct":      round(waste_pct, 2),
            "estimated_cost_usd":       round(total_grid * 0.12, 2),
            "co2_avoided_kg":           round((total_renewable - total_curtailed) * 0.42, 2),
        }

    # ── Natural-language recommendations ────────────────────────────────────

    @staticmethod
    def _generate_recommendations(
        alloc: pd.DataFrame,
        kpis:  Dict,
        capacity: float,
    ) -> List[str]:
        recs = []

        # Self-sufficiency
        ss = kpis["avg_self_sufficiency_pct"]
        if ss >= 80:
            recs.append(f"✅ Excellent self-sufficiency ({ss:.1f}%). The site is largely energy-independent today.")
        elif ss >= 60:
            recs.append(f"⚠️  Moderate self-sufficiency ({ss:.1f}%). Consider increasing battery capacity or adding PV panels.")
        else:
            recs.append(f"🔴 Low self-sufficiency ({ss:.1f}%). High fossil dependency — review load shifting opportunities.")

        # Curtailment
        wasted = kpis["renewable_waste_pct"]
        if wasted > 10:
            recs.append(f"♻️  {wasted:.1f}% of renewable generation is curtailed. "
                        "Increase battery capacity or enable grid export to reduce waste.")
        elif wasted < 2:
            recs.append("🔋 Near-zero curtailment — battery is absorbing surplus effectively.")

        # Peak hours
        peak_hours = alloc[alloc["grid_import_kw"] > alloc["grid_import_kw"].quantile(0.85)]["hour"].tolist()
        if peak_hours:
            peak_str = ", ".join(f"{int(h):02d}:00" for h in peak_hours[:5])
            recs.append(f"⚡ Grid import peaks at hours: {peak_str}. "
                        "Shift flexible loads away from these windows.")

        # EV deferral
        ev_def = kpis["total_deferred_ev_kwh"]
        if ev_def > 0:
            recs.append(f"🚗 {ev_def:.1f} kWh of EV charging deferred to protect battery reserves. "
                        "Schedule EV charging during surplus hours (typically 10:00–15:00).")

        # Battery low nights
        night_mask = (alloc["hour"] >= 22) | (alloc["hour"] <= 5)
        low_soc_nights = alloc[night_mask & (alloc["soc_pct"] < 25)]
        if len(low_soc_nights) > 2:
            recs.append("🌙 Battery SOC drops below 25% on multiple night hours. "
                        "Consider pre-charging to 60–70% before sunset.")

        # CO2
        co2 = kpis["co2_avoided_kg"]
        recs.append(f"🌱 Estimated CO₂ avoided today: {co2:.1f} kg "
                    f"(equivalent to {co2/21:.1f} tree-days of absorption).")

        # Cost
        cost = kpis["estimated_cost_usd"]
        recs.append(f"💰 Estimated grid electricity cost today: ${cost:.2f} USD "
                    f"at $0.12/kWh tariff.")

        return recs
