"""
battery_optimizer.py
---------------------
Linear-programming optimizer for battery charge/discharge scheduling
over a 24-hour horizon.

Objective
---------
Minimize:
  α · grid_import  +  β · curtailment  +  γ · fossil_fraction

Subject to:
  - Energy balance: pv + wind + discharge + grid_import = load + charge + curtailment
  - Battery SOC dynamics with charge/discharge efficiencies
  - SOC bounds: [soc_min, soc_max]
  - Charge / discharge power limits
  - Grid import non-negative

Uses scipy.optimize.linprog (HiGHS backend, bundled with scipy) — no external
solver binary required.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from dataclasses import dataclass
from typing import Dict, List, Optional


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class BatteryConfig:
    capacity_kwh:     float = 40.0    # usable energy capacity
    soc_min_pct:      float = 0.10    # 10 % floor
    soc_max_pct:      float = 0.95    # 95 % ceiling
    max_charge_kw:    float = 10.0    # max charge rate
    max_discharge_kw: float = 10.0    # max discharge rate
    charge_eff:       float = 0.95
    discharge_eff:    float = 0.95
    initial_soc_pct:  float = 0.50    # start at 50 %

    @property
    def soc_min(self) -> float: return self.capacity_kwh * self.soc_min_pct
    @property
    def soc_max(self) -> float: return self.capacity_kwh * self.soc_max_pct
    @property
    def initial_soc(self) -> float: return self.capacity_kwh * self.initial_soc_pct


@dataclass
class OptimizationWeights:
    alpha_grid_import:  float = 1.0
    beta_curtailment:   float = 0.3
    gamma_peak:         float = 0.5
    grid_tariff:        float = 0.12


# ── LP Optimizer ───────────────────────────────────────────────────────────────

class BatteryOptimizer:
    """
    Solves battery scheduling as an LP using scipy.optimize.linprog (HiGHS).

    Decision variable vector per timestep t (5 vars × T timesteps):
      x = [ch_0, dis_0, gi_0, cu_0, soc_0, ch_1, dis_1, ...]
    """

    def __init__(
        self,
        battery: Optional[BatteryConfig] = None,
        weights: Optional[OptimizationWeights] = None,
        dt: float = 1.0,
    ):
        self.battery = battery or BatteryConfig()
        self.weights = weights or OptimizationWeights()
        self.dt = dt

    # ── Variable layout helpers ─────────────────────────────────────────────

    @staticmethod
    def _idx(t: int, var: int, n_vars: int = 5) -> int:
        """Flat index for variable `var` at timestep `t`."""
        return t * n_vars + var
    # var indices: 0=ch, 1=dis, 2=gi, 3=cu, 4=soc

    # ── Core LP ─────────────────────────────────────────────────────────────

    def optimize(
        self,
        pv_forecast:   np.ndarray,
        wind_forecast: np.ndarray,
        load_forecast: np.ndarray,
        initial_soc:   Optional[float] = None,
    ) -> pd.DataFrame:

        T    = len(pv_forecast)
        bat  = self.battery
        w    = self.weights
        soc0 = initial_soc if initial_soc is not None else bat.initial_soc
        N    = 5 * T                         # total decision variables

        idx = self._idx

        # ── Objective coefficients ─────────────────────────────────────────
        c = np.zeros(N)
        for t in range(T):
            c[idx(t,2)] = w.alpha_grid_import + w.gamma_peak   # grid import
            c[idx(t,3)] = w.beta_curtailment                   # curtailment

        # ── Variable bounds ────────────────────────────────────────────────
        bounds = []
        for t in range(T):
            bounds.append((0, bat.max_charge_kw))    # ch
            bounds.append((0, bat.max_discharge_kw)) # dis
            bounds.append((0, None))                  # gi
            bounds.append((0, None))                  # cu
            bounds.append((bat.soc_min, bat.soc_max)) # soc

        # ── Equality constraints ───────────────────────────────────────────
        # Rows: T energy-balance + T SOC-dynamics = 2T equations
        A_eq = np.zeros((2 * T, N))
        b_eq = np.zeros(2 * T)

        renewable = pv_forecast + wind_forecast

        for t in range(T):
            ren = float(renewable[t])
            lod = float(load_forecast[t])

            # Energy balance:
            # ren + dis*eff_d + gi = load + ch + cu
            # → ch - dis*eff_d - gi + cu = ren - load
            row_e = t
            A_eq[row_e, idx(t,0)] =  1.0                  # ch
            A_eq[row_e, idx(t,1)] = -bat.discharge_eff     # -dis*eff
            A_eq[row_e, idx(t,2)] = -1.0                  # -gi
            A_eq[row_e, idx(t,3)] =  1.0                  # cu
            b_eq[row_e]           = ren - lod

            # SOC dynamics:
            # soc_t = soc_{t-1} + ch*eff_c - dis
            # → soc_t - ch*eff_c + dis = soc_{t-1}
            row_s = T + t
            A_eq[row_s, idx(t,4)] =  1.0                  # soc_t
            A_eq[row_s, idx(t,0)] = -bat.charge_eff        # -ch*eff
            A_eq[row_s, idx(t,1)] =  1.0                  # dis
            if t == 0:
                b_eq[row_s] = soc0
            else:
                A_eq[row_s, idx(t-1,4)] = -1.0            # -soc_{t-1}
                b_eq[row_s] = 0.0

        # ── Inequality constraints ─────────────────────────────────────────
        # No simultaneous charge+discharge: ch + dis <= max(limits)
        A_ub = np.zeros((T, N))
        b_ub = np.zeros(T)
        lim  = max(bat.max_charge_kw, bat.max_discharge_kw)
        for t in range(T):
            A_ub[t, idx(t,0)] = 1.0
            A_ub[t, idx(t,1)] = 1.0
            b_ub[t]           = lim

        # ── Solve ──────────────────────────────────────────────────────────
        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options={"presolve": True, "disp": False},
        )

        if result.status != 0:
            print(f"[BatteryOptimizer] LP status {result.status}: {result.message} — using fallback")
            return self._rule_based_fallback(pv_forecast, wind_forecast,
                                             load_forecast, initial_soc)

        x = result.x
        ch_  = np.array([x[idx(t,0)] for t in range(T)])
        dis_ = np.array([x[idx(t,1)] for t in range(T)])
        gi_  = np.array([x[idx(t,2)] for t in range(T)])
        cu_  = np.array([x[idx(t,3)] for t in range(T)])
        s_   = np.array([x[idx(t,4)] for t in range(T)])

        return self._build_result(T, pv_forecast, wind_forecast,
                                  load_forecast, ch_, dis_, gi_, cu_, s_)

    # ── Rule-based fallback ─────────────────────────────────────────────────

    def _rule_based_fallback(
        self,
        pv:   np.ndarray,
        wind: np.ndarray,
        load: np.ndarray,
        initial_soc: Optional[float],
    ) -> pd.DataFrame:
        """Greedy: surplus → charge; deficit → discharge → grid."""
        T    = len(pv)
        bat  = self.battery
        soc0 = initial_soc if initial_soc is not None else bat.initial_soc

        ch_, dis_, gi_, cu_, s_ = (np.zeros(T) for _ in range(5))
        soc_t = soc0

        for t in range(T):
            net = (pv[t] + wind[t]) - load[t]
            if net >= 0:
                ch_t = min(net, bat.max_charge_kw,
                           (bat.soc_max - soc_t) / bat.charge_eff)
                ch_t = max(ch_t, 0)
                cu_t = max(net - ch_t, 0)
                ch_[t], cu_[t] = ch_t, cu_t
            else:
                deficit = -net
                dis_t = min(deficit / bat.discharge_eff,
                            bat.max_discharge_kw,
                            (soc_t - bat.soc_min) * bat.discharge_eff)
                dis_t = max(dis_t, 0)
                gi_t  = max(deficit - dis_t * bat.discharge_eff, 0)
                dis_[t], gi_[t] = dis_t, gi_t

            soc_t = np.clip(
                soc_t + (ch_[t] * bat.charge_eff - dis_[t]) * self.dt,
                bat.soc_min, bat.soc_max,
            )
            s_[t] = soc_t

        return self._build_result(T, pv, wind, load, ch_, dis_, gi_, cu_, s_)

    # ── Assemble result ─────────────────────────────────────────────────────

    def _build_result(
        self, T, pv, wind, load, ch, dis, gi, cu, soc
    ) -> pd.DataFrame:
        renewable = pv + wind
        self_suff = np.where(
            load > 0,
            np.clip((load - gi) / load * 100, 0, 100),
            100.0,
        )
        return pd.DataFrame({
            "hour":               np.arange(T),
            "pv_kw":              np.clip(pv, 0, None).round(3),
            "wind_kw":            np.clip(wind, 0, None).round(3),
            "renewable_kw":       np.clip(renewable, 0, None).round(3),
            "load_kw":            load.round(3),
            "charge_kw":          np.clip(ch,  0, None).round(3),
            "discharge_kw":       np.clip(dis, 0, None).round(3),
            "grid_import_kw":     np.clip(gi,  0, None).round(3),
            "curtailment_kw":     np.clip(cu,  0, None).round(3),
            "soc_kwh":            soc.round(3),
            "net_fossil_kw":      np.clip(gi,  0, None).round(3),
            "self_sufficiency_pct": self_suff.round(2),
        })

    # ── KPI summary ─────────────────────────────────────────────────────────

    @staticmethod
    def compute_kpis(schedule: pd.DataFrame) -> Dict:
        total_load        = schedule["load_kw"].sum()
        total_renewable   = schedule["renewable_kw"].sum()
        total_grid_import = schedule["grid_import_kw"].sum()
        total_curtailed   = schedule["curtailment_kw"].sum()
        avg_soc           = schedule["soc_kwh"].mean()
        fossil_fraction   = (total_grid_import / total_load * 100) if total_load > 0 else 0
        renewable_fraction = ((total_load - total_grid_import) / total_load * 100) if total_load > 0 else 100

        return {
            "total_load_kwh":           round(total_load, 2),
            "total_renewable_kwh":      round(total_renewable, 2),
            "total_grid_import_kwh":    round(total_grid_import, 2),
            "total_curtailed_kwh":      round(total_curtailed, 2),
            "avg_battery_soc_kwh":      round(avg_soc, 2),
            "fossil_fraction_pct":      round(fossil_fraction, 2),
            "renewable_fraction_pct":   round(renewable_fraction, 2),
            "estimated_cost_usd":       round(total_grid_import * 0.12, 2),
        }
