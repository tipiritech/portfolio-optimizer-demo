"""
Sensitivity Analysis Module.
Varies key inputs +/-20% and measures impact on portfolio metrics.
Produces tornado-style rankings showing which variables matter most.
"""

import numpy as np
import pandas as pd
from copy import deepcopy

from src.simulation.monte_carlo import run_monte_carlo, summarize_results
from src.optimization.correlation import compute_correlation_index


def _modify_triangular_params(params, table_name, col_prefix, multiplier):
    """Multiply all Min/Mode/Max values in a parameter table by a factor."""
    modified = deepcopy(params)
    df = modified[table_name].copy()
    for suffix in ["_Min", "_Mode", "_Max"]:
        col = f"{col_prefix}{suffix}"
        if col in df.columns:
            df[col] = df[col].astype(float) * multiplier
    modified[table_name] = df
    return modified


def _modify_regime_prob(params, regime_name, new_prob):
    """Set a specific regime probability and normalize others."""
    modified = deepcopy(params)
    df = modified["regime"].copy()
    other_total = 1.0 - new_prob
    other_count = len(df) - 1
    for i, row in df.iterrows():
        if row["Regime"] == regime_name:
            df.at[i, "Probability"] = new_prob
        else:
            df.at[i, "Probability"] = other_total / other_count
    modified["regime"] = df
    return modified


SENSITIVITY_FACTORS = [
    {
        "name": "Tech Probability (Tier-1)",
        "short": "Tech_Prob",
        "modify": lambda params, mult: _modify_triangular_params(params, "tier1_tech", "Tech", mult),
    },
    {
        "name": "Deal Probability (Tier-1)",
        "short": "Deal_Prob",
        "modify": lambda params, mult: _modify_triangular_params(params, "tier1_deal", "Deal", mult),
    },
    {
        "name": "Development Cost",
        "short": "Dev_Cost",
        "modify": lambda params, mult: _modify_triangular_params(params, "tier1_cost_time", "DevCost", mult),
    },
    {
        "name": "Time to Exit",
        "short": "Time_Exit",
        "modify": lambda params, mult: _modify_triangular_params(params, "tier1_cost_time", "TimeToExit", mult),
    },
    {
        "name": "Upfront Economics",
        "short": "Upfront",
        "modify": lambda params, mult: _modify_triangular_params(params, "tier1_econ", "Upfront", mult),
    },
    {
        "name": "Near-Term Milestones",
        "short": "Milestones",
        "modify": lambda params, mult: _modify_triangular_params(params, "tier1_econ", "NearMilestones", mult),
    },
    {
        "name": "Transaction Cost",
        "short": "Trans_Cost",
        "modify": lambda params, mult: _modify_triangular_params(params, "tier1_cost_time", "TransCost", mult),
    },
]


def run_sensitivity_analysis(
    asset_state: pd.DataFrame,
    tranches: pd.DataFrame,
    params: dict,
    n_sims: int = 2000,
    shock_pct: float = 0.20,
    duration: int = 36,
    annual_overhead: float = 2_640_000.0,
    channel_lookup: dict | None = None,
) -> dict:
    """
    Run sensitivity analysis: vary each factor by +/-shock_pct.

    Returns dict with:
      - base_summary: baseline results
      - sensitivities: list of dicts with factor name, up/down results, deltas
      - tornado_data: sorted by impact magnitude for tornado chart
    """
    seed = np.random.randint(0, 2**31)

    # Base case
    base_results = run_monte_carlo(
        asset_state=asset_state, tranches=tranches, params=params,
        n_sims=n_sims, duration=duration, seed=seed,
        annual_overhead=annual_overhead, enable_tranche_kill=True,
        channel_lookup=channel_lookup,
    )
    base_summary = summarize_results(base_results)

    sensitivities = []

    for factor in SENSITIVITY_FACTORS:
        # Upside shock
        params_up = factor["modify"](params, 1.0 + shock_pct)
        results_up = run_monte_carlo(
            asset_state=asset_state, tranches=tranches, params=params_up,
            n_sims=n_sims, duration=duration, seed=seed,
            annual_overhead=annual_overhead, enable_tranche_kill=True,
            channel_lookup=channel_lookup,
        )
        summary_up = summarize_results(results_up)

        # Downside shock
        params_down = factor["modify"](params, 1.0 - shock_pct)
        results_down = run_monte_carlo(
            asset_state=asset_state, tranches=tranches, params=params_down,
            n_sims=n_sims, duration=duration, seed=seed,
            annual_overhead=annual_overhead, enable_tranche_kill=True,
            channel_lookup=channel_lookup,
        )
        summary_down = summarize_results(results_down)

        # Compute deltas across key metrics
        metrics = ["Median_MOIC", "Median_Annual_IRR", "IRR_P10", "P_ThreePlus_Exits", "P_Exits_LE1"]
        deltas = {}
        for m in metrics:
            base_val = base_summary.get(m, 0)
            up_val = summary_up.get(m, 0)
            down_val = summary_down.get(m, 0)
            deltas[m] = {
                "base": base_val,
                "up": up_val,
                "down": down_val,
                "delta_up": up_val - base_val,
                "delta_down": down_val - base_val,
                "range": abs(up_val - down_val),
            }

        sensitivities.append({
            "factor": factor["name"],
            "short": factor["short"],
            "shock_pct": shock_pct,
            "deltas": deltas,
        })

    # Sort by impact on Median MOIC for tornado ranking
    tornado_data = sorted(
        sensitivities,
        key=lambda s: s["deltas"]["Median_MOIC"]["range"],
        reverse=True,
    )

    return {
        "base_summary": base_summary,
        "sensitivities": sensitivities,
        "tornado_data": tornado_data,
        "shock_pct": shock_pct,
        "n_sims": n_sims,
    }


def format_sensitivity_report(analysis: dict) -> str:
    """Format sensitivity analysis as a text report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  SENSITIVITY ANALYSIS")
    lines.append(f"  Shock: +/-{analysis['shock_pct']:.0%}  |  Simulations: {analysis['n_sims']:,}")
    lines.append("=" * 70)

    base = analysis["base_summary"]
    lines.append(f"\n  BASE CASE:")
    lines.append(f"    Median MOIC:     {base['Median_MOIC']:.2f}x")
    irr = base['Median_Annual_IRR']
    lines.append(f"    Median IRR:      {irr:.1%}" if pd.notna(irr) else "    Median IRR:      N/A")
    p10 = base['IRR_P10']
    lines.append(f"    IRR P10:         {p10:.1%}" if pd.notna(p10) else "    IRR P10:         N/A")
    lines.append(f"    P(3+ exits):     {base['P_ThreePlus_Exits']:.1%}")
    lines.append(f"    P(<=1 exit):     {base['P_Exits_LE1']:.1%}")

    lines.append(f"\n  TORNADO RANKING (by MOIC impact):")
    lines.append(f"  {'Factor':<25} {'Down':>10} {'Base':>10} {'Up':>10} {'Range':>10}")
    lines.append(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for s in analysis["tornado_data"]:
        d = s["deltas"]["Median_MOIC"]
        lines.append(f"  {s['factor']:<25} {d['down']:>10.2f}x {d['base']:>10.2f}x {d['up']:>10.2f}x {d['range']:>10.2f}x")

    lines.append(f"\n  P(3+ EXITS) SENSITIVITY:")
    lines.append(f"  {'Factor':<25} {'Down':>10} {'Base':>10} {'Up':>10}")
    lines.append(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

    for s in analysis["tornado_data"]:
        d = s["deltas"]["P_ThreePlus_Exits"]
        lines.append(f"  {s['factor']:<25} {d['down']:>10.1%} {d['base']:>10.1%} {d['up']:>10.1%}")

    lines.append("")
    return "\n".join(lines)
