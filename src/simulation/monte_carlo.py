"""
Monte Carlo simulation engine.
Includes: tranche kill (#4), IRR cap (#10), overhead (#12), rollover valuation (#15).
"""

import numpy as np
import pandas as pd

from src.data.cashflows import build_asset_monthly_outflows
from src.data.inflows import simulate_asset_inflows
from src.data.net_cashflows import build_asset_net_cashflows, build_portfolio_cashflow
from src.metrics import monthly_irr, annualize_monthly_irr, compute_moic

IRR_CAP = 10.0  # #10: Cap annual IRR at 1000% to prevent outlier distortion


def _apply_tranche_kill(outflows: dict, inflow_df: pd.DataFrame, duration: int = 36) -> dict:
    """
    #4: Cancel remaining capital deployment for assets that fail.

    For each failed asset, zero out outflows from the exit_month onward.
    If exit_month is None (never reached exit), use the relative_exit_month
    as the point where failure would be realized.
    """
    killed_outflows = {}
    for asset_id, vec in outflows.items():
        asset_rows = inflow_df[inflow_df["Asset_ID"] == asset_id]
        if asset_rows.empty:
            killed_outflows[asset_id] = vec.copy()
            continue

        row = asset_rows.iloc[0]
        if not row["Success"]:
            # Estimate when failure is realized: use relative exit month or midpoint
            rel_exit = row.get("Relative_Exit_Month")
            entry = row.get("Entry_Month", 0)
            if pd.notna(rel_exit):
                fail_month = int(entry + rel_exit)
            else:
                fail_month = int(entry + 12)  # conservative: realize failure after 12 months

            fail_month = min(max(fail_month, 0), duration)
            new_vec = vec.copy()
            # Keep spending up to fail_month, zero out after
            if fail_month + 1 <= duration:
                new_vec[fail_month + 1:] = 0.0
            killed_outflows[asset_id] = new_vec
        else:
            killed_outflows[asset_id] = vec.copy()

    return killed_outflows


def _apply_overhead(portfolio_cf: np.ndarray, annual_overhead: float, duration: int = 36) -> np.ndarray:
    """
    #12: Subtract monthly operating overhead from portfolio cashflows.
    Overhead = annual_overhead / 12 per month.
    """
    monthly_oh = annual_overhead / 12.0
    adjusted = portfolio_cf.copy()
    for m in range(1, min(duration + 1, len(adjusted))):  # skip month 0
        adjusted[m] -= monthly_oh
    return adjusted


def _compute_rollover_value(inflow_df: pd.DataFrame, params: dict, duration: int = 36) -> float:
    """
    #15: Estimate residual value for unsold assets at month 36.
    Uses Low/Base midpoint of expected upfront as terminal value.
    Only applies to assets that were technically successful but didn't exit in time.
    """
    from src.data.loader import get_tier_tables
    from src.governance.param_lookup import build_ds_ra_lookup

    rollover_value = 0.0
    for _, row in inflow_df.iterrows():
        # Asset that succeeded technically but exit was beyond duration
        if not row["Success"] and pd.notna(row.get("Relative_Exit_Month")):
            rel_exit = row["Relative_Exit_Month"]
            entry = row.get("Entry_Month", 0)
            actual_exit = entry + rel_exit
            # Only count if exit would have happened within reasonable window (36-48 months)
            if duration < actual_exit <= duration + 12:
                ds = row["DS_Current"]
                ra = row["RA_Current"]
                tier = row.get("Tier", "Tier-1")
                tables = get_tier_tables(params, tier)
                econ_lk = build_ds_ra_lookup(tables["econ"],
                    ["Upfront_Min", "Upfront_Mode", "Upfront_Max",
                     "NearMilestones_Min", "NearMilestones_Mode", "NearMilestones_Max"])
                econ = econ_lk.get((ds, ra))
                if econ:
                    # Low/Base midpoint
                    low_upfront = econ[0]
                    base_upfront = econ[1]
                    midpoint = (low_upfront + base_upfront) / 2.0
                    # Discount by probability it would actually close
                    rollover_value += midpoint * 0.5  # 50% haircut for rollover
    return rollover_value


def run_one_simulation(
    asset_state: pd.DataFrame,
    tranches: pd.DataFrame,
    params: dict,
    corr_config: dict,
    duration: int = 36,
    contingency_mult: float = 1.10,
    use_corr_stress: bool = False,
    use_tight_only: bool = False,
    upfront_threshold: float = 5_000_000.0,
    annual_overhead: float = 0.0,
    enable_tranche_kill: bool = True,
    enable_rollover: bool = False,
    channel_lookup: dict | None = None,
) -> dict:
    """Run one full portfolio simulation."""
    # Build outflows from tranches
    outflows = build_asset_monthly_outflows(
        tranches, asset_state, duration=duration, contingency_mult=contingency_mult
    )

    # Simulate inflows
    inflow_df = simulate_asset_inflows(
        asset_state=asset_state,
        params=params,
        corr_config=corr_config,
        use_corr_stress=use_corr_stress,
        use_tight_only=use_tight_only,
        duration=duration,
        upfront_threshold=upfront_threshold,
        channel_lookup=channel_lookup,
    )

    # #4: Tranche kill — cancel remaining spend for failed assets
    if enable_tranche_kill:
        outflows = _apply_tranche_kill(outflows, inflow_df, duration)

    # Build net cashflows
    net_cashflows = build_asset_net_cashflows(outflows, inflow_df, duration=duration)
    portfolio_cf = build_portfolio_cashflow(net_cashflows, duration=duration)

    # #12: Subtract overhead
    if annual_overhead > 0:
        portfolio_cf = _apply_overhead(portfolio_cf, annual_overhead, duration)

    # #15: Add rollover value at terminal month
    rollover_val = 0.0
    if enable_rollover:
        rollover_val = _compute_rollover_value(inflow_df, params, duration)
        if rollover_val > 0 and duration < len(portfolio_cf):
            portfolio_cf[duration] += rollover_val

    # Compute metrics
    total_outflows = sum(v.sum() for v in outflows.values())
    if annual_overhead > 0:
        total_outflows += annual_overhead * (duration / 12.0)
    total_inflows = inflow_df["Total_Inflow"].sum() + rollover_val
    moic = compute_moic(total_inflows, total_outflows)
    irr_m = monthly_irr(portfolio_cf)
    irr_a = annualize_monthly_irr(irr_m)

    # #10: Cap extreme IRR values
    if pd.notna(irr_a) and irr_a > IRR_CAP:
        irr_a = IRR_CAP

    num_exits = int(inflow_df["Success"].sum())

    # Time to first distribution
    first_dist_month = None
    for m in range(len(portfolio_cf)):
        if portfolio_cf[m] > 0:
            first_dist_month = m
            break

    return {
        "Num_Exits": num_exits,
        "Total_Outflows": round(total_outflows, 2),
        "Total_Inflows": round(total_inflows, 2),
        "MOIC": moic,
        "Monthly_IRR": irr_m,
        "Annual_IRR": irr_a,
        "First_Dist_Month": first_dist_month,
        "Rollover_Value": round(rollover_val, 2),
    }


def run_monte_carlo(
    asset_state: pd.DataFrame,
    tranches: pd.DataFrame,
    params: dict,
    n_sims: int = 1000,
    duration: int = 36,
    contingency_mult: float = 1.10,
    use_corr_stress: bool = False,
    use_tight_only: bool = False,
    upfront_threshold: float = 5_000_000.0,
    seed: int | None = None,
    annual_overhead: float = 0.0,
    enable_tranche_kill: bool = True,
    enable_rollover: bool = False,
    channel_lookup: dict | None = None,
) -> pd.DataFrame:
    """Run N Monte Carlo simulations."""
    if seed is not None:
        np.random.seed(seed)

    corr_config = params.get("correlation", {})
    results = []

    for sim in range(n_sims):
        sim_result = run_one_simulation(
            asset_state=asset_state,
            tranches=tranches,
            params=params,
            corr_config=corr_config,
            duration=duration,
            contingency_mult=contingency_mult,
            use_corr_stress=use_corr_stress,
            use_tight_only=use_tight_only,
            upfront_threshold=upfront_threshold,
            annual_overhead=annual_overhead,
            enable_tranche_kill=enable_tranche_kill,
            enable_rollover=enable_rollover,
            channel_lookup=channel_lookup,
        )
        sim_result["Sim_ID"] = sim + 1
        results.append(sim_result)

    return pd.DataFrame(results)


def summarize_results(results_df: pd.DataFrame) -> dict:
    """Compute summary statistics from Monte Carlo results."""
    valid_irr = results_df["Annual_IRR"].dropna()
    valid_moic = results_df["MOIC"].dropna()
    valid_dist = results_df["First_Dist_Month"].dropna()

    return {
        "Num_Sims": len(results_df),
        "Mean_MOIC": valid_moic.mean() if len(valid_moic) else float("nan"),
        "Median_MOIC": valid_moic.median() if len(valid_moic) else float("nan"),
        "MOIC_P10": valid_moic.quantile(0.10) if len(valid_moic) else float("nan"),
        "Mean_Annual_IRR": valid_irr.mean() if len(valid_irr) else float("nan"),
        "Median_Annual_IRR": valid_irr.median() if len(valid_irr) else float("nan"),
        "IRR_P10": valid_irr.quantile(0.10) if len(valid_irr) else float("nan"),
        "P_Zero_Exits": float((results_df["Num_Exits"] == 0).mean()),
        "P_One_Exit": float((results_df["Num_Exits"] == 1).mean()),
        "P_Two_Exits": float((results_df["Num_Exits"] == 2).mean()),
        "P_ThreePlus_Exits": float((results_df["Num_Exits"] >= 3).mean()),
        "P_Exits_LE1": float((results_df["Num_Exits"] <= 1).mean()),
        "P_Exits_GE2": float((results_df["Num_Exits"] >= 2).mean()),
        "Median_First_Dist_Month": valid_dist.median() if len(valid_dist) else float("nan"),
    }
