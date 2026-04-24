"""
Curation / Hedge Analysis Module.
All analyses use common random numbers (CRN) for stable paired comparisons.
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.simulation.monte_carlo import run_monte_carlo, summarize_results
from src.optimization.correlation import compute_correlation_index


def run_standalone_profile(
    asset_state: pd.DataFrame, tranches: pd.DataFrame, params: dict,
    n_sims: int = 1000, seed: int = 42, duration: int = 36,
    contingency_mult: float = 1.10, annual_overhead: float = 0.0,
    upfront_threshold: float = 5_000_000.0,
    channel_lookup: dict | None = None,
) -> dict:
    results_df = run_monte_carlo(
        asset_state=asset_state, tranches=tranches, params=params,
        n_sims=n_sims, seed=seed, duration=duration,
        contingency_mult=contingency_mult, upfront_threshold=upfront_threshold,
        annual_overhead=annual_overhead, enable_tranche_kill=True, enable_rollover=False,
        channel_lookup=channel_lookup,
    )
    summary = summarize_results(results_df)
    corr_index = compute_correlation_index(asset_state, params.get("correlation", {}))
    n_assets = len(asset_state)
    exit_rate = summary["P_Exits_GE2"] if n_assets > 1 else (1.0 - summary["P_Zero_Exits"])
    total_capital = tranches["Budget"].sum() * contingency_mult
    if annual_overhead > 0:
        total_capital += annual_overhead * (duration / 12.0)
    return {
        "summary": summary, "results_df": results_df, "corr_index": corr_index,
        "n_assets": n_assets, "exit_rate": exit_rate, "total_capital": total_capital,
        "asset_ids": asset_state["Asset_ID"].tolist(),
    }


def run_portfolio_hedge(
    base_asset_state: pd.DataFrame, base_tranches: pd.DataFrame,
    candidate_asset_state: pd.DataFrame, candidate_tranches: pd.DataFrame,
    params: dict, n_sims: int = 1000, seed: int = 42, duration: int = 36,
    contingency_mult: float = 1.10, annual_overhead: float = 2_640_000.0,
    upfront_threshold: float = 5_000_000.0,
    channel_lookup: dict | None = None,
) -> dict:
    baseline_results = run_monte_carlo(
        asset_state=base_asset_state, tranches=base_tranches, params=params,
        n_sims=n_sims, seed=seed, duration=duration, contingency_mult=contingency_mult,
        upfront_threshold=upfront_threshold, annual_overhead=annual_overhead,
        enable_tranche_kill=True, enable_rollover=True,
        channel_lookup=channel_lookup,
    )
    baseline_summary = summarize_results(baseline_results)
    baseline_corr = compute_correlation_index(base_asset_state, params.get("correlation", {}))
    aug_asset_state = pd.concat([base_asset_state, candidate_asset_state], ignore_index=True)
    aug_tranches = pd.concat([base_tranches, candidate_tranches], ignore_index=True)
    aug_results = run_monte_carlo(
        asset_state=aug_asset_state, tranches=aug_tranches, params=params,
        n_sims=n_sims, seed=seed, duration=duration, contingency_mult=contingency_mult,
        upfront_threshold=upfront_threshold, annual_overhead=annual_overhead,
        enable_tranche_kill=True, enable_rollover=True,
        channel_lookup=channel_lookup,
    )
    aug_summary = summarize_results(aug_results)
    aug_corr = compute_correlation_index(aug_asset_state, params.get("correlation", {}))
    stress_baseline = run_monte_carlo(
        asset_state=base_asset_state, tranches=base_tranches, params=params,
        n_sims=n_sims, seed=seed, duration=duration, contingency_mult=contingency_mult,
        use_corr_stress=True, use_tight_only=True, upfront_threshold=upfront_threshold,
        annual_overhead=annual_overhead, enable_tranche_kill=True, enable_rollover=True,
        channel_lookup=channel_lookup,
    )
    stress_baseline_summary = summarize_results(stress_baseline)
    stress_aug = run_monte_carlo(
        asset_state=aug_asset_state, tranches=aug_tranches, params=params,
        n_sims=n_sims, seed=seed, duration=duration, contingency_mult=contingency_mult,
        use_corr_stress=True, use_tight_only=True, upfront_threshold=upfront_threshold,
        annual_overhead=annual_overhead, enable_tranche_kill=True, enable_rollover=True,
        channel_lookup=channel_lookup,
    )
    stress_aug_summary = summarize_results(stress_aug)
    def _delta(aug_s, base_s, key):
        a = aug_s.get(key, float("nan")); b = base_s.get(key, float("nan"))
        return a - b if pd.notna(a) and pd.notna(b) else float("nan")
    deltas = {
        "EDC": _delta(aug_summary, baseline_summary, "P_ThreePlus_Exits"),
        "IRC": _delta(aug_summary, baseline_summary, "Median_Annual_IRR"),
        "DPC": _delta(aug_summary, baseline_summary, "P_Exits_LE1"),
        "LAC": _delta(aug_summary, baseline_summary, "Median_First_Dist_Month"),
        "CDC": aug_corr - baseline_corr,
        "MOIC_Delta": _delta(aug_summary, baseline_summary, "Median_MOIC"),
    }
    stress_deltas = {
        "EDC_Stress": _delta(stress_aug_summary, stress_baseline_summary, "P_ThreePlus_Exits"),
        "IRC_Stress": _delta(stress_aug_summary, stress_baseline_summary, "Median_Annual_IRR"),
        "DPC_Stress": _delta(stress_aug_summary, stress_baseline_summary, "P_Exits_LE1"),
    }
    return {
        "baseline_summary": baseline_summary, "augmented_summary": aug_summary,
        "stress_baseline_summary": stress_baseline_summary,
        "stress_augmented_summary": stress_aug_summary,
        "baseline_results": baseline_results, "augmented_results": aug_results,
        "baseline_corr": baseline_corr, "augmented_corr": aug_corr,
        "deltas": deltas, "stress_deltas": stress_deltas,
        "candidate_ids": candidate_asset_state["Asset_ID"].tolist(),
    }


def run_hedge_sensitivity(
    base_asset_state: pd.DataFrame, base_tranches: pd.DataFrame,
    candidate_asset_state: pd.DataFrame, candidate_tranches: pd.DataFrame,
    params: dict, sweep_param: str = "Equity_to_IP_Pct",
    sweep_values: Optional[list] = None, n_sims: int = 500, seed: int = 42,
    duration: int = 36, contingency_mult: float = 1.10,
    annual_overhead: float = 2_640_000.0, upfront_threshold: float = 5_000_000.0,
    channel_lookup: dict | None = None,
) -> pd.DataFrame:
    if sweep_values is None:
        if sweep_param == "Equity_to_IP_Pct": sweep_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
        elif sweep_param == "AcqCash_to_IP": sweep_values = [0, 500_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000]
        elif sweep_param == "EarlyPassThrough_Pct": sweep_values = [0.0, 0.05, 0.10, 0.15, 0.20]
        elif sweep_param == "EarlyDeferredCash": sweep_values = [0, 250_000, 500_000, 1_000_000, 2_000_000]
        else: raise ValueError(f"Unknown sweep_param: {sweep_param}")
    rows = []
    for val in sweep_values:
        cand_state = candidate_asset_state.copy(); cand_state[sweep_param] = val
        aug_state = pd.concat([base_asset_state, cand_state], ignore_index=True)
        aug_tr = pd.concat([base_tranches, candidate_tranches.copy()], ignore_index=True)
        results = run_monte_carlo(
            asset_state=aug_state, tranches=aug_tr, params=params,
            n_sims=n_sims, seed=seed, duration=duration, contingency_mult=contingency_mult,
            upfront_threshold=upfront_threshold, annual_overhead=annual_overhead,
            enable_tranche_kill=True, enable_rollover=True,
            channel_lookup=channel_lookup,
        )
        s = summarize_results(results)
        rows.append({"Sweep_Param": sweep_param, "Sweep_Value": val,
            "Median_MOIC": s["Median_MOIC"], "Median_IRR": s["Median_Annual_IRR"],
            "IRR_P10": s["IRR_P10"], "P_3Plus_Exits": s["P_ThreePlus_Exits"],
            "P_LE1_Exit": s["P_Exits_LE1"], "P_Zero_Exits": s["P_Zero_Exits"],
            "Median_First_Dist": s["Median_First_Dist_Month"]})
    return pd.DataFrame(rows)


def run_asset_comparison(
    assets: list[dict], params: dict, n_sims: int = 1000, seed: int = 42,
    duration: int = 36, contingency_mult: float = 1.10,
    annual_overhead: float = 0.0, upfront_threshold: float = 5_000_000.0,
    channel_lookup: dict | None = None,
) -> pd.DataFrame:
    rows = []
    for asset_info in assets:
        profile = run_standalone_profile(
            asset_state=asset_info["asset_state"], tranches=asset_info["tranches"],
            params=params, n_sims=n_sims, seed=seed, duration=duration,
            contingency_mult=contingency_mult, annual_overhead=annual_overhead,
            upfront_threshold=upfront_threshold,
            channel_lookup=channel_lookup,
        )
        s = profile["summary"]
        rows.append({"Asset": asset_info["label"], "Asset_ID": ", ".join(profile["asset_ids"]),
            "Exit_Rate": profile["exit_rate"], "Median_MOIC": s["Median_MOIC"],
            "Median_IRR": s["Median_Annual_IRR"], "IRR_P10": s["IRR_P10"],
            "Total_Capital": profile["total_capital"], "Corr_Index": profile["corr_index"],
            "Median_First_Dist": s["Median_First_Dist_Month"]})
    return pd.DataFrame(rows)


def compute_marginal_contribution(
    portfolio_asset_state: pd.DataFrame, portfolio_tranches: pd.DataFrame,
    params: dict, n_sims: int = 1000, seed: int = 42, duration: int = 36,
    contingency_mult: float = 1.10, annual_overhead: float = 2_640_000.0,
    upfront_threshold: float = 5_000_000.0,
    channel_lookup: dict | None = None,
) -> pd.DataFrame:
    full_results = run_monte_carlo(
        asset_state=portfolio_asset_state, tranches=portfolio_tranches, params=params,
        n_sims=n_sims, seed=seed, duration=duration, contingency_mult=contingency_mult,
        upfront_threshold=upfront_threshold, annual_overhead=annual_overhead,
        enable_tranche_kill=True, enable_rollover=True,
        channel_lookup=channel_lookup,
    )
    full_summary = summarize_results(full_results)
    full_corr = compute_correlation_index(portfolio_asset_state, params.get("correlation", {}))
    rows = []
    for asset_id in portfolio_asset_state["Asset_ID"].unique():
        red_state = portfolio_asset_state[portfolio_asset_state["Asset_ID"] != asset_id].reset_index(drop=True)
        red_tr = portfolio_tranches[portfolio_tranches["Asset_ID"] != asset_id].reset_index(drop=True)
        if len(red_state) == 0:
            rows.append({"Asset_ID": asset_id, "EDC": full_summary["P_ThreePlus_Exits"],
                "IRC": full_summary["Median_Annual_IRR"], "DPC": -full_summary["P_Exits_LE1"],
                "LAC": float("nan"), "CDC": -full_corr, "MOIC_Delta": full_summary["Median_MOIC"]})
            continue
        red_results = run_monte_carlo(
            asset_state=red_state, tranches=red_tr, params=params,
            n_sims=n_sims, seed=seed, duration=duration, contingency_mult=contingency_mult,
            upfront_threshold=upfront_threshold, annual_overhead=annual_overhead,
            enable_tranche_kill=True, enable_rollover=True,
            channel_lookup=channel_lookup,
        )
        red_summary = summarize_results(red_results)
        red_corr = compute_correlation_index(red_state, params.get("correlation", {}))
        def _d(key):
            f = full_summary.get(key, float("nan")); r = red_summary.get(key, float("nan"))
            return f - r if pd.notna(f) and pd.notna(r) else float("nan")
        rows.append({"Asset_ID": asset_id, "EDC": _d("P_ThreePlus_Exits"),
            "IRC": _d("Median_Annual_IRR"), "DPC": _d("P_Exits_LE1"),
            "LAC": _d("Median_First_Dist_Month"), "CDC": full_corr - red_corr,
            "MOIC_Delta": _d("Median_MOIC")})
    return pd.DataFrame(rows)
