"""
Contribution engine.
Computes multi-axis contribution deltas (EDC/IRC/DPC/LAC/CDC) for candidate assets.
Uses common random numbers (CRN) for stable paired comparisons.
"""

import numpy as np
import pandas as pd

from src.simulation.monte_carlo import run_monte_carlo, summarize_results
from src.optimization.correlation import compute_correlation_index


def run_contribution_analysis(
    current_asset_state: pd.DataFrame,
    current_tranches: pd.DataFrame,
    candidate_asset_state: pd.DataFrame,
    candidate_tranches: pd.DataFrame,
    params: dict,
    n_sims: int = 5000,
    duration: int = 36,
    contingency_mult: float = 1.10,
    upfront_threshold: float = 5_000_000.0,
    channel_lookup: dict | None = None,
) -> dict:
    """
    Compute contribution profile for a candidate asset.

    Runs Monte Carlo twice:
      1. Portfolio WITHOUT candidate (baseline)
      2. Portfolio WITH candidate (augmented)

    Uses common random number seeds for stable deltas.

    Returns dict with EDC, IRC, DPC, LAC, CDC and Low/Base/High profiles.
    """
    seed_base = np.random.randint(0, 2**31)

    # --- Baseline: portfolio without candidate ---
    baseline_results = run_monte_carlo(
        asset_state=current_asset_state,
        tranches=current_tranches,
        params=params,
        n_sims=n_sims,
        duration=duration,
        contingency_mult=contingency_mult,
        upfront_threshold=upfront_threshold,
        seed=seed_base,
        channel_lookup=channel_lookup,
    )
    baseline_summary = summarize_results(baseline_results)
    baseline_corr = compute_correlation_index(current_asset_state, params.get("correlation", {}))

    # --- Augmented: portfolio with candidate ---
    aug_asset_state = pd.concat(
        [current_asset_state, candidate_asset_state], ignore_index=True
    )
    aug_tranches = pd.concat(
        [current_tranches, candidate_tranches], ignore_index=True
    )

    aug_results = run_monte_carlo(
        asset_state=aug_asset_state,
        tranches=aug_tranches,
        params=params,
        n_sims=n_sims,
        duration=duration,
        contingency_mult=contingency_mult,
        upfront_threshold=upfront_threshold,
        seed=seed_base,
        channel_lookup=channel_lookup,
    )
    aug_summary = summarize_results(aug_results)
    aug_corr = compute_correlation_index(aug_asset_state, params.get("correlation", {}))

    # --- Compute deltas (Base case) ---
    edc = aug_summary["P_ThreePlus_Exits"] - baseline_summary["P_ThreePlus_Exits"]
    irc = aug_summary["Median_Annual_IRR"] - baseline_summary["Median_Annual_IRR"]
    dpc = aug_summary["P_Exits_LE1"] - baseline_summary["P_Exits_LE1"]

    baseline_dist = baseline_summary.get("Median_First_Dist_Month", float("nan"))
    aug_dist = aug_summary.get("Median_First_Dist_Month", float("nan"))
    if pd.notna(baseline_dist) and pd.notna(aug_dist):
        lac = aug_dist - baseline_dist
    else:
        lac = float("nan")

    cdc = aug_corr - baseline_corr

    # --- Low/Base/High profiles ---
    # Low: tight regime, correlation stress
    low_results = run_monte_carlo(
        asset_state=aug_asset_state,
        tranches=aug_tranches,
        params=params,
        n_sims=n_sims,
        duration=duration,
        contingency_mult=contingency_mult,
        use_corr_stress=True,
        use_tight_only=True,
        upfront_threshold=upfront_threshold,
        seed=seed_base,
        channel_lookup=channel_lookup,
    )
    low_summary = summarize_results(low_results)

    low_baseline_results = run_monte_carlo(
        asset_state=current_asset_state,
        tranches=current_tranches,
        params=params,
        n_sims=n_sims,
        duration=duration,
        contingency_mult=contingency_mult,
        use_corr_stress=True,
        use_tight_only=True,
        upfront_threshold=upfront_threshold,
        seed=seed_base,
        channel_lookup=channel_lookup,
    )
    low_baseline_summary = summarize_results(low_baseline_results)

    edc_low = low_summary["P_ThreePlus_Exits"] - low_baseline_summary["P_ThreePlus_Exits"]
    irc_low = low_summary["Median_Annual_IRR"] - low_baseline_summary["Median_Annual_IRR"]
    dpc_low = low_summary["P_Exits_LE1"] - low_baseline_summary["P_Exits_LE1"]

    low_dist = low_summary.get("Median_First_Dist_Month", float("nan"))
    low_base_dist = low_baseline_summary.get("Median_First_Dist_Month", float("nan"))
    lac_low = (low_dist - low_base_dist) if pd.notna(low_dist) and pd.notna(low_base_dist) else float("nan")
    cdc_low = cdc  # correlation index is structural, not regime-dependent

    # High case: use base results with hot regime bias (approximate with base * 1.1 uplift)
    edc_high = edc * 1.1
    irc_high = irc * 1.1
    dpc_high = dpc * 0.9
    lac_high = lac * 0.95 if pd.notna(lac) else float("nan")
    cdc_high = cdc

    return {
        "baseline_summary": baseline_summary,
        "augmented_summary": aug_summary,
        "low_summary": low_summary,
        # Base deltas
        "EDC": round(edc, 4),
        "IRC": round(irc, 4),
        "DPC": round(dpc, 4),
        "LAC_Months": round(lac, 2) if pd.notna(lac) else None,
        "CDC": round(cdc, 4),
        # Low/Base/High
        "EDC_Low": round(edc_low, 4),
        "EDC_Base": round(edc, 4),
        "EDC_High": round(edc_high, 4),
        "IRC_Low": round(irc_low, 4),
        "IRC_Base": round(irc, 4),
        "IRC_High": round(irc_high, 4),
        "DPC_Low": round(dpc_low, 4),
        "DPC_Base": round(dpc, 4),
        "DPC_High": round(dpc_high, 4),
        "LAC_Low": round(lac_low, 2) if pd.notna(lac_low) else None,
        "LAC_Base": round(lac, 2) if pd.notna(lac) else None,
        "LAC_High": round(lac_high, 2) if pd.notna(lac_high) else None,
        "CDC_Low": round(cdc_low, 4),
        "CDC_Base": round(cdc, 4),
        "CDC_High": round(cdc_high, 4),
        # Metadata
        "baseline_corr_index": round(baseline_corr, 4),
        "augmented_corr_index": round(aug_corr, 4),
    }
