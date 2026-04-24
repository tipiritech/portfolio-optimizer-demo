"""
Maximum Economic Capacity (MEC) solver.
Finds the maximum inventor/IP economics such that all governance constraints remain satisfied.

Searches across levers:
  1. AcqCash_to_IP (primary)
  2. Equity_to_IP_Pct
  3. EarlyPassThrough_Pct
  4. EarlyDeferredCash

Uses binary search on primary lever (AcqCash) for efficiency.
"""

import numpy as np
import pandas as pd

from src.simulation.monte_carlo import run_monte_carlo, summarize_results
from src.optimization.correlation import compute_correlation_index


def _passes_constraints(summary: dict, corr_index: float, envelope: dict) -> tuple[bool, str | None]:
    """
    Check if simulation results pass all governance constraints.
    Returns (pass_bool, first_failed_constraint_name).
    """
    target_irr = envelope.get("Target_Median_IRR", 0.25)
    floor_irr = envelope.get("Floor_IRR_P10", 0.0)
    min_p3 = envelope.get("Min_P_Exits_GE3", 0.60)
    max_p_le1 = envelope.get("Max_P_Exits_LE1", 0.15)
    max_corr = envelope.get("Max_CorrIndex", 0.9)

    checks = [
        ("Median_IRR", pd.notna(summary.get("Median_Annual_IRR")) and summary["Median_Annual_IRR"] >= target_irr),
        ("IRR_P10", pd.notna(summary.get("IRR_P10")) and summary["IRR_P10"] >= floor_irr),
        ("P_Exits_GE3", summary.get("P_ThreePlus_Exits", 0) >= min_p3),
        ("P_Exits_LE1", summary.get("P_Exits_LE1", 1) <= max_p_le1),
        ("Corr_Index", corr_index <= max_corr),
    ]

    for name, passed in checks:
        if not passed:
            return False, name

    return True, None


def _run_with_economics(
    current_asset_state: pd.DataFrame,
    current_tranches: pd.DataFrame,
    candidate_row: pd.Series,
    candidate_tranches: pd.DataFrame,
    acq_cash: float,
    equity_pct: float,
    passthru_pct: float,
    deferred_cash: float,
    params: dict,
    n_sims: int,
    duration: int,
    contingency_mult: float,
    upfront_threshold: float,
    seed: int,
    channel_lookup: dict | None = None,
) -> tuple[dict, float]:
    """Run Monte Carlo with specific IP economics and return summary + corr_index."""
    candidate = candidate_row.copy()
    candidate["AcqCash_to_IP"] = acq_cash
    candidate["Equity_to_IP_Pct"] = equity_pct
    candidate["EarlyPassThrough_Pct"] = passthru_pct
    candidate["EarlyDeferredCash"] = deferred_cash

    candidate_df = pd.DataFrame([candidate])
    aug_state = pd.concat([current_asset_state, candidate_df], ignore_index=True)
    aug_tranches = pd.concat([current_tranches, candidate_tranches], ignore_index=True)

    results = run_monte_carlo(
        asset_state=aug_state,
        tranches=aug_tranches,
        params=params,
        n_sims=n_sims,
        duration=duration,
        contingency_mult=contingency_mult,
        upfront_threshold=upfront_threshold,
        seed=seed,
        channel_lookup=channel_lookup,
    )
    summary = summarize_results(results)
    corr_index = compute_correlation_index(aug_state, params.get("correlation", {}))

    return summary, corr_index


def solve_mec(
    current_asset_state: pd.DataFrame,
    current_tranches: pd.DataFrame,
    candidate_row: pd.Series,
    candidate_tranches: pd.DataFrame,
    params: dict,
    envelope: dict,
    n_sims: int = 3000,
    duration: int = 36,
    contingency_mult: float = 1.10,
    upfront_threshold: float = 5_000_000.0,
    acq_cash_range: tuple = (0, 25_000_000, 500_000),
    equity_range: tuple = (0.0, 0.40, 0.02),
    passthru_range: tuple = (0.0,),
    deferred_range: tuple = (0.0,),
    channel_lookup: dict | None = None,
) -> dict:
    """
    Find the Maximum Economic Capacity for a candidate asset.

    Strategy:
      1. Verify candidate is admissible at minimal economics (AC=0, equity=0)
      2. Binary search on AcqCash to find breakpoint
      3. For each AC level, optionally search equity

    Returns dict with MEC breakpoints and first-failed constraint.
    """
    seed = np.random.randint(0, 2**31)

    # Step 1: Check admissibility at zero economics
    summary_zero, corr_zero = _run_with_economics(
        current_asset_state, current_tranches,
        candidate_row, candidate_tranches,
        acq_cash=0, equity_pct=0, passthru_pct=0, deferred_cash=0,
        params=params, n_sims=n_sims, duration=duration,
        contingency_mult=contingency_mult, upfront_threshold=upfront_threshold,
        seed=seed, channel_lookup=channel_lookup,
    )

    passes_zero, fail_zero = _passes_constraints(summary_zero, corr_zero, envelope)
    if not passes_zero:
        return {
            "admissible": False,
            "reason": f"Not admissible even at zero economics (failed: {fail_zero})",
            "MEC_AcqCash": 0,
            "MEC_Equity": 0,
            "MEC_PassThrough": 0,
            "MEC_DeferredCash": 0,
            "first_failed_constraint": fail_zero,
        }

    # Step 2: Binary search on AcqCash
    ac_min, ac_max, ac_step = acq_cash_range
    ac_low = ac_min
    ac_high = ac_max
    best_ac = 0
    first_fail_constraint = None

    summary_max, corr_max = _run_with_economics(
        current_asset_state, current_tranches,
        candidate_row, candidate_tranches,
        acq_cash=ac_high, equity_pct=0, passthru_pct=0, deferred_cash=0,
        params=params, n_sims=n_sims, duration=duration,
        contingency_mult=contingency_mult, upfront_threshold=upfront_threshold,
        seed=seed, channel_lookup=channel_lookup,
    )
    passes_max, _ = _passes_constraints(summary_max, corr_max, envelope)

    if passes_max:
        best_ac = ac_high
    else:
        while ac_high - ac_low > ac_step:
            mid = round((ac_low + ac_high) / 2 / ac_step) * ac_step

            summary_mid, corr_mid = _run_with_economics(
                current_asset_state, current_tranches,
                candidate_row, candidate_tranches,
                acq_cash=mid, equity_pct=0, passthru_pct=0, deferred_cash=0,
                params=params, n_sims=n_sims, duration=duration,
                contingency_mult=contingency_mult, upfront_threshold=upfront_threshold,
                seed=seed, channel_lookup=channel_lookup,
            )

            passes_mid, fail_mid = _passes_constraints(summary_mid, corr_mid, envelope)

            if passes_mid:
                ac_low = mid
                best_ac = mid
            else:
                ac_high = mid
                first_fail_constraint = fail_mid

        best_ac = ac_low

    # Step 3: At best AC, search equity
    best_equity = 0.0
    eq_min, eq_max, eq_step = equity_range[0], equity_range[1], equity_range[2] if len(equity_range) > 2 else 0.02

    summary_eq_max, corr_eq_max = _run_with_economics(
        current_asset_state, current_tranches,
        candidate_row, candidate_tranches,
        acq_cash=best_ac, equity_pct=eq_max, passthru_pct=0, deferred_cash=0,
        params=params, n_sims=n_sims, duration=duration,
        contingency_mult=contingency_mult, upfront_threshold=upfront_threshold,
        seed=seed, channel_lookup=channel_lookup,
    )
    passes_eq_max, _ = _passes_constraints(summary_eq_max, corr_eq_max, envelope)

    if passes_eq_max:
        best_equity = eq_max
    else:
        eq_low, eq_high = eq_min, eq_max
        while eq_high - eq_low > eq_step:
            mid_eq = round((eq_low + eq_high) / 2 / eq_step) * eq_step

            summary_eq, corr_eq = _run_with_economics(
                current_asset_state, current_tranches,
                candidate_row, candidate_tranches,
                acq_cash=best_ac, equity_pct=mid_eq, passthru_pct=0, deferred_cash=0,
                params=params, n_sims=n_sims, duration=duration,
                contingency_mult=contingency_mult, upfront_threshold=upfront_threshold,
                seed=seed, channel_lookup=channel_lookup,
            )
            passes_eq, fail_eq = _passes_constraints(summary_eq, corr_eq, envelope)

            if passes_eq:
                eq_low = mid_eq
                best_equity = mid_eq
            else:
                eq_high = mid_eq
                if first_fail_constraint is None:
                    first_fail_constraint = fail_eq

    return {
        "admissible": True,
        "MEC_AcqCash": best_ac,
        "MEC_Equity": best_equity,
        "MEC_PassThrough": 0.0,
        "MEC_DeferredCash": 0.0,
        "first_failed_constraint": first_fail_constraint,
        "search_details": {
            "ac_range": acq_cash_range,
            "equity_range": equity_range,
            "n_sims": n_sims,
        },
    }


def format_mec_report(candidate_id: str, mec_result: dict) -> str:
    """Format MEC results as a text report."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"  MEC ANALYSIS: {candidate_id}")
    lines.append("=" * 60)

    if not mec_result["admissible"]:
        lines.append(f"\n  NOT ADMISSIBLE: {mec_result['reason']}")
    else:
        lines.append(f"\n  Maximum Economic Capacity:")
        lines.append(f"    AcqCash to IP:       ${mec_result['MEC_AcqCash']:,.0f}")
        lines.append(f"    Equity to IP:         {mec_result['MEC_Equity']:.0%}")
        lines.append(f"    Pass-Through:         {mec_result['MEC_PassThrough']:.0%}")
        lines.append(f"    Deferred Cash:        ${mec_result['MEC_DeferredCash']:,.0f}")

        if mec_result.get("first_failed_constraint"):
            lines.append(f"\n  First constraint at boundary: {mec_result['first_failed_constraint']}")

    lines.append("")
    return "\n".join(lines)
