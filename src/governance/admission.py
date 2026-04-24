"""
Admission logic.
Checks all admission gates from the Governance Doctrine for candidate assets.

An asset may be admitted only if:
  1. Δ P(≥3 exits) > 0  (EDC positive)
  2. Median IRR ≥ target
  3. 10th percentile IRR ≥ floor
  4. P(≤1 exit) ≤ catastrophic ceiling
  5. Correlation ceiling not breached
  6. Tight-regime stress case passes
"""

import pandas as pd
from src.optimization.contribution import run_contribution_analysis
from src.simulation.monte_carlo import run_monte_carlo, summarize_results
from src.optimization.correlation import compute_correlation_index


def check_admission_gates(
    contribution: dict,
    envelope: dict,
    edc_stop_threshold: float = 0.01,
    current_portfolio_size: int = 0,
    hard_cap_assets: int = 12,
) -> dict:
    """
    Check all admission gates for a candidate asset.

    Args:
        contribution: output from run_contribution_analysis
        envelope: envelope thresholds dict
        edc_stop_threshold: minimum marginal EDC (default 1%)
        current_portfolio_size: number of assets currently in portfolio
        hard_cap_assets: maximum allowed assets (default 12)

    Returns:
        dict with gate results, overall pass/fail, and failed gates list
    """
    aug = contribution["augmented_summary"]
    low = contribution["low_summary"]

    target_irr = envelope.get("Target_Median_IRR", 0.25)
    floor_irr = envelope.get("Floor_IRR_P10", 0.0)
    max_p_le1 = envelope.get("Max_P_Exits_LE1", 0.15)
    max_corr = envelope.get("Max_CorrIndex", 0.9)

    gates = {}

    # Gate 0: Hard cap on portfolio size
    proposed_size = current_portfolio_size + 1
    gates["Hard_Cap"] = {
        "rule": f"Portfolio size ≤ {hard_cap_assets}",
        "actual": proposed_size,
        "threshold": hard_cap_assets,
        "pass": proposed_size <= hard_cap_assets,
    }

    # Gate 1: EDC > 0 (exit density contribution positive)
    edc = contribution["EDC"]
    gates["EDC_Positive"] = {
        "rule": "Δ P(≥3 exits) > 0",
        "actual": edc,
        "threshold": 0.0,
        "pass": edc > 0,
    }

    # Gate 1b: EDC above stop threshold (admission density guardrail)
    gates["EDC_Above_Stop"] = {
        "rule": f"Δ P(≥3 exits) ≥ {edc_stop_threshold:.1%}",
        "actual": edc,
        "threshold": edc_stop_threshold,
        "pass": edc >= edc_stop_threshold,
    }

    # Gate 2: Median IRR ≥ target (with candidate included)
    median_irr = aug.get("Median_Annual_IRR", float("nan"))
    gates["Median_IRR"] = {
        "rule": f"Median IRR ≥ {target_irr:.0%}",
        "actual": median_irr,
        "threshold": target_irr,
        "pass": pd.notna(median_irr) and median_irr >= target_irr,
    }

    # Gate 3: IRR P10 ≥ floor (with candidate included)
    p10_irr = aug.get("IRR_P10", float("nan"))
    gates["IRR_P10"] = {
        "rule": f"IRR P10 ≥ {floor_irr:.0%}",
        "actual": p10_irr,
        "threshold": floor_irr,
        "pass": pd.notna(p10_irr) and p10_irr >= floor_irr,
    }

    # Gate 4: P(≤1 exit) ≤ catastrophic ceiling
    p_le1 = aug.get("P_Exits_LE1", 1.0)
    gates["Catastrophic"] = {
        "rule": f"P(≤1 exit) ≤ {max_p_le1:.0%}",
        "actual": p_le1,
        "threshold": max_p_le1,
        "pass": p_le1 <= max_p_le1,
    }

    # Gate 5: Correlation ceiling
    aug_corr = contribution.get("augmented_corr_index", 0.0)
    gates["Correlation"] = {
        "rule": f"Corr Index ≤ {max_corr}",
        "actual": aug_corr,
        "threshold": max_corr,
        "pass": aug_corr <= max_corr,
    }

    # Gate 6: Tight-regime stress case passes
    # Under Low (tight + corr stress), key metrics must still hold
    low_median_irr = low.get("Median_Annual_IRR", float("nan"))
    low_p_le1 = low.get("P_Exits_LE1", 1.0)
    stress_pass = (
        pd.notna(low_median_irr)
        and low_median_irr >= floor_irr
        and low_p_le1 <= max_p_le1 * 1.5  # Allow 50% relaxation under stress
    )
    gates["Tight_Stress"] = {
        "rule": "Stress case (Tight + corr stress) within bounds",
        "actual": f"IRR={low_median_irr:.1%}, P(≤1)={low_p_le1:.1%}" if pd.notna(low_median_irr) else "N/A",
        "threshold": f"IRR≥{floor_irr:.0%}, P(≤1)≤{max_p_le1*1.5:.0%}",
        "pass": stress_pass,
    }

    # Overall
    all_pass = all(g["pass"] for g in gates.values())
    failed = [k for k, v in gates.items() if not v["pass"]]

    return {
        "gates": gates,
        "all_pass": all_pass,
        "failed_gates": failed,
        "requires_override": not all_pass,
        "override_type": "Board Supermajority (≥80%)" if not all_pass else None,
    }


def format_admission_report(
    candidate_id: str,
    contribution: dict,
    gate_result: dict,
) -> str:
    """Format a text report for an admission decision."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"  ADMISSION ANALYSIS: {candidate_id}")
    lines.append("=" * 60)

    # Contribution profile
    lines.append("\n  Contribution Profile (Base Case):")
    lines.append(f"    EDC (Δ P(≥3 exits)):     {contribution['EDC']:+.2%}")
    lines.append(f"    IRC (Δ Median IRR):       {contribution['IRC']:+.4f}")
    lines.append(f"    DPC (Δ P(≤1 exit)):       {contribution['DPC']:+.2%}")
    if contribution['LAC_Months'] is not None:
        lines.append(f"    LAC (Δ 1st Dist Month):   {contribution['LAC_Months']:+.1f} months")
    lines.append(f"    CDC (Δ Corr Index):        {contribution['CDC']:+.4f}")

    # Low/Base/High
    lines.append("\n  Low / Base / High:")
    lines.append(f"    EDC:  {contribution['EDC_Low']:+.2%} / {contribution['EDC_Base']:+.2%} / {contribution['EDC_High']:+.2%}")
    lines.append(f"    IRC:  {contribution['IRC_Low']:+.4f} / {contribution['IRC_Base']:+.4f} / {contribution['IRC_High']:+.4f}")
    lines.append(f"    DPC:  {contribution['DPC_Low']:+.2%} / {contribution['DPC_Base']:+.2%} / {contribution['DPC_High']:+.2%}")

    # Gate results
    lines.append("\n  Admission Gates:")
    for name, gate in gate_result["gates"].items():
        status = "PASS" if gate["pass"] else "FAIL"
        lines.append(f"    {name:20s}  {status}  ({gate['rule']})")

    # Verdict
    lines.append("")
    if gate_result["all_pass"]:
        lines.append("  VERDICT: ADMITTED")
    else:
        lines.append(f"  VERDICT: REQUIRES OVERRIDE")
        lines.append(f"  Failed: {', '.join(gate_result['failed_gates'])}")
        lines.append(f"  Override: {gate_result['override_type']}")

    lines.append("")
    return "\n".join(lines)
