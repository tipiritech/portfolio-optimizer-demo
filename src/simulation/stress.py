"""
Stress Testing Suite & Reproducibility Verification.

Three mandated stress scenarios:
  1. Tight regime only (worst market conditions)
  2. Correlation stress (+0.10 on base floor)
  3. Combined worst-case (tight + correlation stress)

Plus reproducibility verification (same seed = same output).
"""

import numpy as np
import pandas as pd

from src.simulation.monte_carlo import run_monte_carlo, summarize_results
from src.optimization.correlation import compute_correlation_index
from src.optimization.envelope import check_envelope


def run_stress_suite(
    asset_state: pd.DataFrame,
    tranches: pd.DataFrame,
    params: dict,
    envelope: dict,
    n_sims: int = 5000,
    duration: int = 36,
    annual_overhead: float = 2_640_000.0,
    channel_lookup: dict | None = None,
) -> dict:
    """
    Run all three mandated stress scenarios plus base case.

    Returns dict with base + 3 stress scenario results and envelope checks.
    Computes weighted_time and passes to envelope.
    Under correlation stress, recomputes corr_index with stress add-on.
    """
    from src.optimization.envelope import compute_weighted_time, check_capital_concentration

    seed = np.random.randint(0, 2**31)
    corr_config = params.get("correlation", {})

    # Compute weighted time for envelope gate
    weighted_time = compute_weighted_time(asset_state, params)

    # Compute capital concentration for envelope gate
    conc = check_capital_concentration(tranches)

    # Base correlation index (no stress)
    corr_index_base = compute_correlation_index(asset_state, corr_config)

    # Stressed correlation index: add stress_add to base_floor, increasing pairwise overlap
    stress_add = corr_config.get("CorrelationStressAdd", {}).get("Default_Loading", 0.10)
    corr_index_stressed = corr_index_base + stress_add

    scenarios = {
        "Base Case": {"use_corr_stress": False, "use_tight_only": False},
        "Tight Regime Only": {"use_corr_stress": False, "use_tight_only": True},
        "Correlation Stress": {"use_corr_stress": True, "use_tight_only": False},
        "Combined Worst-Case": {"use_corr_stress": True, "use_tight_only": True},
    }

    results = {}

    for scenario_name, kwargs in scenarios.items():
        sim_results = run_monte_carlo(
            asset_state=asset_state, tranches=tranches, params=params,
            n_sims=n_sims, duration=duration, seed=seed,
            annual_overhead=annual_overhead, enable_tranche_kill=True,
            channel_lookup=channel_lookup,
            **kwargs,
        )
        summary = summarize_results(sim_results)

        # Use stressed corr_index for scenarios with correlation stress
        ci = corr_index_stressed if kwargs["use_corr_stress"] else corr_index_base

        env_check = check_envelope(
            summary, envelope, corr_index=ci,
            weighted_time=weighted_time,
            concentration_issues=conc.get("breaches", []),
        )

        results[scenario_name] = {
            "summary": summary,
            "envelope": env_check,
            "all_pass": env_check["all_pass"],
            "failed_gates": env_check["failed_gates"],
            "corr_index": ci,
            "weighted_time": weighted_time,
        }

    return results


def verify_reproducibility(
    asset_state: pd.DataFrame,
    tranches: pd.DataFrame,
    params: dict,
    n_sims: int = 500,
    duration: int = 36,
) -> dict:
    """
    Verify deterministic reproducibility: same seed = identical results.

    Runs the simulation twice with the same seed and compares outputs.
    """
    seed = 42  # fixed seed for reproducibility test

    results_a = run_monte_carlo(
        asset_state=asset_state, tranches=tranches, params=params,
        n_sims=n_sims, duration=duration, seed=seed,
    )
    summary_a = summarize_results(results_a)

    results_b = run_monte_carlo(
        asset_state=asset_state, tranches=tranches, params=params,
        n_sims=n_sims, duration=duration, seed=seed,
    )
    summary_b = summarize_results(results_b)

    # Compare key metrics
    metrics_to_check = [
        "Median_MOIC", "Median_Annual_IRR", "IRR_P10",
        "P_ThreePlus_Exits", "P_Exits_LE1",
    ]

    matches = {}
    all_match = True
    for m in metrics_to_check:
        a_val = summary_a.get(m, float("nan"))
        b_val = summary_b.get(m, float("nan"))
        if pd.isna(a_val) and pd.isna(b_val):
            match = True
        elif pd.isna(a_val) or pd.isna(b_val):
            match = False
        else:
            match = abs(a_val - b_val) < 1e-10
        matches[m] = {"run_a": a_val, "run_b": b_val, "match": match}
        if not match:
            all_match = False

    return {
        "reproducible": all_match,
        "seed": seed,
        "n_sims": n_sims,
        "metrics": matches,
    }


def format_stress_report(stress_results: dict) -> str:
    """Format stress test results as a text report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  STRESS TESTING SUITE")
    lines.append("=" * 70)

    metrics_display = [
        ("Median MOIC", "Median_MOIC", ".2f", "x"),
        ("MOIC P10", "MOIC_P10", ".2f", "x"),
        ("Median IRR", "Median_Annual_IRR", ".1%", ""),
        ("IRR P10", "IRR_P10", ".1%", ""),
        ("P(3+ exits)", "P_ThreePlus_Exits", ".1%", ""),
        ("P(<=1 exit)", "P_Exits_LE1", ".1%", ""),
    ]

    # Header
    scenarios = list(stress_results.keys())
    header = f"  {'Metric':<18}"
    for s in scenarios:
        header += f" {s:>18}"
    lines.append(f"\n{header}")
    lines.append(f"  {'-'*18}" + f" {'-'*18}" * len(scenarios))

    for display_name, key, fmt, suffix in metrics_display:
        row = f"  {display_name:<18}"
        for s in scenarios:
            val = stress_results[s]["summary"].get(key, float("nan"))
            if pd.notna(val):
                formatted = format(val, fmt) + suffix
                row += f" {formatted:>18}"
            else:
                row += f" {'N/A':>18}"
        lines.append(row)

    # Envelope pass/fail
    lines.append("")
    lines.append(f"  Envelope Results:")
    for s in scenarios:
        status = "ALL PASS" if stress_results[s]["all_pass"] else f"FAIL: {', '.join(stress_results[s]['failed_gates'])}"
        lines.append(f"    {s}: {status}")

    # Key finding
    combined = stress_results.get("Combined Worst-Case", {})
    if combined:
        cs = combined["summary"]
        lines.append(f"\n  COMBINED WORST-CASE FINDING:")
        lines.append(f"    Under simultaneous tight market + correlation stress:")
        lines.append(f"    Portfolio still produces {cs.get('Median_MOIC', 0):.2f}x MOIC")
        lines.append(f"    with {cs.get('P_ThreePlus_Exits', 0):.1%} probability of 3+ exits")
        if combined["all_pass"]:
            lines.append(f"    ALL GOVERNANCE CONSTRAINTS SATISFIED under stress")
        else:
            lines.append(f"    CONSTRAINTS BREACHED: {', '.join(combined['failed_gates'])}")

    lines.append("")
    return "\n".join(lines)


def format_reproducibility_report(repro: dict) -> str:
    """Format reproducibility verification as a text report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  REPRODUCIBILITY VERIFICATION")
    lines.append("=" * 70)
    lines.append(f"\n  Seed: {repro['seed']}  |  Simulations: {repro['n_sims']}")
    lines.append(f"  Result: {'PASSED — deterministically reproducible' if repro['reproducible'] else 'FAILED'}")
    lines.append(f"\n  {'Metric':<25} {'Run A':>15} {'Run B':>15} {'Match':>8}")
    lines.append(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*8}")

    for m, vals in repro["metrics"].items():
        a = f"{vals['run_a']:.6f}" if pd.notna(vals['run_a']) else "N/A"
        b = f"{vals['run_b']:.6f}" if pd.notna(vals['run_b']) else "N/A"
        match = "✓" if vals["match"] else "✗"
        lines.append(f"  {m:<25} {a:>15} {b:>15} {match:>8}")

    lines.append("")
    return "\n".join(lines)
