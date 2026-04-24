"""
Investor Validation Report Generator.

Produces a comprehensive markdown report covering:
  1. Portfolio structure and governance framework
  2. Base case simulation results
  3. Stress testing (3 scenarios)
  4. Sensitivity analysis (tornado)
  5. Governance validation (all gates)
  6. Reproducibility verification
  7. Parameter provenance template
"""

import pandas as pd
from datetime import datetime


def generate_investor_report(
    state: dict,
    params: dict,
    base_summary: dict,
    stress_results: dict,
    sensitivity: dict,
    envelope_result: dict,
    signal: dict,
    repro: dict,
    corr_index: float,
    weighted_time: float,
    concentration: dict,
    activation: dict,
    combined_prob: dict,
    run_id: str = "",
) -> str:
    """Generate a complete investor validation report in markdown."""

    asset_state = state["asset_state"]
    tranches = state["tranches"]
    roster = state["roster"]
    envelope = params["envelope"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []

    # === HEADER ===
    lines.append("# Discovery Portfolio Simulation — Investor Validation Report")
    lines.append(f"\n**Generated:** {now}")
    lines.append(f"**Run ID:** {run_id}")
    lines.append(f"**Status:** {signal['label']}")
    lines.append("")

    # === EXECUTIVE SUMMARY ===
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append("The Discovery Portfolio Simulation Engine is an internal governance tool that models")
    lines.append("portfolio-level outcomes for early-stage drug assets using Monte Carlo simulation.")
    lines.append("It is designed to shift drug development investment from discretionary decision-making")
    lines.append("to a systematic, probabilistic framework governed by quantitative constraints.")
    lines.append("")
    lines.append(f"**Portfolio Quality Signal: {signal['label']}** — {signal['message']}")
    lines.append("")

    b = base_summary
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Assets in Portfolio | {len(roster)} |")
    lines.append(f"| Total Planned Capital | ${tranches['Budget'].sum():,.0f} |")
    lines.append(f"| Median MOIC | {b['Median_MOIC']:.2f}x |")
    lines.append(f"| MOIC P10 (downside) | {b['MOIC_P10']:.2f}x |")
    lines.append(f"| Median Annual IRR | {b['Median_Annual_IRR']:.1%} |")
    lines.append(f"| IRR P10 (downside) | {b['IRR_P10']:.1%} |")
    lines.append(f"| P(3+ exits) | {b['P_ThreePlus_Exits']:.1%} |")
    lines.append(f"| P(≤1 exit) | {b['P_Exits_LE1']:.1%} |")
    lines.append(f"| Median First Distribution | Month {b.get('Median_First_Dist_Month', 'N/A')} |")
    lines.append(f"| Correlation Index | {corr_index:.3f} |")
    lines.append(f"| Weighted Time to Exit | {weighted_time:.1f} months |")
    lines.append("")

    # === PORTFOLIO STRUCTURE ===
    lines.append("## 2. Portfolio Structure")
    lines.append("")
    lines.append("### Vehicle Design")
    lines.append("- **Finite-life:** 36-month maximum duration")
    lines.append("- **Single-class equity:** one class of common equity, single capital raise")
    lines.append("- **No capital recycling:** proceeds distributed immediately upon monetization (90-95%)")
    lines.append("- **Portfolio size:** 6-10 assets (hard cap 12)")
    lines.append("")

    lines.append("### Asset Roster")
    lines.append("")
    display_cols = ["Asset_ID", "Asset_Name", "MechCluster_ID", "IndicationCluster_ID", "GeoRACluster_ID"]
    avail = [c for c in display_cols if c in roster.columns]
    lines.append("| " + " | ".join(avail) + " |")
    lines.append("| " + " | ".join(["---"] * len(avail)) + " |")
    for _, row in roster.iterrows():
        vals = [str(row.get(c, "")) for c in avail]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")

    lines.append("### Asset Classification")
    lines.append("")
    class_cols = ["Asset_ID", "Tier", "DS_Current", "RA_Current", "Entry_Month"]
    avail_c = [c for c in class_cols if c in asset_state.columns]
    lines.append("| " + " | ".join(avail_c) + " |")
    lines.append("| " + " | ".join(["---"] * len(avail_c)) + " |")
    for _, row in asset_state.iterrows():
        vals = [str(row.get(c, "")) for c in avail_c]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")

    # === GOVERNANCE FRAMEWORK ===
    lines.append("## 3. Governance Framework")
    lines.append("")
    lines.append("All simulations are governed by locked parameters and quantitative constraints.")
    lines.append("Parameters are frozen at vehicle activation and cannot drift.")
    lines.append("")

    lines.append("### Envelope Thresholds")
    lines.append("")
    lines.append("| Constraint | Threshold | Purpose |")
    lines.append("|-----------|-----------|---------|")
    lines.append(f"| Target Median IRR | ≥{envelope.get('Target_Median_IRR', 0.25):.0%} | Return target |")
    lines.append(f"| Floor IRR P10 | ≥{envelope.get('Floor_IRR_P10', 0):.0%} | Downside floor |")
    lines.append(f"| Min P(≥3 exits) | ≥{envelope.get('Min_P_Exits_GE3', 0.6):.0%} | Exit density |")
    lines.append(f"| Max P(≤1 exit) | ≤{envelope.get('Max_P_Exits_LE1', 0.15):.0%} | Catastrophic ceiling |")
    lines.append(f"| Max Correlation Index | ≤{envelope.get('Max_CorrIndex', 0.9)} | Diversification |")
    lines.append(f"| Max Weighted Time | ≤{envelope.get('Max_Weighted_Time', 24)} months | Capital velocity |")
    lines.append(f"| Max Duration | {envelope.get('Max_Duration', 36)} months | Hard stop |")
    lines.append(f"| Min Upfront Threshold | ${envelope.get('Min_Upfront_Threshold', 5e6):,.0f} | Qualifying exit |")
    lines.append("")

    lines.append("### Gate Check Results")
    lines.append("")
    lines.append("| Gate | Status | Actual | Threshold |")
    lines.append("|------|--------|--------|-----------|")
    for name, check in envelope_result["checks"].items():
        status = "✅ PASS" if check["pass"] else "❌ FAIL"
        actual = f"{check['actual']:.4f}" if isinstance(check["actual"], float) else str(check["actual"])
        lines.append(f"| {name} | {status} | {actual} | {check['threshold']} |")
    lines.append("")

    # === STRESS TESTING ===
    lines.append("## 4. Stress Testing")
    lines.append("")
    lines.append("Three stress scenarios are mandated by the governance doctrine to ensure")
    lines.append("the portfolio envelope remains valid under adverse conditions.")
    lines.append("")

    lines.append("| Metric | Base Case | Tight Only | Corr Stress | Combined |")
    lines.append("|--------|-----------|------------|-------------|----------|")

    metrics_show = [
        ("Median MOIC", "Median_MOIC", ".2f", "x"),
        ("MOIC P10", "MOIC_P10", ".2f", "x"),
        ("Median IRR", "Median_Annual_IRR", ".1%", ""),
        ("IRR P10", "IRR_P10", ".1%", ""),
        ("P(3+ exits)", "P_ThreePlus_Exits", ".1%", ""),
        ("P(≤1 exit)", "P_Exits_LE1", ".1%", ""),
    ]

    scenario_keys = ["Base Case", "Tight Regime Only", "Correlation Stress", "Combined Worst-Case"]
    for display_name, key, fmt, suffix in metrics_show:
        row_parts = [f"| {display_name}"]
        for sk in scenario_keys:
            if sk in stress_results:
                val = stress_results[sk]["summary"].get(key, float("nan"))
                if pd.notna(val):
                    row_parts.append(f" {val:{fmt}}{suffix}")
                else:
                    row_parts.append(" N/A")
            else:
                row_parts.append(" N/A")
        lines.append(" |".join(row_parts) + " |")

    lines.append("")

    # Stress verdict
    combined = stress_results.get("Combined Worst-Case", {})
    if combined:
        if combined["all_pass"]:
            lines.append("**Combined worst-case finding:** All governance constraints remain satisfied")
            lines.append("under simultaneous tight market conditions and elevated correlation stress.")
        else:
            lines.append(f"**Combined worst-case finding:** Constraints breached: {', '.join(combined['failed_gates'])}")
    lines.append("")

    # === SENSITIVITY ANALYSIS ===
    lines.append("## 5. Sensitivity Analysis")
    lines.append("")
    lines.append(f"Each key input was varied by ±{sensitivity['shock_pct']:.0%} to assess model resilience.")
    lines.append("The table below ranks factors by their impact on Median MOIC.")
    lines.append("")

    lines.append("| Factor | -20% MOIC | Base MOIC | +20% MOIC | Range |")
    lines.append("|--------|-----------|-----------|-----------|-------|")
    for s in sensitivity["tornado_data"]:
        d = s["deltas"]["Median_MOIC"]
        lines.append(f"| {s['factor']} | {d['down']:.2f}x | {d['base']:.2f}x | {d['up']:.2f}x | {d['range']:.2f}x |")
    lines.append("")

    lines.append("### Key Finding")
    if sensitivity["tornado_data"]:
        top = sensitivity["tornado_data"][0]
        lines.append(f"The most sensitive input is **{top['factor']}**. Even under a {sensitivity['shock_pct']:.0%}")
        d = top["deltas"]["Median_MOIC"]
        lines.append(f"adverse shock, Median MOIC remains {d['down']:.2f}x (base: {d['base']:.2f}x).")
    lines.append("")

    # === REPRODUCIBILITY ===
    lines.append("## 6. Reproducibility Verification")
    lines.append("")
    if repro["reproducible"]:
        lines.append(f"**PASSED.** Two independent runs with seed={repro['seed']} produced identical results")
        lines.append(f"across all {len(repro['metrics'])} metrics tested ({repro['n_sims']} simulations each).")
    else:
        lines.append("**FAILED.** Results differed between runs. Investigation required.")
    lines.append("")

    lines.append("| Metric | Run A | Run B | Match |")
    lines.append("|--------|-------|-------|-------|")
    for m, vals in repro["metrics"].items():
        a = f"{vals['run_a']:.6f}" if pd.notna(vals["run_a"]) else "N/A"
        b = f"{vals['run_b']:.6f}" if pd.notna(vals["run_b"]) else "N/A"
        match = "✓" if vals["match"] else "✗"
        lines.append(f"| {m} | {a} | {b} | {match} |")
    lines.append("")

    # === PRE-FLIGHT CHECKS ===
    lines.append("## 7. Pre-Flight Validation")
    lines.append("")

    if activation["activated"]:
        lines.append("- **Activation Requirements:** ✅ All satisfied")
    else:
        lines.append(f"- **Activation Requirements:** ⚠️ {len(activation['issues'])} issue(s)")
        for issue in activation["issues"]:
            lines.append(f"  - {issue}")

    if not concentration["breaches"]:
        lines.append("- **Capital Concentration:** ✅ No breaches (hard cap 20%)")
    else:
        lines.append(f"- **Capital Concentration:** ❌ {len(concentration['breaches'])} breach(es)")

    if combined_prob["all_pass"]:
        lines.append("- **Combined Probability (≥45%):** ✅ All assets pass")
    else:
        lines.append(f"- **Combined Probability:** ⚠️ {len(combined_prob['failures'])} below threshold")

    lines.append(f"- **Weighted Time:** {weighted_time:.1f} months {'✅' if weighted_time <= 24 else '⚠️ Above 24mo target'}")
    lines.append("")

    # === NON-NEGOTIABLE PRINCIPLES ===
    lines.append("## 8. Non-Negotiable Principles")
    lines.append("")
    lines.append("The following structural constraints are enforced by the engine and cannot")
    lines.append("be overridden, even by Board vote:")
    lines.append("")
    lines.append("1. **No capital recycling** within the vehicle")
    lines.append("2. **No parameter drift** after activation")
    lines.append("3. **No narrative-based admission** — all admissions require quantitative gate passage")
    lines.append("4. **No time extension** beyond 36 months")
    lines.append("5. **No MEC violations** — inventor economics cannot exceed maximum economic capacity")
    lines.append("6. **No hidden correlation clustering** — factor model enforces transparency")
    lines.append("")

    # === METHODOLOGY ===
    lines.append("## 9. Methodology Notes")
    lines.append("")
    lines.append("- All distributions use **triangular (Min/Mode/Max)** parameterization")
    lines.append("- Probabilities are anchored to **median-observed transaction data**, not means")
    lines.append("- Correlation uses a **latent normal factor model** (market + mechanism + indication + geography + idiosyncratic)")
    lines.append("- IRR is computed via **monthly cashflow bisection**, then annualized via compounding")
    lines.append("- Only **early economics** are modeled (upfront + near-term milestones within 36 months)")
    lines.append("- Long-tail royalties are **excluded** to maintain conservatism")
    lines.append("- Acquisition exits are modeled as a **separate low-probability / high-value path**")
    lines.append("- Operating overhead is **deducted monthly** from portfolio cashflows")
    lines.append("- Failed assets trigger **tranche kill** — remaining capital deployment is cancelled")
    lines.append("")

    lines.append("---")
    lines.append(f"*Report generated by Discovery Portfolio Simulation Engine v2 — {now}*")
    lines.append("")

    return "\n".join(lines)
