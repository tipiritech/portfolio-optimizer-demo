"""
Sandbox: User-driven scenario analysis layer.

Allows interactive what-if exploration WITHOUT modifying locked parameters.
All sandbox runs are clearly marked as exploratory — not governance-grade.

Design constraints:
    - Max_Duration (36 months) is NEVER overridable, even in sandbox
    - Locked parameters are read-only; sandbox applies temporary overlays
    - Overlay = shallow copy of params/state with user adjustments applied
    - Results carry a sandbox_flag so they can never be confused with production
    - No capital recycling logic changes — structural governance is inviolable

Sandbox scenario types:
    1. Parameter shocks (tech prob, deal prob, economics ±X%)
    2. Asset roster changes (add hypothetical / remove existing)
    3. Channel swaps (reassign CRO / pharma targets)
    4. Overhead / contingency adjustments
    5. Regime forcing (Tight / Neutral / Hot)
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Optional

from src.simulation.monte_carlo import run_monte_carlo, summarize_results
from src.optimization.correlation import compute_correlation_index
from src.optimization.envelope import (
    check_envelope, compute_weighted_time,
    check_capital_concentration, portfolio_quality_signal,
)


# ── Governance locks: these cannot be overridden even in sandbox ──────────

NON_OVERRIDABLE = frozenset({
    "Max_Duration",       # 36-month hard stop
    "no_capital_recycle", # structural constraint
})


def _validate_sandbox_request(scenario: dict) -> list[str]:
    """Check that the scenario doesn't attempt to override locked constraints."""
    violations = []
    if scenario.get("duration") and scenario["duration"] != 36:
        violations.append("Max_Duration is non-overridable — cannot change from 36 months")
    if scenario.get("enable_capital_recycle"):
        violations.append("Capital recycling is structurally prohibited")
    return violations


# ── Parameter overlay ─────────────────────────────────────────────────────

def apply_param_shock(
    params: dict,
    shock_map: dict,
) -> dict:
    """
    Create a shallow copy of params with multiplicative shocks applied.

    shock_map: {
        "tech_prob": 0.90,     # multiply all tech probs by 0.90 (10% haircut)
        "deal_prob": 1.10,     # multiply all deal probs by 1.10 (10% boost)
        "upfront": 0.80,       # multiply all upfronts by 0.80
        "dev_cost": 1.15,      # multiply all dev costs by 1.15
        "time_to_exit": 1.10,  # multiply all exit times by 1.10
    }

    Returns a new params dict with shocked values. Original is untouched.
    """
    shocked = deepcopy(params)

    tech_mult = shock_map.get("tech_prob", 1.0)
    deal_mult = shock_map.get("deal_prob", 1.0)
    upfront_mult = shock_map.get("upfront", 1.0)
    milestone_mult = shock_map.get("milestones", upfront_mult)  # default same as upfront
    cost_mult = shock_map.get("dev_cost", 1.0)
    time_mult = shock_map.get("time_to_exit", 1.0)

    for tier_prefix in ("tier1", "tier2"):
        # Tech probability tables
        tech_key = f"{tier_prefix}_tech"
        if tech_key in shocked and shocked[tech_key] is not None:
            df = shocked[tech_key].copy()
            for col in ["Tech_Min", "Tech_Mode", "Tech_Max"]:
                if col in df.columns:
                    df[col] = (df[col] * tech_mult).clip(0.01, 0.99)
            shocked[tech_key] = df

        # Deal probability tables
        deal_key = f"{tier_prefix}_deal"
        if deal_key in shocked and shocked[deal_key] is not None:
            df = shocked[deal_key].copy()
            for col in ["Deal_Min", "Deal_Mode", "Deal_Max"]:
                if col in df.columns:
                    df[col] = (df[col] * deal_mult).clip(0.01, 0.99)
            shocked[deal_key] = df

        # Economics tables
        econ_key = f"{tier_prefix}_econ"
        if econ_key in shocked and shocked[econ_key] is not None:
            df = shocked[econ_key].copy()
            for col in ["Upfront_Min", "Upfront_Mode", "Upfront_Max"]:
                if col in df.columns:
                    df[col] = df[col] * upfront_mult
            for col in ["NearMilestones_Min", "NearMilestones_Mode", "NearMilestones_Max"]:
                if col in df.columns:
                    df[col] = df[col] * milestone_mult
            shocked[econ_key] = df

        # Cost/time tables
        ct_key = f"{tier_prefix}_cost_time"
        if ct_key in shocked and shocked[ct_key] is not None:
            df = shocked[ct_key].copy()
            for col in ["DevCost_Min", "DevCost_Mode", "DevCost_Max"]:
                if col in df.columns:
                    df[col] = df[col] * cost_mult
            for col in ["TimeToExit_Min", "TimeToExit_Mode", "TimeToExit_Max"]:
                if col in df.columns:
                    df[col] = df[col] * time_mult
            shocked[ct_key] = df

    return shocked


# ── Asset roster overlay ──────────────────────────────────────────────────

def add_hypothetical_asset(
    asset_state: pd.DataFrame,
    tranches: pd.DataFrame,
    asset_id: str,
    ds: str,
    ra: str,
    tier: str = "Tier-1",
    entry_month: int = 0,
    budget: float = 5_000_000.0,
    stop_month: int = 14,
    cro_id: str | None = None,
    target_pharma_ids: str | None = None,
    mech_cluster: str = "MECH-NEW",
    ind_cluster: str = "IND-NEW",
    geo_cluster: str = "GEO-US1",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add a hypothetical asset to the portfolio for sandbox analysis.

    Returns (new_asset_state, new_tranches) — copies with the addition.
    """
    new_as = asset_state.copy()
    new_tr = tranches.copy()

    new_row = {
        "Asset_ID": asset_id,
        "DS_Current": ds,
        "RA_Current": ra,
        "Tier": tier,
        "Entry_Month": entry_month,
        "Equity_to_IP_Pct": 0.10,
        "AcqCash_to_IP": 0,
        "EarlyPassThrough_Pct": 0.0,
        "EarlyDeferredCash": 0,
        "MechCluster_ID": mech_cluster,
        "IndicationCluster_ID": ind_cluster,
        "GeoRACluster_ID": geo_cluster,
    }
    if cro_id:
        new_row["CRO_ID"] = cro_id
    if target_pharma_ids:
        new_row["Target_Pharma_IDs"] = target_pharma_ids

    new_as = pd.concat([new_as, pd.DataFrame([new_row])], ignore_index=True)

    # Add default tranches (dev + BD)
    dev_budget = budget * 0.75
    bd_budget = budget * 0.25
    new_tranches = pd.DataFrame([
        {"Asset_ID": asset_id, "Tranche_ID": "T1", "Purpose": "IND-Enabling Development",
         "Budget": dev_budget, "Start_Month": entry_month,
         "Stop_Month": stop_month, "Status": "Planned"},
        {"Asset_ID": asset_id, "Tranche_ID": "T2", "Purpose": "Transaction / BD",
         "Budget": bd_budget, "Start_Month": max(0, stop_month - 4),
         "Stop_Month": stop_month + 4, "Status": "Planned"},
    ])
    new_tr = pd.concat([new_tr, new_tranches], ignore_index=True)

    return new_as, new_tr


def remove_asset(
    asset_state: pd.DataFrame,
    tranches: pd.DataFrame,
    asset_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove an asset from the portfolio for sandbox analysis."""
    new_as = asset_state[asset_state["Asset_ID"] != asset_id].copy().reset_index(drop=True)
    new_tr = tranches[tranches["Asset_ID"] != asset_id].copy().reset_index(drop=True)
    return new_as, new_tr


# ── Channel swap ──────────────────────────────────────────────────────────

def swap_cro(
    asset_state: pd.DataFrame,
    asset_id: str,
    new_cro_id: str,
    new_pharma_ids: str | None = None,
) -> pd.DataFrame:
    """Reassign CRO for an asset in sandbox. Returns new asset_state copy."""
    new_as = asset_state.copy()
    mask = new_as["Asset_ID"] == asset_id
    if not mask.any():
        raise ValueError(f"Asset {asset_id} not found")
    new_as.loc[mask, "CRO_ID"] = new_cro_id
    if new_pharma_ids is not None:
        new_as.loc[mask, "Target_Pharma_IDs"] = new_pharma_ids
    return new_as


# ── Core sandbox runner ───────────────────────────────────────────────────

def run_sandbox(
    asset_state: pd.DataFrame,
    tranches: pd.DataFrame,
    params: dict,
    envelope: dict,
    scenario: dict,
    n_sims: int = 1000,
    seed: int | None = None,
    channel_lookup: dict | None = None,
) -> dict:
    """
    Run a sandbox scenario and return results with governance comparison.

    scenario dict supports:
        "shock_map": dict of parameter shocks (see apply_param_shock)
        "add_assets": list of dicts for add_hypothetical_asset()
        "remove_assets": list of Asset_ID strings to remove
        "cro_swaps": list of {"asset_id", "new_cro_id", "new_pharma_ids"} dicts
        "overhead": float override
        "contingency": float override
        "use_corr_stress": bool
        "use_tight_only": bool
        "label": str (human-readable scenario name)

    Returns dict with:
        "sandbox_flag": True (always — marks this as non-governance)
        "label": scenario label
        "violations": list of governance violations attempted
        "summary": MC summary dict
        "envelope": envelope check results
        "asset_count": number of assets in scenario
        "delta_vs_base": dict of metric deltas (if base_summary provided)
    """
    # Check for governance violations
    violations = _validate_sandbox_request(scenario)
    if violations:
        return {
            "sandbox_flag": True,
            "label": scenario.get("label", "Unnamed"),
            "violations": violations,
            "summary": None,
            "envelope": None,
            "error": "Governance violation — scenario rejected",
        }

    # Apply parameter shocks
    working_params = params
    if scenario.get("shock_map"):
        working_params = apply_param_shock(params, scenario["shock_map"])

    # Apply asset roster changes
    working_as = asset_state.copy()
    working_tr = tranches.copy()

    for remove_id in scenario.get("remove_assets", []):
        working_as, working_tr = remove_asset(working_as, working_tr, remove_id)

    for add_spec in scenario.get("add_assets", []):
        working_as, working_tr = add_hypothetical_asset(
            working_as, working_tr, **add_spec
        )

    # Apply CRO swaps
    for swap in scenario.get("cro_swaps", []):
        working_as = swap_cro(
            working_as,
            swap["asset_id"],
            swap["new_cro_id"],
            swap.get("new_pharma_ids"),
        )

    # Run overrides
    overhead = scenario.get("overhead", 2_640_000.0)
    contingency = scenario.get("contingency", 1.10)
    use_corr_stress = scenario.get("use_corr_stress", False)
    use_tight_only = scenario.get("use_tight_only", False)

    # Rebuild channel lookup if CRO swaps were applied
    working_channel = channel_lookup
    if scenario.get("cro_swaps") and channel_lookup is not None:
        # Re-derive channel from modified asset_state
        # Caller should rebuild; we pass through as-is for now
        pass

    # Run MC
    results_df = run_monte_carlo(
        asset_state=working_as,
        tranches=working_tr,
        params=working_params,
        n_sims=n_sims,
        duration=36,  # NON-OVERRIDABLE
        contingency_mult=contingency,
        use_corr_stress=use_corr_stress,
        use_tight_only=use_tight_only,
        seed=seed,
        annual_overhead=overhead,
        enable_tranche_kill=True,
        enable_rollover=True,
        channel_lookup=working_channel,
    )
    summary = summarize_results(results_df)

    # Envelope check
    corr_config = working_params.get("correlation", {})
    corr_index = compute_correlation_index(working_as, corr_config)
    weighted_time = compute_weighted_time(working_as, working_params)
    conc = check_capital_concentration(working_tr)

    env_check = check_envelope(
        summary, envelope, corr_index=corr_index,
        weighted_time=weighted_time,
        concentration_issues=conc.get("breaches", []),
    )

    signal = portfolio_quality_signal(summary, envelope)

    return {
        "sandbox_flag": True,
        "label": scenario.get("label", "Unnamed Scenario"),
        "violations": [],
        "summary": summary,
        "envelope": env_check,
        "signal": signal,
        "asset_count": len(working_as),
        "corr_index": corr_index,
        "weighted_time": weighted_time,
        "concentration": conc,
        "results_df": results_df,
    }


# ── Comparison helper ─────────────────────────────────────────────────────

def compare_scenarios(base_result: dict, sandbox_result: dict) -> dict:
    """
    Compute deltas between base case and sandbox scenario.

    Returns dict of metric name -> {base, sandbox, delta, direction}
    """
    if sandbox_result.get("summary") is None:
        return {"error": "Sandbox scenario was rejected"}

    base = base_result["summary"]
    sand = sandbox_result["summary"]

    metrics = [
        ("Median_MOIC", "x", True),       # higher is better
        ("MOIC_P10", "x", True),
        ("Median_Annual_IRR", "%", True),
        ("IRR_P10", "%", True),
        ("P_ThreePlus_Exits", "%", True),  # higher is better
        ("P_Exits_LE1", "%", False),       # lower is better
        ("P_Zero_Exits", "%", False),      # lower is better
    ]

    deltas = {}
    for key, unit, higher_better in metrics:
        b = base.get(key, float("nan"))
        s = sand.get(key, float("nan"))
        if pd.notna(b) and pd.notna(s):
            delta = s - b
            if higher_better:
                direction = "better" if delta > 0.001 else ("worse" if delta < -0.001 else "flat")
            else:
                direction = "better" if delta < -0.001 else ("worse" if delta > 0.001 else "flat")
            deltas[key] = {"base": b, "sandbox": s, "delta": delta,
                           "direction": direction, "unit": unit}
        else:
            deltas[key] = {"base": b, "sandbox": s, "delta": float("nan"),
                           "direction": "N/A", "unit": unit}

    # Envelope pass/fail comparison
    base_pass = base_result.get("envelope", {}).get("all_pass", False)
    sand_pass = sandbox_result.get("envelope", {}).get("all_pass", False)
    deltas["_envelope"] = {
        "base_pass": base_pass,
        "sandbox_pass": sand_pass,
        "changed": base_pass != sand_pass,
    }

    return deltas


def format_comparison(deltas: dict, label: str = "Scenario") -> str:
    """Format scenario comparison as a text report."""
    lines = []
    lines.append("=" * 65)
    lines.append(f"  SANDBOX COMPARISON: {label}")
    lines.append("  ⚠️  EXPLORATORY ONLY — NOT GOVERNANCE-GRADE")
    lines.append("=" * 65)

    lines.append(f"\n  {'Metric':<25} {'Base':>10} {'Sandbox':>10} {'Delta':>10} {'':>8}")
    lines.append(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for key, d in deltas.items():
        if key.startswith("_"):
            continue
        b = d["base"]
        s = d["sandbox"]
        delta = d["delta"]
        direction = d["direction"]
        unit = d["unit"]

        if unit == "x":
            b_str = f"{b:.2f}x" if pd.notna(b) else "N/A"
            s_str = f"{s:.2f}x" if pd.notna(s) else "N/A"
            d_str = f"{delta:+.2f}x" if pd.notna(delta) else "N/A"
        else:
            b_str = f"{b:.1%}" if pd.notna(b) else "N/A"
            s_str = f"{s:.1%}" if pd.notna(s) else "N/A"
            d_str = f"{delta:+.1%}" if pd.notna(delta) else "N/A"

        icon = "↑" if direction == "better" else ("↓" if direction == "worse" else "—")
        lines.append(f"  {key:<25} {b_str:>10} {s_str:>10} {d_str:>10} {icon:>8}")

    env = deltas.get("_envelope", {})
    if env:
        b_env = "PASS" if env.get("base_pass") else "FAIL"
        s_env = "PASS" if env.get("sandbox_pass") else "FAIL"
        lines.append(f"\n  Envelope: {b_env} → {s_env}")

    lines.append("")
    return "\n".join(lines)


# ── Preset scenarios ──────────────────────────────────────────────────────

PRESET_SCENARIOS = {
    "bear_case": {
        "label": "Bear Case (tech -15%, deal -15%, upfronts -20%)",
        "shock_map": {"tech_prob": 0.85, "deal_prob": 0.85, "upfront": 0.80},
    },
    "bull_case": {
        "label": "Bull Case (deal +10%, upfronts +15%)",
        "shock_map": {"deal_prob": 1.10, "upfront": 1.15},
    },
    "cost_blowout": {
        "label": "Cost Blowout (dev cost +25%, time +15%)",
        "shock_map": {"dev_cost": 1.25, "time_to_exit": 1.15},
    },
    "tight_market": {
        "label": "Tight Market (forced tight regime)",
        "use_tight_only": True,
    },
    "correlation_stress": {
        "label": "Correlation Stress (+0.10 floor)",
        "use_corr_stress": True,
    },
    "combined_stress": {
        "label": "Combined Stress (tight + correlation)",
        "use_tight_only": True,
        "use_corr_stress": True,
    },
}
