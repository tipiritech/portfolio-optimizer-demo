"""
Envelope validation and Portfolio Quality Signal.
Includes: weighted time (#1), capital concentration (#2), combined probability (#3),
activation requirements (#5), parameter validation (#16), quality signal.
"""

import numpy as np
import pandas as pd


def check_envelope(summary, envelope, corr_index=0.0, weighted_time=None,
                   concentration_issues=None, portfolio_duration=36):
    checks = {}
    target_irr = envelope.get("Target_Median_IRR", 0.25)
    median_irr = summary.get("Median_Annual_IRR", float("nan"))
    checks["Median_IRR"] = {"threshold": target_irr, "actual": median_irr,
        "pass": pd.notna(median_irr) and median_irr >= target_irr}

    floor_irr = envelope.get("Floor_IRR_P10", 0.0)
    p10_irr = summary.get("IRR_P10", float("nan"))
    checks["IRR_P10"] = {"threshold": floor_irr, "actual": p10_irr,
        "pass": pd.notna(p10_irr) and p10_irr >= floor_irr}

    min_p3 = envelope.get("Min_P_Exits_GE3", 0.60)
    p3 = summary.get("P_ThreePlus_Exits", 0.0)
    checks["P_Exits_GE3"] = {"threshold": min_p3, "actual": p3, "pass": p3 >= min_p3}

    max_p1 = envelope.get("Max_P_Exits_LE1", 0.15)
    p_le1 = summary.get("P_Exits_LE1", summary.get("P_Zero_Exits", 0) + summary.get("P_One_Exit", 0))
    checks["P_Exits_LE1"] = {"threshold": max_p1, "actual": p_le1, "pass": p_le1 <= max_p1}

    max_corr = envelope.get("Max_CorrIndex", 0.9)
    checks["Corr_Index"] = {"threshold": max_corr, "actual": corr_index, "pass": corr_index <= max_corr}

    max_wt = envelope.get("Max_Weighted_Time", 24)
    if weighted_time is not None:
        checks["Weighted_Time"] = {"threshold": max_wt, "actual": round(weighted_time, 2),
            "pass": weighted_time <= max_wt}
        if weighted_time > 30:
            checks["Capital_Pause_Trigger"] = {"threshold": 30, "actual": round(weighted_time, 2), "pass": False}

    if concentration_issues is not None:
        has_breach = len(concentration_issues) > 0
        checks["Capital_Concentration"] = {"threshold": "<=20% per asset",
            "actual": f"{len(concentration_issues)} breach(es)" if has_breach else "OK",
            "pass": not has_breach}

    # Max_Duration: compare actual portfolio duration against hard stop
    max_dur = envelope.get("Max_Duration", 36)
    checks["Max_Duration"] = {"threshold": max_dur, "actual": portfolio_duration,
        "pass": portfolio_duration <= max_dur}

    all_pass = all(c["pass"] for c in checks.values())
    failed_gates = [k for k, v in checks.items() if not v["pass"]]
    return {"checks": checks, "all_pass": all_pass, "failed_gates": failed_gates}


def compute_weighted_time(asset_state, params):
    from src.data.loader import get_tier_tables
    from src.governance.param_lookup import build_ds_ra_lookup
    times = []
    for _, row in asset_state.iterrows():
        ds, ra, tier = row["DS_Current"], row["RA_Current"], row.get("Tier", "Tier-1")
        entry = int(row.get("Entry_Month", 0))
        tables = get_tier_tables(params, tier)
        ct_lk = build_ds_ra_lookup(tables["cost_time"],
            ["TimeToExit_Min", "TimeToExit_Mode", "TimeToExit_Max",
             "MilestoneLag_Min", "MilestoneLag_Mode", "MilestoneLag_Max"])
        p = ct_lk.get((ds, ra))
        if p is None:
            raise KeyError(f"DS/RA pair ('{ds}', '{ra}') not found in {tier} cost_time table "
                           f"for weighted time calculation. Available: {sorted(ct_lk.keys())}")
        times.append(entry + p[1])
    return float(np.mean(times)) if times else 0.0


def check_capital_concentration(tranches, hard_cap=0.20, soft_target=0.15):
    budgets = tranches.groupby("Asset_ID")["Budget"].sum()
    total = budgets.sum()
    if total <= 0:
        return {"breaches": [], "warnings": [], "concentrations": {}}
    conc = (budgets / total).to_dict()
    return {
        "breaches": [a for a, c in conc.items() if c > hard_cap],
        "warnings": [a for a, c in conc.items() if soft_target < c <= hard_cap],
        "concentrations": conc,
    }


def check_combined_probability(asset_state, params, min_combined=0.45):
    from src.data.loader import get_tier_tables
    from src.governance.param_lookup import build_tech_lookup, build_deal_lookup, build_ra_modifier_lookup
    ra_mod_lk = build_ra_modifier_lookup(params["ds_ra_map"])
    failures = []
    for _, row in asset_state.iterrows():
        ds, ra = row["DS_Current"], row["RA_Current"]
        tables = get_tier_tables(params, row.get("Tier", "Tier-1"))
        tech_lk = build_tech_lookup(tables["tech"])
        deal_lk = build_deal_lookup(tables["deal"])
        if ds not in tech_lk:
            raise KeyError(f"DS '{ds}' not found in tech probability table for "
                           f"combined probability check. Available: {sorted(tech_lk.keys())}")
        if ds not in deal_lk:
            raise KeyError(f"DS '{ds}' not found in deal probability table for "
                           f"combined probability check. Available: {sorted(deal_lk.keys())}")
        if (ds, ra) not in ra_mod_lk:
            raise KeyError(f"DS/RA pair ('{ds}', '{ra}') not found in DS_RA_MAP for "
                           f"combined probability check. Available: {sorted(ra_mod_lk.keys())}")
        tech_mode = tech_lk[ds][1]
        deal_mode = deal_lk[ds][1]
        ra_mode = ra_mod_lk[(ds, ra)][1]
        combined = tech_mode * deal_mode * ra_mode
        if combined < min_combined:
            failures.append({"Asset_ID": row["Asset_ID"], "Combined": round(combined, 3), "Threshold": min_combined})
    return {"all_pass": len(failures) == 0, "failures": failures}


def check_activation_requirements(asset_state, ds_ra_map):
    from src.governance.param_lookup import build_allowed_lookup
    issues = []
    n = len(asset_state)
    if n < 3:
        issues.append(f"Only {n} assets (minimum 3 required)")
    valid_ds, valid_ra = {"DS-3", "DS-4", "DS-5"}, {"RA-1", "RA-2"}
    for _, row in asset_state.iterrows():
        if row["DS_Current"] not in valid_ds:
            issues.append(f"{row['Asset_ID']}: DS={row['DS_Current']} below DS-3")
        if row["RA_Current"] not in valid_ra:
            issues.append(f"{row['Asset_ID']}: RA={row['RA_Current']} not RA-1/RA-2")
    if n >= 3:
        first3 = asset_state.head(3)
        for col in ["MechCluster_ID", "IndicationCluster_ID"]:
            if col in first3.columns and first3[col].nunique() < 2:
                issues.append(f"Insufficient diversification in {col} among first 3 assets")
    allowed = build_allowed_lookup(ds_ra_map)
    for _, row in asset_state.iterrows():
        key = (row["DS_Current"], row["RA_Current"])
        if key in allowed and not allowed[key]:
            issues.append(f"{row['Asset_ID']}: {key} not allowed")
    return {"activated": len(issues) == 0, "issues": issues}


def validate_params(params):
    issues = []
    for tbl in ["tier1_tech", "tier1_deal", "tier2_tech", "tier2_deal"]:
        df = params.get(tbl)
        if df is None or df.empty:
            continue
        cols = [c for c in df.columns if c != "DS"]
        for _, row in df.iterrows():
            vals = [float(row[c]) for c in cols]
            if not (vals[0] <= vals[1] <= vals[2]):
                issues.append(f"{tbl} DS={row['DS']}: Min<=Mode<=Max violated ({vals})")
            if any(v < 0 or v > 1 for v in vals):
                issues.append(f"{tbl} DS={row['DS']}: value outside [0,1]")
    for tbl in ["tier1_cost_time", "tier2_cost_time"]:
        df = params.get(tbl)
        if df is None or df.empty:
            continue
        groups = [("DevCost_Min","DevCost_Mode","DevCost_Max"),
                  ("TransCost_Min","TransCost_Mode","TransCost_Max"),
                  ("TimeToExit_Min","TimeToExit_Mode","TimeToExit_Max"),
                  ("MilestoneLag_Min","MilestoneLag_Mode","MilestoneLag_Max")]
        for _, row in df.iterrows():
            for mn, md, mx in groups:
                v = [float(row[mn]), float(row[md]), float(row[mx])]
                if not (v[0] <= v[1] <= v[2]):
                    issues.append(f"{tbl} {row['DS']}/{row['RA']}: {mn} Min<=Mode<=Max violated")
    regime = params.get("regime")
    if regime is not None and not regime.empty:
        s = regime["Probability"].astype(float).sum()
        if abs(s - 1.0) > 0.001:
            issues.append(f"Regime probabilities sum to {s:.4f} (should be 1.0)")
    return issues


def portfolio_quality_signal(summary, envelope):
    min_p3 = envelope.get("Min_P_Exits_GE3", 0.60)
    floor_irr = envelope.get("Floor_IRR_P10", 0.0)
    p3 = summary.get("P_ThreePlus_Exits", 0.0)
    irr_p10 = summary.get("IRR_P10", float("nan"))
    irr_p10 = irr_p10 if pd.notna(irr_p10) else -999.0
    if p3 >= min_p3 and irr_p10 >= floor_irr:
        return {"label": "GREEN", "bg": "#d1fae5", "border": "#10b981", "text": "#065f46",
            "message": f"Strong profile — P(3+ exits)={p3:.1%}, IRR P10={irr_p10:.1%}"}
    elif p3 >= 0.40 and irr_p10 >= -0.10:
        return {"label": "YELLOW", "bg": "#fef3c7", "border": "#f59e0b", "text": "#92400e",
            "message": f"Borderline profile — P(3+ exits)={p3:.1%}, IRR P10={irr_p10:.1%}"}
    else:
        return {"label": "RED", "bg": "#fee2e2", "border": "#ef4444", "text": "#991b1b",
            "message": f"Weak profile — P(3+ exits)={p3:.1%}, IRR P10={irr_p10:.1%}"}
