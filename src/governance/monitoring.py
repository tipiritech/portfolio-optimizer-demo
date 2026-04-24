"""
Operational monitoring: conditional capital status (#11) and wind-down classification (#13).
"""

import pandas as pd
from datetime import datetime, timedelta


def check_conditional_status(
    asset_state: pd.DataFrame,
    params: dict,
    current_date: datetime | None = None,
) -> dict:
    """
    #11: Identify assets in conditional capital status.

    Triggers:
      - IRR proxy < 22% (based on mode assumptions)
      - Weighted liquidity approaching 30 months

    Returns dict with flagged assets and milestone deadlines.
    """
    from src.data.loader import get_tier_tables
    from src.governance.param_lookup import build_ds_ra_lookup, build_tech_lookup, build_deal_lookup

    if current_date is None:
        current_date = datetime.now()

    irr_threshold = 0.22
    liquidity_ceiling = 30

    flagged = []

    for _, row in asset_state.iterrows():
        asset_id = row["Asset_ID"]
        ds = row["DS_Current"]
        ra = row["RA_Current"]
        tier = row.get("Tier", "Tier-1")
        entry = int(row.get("Entry_Month", 0))

        tables = get_tier_tables(params, tier)

        # Get mode values for IRR proxy
        tech_lk = build_tech_lookup(tables["tech"])
        deal_lk = build_deal_lookup(tables["deal"])
        ct_lk = build_ds_ra_lookup(tables["cost_time"],
            ["TimeToExit_Min", "TimeToExit_Mode", "TimeToExit_Max",
             "MilestoneLag_Min", "MilestoneLag_Mode", "MilestoneLag_Max"])
        # Separate lookup for DevCost
        devcost_lk = build_ds_ra_lookup(tables["cost_time"],
            ["DevCost_Min", "DevCost_Mode", "DevCost_Max"])
        econ_lk = build_ds_ra_lookup(tables["econ"],
            ["Upfront_Min", "Upfront_Mode", "Upfront_Max",
             "NearMilestones_Min", "NearMilestones_Mode", "NearMilestones_Max"])

        if ds not in tech_lk:
            raise KeyError(f"DS '{ds}' not found in {tier} tech table for monitoring. "
                           f"Available: {sorted(tech_lk.keys())}")
        if ds not in deal_lk:
            raise KeyError(f"DS '{ds}' not found in {tier} deal table for monitoring. "
                           f"Available: {sorted(deal_lk.keys())}")
        ct = ct_lk.get((ds, ra))
        if ct is None:
            raise KeyError(f"DS/RA pair ('{ds}', '{ra}') not found in {tier} cost_time table "
                           f"for monitoring. Available: {sorted(ct_lk.keys())}")
        econ = econ_lk.get((ds, ra))
        if econ is None:
            raise KeyError(f"DS/RA pair ('{ds}', '{ra}') not found in {tier} econ table "
                           f"for monitoring. Available: {sorted(econ_lk.keys())}")
        devcost = devcost_lk.get((ds, ra))
        if devcost is None:
            raise KeyError(f"DS/RA pair ('{ds}', '{ra}') not found in {tier} devcost table "
                           f"for monitoring. Available: {sorted(devcost_lk.keys())}")

        tech_mode = tech_lk[ds][1]
        deal_mode = deal_lk[ds][1]
        time_mode = ct[1]
        upfront_mode = econ[1]
        milestone_mode = econ[4]
        dev_cost_mode = devcost[1]  # actual DevCost_Mode from parameter table

        # Simple IRR proxy
        combined_prob = tech_mode * deal_mode
        expected_proceeds = combined_prob * (upfront_mode + milestone_mode)
        # Capital estimate from actual DevCost_Mode (not a proxy)
        est_capital = dev_cost_mode
        total_time = entry + time_mode

        if est_capital > 0 and total_time > 0:
            irr_proxy = (expected_proceeds / est_capital) ** (12.0 / total_time) - 1.0
        else:
            irr_proxy = 0.0

        reasons = []
        if irr_proxy < irr_threshold:
            reasons.append(f"IRR proxy {irr_proxy:.1%} < {irr_threshold:.0%}")
        if total_time > liquidity_ceiling:
            reasons.append(f"Liquidity {total_time:.0f}mo > {liquidity_ceiling}mo ceiling")

        if reasons:
            # Milestone deadline: 90 days from now
            milestone_deadline = current_date + timedelta(days=90)
            flagged.append({
                "Asset_ID": asset_id,
                "Status": "CONDITIONAL",
                "Reasons": "; ".join(reasons),
                "IRR_Proxy": round(irr_proxy, 4),
                "Liquidity_Months": total_time,
                "Milestone_Deadline": milestone_deadline.strftime("%Y-%m-%d"),
                "Action": "Capital limited to preservation spending; milestone required within 90 days",
            })

    return {
        "has_conditional": len(flagged) > 0,
        "flagged_assets": flagged,
    }


def classify_month30_winddown(
    asset_state: pd.DataFrame,
    params: dict,
    results_df: pd.DataFrame = None,
) -> list:
    """
    #13: At month 30, classify each asset for wind-down.

    Categories:
      - Monetization Imminent: exit expected within months 30-36
      - Secondary Sale Candidate: some value, could be sold to another vehicle
      - Bundle Sale Candidate: low individual value, bundle with others
      - Termination Candidate: no realistic path to exit

    Uses mode time-to-exit vs remaining runway.
    """
    from src.data.loader import get_tier_tables
    from src.governance.param_lookup import build_ds_ra_lookup, build_tech_lookup, build_deal_lookup

    classifications = []

    for _, row in asset_state.iterrows():
        asset_id = row["Asset_ID"]
        ds = row["DS_Current"]
        ra = row["RA_Current"]
        tier = row.get("Tier", "Tier-1")
        entry = int(row.get("Entry_Month", 0))

        tables = get_tier_tables(params, tier)
        tech_lk = build_tech_lookup(tables["tech"])
        deal_lk = build_deal_lookup(tables["deal"])
        ct_lk = build_ds_ra_lookup(tables["cost_time"],
            ["TimeToExit_Min", "TimeToExit_Mode", "TimeToExit_Max",
             "MilestoneLag_Min", "MilestoneLag_Mode", "MilestoneLag_Max"])

        if ds not in tech_lk:
            raise KeyError(f"DS '{ds}' not found in {tier} tech table for wind-down classification. "
                           f"Available: {sorted(tech_lk.keys())}")
        if ds not in deal_lk:
            raise KeyError(f"DS '{ds}' not found in {tier} deal table for wind-down classification. "
                           f"Available: {sorted(deal_lk.keys())}")
        ct = ct_lk.get((ds, ra))
        if ct is None:
            raise KeyError(f"DS/RA pair ('{ds}', '{ra}') not found in {tier} cost_time table "
                           f"for wind-down classification. Available: {sorted(ct_lk.keys())}")

        tech_mode = tech_lk[ds][1]
        deal_mode = deal_lk[ds][1]
        time_mode = ct[1]

        expected_exit = entry + time_mode
        combined_prob = tech_mode * deal_mode
        remaining_runway = 36 - 30  # 6 months from month 30

        if expected_exit <= 36 and combined_prob >= 0.50:
            category = "Monetization Imminent"
        elif expected_exit <= 42 and combined_prob >= 0.35:
            category = "Secondary Sale Candidate"
        elif combined_prob >= 0.20:
            category = "Bundle Sale Candidate"
        else:
            category = "Termination Candidate"

        classifications.append({
            "Asset_ID": asset_id,
            "DS": ds,
            "RA": ra,
            "Expected_Exit_Month": expected_exit,
            "Combined_Prob": round(combined_prob, 3),
            "Classification": category,
        })

    return classifications
