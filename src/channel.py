"""
CRO/Pharma Channel Integration Module.

Loads CRO_Master and Pharma_Master workbooks, builds lookup tables,
and computes channel effects (deal probability boost + time compression)
for use in the Monte Carlo inflow simulation.

Design rationale:
    - CRO selection is a structural advantage, not a market condition
    - Channel effect = CRO boost × (1 + AVL alignment if pharma target matched)
    - Time compression = CRO time impact (additive shift on sampled exit month)
    - Only "Confirmed" AVL relationships enter base case; "Likely" for upside
    - All boosts are multiplicative on deal probability, additive on time

Integration point: inflows.py → simulate_asset_inflows()
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def load_cro_master(path: str | Path) -> dict:
    """
    Load CRO_Master workbook. Returns dict of DataFrames.
    Engine consumes CRO_Interface + CRO_IND_Estimates; other sheets are master data.
    """
    p = pd.ExcelFile(path)
    cro_interface = pd.read_excel(p, "CRO_Interface")
    cro_services = pd.read_excel(p, "CRO_Services")

    # Load IND estimates if available
    ind_estimates = {}
    if "CRO_IND_Estimates" in p.sheet_names:
        ind_df = pd.read_excel(p, "CRO_IND_Estimates")
        for _, row in ind_df.iterrows():
            cro_id = row["CRO_ID"]
            ind_estimates[cro_id] = {
                "cost_to_ind": (
                    float(row.get("Est_Cost_To_IND_Min", 3e6)),
                    float(row.get("Est_Cost_To_IND_Mode", 4.5e6)),
                    float(row.get("Est_Cost_To_IND_Max", 6e6)),
                ),
                "time_to_ind": (
                    float(row.get("Est_Time_To_IND_Min", 12)),
                    float(row.get("Est_Time_To_IND_Mode", 15)),
                    float(row.get("Est_Time_To_IND_Max", 18)),
                ),
                "industry_engagement_time": (
                    float(row.get("Industry_Time_To_Engagement_Min", 3)),
                    float(row.get("Industry_Time_To_Engagement_Mode", 4.5)),
                    float(row.get("Industry_Time_To_Engagement_Max", 6)),
                ),
                "discovery_engagement_time": (
                    float(row.get("Discovery_Time_To_Engagement_Min", 0.5)),
                    float(row.get("Discovery_Time_To_Engagement_Mode", 1.0)),
                    float(row.get("Discovery_Time_To_Engagement_Max", 2.0)),
                ),
            }

    # Build CRO lookup: CRO_ID -> channel params
    cro_lookup = {}
    for _, row in cro_interface.iterrows():
        cro_id = row["CRO_ID"]
        cro_lookup[cro_id] = {
            "name": row["CRO_Name"],
            "partner_boost": float(row.get("Partner_Probability_Boost_Percent", 0.0)),
            "time_impact": float(row.get("Time_To_Deal_Impact_Months", 0.0)),
            "cost_score": float(row.get("Cost_Score_1to5", 3.0)),
            "speed_score": float(row.get("Execution_Speed_Score_1to5", 3.0)),
            "oncology_focus": row.get("Oncology_Focus_Level", "Medium"),
            "sm_strength": row.get("Small_Molecule_Strength", "Medium"),
        }

    # Build service index: CRO_ID -> set of (Service_Category, Phase_Supported)
    svc_index = {}
    for _, row in cro_services.iterrows():
        cro_id = row["CRO_ID"]
        if cro_id not in svc_index:
            svc_index[cro_id] = []
        svc_index[cro_id].append({
            "category": row.get("Service_Category", ""),
            "subcategory": row.get("Service_Subcategory", ""),
            "phase": row.get("Phase_Supported", ""),
            "glp": row.get("GLP_Status", ""),
            "onc_specific": row.get("Oncology_Specific", ""),
        })

    return {
        "cro_lookup": cro_lookup,
        "cro_services": svc_index,
        "cro_interface_df": cro_interface,
        "ind_estimates": ind_estimates,
    }


def load_pharma_master(path: str | Path) -> dict:
    """
    Load Pharma_Master workbook. Returns dict of DataFrames.
    Engine consumes Pharma_Interface + DealTerms + AVL.
    """
    p = pd.ExcelFile(path)
    pharma_interface = pd.read_excel(p, "Pharma_Interface")
    deal_terms = pd.read_excel(p, "DealTerms_By_Stage")
    avl = pd.read_excel(p, "AVL_Pharma_CRO")

    # Build pharma lookup: Pharma_ID -> channel params
    pharma_lookup = {}
    for _, row in pharma_interface.iterrows():
        ph_id = row["Pharma_ID"]
        pharma_lookup[ph_id] = {
            "name": row["Pharma_Name"],
            "base_deal_prob": float(row.get("Base_Deal_Probability", 0.5)),
            "time_to_decision": float(row.get("Time_To_Decision_Months", 12.0)),
            "appetite": row.get("BD_Appetite_Level", "Moderate"),
            "stage_pref": row.get("Stage_Preference", "IND"),
        }

    # Build deal terms lookup: (Pharma_ID, Stage_ID) -> economics
    deal_terms_lookup = {}
    for _, row in deal_terms.iterrows():
        key = (row["Pharma_ID"], row["Stage_ID"])
        deal_terms_lookup[key] = {
            "upfront": float(row.get("Expected_Upfront_USD", 0)),
            "milestones": float(row.get("Expected_Milestones_USD", 0)),
            "royalty_pct": float(row.get("Expected_Royalty_Percent", 0)),
            "time_to_deal": float(row.get("Expected_Time_To_Deal_Months", 12)),
        }

    # Build AVL index: (Pharma_ID, CRO_ID) -> alignment data
    avl_lookup = {}
    for _, row in avl.iterrows():
        key = (row["Pharma_ID"], row["CRO_ID"])
        avl_lookup[key] = {
            "vendor_status": row.get("Vendor_Status", "Unknown"),
            "alignment_boost": float(row.get("Vendor_Alignment_Boost_Percent", 0.0)),
        }

    return {
        "pharma_lookup": pharma_lookup,
        "deal_terms_lookup": deal_terms_lookup,
        "avl_lookup": avl_lookup,
        "pharma_interface_df": pharma_interface,
        "avl_df": avl,
    }


def build_channel_lookup(
    asset_state: pd.DataFrame,
    cro_data: dict,
    pharma_data: dict,
    avl_confirmed_only: bool = True,
) -> dict:
    """
    Build per-asset channel effect lookup.

    For each asset that has a CRO_ID assigned (and optionally Target_Pharma_IDs),
    compute the composite deal probability boost, time compression,
    CRO IND cost/time estimates, and engagement arbitrage.

    Args:
        asset_state: must have CRO_ID column; optionally Target_Pharma_IDs (comma-sep)
                     and Engagement_Complete (bool)
        cro_data: output of load_cro_master()
        pharma_data: output of load_pharma_master()
        avl_confirmed_only: if True, only "Confirmed" AVL entries enter base case

    Returns:
        dict: Asset_ID -> {deal_prob_mult, time_shift_months, channel_detail,
                           ind_cost, ind_time, engagement_arbitrage, engagement_complete}
    """
    cro_lookup = cro_data["cro_lookup"]
    ind_estimates = cro_data.get("ind_estimates", {})
    pharma_lookup = pharma_data["pharma_lookup"]
    avl_lookup = pharma_data["avl_lookup"]

    channel = {}

    for _, row in asset_state.iterrows():
        asset_id = row["Asset_ID"]
        cro_id = row.get("CRO_ID")
        engagement_complete = bool(row.get("Engagement_Complete", False))

        # Default: no channel effect
        if pd.isna(cro_id) or cro_id not in cro_lookup:
            channel[asset_id] = {
                "deal_prob_mult": 1.0,
                "time_shift_months": 0.0,
                "cro_boost": 0.0,
                "avl_boost": 0.0,
                "cro_id": None,
                "matched_pharmas": [],
                "ind_cost": None,
                "ind_time": None,
                "engagement_arbitrage_months": None,
                "engagement_complete": False,
            }
            continue

        cro = cro_lookup[cro_id]
        cro_boost = cro["partner_boost"]       # e.g. 0.10
        time_shift = cro["time_impact"]         # e.g. -2.0

        # Check for pharma target alignment via AVL
        target_pharma_str = row.get("Target_Pharma_IDs", "")
        avl_boost = 0.0
        matched_pharmas = []

        if pd.notna(target_pharma_str) and str(target_pharma_str).strip():
            target_ids = [x.strip() for x in str(target_pharma_str).split(",")]

            for ph_id in target_ids:
                avl_key = (ph_id, cro_id)
                if avl_key in avl_lookup:
                    avl_entry = avl_lookup[avl_key]
                    status = avl_entry["vendor_status"]

                    # Filter by confirmation level
                    if avl_confirmed_only and status not in ("Confirmed", "Likely"):
                        continue

                    matched_pharmas.append({
                        "pharma_id": ph_id,
                        "pharma_name": pharma_lookup.get(ph_id, {}).get("name", "?"),
                        "vendor_status": status,
                        "alignment_boost": avl_entry["alignment_boost"],
                    })
                    # Take the max AVL boost across matched pharmas (not additive)
                    avl_boost = max(avl_boost, avl_entry["alignment_boost"])

        # Composite: (1 + CRO_boost) × (1 + AVL_boost)
        deal_prob_mult = (1.0 + cro_boost) * (1.0 + avl_boost)

        # CRO IND cost/time estimates (triangular min/mode/max)
        ind_est = ind_estimates.get(cro_id)
        ind_cost = ind_est["cost_to_ind"] if ind_est else None
        ind_time = ind_est["time_to_ind"] if ind_est else None

        # Engagement arbitrage: industry avg engagement time - Discovery engagement time
        engagement_arb = None
        if ind_est:
            ind_eng = ind_est["industry_engagement_time"]    # (min, mode, max)
            disc_eng = ind_est["discovery_engagement_time"]  # (min, mode, max)
            # Arbitrage = mode(industry) - mode(discovery)
            engagement_arb = round(ind_eng[1] - disc_eng[1], 1)

        channel[asset_id] = {
            "deal_prob_mult": round(deal_prob_mult, 4),
            "time_shift_months": time_shift,
            "cro_boost": cro_boost,
            "avl_boost": avl_boost,
            "cro_id": cro_id,
            "matched_pharmas": matched_pharmas,
            "ind_cost": ind_cost,
            "ind_time": ind_time,
            "engagement_arbitrage_months": engagement_arb,
            "engagement_complete": engagement_complete,
        }

    return channel


def compute_channel_effects(
    deal_p_base: float,
    relative_exit_month: int,
    channel_entry: dict,
) -> tuple[float, int]:
    """
    Apply channel effects to a single asset's sampled deal probability and timing.

    Called inside simulate_asset_inflows() per-asset loop.

    Args:
        deal_p_base: deal probability after RA modifier, before regime mult
        relative_exit_month: sampled time-to-exit in months
        channel_entry: single asset's entry from build_channel_lookup()

    Returns:
        (adjusted_deal_p, adjusted_exit_month)
    """
    adjusted_deal_p = deal_p_base * channel_entry["deal_prob_mult"]
    adjusted_exit = max(1, relative_exit_month + int(channel_entry["time_shift_months"]))
    return adjusted_deal_p, adjusted_exit


def sample_cro_ind_cost(channel_entry: dict, rng=None) -> float | None:
    """
    Sample CRO estimated cost-to-IND from triangular distribution.

    Args:
        channel_entry: single asset's entry from build_channel_lookup()
        rng: numpy random generator (or None to use np.random)

    Returns:
        Sampled cost in USD, or None if no IND estimates available
    """
    import numpy as np
    tri = channel_entry.get("ind_cost")
    if tri is None:
        return None
    low, mode, high = tri
    if rng is not None:
        return float(rng.triangular(low, mode, high))
    return float(np.random.triangular(low, mode, high))


def sample_cro_ind_time(channel_entry: dict, rng=None) -> float | None:
    """
    Sample CRO estimated time-to-IND from triangular distribution.

    Args:
        channel_entry: single asset's entry from build_channel_lookup()
        rng: numpy random generator (or None to use np.random)

    Returns:
        Sampled time in months, or None if no IND estimates available
    """
    import numpy as np
    tri = channel_entry.get("ind_time")
    if tri is None:
        return None
    low, mode, high = tri
    if rng is not None:
        return float(rng.triangular(low, mode, high))
    return float(np.random.triangular(low, mode, high))


def compute_engagement_arbitrage(channel_entry: dict) -> dict | None:
    """
    Compute the engagement time arbitrage for a channeled asset.

    Discovery's conditional deal (LOI) structure pre-negotiates CRO engagement,
    cutting industry-standard engagement timelines (RFP → evaluation → contracting)
    from 3-6 months down to 0.5-2 months.

    Args:
        channel_entry: single asset's entry from build_channel_lookup()

    Returns:
        dict with arbitrage analysis, or None if no CRO assigned
    """
    arb = channel_entry.get("engagement_arbitrage_months")
    if arb is None:
        return None

    return {
        "arbitrage_months": arb,
        "engagement_complete": channel_entry.get("engagement_complete", False),
        "cro_id": channel_entry.get("cro_id"),
        "status": "Active" if channel_entry.get("engagement_complete") else "Pending",
    }


def validate_cro_budget_fit(
    channel_entry: dict,
    tranche_budget: float,
    confidence: str = "mode",
) -> dict:
    """
    Check whether CRO estimated IND cost fits within the tranche budget allocation.

    Args:
        channel_entry: single asset's entry from build_channel_lookup()
        tranche_budget: total budget allocated to the asset's IND tranches
        confidence: which point estimate to use — "min", "mode", or "max"

    Returns:
        dict with fit analysis
    """
    tri = channel_entry.get("ind_cost")
    if tri is None:
        return {"has_estimate": False}

    idx = {"min": 0, "mode": 1, "max": 2}.get(confidence, 1)
    est_cost = tri[idx]
    headroom = tranche_budget - est_cost
    pct_of_budget = est_cost / tranche_budget if tranche_budget > 0 else float("inf")

    return {
        "has_estimate": True,
        "est_cost": est_cost,
        "tranche_budget": tranche_budget,
        "headroom": headroom,
        "pct_of_budget": round(pct_of_budget, 3),
        "fits": headroom >= 0,
        "confidence": confidence,
    }


def summarize_channel(channel_lookup: dict) -> pd.DataFrame:
    """Produce a summary table of channel effects for reporting/audit."""
    rows = []
    for asset_id, ch in channel_lookup.items():
        ind_cost_mode = ch["ind_cost"][1] if ch.get("ind_cost") else None
        ind_time_mode = ch["ind_time"][1] if ch.get("ind_time") else None
        rows.append({
            "Asset_ID": asset_id,
            "CRO_ID": ch["cro_id"],
            "CRO_Boost": ch["cro_boost"],
            "AVL_Boost": ch["avl_boost"],
            "Deal_Prob_Multiplier": ch["deal_prob_mult"],
            "Time_Shift_Months": ch["time_shift_months"],
            "Matched_Pharmas": len(ch["matched_pharmas"]),
            "Pharma_Names": ", ".join(m["pharma_name"] for m in ch["matched_pharmas"]),
            "Est_IND_Cost_Mode": ind_cost_mode,
            "Est_IND_Time_Mode": ind_time_mode,
            "Engagement_Arbitrage_Mo": ch.get("engagement_arbitrage_months"),
            "Engagement_Complete": ch.get("engagement_complete", False),
        })
    return pd.DataFrame(rows)
