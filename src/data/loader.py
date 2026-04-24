"""
Workbook loader and validator.
Reads Parameter and Portfolio State workbooks, validates data, returns clean structures.
"""

import pandas as pd
from pathlib import Path


def _clean_df(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """Drop rows where key column is NaN and reset index."""
    return df[df[key_col].notna()].reset_index(drop=True)


def load_params(path: str | Path) -> dict:
    """
    Load the Parameter Workbook.
    Returns dict of DataFrames keyed by logical name.
    """
    p = pd.ExcelFile(path)

    tier1_tech = _clean_df(pd.read_excel(p, "TIER1_TECH"), "DS")
    tier1_deal = _clean_df(pd.read_excel(p, "TIER1_DEAL"), "DS")
    tier1_cost_time = _clean_df(pd.read_excel(p, "TIER1_COST_TIME"), "DS")
    tier1_econ = _clean_df(pd.read_excel(p, "TIER1_ECON"), "DS")

    tier2_tech = _clean_df(pd.read_excel(p, "TIER2_TECH"), "DS")
    tier2_deal = _clean_df(pd.read_excel(p, "TIER2_DEAL"), "DS")
    tier2_cost_time = _clean_df(pd.read_excel(p, "TIER2_COST_TIME"), "DS")
    tier2_econ = _clean_df(pd.read_excel(p, "TIER2_ECON"), "DS")

    ds_ra_map = _clean_df(pd.read_excel(p, "DS_RA_MAP"), "DS")
    regime = _clean_df(pd.read_excel(p, "REGIME"), "Regime")
    envelope = _clean_df(pd.read_excel(p, "ENVELOPE_THRESHOLDS"), "Metric")
    correlation = _clean_df(pd.read_excel(p, "CORRELATION_FACTORS"), "Factor")

    # Parse envelope into a flat dict for easy access
    envelope_dict = {}
    for _, row in envelope.iterrows():
        envelope_dict[row["Metric"]] = float(row["Value"])

    # Parse correlation factors into a dict
    corr_dict = {}
    for _, row in correlation.iterrows():
        factor = row["Factor"]
        corr_dict[factor] = {
            "StdDev": float(row["StdDev"]) if pd.notna(row.get("StdDev")) else None,
            "Default_Loading": float(row["Default_Loading"]),
        }

    return {
        "tier1_tech": tier1_tech,
        "tier1_deal": tier1_deal,
        "tier1_cost_time": tier1_cost_time,
        "tier1_econ": tier1_econ,
        "tier2_tech": tier2_tech,
        "tier2_deal": tier2_deal,
        "tier2_cost_time": tier2_cost_time,
        "tier2_econ": tier2_econ,
        "ds_ra_map": ds_ra_map,
        "regime": regime,
        "envelope": envelope_dict,
        "correlation": corr_dict,
    }


def load_state(path: str | Path) -> dict:
    """
    Load the Portfolio State Workbook.
    Returns dict of DataFrames keyed by logical name.
    """
    s = pd.ExcelFile(path)

    roster = _clean_df(pd.read_excel(s, "ASSET_ROSTER"), "Asset_ID")
    asset_state = _clean_df(pd.read_excel(s, "ASSET_STATE"), "Asset_ID")
    tranches = _clean_df(pd.read_excel(s, "CAPITAL_TRANCHES"), "Asset_ID")

    # Read control panel settings
    control_raw = pd.read_excel(s, "CONTROL_PANEL", header=None)
    control = {}
    for _, row in control_raw.iterrows():
        key = row.iloc[0]
        val = row.iloc[1] if len(row) > 1 else None
        if pd.notna(key) and isinstance(key, str) and key.strip():
            control[key.strip()] = val

    # Merge cluster IDs from roster into asset_state
    cluster_cols = ["Asset_ID", "MechCluster_ID", "IndicationCluster_ID", "GeoRACluster_ID"]
    available_cols = [c for c in cluster_cols if c in roster.columns]
    if len(available_cols) > 1:
        asset_state = asset_state.merge(
            roster[available_cols], on="Asset_ID", how="left"
        )

    # Ensure Entry_Month exists and is numeric — reject non-numeric values
    if "Entry_Month" not in asset_state.columns:
        asset_state["Entry_Month"] = 0
    else:
        raw = asset_state["Entry_Month"]
        coerced = pd.to_numeric(raw, errors="coerce")
        bad_rows = coerced.isna() & raw.notna()
        if bad_rows.any():
            bad_ids = asset_state.loc[bad_rows, "Asset_ID"].tolist()
            raise ValueError(f"Non-numeric Entry_Month values found for assets: {bad_ids}. "
                             f"Fix the Portfolio State workbook.")
        asset_state["Entry_Month"] = coerced.fillna(0).astype(int)

    # Validate IP economics columns — reject non-numeric values instead of silently zeroing
    for col, default in [
        ("AcqCash_to_IP", 0.0),
        ("Equity_to_IP_Pct", 0.0),
        ("EarlyPassThrough_Pct", 0.0),
        ("EarlyDeferredCash", 0.0),
    ]:
        if col not in asset_state.columns:
            asset_state[col] = default
        else:
            raw = asset_state[col]
            coerced = pd.to_numeric(raw, errors="coerce")
            bad_rows = coerced.isna() & raw.notna()
            if bad_rows.any():
                bad_ids = asset_state.loc[bad_rows, "Asset_ID"].tolist()
                raise ValueError(f"Non-numeric {col} values found for assets: {bad_ids}. "
                                 f"Fix the Portfolio State workbook.")
            asset_state[col] = coerced.fillna(default)

    return {
        "roster": roster,
        "asset_state": asset_state,
        "tranches": tranches,
        "control": control,
    }


def get_tier_tables(params: dict, tier: str) -> dict:
    """
    Return the correct tech/deal/cost_time/econ tables for a given tier.
    tier should be 'Tier-1' or 'Tier-2'.
    """
    if tier == "Tier-2":
        return {
            "tech": params["tier2_tech"],
            "deal": params["tier2_deal"],
            "cost_time": params["tier2_cost_time"],
            "econ": params["tier2_econ"],
        }
    # Default to Tier-1
    return {
        "tech": params["tier1_tech"],
        "deal": params["tier1_deal"],
        "cost_time": params["tier1_cost_time"],
        "econ": params["tier1_econ"],
    }
