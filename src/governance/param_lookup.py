"""
Parameter lookup utilities.
Triangular sampling and lookup-table builders for DS, RA, Tier, and regime parameters.
"""

import numpy as np
import pandas as pd


def triangular_sample(min_val, mode_val, max_val):
    """Sample from a triangular distribution with guardrails."""
    min_val, mode_val, max_val = float(min_val), float(mode_val), float(max_val)

    if min_val == mode_val == max_val:
        return min_val
    if mode_val < min_val:
        mode_val = min_val
    if mode_val > max_val:
        mode_val = max_val
    if min_val == max_val:
        return min_val

    return np.random.triangular(min_val, mode_val, max_val)


def build_tech_lookup(tech_df: pd.DataFrame) -> dict:
    """Build lookup: DS -> (Tech_Min, Tech_Mode, Tech_Max)."""
    lookup = {}
    for _, row in tech_df.iterrows():
        lookup[row["DS"]] = (
            float(row["Tech_Min"]),
            float(row["Tech_Mode"]),
            float(row["Tech_Max"]),
        )
    return lookup


def build_deal_lookup(deal_df: pd.DataFrame) -> dict:
    """Build lookup: DS -> (Deal_Min, Deal_Mode, Deal_Max)."""
    lookup = {}
    for _, row in deal_df.iterrows():
        lookup[row["DS"]] = (
            float(row["Deal_Min"]),
            float(row["Deal_Mode"]),
            float(row["Deal_Max"]),
        )
    return lookup


def build_ra_modifier_lookup(ds_ra_map: pd.DataFrame) -> dict:
    """Build lookup: (DS, RA) -> (RA_Deal_Mod_Min, Mode, Max)."""
    lookup = {}
    for _, row in ds_ra_map.iterrows():
        lookup[(row["DS"], row["RA"])] = (
            float(row["RA_Deal_Mod_Min"]),
            float(row["RA_Deal_Mod_Mode"]),
            float(row["RA_Deal_Mod_Max"]),
        )
    return lookup


def build_ds_ra_lookup(df: pd.DataFrame, value_cols: list) -> dict:
    """Build lookup: (DS, RA) -> tuple of values from specified columns."""
    lookup = {}
    for _, row in df.iterrows():
        lookup[(row["DS"], row["RA"])] = tuple(float(row[col]) for col in value_cols)
    return lookup


def build_allowed_lookup(ds_ra_map: pd.DataFrame) -> dict:
    """Build lookup: (DS, RA) -> bool (whether combination is allowed)."""
    lookup = {}
    for _, row in ds_ra_map.iterrows():
        allowed = str(row["Allowed"]).strip().upper() == "TRUE"
        lookup[(row["DS"], row["RA"])] = allowed
    return lookup
