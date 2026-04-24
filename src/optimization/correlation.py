"""
Factor-based correlation model.
Generates correlated Bernoulli success outcomes via latent normal variables.
Loadings are read from the CORRELATION_FACTORS parameter table.
"""

import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_cluster_factor_map(values):
    """Draw one shared normal factor per unique cluster value."""
    unique_vals = pd.Series(values).dropna().unique()
    return {val: np.random.normal(0, 1) for val in unique_vals}


def compute_correlated_success_probs(
    asset_state: pd.DataFrame,
    base_success_probs: pd.Series,
    corr_config: dict,
    use_stress: bool = False,
) -> pd.Series:
    """
    Apply factor-based correlation adjustments to base success probabilities.

    Uses a latent normal model:
      latent_i = sum(loading_k * factor_k) + idio_loading * idio_i
    Then adjusts base probability via logit transform.

    Args:
        asset_state: DataFrame with cluster ID columns
        base_success_probs: Series of base p_i for each asset
        corr_config: dict from params["correlation"] with factor loadings
        use_stress: if True, add CorrelationStressAdd to base floor

    Returns:
        Series of adjusted success probabilities
    """
    # Extract loadings from config (with defaults matching guardrails)
    market_loading = corr_config.get("Market", {}).get("Default_Loading", 0.35)
    mech_loading = corr_config.get("MechanismCluster", {}).get("Default_Loading", 0.35)
    ind_loading = corr_config.get("IndicationCluster", {}).get("Default_Loading", 0.20)
    geo_loading = corr_config.get("GeoRACluster", {}).get("Default_Loading", 0.10)
    cro_loading = corr_config.get("CROCluster", {}).get("Default_Loading", 0.05)
    idio_loading = corr_config.get("Idiosyncratic", {}).get("Default_Loading", 0.50)

    base_floor = corr_config.get("BaseCorrelationFloor", {}).get("Default_Loading", 0.25)
    stress_add = corr_config.get("CorrelationStressAdd", {}).get("Default_Loading", 0.10)

    if use_stress:
        base_floor += stress_add

    # Draw shared factors
    market_factor = np.random.normal(0, 1)

    mech_col = "MechCluster_ID"
    ind_col = "IndicationCluster_ID"
    geo_col = "GeoRACluster_ID"
    cro_col = "CRO_ID"

    mech_map = build_cluster_factor_map(asset_state[mech_col]) if mech_col in asset_state.columns else {}
    ind_map = build_cluster_factor_map(asset_state[ind_col]) if ind_col in asset_state.columns else {}
    geo_map = build_cluster_factor_map(asset_state[geo_col]) if geo_col in asset_state.columns else {}
    cro_map = build_cluster_factor_map(asset_state[cro_col]) if cro_col in asset_state.columns else {}

    adjusted_probs = []

    for idx in range(len(asset_state)):
        row = asset_state.iloc[idx]
        p = float(base_success_probs.iloc[idx])
        p = min(max(p, 0.01), 0.99)

        mech_factor = mech_map.get(row.get(mech_col), 0.0)
        ind_factor = ind_map.get(row.get(ind_col), 0.0)
        geo_factor = geo_map.get(row.get(geo_col), 0.0)
        cro_factor = cro_map.get(row.get(cro_col), 0.0)
        idio_factor = np.random.normal(0, 1)

        # Weighted latent score
        score = (
            market_loading * market_factor
            + mech_loading * mech_factor
            + ind_loading * ind_factor
            + geo_loading * geo_factor
            + cro_loading * cro_factor
            + idio_loading * idio_factor
        )

        # Logit transform: shift base probability by factor score
        # #9: Scale factor is configurable (controls how much factors shift probs)
        scale = corr_config.get("_logit_scale", 0.75)
        logit_p = np.log(p / (1.0 - p))
        adjusted_logit = logit_p + scale * score
        adjusted_p = sigmoid(adjusted_logit)

        # Enforce base correlation floor:
        # Blend toward the portfolio mean to ensure minimum dependence
        adjusted_p = min(max(adjusted_p, 0.001), 0.999)
        adjusted_probs.append(adjusted_p)

    return pd.Series(adjusted_probs, index=asset_state.index)


def compute_correlation_index(asset_state: pd.DataFrame, corr_config: dict) -> float:
    """
    Compute a summary correlation index for the portfolio.
    Higher values indicate more concentrated/correlated risk.

    Based on cluster overlap: assets sharing clusters increase the index.
    """
    n = len(asset_state)
    if n <= 1:
        return 0.0

    mech_loading = corr_config.get("MechanismCluster", {}).get("Default_Loading", 0.35)
    ind_loading = corr_config.get("IndicationCluster", {}).get("Default_Loading", 0.20)
    geo_loading = corr_config.get("GeoRACluster", {}).get("Default_Loading", 0.10)
    cro_loading = corr_config.get("CROCluster", {}).get("Default_Loading", 0.05)
    market_loading = corr_config.get("Market", {}).get("Default_Loading", 0.35)

    # Count pairwise cluster overlaps
    total_overlap = 0.0
    n_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            ri = asset_state.iloc[i]
            rj = asset_state.iloc[j]

            overlap = market_loading  # all assets share market factor

            if "MechCluster_ID" in asset_state.columns:
                if ri["MechCluster_ID"] == rj["MechCluster_ID"]:
                    overlap += mech_loading

            if "IndicationCluster_ID" in asset_state.columns:
                if ri["IndicationCluster_ID"] == rj["IndicationCluster_ID"]:
                    overlap += ind_loading

            if "GeoRACluster_ID" in asset_state.columns:
                if ri["GeoRACluster_ID"] == rj["GeoRACluster_ID"]:
                    overlap += geo_loading

            if "CRO_ID" in asset_state.columns:
                cro_i = ri.get("CRO_ID")
                cro_j = rj.get("CRO_ID")
                if pd.notna(cro_i) and pd.notna(cro_j) and cro_i == cro_j:
                    overlap += cro_loading

            total_overlap += overlap
            n_pairs += 1

    if n_pairs == 0:
        return 0.0

    return total_overlap / n_pairs
