"""
Net cashflow assembly.
Combines outflows and inflows into monthly net cashflow vectors per asset and portfolio.
"""

import numpy as np
import pandas as pd


def build_asset_net_cashflows(
    outflows: dict,
    inflow_df: pd.DataFrame,
    duration: int = 36,
) -> dict:
    """
    Combine monthly outflows with simulated inflows into net cashflow vectors.

    Returns:
        dict: Asset_ID -> numpy array of monthly net cashflows
    """
    net_cashflows = {}

    for asset_id, outflow_vec in outflows.items():
        net = np.zeros(duration + 1)
        net -= outflow_vec

        asset_rows = inflow_df[inflow_df["Asset_ID"] == asset_id]
        if asset_rows.empty:
            net_cashflows[asset_id] = net
            continue

        row = asset_rows.iloc[0]

        if row["Success"] and pd.notna(row["Exit_Month"]):
            exit_month = int(row["Exit_Month"])
            upfront = float(row["Upfront"])
            milestone = float(row["Near_Milestone"])

            if 0 <= exit_month <= duration:
                net[exit_month] += upfront

            milestone_lag = int(row["Milestone_Lag"]) if pd.notna(row["Milestone_Lag"]) else 0
            milestone_month = exit_month + milestone_lag
            if 0 <= milestone_month <= duration:
                net[milestone_month] += milestone

        net_cashflows[asset_id] = net

    return net_cashflows


def build_portfolio_cashflow(net_cashflows: dict, duration: int = 36) -> np.ndarray:
    """Sum asset monthly cashflows into one portfolio cashflow vector."""
    portfolio = np.zeros(duration + 1)
    for vec in net_cashflows.values():
        portfolio += vec
    return portfolio


def net_cashflows_to_dataframe(net_cashflows: dict, duration: int = 36) -> pd.DataFrame:
    """Convert net cashflow dict to a readable DataFrame."""
    rows = []
    for asset_id, vec in net_cashflows.items():
        row = {"Asset_ID": asset_id}
        for m in range(duration + 1):
            row[f"Month_{m}"] = vec[m]
        row["Total_Net"] = vec.sum()
        rows.append(row)
    return pd.DataFrame(rows)
