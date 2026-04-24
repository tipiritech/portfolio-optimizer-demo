"""
Capital outflow construction from tranche schedules.
Supports Entry_Month offset, tranche kill logic, and contingency multiplier.
"""

import numpy as np
import pandas as pd


def build_asset_monthly_outflows(
    tranches: pd.DataFrame,
    asset_state: pd.DataFrame,
    duration: int = 36,
    contingency_mult: float = 1.10,
) -> dict:
    """
    Build monthly outflow vectors for each asset from tranche schedules.

    Features:
        - Entry_Month offset from asset_state
        - Stopped tranches excluded (tranche kill)
        - Contingency multiplier on DevCost (guardrail: 10-15%)
        - AcqCash_to_IP added at entry month

    Args:
        tranches: CAPITAL_TRANCHES DataFrame
        asset_state: ASSET_STATE DataFrame (for Entry_Month, AcqCash_to_IP)
        duration: portfolio duration in months (default 36)
        contingency_mult: multiplier on tranche budgets (default 1.10 = 10%)

    Returns:
        dict: Asset_ID -> numpy array of length duration+1
    """
    asset_ids = tranches["Asset_ID"].dropna().unique()
    outflows = {}

    entry_lookup = asset_state.set_index("Asset_ID")["Entry_Month"].to_dict()
    acq_cash_lookup = asset_state.set_index("Asset_ID")["AcqCash_to_IP"].to_dict()

    for asset_id in asset_ids:
        asset_tranches = tranches[tranches["Asset_ID"] == asset_id].copy()
        monthly = np.zeros(duration + 1)
        entry_month = int(entry_lookup.get(asset_id, 0))

        # Add acquisition cash at entry month
        acq_cash = float(acq_cash_lookup.get(asset_id, 0.0))
        if acq_cash < 0:
            raise ValueError(f"Asset {asset_id}: AcqCash_to_IP={acq_cash} is negative. "
                             f"Acquisition cash must be >= 0.")
        if acq_cash > 0 and 0 <= entry_month <= duration:
            monthly[entry_month] += acq_cash

        for _, row in asset_tranches.iterrows():
            budget = float(row["Budget"])
            status = str(row["Status"]).strip().title()  # Normalize: "active" → "Active"

            # Tranche kill: skip stopped/unknown tranches
            # Completed = already spent (historical), not future deployment
            if status not in ("Planned", "Active"):
                continue

            # Handle both Start_Month/Stop_Month and Start_Date/Stop_Date
            if "Start_Month" in row.index and pd.notna(row.get("Start_Month")):
                tranche_start = int(row["Start_Month"])
                tranche_stop = int(row["Stop_Month"])
            else:
                tranche_start = 0
                tranche_stop = 0

            # Apply contingency multiplier
            budget *= contingency_mult

            # Offset by entry month
            start = entry_month + tranche_start
            stop = entry_month + tranche_stop

            if stop < start or start > duration:
                continue

            stop = min(stop, duration)
            n_months = stop - start + 1
            if n_months <= 0:
                continue

            monthly_spend = budget / n_months
            monthly[start:stop + 1] += monthly_spend

        outflows[asset_id] = monthly

    # Validate: if total outflows are zero, all tranches may be Completed/invalid
    total_all = sum(v.sum() for v in outflows.values())
    if total_all == 0 and len(outflows) > 0:
        raise ValueError("Total deployable capital is zero. All tranches may have Status='Completed' "
                         "or invalid timing. Check CAPITAL_TRANCHES for active/planned tranches.")

    return outflows


def outflows_to_dataframe(outflows: dict, duration: int = 36) -> pd.DataFrame:
    """Convert outflow dict to a readable DataFrame."""
    rows = []
    for asset_id, vec in outflows.items():
        row = {"Asset_ID": asset_id}
        for m in range(duration + 1):
            row[f"Month_{m}"] = vec[m]
        row["Total_Outflow"] = vec.sum()
        rows.append(row)
    return pd.DataFrame(rows)
