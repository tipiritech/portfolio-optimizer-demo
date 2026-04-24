"""
Asset inflow simulation.
Samples success/failure, exit timing, and economics for each asset per simulation run.
Supports Tier-1 and Tier-2 parameters, IP economics netting, and qualifying exit threshold.
"""

import numpy as np
import pandas as pd

from src.governance.param_lookup import (
    triangular_sample,
    build_tech_lookup,
    build_deal_lookup,
    build_ra_modifier_lookup,
    build_ds_ra_lookup,
)
from src.optimization.correlation import compute_correlated_success_probs


def simulate_asset_inflows(
    asset_state: pd.DataFrame,
    params: dict,
    corr_config: dict,
    use_corr_stress: bool = False,
    use_tight_only: bool = False,
    duration: int = 36,
    upfront_threshold: float = 5_000_000.0,
    channel_lookup: dict | None = None,
) -> pd.DataFrame:
    """
    Simulate inflows for all assets in one Monte Carlo run.

    Key improvements over original:
        - Reads deal probability from TIER1_DEAL / TIER2_DEAL (not hardcoded)
        - Supports Tier-1 and Tier-2 parameter sets
        - Applies IP economics netting (equity, pass-through, deferred cash)
        - Enforces upfront threshold for qualifying exits
        - 36-month hard stop: exit beyond duration = failure

    Args:
        asset_state: merged asset state with cluster IDs
        params: full parameter dict from loader
        corr_config: correlation configuration dict
        use_corr_stress: apply correlation stress add-on
        use_tight_only: force Tight regime (stress test)
        duration: hard stop in months
        upfront_threshold: minimum upfront for qualifying exit

    Returns:
        DataFrame with one row per asset containing success, timing, economics
    """
    from src.data.loader import get_tier_tables

    # Build lookups for both tiers
    tier_lookups = {}
    for tier_name in ("Tier-1", "Tier-2"):
        tables = get_tier_tables(params, tier_name)
        tier_lookups[tier_name] = {
            "tech": build_tech_lookup(tables["tech"]),
            "deal": build_deal_lookup(tables["deal"]),
            "cost_time": build_ds_ra_lookup(
                tables["cost_time"],
                ["TimeToExit_Min", "TimeToExit_Mode", "TimeToExit_Max",
                 "MilestoneLag_Min", "MilestoneLag_Mode", "MilestoneLag_Max"],
            ),
            "econ": build_ds_ra_lookup(
                tables["econ"],
                ["Upfront_Min", "Upfront_Mode", "Upfront_Max",
                 "NearMilestones_Min", "NearMilestones_Mode", "NearMilestones_Max"],
            ),
        }

    ra_mod_lookup = build_ra_modifier_lookup(params["ds_ra_map"])

    # Sample regime for this run
    regime_df = params["regime"]
    if use_tight_only:
        sampled_regime = "Tight"
    else:
        regimes = regime_df["Regime"].tolist()
        regime_probs = regime_df["Probability"].astype(float).tolist()
        sampled_regime = np.random.choice(regimes, p=regime_probs)

    regime_row = regime_df[regime_df["Regime"] == sampled_regime].iloc[0]
    deal_mult = float(regime_row["Deal_Multiplier"])
    upfront_mult = float(regime_row["Upfront_Multiplier"])
    time_mult = float(regime_row["Time_Multiplier"])

    base_success_probs = []
    sampled_rows = []

    for _, row in asset_state.iterrows():
        asset_id = row["Asset_ID"]
        ds = row["DS_Current"]
        ra = row["RA_Current"]
        tier = row.get("Tier", "Tier-1")
        entry_month = int(row["Entry_Month"])

        # Select correct tier lookups
        if tier not in tier_lookups:
            tier = "Tier-1"
        lk = tier_lookups[tier]

        # Sample technical probability
        tech_min, tech_mode, tech_max = lk["tech"].get(ds, (0.5, 0.6, 0.7))
        tech_p = triangular_sample(tech_min, tech_mode, tech_max)

        # Sample deal probability from DEAL table (NOT hardcoded)
        deal_min, deal_mode, deal_max = lk["deal"].get(ds, (0.5, 0.6, 0.7))
        deal_p_base = triangular_sample(deal_min, deal_mode, deal_max)

        # Apply RA modifier
        ra_min, ra_mode, ra_max = ra_mod_lookup.get((ds, ra), (1.0, 1.0, 1.0))
        ra_mod = triangular_sample(ra_min, ra_mode, ra_max)

        # Apply regime deal multiplier
        deal_p = deal_p_base * ra_mod * deal_mult
        deal_p = min(max(deal_p, 0.01), 0.99)

        # Apply channel effects (CRO + AVL boost on deal prob, time shift)
        ch_entry = (channel_lookup or {}).get(asset_id)
        if ch_entry is not None:
            ch_deal_mult = ch_entry["deal_prob_mult"]
            ch_time_shift = ch_entry["time_shift_months"]
        else:
            ch_deal_mult = 1.0
            ch_time_shift = 0.0

        deal_p_channeled = deal_p * ch_deal_mult
        deal_p_channeled = min(max(deal_p_channeled, 0.01), 0.99)

        # Combined base success probability
        base_success_p = min(tech_p * deal_p_channeled, 0.99)

        # Sample timing
        time_params = lk["cost_time"].get((ds, ra))
        if time_params:
            tmin, tmode, tmax, lag_min, lag_mode, lag_max = time_params
        else:
            tmin, tmode, tmax = 12, 18, 24
            lag_min, lag_mode, lag_max = 0, 6, 12

        relative_exit_month = int(round(triangular_sample(tmin, tmode, tmax) * time_mult))
        relative_exit_month_raw = relative_exit_month  # pre-channel for audit
        # Apply channel time shift
        relative_exit_month = max(1, relative_exit_month + int(ch_time_shift))
        exit_month = entry_month + relative_exit_month
        milestone_lag = int(round(triangular_sample(lag_min, lag_mode, lag_max) * time_mult))  # #6: time_mult on lag too

        # Sample economics
        econ_params = lk["econ"].get((ds, ra))
        if econ_params:
            umin, umode, umax, mmin, mmode, mmax = econ_params
        else:
            umin, umode, umax = 5e6, 10e6, 25e6
            mmin, mmode, mmax = 0, 5e6, 15e6

        # [#14] Acquisition exit mode: small probability of acquisition vs license
        # Acquisition probs by stage (from transaction benchmark synthesis)
        acq_probs = {"DS-1": 0.02, "DS-2": 0.04, "DS-3": 0.06, "DS-4": 0.06, "DS-5": 0.10}
        acq_mults = {"DS-1": 6.0, "DS-2": 5.0, "DS-3": 4.0, "DS-4": 3.5, "DS-5": 3.0}
        is_acquisition = np.random.rand() < acq_probs.get(ds, 0.06)

        if is_acquisition:
            # Acquisition: higher upfront, no near-term milestones (all cash at close)
            acq_mult = acq_mults.get(ds, 4.0)
            upfront_gross = triangular_sample(umin, umode, umax) * upfront_mult * acq_mult
            milestone_gross = 0.0  # acquisition = all upfront
        else:
            upfront_gross = triangular_sample(umin, umode, umax) * upfront_mult
            milestone_gross = triangular_sample(mmin, mmode, mmax)

        base_success_probs.append(base_success_p)
        sampled_rows.append({
            "Asset_ID": asset_id,
            "DS_Current": ds,
            "RA_Current": ra,
            "Tier": tier,
            "Entry_Month": entry_month,
            "Sampled_Regime": sampled_regime,
            "Tech_Prob": round(tech_p, 4),
            "Deal_Prob_Base": round(deal_p_base, 4),
            "RA_Modifier": round(ra_mod, 4),
            "Deal_Prob_Adjusted": round(deal_p_channeled, 4),
            "Base_Success_Prob": round(base_success_p, 4),
            "Relative_Exit_Month": relative_exit_month,
            "Relative_Exit_Month_Raw": relative_exit_month_raw,
            "Exit_Month": exit_month,
            "Milestone_Lag": milestone_lag,
            "Upfront_Gross": upfront_gross,
            "Near_Milestone_Gross": milestone_gross,
            "Exit_Type": "Acquisition" if is_acquisition else "License",
            "Channel_Deal_Mult": round(ch_deal_mult, 4),
            "Channel_Time_Shift": ch_time_shift,
        })

    # Apply correlation
    base_probs_series = pd.Series(base_success_probs)
    correlated_probs = compute_correlated_success_probs(
        asset_state, base_probs_series, corr_config, use_stress=use_corr_stress
    )

    # Determine success/failure and apply IP netting
    results = []
    for i, sampled in enumerate(sampled_rows):
        success_p = float(correlated_probs.iloc[i])
        success = np.random.rand() < success_p

        exit_month = sampled["Exit_Month"]
        upfront_gross = sampled["Upfront_Gross"]
        milestone_gross = sampled["Near_Milestone_Gross"]
        milestone_lag = sampled["Milestone_Lag"]

        # Hard stop: exit beyond duration = failure
        if exit_month > duration:
            success = False

        # Check upfront threshold for qualifying exit
        if success and upfront_gross < upfront_threshold:
            success = False

        if not success:
            exit_month = None
            milestone_lag = None
            upfront_net = 0.0
            milestone_net = 0.0
            upfront_gross = 0.0
            milestone_gross = 0.0
        else:
            # Apply IP economics netting
            asset_row = asset_state.iloc[i]
            equity_pct = float(asset_row.get("Equity_to_IP_Pct", 0.0))
            passthru_pct = float(asset_row.get("EarlyPassThrough_Pct", 0.0))
            deferred_cash = float(asset_row.get("EarlyDeferredCash", 0.0))

            # Net_i = (1 - e_i) * (1 - q_i) * G_i - d_i
            gross_total = upfront_gross + milestone_gross
            investor_share = (1.0 - equity_pct) * (1.0 - passthru_pct)
            net_total = investor_share * gross_total - deferred_cash
            net_total = max(net_total, 0.0)

            # Proportionally split net back to upfront and milestone
            if gross_total > 0:
                upfront_net = net_total * (upfront_gross / gross_total)
                milestone_net = net_total * (milestone_gross / gross_total)
            else:
                upfront_net = 0.0
                milestone_net = 0.0

        results.append({
            "Asset_ID": sampled["Asset_ID"],
            "DS_Current": sampled["DS_Current"],
            "RA_Current": sampled["RA_Current"],
            "Tier": sampled["Tier"],
            "Entry_Month": sampled["Entry_Month"],
            "Sampled_Regime": sampled["Sampled_Regime"],
            "Tech_Prob": sampled["Tech_Prob"],
            "Deal_Prob_Base": sampled["Deal_Prob_Base"],
            "RA_Modifier": sampled["RA_Modifier"],
            "Deal_Prob_Adjusted": sampled["Deal_Prob_Adjusted"],
            "Base_Success_Prob": sampled["Base_Success_Prob"],
            "Correlated_Success_Prob": round(success_p, 4),
            "Success": success,
            "Exit_Type": sampled["Exit_Type"] if success else None,
            "Relative_Exit_Month": sampled["Relative_Exit_Month"],
            "Relative_Exit_Month_Raw": sampled["Relative_Exit_Month_Raw"],
            "Exit_Month": exit_month,
            "Milestone_Lag": milestone_lag,
            "Upfront_Gross": round(sampled["Upfront_Gross"], 2) if success else 0.0,
            "Near_Milestone_Gross": round(sampled["Near_Milestone_Gross"], 2) if success else 0.0,
            "Upfront": round(upfront_net, 2),
            "Near_Milestone": round(milestone_net, 2),
            "Total_Inflow": round(upfront_net + milestone_net, 2),
            "Channel_Deal_Mult": sampled["Channel_Deal_Mult"],
            "Channel_Time_Shift": sampled["Channel_Time_Shift"],
        })

    return pd.DataFrame(results)
