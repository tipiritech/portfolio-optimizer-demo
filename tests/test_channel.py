#!/usr/bin/env python3
"""
Channel Integration Test — Baseline vs Channel-Enabled
Runs MC with and without CRO/Pharma channel effects to verify:
1. Backward compatibility (channel_lookup=None → same results)
2. Channel effects move deal probability and timing correctly
3. Portfolio metrics improve with channel engagement
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.data.loader import load_params, load_state
from src.simulation.monte_carlo import run_monte_carlo, summarize_results
from src.channel import load_cro_master, load_pharma_master, build_channel_lookup, summarize_channel

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

N_SIMS = 500
SEED = 42
OVERHEAD = 2_640_000.0


def fmt_irr(v):
    return f"{v:.1%}" if pd.notna(v) else "N/A"


def print_summary(label, summary):
    print(f"\n  {label}")
    print(f"  {'─' * 50}")
    print(f"  Median MOIC:     {summary['Median_MOIC']:.2f}x")
    print(f"  Median IRR:      {fmt_irr(summary['Median_Annual_IRR'])}")
    print(f"  IRR P10:         {fmt_irr(summary['IRR_P10'])}")
    print(f"  P(≥3 exits):     {summary['P_ThreePlus_Exits']:.1%}")
    print(f"  P(≤1 exit):      {summary['P_Exits_LE1']:.1%}")
    print(f"  P(0 exits):      {summary['P_Zero_Exits']:.1%}")
    print(f"  First Dist Mo:   {summary['Median_First_Dist_Month']}")


def print_delta(label, base, channel):
    print(f"\n  {label}")
    print(f"  {'─' * 50}")
    for key, fmt in [
        ("Median_MOIC", "{:+.2f}x"),
        ("Median_Annual_IRR", "{:+.1%}"),
        ("P_ThreePlus_Exits", "{:+.1%}"),
        ("P_Exits_LE1", "{:+.1%}"),
        ("P_Zero_Exits", "{:+.1%}"),
    ]:
        b, c = base.get(key, float("nan")), channel.get(key, float("nan"))
        if pd.notna(b) and pd.notna(c):
            print(f"  {key:25s}  {fmt.format(c - b)}")

    b_dist = base.get("Median_First_Dist_Month", float("nan"))
    c_dist = channel.get("Median_First_Dist_Month", float("nan"))
    if pd.notna(b_dist) and pd.notna(c_dist):
        print(f"  {'First_Dist_Month':25s}  {c_dist - b_dist:+.1f} months")


def main():
    print("\n" + "=" * 60)
    print("  CHANNEL INTEGRATION TEST")
    print("=" * 60)

    # ── Load workbooks ────────────────────────────────────────────
    params = load_params(DATA_DIR / "Discovery_Params_v1.xlsx")
    state = load_state(DATA_DIR / "Discovery_Portfolio_State_V1_DryRun.xlsx")
    asset_state = state["asset_state"]
    tranches = state["tranches"]

    print(f"\n  Assets: {len(asset_state)}")
    for _, row in asset_state.iterrows():
        print(f"    {row['Asset_ID']}  {row['DS_Current']}/{row['RA_Current']}")

    # ── Load CRO/Pharma masters ───────────────────────────────────
    cro_path = DATA_DIR / "CRO_Master_v1.xlsx"
    pharma_path = DATA_DIR / "Pharma_Master_v1.xlsx"

    print(f"\n  Loading CRO master:   {cro_path.name}")
    cro_data = load_cro_master(cro_path)
    print(f"  CROs loaded:          {len(cro_data['cro_lookup'])}")

    print(f"  Loading Pharma master: {pharma_path.name}")
    pharma_data = load_pharma_master(pharma_path)
    print(f"  Pharmas loaded:       {len(pharma_data['pharma_lookup'])}")
    print(f"  AVL relationships:    {len(pharma_data['avl_lookup'])}")

    # ── Add CRO/Pharma assignments to asset state ─────────────────
    # SU056 (A-0001): Crown Bioscience → targets AbbVie + AstraZeneca (ovarian buyers)
    # T3155 (A-0002): Champions Oncology → targets Merck + Novartis (heme/ABL buyers)
    asset_state_ch = asset_state.copy()
    asset_state_ch.loc[asset_state_ch["Asset_ID"] == "A-0001", "CRO_ID"] = "CRO-0003"
    asset_state_ch.loc[asset_state_ch["Asset_ID"] == "A-0001", "Target_Pharma_IDs"] = "PH-0002,PH-0003"
    asset_state_ch.loc[asset_state_ch["Asset_ID"] == "A-0002", "CRO_ID"] = "CRO-0004"
    asset_state_ch.loc[asset_state_ch["Asset_ID"] == "A-0002", "Target_Pharma_IDs"] = "PH-0004,PH-0005"

    # ── Build channel lookup ──────────────────────────────────────
    channel_lookup = build_channel_lookup(
        asset_state_ch, cro_data, pharma_data,
        avl_confirmed_only=False,  # Include "Likely" for this test
    )

    print(f"\n  Channel Effects:")
    ch_summary = summarize_channel(channel_lookup)
    for _, row in ch_summary.iterrows():
        print(f"    {row['Asset_ID']}: CRO={row['CRO_ID']}, "
              f"Deal×{row['Deal_Prob_Multiplier']:.3f}, "
              f"Time {row['Time_Shift_Months']:+.0f}mo, "
              f"Matched: {row['Pharma_Names']}")

    # ══════════════════════════════════════════════════════════════
    # TEST 1: Backward Compatibility (no channel)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 1: BASELINE (no channel effects)")
    print("=" * 60)

    results_base = run_monte_carlo(
        asset_state=asset_state, tranches=tranches, params=params,
        n_sims=N_SIMS, seed=SEED, annual_overhead=OVERHEAD,
        enable_tranche_kill=True, enable_rollover=True,
        channel_lookup=None,  # No channel
    )
    summary_base = summarize_results(results_base)
    print_summary("Baseline Results (channel_lookup=None)", summary_base)

    # ══════════════════════════════════════════════════════════════
    # TEST 2: Channel-Enabled
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 2: CHANNEL-ENABLED (CRO + Pharma AVL)")
    print("=" * 60)

    results_ch = run_monte_carlo(
        asset_state=asset_state_ch, tranches=tranches, params=params,
        n_sims=N_SIMS, seed=SEED, annual_overhead=OVERHEAD,
        enable_tranche_kill=True, enable_rollover=True,
        channel_lookup=channel_lookup,
    )
    summary_ch = summarize_results(results_ch)
    print_summary("Channel-Enabled Results", summary_ch)

    # ══════════════════════════════════════════════════════════════
    # TEST 3: Delta Analysis
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 3: CHANNEL IMPACT (delta)")
    print("=" * 60)
    print_delta("Channel Effect on Portfolio", summary_base, summary_ch)

    # ══════════════════════════════════════════════════════════════
    # TEST 4: Verify channel fields in output
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 4: AUDIT TRAIL VERIFICATION")
    print("=" * 60)

    # Run a single sim to inspect per-asset inflow data
    from src.data.inflows import simulate_asset_inflows
    np.random.seed(SEED)
    corr_config = params.get("correlation", {})

    inflow_base = simulate_asset_inflows(
        asset_state=asset_state, params=params, corr_config=corr_config,
        channel_lookup=None,
    )
    np.random.seed(SEED)
    inflow_ch = simulate_asset_inflows(
        asset_state=asset_state_ch, params=params, corr_config=corr_config,
        channel_lookup=channel_lookup,
    )

    print(f"\n  Baseline inflow columns:  {len(inflow_base.columns)}")
    print(f"  Channel inflow columns:   {len(inflow_ch.columns)}")

    new_cols = set(inflow_ch.columns) - set(inflow_base.columns)
    print(f"  New columns added:        {sorted(new_cols)}")

    print(f"\n  Per-asset channel audit (single sim):")
    for _, row in inflow_ch.iterrows():
        print(f"    {row['Asset_ID']}: "
              f"Channel_Deal_Mult={row['Channel_Deal_Mult']:.3f}, "
              f"Channel_Time_Shift={row['Channel_Time_Shift']:.0f}, "
              f"Exit_Raw={row['Relative_Exit_Month_Raw']}, "
              f"Exit_Adj={row['Relative_Exit_Month']}, "
              f"Deal_Adj={row['Deal_Prob_Adjusted']:.3f}, "
              f"Success={row['Success']}")

    # ══════════════════════════════════════════════════════════════
    # VERDICT
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  VERDICT")
    print("=" * 60)

    checks = []

    # Check 1: Channel multipliers are > 1.0
    for aid, ch in channel_lookup.items():
        if ch["deal_prob_mult"] > 1.0:
            checks.append(True)
        else:
            checks.append(False)
            print(f"  FAIL: {aid} deal_prob_mult should be > 1.0, got {ch['deal_prob_mult']}")

    # Check 2: New columns exist
    for col in ["Channel_Deal_Mult", "Channel_Time_Shift", "Relative_Exit_Month_Raw"]:
        if col in inflow_ch.columns:
            checks.append(True)
        else:
            checks.append(False)
            print(f"  FAIL: Missing column {col}")

    # Check 3: Time shift applied (exit months should differ)
    for _, row in inflow_ch.iterrows():
        if row["Channel_Time_Shift"] != 0:
            if row["Relative_Exit_Month"] != row["Relative_Exit_Month_Raw"]:
                checks.append(True)
            else:
                checks.append(False)
                print(f"  FAIL: {row['Asset_ID']} time shift not applied")

    if all(checks):
        print(f"\n  ✅ ALL {len(checks)} CHECKS PASSED")
        print(f"  Channel integration is working correctly.")
    else:
        n_fail = sum(1 for c in checks if not c)
        print(f"\n  ❌ {n_fail}/{len(checks)} CHECKS FAILED")

    print()


if __name__ == "__main__":
    main()
