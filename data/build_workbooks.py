"""Build Discovery_Params_v1.xlsx and Discovery_Portfolio_State_V1_DryRun.xlsx
Matches the exact schema expected by src/loader.py AND src/inflows.py AND src/cashflows.py
All 6 schema fixes applied."""
import pandas as pd
import os

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

params_path = os.path.join(DATA_DIR, "Discovery_Params_v1.xlsx")

# TIER1_TECH
tier1_tech = pd.DataFrame([
    {"DS": "DS-1", "Tech_Min": 0.20, "Tech_Mode": 0.30, "Tech_Max": 0.40},
    {"DS": "DS-2", "Tech_Min": 0.30, "Tech_Mode": 0.40, "Tech_Max": 0.50},
    {"DS": "DS-3", "Tech_Min": 0.45, "Tech_Mode": 0.55, "Tech_Max": 0.65},
    {"DS": "DS-4", "Tech_Min": 0.55, "Tech_Mode": 0.65, "Tech_Max": 0.75},
    {"DS": "DS-5", "Tech_Min": 0.65, "Tech_Mode": 0.75, "Tech_Max": 0.85},
])

# TIER1_DEAL
tier1_deal = pd.DataFrame([
    {"DS": "DS-1", "Deal_Min": 0.30, "Deal_Mode": 0.40, "Deal_Max": 0.50},
    {"DS": "DS-2", "Deal_Min": 0.40, "Deal_Mode": 0.50, "Deal_Max": 0.60},
    {"DS": "DS-3", "Deal_Min": 0.55, "Deal_Mode": 0.65, "Deal_Max": 0.75},
    {"DS": "DS-4", "Deal_Min": 0.65, "Deal_Mode": 0.75, "Deal_Max": 0.85},
    {"DS": "DS-5", "Deal_Min": 0.70, "Deal_Mode": 0.80, "Deal_Max": 0.90},
])

# TIER1_ECON — FIX #6: NearMilestones_* (not NearMS_*)
tier1_econ = pd.DataFrame([
    {"DS": "DS-1", "RA": "RA-1", "Upfront_Min": 5e6, "Upfront_Mode": 10e6, "Upfront_Max": 20e6,
     "NearMilestones_Min": 3e6, "NearMilestones_Mode": 8e6, "NearMilestones_Max": 15e6},
    {"DS": "DS-2", "RA": "RA-1", "Upfront_Min": 10e6, "Upfront_Mode": 20e6, "Upfront_Max": 40e6,
     "NearMilestones_Min": 7e6, "NearMilestones_Mode": 15e6, "NearMilestones_Max": 30e6},
    {"DS": "DS-3", "RA": "RA-1", "Upfront_Min": 15e6, "Upfront_Mode": 30e6, "Upfront_Max": 55e6,
     "NearMilestones_Min": 10e6, "NearMilestones_Mode": 25e6, "NearMilestones_Max": 50e6},
    {"DS": "DS-4", "RA": "RA-1", "Upfront_Min": 20e6, "Upfront_Mode": 40e6, "Upfront_Max": 75e6,
     "NearMilestones_Min": 15e6, "NearMilestones_Mode": 35e6, "NearMilestones_Max": 65e6},
    {"DS": "DS-5", "RA": "RA-1", "Upfront_Min": 30e6, "Upfront_Mode": 55e6, "Upfront_Max": 100e6,
     "NearMilestones_Min": 20e6, "NearMilestones_Mode": 45e6, "NearMilestones_Max": 85e6},
    {"DS": "DS-3", "RA": "RA-2", "Upfront_Min": 13.5e6, "Upfront_Mode": 27e6, "Upfront_Max": 49.5e6,
     "NearMilestones_Min": 9e6, "NearMilestones_Mode": 22.5e6, "NearMilestones_Max": 45e6},
    {"DS": "DS-4", "RA": "RA-2", "Upfront_Min": 18e6, "Upfront_Mode": 36e6, "Upfront_Max": 67.5e6,
     "NearMilestones_Min": 13.5e6, "NearMilestones_Mode": 31.5e6, "NearMilestones_Max": 58.5e6},
    {"DS": "DS-5", "RA": "RA-2", "Upfront_Min": 27e6, "Upfront_Mode": 49.5e6, "Upfront_Max": 90e6,
     "NearMilestones_Min": 18e6, "NearMilestones_Mode": 40.5e6, "NearMilestones_Max": 76.5e6},
])

# TIER1_COST_TIME — includes MilestoneLag columns (Fix #5)
tier1_cost_time = pd.DataFrame([
    {"DS": "DS-1", "RA": "RA-1", "DevCost_Min": 8e6, "DevCost_Mode": 12e6, "DevCost_Max": 18e6,
     "TimeToExit_Min": 18, "TimeToExit_Mode": 24, "TimeToExit_Max": 34,
     "MilestoneLag_Min": 0, "MilestoneLag_Mode": 6, "MilestoneLag_Max": 12},
    {"DS": "DS-2", "RA": "RA-1", "DevCost_Min": 6e6, "DevCost_Mode": 9e6, "DevCost_Max": 14e6,
     "TimeToExit_Min": 15, "TimeToExit_Mode": 21, "TimeToExit_Max": 30,
     "MilestoneLag_Min": 0, "MilestoneLag_Mode": 6, "MilestoneLag_Max": 12},
    {"DS": "DS-3", "RA": "RA-1", "DevCost_Min": 4e6, "DevCost_Mode": 7e6, "DevCost_Max": 10e6,
     "TimeToExit_Min": 12, "TimeToExit_Mode": 18, "TimeToExit_Max": 24,
     "MilestoneLag_Min": 0, "MilestoneLag_Mode": 6, "MilestoneLag_Max": 12},
    {"DS": "DS-4", "RA": "RA-1", "DevCost_Min": 3e6, "DevCost_Mode": 4.5e6, "DevCost_Max": 6e6,
     "TimeToExit_Min": 10, "TimeToExit_Mode": 14, "TimeToExit_Max": 20,
     "MilestoneLag_Min": 0, "MilestoneLag_Mode": 6, "MilestoneLag_Max": 12},
    {"DS": "DS-5", "RA": "RA-1", "DevCost_Min": 2e6, "DevCost_Mode": 3e6, "DevCost_Max": 5e6,
     "TimeToExit_Min": 6, "TimeToExit_Mode": 10, "TimeToExit_Max": 16,
     "MilestoneLag_Min": 0, "MilestoneLag_Mode": 6, "MilestoneLag_Max": 12},
    {"DS": "DS-3", "RA": "RA-2", "DevCost_Min": 4.5e6, "DevCost_Mode": 7.5e6, "DevCost_Max": 11e6,
     "TimeToExit_Min": 13, "TimeToExit_Mode": 19, "TimeToExit_Max": 26,
     "MilestoneLag_Min": 3, "MilestoneLag_Mode": 9, "MilestoneLag_Max": 15},
    {"DS": "DS-4", "RA": "RA-2", "DevCost_Min": 3.5e6, "DevCost_Mode": 5e6, "DevCost_Max": 7e6,
     "TimeToExit_Min": 11, "TimeToExit_Mode": 15, "TimeToExit_Max": 22,
     "MilestoneLag_Min": 3, "MilestoneLag_Mode": 9, "MilestoneLag_Max": 15},
    {"DS": "DS-5", "RA": "RA-2", "DevCost_Min": 2.5e6, "DevCost_Mode": 3.5e6, "DevCost_Max": 5.5e6,
     "TimeToExit_Min": 7, "TimeToExit_Mode": 11, "TimeToExit_Max": 18,
     "MilestoneLag_Min": 3, "MilestoneLag_Mode": 9, "MilestoneLag_Max": 15},
])

# TIER2 = same structure, economics discounted 20%
tier2_tech = tier1_tech.copy()
tier2_deal = tier1_deal.copy()
tier2_cost_time = tier1_cost_time.copy()
tier2_econ = tier1_econ.copy()
for col in ["Upfront_Min", "Upfront_Mode", "Upfront_Max",
            "NearMilestones_Min", "NearMilestones_Mode", "NearMilestones_Max"]:
    tier2_econ[col] = tier2_econ[col] * 0.80

# REGIME — FULL COLUMN NAMES (Fix #3)
regime = pd.DataFrame([
    {"Regime": "Tight", "Probability": 0.25, "Deal_Multiplier": 0.85, "Upfront_Multiplier": 0.80, "Time_Multiplier": 1.15},
    {"Regime": "Neutral", "Probability": 0.50, "Deal_Multiplier": 1.00, "Upfront_Multiplier": 1.00, "Time_Multiplier": 1.00},
    {"Regime": "Hot", "Probability": 0.25, "Deal_Multiplier": 1.15, "Upfront_Multiplier": 1.20, "Time_Multiplier": 0.90},
])

# DS_RA_MAP — WITH RA_Deal_Mod columns AND boolean Allowed (Fix #4)
ds_ra_map = pd.DataFrame([
    {"DS": "DS-1", "RA": "RA-1", "Allowed": True, "RA_Deal_Mod_Min": 0.95, "RA_Deal_Mod_Mode": 1.00, "RA_Deal_Mod_Max": 1.05},
    {"DS": "DS-2", "RA": "RA-1", "Allowed": True, "RA_Deal_Mod_Min": 0.95, "RA_Deal_Mod_Mode": 1.00, "RA_Deal_Mod_Max": 1.05},
    {"DS": "DS-3", "RA": "RA-1", "Allowed": True, "RA_Deal_Mod_Min": 0.95, "RA_Deal_Mod_Mode": 1.00, "RA_Deal_Mod_Max": 1.05},
    {"DS": "DS-3", "RA": "RA-2", "Allowed": True, "RA_Deal_Mod_Min": 0.85, "RA_Deal_Mod_Mode": 0.90, "RA_Deal_Mod_Max": 0.95},
    {"DS": "DS-4", "RA": "RA-1", "Allowed": True, "RA_Deal_Mod_Min": 0.95, "RA_Deal_Mod_Mode": 1.00, "RA_Deal_Mod_Max": 1.05},
    {"DS": "DS-4", "RA": "RA-2", "Allowed": True, "RA_Deal_Mod_Min": 0.85, "RA_Deal_Mod_Mode": 0.90, "RA_Deal_Mod_Max": 0.95},
    {"DS": "DS-5", "RA": "RA-1", "Allowed": True, "RA_Deal_Mod_Min": 0.95, "RA_Deal_Mod_Mode": 1.00, "RA_Deal_Mod_Max": 1.05},
    {"DS": "DS-5", "RA": "RA-2", "Allowed": True, "RA_Deal_Mod_Min": 0.85, "RA_Deal_Mod_Mode": 0.90, "RA_Deal_Mod_Max": 0.95},
])

# CORRELATION_FACTORS
corr_factors = pd.DataFrame([
    {"Factor": "Market", "StdDev": 1.0, "Default_Loading": 0.35},
    {"Factor": "MechanismCluster", "StdDev": 1.0, "Default_Loading": 0.35},
    {"Factor": "IndicationCluster", "StdDev": 1.0, "Default_Loading": 0.20},
    {"Factor": "GeoRACluster", "StdDev": 1.0, "Default_Loading": 0.10},
    {"Factor": "CROCluster", "StdDev": 1.0, "Default_Loading": 0.05},
    {"Factor": "Idiosyncratic", "StdDev": 1.0, "Default_Loading": 0.50},
    {"Factor": "BaseCorrelationFloor", "StdDev": None, "Default_Loading": 0.25},
    {"Factor": "CorrelationStressAdd", "StdDev": None, "Default_Loading": 0.10},
])

# ENVELOPE_THRESHOLDS
envelope = pd.DataFrame([
    {"Metric": "Target_Median_IRR", "Value": 0.25},
    {"Metric": "Floor_IRR_P10", "Value": 0.00},
    {"Metric": "Min_P_Exits_GE3", "Value": 0.60},
    {"Metric": "Max_P_Exits_LE1", "Value": 0.15},
    {"Metric": "Max_CorrIndex", "Value": 0.90},
    {"Metric": "Max_Weighted_Time", "Value": 24.0},
    {"Metric": "Max_Duration", "Value": 36.0},
    {"Metric": "Min_Upfront_Threshold", "Value": 5_000_000.0},
    {"Metric": "EDC_Stop_Threshold", "Value": 0.01},
    {"Metric": "Capital_Pause_Time", "Value": 30.0},
    {"Metric": "Max_Concentration", "Value": 0.20},
    {"Metric": "Warn_Concentration", "Value": 0.15},
    {"Metric": "Min_Combined_Prob", "Value": 0.45},
])

with pd.ExcelWriter(params_path, engine="openpyxl") as writer:
    tier1_tech.to_excel(writer, sheet_name="TIER1_TECH", index=False)
    tier1_deal.to_excel(writer, sheet_name="TIER1_DEAL", index=False)
    tier1_econ.to_excel(writer, sheet_name="TIER1_ECON", index=False)
    tier1_cost_time.to_excel(writer, sheet_name="TIER1_COST_TIME", index=False)
    tier2_tech.to_excel(writer, sheet_name="TIER2_TECH", index=False)
    tier2_deal.to_excel(writer, sheet_name="TIER2_DEAL", index=False)
    tier2_econ.to_excel(writer, sheet_name="TIER2_ECON", index=False)
    tier2_cost_time.to_excel(writer, sheet_name="TIER2_COST_TIME", index=False)
    regime.to_excel(writer, sheet_name="REGIME", index=False)
    ds_ra_map.to_excel(writer, sheet_name="DS_RA_MAP", index=False)
    corr_factors.to_excel(writer, sheet_name="CORRELATION_FACTORS", index=False)
    envelope.to_excel(writer, sheet_name="ENVELOPE_THRESHOLDS", index=False)

print(f"✓ Params: {params_path}")

# ============================================================
# PORTFOLIO STATE WORKBOOK
# ============================================================
state_path = os.path.join(DATA_DIR, "Discovery_Portfolio_State_V1_DryRun.xlsx")

# ASSET_ROSTER
asset_roster = pd.DataFrame([
    {"Asset_ID": "A-0001", "Name": "SU056", "Tier": "Tier-1", "DS": "DS-3", "RA": "RA-1",
     "Mechanism": "YB-1 inhibitor (first-in-class)", "Indication": "Ovarian cancer / chemo-resistance",
     "MechCluster_ID": "MECH-YB1", "IndicationCluster_ID": "IND-OV", "GeoRACluster_ID": "GEO-US1"},
    {"Asset_ID": "A-0002", "Name": "T3155", "Tier": "Tier-1", "DS": "DS-4", "RA": "RA-1",
     "Mechanism": "BCR-ABL TKI (next-gen)", "Indication": "Resistant myeloid leukemias (T315I+)",
     "MechCluster_ID": "MECH-ABL", "IndicationCluster_ID": "IND-HEM", "GeoRACluster_ID": "GEO-US1"},
])

# ASSET_STATE — DS_Current/RA_Current (Fix #2)
asset_state = pd.DataFrame([
    {"Asset_ID": "A-0001", "DS_Current": "DS-3", "RA_Current": "RA-1", "Tier": "Tier-1",
     "Entry_Month": 0, "Equity_to_IP_Pct": 0.10, "AcqCash_to_IP": 0,
     "EarlyPassThrough_Pct": 0.0, "EarlyDeferredCash": 0},
    {"Asset_ID": "A-0002", "DS_Current": "DS-4", "RA_Current": "RA-1", "Tier": "Tier-1",
     "Entry_Month": 0, "Equity_to_IP_Pct": 0.10, "AcqCash_to_IP": 0,
     "EarlyPassThrough_Pct": 0.0, "EarlyDeferredCash": 0},
])

# CAPITAL_TRANCHES — with Status column (Fix #1)
tranches = pd.DataFrame([
    {"Asset_ID": "A-0001", "Tranche_ID": "T1", "Purpose": "IND-Enabling Development",
     "Budget": 5_500_000, "Start_Month": 0, "Stop_Month": 14, "Status": "Active"},
    {"Asset_ID": "A-0001", "Tranche_ID": "T2", "Purpose": "Transaction / BD",
     "Budget": 1_500_000, "Start_Month": 10, "Stop_Month": 18, "Status": "Planned"},
    {"Asset_ID": "A-0002", "Tranche_ID": "T1", "Purpose": "IND-Ready Development",
     "Budget": 3_500_000, "Start_Month": 0, "Stop_Month": 10, "Status": "Active"},
    {"Asset_ID": "A-0002", "Tranche_ID": "T2", "Purpose": "Transaction / BD",
     "Budget": 1_000_000, "Start_Month": 8, "Stop_Month": 14, "Status": "Planned"},
])

# CONTROL_PANEL
control_panel = pd.DataFrame([
    ["Vehicle_ID", "V-001"],
    ["Param_Set_ID", "PS-0001"],
    ["Max_Duration", 36],
    ["Annual_Overhead", 2640000],
    ["Contingency_Mult", 1.10],
    ["Upfront_Threshold", 5000000],
    ["Target_Assets_Min", 6],
    ["Target_Assets_Max", 10],
    ["Hard_Cap_Assets", 12],
])

with pd.ExcelWriter(state_path, engine="openpyxl") as writer:
    asset_roster.to_excel(writer, sheet_name="ASSET_ROSTER", index=False)
    asset_state.to_excel(writer, sheet_name="ASSET_STATE", index=False)
    tranches.to_excel(writer, sheet_name="CAPITAL_TRANCHES", index=False)
    control_panel.to_excel(writer, sheet_name="CONTROL_PANEL", index=False, header=False)

print(f"✓ State: {state_path}")
