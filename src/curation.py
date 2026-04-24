"""
Portfolio Curation Optimizer.
Generates target asset profiles and optimizes portfolio construction
via greedy sequential fill with lexicographic ranking.

Design: 06_Curation_Optimizer_Architecture.md
Scoring: Lexicographic ordering (EDC > IRC > DPC > LAC > CDC)
  - No subjective weights per doctrine
Governance: Hard boundary — any envelope breach = infeasible
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.simulation.monte_carlo import run_monte_carlo, summarize_results
from src.optimization.correlation import compute_correlation_index
from src.optimization.envelope import check_envelope

# ── Cluster Universes ──────────────────────────────────────────────────────────
MECH_CLUSTERS = [
    "MECH-YB1", "MECH-ABL",  # locked
    "MECH-CDK", "MECH-KRAS", "MECH-ICK", "MECH-ADC", "MECH-PROTAC", "MECH-EPIG",
]
IND_CLUSTERS = [
    "IND-OV", "IND-HEM",  # locked
    "IND-NSCLC", "IND-CRC", "IND-MEL", "IND-BREAST", "IND-PROST",
    "IND-LIVER", "IND-PANC", "IND-GLIO",
]
GEO_CLUSTERS = ["GEO-US1", "GEO-US2", "GEO-EU1"]

# DS/RA candidates (DS-1/DS-2 excluded per doctrine — activation requires DS-3+)
DS_CANDIDATES = ["DS-3", "DS-4", "DS-5"]
RA_CANDIDATES = ["RA-1"]  # default; RA-2 available but optional

# Lexicographic ranking axes (doctrine priority order)
# EDC first (primary objective), then IRC, DPC (invert), LAC (invert), CDC (invert)
LEXICO_AXES = [
    ("EDC", False),   # higher is better
    ("IRC", False),   # higher is better
    ("DPC", True),    # lower (more negative) is better — invert for sort
    ("LAC", True),    # lower (more negative) is better — invert for sort
    ("CDC", True),    # lower (more negative) is better — invert for sort
]

# ── Grid Construction ──────────────────────────────────────────────────────────

def build_candidate_grid(
    params: dict,
    ds_list: Optional[list] = None,
    ra_list: Optional[list] = None,
    mech_list: Optional[list] = None,
    ind_list: Optional[list] = None,
    geo_list: Optional[list] = None,
    exclude_locked_clusters: bool = False,
) -> pd.DataFrame:
    """Generate all feasible (DS, RA, Mech, Ind, Geo) candidate profiles."""
    ds_list = ds_list or DS_CANDIDATES
    ra_list = ra_list or RA_CANDIDATES
    mech_list = mech_list or MECH_CLUSTERS
    ind_list = ind_list or IND_CLUSTERS
    geo_list = geo_list or GEO_CLUSTERS

    if exclude_locked_clusters:
        mech_list = [m for m in mech_list if m not in ("MECH-YB1", "MECH-ABL")]
        ind_list = [i for i in ind_list if i not in ("IND-OV", "IND-HEM")]

    # Filter by DS_RA_MAP allowed combinations
    ds_ra_map = params.get("ds_ra_map", pd.DataFrame())
    allowed_pairs = set()
    if len(ds_ra_map) > 0:
        for _, row in ds_ra_map.iterrows():
            if row.get("Allowed", True):
                allowed_pairs.add((row["DS"], row["RA"]))

    rows = []
    idx = 0
    for ds in ds_list:
        for ra in ra_list:
            if allowed_pairs and (ds, ra) not in allowed_pairs:
                continue
            for mech in mech_list:
                for ind in ind_list:
                    for geo in geo_list:
                        rows.append({
                            "Candidate_ID": f"SYN-CAND-{idx:04d}",
                            "DS": ds, "RA": ra,
                            "MechCluster_ID": mech,
                            "IndicationCluster_ID": ind,
                            "GeoRACluster_ID": geo,
                        })
                        idx += 1
    return pd.DataFrame(rows)


# ── Synthetic Channel Lookup Builder ──────────────────────────────────────────

def build_synthetic_channel_entry(
    cro_data: Optional[dict] = None,
    pharma_data: Optional[dict] = None,
    cro_id: Optional[str] = None,
    target_pharma_ids: Optional[list[str]] = None,
    avl_confirmed_only: bool = False,
) -> dict | None:
    """Build a channel_lookup entry for a synthetic candidate asset.

    This allows the curation optimizer to evaluate "what if we assign CRO-X
    to candidate asset Y?" by constructing the same channel effects that
    build_channel_lookup would produce for a real asset.

    Args:
        cro_data: output of load_cro_master() (dict with 'cro_lookup' key)
        pharma_data: output of load_pharma_master() (dict with 'pharma_lookup', 'avl_lookup')
        cro_id: which CRO to assign (e.g. "CRO-0003")
        target_pharma_ids: which pharma partners (e.g. ["PH-0002", "PH-0003"])
        avl_confirmed_only: only count Confirmed/Likely AVL relationships

    Returns:
        dict with deal_prob_mult and time_shift_months, or None if no CRO assigned
    """
    if cro_data is None or cro_id is None:
        return None

    cro_lookup = cro_data.get("cro_lookup", {})
    if cro_id not in cro_lookup:
        return None

    cro = cro_lookup[cro_id]
    cro_boost = cro.get("partner_boost", 0.0)
    time_shift = cro.get("time_impact", 0.0)

    # AVL pharma boost — replicate real logic from channel.py build_channel_lookup
    avl_boost = 0.0
    matched_pharmas = []

    if pharma_data is not None and target_pharma_ids:
        avl_lookup = pharma_data.get("avl_lookup", {})
        pharma_lookup = pharma_data.get("pharma_lookup", {})

        for ph_id in target_pharma_ids:
            avl_key = (ph_id, cro_id)
            if avl_key in avl_lookup:
                avl_entry = avl_lookup[avl_key]
                status = avl_entry.get("vendor_status", "")

                if avl_confirmed_only and status not in ("Confirmed", "Likely"):
                    continue

                matched_pharmas.append({
                    "pharma_id": ph_id,
                    "pharma_name": pharma_lookup.get(ph_id, {}).get("name", "?"),
                    "vendor_status": status,
                    "alignment_boost": avl_entry.get("alignment_boost", 0.0),
                })
                # Max AVL boost across matched pharmas (same as real logic)
                avl_boost = max(avl_boost, avl_entry.get("alignment_boost", 0.0))

    # Composite: (1 + CRO_boost) * (1 + AVL_boost) — matches real channel math
    deal_prob_mult = (1.0 + cro_boost) * (1.0 + avl_boost)

    return {
        "deal_prob_mult": round(deal_prob_mult, 4),
        "time_shift_months": time_shift,
        "cro_id": cro_id,
        "cro_boost": cro_boost,
        "avl_boost": avl_boost,
        "target_pharma_ids": target_pharma_ids or [],
        "matched_pharmas": matched_pharmas,
        "synthetic": True,
    }


def augment_channel_lookup(
    base_lookup: dict | None,
    candidate_id: str,
    candidate_channel: dict | None,
) -> dict | None:
    """Merge a synthetic candidate's channel entry into an existing lookup.

    Returns a new dict (never mutates base_lookup).
    """
    if candidate_channel is None and base_lookup is None:
        return None
    if candidate_channel is None:
        return base_lookup

    merged = dict(base_lookup) if base_lookup else {}
    merged[candidate_id] = candidate_channel
    return merged


# ── Synthetic Asset Builder ────────────────────────────────────────────────────

def build_synthetic_asset(
    candidate: dict, params: dict, entry_month: int = 3,
    equity_pct: float = 0.10, acq_cash: float = 0,
    passthrough_pct: float = 0.0, deferred_cash: float = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create asset_state + tranches for a synthetic candidate using parameter table lookups."""
    ds, ra = candidate["DS"], candidate["RA"]
    cand_id = candidate["Candidate_ID"]

    # Look up cost/time from tier tables to set realistic tranche budgets
    tier = "Tier-1"  # default
    cost_time_table = params.get("tier1_cost_time", pd.DataFrame())
    row = cost_time_table[(cost_time_table["DS"] == ds) & (cost_time_table["RA"] == ra)]
    if len(row) == 0:
        # Fallback: use DS-only match
        row = cost_time_table[cost_time_table["DS"] == ds]
    if len(row) > 0:
        row = row.iloc[0]
        dev_cost_mode = row["DevCost_Mode"]
        time_mode = row["TimeToExit_Mode"]
    else:
        dev_cost_mode = 5_000_000
        time_mode = 14

    # Set tranche structure: T1 = development, T2 = transaction
    t1_budget = dev_cost_mode * 0.75
    t2_budget = dev_cost_mode * 0.25
    t1_stop = entry_month + int(time_mode * 0.7)
    t2_start = entry_month + int(time_mode * 0.5)
    t2_stop = entry_month + int(time_mode)

    asset_state = pd.DataFrame([{
        "Asset_ID": cand_id,
        "DS_Current": ds,
        "RA_Current": ra,
        "Tier": tier,
        "Entry_Month": entry_month,
        "Equity_to_IP_Pct": equity_pct,
        "AcqCash_to_IP": acq_cash,
        "EarlyPassThrough_Pct": passthrough_pct,
        "EarlyDeferredCash": deferred_cash,
        "MechCluster_ID": candidate["MechCluster_ID"],
        "IndicationCluster_ID": candidate["IndicationCluster_ID"],
        "GeoRACluster_ID": candidate["GeoRACluster_ID"],
    }])

    tranches = pd.DataFrame([
        {"Asset_ID": cand_id, "Tranche_ID": "T1", "Purpose": "Development",
         "Budget": t1_budget, "Start_Month": entry_month, "Stop_Month": t1_stop, "Status": "Planned"},
        {"Asset_ID": cand_id, "Tranche_ID": "T2", "Purpose": "Transaction / BD",
         "Budget": t2_budget, "Start_Month": t2_start, "Stop_Month": t2_stop, "Status": "Planned"},
    ])

    return asset_state, tranches


# ── Candidate Evaluation ──────────────────────────────────────────────────────

def evaluate_candidate(
    base_asset_state: pd.DataFrame, base_tranches: pd.DataFrame,
    candidate: dict, params: dict,
    baseline_summary: Optional[dict] = None,
    baseline_corr: Optional[float] = None,
    n_sims: int = 100, seed: int = 42, duration: int = 36,
    contingency_mult: float = 1.10, annual_overhead: float = 2_640_000.0,
    upfront_threshold: float = 5_000_000.0,
    channel_lookup: dict | None = None,
    candidate_channel: dict | None = None,
) -> dict:
    """Evaluate a single candidate addition against baseline. Returns deltas + hedge score.

    Args:
        channel_lookup: existing channel lookup for portfolio assets
        candidate_channel: optional channel entry for the candidate asset itself
            (built via build_synthetic_channel_entry). If provided, it is merged
            into channel_lookup for the augmented simulation.
    """

    # Build synthetic asset
    cand_state, cand_tr = build_synthetic_asset(candidate, params)

    # Compute baseline if not cached
    if baseline_summary is None or baseline_corr is None:
        base_results = run_monte_carlo(
            asset_state=base_asset_state, tranches=base_tranches, params=params,
            n_sims=n_sims, seed=seed, duration=duration, contingency_mult=contingency_mult,
            upfront_threshold=upfront_threshold, annual_overhead=annual_overhead,
            enable_tranche_kill=True, enable_rollover=True,
            channel_lookup=channel_lookup,
        )
        baseline_summary = summarize_results(base_results)
        baseline_corr = compute_correlation_index(base_asset_state, params.get("correlation", {}))

    # Augmented portfolio
    aug_state = pd.concat([base_asset_state, cand_state], ignore_index=True)
    aug_tr = pd.concat([base_tranches, cand_tr], ignore_index=True)

    # Build augmented channel lookup (merge candidate's channel if provided)
    aug_channel = augment_channel_lookup(
        channel_lookup, candidate["Candidate_ID"], candidate_channel
    )

    aug_results = run_monte_carlo(
        asset_state=aug_state, tranches=aug_tr, params=params,
        n_sims=n_sims, seed=seed, duration=duration, contingency_mult=contingency_mult,
        upfront_threshold=upfront_threshold, annual_overhead=annual_overhead,
        enable_tranche_kill=True, enable_rollover=True,
        channel_lookup=aug_channel,
    )
    aug_summary = summarize_results(aug_results)
    aug_corr = compute_correlation_index(aug_state, params.get("correlation", {}))

    # Compute deltas
    def _d(key):
        a = aug_summary.get(key, float("nan"))
        b = baseline_summary.get(key, float("nan"))
        return a - b if pd.notna(a) and pd.notna(b) else float("nan")

    deltas = {
        "EDC": _d("P_ThreePlus_Exits"),
        "IRC": _d("Median_Annual_IRR"),
        "DPC": _d("P_Exits_LE1"),
        "LAC": _d("Median_First_Dist_Month"),
        "CDC": aug_corr - baseline_corr,
    }

    # Check envelope on augmented portfolio
    # During accumulation, only enforce gates achievable at current portfolio size:
    # - P(>=3 exits) / P(<=1 exit) unreachable with <6 assets
    # - IRR gates produce NaN with too few exits — structural, not a governance failure
    # - Correlation and time gates are always enforced
    n_aug = len(aug_state)

    # Compute weighted time for the augmented portfolio (Critical 5 fix)
    from src.optimization.envelope import compute_weighted_time
    aug_weighted_time = compute_weighted_time(aug_state, params)

    envelope_result = check_envelope(
        aug_summary, params.get("envelope", {}),
        corr_index=aug_corr, weighted_time=aug_weighted_time,
    )
    if isinstance(envelope_result, dict):
        failed = envelope_result.get("failed_gates", [])
        # Exempt density gates when portfolio is too small to meet them
        if n_aug < 6:
            failed = [g for g in failed if g not in ("P_Exits_GE3", "P_Exits_LE1")]
        # Exempt IRR gates when IRR is NaN (too few exits to compute)
        if pd.isna(aug_summary.get("Median_Annual_IRR")):
            failed = [g for g in failed if g not in ("Median_IRR", "IRR_P10")]
        breach = len(failed) > 0
    else:
        breach = False

    return {
        "candidate": candidate,
        "deltas": deltas,
        "aug_summary": aug_summary,
        "aug_corr": aug_corr,
        "breach": breach,
        "cand_state": cand_state,
        "cand_tranches": cand_tr,
    }


# ── Ranking ───────────────────────────────────────────────────────────────────

def rank_candidates(eval_results: list[dict]) -> pd.DataFrame:
    """Rank candidates by lexicographic ordering per doctrine.

    Priority: EDC > IRC > DPC (inverted) > LAC (inverted) > CDC (inverted).
    Infeasible candidates (governance breach) sort to bottom.
    """
    rows = []
    for er in eval_results:
        c = er["candidate"]
        d = er["deltas"]
        rows.append({
            "Candidate_ID": c["Candidate_ID"],
            "DS": c["DS"], "RA": c["RA"],
            "MechCluster_ID": c["MechCluster_ID"],
            "IndicationCluster_ID": c["IndicationCluster_ID"],
            "GeoRACluster_ID": c["GeoRACluster_ID"],
            "EDC": d["EDC"],
            "IRC": d["IRC"],
            "DPC": d["DPC"],
            "LAC": d["LAC"],
            "CDC": d["CDC"],
            "Breach": er["breach"],
        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    # Build sort keys for lexicographic ordering
    # For "lower is better" axes, negate so that sort descending still works
    for col, invert in LEXICO_AXES:
        vals = df[col].values.astype(float)
        # Replace NaN with worst possible value for sorting
        fill_val = float("-inf") if not invert else float("inf")
        clean = np.where(np.isfinite(vals), vals, fill_val)
        df[f"_sort_{col}"] = -clean if invert else clean

    # Governance hard boundary: breach -> all sort keys = -inf
    for col, _ in LEXICO_AXES:
        df.loc[df["Breach"], f"_sort_{col}"] = float("-inf")

    # Compute HedgeScore as EDC for backward compatibility with optimizer stop logic
    df["HedgeScore"] = df["_sort_EDC"]
    df.loc[df["Breach"], "HedgeScore"] = float("-inf")

    # Sort lexicographically: primary axis first, then tiebreakers
    sort_cols = [f"_sort_{col}" for col, _ in LEXICO_AXES]
    df = df.sort_values(sort_cols, ascending=False).reset_index(drop=True)

    # Drop internal sort columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("_sort_")])

    return df


# ── Sequential Optimizer ──────────────────────────────────────────────────────

def optimize_sequential(
    current_asset_state: pd.DataFrame, current_tranches: pd.DataFrame,
    params: dict, n_sims: int = 100, seed: int = 42,
    duration: int = 36, contingency_mult: float = 1.10,
    annual_overhead: float = 2_640_000.0, upfront_threshold: float = 5_000_000.0,
    top_n: int = 3, candidate_grid: Optional[pd.DataFrame] = None,
    exclude_locked_clusters: bool = True,
    channel_lookup: dict | None = None,
    candidate_channel: dict | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Find best next asset. Returns (ranked_df, eval_results)."""

    if candidate_grid is None:
        candidate_grid = build_candidate_grid(params, exclude_locked_clusters=exclude_locked_clusters)

    # Cache baseline
    base_results = run_monte_carlo(
        asset_state=current_asset_state, tranches=current_tranches, params=params,
        n_sims=n_sims, seed=seed, duration=duration, contingency_mult=contingency_mult,
        upfront_threshold=upfront_threshold, annual_overhead=annual_overhead,
        enable_tranche_kill=True, enable_rollover=True,
        channel_lookup=channel_lookup,
    )
    baseline_summary = summarize_results(base_results)
    baseline_corr = compute_correlation_index(current_asset_state, params.get("correlation", {}))

    eval_results = []
    for _, row in candidate_grid.iterrows():
        candidate = row.to_dict()
        er = evaluate_candidate(
            current_asset_state, current_tranches, candidate, params,
            baseline_summary=baseline_summary, baseline_corr=baseline_corr,
            n_sims=n_sims, seed=seed, duration=duration,
            contingency_mult=contingency_mult, annual_overhead=annual_overhead,
            upfront_threshold=upfront_threshold,
            channel_lookup=channel_lookup,
            candidate_channel=candidate_channel,
        )
        eval_results.append(er)

    ranked = rank_candidates(eval_results)
    return ranked.head(top_n), eval_results


# ── Simultaneous Optimizer (Greedy Fill) ──────────────────────────────────────

def optimize_simultaneous(
    anchor_asset_state: pd.DataFrame, anchor_tranches: pd.DataFrame,
    params: dict, n_slots: int = 5, n_sims: int = 100, seed: int = 42,
    duration: int = 36, contingency_mult: float = 1.10,
    annual_overhead: float = 2_640_000.0, upfront_threshold: float = 5_000_000.0,
    edc_stop_threshold: float = 0.01,
    candidate_grid: Optional[pd.DataFrame] = None,
    exclude_locked_clusters: bool = True,
    channel_lookup: dict | None = None,
    candidate_channel: dict | None = None,
) -> dict:
    """Greedy fill N slots. Returns dict with filled assets and convergence log."""

    if candidate_grid is None:
        candidate_grid = build_candidate_grid(params, exclude_locked_clusters=exclude_locked_clusters)

    current_state = anchor_asset_state.copy()
    current_tr = anchor_tranches.copy()
    used_ids = set()
    filled = []
    convergence = []

    for slot in range(n_slots):
        # Filter out already-used candidates
        avail = candidate_grid[~candidate_grid["Candidate_ID"].isin(used_ids)]
        if len(avail) == 0:
            break

        ranked, evals = optimize_sequential(
            current_state, current_tr, params, n_sims=n_sims, seed=seed,
            duration=duration, contingency_mult=contingency_mult,
            annual_overhead=annual_overhead, upfront_threshold=upfront_threshold,
            top_n=1, candidate_grid=avail, exclude_locked_clusters=False,
            channel_lookup=channel_lookup,
            candidate_channel=candidate_channel,
        )

        if len(ranked) == 0 or ranked.iloc[0]["HedgeScore"] == float("-inf"):
            break

        best = ranked.iloc[0]
        best_id = best["Candidate_ID"]

        # EDC stop threshold check — applies to ALL slots including slot 0
        # Negative EDC means adding the asset makes P(>=3 exits) worse
        if best["EDC"] < 0:
            break
        if best["EDC"] < edc_stop_threshold and slot > 0:
            break

        # Find the eval result for the best candidate
        best_er = next(er for er in evals if er["candidate"]["Candidate_ID"] == best_id)

        # Add to portfolio
        current_state = pd.concat([current_state, best_er["cand_state"]], ignore_index=True)
        current_tr = pd.concat([current_tr, best_er["cand_tranches"]], ignore_index=True)
        used_ids.add(best_id)

        # Persist candidate's channel effect into running channel_lookup
        # so subsequent slots see the accumulated channel environment
        if candidate_channel is not None:
            channel_lookup = augment_channel_lookup(channel_lookup, best_id, candidate_channel)

        filled.append({
            "slot": slot + 1,
            "candidate_id": best_id,
            "ds": best["DS"],
            "ra": best["RA"],
            "mech": best["MechCluster_ID"],
            "ind": best["IndicationCluster_ID"],
            "geo": best["GeoRACluster_ID"],
            "hedge_score": best["HedgeScore"],
            "edc_delta": best["EDC"],
        })

        # Log convergence
        conv_results = run_monte_carlo(
            asset_state=current_state, tranches=current_tr, params=params,
            n_sims=n_sims, seed=seed, duration=duration, contingency_mult=contingency_mult,
            upfront_threshold=upfront_threshold, annual_overhead=annual_overhead,
            enable_tranche_kill=True, enable_rollover=True,
            channel_lookup=channel_lookup,
        )
        conv_summary = summarize_results(conv_results)
        conv_corr = compute_correlation_index(current_state, params.get("correlation", {}))
        convergence.append({
            "n_assets": len(current_state),
            "median_moic": conv_summary["Median_MOIC"],
            "median_irr": conv_summary["Median_Annual_IRR"],
            "irr_p10": conv_summary["IRR_P10"],
            "p_ge3_exits": conv_summary["P_ThreePlus_Exits"],
            "p_le1_exit": conv_summary["P_Exits_LE1"],
            "corr_index": conv_corr,
            "first_dist_month": conv_summary["Median_First_Dist_Month"],
        })

    return {
        "filled": filled,
        "convergence": pd.DataFrame(convergence),
        "final_state": current_state,
        "final_tranches": current_tr,
    }


# ── Sourcing Spec Generator ──────────────────────────────────────────────────

def generate_sourcing_spec(
    candidate: dict, params: dict, slot_number: int = 0,
    portfolio_context: Optional[dict] = None,
) -> str:
    """Generate human-readable sourcing target profile."""
    ds, ra = candidate.get("ds", candidate.get("DS", "?")), candidate.get("ra", candidate.get("RA", "?"))
    mech = candidate.get("mech", candidate.get("MechCluster_ID", "?"))
    ind = candidate.get("ind", candidate.get("IndicationCluster_ID", "?"))
    geo = candidate.get("geo", candidate.get("GeoRACluster_ID", "?"))

    # Look up parameter ranges
    cost_time = params.get("tier1_cost_time", pd.DataFrame())
    econ = params.get("tier1_econ", pd.DataFrame())

    ct_row = cost_time[(cost_time["DS"] == ds) & (cost_time["RA"] == ra)]
    ec_row = econ[(econ["DS"] == ds) & (econ["RA"] == ra)]

    lines = [
        f"ASSET SLOT {slot_number} -- SOURCING TARGET PROFILE",
        "=" * 50,
        f"Development State:      {ds}",
        f"Regulatory Access:      {ra}",
        f"Mechanism Cluster:      {mech}",
        f"Indication Cluster:     {ind}",
        f"Geography Cluster:      {geo}",
        "",
    ]

    if len(ct_row) > 0:
        r = ct_row.iloc[0]
        lines.append(f"Dev Cost Range:         ${r['DevCost_Min']/1e6:.1f}M - ${r['DevCost_Mode']/1e6:.1f}M - ${r['DevCost_Max']/1e6:.1f}M")
        lines.append(f"Time to Exit Range:     {r['TimeToExit_Min']:.0f} - {r['TimeToExit_Mode']:.0f} - {r['TimeToExit_Max']:.0f} months")

    if len(ec_row) > 0:
        r = ec_row.iloc[0]
        lines.append(f"Upfront Economics:      ${r['Upfront_Min']/1e6:.0f}M - ${r['Upfront_Mode']/1e6:.0f}M - ${r['Upfront_Max']/1e6:.0f}M")
        lines.append(f"Near Milestones:        ${r['NearMilestones_Min']/1e6:.0f}M - ${r['NearMilestones_Mode']/1e6:.0f}M - ${r['NearMilestones_Max']/1e6:.0f}M")

    lines.append("")
    lines.append("Equity Tolerance:       10% preferred (matches anchor structure)")
    lines.append("Combined Probability:   >= 45% (governance minimum)")

    if portfolio_context:
        lines.append("")
        lines.append(f"Hedge Score:            {candidate.get('hedge_score', '?'):.3f}")
        lines.append(f"EDC Delta:              {candidate.get('edc_delta', '?'):.1%}")

    return "\n".join(lines)


# ── Asset Pool Builder ────────────────────────────────────────────────────────

def build_asset_pool(
    real_asset_state: pd.DataFrame, real_tranches: pd.DataFrame,
    params: dict, candidate_grid: Optional[pd.DataFrame] = None,
    exclude_locked_clusters: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build unified pool of real + simulated assets.
    Returns (pool_state, pool_tranches, pool_index).
    """
    if candidate_grid is None:
        candidate_grid = build_candidate_grid(params, exclude_locked_clusters=exclude_locked_clusters)

    all_states = [real_asset_state.copy()]
    all_tranches = [real_tranches.copy()]
    index_rows = []

    # Real assets
    for _, row in real_asset_state.iterrows():
        index_rows.append({
            "Asset_ID": row["Asset_ID"],
            "Type": "real",
            "DS": row["DS_Current"],
            "RA": row["RA_Current"],
            "MechCluster_ID": row.get("MechCluster_ID", ""),
            "IndicationCluster_ID": row.get("IndicationCluster_ID", ""),
            "GeoRACluster_ID": row.get("GeoRACluster_ID", ""),
        })

    # Synthetic assets
    for _, row in candidate_grid.iterrows():
        candidate = row.to_dict()
        s, t = build_synthetic_asset(candidate, params)
        all_states.append(s)
        all_tranches.append(t)
        index_rows.append({
            "Asset_ID": candidate["Candidate_ID"],
            "Type": "simulated",
            "DS": candidate["DS"],
            "RA": candidate["RA"],
            "MechCluster_ID": candidate["MechCluster_ID"],
            "IndicationCluster_ID": candidate["IndicationCluster_ID"],
            "GeoRACluster_ID": candidate["GeoRACluster_ID"],
        })

    pool_state = pd.concat(all_states, ignore_index=True)
    pool_tranches = pd.concat(all_tranches, ignore_index=True)
    pool_index = pd.DataFrame(index_rows)

    return pool_state, pool_tranches, pool_index


# ── Run Selected Portfolio ────────────────────────────────────────────────────

def run_selected_portfolio(
    pool_state: pd.DataFrame, pool_tranches: pd.DataFrame,
    selected_ids: list[str], params: dict,
    n_sims: int = 500, seed: int = 42, duration: int = 36,
    contingency_mult: float = 1.10, annual_overhead: float = 2_640_000.0,
    upfront_threshold: float = 5_000_000.0,
    channel_lookup: dict | None = None,
) -> dict:
    """Run MC on a user-selected subset of the asset pool."""
    sel_state = pool_state[pool_state["Asset_ID"].isin(selected_ids)].reset_index(drop=True)
    sel_tr = pool_tranches[pool_tranches["Asset_ID"].isin(selected_ids)].reset_index(drop=True)

    if len(sel_state) == 0:
        return {"error": "No matching assets found"}

    results = run_monte_carlo(
        asset_state=sel_state, tranches=sel_tr, params=params,
        n_sims=n_sims, seed=seed, duration=duration, contingency_mult=contingency_mult,
        upfront_threshold=upfront_threshold, annual_overhead=annual_overhead,
        enable_tranche_kill=True, enable_rollover=True,
        channel_lookup=channel_lookup,
    )
    summary = summarize_results(results)
    corr = compute_correlation_index(sel_state, params.get("correlation", {}))

    return {
        "summary": summary,
        "results_df": results,
        "corr_index": corr,
        "n_assets": len(sel_state),
        "asset_ids": sel_state["Asset_ID"].tolist(),
        "asset_state": sel_state,
        "tranches": sel_tr,
    }


# ── Full Pipeline Entry Point ─────────────────────────────────────────────────

def run_full_curation_pipeline(
    anchor_asset_state: pd.DataFrame, anchor_tranches: pd.DataFrame,
    params: dict, n_target_assets: int = 7, n_sims: int = 100, seed: int = 42,
    duration: int = 36, contingency_mult: float = 1.10,
    annual_overhead: float = 2_640_000.0, upfront_threshold: float = 5_000_000.0,
    edc_stop_threshold: float = 0.01,
    channel_lookup: dict | None = None,
    candidate_channel: dict | None = None,
) -> dict:
    """End-to-end curation: build grid, greedy fill, generate sourcing specs."""
    n_anchor = len(anchor_asset_state)
    n_slots = n_target_assets - n_anchor

    # Build grid (exclude locked clusters for diversification)
    grid = build_candidate_grid(params, exclude_locked_clusters=True)

    # Greedy fill
    result = optimize_simultaneous(
        anchor_asset_state, anchor_tranches, params,
        n_slots=n_slots, n_sims=n_sims, seed=seed,
        duration=duration, contingency_mult=contingency_mult,
        annual_overhead=annual_overhead, upfront_threshold=upfront_threshold,
        edc_stop_threshold=edc_stop_threshold, candidate_grid=grid,
        exclude_locked_clusters=False,
        channel_lookup=channel_lookup,
        candidate_channel=candidate_channel,
    )

    # Generate sourcing specs for each filled slot
    specs = []
    for f in result["filled"]:
        spec = generate_sourcing_spec(f, params, slot_number=n_anchor + f["slot"],
                                       portfolio_context=result)
        specs.append(spec)

    result["sourcing_specs"] = specs
    result["grid_size"] = len(grid)

    return result
