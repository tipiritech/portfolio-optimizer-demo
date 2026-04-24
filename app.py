#!/usr/bin/env python3
"""
Discovery Portfolio Simulation — Streamlit Dashboard v1.0
Full governance engine with:
  - Info tooltips on every metric and header
  - User guide (sidebar)
  - CRO/Pharma channel display
  - Sandbox scenario analysis UI
  - Pre-flight checks, envelope gates, distributions, monitoring
"""

import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


def _add_median_line(fig, x_val, label, color="#E53E3E", y_max=None):
    """Add a dashed median line with annotation (Plotly 6.x compatible)."""
    fig.add_shape(type="line", x0=x_val, x1=x_val,
        y0=0, y1=y_max if y_max else 1, yref="paper" if not y_max else "y",
        line=dict(color=color, dash="dash", width=1.5))
    fig.add_annotation(x=x_val, y=1, yref="paper",
        text=label, showarrow=False, yshift=10,
        font=dict(color=color, size=10))


from src.data.loader import load_params, load_state
from src.simulation.monte_carlo import run_monte_carlo, summarize_results
from src.optimization.correlation import compute_correlation_index
from src.optimization.envelope import (
    check_envelope, portfolio_quality_signal, validate_params,
    check_activation_requirements, check_capital_concentration,
    check_combined_probability, compute_weighted_time,
)
from src.governance.monitoring import check_conditional_status, classify_month30_winddown
from src.auth import (
    init_auth, login_gate, logout, get_current_user,
    has_permission, require_permission, user_sidebar,
    log_action, confidentiality_footer, compute_state_hash,
)

# ── Tooltip dictionary ────────────────────────────────────────────────────

TIPS = {
    "assets": (
        "Number of active drug assets in the portfolio. "
        "Diversification across 6-10 assets is required to hit "
        "P(>=3 exits) >= 60%. Fewer assets = concentrated risk."
    ),
    "planned_capital": (
        "Total budgeted capital across all tranches (development + BD). "
        "This is the denominator in MOIC — every dollar deployed "
        "must earn its return within the 36-month window."
    ),
    "tranches": (
        "Number of discrete capital deployment stages across all assets. "
        "Tranche structure controls when capital is at risk — "
        "tranche kill stops spending on failed assets."
    ),
    "corr_index": (
        "Pairwise correlation index (0-1) based on shared mechanism, indication, "
        "geography, and CRO cluster factors. "
        "High correlation means assets fail together. Threshold: <= 0.90."
    ),
    "weighted_time": (
        "Capital-weighted average time-to-exit across all assets (months from entry). "
        "Liquidity speed protects IRR. "
        "Target: <= 24 months. Capital pause trigger: 30 months."
    ),
    "median_moic": (
        "Median Multiple on Invested Capital across all simulations. "
        "MOIC is the primary return metric — "
        "it answers 'for every dollar in, how many come back?'"
    ),
    "moic_p10": (
        "10th percentile MOIC — the return in a bad-but-not-worst scenario. "
        "Investors care about downside protection. "
        "A positive P10 means even poor outcomes return some capital."
    ),
    "median_irr": (
        "Median annualized Internal Rate of Return. "
        "IRR captures the TIME value of returns — "
        "a 3x in 18 months is much better than 3x in 36 months. Target: >= 25%."
    ),
    "irr_p10": (
        "10th percentile annualized IRR — downside return scenario. "
        "Governance requires IRR P10 >= 0% — "
        "even in a bad case, investors should not lose money on a time-adjusted basis."
    ),
    "p_zero_exits": (
        "Probability of zero successful exits across the portfolio. "
        "Zero exits = total capital loss. "
        "This is the catastrophic scenario investors most fear."
    ),
    "p_one_exit": (
        "Probability of exactly one successful exit. "
        "One exit may not return enough capital to cover "
        "overhead + failed assets — borderline survival scenario."
    ),
    "p_two_exits": (
        "Probability of exactly two successful exits. "
        "Two exits typically covers capital + modest return — "
        "acceptable but below governance target."
    ),
    "p_three_plus": (
        "Probability of three or more successful exits. "
        "Hard governance gate: P(>=3 exits) >= 60% required. "
        "Three exits generally delivers strong MOIC and IRR."
    ),
    "preflight": (
        "Automated validation checks that run before any simulation. "
        "These catch data errors, constraint violations, "
        "and structural problems before you waste compute time."
    ),
    "envelope": (
        "Governance constraint validation against simulation output. "
        "All gates must pass for the portfolio to be approved. "
        "These are the quantitative guardrails that protect investors."
    ),
    "winddown": (
        "Month-30 wind-down classification for each asset. "
        "Assets projecting beyond 30 months trigger "
        "capital pause and Board review — speed protects IRR."
    ),
    "conditional": (
        "Assets flagged for conditional capital status (IRR < 22% or "
        "liquidity approaching 30 months). "
        "Conditional status limits spending to preservation "
        "and requires a milestone within 60-90 days."
    ),
    "channel": (
        "CRO/Pharma channel effects — structural advantages from CRO selection "
        "and pharma vendor alignment. "
        "Channel engagement compresses timelines and boosts "
        "deal probability, directly improving portfolio economics."
    ),
    "sandbox": (
        "Exploratory scenario analysis — test what-if adjustments without "
        "modifying locked governance parameters. "
        "Results are clearly marked NON-GOVERNANCE. "
        "Lets you pressure-test decisions before committing."
    ),
    "n_sims": (
        "Number of Monte Carlo simulations to run. "
        "Grades: <500 Quick Check (fast, noisy) | "
        "500-1,999 Exploratory (reasonable estimates) | "
        "2,000-4,999 Standard (stable statistics) | "
        "5,000-9,999 Governance-Grade (investor reporting) | "
        "10,000+ High-Precision (convergence verification). "
        "Governance-grade runs at 5,000+ sims converge within +/-0.02x MOIC."
    ),
    "corr_stress_toggle": (
        "Add +0.10 to the base correlation floor (0.25 to 0.35). "
        "Simulates a scenario where assets are more correlated than expected."
    ),
    "tight_regime_toggle": (
        "Force all simulations into the Tight market regime "
        "(Deal x 0.85, Upfront x 0.80, Time x 1.15). "
        "Simulates worst-case market conditions."
    ),
    "contingency": (
        "Multiplier applied to all tranche budgets (e.g. 1.10 = 10% buffer). "
        "Cost overruns are common in drug development — "
        "the contingency ensures capital adequacy."
    ),
    "overhead": (
        "Annual operating overhead for the vehicle (legal, admin, mgmt fees). "
        "Subtracted monthly from portfolio cashflows. "
        "Every dollar of overhead must be earned back before investors see profit."
    ),
    "tranche_kill": (
        "When enabled, stops capital deployment for assets that fail. "
        "Continuing to spend on failed assets destroys returns. "
        "This is a core capital discipline mechanism."
    ),
    "rollover": (
        "Estimate residual value for assets that succeeded technically "
        "but didn't exit within 36 months. Uses Low/Base midpoint x 50% haircut. "
        "Captures optionality that pure binary models miss."
    ),
}

USER_GUIDE = """
### How This Engine Works

**Discovery Biotech** curates high-value novel drug assets and develops them 
for delivery to biopharma customers. Private investors invest into a portfolio 
vehicle (not individual assets). This engine models the probabilistic outcomes 
of that portfolio.

---

#### Key Concepts

**Monte Carlo Simulation** — Runs thousands of random scenarios sampling 
technical success probability, deal probability, exit timing, and economics 
from triangular distributions. Each simulation produces a complete portfolio 
outcome (MOIC, IRR, exit count).

**Governance Envelope** — Hard quantitative constraints:
- Median IRR >= 25%
- IRR P10 >= 0%
- P(>=3 exits) >= 60%
- P(<=1 exit) <= 15%
- Weighted time <= 24 months
- Capital concentration <= 20% per asset
- Combined probability >= 45% per asset
- 36-month hard stop (non-overridable)

**Channel Effects** — CRO selection and pharma vendor alignment create 
structural advantages: higher deal probability and compressed timelines.

**Tranche Kill** — When an asset fails, remaining capital deployment for 
that asset is cancelled. This preserves capital for surviving assets.

---

#### Running a Simulation

1. Set the number of simulations (1,000 for quick runs, 5,000+ for governance)
2. Choose stress options (correlation stress, tight regime) if desired
3. Click **Run Monte Carlo**
4. Review the Quality Signal (GREEN / YELLOW / RED)
5. Check the Envelope Gates — all must PASS for governance approval
6. Use the **Sandbox** tab to explore what-if scenarios

---

#### Reading the Results

**MOIC** (Multiple on Invested Capital) — How many dollars come back for 
each dollar invested. 2.0x means investors double their money.

**IRR** (Internal Rate of Return) — Annualized return accounting for the 
timing of cash flows. Higher is better; the speed of returns matters.

**Exit Count** — Number of assets that successfully monetize (upfront >= $5M). 
The portfolio needs >= 3 exits to hit governance targets.

**Quality Signal:**
- GREEN: Strong profile — passes all governance gates
- YELLOW: Borderline — some gates at risk
- RED: Weak — governance gates failing

---

#### Parameters Are Locked

All parameters are frozen at vehicle activation. No drift is permitted. 
The sandbox allows exploratory what-if analysis, but sandbox results are 
clearly marked NON-GOVERNANCE and cannot be used for investor reporting.
"""

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DEFAULT_PARAM_FILE = DATA_DIR / "Discovery_Params_v1.xlsx"
DEFAULT_STATE_FILE = DATA_DIR / "Demo_Portfolio_10_Assets.xlsx"
LOGO_PATH = None
for _logo_candidate in [
    DATA_DIR / "discovery_logo.png",
    APP_DIR / "discovery_logo.png",
    APP_DIR / "src" / "discovery_logo.png",
]:
    if _logo_candidate.exists():
        LOGO_PATH = str(_logo_candidate)
        break


def _pdf_bytes(print_fn, *args, **kwargs) -> bytes:
    """Call a printer function into a temp file and return the bytes."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        kwargs["output_path"] = tmp.name
        kwargs.setdefault("logo_path", LOGO_PATH)
        print_fn(*args, **kwargs)
        tmp.seek(0)
        return Path(tmp.name).read_bytes()

st.set_page_config(page_title="Discovery Portfolio Optimizer v1.0", layout="wide")

# ─── DEMO MODE BANNER ───
DEMO_MODE = True
if DEMO_MODE:
    st.warning("🧪 **Demonstration build — synthetic data only.** Channel/CRO integration disabled in demo mode.")

# ── Global CSS: table text wrap + sticky tabs + styling ──
st.markdown("""
<style>
    /* Reduce top margin */
    .block-container { padding-top: 3.0rem !important; }

    /* Wrap text in dataframe cells */
    .stDataFrame div[data-testid="stDataFrameResizable"] td {
        white-space: normal !important;
        word-wrap: break-word !important;
        max-width: 300px;
    }
    .stDataFrame div[data-testid="stDataFrameResizable"] th {
        white-space: normal !important;
        word-wrap: break-word !important;
    }
    /* Ensure all table cells wrap */
    [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
    /* DataFrame container */
    .dataframe td { white-space: normal !important; }
    .dataframe th { white-space: normal !important; }

    /* Ensure sidebar scrolls */
    [data-testid="stSidebar"] > div:first-child {
        overflow-y: auto !important;
        max-height: 100vh;
    }
    section[data-testid="stSidebar"] {
        overflow-y: auto !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Authentication ──
# Set SKIP_AUTH=True to bypass login during development/demos
SKIP_AUTH = True

if SKIP_AUTH:
    # Force auth on EVERY render — prevents any logout/state loss
    st.session_state.authenticated = True
    st.session_state.user_email = "admin@discoverybiotech.com"
    st.session_state.user_name = "Admin"
    st.session_state.user_role = "board"
    if "login_time" not in st.session_state:
        st.session_state.login_time = None
else:
    init_auth(DATA_DIR)
    if not login_gate(DATA_DIR):
        st.stop()

# ── Header with logo ──
_header_cols = st.columns([0.04, 0.96])
with _header_cols[0]:
    if LOGO_PATH and Path(LOGO_PATH).exists():
        st.markdown('<div style="margin-top:2px;">', unsafe_allow_html=True)
        st.image(LOGO_PATH, width=44)
        st.markdown('</div>', unsafe_allow_html=True)
with _header_cols[1]:
    st.markdown(
        '<div>'
        '<div style="font-size:22pt;font-weight:700;color:#2D3748;line-height:1.1;">'
        'Discovery Portfolio Optimizer v1.0</div>'
        '<div style="font-size:12pt;color:#718096;margin-top:1px;line-height:1.2;">'
        'Portfolio governance &amp; optimization engine — drug development modeling, curation &amp; scenario analysis</div>'
        '</div>',
        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    if not SKIP_AUTH:
        user_sidebar(DATA_DIR)
        st.divider()
    else:
        pass  # Demo mode badge at bottom

    run_button = st.button("▶ Run Monte Carlo", type="primary", use_container_width=True)
    st.header("Run Controls")
    n_sims = st.number_input("Simulations", 100, 50000, 1000, 100, help=TIPS["n_sims"])

    # Simulation grade label
    if n_sims < 500:
        st.caption("⚡ **Quick Check** — fast but noisy, not for decisions")
    elif n_sims < 2000:
        st.caption("🔍 **Exploratory** — reasonable estimates, not governance-grade")
    elif n_sims < 5000:
        st.caption("📊 **Standard** — stable statistics for most analysis")
    elif n_sims < 10000:
        st.caption("✅ **Governance-Grade** — suitable for investor reporting")
    else:
        st.caption("🔬 **High-Precision** — sanity check / convergence verification")

    use_corr_stress = st.checkbox("Correlation Stress (+0.10)", help=TIPS["corr_stress_toggle"])
    use_tight_only = st.checkbox("Force Tight Regime", help=TIPS["tight_regime_toggle"])
    contingency = st.slider("Contingency Multiplier", 1.00, 1.20, 1.10, 0.01, help=TIPS["contingency"])
    overhead = st.number_input("Annual Overhead ($)", 0, 10_000_000, 2_640_000, 100_000, help=TIPS["overhead"])
    enable_tranche_kill = st.checkbox("Tranche Kill", value=True, help=TIPS["tranche_kill"])
    enable_rollover = st.checkbox("Rollover Valuation", value=True, help=TIPS["rollover"])

    # Demo mode badge at bottom
    if SKIP_AUTH:
        st.divider()
        st.markdown(
            '<div style="background:#EDF2F7;padding:8px 12px;border-radius:6px;">'
            '<div style="font-weight:600;font-size:12px;color:#2D3748;">Admin (Demo Mode)</div>'
            '<div style="font-size:10px;color:#718096;">Auth bypassed — SKIP_AUTH=True</div>'
            '</div>', unsafe_allow_html=True)

# ── Default paths (configurable in Settings tab) ──
if "param_path" not in st.session_state:
    st.session_state.param_path = str(DEFAULT_PARAM_FILE)
if "state_path" not in st.session_state:
    st.session_state.state_path = str(DEFAULT_STATE_FILE)
if "enable_channel" not in st.session_state:
    st.session_state.enable_channel = False
if "cro_path" not in st.session_state:
    st.session_state.cro_path = None  # DEMO
if "pharma_path" not in st.session_state:
    st.session_state.pharma_path = None  # DEMO
if "annual_overhead" not in st.session_state:
    st.session_state.annual_overhead = 2_640_000

param_path = st.session_state.param_path
state_path = st.session_state.state_path
enable_channel = st.session_state.enable_channel
cro_path_str = st.session_state.cro_path
pharma_path_str = st.session_state.pharma_path
overhead = st.session_state.annual_overhead

# ══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════

try:
    params = load_params(param_path)
    state = load_state(state_path)
except Exception as e:
    st.error(f"Could not load workbooks: {e}")
    st.stop()

asset_state = state["asset_state"]
tranches = state["tranches"]
roster = state["roster"]
envelope = params["envelope"]

channel_lookup = None
if enable_channel:
    try:
        from src.channel import load_cro_master, load_pharma_master, build_channel_lookup
        cro_data = load_cro_master(cro_path_str)
        pharma_data = load_pharma_master(pharma_path_str)
        channel_lookup = build_channel_lookup(asset_state, cro_data, pharma_data, avl_confirmed_only=False)
    except Exception as e:
        st.warning(f"Channel loading failed: {e}")

# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════

tab_main, tab_assets, tab_explorer, tab_sandbox, tab_settings = st.tabs([
    "📊 Portfolio",
    "🧬 Asset Profiles",
    "🎛️ Interactive Portfolio",
    "🧪 Sandbox",
    "⚙️ Settings",
])

with tab_main:
    # ── Pre-Flight ──
    st.subheader("Pre-Flight Checks", help=TIPS["preflight"])
    pf1, pf2, pf3, pf4 = st.columns(4)

    param_issues = []
    try:
        param_issues = validate_params(params)
    except Exception:
        pass
    with pf1:
        if not param_issues:
            st.success("Parameters OK")
        else:
            st.error(f"Params: {len(param_issues)} issue(s)")

    activation = check_activation_requirements(asset_state, params["ds_ra_map"])
    with pf2:
        if activation["activated"]:
            st.success("Activation OK")
        else:
            st.warning(f"Activation: {len(activation['issues'])} issue(s)")

    conc = check_capital_concentration(tranches)
    with pf3:
        if conc["breaches"]:
            st.error(f"Concentration: {len(conc['breaches'])} breach(es)")
        elif conc["warnings"]:
            st.warning(f"Concentration: {len(conc['warnings'])} warning(s)")
        else:
            st.success("Concentration OK")

    comb_prob = check_combined_probability(asset_state, params)
    with pf4:
        if comb_prob["all_pass"]:
            st.success("Combined Prob OK")
        else:
            st.warning(f"Combined Prob: {len(comb_prob['failures'])} below 45%")

    with st.expander("Pre-Flight Details"):
        if param_issues:
            st.markdown("**Parameter Issues:**")
            for issue in param_issues:
                st.write(f"- {issue}")
        if not activation["activated"]:
            st.markdown("**Activation Issues:**")
            for issue in activation["issues"]:
                st.write(f"- {issue}")
        if conc["breaches"] or conc["warnings"]:
            st.markdown("**Capital Concentration:**")
            st.dataframe(pd.DataFrame([
                {"Asset_ID": k, "Concentration": f"{v:.1%}",
                 "Status": "BREACH" if v > 0.20 else ("WARNING" if v > 0.15 else "OK")}
                for k, v in conc["concentrations"].items()
            ]), use_container_width=True, hide_index=True)
        if not comb_prob["all_pass"]:
            st.markdown("**Combined Probability Failures:**")
            st.dataframe(pd.DataFrame(comb_prob["failures"]), use_container_width=True, hide_index=True)
        from src.printer import print_preflight
        st.download_button("📄 Print Pre-Flight", key="dl_preflight",
            data=_pdf_bytes(print_preflight, param_issues, activation, conc, comb_prob),
            file_name="preflight.pdf", mime="application/pdf")

    # ── Portfolio Overview ──
    st.subheader("Portfolio Overview")
    corr_config = params.get("correlation", {})
    corr_index = compute_correlation_index(asset_state, corr_config)
    weighted_time = compute_weighted_time(asset_state, params)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Assets", len(roster), help=TIPS["assets"])
    with c2: st.metric("Planned Capital", f"${tranches['Budget'].sum():,.0f}", help=TIPS["planned_capital"])
    with c3: st.metric("Tranches", len(tranches), help=TIPS["tranches"])
    with c4: st.metric("Corr Index", f"{corr_index:.3f}", help=TIPS["corr_index"])
    with c5: st.metric("Weighted Time", f"{weighted_time:.1f}mo", help=TIPS["weighted_time"])

    left, right = st.columns(2)
    with left:
        with st.expander("Asset Roster", expanded=True):
            display_cols = [c for c in roster.columns if c != "Notes"]
            st.dataframe(roster[display_cols], use_container_width=True, hide_index=True)
        with st.expander("Asset State"):
            state_cols = ["Asset_ID", "Tier", "DS_Current", "RA_Current", "Entry_Month",
                          "Equity_to_IP_Pct", "CRO_ID", "Target_Pharma_IDs", "Engagement_Complete"]
            available = [c for c in state_cols if c in asset_state.columns]
            st.dataframe(asset_state[available], use_container_width=True, hide_index=True)
    with right:
        with st.expander("Capital Tranches", expanded=True):
            tranche_cols = ["Asset_ID", "Tranche_ID", "Purpose", "Budget", "Start_Month", "Stop_Month", "Status"]
            available_t = [c for c in tranche_cols if c in tranches.columns]
            st.dataframe(tranches[available_t], use_container_width=True, hide_index=True)
        with st.expander("Key Parameters"):
            st.markdown("**Market Regimes**")
            st.dataframe(params["regime"], use_container_width=True, hide_index=True)

    from src.printer import print_portfolio_overview
    st.download_button("📄 Print Portfolio Overview", key="dl_overview",
        data=_pdf_bytes(print_portfolio_overview, roster, asset_state, tranches, params),
        file_name="portfolio_overview.pdf", mime="application/pdf")

    if channel_lookup:
        with st.expander("CRO / Pharma Channel Effects", expanded=False):
            st.caption(TIPS["channel"])
            from src.channel import summarize_channel
            ch_summary = summarize_channel(channel_lookup)
            st.dataframe(ch_summary, use_container_width=True, hide_index=True)
            from src.printer import print_channel_summary
            st.download_button("📄 Print Channel Summary", key="dl_channel",
                data=_pdf_bytes(print_channel_summary, channel_lookup),
                file_name="channel_summary.pdf", mime="application/pdf")

    # ── Simulation ──
    if run_button:
        upfront_threshold = envelope.get("Min_Upfront_Threshold", 5_000_000.0)
        with st.spinner(f"Running {n_sims:,} simulations..."):
            results_df = run_monte_carlo(
                asset_state=asset_state, tranches=tranches, params=params,
                n_sims=int(n_sims), duration=36, contingency_mult=contingency,
                use_corr_stress=use_corr_stress, use_tight_only=use_tight_only,
                upfront_threshold=upfront_threshold, annual_overhead=float(overhead),
                enable_tranche_kill=enable_tranche_kill, enable_rollover=enable_rollover,
                channel_lookup=channel_lookup,
            )
            summary = summarize_results(results_df)

        log_action(DATA_DIR, st.session_state.get("user_email", "unknown"), "run_mc", {
            "n_sims": int(n_sims), "corr_stress": use_corr_stress,
            "tight_only": use_tight_only, "channel": channel_lookup is not None,
            "median_moic": round(summary["Median_MOIC"], 3),
        }, state_hash=compute_state_hash(asset_state, tranches))

        signal = portfolio_quality_signal(summary, envelope)
        st.markdown(
            f'<div style="background-color:{signal["bg"]};border-left:10px solid {signal["border"]};'
            f'padding:18px;border-radius:8px;margin:12px 0 18px 0;">'
            f'<div style="font-size:28px;font-weight:700;color:{signal["text"]};">'
            f'Portfolio Quality: {signal["label"]}</div>'
            f'<div style="font-size:18px;color:{signal["text"]};margin-top:6px;">'
            f'{signal["message"]}</div></div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Median MOIC", f"{summary['Median_MOIC']:.2f}x", help=TIPS["median_moic"])
        with m2: st.metric("MOIC P10", f"{summary['MOIC_P10']:.2f}x", help=TIPS["moic_p10"])
        with m3:
            v = summary["Median_Annual_IRR"]
            st.metric("Median IRR", f"{v:.1%}" if pd.notna(v) else "N/A", help=TIPS["median_irr"])
        with m4:
            v = summary["IRR_P10"]
            st.metric("IRR P10", f"{v:.1%}" if pd.notna(v) else "N/A", help=TIPS["irr_p10"])

        m5, m6, m7, m8 = st.columns(4)
        with m5: st.metric("P(0 exits)", f"{summary['P_Zero_Exits']:.1%}", help=TIPS["p_zero_exits"])
        with m6: st.metric("P(1 exit)", f"{summary['P_One_Exit']:.1%}", help=TIPS["p_one_exit"])
        with m7: st.metric("P(2 exits)", f"{summary['P_Two_Exits']:.1%}", help=TIPS["p_two_exits"])
        with m8: st.metric("P(3+ exits)", f"{summary['P_ThreePlus_Exits']:.1%}", help=TIPS["p_three_plus"])

        envelope_result = check_envelope(summary, envelope, corr_index,
            weighted_time=weighted_time, concentration_issues=conc["breaches"])
        with st.expander("Envelope Gate Checks", expanded=True):
            st.caption(TIPS["envelope"])
            gate_rows = []
            for name, check in envelope_result["checks"].items():
                gate_rows.append({
                    "Gate": name,
                    "Status": "✅ PASS" if check["pass"] else "❌ FAIL",
                    "Actual": f"{check['actual']:.4f}" if isinstance(check["actual"], float) else str(check["actual"]),
                    "Threshold": str(check["threshold"]),
                })
            st.dataframe(pd.DataFrame(gate_rows), use_container_width=True, hide_index=True)
            from src.printer import print_envelope_gates
            st.download_button("📄 Print Envelope Gates", key="dl_envelope",
                data=_pdf_bytes(print_envelope_gates, envelope_result),
                file_name="envelope_gates.pdf", mime="application/pdf")

        from src.printer import print_mc_results
        st.download_button("📄 Print MC Results", key="dl_mc",
            data=_pdf_bytes(print_mc_results, summary, envelope_result, n_sims=int(n_sims)),
            file_name="mc_results.pdf", mime="application/pdf")

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Exit Count Distribution")
            exit_counts = results_df["Num_Exits"].value_counts().sort_index()
            fig_exit = go.Figure(go.Bar(
                x=exit_counts.index, y=exit_counts.values,
                marker_color="#3182CE", hovertemplate="Exits: %{x}<br>Simulations: %{y:,}<extra></extra>",
            ))
            fig_exit.update_layout(
                xaxis_title="Exit Count", yaxis_title="Simulations",
                height=350, margin=dict(l=40, r=20, t=20, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_exit, use_container_width=True)

        with col_r:
            st.subheader("MOIC Distribution")
            valid_moic = results_df["MOIC"].dropna()
            if len(valid_moic):
                fig_moic = go.Figure(go.Histogram(
                    x=valid_moic, nbinsx=25,
                    marker_color="#3182CE", opacity=0.85,
                    hovertemplate="MOIC: %{x:.1f}x<br>Count: %{y}<extra></extra>",
                ))
                _add_median_line(fig_moic, valid_moic.median(), f"Median: {valid_moic.median():.2f}x")
                fig_moic.update_layout(
                    xaxis_title="MOIC (x)", yaxis_title="Simulations",
                    height=350, margin=dict(l=40, r=20, t=20, b=40),
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_moic, use_container_width=True)

        col_l2, col_r2 = st.columns(2)
        with col_l2:
            st.subheader("Annual IRR Distribution")
            valid_irr = results_df["Annual_IRR"].dropna()
            if len(valid_irr):
                fig_irr = go.Figure(go.Histogram(
                    x=valid_irr * 100, nbinsx=25,
                    marker_color="#38A169", opacity=0.85,
                    hovertemplate="IRR: %{x:.0f}%<br>Count: %{y}<extra></extra>",
                ))
                _add_median_line(fig_irr, valid_irr.median() * 100, f"Median: {valid_irr.median():.0%}")
                fig_irr.update_layout(
                    xaxis_title="Annual IRR (%)", yaxis_title="Simulations",
                    height=350, margin=dict(l=40, r=20, t=20, b=40),
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_irr, use_container_width=True)

        with col_r2:
            st.subheader("First Distribution Month")
            valid_dist = results_df["First_Dist_Month"].dropna()
            if len(valid_dist):
                dist_counts = valid_dist.astype(int).value_counts().sort_index()
                med_dist = float(valid_dist.median())
                fig_dist = go.Figure(go.Bar(
                    x=dist_counts.index, y=dist_counts.values,
                    marker_color="#805AD5", hovertemplate="Month %{x}<br>Simulations: %{y:,}<extra></extra>",
                ))
                fig_dist.add_shape(type="line", x0=med_dist, x1=med_dist,
                    y0=0, y1=dist_counts.values.max(), line=dict(color="#E53E3E", dash="dash", width=2))
                fig_dist.add_annotation(x=med_dist, y=dist_counts.values.max(),
                    text=f"Median: Mo {med_dist:.0f}", showarrow=False,
                    yshift=15, font=dict(color="#E53E3E", size=11))
                fig_dist.update_layout(
                    xaxis_title="Month", yaxis_title="Simulations",
                    height=350, margin=dict(l=40, r=20, t=30, b=40),
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_dist, use_container_width=True)

        with st.expander("Month-30 Wind-Down Classification"):
            st.caption(TIPS["winddown"])
            try:
                classifications = classify_month30_winddown(asset_state, params)
                st.dataframe(pd.DataFrame(classifications), use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"Wind-down classification error: {e}")

        with st.expander("Conditional Capital Status"):
            st.caption(TIPS["conditional"])
            try:
                cond_status = check_conditional_status(asset_state, params)
                if cond_status["has_conditional"]:
                    st.dataframe(pd.DataFrame(cond_status["flagged_assets"]), use_container_width=True, hide_index=True)
                else:
                    st.success("No assets in conditional status.")
            except Exception as e:
                st.warning(f"Conditional status error: {e}")

        with st.expander("Sample Simulation Results"):
            st.dataframe(results_df.head(50), use_container_width=True, hide_index=True)
    else:
        st.info("Click **▶ Run Monte Carlo** in the sidebar to generate results.")

    # ══════════════════════════════════════════════════════════════════════
    # ADMISSION REVIEW & SIGN-OFF (always visible)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📋 Asset Admission — Review & Sign-Off")

    if "admission_queue" not in st.session_state:
        st.session_state.admission_queue = []
    live_queue = st.session_state.admission_queue

    pending_items = [a for a in live_queue if a.get("_admission_status", "Pending") == "Pending"]
    approved_items = [a for a in live_queue if a.get("_admission_status") == "Approved"]
    rejected_items = [a for a in live_queue if a.get("_admission_status") == "Rejected"]

    # Status summary
    aq1, aq2, aq3, aq4 = st.columns(4)
    with aq1: st.metric("Pending Review", len(pending_items))
    with aq2: st.metric("Approved", len(approved_items))
    with aq3: st.metric("Rejected", len(rejected_items))
    with aq4: st.metric("Total Submissions", len(live_queue))

    if not live_queue:
        st.info("No asset submissions yet. Use the **Sandbox** tab to build and analyze candidates, "
                "then click **📋 Submit** to send them here for formal review.")
    else:
        # ── Pending Review ──
        if pending_items:
            st.markdown("### ⏳ Pending Review")
            for qi, qa in enumerate(live_queue):
                if qa.get("_admission_status", "Pending") != "Pending":
                    continue

                with st.expander(f"⏳ {qa['Asset_ID']} — {qa['DS_Current']}/{qa['RA_Current']} "
                                 f"{qa.get('MechCluster_ID', '')} / {qa.get('IndicationCluster_ID', '')}",
                                 expanded=True):

                    # Asset details
                    st.markdown("**Asset Summary**")
                    rd1, rd2, rd3 = st.columns(3)
                    with rd1:
                        st.markdown(f"**Dev Stage:** {qa['DS_Current']}  \n"
                                    f"**Reg Access:** {qa['RA_Current']}  \n"
                                    f"**Tier:** {qa.get('Tier', 'Tier-1')}")
                    with rd2:
                        st.markdown(f"**Budget:** ${qa.get('_budget', 0):,.0f}  \n"
                                    f"**Equity:** {qa.get('Equity_to_IP_Pct', 0.10):.0%}  \n"
                                    f"**Entry Month:** {qa.get('Entry_Month', '?')}")
                    with rd3:
                        st.markdown(f"**Mechanism:** {qa.get('MechCluster_ID', '?')}  \n"
                                    f"**Indication:** {qa.get('IndicationCluster_ID', '?')}  \n"
                                    f"**Source:** {qa.get('_source', '?')}")

                    # Sandbox analysis results (if attached)
                    if qa.get("_hedge_score") is not None or qa.get("_hedge_rank") is not None:
                        sa1, sa2, sa3 = st.columns(3)
                        with sa1:
                            score = qa.get("_hedge_score")
                            if score is not None:
                                color = "🟢" if score > 1 else ("🟡" if score > 0 else "🔴")
                                st.metric(f"{color} Hedge Score", f"{score:.2f}")
                        with sa2:
                            rank = qa.get("_hedge_rank")
                            if rank is not None:
                                st.metric("Pipeline Rank", f"#{rank}")
                        with sa3:
                            gpass = qa.get("_gate_check_pass")
                            if gpass is not None:
                                st.metric("Sandbox Gate Check", "✅ PASS" if gpass else "❌ FAIL")

                    # ── Checklist ──
                    st.markdown("**Review Checklist**")
                    chk1 = st.checkbox("Governance gate check completed", key=f"chk_gate_{qi}",
                        value=qa.get("_last_admission_result") is not None)
                    chk2 = st.checkbox("Hedge profile reviewed (EDC/IRC/DPC/LAC/CDC)", key=f"chk_hedge_{qi}")
                    chk3 = st.checkbox("Budget and capital concentration verified", key=f"chk_budget_{qi}")
                    chk4 = st.checkbox("CRO/Pharma channel pathway confirmed", key=f"chk_channel_{qi}")
                    chk5 = st.checkbox("Sponsor / Board member sign-off obtained", key=f"chk_signoff_{qi}")
                    all_checked = all([chk1, chk2, chk3, chk4, chk5])

                    # Gate check results
                    last_adm = qa.get("_last_admission_result")
                    if last_adm:
                        if last_adm.get("all_pass"):
                            st.success("7-Gate Check: ✅ ALL PASS")
                        else:
                            st.error(f"7-Gate Check: ❌ FAILS — {', '.join(last_adm.get('failed_gates', []))}")
                        # Show gate detail
                        gr = [{"Gate": n, "Status": "✅ PASS" if c.get("pass") else "❌ FAIL",
                               "Detail": str(c.get("detail", ""))}
                              for n, c in last_adm.get("gates", {}).items()]
                        if gr:
                            st.dataframe(pd.DataFrame(gr), use_container_width=True, hide_index=True)

                    # Hedge profile
                    last_contrib = qa.get("_last_contrib")
                    if last_contrib:
                        st.markdown("**Hedge Profile**")
                        hc1, hc2, hc3, hc4, hc5 = st.columns(5)
                        with hc1: st.metric("EDC", f"{last_contrib.get('EDC', 0):+.1%}" if isinstance(last_contrib.get('EDC'), float) else "—")
                        with hc2: st.metric("IRC", f"{last_contrib.get('IRC', 0):+.2f}" if isinstance(last_contrib.get('IRC'), float) else "—")
                        with hc3: st.metric("DPC", f"{last_contrib.get('DPC', 0):+.1%}" if isinstance(last_contrib.get('DPC'), float) else "—")
                        with hc4: st.metric("LAC", f"{last_contrib.get('LAC', 0):+.1f}" if isinstance(last_contrib.get('LAC'), (int, float)) else "—")
                        with hc5: st.metric("CDC", f"{last_contrib.get('CDC', 0):+.3f}" if isinstance(last_contrib.get('CDC'), float) else "—")

                    # Actions
                    st.markdown("**Actions**")
                    act1, act2, act3 = st.columns(3)
                    with act1:
                        rv_sims = st.number_input("Sims", 200, 10000, 500, 100, key=f"lv_sims_{qi}")
                        if st.button("▶ Run 7-Gate Check", key=f"lv_check_{qi}", use_container_width=True):
                            from src.optimization.contribution import run_contribution_analysis
                            from src.governance.admission import check_admission_gates
                            xa = pd.DataFrame([{k: v for k, v in qa.items() if not k.startswith("_")}])
                            b = qa.get("_budget", 5e6); e = int(qa.get("Entry_Month", 3))
                            xt = pd.DataFrame([
                                {"Asset_ID": qa["Asset_ID"], "Tranche_ID": "T1", "Purpose": "Dev",
                                 "Budget": b*0.75, "Start_Month": e, "Stop_Month": e+14, "Status": "Planned"},
                                {"Asset_ID": qa["Asset_ID"], "Tranche_ID": "T2", "Purpose": "BD",
                                 "Budget": b*0.25, "Start_Month": e+10, "Stop_Month": e+18, "Status": "Planned"},
                            ])
                            with st.spinner(f"Running admission for {qa['Asset_ID']}..."):
                                contrib = run_contribution_analysis(asset_state, tranches, xa, xt, params, n_sims=int(rv_sims))
                                adm_result = check_admission_gates(contrib, envelope)
                            live_queue[qi]["_last_admission_result"] = adm_result
                            live_queue[qi]["_last_contrib"] = {k: v for k, v in contrib.items() if isinstance(v, (int, float, str, bool))}
                            st.session_state.admission_queue = live_queue
                            st.rerun()

                    with act2:
                        approver = st.text_input("Approver Name", st.session_state.get("user_name", ""),
                                                 key=f"lv_approver_{qi}")
                        approve_disabled = not all_checked
                        if st.button("✅ APPROVE — Add to Portfolio", key=f"lv_approve_{qi}",
                                     type="primary", use_container_width=True, disabled=approve_disabled):
                            live_queue[qi]["_admission_status"] = "Approved"
                            live_queue[qi]["_approved_by"] = approver
                            live_queue[qi]["_approval_timestamp"] = pd.Timestamp.now().isoformat()
                            st.session_state.admission_queue = live_queue
                            log_action(DATA_DIR, st.session_state.get("user_email", "unknown"),
                                "approve_admission_live", {
                                    "asset": qa["Asset_ID"], "approver": approver,
                                    "gate_check_pass": bool(last_adm.get("all_pass")) if last_adm else None,
                                })
                            st.rerun()
                        if approve_disabled:
                            st.caption("Complete all checklist items to enable approval")

                    with act3:
                        reject_reason = st.text_input("Rejection Reason", "", key=f"lv_reason_{qi}")
                        if st.button("❌ REJECT", key=f"lv_reject_{qi}", use_container_width=True):
                            live_queue[qi]["_admission_status"] = "Rejected"
                            live_queue[qi]["_rejected_by"] = st.session_state.get("user_name", "")
                            live_queue[qi]["_rejection_reason"] = reject_reason
                            live_queue[qi]["_rejection_timestamp"] = pd.Timestamp.now().isoformat()
                            st.session_state.admission_queue = live_queue
                            log_action(DATA_DIR, st.session_state.get("user_email", "unknown"),
                                "reject_admission_live", {"asset": qa["Asset_ID"], "reason": reject_reason})
                            st.rerun()

        # ── Approved assets ──
        if approved_items:
            st.markdown("### ✅ Approved Assets")
            approved_rows = []
            for qa in approved_items:
                approved_rows.append({
                    "Asset_ID": qa["Asset_ID"],
                    "DS": qa["DS_Current"],
                    "RA": qa["RA_Current"],
                    "Mechanism": qa.get("MechCluster_ID", "—"),
                    "Indication": qa.get("IndicationCluster_ID", "—"),
                    "Budget": f"${qa.get('_budget', 0):,.0f}",
                    "Approved By": qa.get("_approved_by", "—"),
                    "Approved At": qa.get("_approval_timestamp", "—")[:16] if qa.get("_approval_timestamp") else "—",
                })
            st.dataframe(pd.DataFrame(approved_rows), use_container_width=True, hide_index=True)

        # ── Rejected assets ──
        if rejected_items:
            with st.expander(f"❌ Rejected ({len(rejected_items)})", expanded=False):
                rejected_rows = []
                for qa in rejected_items:
                    rejected_rows.append({
                        "Asset_ID": qa["Asset_ID"],
                        "DS": qa["DS_Current"],
                        "Reason": qa.get("_rejection_reason", "—"),
                        "Rejected By": qa.get("_rejected_by", "—"),
                    })
                st.dataframe(pd.DataFrame(rejected_rows), use_container_width=True, hide_index=True)

        # Clear completed
        if approved_items or rejected_items:
            if st.button("Clear Completed Submissions", key="lv_clear_completed"):
                st.session_state.admission_queue = [a for a in live_queue
                    if a.get("_admission_status", "Pending") == "Pending"]
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════
# TAB: ASSET PROFILES
# ══════════════════════════════════════════════════════════════════════════

with tab_assets:
    st.subheader("Asset Profiles & Master Data", help=(
        "Complete CRO and Pharma master data from the Excel workbooks. "
        "View capabilities, services, deal terms, and vendor relationships."
    ))

    # Load master data if available
    _cro_loaded = False
    _pharma_loaded = False
    try:
        from src.channel import load_cro_master, load_pharma_master
        _cro_data = load_cro_master(cro_path_str)
        _cro_loaded = True
    except Exception as e:
        st.warning(f"CRO Master not loaded: {e}")
    try:
        _pharma_data = load_pharma_master(pharma_path_str)
        _pharma_loaded = True
    except Exception:
        pass

    asset_section = st.radio("View", [
        "Portfolio Assets",
        "CRO Master",
        "Pharma Master",
        "AVL Relationships",
    ], horizontal=True, key="asset_section")

    if asset_section == "Portfolio Assets":
        st.markdown("**Current Portfolio Assets**")
        # Show full asset state with all available columns
        st.dataframe(asset_state, use_container_width=True, hide_index=True)

        st.markdown("**Capital Tranches**")
        st.dataframe(tranches, use_container_width=True, hide_index=True)

        # Per-asset detail
        if len(asset_state) > 0:
            selected_asset = st.selectbox("Asset Detail View",
                                          asset_state["Asset_ID"].tolist(), key="asset_detail_sel")
            if selected_asset:
                row = asset_state[asset_state["Asset_ID"] == selected_asset].iloc[0]
                asset_tr = tranches[tranches["Asset_ID"] == selected_asset]
                total_budget = asset_tr["Budget"].sum()

                dc1, dc2, dc3, dc4 = st.columns(4)
                with dc1: st.metric("Dev Stage", row["DS_Current"])
                with dc2: st.metric("Reg Access", row["RA_Current"])
                with dc3: st.metric("Total Budget", f"${total_budget:,.0f}")
                with dc4: st.metric("Entry Month", str(row.get("Entry_Month", 0)))

                if "CRO_ID" in row.index and pd.notna(row.get("CRO_ID")):
                    st.markdown(f"**CRO Assignment:** {row['CRO_ID']}")
                if "Target_Pharma_IDs" in row.index and pd.notna(row.get("Target_Pharma_IDs")):
                    st.markdown(f"**Target Pharmas:** {row['Target_Pharma_IDs']}")

                # Channel effects for this asset
                if channel_lookup and selected_asset in channel_lookup:
                    ch = channel_lookup[selected_asset]
                    st.markdown("**Channel Effects**")
                    ce1, ce2, ce3, ce4 = st.columns(4)
                    with ce1: st.metric("Deal Multiplier", f"{ch['deal_prob_mult']:.3f}x")
                    with ce2: st.metric("Time Shift", f"{ch['time_shift_months']:+.0f} mo")
                    with ce3:
                        arb = ch.get("engagement_arbitrage_months")
                        st.metric("Engagement Arbitrage", f"{arb:+.1f} mo" if arb else "—")
                    with ce4:
                        eng = ch.get("engagement_complete", False)
                        st.metric("Engagement", "✅ Complete" if eng else "⏳ Pending")

                    if ch.get("ind_cost"):
                        st.markdown(f"**Est. CRO IND Cost:** ${ch['ind_cost'][0]:,.0f} / "
                                    f"${ch['ind_cost'][1]:,.0f} / ${ch['ind_cost'][2]:,.0f} (Min/Mode/Max)")
                    if ch.get("ind_time"):
                        st.markdown(f"**Est. CRO IND Time:** {ch['ind_time'][0]:.0f} / "
                                    f"{ch['ind_time'][1]:.0f} / {ch['ind_time'][2]:.0f} months (Min/Mode/Max)")

    elif asset_section == "CRO Master":
        if _cro_loaded:
            st.markdown("**CRO Interface** (engine-consumed parameters)")
            st.dataframe(_cro_data["cro_interface_df"], use_container_width=True, hide_index=True)

            # IND Estimates
            if _cro_data.get("ind_estimates"):
                st.markdown("**CRO IND Cost/Time Estimates**")
                ind_rows = []
                for cro_id, est in _cro_data["ind_estimates"].items():
                    ind_rows.append({
                        "CRO_ID": cro_id,
                        "Cost Min": f"${est['cost_to_ind'][0]:,.0f}",
                        "Cost Mode": f"${est['cost_to_ind'][1]:,.0f}",
                        "Cost Max": f"${est['cost_to_ind'][2]:,.0f}",
                        "Time Min": f"{est['time_to_ind'][0]:.0f}mo",
                        "Time Mode": f"{est['time_to_ind'][1]:.0f}mo",
                        "Time Max": f"{est['time_to_ind'][2]:.0f}mo",
                        "Engagement Arb": f"{est['industry_engagement_time'][1] - est['discovery_engagement_time'][1]:+.1f}mo",
                    })
                st.dataframe(pd.DataFrame(ind_rows), use_container_width=True, hide_index=True)

            # Services
            st.markdown("**CRO Services**")
            svc_rows = []
            for cro_id, services in _cro_data["cro_services"].items():
                for svc in services:
                    svc_rows.append({
                        "CRO_ID": cro_id,
                        "Category": svc["category"],
                        "Subcategory": svc["subcategory"],
                        "Phase": svc["phase"],
                        "GLP": svc["glp"],
                        "Oncology": svc["onc_specific"],
                    })
            st.dataframe(pd.DataFrame(svc_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Load CRO Master workbook via the Channel sidebar toggle.")

    elif asset_section == "Pharma Master":
        if _pharma_loaded:
            st.markdown("**Pharma Interface** (engine-consumed parameters)")
            st.dataframe(_pharma_data["pharma_interface_df"], use_container_width=True, hide_index=True)

            st.markdown("**Deal Terms by Stage**")
            deal_rows = []
            for (ph_id, stage), terms in _pharma_data["deal_terms_lookup"].items():
                deal_rows.append({
                    "Pharma_ID": ph_id,
                    "Stage": stage,
                    "Upfront": f"${terms['upfront']:,.0f}",
                    "Milestones": f"${terms['milestones']:,.0f}",
                    "Royalty": f"{terms['royalty_pct']:.0%}",
                    "Time to Deal": f"{terms['time_to_deal']:.0f}mo",
                })
            st.dataframe(pd.DataFrame(deal_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Load Pharma Master workbook via the Channel sidebar toggle.")

    elif asset_section == "AVL Relationships":
        if _pharma_loaded:
            st.markdown("**Approved Vendor List — Pharma ↔ CRO Relationships**")
            st.caption("Vendor alignment creates a structural deal probability boost. "
                       "Confirmed relationships enter the base case; Likely for upside scenarios.")
            st.dataframe(_pharma_data["avl_df"], use_container_width=True, hide_index=True)
        else:
            st.info("Load Pharma Master workbook via the Channel sidebar toggle.")



    # ── Sensitivity Analysis ──
    with st.expander("🌪️ Sensitivity Analysis (Tornado)", expanded=False):
        st.caption("Shocks each key parameter up/down by a fixed percentage and ranks by MOIC impact.")
        sc1, sc2 = st.columns([1, 3])
        with sc1:
            sens_shock = st.slider("Shock (%)", 5, 40, 20, 5, key="sens_shock",
                                   help="Percentage shock applied up and down to each factor")
            sens_sims = st.number_input("Simulations", 200, 20000, 1000, 100, key="sens_sims")
            sens_run = st.button("▶ Run Sensitivity", key="sens_run_btn")
        with sc2:
            if sens_run:
                from src.simulation.sensitivity import run_sensitivity_analysis
                with st.spinner(f"Running tornado analysis (±{sens_shock}%, {sens_sims:,} sims per factor)..."):
                    sa = run_sensitivity_analysis(asset_state, tranches, params,
                        n_sims=int(sens_sims), shock_pct=sens_shock / 100.0, annual_overhead=float(overhead))
                log_action(DATA_DIR, st.session_state.get("user_email", "unknown"),
                    "run_sensitivity", {"shock_pct": sens_shock, "n_sims": int(sens_sims)})
                base = sa["base_summary"]
                bm1, bm2, bm3, bm4 = st.columns(4)
                with bm1: st.metric("Base MOIC", f"{base['Median_MOIC']:.2f}x")
                with bm2: st.metric("Base IRR", f"{base['Median_Annual_IRR']:.1%}" if pd.notna(base.get('Median_Annual_IRR')) else "N/A")
                with bm3: st.metric("Base P(3+)", f"{base['P_ThreePlus_Exits']:.1%}")
                with bm4: st.metric("Base P(≤1)", f"{base['P_Exits_LE1']:.1%}")
                if sa["tornado_data"]:
                    st.markdown("**MOIC Tornado Ranking**")
                    tornado_rows = [{"Factor": t["factor"], "Down": f"{t['deltas']['Median_MOIC']['down']:.2f}x",
                        "Base": f"{t['deltas']['Median_MOIC']['base']:.2f}x", "Up": f"{t['deltas']['Median_MOIC']['up']:.2f}x",
                        "Range": f"{t['deltas']['Median_MOIC']['range']:.2f}x"} for t in sa["tornado_data"]]
                    st.dataframe(pd.DataFrame(tornado_rows), use_container_width=True, hide_index=True)
                    st.markdown("**P(3+ Exits) Sensitivity**")
                    p3_rows = [{"Factor": t["factor"], "Down": f"{t['deltas']['P_ThreePlus_Exits']['down']:.1%}",
                        "Base": f"{t['deltas']['P_ThreePlus_Exits']['base']:.1%}", "Up": f"{t['deltas']['P_ThreePlus_Exits']['up']:.1%}"}
                        for t in sa["tornado_data"]]
                    st.dataframe(pd.DataFrame(p3_rows), use_container_width=True, hide_index=True)

    # ── Marginal Contribution ──
    with st.expander("📐 Marginal Contribution (Leave-One-Out)", expanded=False):
        st.caption("Removes each asset one at a time and measures 5-axis contribution deltas.")
        cc1, cc2 = st.columns([1, 3])
        with cc1:
            cont_sims = st.number_input("Simulations", 200, 20000, 1000, 100, key="cont_sims")
            cont_run = st.button("▶ Run Contribution", key="cont_run_btn")
        with cc2:
            if cont_run:
                from src.optimization.hedge import compute_marginal_contribution
                with st.spinner(f"Computing contributions ({cont_sims:,} sims per leave-one-out)..."):
                    mc_df = compute_marginal_contribution(asset_state, tranches, params,
                        n_sims=int(cont_sims), seed=42, annual_overhead=float(overhead))
                log_action(DATA_DIR, st.session_state.get("user_email", "unknown"),
                    "run_contribution", {"n_sims": int(cont_sims)})
                st.markdown("**Marginal Contribution**")
                st.caption("EDC = Exit Density | IRC = IRR | DPC = Downside Protection | LAC = Liquidity | CDC = Correlation")
                st.dataframe(mc_df, use_container_width=True, hide_index=True)
                if len(mc_df) >= 2 and "MOIC_Delta" in mc_df.columns:
                    valid = mc_df.dropna(subset=["MOIC_Delta"])
                    if len(valid) >= 2:
                        best = valid.loc[valid["MOIC_Delta"].idxmin()]
                        worst = valid.loc[valid["MOIC_Delta"].idxmax()]
                        st.markdown(f"**Most critical:** {best['Asset_ID']} (removing drops MOIC by {best['MOIC_Delta']:.2f}x)  \n"
                                    f"**Least critical:** {worst['Asset_ID']} (MOIC delta: {worst['MOIC_Delta']:+.2f}x)")

# ══════════════════════════════════════════════════════════════════════════
# TAB 6: SANDBOX
# ══════════════════════════════════════════════════════════════════════════

with tab_sandbox:
    st.subheader("Portfolio Construction Workbench", help=TIPS["sandbox"])
    st.markdown(
        '<div style="background:#fef3c7;border-left:6px solid #f59e0b;padding:12px;'
        'border-radius:6px;margin-bottom:16px;">'
        '<strong>⚠️ SANDBOX — EXPLORATORY ONLY</strong> — Build and test portfolio compositions '
        'without modifying locked parameters. Nothing here writes back to the production state.'
        '</div>', unsafe_allow_html=True)

    # ── Sub-navigation ──
    wb_mode = st.radio("Mode", [
        "🔧 Manual Build",
        "🤖 Auto-Optimize",
        "🚪 Admission",
        "📊 Run Engine",
        "⚖️ Governance",
    ], horizontal=True, key="wb_mode")

    # ── Shared state: sandbox portfolio ──
    # Initialize sandbox roster in session state
    if "sb_portfolio" not in st.session_state:
        # Start with copies of real assets, all toggled ON
        sb_assets = []
        for _, row in asset_state.iterrows():
            entry = row.to_dict()
            entry["_source"] = "portfolio"
            entry["_enabled"] = True
            sb_assets.append(entry)
        st.session_state.sb_portfolio = sb_assets
        st.session_state.sb_custom_counter = 0

    sb_portfolio = st.session_state.sb_portfolio

    # ══════════════════════════════════════════════════════════════════════
    # MODE 1: MANUAL BUILD
    # ══════════════════════════════════════════════════════════════════════
    if wb_mode == "🔧 Manual Build":
        st.markdown("### Portfolio Roster")
        st.caption("Toggle existing assets on/off. Add candidate assets manually or from the optimizer.")

        # ── Existing assets toggle ──
        st.markdown("**Current Assets**")
        for i, asset in enumerate(sb_portfolio):
            if asset["_source"] == "portfolio":
                col_a, col_b, col_c = st.columns([0.5, 2, 0.5])
                with col_a:
                    enabled = st.checkbox("", value=asset["_enabled"],
                                         key=f"sb_toggle_{i}", label_visibility="collapsed")
                    sb_portfolio[i]["_enabled"] = enabled
                with col_b:
                    status = "✅" if enabled else "❌"
                    st.markdown(f"{status} **{asset['Asset_ID']}** — {asset['DS_Current']}/{asset['RA_Current']} "
                                f"({asset.get('Tier', 'Tier-1')})")
                with col_c:
                    pass  # can't remove portfolio assets, only toggle

        # ── Custom asset candidates ──
        st.markdown("---")
        st.markdown("**Add Candidate Asset**")

        with st.form("add_candidate_form"):
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                cand_ds = st.selectbox("Dev Stage", ["DS-3", "DS-4", "DS-5"], key="cand_ds")
                cand_ra = st.selectbox("Reg Access", ["RA-1", "RA-2"], key="cand_ra")
                cand_tier = st.selectbox("Tier", ["Tier-1", "Tier-2"], key="cand_tier")
            with ac2:
                cand_mech = st.text_input("Mechanism Cluster", "MECH-NEW", key="cand_mech")
                cand_ind = st.text_input("Indication Cluster", "IND-NEW", key="cand_ind")
                cand_geo = st.text_input("Geography Cluster", "GEO-US1", key="cand_geo")
            with ac3:
                cand_budget = st.number_input("Total Budget ($)", 1_000_000, 15_000_000,
                                              5_000_000, 500_000, key="cand_budget")
                cand_equity = st.slider("Equity to IP (%)", 0, 25, 10, 1, key="cand_equity")
                cand_entry = st.number_input("Entry Month", 0, 24, 3, 1, key="cand_entry")

            add_btn = st.form_submit_button("Add to Sandbox Portfolio")

        if add_btn:
            st.session_state.sb_custom_counter += 1
            n = st.session_state.sb_custom_counter
            new_id = f"SB-CAND-{n:04d}"
            new_asset = {
                "Asset_ID": new_id,
                "DS_Current": cand_ds,
                "RA_Current": cand_ra,
                "Tier": cand_tier,
                "Entry_Month": cand_entry,
                "Equity_to_IP_Pct": cand_equity / 100.0,
                "AcqCash_to_IP": 0,
                "EarlyPassThrough_Pct": 0.0,
                "EarlyDeferredCash": 0,
                "MechCluster_ID": cand_mech,
                "IndicationCluster_ID": cand_ind,
                "GeoRACluster_ID": cand_geo,
                "_source": "sandbox",
                "_enabled": True,
                "_budget": cand_budget,
            }
            sb_portfolio.append(new_asset)
            st.session_state.sb_portfolio = sb_portfolio
            st.rerun()

        # ── Upload candidates from Excel ──
        st.markdown("---")
        st.markdown("**Add from Candidate Pipeline**")
        st.caption(
            "Upload a candidate pipeline Excel file (e.g. Candidate_Pipeline.xlsx). "
            "Browse candidates, view details, and add individually or in bulk."
        )
        uploaded_file = st.file_uploader("Upload Pipeline (.xlsx)", type=["xlsx"], key="sb_upload")
        if uploaded_file is not None:
            try:
                sb_xl = pd.ExcelFile(uploaded_file)
                sb_data_sheets = [s for s in sb_xl.sheet_names if s.upper() != "README"]
                upload_df = pd.read_excel(sb_xl, sb_data_sheets[0] if sb_data_sheets else 0)

                # Detect column format — pipeline uses DS/RA, generic uses DS_Current/RA_Current
                if "DS" in upload_df.columns and "DS_Current" not in upload_df.columns:
                    upload_df = upload_df.rename(columns={"DS": "DS_Current", "RA": "RA_Current"})

                required_cols = {"Asset_ID", "DS_Current", "RA_Current"}
                if not required_cols.issubset(set(upload_df.columns)):
                    # Try alternate column names
                    if "DS" in upload_df.columns:
                        upload_df = upload_df.rename(columns={"DS": "DS_Current", "RA": "RA_Current"})
                    if not required_cols.issubset(set(upload_df.columns)):
                        st.error(f"Missing required columns: {required_cols - set(upload_df.columns)}")
                        upload_df = None

                if upload_df is not None and len(upload_df) > 0:
                    # Show pipeline overview
                    display_cols = [c for c in ["Asset_ID", "Name", "DS_Current", "RA_Current",
                                                "Mechanism", "Indication", "Budget", "Tier"]
                                   if c in upload_df.columns]
                    st.dataframe(upload_df[display_cols], use_container_width=True, hide_index=True)

                    # ── Auto-Rank: Hedge Doctrine Recommendation ──
                    def _build_sb_asset_for_rank(row):
                        asset = {
                            "Asset_ID": str(row["Asset_ID"]),
                            "DS_Current": str(row["DS_Current"]),
                            "RA_Current": str(row["RA_Current"]),
                            "Tier": str(row.get("Tier", "Tier-1")),
                            "Entry_Month": int(row.get("Entry_Month", 3)),
                            "Equity_to_IP_Pct": float(row.get("Equity_Pct", 10)) / 100.0 if float(row.get("Equity_Pct", 10)) > 1 else float(row.get("Equity_Pct", 0.10)),
                            "AcqCash_to_IP": 0, "EarlyPassThrough_Pct": 0.0, "EarlyDeferredCash": 0,
                            "MechCluster_ID": str(row.get("MechCluster_ID", "MECH-NEW")),
                            "IndicationCluster_ID": str(row.get("IndicationCluster_ID", "IND-NEW")),
                            "GeoRACluster_ID": str(row.get("GeoRACluster_ID", "GEO-US1")),
                            "_source": "pipeline", "_enabled": True,
                            "_budget": float(row.get("Budget", 5_000_000)),
                        }
                        if "CRO_ID" in row.index and pd.notna(row.get("CRO_ID")):
                            asset["CRO_ID"] = str(row["CRO_ID"])
                        if "Target_Pharma_IDs" in row.index and pd.notna(row.get("Target_Pharma_IDs")):
                            asset["Target_Pharma_IDs"] = str(row["Target_Pharma_IDs"])
                        return asset

                    st.markdown("---")
                    rank_col1, rank_col2 = st.columns([3, 1])
                    with rank_col1:
                        st.markdown("#### Recommended Next Assets (Hedge Ranked)")
                    with rank_col2:
                        rank_sims = st.number_input("Sims per candidate", 30, 300, 50, 10,
                                                    key="sb_rank_sims")

                    run_rank = st.button("▶ Rank All Candidates by Portfolio Fit",
                                        key="sb_rank_btn", use_container_width=True)

                    if run_rank:
                        from src.optimization.contribution import run_contribution_analysis
                        from src.governance.admission import check_admission_gates

                        # Build current sandbox portfolio
                        sb_enabled = [a for a in sb_portfolio if a["_enabled"]]
                        if sb_enabled:
                            sb_base_as = pd.DataFrame([{k: v for k, v in a.items() if not k.startswith("_")} for a in sb_enabled])
                            sb_base_tr_list = []
                            for a in sb_enabled:
                                aid = a["Asset_ID"]
                                if a.get("_source") == "portfolio":
                                    sb_base_tr_list.append(tranches[tranches["Asset_ID"] == aid])
                                else:
                                    bgt = a.get("_budget", 5_000_000)
                                    ent = int(a.get("Entry_Month", 3))
                                    sb_base_tr_list.append(pd.DataFrame([
                                        {"Asset_ID": aid, "Tranche_ID": "T1", "Purpose": "Development",
                                         "Budget": bgt * 0.75, "Start_Month": ent, "Stop_Month": ent + 14, "Status": "Planned"},
                                        {"Asset_ID": aid, "Tranche_ID": "T2", "Purpose": "BD",
                                         "Budget": bgt * 0.25, "Start_Month": ent + 10, "Stop_Month": ent + 18, "Status": "Planned"},
                                    ]))
                            sb_base_tr = pd.concat(sb_base_tr_list, ignore_index=True)
                        else:
                            sb_base_as = asset_state
                            sb_base_tr = tranches

                        rank_results = []
                        progress = st.progress(0, text="Evaluating candidates...")
                        for ci, (_, crow) in enumerate(upload_df.iterrows()):
                            progress.progress((ci + 1) / len(upload_df),
                                text=f"Evaluating {crow['Asset_ID']} ({ci+1}/{len(upload_df)})...")
                            c_asset = _build_sb_asset_for_rank(crow)
                            c_as = pd.DataFrame([{k: v for k, v in c_asset.items() if not k.startswith("_")}])
                            c_bgt = c_asset.get("_budget", 5_000_000)
                            c_ent = int(c_asset.get("Entry_Month", 3))
                            c_tr = pd.DataFrame([
                                {"Asset_ID": c_asset["Asset_ID"], "Tranche_ID": "T1", "Purpose": "Development",
                                 "Budget": c_bgt * 0.75, "Start_Month": c_ent, "Stop_Month": c_ent + 14, "Status": "Planned"},
                                {"Asset_ID": c_asset["Asset_ID"], "Tranche_ID": "T2", "Purpose": "BD",
                                 "Budget": c_bgt * 0.25, "Start_Month": c_ent + 10, "Stop_Month": c_ent + 18, "Status": "Planned"},
                            ])
                            try:
                                contrib = run_contribution_analysis(
                                    sb_base_as, sb_base_tr, c_as, c_tr, params, n_sims=int(rank_sims))
                                admission = check_admission_gates(contrib, envelope)

                                edc_val = contrib.get("EDC", 0) if isinstance(contrib.get("EDC"), float) else 0
                                irc_val = contrib.get("IRC", 0) if isinstance(contrib.get("IRC"), float) else 0
                                dpc_val = contrib.get("DPC", 0) if isinstance(contrib.get("DPC"), float) else 0
                                lac_val = contrib.get("LAC", 0) if isinstance(contrib.get("LAC"), (int, float)) else 0
                                cdc_val = contrib.get("CDC", 0) if isinstance(contrib.get("CDC"), float) else 0
                                moic_d = contrib.get("MOIC_Delta", 0) if isinstance(contrib.get("MOIC_Delta"), float) else 0

                                # Composite hedge score: weight EDC highest, penalize positive CDC
                                hedge_score = (0.30 * edc_val * 100  # scale % to points
                                             + 0.25 * min(irc_val, 2.0)  # cap IRC contribution
                                             - 0.20 * dpc_val * 100  # negative DPC is good
                                             - 0.15 * lac_val  # negative LAC is good
                                             - 0.10 * cdc_val * 100)  # negative CDC is good

                                rank_results.append({
                                    "Asset_ID": c_asset["Asset_ID"],
                                    "Name": crow.get("Name", "?"),
                                    "DS": c_asset["DS_Current"],
                                    "Admission": admission.get("all_pass", False),
                                    "EDC": edc_val,
                                    "IRC": irc_val,
                                    "DPC": dpc_val,
                                    "LAC": lac_val,
                                    "CDC": cdc_val,
                                    "MOIC_Delta": moic_d,
                                    "Hedge_Score": hedge_score,
                                })
                            except Exception:
                                rank_results.append({
                                    "Asset_ID": c_asset["Asset_ID"],
                                    "Name": crow.get("Name", "?"),
                                    "DS": c_asset["DS_Current"],
                                    "Admission": False,
                                    "EDC": 0, "IRC": 0, "DPC": 0, "LAC": 0, "CDC": 0,
                                    "MOIC_Delta": 0, "Hedge_Score": -999,
                                })
                        progress.empty()

                        # Sort by hedge score descending
                        rank_df = pd.DataFrame(rank_results).sort_values("Hedge_Score", ascending=False).reset_index(drop=True)
                        rank_df.insert(0, "Rank", range(1, len(rank_df) + 1))

                        # Color code: top 1/3 green, middle 1/3 yellow, bottom 1/3 red
                        n = len(rank_df)
                        def _fit_label(i):
                            if i < n / 3:
                                return "🟢 Strong"
                            elif i < 2 * n / 3:
                                return "🟡 Moderate"
                            else:
                                return "🔴 Weak"

                        display_rank = rank_df.copy()
                        display_rank["Fit"] = [_fit_label(i) for i in range(n)]
                        display_rank["Admission"] = display_rank["Admission"].map({True: "✅ PASS", False: "❌ FAIL"})
                        display_rank["EDC"] = display_rank["EDC"].apply(lambda x: f"{x:+.1%}")
                        display_rank["IRC"] = display_rank["IRC"].apply(lambda x: f"{x:+.2f}")
                        display_rank["DPC"] = display_rank["DPC"].apply(lambda x: f"{x:+.1%}")
                        display_rank["LAC"] = display_rank["LAC"].apply(lambda x: f"{x:+.1f}")
                        display_rank["CDC"] = display_rank["CDC"].apply(lambda x: f"{x:+.3f}")
                        display_rank["MOIC_Delta"] = display_rank["MOIC_Delta"].apply(lambda x: f"{x:+.2f}x")
                        display_rank["Score"] = rank_df["Hedge_Score"].apply(lambda x: f"{x:.2f}")

                        st.dataframe(
                            display_rank[["Rank", "Fit", "Asset_ID", "Name", "DS", "Admission",
                                         "EDC", "DPC", "CDC", "MOIC_Delta", "Score"]],
                            use_container_width=True, hide_index=True)

                        st.caption(
                            "**Hedge Score** = weighted composite: EDC (30%) + IRC (25%) − DPC (20%) − LAC (15%) − CDC (10%). "
                            "Higher is better. 🟢 Strong = top third, 🟡 Moderate = middle, 🔴 Weak = bottom third. "
                            "Candidates that FAIL admission are penalized but still shown for comparison."
                        )

                        # Top recommendation callout
                        top = rank_df.iloc[0]
                        if top["Admission"]:
                            st.success(
                                f"**Recommended:** {top['Asset_ID']} ({top['Name']}) — "
                                f"Score {top['Hedge_Score']:.2f}, MOIC {top['MOIC_Delta']:+.2f}x, "
                                f"EDC {top['EDC']:+.1%}, CDC {top['CDC']:+.3f}")
                        else:
                            # Find top that passes admission
                            passing = rank_df[rank_df["Admission"] == True]
                            if len(passing) > 0:
                                top_pass = passing.iloc[0]
                                st.success(
                                    f"**Recommended (passes gates):** {top_pass['Asset_ID']} ({top_pass['Name']}) — "
                                    f"Score {top_pass['Hedge_Score']:.2f}, MOIC {top_pass['MOIC_Delta']:+.2f}x")
                            else:
                                st.warning("No candidates pass all admission gates with current portfolio composition")

                        # Store ranking in session for picker default and submission
                        st.session_state["sb_rank_order"] = rank_df["Asset_ID"].tolist()
                        # Store per-asset analysis results for attaching to submissions
                        rank_lookup = {}
                        for _, rrow in rank_df.iterrows():
                            rank_lookup[rrow["Asset_ID"]] = {
                                "Hedge_Score": rrow["Hedge_Score"],
                                "EDC": rrow["EDC"],
                                "IRC": rrow["IRC"],
                                "DPC": rrow["DPC"],
                                "LAC": rrow["LAC"],
                                "CDC": rrow["CDC"],
                                "MOIC_Delta": rrow["MOIC_Delta"],
                                "Admission": rrow["Admission"],
                                "Rank": int(rrow["Rank"]),
                            }
                        st.session_state["sb_rank_results"] = rank_lookup

                    st.markdown("---")

                    # ── Multi-select picker (uses rank order if available) ──
                    if "Name" in upload_df.columns:
                        pick_options = [f"{row['Asset_ID']} — {row['Name']} ({row['DS_Current']}/{row['RA_Current']})"
                                       for _, row in upload_df.iterrows()]
                    else:
                        pick_options = [f"{row['Asset_ID']} ({row['DS_Current']}/{row['RA_Current']})"
                                       for _, row in upload_df.iterrows()]

                    sb_pick1, sb_pick2 = st.columns([3, 1])
                    with sb_pick1:
                        pick_indices = st.multiselect("Select candidates to add",
                                                     range(len(pick_options)),
                                                     format_func=lambda i: pick_options[i],
                                                     key="sb_pipeline_pick")
                    with sb_pick2:
                        st.markdown("")  # spacer
                        add_selected = st.button("Add Selected", key="sb_pipeline_add_sel")

                    # Hedge doctrine toggle
                    sb_hedge_check = st.checkbox("Run Hedge Doctrine check before adding",
                        value=False, key="sb_hedge_doctrine",
                        help="Evaluates each candidate's 5-axis contribution (EDC/IRC/DPC/LAC/CDC) "
                             "against the current sandbox portfolio. Shows which candidates improve "
                             "diversification vs which add correlation risk.")

                    # Show selected candidates detail
                    if pick_indices:
                        st.markdown(f"**{len(pick_indices)} candidate(s) selected**")
                        for pidx in pick_indices:
                            pick_row = upload_df.iloc[pidx]
                            with st.expander(f"{pick_row['Asset_ID']} — {pick_row.get('Name', '?')} "
                                            f"({pick_row['DS_Current']}/{pick_row['RA_Current']})", expanded=False):
                                pd1, pd2 = st.columns(2)
                                with pd1:
                                    st.markdown(f"**Mechanism:** {pick_row.get('Mechanism', pick_row.get('MechCluster_ID', '?'))}")
                                    st.markdown(f"**Indication:** {pick_row.get('Indication', pick_row.get('IndicationCluster_ID', '?'))}")
                                with pd2:
                                    st.markdown(f"**Budget:** ${pick_row.get('Budget', 0):,.0f}")
                                    st.markdown(f"**CRO:** {pick_row.get('CRO_ID', 'None')}")
                                if "Rationale" in pick_row.index and pd.notna(pick_row.get("Rationale")):
                                    st.markdown(f"**Rationale:** {pick_row['Rationale']}")

                    def _build_sb_asset(row):
                        asset = {
                            "Asset_ID": str(row["Asset_ID"]),
                            "DS_Current": str(row["DS_Current"]),
                            "RA_Current": str(row["RA_Current"]),
                            "Tier": str(row.get("Tier", "Tier-1")),
                            "Entry_Month": int(row.get("Entry_Month", 3)),
                            "Equity_to_IP_Pct": float(row.get("Equity_Pct", 10)) / 100.0 if float(row.get("Equity_Pct", 10)) > 1 else float(row.get("Equity_Pct", 0.10)),
                            "AcqCash_to_IP": 0,
                            "EarlyPassThrough_Pct": 0.0,
                            "EarlyDeferredCash": 0,
                            "MechCluster_ID": str(row.get("MechCluster_ID", "MECH-NEW")),
                            "IndicationCluster_ID": str(row.get("IndicationCluster_ID", "IND-NEW")),
                            "GeoRACluster_ID": str(row.get("GeoRACluster_ID", "GEO-US1")),
                            "_source": "pipeline",
                            "_enabled": True,
                            "_budget": float(row.get("Budget", 5_000_000)),
                        }
                        if "CRO_ID" in row.index and pd.notna(row.get("CRO_ID")):
                            asset["CRO_ID"] = str(row["CRO_ID"])
                        if "Target_Pharma_IDs" in row.index and pd.notna(row.get("Target_Pharma_IDs")):
                            asset["Target_Pharma_IDs"] = str(row["Target_Pharma_IDs"])
                        return asset

                    if add_selected and pick_indices:
                        # ── Hedge Doctrine Check ──
                        if sb_hedge_check:
                            from src.optimization.contribution import run_contribution_analysis
                            from src.governance.admission import check_admission_gates

                            # Build current sandbox portfolio state
                            sb_enabled = [a for a in sb_portfolio if a["_enabled"]]
                            if sb_enabled:
                                sb_base_as = pd.DataFrame([{k: v for k, v in a.items() if not k.startswith("_")} for a in sb_enabled])
                                sb_base_tr_list = []
                                for a in sb_enabled:
                                    aid = a["Asset_ID"]
                                    if a.get("_source") == "portfolio":
                                        sb_base_tr_list.append(tranches[tranches["Asset_ID"] == aid])
                                    else:
                                        bgt = a.get("_budget", 5_000_000)
                                        ent = int(a.get("Entry_Month", 3))
                                        sb_base_tr_list.append(pd.DataFrame([
                                            {"Asset_ID": aid, "Tranche_ID": "T1", "Purpose": "Development",
                                             "Budget": bgt * 0.75, "Start_Month": ent, "Stop_Month": ent + 14, "Status": "Planned"},
                                            {"Asset_ID": aid, "Tranche_ID": "T2", "Purpose": "BD",
                                             "Budget": bgt * 0.25, "Start_Month": ent + 10, "Stop_Month": ent + 18, "Status": "Planned"},
                                        ]))
                                sb_base_tr = pd.concat(sb_base_tr_list, ignore_index=True)
                            else:
                                sb_base_as = asset_state
                                sb_base_tr = tranches

                            st.markdown("#### Hedge Doctrine Analysis")
                            hedge_results = []
                            with st.spinner(f"Running hedge analysis on {len(pick_indices)} candidate(s)..."):
                                for pidx in pick_indices:
                                    crow = upload_df.iloc[pidx]
                                    c_asset = _build_sb_asset(crow)
                                    c_as = pd.DataFrame([{k: v for k, v in c_asset.items() if not k.startswith("_")}])
                                    c_bgt = c_asset.get("_budget", 5_000_000)
                                    c_ent = int(c_asset.get("Entry_Month", 3))
                                    c_tr = pd.DataFrame([
                                        {"Asset_ID": c_asset["Asset_ID"], "Tranche_ID": "T1", "Purpose": "Development",
                                         "Budget": c_bgt * 0.75, "Start_Month": c_ent, "Stop_Month": c_ent + 14, "Status": "Planned"},
                                        {"Asset_ID": c_asset["Asset_ID"], "Tranche_ID": "T2", "Purpose": "BD",
                                         "Budget": c_bgt * 0.25, "Start_Month": c_ent + 10, "Stop_Month": c_ent + 18, "Status": "Planned"},
                                    ])
                                    try:
                                        contrib = run_contribution_analysis(
                                            sb_base_as, sb_base_tr, c_as, c_tr, params, n_sims=100)
                                        admission = check_admission_gates(contrib, envelope)
                                        hedge_results.append({
                                            "Asset_ID": c_asset["Asset_ID"],
                                            "Name": crow.get("Name", "?"),
                                            "DS": c_asset["DS_Current"],
                                            "Admission": "✅ PASS" if admission.get("all_pass") else "❌ FAIL",
                                            "EDC": f"{contrib.get('EDC', 0):+.1%}" if isinstance(contrib.get('EDC'), float) else "—",
                                            "IRC": f"{contrib.get('IRC', 0):+.2f}" if isinstance(contrib.get('IRC'), float) else "—",
                                            "DPC": f"{contrib.get('DPC', 0):+.1%}" if isinstance(contrib.get('DPC'), float) else "—",
                                            "LAC": f"{contrib.get('LAC', 0):+.1f}" if isinstance(contrib.get('LAC'), (int, float)) else "—",
                                            "CDC": f"{contrib.get('CDC', 0):+.3f}" if isinstance(contrib.get('CDC'), float) else "—",
                                            "MOIC_Delta": f"{contrib.get('MOIC_Delta', 0):+.2f}x" if isinstance(contrib.get('MOIC_Delta'), float) else "—",
                                            "_raw_contrib": {k: v for k, v in contrib.items() if isinstance(v, (int, float, str, bool))},
                                            "_admission_pass": admission.get("all_pass", False),
                                        })
                                    except Exception as e:
                                        hedge_results.append({
                                            "Asset_ID": c_asset["Asset_ID"],
                                            "Name": crow.get("Name", "?"),
                                            "DS": c_asset["DS_Current"],
                                            "Admission": "Error",
                                            "EDC": "—", "IRC": "—", "DPC": "—", "LAC": "—", "CDC": "—",
                                            "MOIC_Delta": str(e)[:30],
                                            "_raw_contrib": None, "_admission_pass": False,
                                        })

                            # Store results for attaching to submissions
                            st.session_state["sb_hedge_doctrine_results"] = {
                                r["Asset_ID"]: r for r in hedge_results if r.get("_raw_contrib")
                            }

                            # Display (without raw fields)
                            display_hr = [{k: v for k, v in r.items() if not k.startswith("_")} for r in hedge_results]
                            st.dataframe(pd.DataFrame(display_hr), use_container_width=True, hide_index=True)
                            st.caption(
                                "EDC = Exit Density Change (P(3+ exits)) | IRC = IRR Change | "
                                "DPC = Downside Protection (P(≤1 exit)) | LAC = Liquidity Acceleration | "
                                "CDC = Correlation Diversification. Green admission = passes all 7 gates."
                            )

                        # Add selected to sandbox (attach analysis if available)
                        existing_ids = {a["Asset_ID"] for a in sb_portfolio}
                        added = 0
                        hedge_data = st.session_state.get("sb_hedge_doctrine_results", {})
                        rank_data = st.session_state.get("sb_rank_results", {})
                        for pidx in pick_indices:
                            row = upload_df.iloc[pidx]
                            rid = str(row["Asset_ID"])
                            if rid not in existing_ids:
                                new_asset = _build_sb_asset(row)
                                # Attach hedge doctrine results
                                if rid in hedge_data and hedge_data[rid].get("_raw_contrib"):
                                    new_asset["_last_contrib"] = hedge_data[rid]["_raw_contrib"]
                                    new_asset["_gate_check_pass"] = hedge_data[rid].get("_admission_pass", False)
                                # Attach rank results
                                if rid in rank_data:
                                    new_asset["_hedge_score"] = rank_data[rid].get("Hedge_Score")
                                    new_asset["_hedge_rank"] = rank_data[rid].get("Rank")
                                sb_portfolio.append(new_asset)
                                existing_ids.add(rid)
                                added += 1
                        st.session_state.sb_portfolio = sb_portfolio
                        if added > 0:
                            st.success(f"Added {added} candidate(s) to sandbox portfolio")
                            st.rerun()
                        else:
                            st.warning("All selected candidates are already in the sandbox")

                    # Bulk add all
                    if st.button("Add ALL pipeline candidates to sandbox", key="sb_pipeline_add_all"):
                        existing_ids = {a["Asset_ID"] for a in sb_portfolio}
                        added = 0
                        for _, row in upload_df.iterrows():
                            rid = str(row["Asset_ID"])
                            if rid not in existing_ids:
                                sb_portfolio.append(_build_sb_asset(row))
                                existing_ids.add(rid)
                                added += 1
                        st.session_state.sb_portfolio = sb_portfolio
                        st.success(f"Added {added} candidates from pipeline")
                        st.rerun()

            except Exception as e:
                st.error(f"Error reading Excel file: {e}")

        # ── Show sandbox candidates ──
        sandbox_assets = [a for a in sb_portfolio if a["_source"] != "portfolio"]
        if sandbox_assets:
            st.markdown("**Sandbox Candidates**")
            for i, asset in enumerate(sb_portfolio):
                if asset["_source"] == "portfolio":
                    continue
                col_a, col_b, col_c, col_d = st.columns([0.4, 2, 0.6, 0.4])
                with col_a:
                    enabled = st.checkbox("", value=asset["_enabled"],
                                         key=f"sb_cand_toggle_{i}", label_visibility="collapsed")
                    sb_portfolio[i]["_enabled"] = enabled
                with col_b:
                    status = "✅" if enabled else "❌"
                    adm_status = " 📋" if asset.get("_submitted_to_admission") else ""
                    budget_str = f"${asset.get('_budget', 0):,.0f}" if asset.get('_budget') else ""
                    st.markdown(f"{status} **{asset['Asset_ID']}** — {asset['DS_Current']}/{asset['RA_Current']} "
                                f"{asset.get('MechCluster_ID', '')} / {asset.get('IndicationCluster_ID', '')} "
                                f"{budget_str}{adm_status}")
                with col_c:
                    if not asset.get("_submitted_to_admission"):
                        if st.button("📋 Submit", key=f"sb_submit_{i}",
                                     help="Submit to Admission queue with analysis results"):
                            sb_portfolio[i]["_submitted_to_admission"] = True
                            # Add to admission queue
                            if "admission_queue" not in st.session_state:
                                st.session_state.admission_queue = []
                            # Avoid duplicates
                            existing_q = {a["Asset_ID"] for a in st.session_state.admission_queue}
                            if asset["Asset_ID"] not in existing_q:
                                submission = dict(asset)
                                # Attach hedge ranking results if available
                                rank_results = st.session_state.get("sb_rank_results", {})
                                if asset["Asset_ID"] in rank_results:
                                    rr = rank_results[asset["Asset_ID"]]
                                    submission["_last_contrib"] = {
                                        "EDC": rr["EDC"], "IRC": rr["IRC"], "DPC": rr["DPC"],
                                        "LAC": rr["LAC"], "CDC": rr["CDC"], "MOIC_Delta": rr["MOIC_Delta"],
                                    }
                                    submission["_hedge_score"] = rr["Hedge_Score"]
                                    submission["_hedge_rank"] = rr["Rank"]
                                    submission["_gate_check_pass"] = rr["Admission"]
                                st.session_state.admission_queue.append(submission)
                            st.session_state.sb_portfolio = sb_portfolio
                            log_action(DATA_DIR, st.session_state.get("user_email", "unknown"),
                                "submit_to_admission", {"asset": asset["Asset_ID"],
                                    "has_analysis": asset["Asset_ID"] in st.session_state.get("sb_rank_results", {})})
                            st.rerun()
                    else:
                        st.caption("📋 Submitted")
                with col_d:
                    if st.button("🗑️", key=f"sb_remove_{i}"):
                        sb_portfolio.pop(i)
                        st.session_state.sb_portfolio = sb_portfolio
                        st.rerun()

        # ── Summary ──
        enabled_assets = [a for a in sb_portfolio if a["_enabled"]]
        submitted_count = sum(1 for a in sb_portfolio if a.get("_submitted_to_admission"))
        st.markdown("---")
        sc1, sc2 = st.columns(2)
        with sc1:
            st.metric("Active Assets in Sandbox", len(enabled_assets))
        with sc2:
            st.metric("Submitted to Admission", submitted_count)

    # ══════════════════════════════════════════════════════════════════════
    # MODE 2: AUTO-OPTIMIZE
    # ══════════════════════════════════════════════════════════════════════
    elif wb_mode == "🤖 Auto-Optimize":
        st.markdown("### Curation Optimizer")
        st.caption(
            "Automatically find the best next assets for your portfolio. "
            "The optimizer evaluates all feasible (DS, RA, Cluster) combinations "
            "and ranks by balanced hedge score (EDC/IRC/DPC/LAC/CDC)."
        )

        opt_mode = st.radio("Optimizer Mode", [
            "Sequential (best next asset)",
            "Simultaneous (fill N slots)",
        ], key="opt_mode")

        oc1, oc2 = st.columns([1, 2])

        with oc1:
            opt_sims = st.number_input("Sims per candidate", 30, 500, 50, 10,
                                       key="opt_sims",
                                       help="More sims = more accurate but slower. 50 is good for exploration, 200+ for decisions.")
            if opt_mode == "Sequential (best next asset)":
                opt_top_n = st.number_input("Top N candidates", 1, 10, 3, 1, key="opt_top_n")
            else:
                opt_slots = st.number_input("Slots to fill", 1, 8, 5, 1, key="opt_slots")

            opt_exclude = st.checkbox("Exclude existing clusters", value=True, key="opt_exclude",
                                      help="Avoid mechanism/indication clusters already in portfolio for diversification")
            opt_run = st.button("▶ Run Optimizer", key="opt_run_btn")

        with oc2:
            if opt_run:
                from src.curation import optimize_sequential, optimize_simultaneous, generate_sourcing_spec

                # Build current portfolio from enabled sandbox assets
                enabled = [a for a in sb_portfolio if a["_enabled"]]
                if not enabled:
                    st.error("No assets enabled in sandbox portfolio.")
                else:
                    sb_as = pd.DataFrame([{k: v for k, v in a.items() if not k.startswith("_")} for a in enabled])
                    # Build tranches for sandbox portfolio
                    sb_tr_list = []
                    for a in enabled:
                        aid = a["Asset_ID"]
                        if a["_source"] == "portfolio":
                            asset_tr = tranches[tranches["Asset_ID"] == aid]
                            sb_tr_list.append(asset_tr)
                        else:
                            budget = a.get("_budget", 5_000_000)
                            entry = int(a.get("Entry_Month", 3))
                            sb_tr_list.append(pd.DataFrame([
                                {"Asset_ID": aid, "Tranche_ID": "T1", "Purpose": "IND-Enabling Development",
                                 "Budget": budget * 0.75, "Start_Month": entry, "Stop_Month": entry + 14, "Status": "Planned"},
                                {"Asset_ID": aid, "Tranche_ID": "T2", "Purpose": "Transaction / BD",
                                 "Budget": budget * 0.25, "Start_Month": entry + 10, "Stop_Month": entry + 18, "Status": "Planned"},
                            ]))
                    sb_tr = pd.concat(sb_tr_list, ignore_index=True) if sb_tr_list else pd.DataFrame()

                    if opt_mode == "Sequential (best next asset)":
                        with st.spinner(f"Evaluating candidates ({opt_sims} sims each)... this may take a minute"):
                            ranked_df, _eval_results = optimize_sequential(
                                sb_as, sb_tr, params,
                                n_sims=int(opt_sims), seed=42,
                                top_n=int(opt_top_n),
                                exclude_locked_clusters=opt_exclude,
                                annual_overhead=float(overhead),
                            )

                        log_action(DATA_DIR, st.session_state.get("user_email", "unknown"),
                            "run_optimizer_sequential", {"n_sims": int(opt_sims), "top_n": int(opt_top_n)})

                        if len(ranked_df) > 0:
                            st.markdown(f"**Top {len(ranked_df)} Candidates**")
                            for j, (_, cand) in enumerate(ranked_df.iterrows()):
                                cand_id = cand.get("Candidate_ID", f"CAND-{j}")
                                with st.expander(f"#{j+1}: {cand_id} — {cand['DS']}/{cand['RA']} "
                                                 f"{cand.get('MechCluster_ID', '')} / {cand.get('IndicationCluster_ID', '')} "
                                                 f"(Score: {cand.get('HedgeScore', 0):.3f})", expanded=(j == 0)):
                                    cc1, cc2 = st.columns(2)
                                    with cc1:
                                        st.markdown(f"**DS:** {cand['DS']}  \n**RA:** {cand['RA']}")
                                        st.markdown(f"**Mechanism:** {cand.get('MechCluster_ID', '?')}  \n"
                                                    f"**Indication:** {cand.get('IndicationCluster_ID', '?')}")
                                    with cc2:
                                        edc = float(cand.get('EDC', 0))
                                        irc = float(cand.get('IRC', 0))
                                        dpc = float(cand.get('DPC', 0))
                                        lac = float(cand.get('LAC', 0))
                                        cdc = float(cand.get('CDC', 0))
                                        st.markdown(f"**EDC:** {edc:+.1%}  \n"
                                                    f"**IRC:** {irc:+.2f}  \n"
                                                    f"**DPC:** {dpc:+.1%}  \n"
                                                    f"**LAC:** {lac:+.1f}  \n"
                                                    f"**CDC:** {cdc:+.3f}")

                                    # Sourcing spec
                                    spec = generate_sourcing_spec(cand.to_dict(), params, slot_number=j+1)
                                    st.code(spec, language="text")

                                    # Add to sandbox + submit to admission
                                    opt_btn1, opt_btn2 = st.columns(2)
                                    with opt_btn1:
                                        if st.button(f"Add {cand_id} to sandbox", key=f"add_opt_{j}"):
                                            new_asset = {
                                                "Asset_ID": cand_id,
                                                "DS_Current": cand["DS"],
                                                "RA_Current": cand["RA"],
                                                "Tier": "Tier-1",
                                                "Entry_Month": 3,
                                                "Equity_to_IP_Pct": 0.10,
                                                "AcqCash_to_IP": 0,
                                                "EarlyPassThrough_Pct": 0.0,
                                                "EarlyDeferredCash": 0,
                                                "MechCluster_ID": cand.get("MechCluster_ID", "MECH-NEW"),
                                                "IndicationCluster_ID": cand.get("IndicationCluster_ID", "IND-NEW"),
                                                "GeoRACluster_ID": cand.get("GeoRACluster_ID", "GEO-US1"),
                                                "_source": "optimizer",
                                                "_enabled": True,
                                                "_budget": 5_000_000,
                                            }
                                            sb_portfolio.append(new_asset)
                                            st.session_state.sb_portfolio = sb_portfolio
                                            st.success(f"Added {cand_id} to sandbox")
                                    with opt_btn2:
                                        if st.button(f"📋 Submit {cand_id} to Admission", key=f"submit_opt_{j}"):
                                            sub_asset = {
                                                "Asset_ID": cand_id,
                                                "DS_Current": cand["DS"],
                                                "RA_Current": cand["RA"],
                                                "Tier": "Tier-1",
                                                "Entry_Month": 3,
                                                "Equity_to_IP_Pct": 0.10,
                                                "AcqCash_to_IP": 0,
                                                "EarlyPassThrough_Pct": 0.0,
                                                "EarlyDeferredCash": 0,
                                                "MechCluster_ID": cand.get("MechCluster_ID", "MECH-NEW"),
                                                "IndicationCluster_ID": cand.get("IndicationCluster_ID", "IND-NEW"),
                                                "GeoRACluster_ID": cand.get("GeoRACluster_ID", "GEO-US1"),
                                                "_source": "optimizer",
                                                "_enabled": True,
                                                "_budget": 5_000_000,
                                                "_submitted_to_admission": True,
                                            }
                                            if "admission_queue" not in st.session_state:
                                                st.session_state.admission_queue = []
                                            eq = {a["Asset_ID"] for a in st.session_state.admission_queue}
                                            if cand_id not in eq:
                                                st.session_state.admission_queue.append(sub_asset)
                                                log_action(DATA_DIR, st.session_state.get("user_email", "unknown"),
                                                    "submit_to_admission", {"asset": cand_id, "source": "optimizer"})
                                                st.success(f"📋 {cand_id} submitted to Admission queue")
                                            else:
                                                st.info(f"{cand_id} already in queue")
                        else:
                            st.warning("No viable candidates found.")

                    else:  # Simultaneous
                        with st.spinner(f"Optimizing {opt_slots} slots ({opt_sims} sims each)... this may take several minutes"):
                            sim_result = optimize_simultaneous(
                                sb_as, sb_tr, params,
                                n_slots=int(opt_slots),
                                n_sims=int(opt_sims), seed=42,
                                exclude_locked_clusters=opt_exclude,
                                annual_overhead=float(overhead),
                            )

                        log_action(DATA_DIR, st.session_state.get("user_email", "unknown"),
                            "run_optimizer_simultaneous", {"n_sims": int(opt_sims), "n_slots": int(opt_slots)})

                        filled = sim_result.get("filled", [])
                        if filled:
                            st.markdown(f"**Optimized Portfolio ({len(filled)} slots filled)**")
                            for j, cand in enumerate(filled):
                                st.markdown(f"  {j+1}. **{cand['candidate_id']}** — {cand['ds']}/{cand['ra']} "
                                            f"{cand.get('mech', '')} / {cand.get('ind', '')} "
                                            f"(EDC: {cand.get('edc_delta', 0):+.1%})")

                            # Show convergence
                            conv_df = sim_result.get("convergence")
                            if conv_df is not None and len(conv_df) > 0:
                                st.markdown("**Convergence**")
                                conv_display = conv_df.copy()
                                for col in conv_display.columns:
                                    if "moic" in col.lower():
                                        conv_display[col] = conv_display[col].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "N/A")
                                    elif "irr" in col.lower() or "p_" in col.lower() or "exit" in col.lower():
                                        conv_display[col] = conv_display[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                                st.dataframe(conv_display, use_container_width=True, hide_index=True)

                            # Add all to sandbox button
                            if st.button("Add all optimized assets to sandbox", key="add_all_opt"):
                                for cand in filled:
                                    new_asset = {
                                        "Asset_ID": cand["candidate_id"],
                                        "DS_Current": cand["ds"],
                                        "RA_Current": cand["ra"],
                                        "Tier": "Tier-1",
                                        "Entry_Month": 3,
                                        "Equity_to_IP_Pct": 0.10,
                                        "AcqCash_to_IP": 0,
                                        "EarlyPassThrough_Pct": 0.0,
                                        "EarlyDeferredCash": 0,
                                        "MechCluster_ID": cand.get("mech", "MECH-NEW"),
                                        "IndicationCluster_ID": cand.get("ind", "IND-NEW"),
                                        "GeoRACluster_ID": cand.get("geo", "GEO-US1"),
                                        "_source": "optimizer",
                                        "_enabled": True,
                                        "_budget": 5_000_000,
                                    }
                                    sb_portfolio.append(new_asset)
                                st.session_state.sb_portfolio = sb_portfolio
                                st.success(f"Added {len(filled)} assets to sandbox")
                        else:
                            st.warning("Optimizer found no viable compositions.")

    # ══════════════════════════════════════════════════════════════════════
    # MODE 3: RUN ENGINE
    # ══════════════════════════════════════════════════════════════════════
    elif wb_mode == "📊 Run Engine":
        st.markdown("### Run Full Engine on Sandbox Portfolio")
        st.caption("Pre-flight checks, MC simulation, envelope gates, distributions, stress, and hedge — "
                   "all on the composed sandbox portfolio.")

        # Build sandbox portfolio DataFrames
        enabled = [a for a in sb_portfolio if a["_enabled"]]
        if not enabled:
            st.warning("No assets enabled. Go to Manual Build to toggle assets on.")
        else:
            # Build asset_state
            sb_as = pd.DataFrame([{k: v for k, v in a.items() if not k.startswith("_")} for a in enabled])

            # Build tranches
            sb_tr_list = []
            for a in enabled:
                aid = a["Asset_ID"]
                if a["_source"] == "portfolio":
                    asset_tr = tranches[tranches["Asset_ID"] == aid]
                    sb_tr_list.append(asset_tr)
                else:
                    budget = a.get("_budget", 5_000_000)
                    entry = int(a.get("Entry_Month", 3))
                    sb_tr_list.append(pd.DataFrame([
                        {"Asset_ID": aid, "Tranche_ID": "T1", "Purpose": "IND-Enabling Development",
                         "Budget": budget * 0.75, "Start_Month": entry, "Stop_Month": entry + 14, "Status": "Planned"},
                        {"Asset_ID": aid, "Tranche_ID": "T2", "Purpose": "Transaction / BD",
                         "Budget": budget * 0.25, "Start_Month": entry + 10, "Stop_Month": entry + 18, "Status": "Planned"},
                    ]))
            sb_tr = pd.concat(sb_tr_list, ignore_index=True)

            # ── Pre-Flight Checks on Sandbox Portfolio ──
            st.markdown("#### Pre-Flight Checks (Sandbox)")
            spf1, spf2, spf3, spf4 = st.columns(4)

            sb_param_issues = []
            try:
                sb_param_issues = validate_params(params)
            except Exception:
                pass
            with spf1:
                if not sb_param_issues:
                    st.success("Parameters OK")
                else:
                    st.error(f"Params: {len(sb_param_issues)} issue(s)")

            sb_activation = check_activation_requirements(sb_as, params["ds_ra_map"])
            with spf2:
                if sb_activation["activated"]:
                    st.success("Activation OK")
                else:
                    st.warning(f"Activation: {len(sb_activation['issues'])} issue(s)")

            sb_conc = check_capital_concentration(sb_tr)
            with spf3:
                if sb_conc["breaches"]:
                    st.error(f"Concentration: {len(sb_conc['breaches'])} breach(es)")
                elif sb_conc["warnings"]:
                    st.warning(f"Concentration: {len(sb_conc['warnings'])} warning(s)")
                else:
                    st.success("Concentration OK")

            sb_comb = check_combined_probability(sb_as, params)
            with spf4:
                if sb_comb["all_pass"]:
                    st.success("Combined Prob OK")
                else:
                    st.warning(f"Combined Prob: {len(sb_comb['failures'])} below 45%")

            with st.expander("Pre-Flight Details (Sandbox)"):
                if sb_param_issues:
                    for issue in sb_param_issues:
                        st.write(f"- {issue}")
                if not sb_activation["activated"]:
                    for issue in sb_activation["issues"]:
                        st.write(f"- {issue}")
                if sb_conc["breaches"] or sb_conc["warnings"]:
                    st.dataframe(pd.DataFrame([
                        {"Asset_ID": k, "Concentration": f"{v:.1%}",
                         "Status": "BREACH" if v > 0.20 else ("WARNING" if v > 0.15 else "OK")}
                        for k, v in sb_conc["concentrations"].items()
                    ]), use_container_width=True, hide_index=True)
                if not sb_comb["all_pass"]:
                    st.dataframe(pd.DataFrame(sb_comb["failures"]), use_container_width=True, hide_index=True)

            # ── Composition table ──
            st.markdown(f"**Sandbox Portfolio: {len(enabled)} assets**")
            roster_rows = []
            for a in enabled:
                roster_rows.append({
                    "Asset_ID": a["Asset_ID"],
                    "DS": a["DS_Current"],
                    "RA": a["RA_Current"],
                    "Mechanism": a.get("MechCluster_ID", "—"),
                    "Indication": a.get("IndicationCluster_ID", "—"),
                    "Source": a["_source"],
                })
            st.dataframe(pd.DataFrame(roster_rows), use_container_width=True, hide_index=True)

            # ── CRO / Pharma Channel for Sandbox ──
            sb_channel_lookup = None
            sb_enable_ch = st.checkbox("Enable CRO/Pharma Channel (Sandbox)", value=False, key="sb_ch_toggle",
                                       help="Assign CRO and pharma targets to sandbox assets")
            if sb_enable_ch:
                try:
                    from src.channel import load_cro_master as _lcro, load_pharma_master as _lph, build_channel_lookup as _bcl

                    _sb_cro = _lcro(cro_path_str)
                    _sb_pharma = _lph(pharma_path_str)
                    cro_opts = ["None"] + list(_sb_cro["cro_lookup"].keys())
                    cro_labels = {"None": "No CRO"}
                    for cid, ci in _sb_cro["cro_lookup"].items():
                        cro_labels[cid] = f"{cid} — {ci['name']}"
                    pharma_opts = list(_sb_pharma["pharma_lookup"].keys())
                    pharma_labels = {}
                    for pid, pi in _sb_pharma["pharma_lookup"].items():
                        pharma_labels[pid] = f"{pid} — {pi['name']}"

                    with st.expander("Sandbox Channel Pathways", expanded=False):
                        for idx_a, a in enumerate(enabled):
                            aid = a["Asset_ID"]
                            ac1, ac2 = st.columns(2)
                            with ac1:
                                cur_cro = a.get("CRO_ID", "None")
                                if pd.isna(cur_cro) if isinstance(cur_cro, float) else not cur_cro:
                                    cur_cro = "None"
                                def_idx = cro_opts.index(cur_cro) if cur_cro in cro_opts else 0
                                sel_cro = st.selectbox(f"{aid} CRO", cro_opts, index=def_idx,
                                    format_func=lambda x: cro_labels.get(x, x), key=f"sb_ch_cro_{idx_a}")
                            with ac2:
                                cur_ph = str(a.get("Target_Pharma_IDs", ""))
                                cur_ph_list = [x.strip() for x in cur_ph.split(",") if x.strip()] if cur_ph else []
                                def_ph = [p for p in cur_ph_list if p in pharma_opts]
                                sel_ph = st.multiselect(f"{aid} Pharmas", pharma_opts, default=def_ph,
                                    format_func=lambda x: pharma_labels.get(x, x), key=f"sb_ch_ph_{idx_a}")

                            # Update the sandbox asset in-place
                            sb_portfolio_idx = next(i for i, x in enumerate(sb_portfolio) if x["Asset_ID"] == aid and x["_enabled"])
                            if sel_cro != "None":
                                sb_portfolio[sb_portfolio_idx]["CRO_ID"] = sel_cro
                            else:
                                sb_portfolio[sb_portfolio_idx].pop("CRO_ID", None)
                            if sel_ph:
                                sb_portfolio[sb_portfolio_idx]["Target_Pharma_IDs"] = ",".join(sel_ph)
                            else:
                                sb_portfolio[sb_portfolio_idx].pop("Target_Pharma_IDs", None)

                    # Rebuild sb_as with updated CRO/Pharma
                    sb_as = pd.DataFrame([{k: v for k, v in a.items() if not k.startswith("_")} for a in enabled])
                    sb_channel_lookup = _bcl(sb_as, _sb_cro, _sb_pharma, avl_confirmed_only=False)

                    # Show channel summary
                    from src.channel import summarize_channel as _sch
                    ch_active = {k: v for k, v in sb_channel_lookup.items() if v.get("cro_id")}
                    if ch_active:
                        st.caption(f"Channel active on {len(ch_active)} asset(s)")
                except Exception as e:
                    st.warning(f"Channel setup failed: {e}")

            # ── Run controls ──
            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1:
                sb_engine_sims = st.number_input("Simulations", 200, 50000, 1000, 100, key="sb_engine_sims")
            with rc2:
                sb_engine_stress = st.checkbox("Include Stress Tests", value=False, key="sb_engine_stress")
            with rc3:
                sb_engine_hedge = st.checkbox("Include Hedge Analysis", value=False, key="sb_engine_hedge")
            with rc4:
                sb_view_mode = st.radio("View", ["Sandbox Only", "Side-by-Side"], key="sb_view_mode",
                                        help="Side-by-Side runs the production portfolio in parallel for comparison")

            sb_engine_run = st.button("▶ Run Sandbox Engine", type="primary", key="sb_engine_run_btn",
                                      use_container_width=True)

            if sb_engine_run:
                from src.sandbox import run_sandbox, compare_scenarios

                # ── Run sandbox MC ──
                with st.spinner(f"Running {sb_engine_sims:,} sims on sandbox portfolio..."):
                    sb_result = run_sandbox(
                        sb_as, sb_tr, params, envelope,
                        scenario={"label": f"Sandbox ({len(enabled)} assets)"},
                        n_sims=int(sb_engine_sims), seed=42,
                        channel_lookup=sb_channel_lookup,
                    )

                # ── Run production MC if side-by-side ──
                prod_result = None
                if sb_view_mode == "Side-by-Side":
                    with st.spinner("Running production baseline for comparison..."):
                        prod_result = run_sandbox(
                            asset_state, tranches, params, envelope,
                            scenario={"label": "Production"},
                            n_sims=int(sb_engine_sims), seed=42,
                        )

                log_action(DATA_DIR, st.session_state.get("user_email", "unknown"),
                    "run_sandbox_engine", {"n_assets": len(enabled), "n_sims": int(sb_engine_sims),
                                           "side_by_side": sb_view_mode == "Side-by-Side"})

                if sb_result.get("summary"):
                    sb_summary = sb_result["summary"]
                    sb_signal = sb_result.get("signal", portfolio_quality_signal(sb_summary, envelope))

                    # ── Quality Signal ──
                    if sb_view_mode == "Side-by-Side" and prod_result and prod_result.get("summary"):
                        prod_summary = prod_result["summary"]
                        prod_signal = portfolio_quality_signal(prod_summary, envelope)
                        qs1, qs2 = st.columns(2)
                        with qs1:
                            st.markdown(
                                f'<div style="background:{prod_signal["bg"]};border-left:8px solid {prod_signal["border"]};'
                                f'padding:14px;border-radius:8px;">'
                                f'<div style="font-size:20px;font-weight:700;color:{prod_signal["text"]};">'
                                f'Production: {prod_signal["label"]}</div>'
                                f'<div style="font-size:14px;color:{prod_signal["text"]};">'
                                f'{prod_signal["message"]}</div></div>', unsafe_allow_html=True)
                        with qs2:
                            st.markdown(
                                f'<div style="background:{sb_signal["bg"]};border-left:8px solid {sb_signal["border"]};'
                                f'padding:14px;border-radius:8px;">'
                                f'<div style="font-size:20px;font-weight:700;color:{sb_signal["text"]};">'
                                f'Sandbox: {sb_signal["label"]}</div>'
                                f'<div style="font-size:14px;color:{sb_signal["text"]};">'
                                f'{sb_signal["message"]}</div></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f'<div style="background-color:{sb_signal["bg"]};border-left:10px solid {sb_signal["border"]};'
                            f'padding:18px;border-radius:8px;margin:12px 0 18px 0;">'
                            f'<div style="font-size:24px;font-weight:700;color:{sb_signal["text"]};">'
                            f'Sandbox Quality: {sb_signal["label"]}</div>'
                            f'<div style="font-size:16px;color:{sb_signal["text"]};margin-top:6px;">'
                            f'{sb_signal["message"]}</div></div>', unsafe_allow_html=True)

                    # ── Metrics (side-by-side or sandbox only) ──
                    def _fmt_irr(v):
                        return f"{v:.1%}" if pd.notna(v) else "N/A"

                    if sb_view_mode == "Side-by-Side" and prod_result and prod_result.get("summary"):
                        ps = prod_summary
                        ss = sb_summary
                        st.markdown("#### Key Metrics Comparison")
                        metric_comp = pd.DataFrame([
                            {"Metric": "Median MOIC", "Production": f"{ps['Median_MOIC']:.2f}x", "Sandbox": f"{ss['Median_MOIC']:.2f}x",
                             "Delta": f"{ss['Median_MOIC'] - ps['Median_MOIC']:+.2f}x"},
                            {"Metric": "MOIC P10", "Production": f"{ps['MOIC_P10']:.2f}x", "Sandbox": f"{ss['MOIC_P10']:.2f}x",
                             "Delta": f"{ss['MOIC_P10'] - ps['MOIC_P10']:+.2f}x"},
                            {"Metric": "Median IRR", "Production": _fmt_irr(ps.get('Median_Annual_IRR')),
                             "Sandbox": _fmt_irr(ss.get('Median_Annual_IRR')), "Delta": "—"},
                            {"Metric": "P(0 exits)", "Production": f"{ps['P_Zero_Exits']:.1%}", "Sandbox": f"{ss['P_Zero_Exits']:.1%}",
                             "Delta": f"{ss['P_Zero_Exits'] - ps['P_Zero_Exits']:+.1%}"},
                            {"Metric": "P(≤1 exit)", "Production": f"{ps['P_Exits_LE1']:.1%}", "Sandbox": f"{ss['P_Exits_LE1']:.1%}",
                             "Delta": f"{ss['P_Exits_LE1'] - ps['P_Exits_LE1']:+.1%}"},
                            {"Metric": "P(3+ exits)", "Production": f"{ps['P_ThreePlus_Exits']:.1%}", "Sandbox": f"{ss['P_ThreePlus_Exits']:.1%}",
                             "Delta": f"{ss['P_ThreePlus_Exits'] - ps['P_ThreePlus_Exits']:+.1%}"},
                        ])
                        st.dataframe(metric_comp, use_container_width=True, hide_index=True)
                    else:
                        m1, m2, m3, m4 = st.columns(4)
                        with m1: st.metric("Median MOIC", f"{sb_summary['Median_MOIC']:.2f}x")
                        with m2: st.metric("MOIC P10", f"{sb_summary['MOIC_P10']:.2f}x")
                        with m3: st.metric("Median IRR", _fmt_irr(sb_summary.get("Median_Annual_IRR")))
                        with m4: st.metric("IRR P10", _fmt_irr(sb_summary.get("IRR_P10")))

                        m5, m6, m7, m8 = st.columns(4)
                        with m5: st.metric("P(0 exits)", f"{sb_summary['P_Zero_Exits']:.1%}")
                        with m6: st.metric("P(1 exit)", f"{sb_summary['P_One_Exit']:.1%}")
                        with m7: st.metric("P(2 exits)", f"{sb_summary['P_Two_Exits']:.1%}")
                        with m8: st.metric("P(3+ exits)", f"{sb_summary['P_ThreePlus_Exits']:.1%}")

                    # ── Envelope Gates ──
                    sb_env = sb_result.get("envelope", {})
                    if sb_env:
                        with st.expander("Envelope Gate Checks", expanded=True):
                            gate_rows = []
                            for name, check in sb_env.get("checks", {}).items():
                                gate_rows.append({
                                    "Gate": name,
                                    "Status": "✅ PASS" if check["pass"] else "❌ FAIL",
                                    "Actual": f"{check['actual']:.4f}" if isinstance(check["actual"], float) else str(check["actual"]),
                                    "Threshold": str(check["threshold"]),
                                })
                            st.dataframe(pd.DataFrame(gate_rows), use_container_width=True, hide_index=True)

                    # ── Distribution Charts ──
                    sb_rdf = sb_result.get("results_df")
                    prod_rdf = prod_result.get("results_df") if prod_result else None

                    if sb_view_mode == "Side-by-Side" and prod_rdf is not None and sb_rdf is not None:
                        # Exit count — overlaid red/blue
                        st.markdown("#### Exit Count Distribution")
                        ec_prod = prod_rdf["Num_Exits"].value_counts().sort_index()
                        ec_sb = sb_rdf["Num_Exits"].value_counts().sort_index()
                        all_exits = sorted(set(ec_prod.index) | set(ec_sb.index))

                        fig_ec = go.Figure()
                        fig_ec.add_trace(go.Bar(
                            name="Production", x=[str(e) for e in all_exits],
                            y=[ec_prod.get(e, 0) for e in all_exits],
                            marker_color="#E53E3E", opacity=0.75,
                            hovertemplate="Exits: %{x}<br>Production: %{y:,}<extra></extra>",
                        ))
                        fig_ec.add_trace(go.Bar(
                            name="Sandbox", x=[str(e) for e in all_exits],
                            y=[ec_sb.get(e, 0) for e in all_exits],
                            marker_color="#3182CE", opacity=0.75,
                            hovertemplate="Exits: %{x}<br>Sandbox: %{y:,}<extra></extra>",
                        ))
                        fig_ec.update_layout(
                            barmode="group", xaxis_title="Exit Count", yaxis_title="Simulations",
                            height=380, margin=dict(l=40, r=20, t=30, b=40),
                            plot_bgcolor="rgba(0,0,0,0)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        )
                        st.plotly_chart(fig_ec, use_container_width=True)

                        # MOIC — overlaid histograms
                        st.markdown("#### MOIC Distribution")
                        vm_prod = prod_rdf["MOIC"].dropna()
                        vm_sb = sb_rdf["MOIC"].dropna()
                        if len(vm_prod) > 0 and len(vm_sb) > 0:
                            fig_moic = go.Figure()
                            fig_moic.add_trace(go.Histogram(
                                x=vm_prod, name="Production", nbinsx=25,
                                marker_color="#E53E3E", opacity=0.6,
                                hovertemplate="MOIC: %{x:.1f}x<br>Production: %{y}<extra></extra>",
                            ))
                            fig_moic.add_trace(go.Histogram(
                                x=vm_sb, name="Sandbox", nbinsx=25,
                                marker_color="#3182CE", opacity=0.6,
                                hovertemplate="MOIC: %{x:.1f}x<br>Sandbox: %{y}<extra></extra>",
                            ))
                            _add_median_line(fig_moic, vm_prod.median(), f"Prod: {vm_prod.median():.2f}x", "#E53E3E")
                            _add_median_line(fig_moic, vm_sb.median(), f"SB: {vm_sb.median():.2f}x", "#3182CE")
                            fig_moic.update_layout(
                                barmode="overlay", xaxis_title="MOIC (x)", yaxis_title="Simulations",
                                height=380, margin=dict(l=40, r=20, t=30, b=40),
                                plot_bgcolor="rgba(0,0,0,0)",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            )
                            st.plotly_chart(fig_moic, use_container_width=True)

                        # Delta summary
                        deltas = compare_scenarios(prod_result, sb_result)
                        comp_rows = []
                        for key, d in deltas.items():
                            if key.startswith("_"): continue
                            unit = d["unit"]
                            if unit == "x":
                                b_s = f"{d['base']:.2f}x" if pd.notna(d["base"]) else "N/A"
                                s_s = f"{d['sandbox']:.2f}x" if pd.notna(d["sandbox"]) else "N/A"
                                d_s = f"{d['delta']:+.2f}x" if pd.notna(d["delta"]) else "—"
                            else:
                                b_s = f"{d['base']:.1%}" if pd.notna(d["base"]) else "N/A"
                                s_s = f"{d['sandbox']:.1%}" if pd.notna(d["sandbox"]) else "N/A"
                                d_s = f"{d['delta']:+.1%}" if pd.notna(d["delta"]) else "—"
                            icon = "🟢" if d["direction"] == "better" else ("🔴" if d["direction"] == "worse" else "⚪")
                            comp_rows.append({"Metric": key,
                                "Production 🔴": b_s, "Sandbox 🔵": s_s, "Delta": d_s, "": icon})
                        with st.expander("Full Comparison Table", expanded=False):
                            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

                    elif sb_rdf is not None:
                        # Sandbox only charts
                        col_l, col_r = st.columns(2)
                        with col_l:
                            st.markdown("**Exit Count Distribution**")
                            exit_counts = sb_rdf["Num_Exits"].value_counts().sort_index()
                            fig_ec_sb = go.Figure(go.Bar(
                                x=exit_counts.index.astype(str), y=exit_counts.values,
                                marker_color="#3182CE",
                                hovertemplate="Exits: %{x}<br>Sims: %{y:,}<extra></extra>",
                            ))
                            fig_ec_sb.update_layout(
                                xaxis_title="Exit Count", yaxis_title="Simulations",
                                height=320, margin=dict(l=40, r=20, t=20, b=40),
                                plot_bgcolor="rgba(0,0,0,0)",
                            )
                            st.plotly_chart(fig_ec_sb, use_container_width=True)
                        with col_r:
                            st.markdown("**MOIC Distribution**")
                            valid_moic = sb_rdf["MOIC"].dropna()
                            if len(valid_moic):
                                fig_moic_sb = go.Figure(go.Histogram(
                                    x=valid_moic, nbinsx=25,
                                    marker_color="#3182CE", opacity=0.85,
                                    hovertemplate="MOIC: %{x:.1f}x<br>Count: %{y}<extra></extra>",
                                ))
                                _add_median_line(fig_moic_sb, valid_moic.median(), f"Median: {valid_moic.median():.2f}x")
                                fig_moic_sb.update_layout(
                                    xaxis_title="MOIC (x)", yaxis_title="Simulations",
                                    height=320, margin=dict(l=40, r=20, t=20, b=40),
                                    plot_bgcolor="rgba(0,0,0,0)",
                                )
                                st.plotly_chart(fig_moic_sb, use_container_width=True)

                    # ── Hedge Analysis ──
                    if sb_engine_hedge and len(enabled) >= 2:
                        with st.expander("Marginal Contribution (Leave-One-Out)", expanded=False):
                            from src.optimization.hedge import compute_marginal_contribution
                            with st.spinner("Computing marginal contributions..."):
                                mc_df = compute_marginal_contribution(
                                    sb_as, sb_tr, params,
                                    n_sims=min(int(sb_engine_sims), 500), seed=42,
                                    annual_overhead=float(overhead),
                                )
                            st.dataframe(mc_df, use_container_width=True, hide_index=True)

                    # ── Stress Tests ──
                    if sb_engine_stress:
                        with st.expander("Stress Test Results", expanded=False):
                            from src.simulation.stress import run_stress_suite
                            with st.spinner("Running 4 stress scenarios..."):
                                stress = run_stress_suite(
                                    sb_as, sb_tr, params, envelope,
                                    n_sims=min(int(sb_engine_sims), 1000),
                                    annual_overhead=float(overhead),
                                )
                            scenarios = list(stress.keys())
                            stress_rows = []
                            for s in scenarios:
                                ss = stress[s]["summary"]
                                stress_rows.append({
                                    "Scenario": s,
                                    "Median MOIC": f"{ss['Median_MOIC']:.2f}x",
                                    "P(3+ exits)": f"{ss['P_ThreePlus_Exits']:.1%}",
                                    "P(≤1 exit)": f"{ss['P_Exits_LE1']:.1%}",
                                    "Envelope": "✅ PASS" if stress[s]["all_pass"] else "❌ FAIL",
                                })
                            st.dataframe(pd.DataFrame(stress_rows), use_container_width=True, hide_index=True)

                else:
                    st.error("Sandbox engine failed. Check asset configuration.")

    # ══════════════════════════════════════════════════════════════════════
    # MODE: ADMISSION
    # ══════════════════════════════════════════════════════════════════════
    elif wb_mode == "🚪 Admission":
        st.markdown("### Asset Admission (7-Gate Check)")

        # ── Admission Queue ──
        if "admission_queue" not in st.session_state:
            st.session_state.admission_queue = []
        adm_queue = st.session_state.admission_queue

        if adm_queue:
            st.markdown("#### 📋 Admission Queue — Pending Review")
            st.caption("Assets submitted from the sandbox for formal admission review. "
                       "Run the 7-gate check, then approve or reject.")

            for qi, qa in enumerate(adm_queue):
                q_status = qa.get("_admission_status", "Pending")
                if q_status == "Pending":
                    icon = "⏳"
                elif q_status == "Approved":
                    icon = "✅"
                else:
                    icon = "❌"

                with st.expander(f"{icon} {qa['Asset_ID']} — {qa['DS_Current']}/{qa['RA_Current']} "
                                 f"{qa.get('MechCluster_ID', '')} / {qa.get('IndicationCluster_ID', '')} "
                                 f"[{q_status}]", expanded=(q_status == "Pending")):
                    qd1, qd2, qd3 = st.columns(3)
                    with qd1:
                        st.markdown(f"**DS:** {qa['DS_Current']} | **RA:** {qa['RA_Current']} | **Tier:** {qa.get('Tier', 'Tier-1')}")
                    with qd2:
                        st.markdown(f"**Budget:** ${qa.get('_budget', 0):,.0f} | **Equity:** {qa.get('Equity_to_IP_Pct', 0.1):.0%}")
                    with qd3:
                        st.markdown(f"**Source:** {qa.get('_source', '?')} | **Entry Mo:** {qa.get('Entry_Month', '?')}")

                    if q_status == "Pending":
                        qr1, qr2, qr3, qr4 = st.columns(4)
                        with qr1:
                            q_sims = st.number_input("Sims", 200, 10000, 500, 100, key=f"q_sims_{qi}")
                        with qr2:
                            if st.button("▶ Run 7-Gate Check", key=f"q_check_{qi}"):
                                from src.optimization.contribution import run_contribution_analysis
                                from src.governance.admission import check_admission_gates
                                xa = pd.DataFrame([{k: v for k, v in qa.items() if not k.startswith("_")}])
                                b = qa.get("_budget", 5e6); e = int(qa.get("Entry_Month", 3))
                                xt = pd.DataFrame([
                                    {"Asset_ID": qa["Asset_ID"], "Tranche_ID": "T1", "Purpose": "Dev",
                                     "Budget": b*0.75, "Start_Month": e, "Stop_Month": e+14, "Status": "Planned"},
                                    {"Asset_ID": qa["Asset_ID"], "Tranche_ID": "T2", "Purpose": "BD",
                                     "Budget": b*0.25, "Start_Month": e+10, "Stop_Month": e+18, "Status": "Planned"},
                                ])
                                with st.spinner(f"Running admission for {qa['Asset_ID']}..."):
                                    contrib = run_contribution_analysis(asset_state, tranches, xa, xt, params, n_sims=int(q_sims))
                                    adm = check_admission_gates(contrib, envelope)
                                adm_queue[qi]["_last_admission_result"] = adm
                                adm_queue[qi]["_last_contrib"] = {k: v for k, v in contrib.items() if isinstance(v, (int, float, str, bool))}
                                st.session_state.admission_queue = adm_queue
                                if adm.get("all_pass"):
                                    st.success(f"✅ {qa['Asset_ID']} PASSES all gates")
                                else:
                                    st.error(f"❌ FAILS: {', '.join(adm.get('failed_gates', []))}")
                                gr = [{"Gate": n, "Status": "✅" if c.get("pass") else "❌", "Detail": str(c.get("detail",""))}
                                      for n, c in adm.get("gates", {}).items()]
                                if gr:
                                    st.dataframe(pd.DataFrame(gr), use_container_width=True, hide_index=True)

                        with qr3:
                            if st.button("✅ Approve", key=f"q_approve_{qi}", type="primary"):
                                if not require_permission("write", silent=True):
                                    st.warning("Approval requires write permission.")
                                else:
                                    adm_queue[qi]["_admission_status"] = "Approved"
                                    st.session_state.admission_queue = adm_queue
                                    log_action(DATA_DIR, st.session_state.get("user_email", "unknown"),
                                        "approve_admission", {"asset": qa["Asset_ID"]})
                                    st.success(f"✅ {qa['Asset_ID']} approved for portfolio addition")
                                    st.rerun()
                        with qr4:
                            if st.button("❌ Reject", key=f"q_reject_{qi}"):
                                adm_queue[qi]["_admission_status"] = "Rejected"
                                st.session_state.admission_queue = adm_queue
                                log_action(DATA_DIR, st.session_state.get("user_email", "unknown"),
                                    "reject_admission", {"asset": qa["Asset_ID"]})
                                st.rerun()

                    # Show last result if available
                    last_contrib = qa.get("_last_contrib")
                    if last_contrib:
                        h1, h2, h3, h4, h5 = st.columns(5)
                        with h1: st.metric("EDC", f"{last_contrib.get('EDC', 0):+.1%}" if isinstance(last_contrib.get('EDC'), float) else "—")
                        with h2: st.metric("IRC", f"{last_contrib.get('IRC', 0):+.2f}" if isinstance(last_contrib.get('IRC'), float) else "—")
                        with h3: st.metric("DPC", f"{last_contrib.get('DPC', 0):+.1%}" if isinstance(last_contrib.get('DPC'), float) else "—")
                        with h4: st.metric("LAC", f"{last_contrib.get('LAC', 0):+.1f}" if isinstance(last_contrib.get('LAC'), (int, float)) else "—")
                        with h5: st.metric("CDC", f"{last_contrib.get('CDC', 0):+.3f}" if isinstance(last_contrib.get('CDC'), float) else "—")

            # Queue summary
            pending = sum(1 for a in adm_queue if a.get("_admission_status", "Pending") == "Pending")
            approved = sum(1 for a in adm_queue if a.get("_admission_status") == "Approved")
            rejected = sum(1 for a in adm_queue if a.get("_admission_status") == "Rejected")
            qs1, qs2, qs3 = st.columns(3)
            with qs1: st.metric("Pending", pending)
            with qs2: st.metric("Approved", approved)
            with qs3: st.metric("Rejected", rejected)

            if st.button("Clear Completed (Approved + Rejected)", key="q_clear"):
                st.session_state.admission_queue = [a for a in adm_queue if a.get("_admission_status", "Pending") == "Pending"]
                st.rerun()

            st.markdown("---")

        st.caption("You can also test assets directly below without going through the queue.")

        adm_src = st.radio("Source", ["Test Existing", "New Candidate"], horizontal=True, key="sb_adm_src")
        sb_en = [a for a in sb_portfolio if a["_enabled"]]
        if sb_en:
            sb_a = pd.DataFrame([{k: v for k, v in a.items() if not k.startswith("_")} for a in sb_en])
            sb_t_list = []
            for a in sb_en:
                aid = a["Asset_ID"]
                if a.get("_source") == "portfolio":
                    sb_t_list.append(tranches[tranches["Asset_ID"] == aid])
                else:
                    b = a.get("_budget", 5e6); e = int(a.get("Entry_Month", 3))
                    sb_t_list.append(pd.DataFrame([
                        {"Asset_ID": aid, "Tranche_ID": "T1", "Purpose": "Dev", "Budget": b*0.75, "Start_Month": e, "Stop_Month": e+14, "Status": "Planned"},
                        {"Asset_ID": aid, "Tranche_ID": "T2", "Purpose": "BD", "Budget": b*0.25, "Start_Month": e+10, "Stop_Month": e+18, "Status": "Planned"},
                    ]))
            sb_t = pd.concat(sb_t_list, ignore_index=True)
        else:
            sb_a = asset_state; sb_t = tranches

        if adm_src == "Test Existing" and len(sb_a) >= 2:
            test_aid = st.selectbox("Asset", sb_a["Asset_ID"].tolist(), key="sb_adm_aid")
            adm_n = st.number_input("Sims", 200, 10000, 500, 100, key="sb_adm_n")
            if st.button("▶ Run Admission", key="sb_adm_go", use_container_width=True):
                from src.optimization.contribution import run_contribution_analysis
                from src.governance.admission import check_admission_gates
                ca = sb_a[sb_a["Asset_ID"]!=test_aid].reset_index(drop=True)
                ct = sb_t[sb_t["Asset_ID"]!=test_aid].reset_index(drop=True)
                xa = sb_a[sb_a["Asset_ID"]==test_aid].reset_index(drop=True)
                xt = sb_t[sb_t["Asset_ID"]==test_aid].reset_index(drop=True)
                with st.spinner(f"Evaluating {test_aid}..."):
                    contrib = run_contribution_analysis(ca, ct, xa, xt, params, n_sims=int(adm_n))
                    adm = check_admission_gates(contrib, envelope)
                if adm.get("all_pass"):
                    st.success(f"✅ {test_aid} PASSES all admission gates")
                else:
                    st.error(f"❌ {test_aid} FAILS: {', '.join(adm.get('failed_gates', []))}")
                gr = [{"Gate": n, "Status": "✅" if c.get("pass") else "❌", "Detail": str(c.get("detail",""))} for n,c in adm.get("gates",{}).items()]
                if gr: st.dataframe(pd.DataFrame(gr), use_container_width=True, hide_index=True)
                h1,h2,h3,h4,h5 = st.columns(5)
                with h1: st.metric("EDC", f"{contrib.get('EDC',0):+.1%}" if isinstance(contrib.get('EDC'),float) else "—")
                with h2: st.metric("IRC", f"{contrib.get('IRC',0):+.2f}" if isinstance(contrib.get('IRC'),float) else "—")
                with h3: st.metric("DPC", f"{contrib.get('DPC',0):+.1%}" if isinstance(contrib.get('DPC'),float) else "—")
                with h4: st.metric("LAC", f"{contrib.get('LAC',0):+.1f}" if isinstance(contrib.get('LAC'),(int,float)) else "—")
                with h5: st.metric("CDC", f"{contrib.get('CDC',0):+.3f}" if isinstance(contrib.get('CDC'),float) else "—")

        elif adm_src == "New Candidate":
            nc1, nc2, nc3 = st.columns(3)
            with nc1:
                nid = st.text_input("Asset ID", "A-NEW-001", key="sb_nc_id")
                nds = st.selectbox("DS", ["DS-3","DS-4","DS-5"], key="sb_nc_ds")
                nra = st.selectbox("RA", ["RA-1","RA-2"], key="sb_nc_ra")
            with nc2:
                nmech = st.text_input("Mechanism Cluster", "MECH-NEW", key="sb_nc_mech")
                nind = st.text_input("Indication Cluster", "IND-NEW", key="sb_nc_ind")
                ntier = st.selectbox("Tier", ["Tier-1","Tier-2"], key="sb_nc_tier")
            with nc3:
                nbgt = st.number_input("Budget ($)", 1e6, 15e6, 5e6, 5e5, key="sb_nc_bgt")
                neq = st.slider("Equity %", 0, 25, 10, 1, key="sb_nc_eq")
                nent = st.number_input("Entry Month", 0, 24, 3, 1, key="sb_nc_ent")
            nn = st.number_input("Sims", 200, 10000, 500, 100, key="sb_nc_n")
            if st.button("▶ Evaluate", type="primary", key="sb_nc_go", use_container_width=True):
                from src.optimization.contribution import run_contribution_analysis
                from src.governance.admission import check_admission_gates
                xa = pd.DataFrame([{"Asset_ID": nid, "DS_Current": nds, "RA_Current": nra, "Tier": ntier,
                    "Entry_Month": nent, "Equity_to_IP_Pct": neq/100, "AcqCash_to_IP": 0,
                    "EarlyPassThrough_Pct": 0, "EarlyDeferredCash": 0,
                    "MechCluster_ID": nmech, "IndicationCluster_ID": nind, "GeoRACluster_ID": "GEO-US1"}])
                xt = pd.DataFrame([
                    {"Asset_ID": nid, "Tranche_ID": "T1", "Purpose": "Dev", "Budget": nbgt*0.75, "Start_Month": nent, "Stop_Month": nent+14, "Status": "Planned"},
                    {"Asset_ID": nid, "Tranche_ID": "T2", "Purpose": "BD", "Budget": nbgt*0.25, "Start_Month": nent+10, "Stop_Month": nent+18, "Status": "Planned"},
                ])
                with st.spinner(f"Evaluating {nid}..."):
                    contrib = run_contribution_analysis(sb_a, sb_t, xa, xt, params, n_sims=int(nn))
                    adm = check_admission_gates(contrib, envelope)
                if adm.get("all_pass"):
                    st.success(f"✅ {nid} PASSES all admission gates")
                else:
                    st.error(f"❌ {nid} FAILS: {', '.join(adm.get('failed_gates', []))}")
                gr = [{"Gate": n, "Status": "✅" if c.get("pass") else "❌", "Detail": str(c.get("detail",""))} for n,c in adm.get("gates",{}).items()]
                if gr: st.dataframe(pd.DataFrame(gr), use_container_width=True, hide_index=True)
                h1,h2,h3,h4,h5 = st.columns(5)
                with h1: st.metric("EDC", f"{contrib.get('EDC',0):+.1%}" if isinstance(contrib.get('EDC'),float) else "—")
                with h2: st.metric("IRC", f"{contrib.get('IRC',0):+.2f}" if isinstance(contrib.get('IRC'),float) else "—")
                with h3: st.metric("DPC", f"{contrib.get('DPC',0):+.1%}" if isinstance(contrib.get('DPC'),float) else "—")
                with h4: st.metric("LAC", f"{contrib.get('LAC',0):+.1f}" if isinstance(contrib.get('LAC'),(int,float)) else "—")
                with h5: st.metric("CDC", f"{contrib.get('CDC',0):+.3f}" if isinstance(contrib.get('CDC'),float) else "—")
                # Diversification check
                emechs = set(sb_a.get("MechCluster_ID", pd.Series()).dropna())
                einds = set(sb_a.get("IndicationCluster_ID", pd.Series()).dropna())
                mn = nmech not in emechs; inn = nind not in einds
                if mn and inn: st.success(f"✅ Both mechanism and indication are NEW — max diversification")
                elif mn: st.success(f"✅ Mechanism is NEW. Indication overlaps.")
                elif inn: st.success(f"✅ Indication is NEW. Mechanism overlaps.")
                else: st.warning(f"⚠️ Both mechanism and indication overlap — limited diversification")

    # ══════════════════════════════════════════════════════════════════════
    # MODE: GOVERNANCE
    # ══════════════════════════════════════════════════════════════════════
    elif wb_mode == "⚖️ Governance":
        st.markdown("### Governance Controls")
        gov_sec = st.radio("Section", ["Board Override", "Monitoring", "Audit Log"], horizontal=True, key="sb_gov_sec")

        if gov_sec == "Board Override":
            st.caption("Requires Board role with ≥80% supermajority. Cannot override Max_Duration.")
            if not has_permission("override"):
                st.warning("Board Override requires Board (BoD Override) role.")
            else:
                from src.governance.override import create_override
                with st.form("sb_override_form"):
                    ov_gates = st.multiselect("Gates to Override",
                        ["Median_IRR","IRR_P10","P_Exits_GE3","P_Exits_LE1","Corr_Index","Weighted_Time","Capital_Concentration"], key="sb_ov_g")
                    ov_vote = st.text_input("Board Vote (e.g. '4/5 = 80%')", key="sb_ov_v")
                    ov_just = st.text_area("Justification", key="sb_ov_j")
                    ov_asset = st.text_input("Asset ID (optional)", key="sb_ov_a")
                    ov_cond = st.text_input("Conditions (optional)", key="sb_ov_c")
                    ov_sub = st.form_submit_button("Submit Override")
                if ov_sub:
                    if not ov_gates or not ov_vote or not ov_just:
                        st.warning("Fill in gates, vote, and justification.")
                    elif "Max_Duration" in ov_gates:
                        st.error("Max_Duration is non-overridable.")
                    else:
                        user = get_current_user()
                        override = create_override(failed_gates=ov_gates, board_vote=ov_vote,
                            justification=ov_just, approved_by=user["email"], asset_id=ov_asset or None, conditions=ov_cond)
                        log_action(DATA_DIR, user["email"], "board_override", {"gates": ov_gates, "vote": ov_vote})
                        st.success("Override recorded"); st.json(override)

        elif gov_sec == "Monitoring":
            try:
                cond_status = check_conditional_status(asset_state, params)
                if cond_status["has_conditional"]:
                    st.warning("Assets in conditional status:")
                    st.dataframe(pd.DataFrame(cond_status["flagged_assets"]), use_container_width=True, hide_index=True)
                else: st.success("No assets in conditional status.")
            except Exception as e: st.warning(f"Error: {e}")
            st.markdown("---")
            try:
                classifications = classify_month30_winddown(asset_state, params)
                st.markdown("**Month-30 Wind-Down Classification**")
                st.dataframe(pd.DataFrame(classifications), use_container_width=True, hide_index=True)
            except Exception as e: st.warning(f"Error: {e}")

        elif gov_sec == "Audit Log":
            from src.auth import read_audit_log
            logs = read_audit_log(DATA_DIR, last_n=50)
            if logs:
                log_df = pd.DataFrame(logs)
                dcols = [c for c in ["timestamp","user","action","details","state_hash"] if c in log_df.columns]
                st.dataframe(log_df[dcols], use_container_width=True, hide_index=True)
            else: st.info("No audit log entries yet.")


# ══════════════════════════════════════════════════════════════════════════
# TAB: SETTINGS
# ══════════════════════════════════════════════════════════════════════════

with tab_settings:
    st.subheader("⚙️ Settings & Administration")

    settings_section = st.radio("Section", [
        "📖 User Guide",
        "👤 Account",
        "📂 Data Sources",
        "🔧 Admin Panel",
        "📄 About",
    ], horizontal=True, key="settings_section")

    if settings_section == "📖 User Guide":
        st.markdown(USER_GUIDE)

    elif settings_section == "📂 Data Sources":
        st.markdown("### File Paths & Channel Configuration")
        st.caption("Changes here take effect on the next page reload.")

        st.markdown("**Parameter & State Files**")
        new_param = st.text_input("Parameters Workbook", st.session_state.param_path, key="set_param_path")
        new_state = st.text_input("Portfolio State Workbook", st.session_state.state_path, key="set_state_path")

        st.markdown("---")
        st.markdown("**Vehicle Configuration**")
        new_overhead = st.number_input("Annual Overhead ($)", 0, 10_000_000,
                                       int(st.session_state.annual_overhead), 100_000,
                                       key="set_overhead", help=TIPS["overhead"])

        st.markdown("---")
        st.markdown("**CRO / Pharma Channel**")
        new_channel = st.checkbox("Enable CRO/Pharma Channel", value=st.session_state.enable_channel,
                                  key="set_channel_toggle", help=TIPS["channel"])
        new_cro = st.text_input("CRO Master Workbook", st.session_state.cro_path, key="set_cro_path")
        new_pharma = st.text_input("Pharma Master Workbook", st.session_state.pharma_path, key="set_pharma_path")

        if st.button("💾 Save Data Source Settings", type="primary", use_container_width=True):
            st.session_state.param_path = new_param
            st.session_state.state_path = new_state
            st.session_state.annual_overhead = new_overhead
            st.session_state.enable_channel = new_channel
            st.session_state.cro_path = new_cro
            st.session_state.pharma_path = new_pharma
            st.success("Settings saved — reload to apply")
            st.rerun()

    elif settings_section == "👤 Account":
        st.markdown("### Account Information")
        user = get_current_user()
        ai1, ai2 = st.columns(2)
        with ai1:
            st.markdown(f"**Name:** {user.get('name', '—')}")
            st.markdown(f"**Email:** {user.get('email', '—')}")
            st.markdown(f"**Role:** {user.get('role', '—')}")
            st.markdown(f"**Authenticated:** {'Yes' if user.get('authenticated') else 'No'}")
        with ai2:
            st.markdown(f"**Login Time:** {st.session_state.get('login_time', 'N/A')}")
            st.markdown(f"**Auth Mode:** {'Demo (SKIP_AUTH)' if SKIP_AUTH else 'Production'}")
            login_hash = st.session_state.get("login_state_hash")
            st.markdown(f"**State Hash:** {login_hash[:16] + '...' if login_hash else 'N/A'}")

        if not SKIP_AUTH:
            st.divider()
            if st.button("Sign Out", key="settings_logout_btn"):
                logout(DATA_DIR)

    elif settings_section == "🔧 Admin Panel":
        st.markdown("### Administration")

        if not has_permission("admin"):
            st.warning("Admin panel requires admin or board role.")
        else:
            admin_sub = st.radio("Admin Section", [
                "User Management",
                "Audit Log",
                "System Status",
            ], horizontal=True, key="admin_sub")

            if admin_sub == "User Management":
                st.markdown("**Registered Users**")
                from src.auth import load_users
                users = load_users(DATA_DIR)
                user_rows = []
                for email, info in users.items():
                    user_rows.append({
                        "Email": email,
                        "Name": info.get("name", "—"),
                        "Role": info.get("role", "—"),
                    })
                st.dataframe(pd.DataFrame(user_rows), use_container_width=True, hide_index=True)

                st.markdown("**Add User**")
                with st.form("add_user_form"):
                    new_email = st.text_input("Email", key="admin_new_email")
                    new_name = st.text_input("Name", key="admin_new_name")
                    new_pw = st.text_input("Password", type="password", key="admin_new_pw")
                    new_role = st.selectbox("Role", ["viewer", "analyst", "board"], key="admin_new_role")
                    add_user_btn = st.form_submit_button("Add User")

                if add_user_btn:
                    if new_email and new_name and new_pw:
                        from src.auth import add_user
                        try:
                            add_user(DATA_DIR, new_email, new_name, new_pw, new_role)
                            log_action(DATA_DIR, st.session_state.get("user_email", "unknown"),
                                "admin_add_user", {"new_user": new_email, "role": new_role})
                            st.success(f"Added user {new_email}")
                        except Exception as e:
                            st.error(f"Error: {e}")
                    else:
                        st.warning("Fill in all fields")

            elif admin_sub == "Audit Log":
                st.markdown("**Activity Audit Log**")
                from src.auth import read_audit_log
                n_logs = st.number_input("Show last N entries", 10, 500, 50, 10, key="admin_log_n")
                logs = read_audit_log(DATA_DIR, last_n=int(n_logs))
                if logs:
                    log_df = pd.DataFrame(logs)
                    dcols = [c for c in ["timestamp", "user", "action", "details", "state_hash"]
                             if c in log_df.columns]
                    st.dataframe(log_df[dcols], use_container_width=True, hide_index=True)
                else:
                    st.info("No audit log entries yet.")

            elif admin_sub == "System Status":
                st.markdown("**Engine Configuration**")
                ss1, ss2 = st.columns(2)
                with ss1:
                    st.markdown(f"**Parameter File:** {param_path}")
                    st.markdown(f"**State File:** {state_path}")
                    st.markdown(f"**CRO Master:** {cro_path_str}")
                    st.markdown(f"**Pharma Master:** {pharma_path_str}")
                with ss2:
                    st.markdown(f"**Assets Loaded:** {len(asset_state)}")
                    st.markdown(f"**Tranches Loaded:** {len(tranches)}")
                    st.markdown(f"**Auth Mode:** {'SKIP_AUTH (Demo)' if SKIP_AUTH else 'Production'}")
                    st.markdown(f"**Channel:** {'Enabled' if channel_lookup else 'Disabled'}")

                st.markdown("**Governance Envelope Thresholds**")
                env_rows = [{"Parameter": k, "Value": str(v)} for k, v in envelope.items()]
                st.dataframe(pd.DataFrame(env_rows), use_container_width=True, hide_index=True)

    elif settings_section == "📄 About":
        st.markdown("""
### Discovery Portfolio Optimizer v1.0

**Purpose:** Monte Carlo governance engine for portfolio-level drug development 
modeling and optimization.

**Discovery Biotech** curates high-value novel drug assets and develops them 
for delivery to biopharma customers through wholly owned development companies. 
Private investors invest into the portfolio vehicle, not individual assets.

**Technical Stack:**
- Python (numpy, pandas, openpyxl, streamlit, plotly)
- Monte Carlo simulation with triangular distributions
- Factor-based latent normal correlation model
- 7-gate admission framework with 5-axis hedge profiling
- Milestone-based capital deployment with tranche kill

**Key Governance Rules:**
- 36-month hard stop (non-overridable)
- No capital recycling within a vehicle
- Median IRR ≥ 25%, IRR P10 ≥ 0%
- P(≥3 exits) ≥ 60%, P(≤1 exit) ≤ 15%
- Capital concentration ≤ 20% per asset
- Combined probability ≥ 45%
- Board override requires ≥80% supermajority

**CONFIDENTIAL** — Discovery Biotech
        """)

# ══════════════════════════════════════════════════════════════════════════
# CONFIDENTIALITY FOOTER
# ══════════════════════════════════════════════════════════════════════════
confidentiality_footer()


# ══════════════════════════════════════════════════════════════════════
# TAB: INTERACTIVE PORTFOLIO
# ══════════════════════════════════════════════════════════════════════
with tab_explorer:
    # BUTTON_COLORS_MARKER_V2 — sibling-selector button styling
    st.markdown("""
    <style>
    /* Hide the marker divs themselves */
    .explorer-add-marker,
    .explorer-remove-marker {
        display: none;
    }

    /* The marker div is a sibling of the button's wrapper div.
       Streamlit puts each element in its own block container, so we
       target the immediately-following stButton. */

    /* + Add buttons — slate blue */
    div:has(> .explorer-add-marker) + div[data-testid="stElementContainer"] button,
    div.element-container:has(.explorer-add-marker) + div.element-container button {
        background-color: #334155 !important;
        color: #ffffff !important;
        border: 1px solid #334155 !important;
        font-weight: 500 !important;
        transition: background-color 0.15s ease !important;
    }
    div:has(> .explorer-add-marker) + div[data-testid="stElementContainer"] button:hover,
    div.element-container:has(.explorer-add-marker) + div.element-container button:hover {
        background-color: #475569 !important;
        border-color: #475569 !important;
        color: #ffffff !important;
    }

    /* − Remove buttons — light grey */
    div:has(> .explorer-remove-marker) + div[data-testid="stElementContainer"] button,
    div.element-container:has(.explorer-remove-marker) + div.element-container button {
        background-color: #718096 !important;
        color: #ffffff !important;
        border: 1px solid #718096 !important;
        font-weight: 500 !important;
        transition: background-color 0.15s ease !important;
    }
    div:has(> .explorer-remove-marker) + div[data-testid="stElementContainer"] button:hover,
    div.element-container:has(.explorer-remove-marker) + div.element-container button:hover {
        background-color: #8a95a5 !important;
        border-color: #8a95a5 !important;
        color: #ffffff !important;
    }

    /* Suggestion highlights — applied via marker class next to suggested buttons */
    .suggested-add-marker + div[data-testid="stElementContainer"] button,
    div.element-container:has(.suggested-add-marker) + div.element-container button {
        background-color: #d4a43c !important;
        color: #1a1a1a !important;
        border: 2px solid #b88a1a !important;
        font-weight: 600 !important;
        box-shadow: 0 0 8px rgba(212, 164, 60, 0.5) !important;
        animation: glow-gold 2s ease-in-out infinite;
    }
    .suggested-add-marker + div[data-testid="stElementContainer"] button:hover,
    div.element-container:has(.suggested-add-marker) + div.element-container button:hover {
        background-color: #e0b552 !important;
        border-color: #d4a43c !important;
    }
    .suggested-remove-marker + div[data-testid="stElementContainer"] button,
    div.element-container:has(.suggested-remove-marker) + div.element-container button {
        background-color: #d97757 !important;
        color: #ffffff !important;
        border: 2px solid #b85c3a !important;
        font-weight: 600 !important;
        box-shadow: 0 0 8px rgba(217, 119, 87, 0.5) !important;
        animation: glow-orange 2s ease-in-out infinite;
    }
    .suggested-remove-marker + div[data-testid="stElementContainer"] button:hover,
    div.element-container:has(.suggested-remove-marker) + div.element-container button:hover {
        background-color: #e08a6e !important;
        border-color: #d97757 !important;
    }
    @keyframes glow-gold {
        0%, 100% { box-shadow: 0 0 8px rgba(212, 164, 60, 0.5); }
        50% { box-shadow: 0 0 14px rgba(212, 164, 60, 0.8); }
    }
    @keyframes glow-orange {
        0%, 100% { box-shadow: 0 0 8px rgba(217, 119, 87, 0.5); }
        50% { box-shadow: 0 0 14px rgba(217, 119, 87, 0.8); }
    }
    
    
    /* Locked asset (IND-OV, IND-HEM) — permanently disabled Remove button */
    .locked-asset-marker + div[data-testid="stElementContainer"] button,
    div.element-container:has(.locked-asset-marker) + div.element-container button {
        background-color: #e2e8f0 !important;
        color: #2d3748 !important;
        border: 1px solid #a0aec0 !important;
        font-weight: 500 !important;
        cursor: not-allowed !important;
        opacity: 0.85 !important;
    }
    .locked-asset-marker + div[data-testid="stElementContainer"] button:disabled,
    div.element-container:has(.locked-asset-marker) + div.element-container button:disabled {
        background-color: #e2e8f0 !important;
        color: #2d3748 !important;
        border-color: #a0aec0 !important;
        opacity: 0.85 !important;
    }
    
        /* Disabled Remove (min 1) */
    div:has(> .explorer-remove-marker) + div[data-testid="stElementContainer"] button:disabled,
    div.element-container:has(.explorer-remove-marker) + div.element-container button:disabled {
        background-color: #cbd5e0 !important;
        color: #4a5568 !important;
        border-color: #cbd5e0 !important;
        cursor: not-allowed !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("## 🎛️ Interactive Portfolio")
    st.caption(
        "Toggle assets on/off and see how the portfolio responds. "
        "Baseline metrics shown for comparison. "
        "All tranches for an asset are included or excluded together."
    )

    # SPLIT_UI_MARKER_V2 — init with first 2 assets ON, rest OFF
    if "explorer_toggles" not in st.session_state:
        _all_ids = asset_state["Asset_ID"].tolist()
        st.session_state.explorer_toggles = {
            aid: (i < 2) for i, aid in enumerate(_all_ids)
        }

    # AUTO_OPTIMIZE_MARKER_V2 — control bar with bidirectional optimizer
    ctl_l, ctl_m, ctl_o, ctl_r = st.columns([1, 1, 1.3, 1.7])
    with ctl_l:
        if st.button("↺ Reset to start", key="explorer_reset",
                     use_container_width=True,
                     help="Return to minimal portfolio (first 2 assets)"):
            _all_ids_r = asset_state["Asset_ID"].tolist()
            st.session_state.explorer_toggles = {
                aid: (i < 2) for i, aid in enumerate(_all_ids_r)
            }
            st.session_state.pop("explorer_optimize_suggestion", None)
            st.rerun()
    with ctl_m:
        run_full = st.button("▶ Run Full (N=1000)", key="explorer_run_full",
                             type="primary", use_container_width=True)
    with ctl_o:
        optimize_clicked = st.button(
            "🔎 Optimize Portfolio", key="explorer_optimize",
            use_container_width=True,
            help="Find optimal adds/removes to maximize MOIC while passing governance gates",
        )
    with ctl_r:
        _sug_v2 = st.session_state.get("explorer_optimize_suggestion")
        if _sug_v2:
            _sug_adds = _sug_v2.get("add", [])
            _sug_rems = _sug_v2.get("remove", [])
            _parts = []
            if _sug_adds:
                _parts.append(f"add {', '.join(_sug_adds)}")
            if _sug_rems:
                _parts.append(f"remove {', '.join(_sug_rems)}")
            if _parts:
                _moic_delta = _sug_v2.get("moic_delta", 0)
                _delta_str = f"  Δ MOIC: **{_moic_delta:+.2f}x**"
                st.markdown(
                    f"🔎 **Suggestion:** {' · '.join(_parts)}.{_delta_str}  "
                    f"<span style='color:#718096;font-size:12px;'>"
                    f"Click highlighted buttons to apply individually.</span>",
                    unsafe_allow_html=True
                )
            else:
                st.caption("🔎 Current portfolio is already optimal — no changes suggested.")
        else:
            st.caption("Click **Optimize Portfolio** for suggested adds/removes.")

    st.divider()

    # Asset toggle grid
    st.markdown("### Assets")
    # LEGEND_MARKER_V1 — coding scheme reference for exec viewers
    st.caption(
        "Assets are labeled by **Asset_ID · Tier · Development Stage · Indication Cluster**. "
        "Click the legend below to decode these codes."
    )
    with st.expander("ℹ️ Coding scheme reference", expanded=False):
        leg_a, leg_b = st.columns(2)
        with leg_a:
            st.markdown("**Tier** — priority/probability set")
            st.markdown(
                "- **Tier-1** · Lead candidate (primary probability tables)\n"
                "- **Tier-2** · Follow-on / backup (separate probability tables)"
            )
            st.markdown("**Development Stage (DS)**")
            st.markdown(
                "| Code | Stage | Acq prob | Multiplier |\n"
                "|---|---|---|---|\n"
                "| DS-1 | Pre-discovery | 2% | 6.0x |\n"
                "| DS-2 | Discovery | 4% | 5.0x |\n"
                "| **DS-3** | **Lead optimization** (activation threshold) | 6% | 4.0x |\n"
                "| DS-4 | IND-enabling | 6% | 3.5x |\n"
                "| DS-5 | Clinical-ready | 10% | 3.0x |"
            )
            st.caption("Governance: DS-1/DS-2 excluded from portfolio admission. Activation requires DS-3 or higher.")
        with leg_b:
            st.markdown("**Regulatory Archetype (RA)**")
            st.markdown(
                "- **RA-1** · Regulatory archetype 1\n"
                "- **RA-2** · Regulatory archetype 2"
            )
            st.markdown("**Indication Cluster (IND)**")
            st.markdown(
                "| Code | Indication | Status |\n"
                "|---|---|---|\n"
                "| IND-OV | Ovarian | Locked |\n"
                "| IND-HEM | Hematologic | Locked |\n"
                "| IND-NSCLC | Non-Small Cell Lung | Candidate |\n"
                "| IND-CRC | Colorectal | Candidate |\n"
                "| IND-MEL | Melanoma | Candidate |\n"
                "| IND-BREAST | Breast | Candidate |\n"
                "| IND-PROST | Prostate | Candidate |"
            )
            st.caption("Locked indications are required in every admitted portfolio. Candidates compete for remaining slots.")
    # BUDGET_FIX_MARKER_V1 — per-asset budget lives on tranches, not asset_state
    asset_ids = asset_state["Asset_ID"].tolist()
    # Sum tranche budgets per asset; result: {asset_id: total_capital}
    if "Budget" in tranches.columns:
        _asset_budget_map = (tranches.groupby("Asset_ID")["Budget"]
                             .sum().to_dict())
    else:
        _asset_budget_map = {}

    def _asset_label(aid):
        """Composite label + total tranche capital: Asset_ID · Tier · DS · IND"""
        row = asset_state[asset_state["Asset_ID"] == aid].iloc[0]
        parts = [aid]
        for col in ("Tier", "DS_Current", "IndicationCluster_ID"):
            if col in asset_state.columns:
                val = row.get(col)
                if val is not None and str(val) != "nan":
                    parts.append(str(val))
        budget = float(_asset_budget_map.get(aid, 0))
        return " · ".join(parts), budget

    # LOCKED_ASSETS_MARKER_V1 — per curation.py: IND-OV and IND-HEM are
    # required in every admitted portfolio (doctrine-level locking).
    _LOCKED_INDICATIONS = {"IND-OV", "IND-HEM"}
    def _is_locked_asset(aid):
        """True if the asset's indication is in the locked set."""
        row = asset_state[asset_state["Asset_ID"] == aid]
        if row.empty or "IndicationCluster_ID" not in row.columns:
            return False
        return str(row.iloc[0]["IndicationCluster_ID"]) in _LOCKED_INDICATIONS

    _locked_ids = [a for a in asset_ids if _is_locked_asset(a)]
    # Force locked assets to be ON (defensive: if a prior session
    # state toggled one off, re-enable it).
    for _la in _locked_ids:
        if not st.session_state.explorer_toggles.get(_la, False):
            st.session_state.explorer_toggles[_la] = True

    active_ids_ui = [a for a in asset_ids
                     if st.session_state.explorer_toggles.get(a, False)]
    available_ids_ui = [a for a in asset_ids
                        if not st.session_state.explorer_toggles.get(a, False)]

    # ── ACTIVE PORTFOLIO ──
    n_act = len(active_ids_ui)
    total_capital = sum(float(_asset_budget_map.get(a, 0)) for a in active_ids_ui)
    cap_txt = f" · ${total_capital:,.0f} committed" if _asset_budget_map else ""
    st.markdown(
        f"#### ✅ Active Portfolio  "
        f"<span style='color:#4a5568;font-weight:400;font-size:14px;'>"
        f"({n_act} asset{'s' if n_act != 1 else ''}{cap_txt})</span>",
        unsafe_allow_html=True
    )
    if _locked_ids:
        st.caption(
            f"🔒 **Locked assets:** {', '.join(_locked_ids)} "
            f"(required indications IND-OV + IND-HEM per governance doctrine)"
        )

    if not active_ids_ui:
        st.info("No active assets. Add from the Available pool below.")
    else:
        act_cols = st.columns(min(4, max(1, n_act)))
        for i, aid in enumerate(active_ids_ui):
            label_top, budget = _asset_label(aid)
            with act_cols[i % len(act_cols)]:
                st.markdown(
                    f"<div style='padding:10px;border:1px solid #c6f6d5;"
                    f"background:#f0fff4;border-radius:6px;margin-bottom:8px;'>"
                    f"<div style='font-weight:600;font-size:13px;color:#22543d;'>{label_top}</div>"
                    f"<div style='font-size:12px;color:#2f855a;'>${budget:,.0f}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                _is_locked = _is_locked_asset(aid)
                _is_suggested_remove = (not _is_locked) and aid in (
                    st.session_state.get("explorer_optimize_suggestion", {})
                    .get("remove", [])
                )
                if _is_locked:
                    _rem_class = "locked-asset-marker"
                    _rem_label = "🔒 Locked"
                    _rem_help = "Required indication (IND-OV or IND-HEM) — cannot be removed per doctrine"
                elif _is_suggested_remove:
                    _rem_class = "suggested-remove-marker"
                    _rem_label = "⚠ − Remove"
                    _rem_help = "Suggested by optimizer — click to apply"
                else:
                    _rem_class = "explorer-remove-marker"
                    _rem_label = "− Remove"
                    _rem_help = None
                st.markdown(
                    f'<div class="{_rem_class}"></div>',
                    unsafe_allow_html=True,
                )
                if st.button(
                    _rem_label,
                    key=f"explorer_remove_{aid}",
                    disabled=_is_locked,
                    use_container_width=True,
                    help=_rem_help,
                ):
                    st.session_state.explorer_toggles[aid] = False
                    _sug_live = st.session_state.get("explorer_optimize_suggestion")
                    if _sug_live and aid in _sug_live.get("remove", []):
                        _sug_live["remove"] = [x for x in _sug_live["remove"] if x != aid]
                    st.rerun()

    st.markdown("")

    # ── AVAILABLE POOL ──
    n_avail = len(available_ids_ui)
    st.markdown(
        f"#### ➕ Available Assets  "
        f"<span style='color:#4a5568;font-weight:400;font-size:14px;'>"
        f"({n_avail} not in portfolio)</span>",
        unsafe_allow_html=True
    )

    if not available_ids_ui:
        st.caption("All assets are in the active portfolio.")
    else:
        avail_cols = st.columns(min(4, max(1, n_avail)))
        for i, aid in enumerate(available_ids_ui):
            label_top, budget = _asset_label(aid)
            with avail_cols[i % len(avail_cols)]:
                st.markdown(
                    f"<div style='padding:10px;border:1px dashed #cbd5e0;"
                    f"background:#f7fafc;border-radius:6px;margin-bottom:8px;'>"
                    f"<div style='font-weight:600;font-size:13px;color:#2d3748;'>{label_top}</div>"
                    f"<div style='font-size:12px;color:#4a5568;'>${budget:,.0f}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                _is_suggested_add = aid in (
                    st.session_state.get("explorer_optimize_suggestion", {})
                    .get("add", [])
                )
                _add_class = "suggested-add-marker" if _is_suggested_add else "explorer-add-marker"
                _add_label = "⭐ + Add" if _is_suggested_add else "+ Add"
                _add_help = ("Suggested by optimizer — click to apply"
                             if _is_suggested_add else None)
                st.markdown(
                    f'<div class="{_add_class}"></div>',
                    unsafe_allow_html=True,
                )
                if st.button(
                    _add_label,
                    key=f"explorer_add_{aid}",
                    use_container_width=True,
                    help=_add_help,
                ):
                    st.session_state.explorer_toggles[aid] = True
                    # Progressive dim: remove this asset from suggestion
                    _sug_live = st.session_state.get("explorer_optimize_suggestion")
                    if _sug_live and aid in _sug_live.get("add", []):
                        _sug_live["add"] = [x for x in _sug_live["add"] if x != aid]
                    st.rerun()

    # Build active set
    active_ids = [aid for aid, on in
                  st.session_state.explorer_toggles.items() if on]
    excluded_ids = [aid for aid, on in
                    st.session_state.explorer_toggles.items() if not on]

    st.divider()

    # Filter dataframes to active assets only
    scenario_assets = asset_state[asset_state["Asset_ID"].isin(active_ids)].copy()
    scenario_tranches = tranches[tranches["Asset_ID"].isin(active_ids)].copy()

    # Decide N
    n_sims_scenario = 1000 if run_full else 200

    # Cache key: frozenset of excluded + n_sims. Same inputs → same output.
    @st.cache_data(show_spinner=False)
    def _explorer_run(excluded_tuple, n_sims):
        _active = [a for a in asset_ids if a not in set(excluded_tuple)]
        _scen_assets = asset_state[asset_state["Asset_ID"].isin(_active)].copy()
        _scen_tr = tranches[tranches["Asset_ID"].isin(_active)].copy()
        _upfront = envelope.get("Min_Upfront_Threshold", 5_000_000.0)
        _results = run_monte_carlo(
            asset_state=_scen_assets, tranches=_scen_tr, params=params,
            n_sims=int(n_sims), duration=36, contingency_mult=1.10,
            use_corr_stress=False, use_tight_only=False,
            upfront_threshold=_upfront, annual_overhead=0.0,
            enable_tranche_kill=True, enable_rollover=False,
            channel_lookup=channel_lookup,
        )
        return summarize_results(_results), _results

    # BASELINE_FIX_MARKER_V1 — baseline = minimal start (first 2 assets),
    # so deltas turn GREEN as viewer adds assets (matches build-up narrative).
    @st.cache_data(show_spinner=False)
    def _explorer_baseline():
        _baseline_ids = asset_state["Asset_ID"].tolist()[:2]
        _bl_assets = asset_state[asset_state["Asset_ID"].isin(_baseline_ids)].copy()
        _bl_tr = tranches[tranches["Asset_ID"].isin(_baseline_ids)].copy()
        _upfront = envelope.get("Min_Upfront_Threshold", 5_000_000.0)
        _results = run_monte_carlo(
            asset_state=_bl_assets, tranches=_bl_tr, params=params,
            n_sims=1000, duration=36, contingency_mult=1.10,
            use_corr_stress=False, use_tight_only=False,
            upfront_threshold=_upfront, annual_overhead=0.0,
            enable_tranche_kill=True, enable_rollover=False,
            channel_lookup=channel_lookup,
        )
        return summarize_results(_results)

    spinner_msg = ("Running full simulation (N=1000)..." if run_full
                   else "Calculating (N=200)...")
    with st.spinner(spinner_msg):
        scenario_summary, scenario_results_df = _explorer_run(
            tuple(sorted(excluded_ids)), n_sims_scenario)
        baseline_summary = _explorer_baseline()

    # ── Side-by-side metrics ──
    st.markdown("### Scenario vs. Minimal Start")
    _added_count = len(active_ids) - 2
    if _added_count > 0:
        st.caption(
            f"Active: {len(active_ids)} of {len(asset_ids)} assets  |  "
            f"+{_added_count} added above the minimal 2-asset start  |  "
            f"N={n_sims_scenario:,}"
        )
    elif _added_count == 0:
        st.caption(
            f"Minimal start (2 assets)  |  "
            f"Deltas vs. self = 0 — add assets to see scenario improve  |  "
            f"N={n_sims_scenario:,}"
        )
    else:
        # fewer than 2 active — user removed one below the minimum
        st.caption(
            f"Active: {len(active_ids)} of {len(asset_ids)} assets  |  "
            f"Below minimal baseline  |  "
            f"N={n_sims_scenario:,}"
        )

    def _delta_chip(scen_val, base_val, is_pct=False, higher_is_better=True):
        """Return colored delta chip HTML."""
        delta = scen_val - base_val
        if abs(delta) < 1e-9:
            color = "#718096"
            sign = ""
        elif (delta > 0) == higher_is_better:
            color = "#22c55e"
            sign = "+"
        else:
            color = "#ef4444"
            sign = ""
        if is_pct:
            dstr = f"{sign}{delta*100:.1f}pp"
        else:
            dstr = f"{sign}{delta:.2f}"
        return (f'<span style="color:{color};font-weight:600;font-size:14px;">'
                f'{dstr}</span>')

    def _fmt_irr(v):
        return "n/a" if v is None else f"{v*100:.1f}%"

    # IRR_REMOVED_MARKER_V1 — IRR hidden due to 1000% cap saturation in demo portfolio
    metric_specs = [
        ("Median MOIC", "Median_MOIC", False, True, lambda x: f"{x:.2f}x"),
        ("P(≥3 exits)", "P_ThreePlus_Exits", True, True, lambda x: f"{x:.1%}"),
        ("P(≤1 exit)", "P_Exits_LE1", True, False, lambda x: f"{x:.1%}"),
    ]

    mcols = st.columns(3)
    for col, (label, key, is_pct, higher_better, fmt) in zip(mcols, metric_specs):
        with col:
            scen_v = scenario_summary.get(key, 0)
            base_v = baseline_summary.get(key, 0)
            if scen_v is None or base_v is None:
                st.metric(label, "n/a")
                continue
            st.markdown(f"**{label}**")
            st.markdown(
                f'<div style="font-size:24px;font-weight:700;'
                f'color:#1a1a1a;line-height:1.2;">{fmt(scen_v)}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div style="color:#4a5568;font-size:12px;">'
                f'Minimal: {fmt(base_v)}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div style="margin-top:4px;">Δ {_delta_chip(scen_v, base_v, is_pct, higher_better)}</div>',
                unsafe_allow_html=True
            )

    st.divider()

    # ═══════════════════════════════════════════════════════════════
    # AUTO-OPTIMIZE V2: bidirectional greedy (adds AND removes)
    # ═══════════════════════════════════════════════════════════════
    def _score_subset_v2(subset_ids, n_sims=200):
        """Score a subset. Returns (moic, all_gates_pass, gate_failures)."""
        if len(subset_ids) < 3:
            return float("-inf"), False, ["too_small"]
        _sub_assets = asset_state[asset_state["Asset_ID"].isin(subset_ids)].copy()
        _sub_tr = tranches[tranches["Asset_ID"].isin(subset_ids)].copy()
        _upfront = envelope.get("Min_Upfront_Threshold", 5_000_000.0)
        try:
            _res = run_monte_carlo(
                asset_state=_sub_assets, tranches=_sub_tr, params=params,
                n_sims=n_sims, duration=36, contingency_mult=1.10,
                use_corr_stress=False, use_tight_only=False,
                upfront_threshold=_upfront, annual_overhead=0.0,
                enable_tranche_kill=True, enable_rollover=False,
                channel_lookup=channel_lookup,
            )
            _sum = summarize_results(_res)
            _corr_cfg = params.get("correlation", {})
            _corr_idx = compute_correlation_index(_sub_assets, _corr_cfg)
            _wt = compute_weighted_time(_sub_assets, params)
            _conc = check_capital_concentration(_sub_tr)
            _env = check_envelope(_sum, envelope, _corr_idx,
                                  weighted_time=_wt,
                                  concentration_issues=_conc["breaches"])
            _fails = [g for g, c in _env["checks"].items() if not c["pass"]]
            _all_pass = len(_fails) == 0
            _moic = float(_sum.get("Median_MOIC", 0))
            return _moic, _all_pass, _fails
        except Exception:
            return float("-inf"), False, ["error"]

    def _greedy_bidirectional(start_ids, all_ids, max_iterations=10):
        """Greedy search considering both add-moves and remove-moves.

        At each step, try every single-move change (add one from pool, or
        remove one from current). Pick the move that most improves score.
        Score = MOIC if gates pass, else MOIC - 1000 (strong penalty).
        Stop when no single move improves or max_iterations reached.
        """
        current = set(start_ids)
        pool = set(all_ids)

        moic, passes, _ = _score_subset_v2(list(current))
        current_score = moic if passes else moic - 1000
        original_moic = moic

        for _ in range(max_iterations):
            best_move = None
            best_score = current_score
            best_moic = moic
            best_pass = passes

            # Try remove moves (min=3, and never remove locked assets)
            if len(current) > 3:
                for aid in list(current):
                    if _is_locked_asset(aid):
                        continue  # locked indications cannot be removed per doctrine
                    cand = current - {aid}
                    m, p, _ = _score_subset_v2(list(cand))
                    s = m if p else m - 1000
                    if s > best_score + 1e-6:
                        best_score = s
                        best_move = ("remove", aid, cand)
                        best_moic = m
                        best_pass = p

            # Try add moves
            for aid in pool - current:
                cand = current | {aid}
                m, p, _ = _score_subset_v2(list(cand))
                s = m if p else m - 1000
                if s > best_score + 1e-6:
                    best_score = s
                    best_move = ("add", aid, cand)
                    best_moic = m
                    best_pass = p

            if best_move is None:
                break
            current = best_move[2]
            current_score = best_score
            moic = best_moic
            passes = best_pass

        return list(current), original_moic, moic, passes

    # Handle optimize click
    if optimize_clicked:
        _all_ids = asset_state["Asset_ID"].tolist()
        _current = [a for a in _all_ids
                    if st.session_state.explorer_toggles.get(a, False)]
        with st.spinner("Searching for optimal portfolio (adds + removes)..."):
            _opt, _start_moic, _end_moic, _end_pass = _greedy_bidirectional(
                _current, _all_ids
            )
        _current_set = set(_current)
        _opt_set = set(_opt)
        _sug_add = sorted(_opt_set - _current_set)
        _sug_remove = sorted(_current_set - _opt_set)
        st.session_state.explorer_optimize_suggestion = {
            "add": _sug_add,
            "remove": _sug_remove,
            "moic_before": _start_moic,
            "moic_after": _end_moic,
            "moic_delta": _end_moic - _start_moic,
            "gates_pass": _end_pass,
        }
        st.rerun()

    # ── Envelope gate checks ──
    st.markdown("### Envelope Gate Checks")
    try:
        corr_cfg_exp = params.get("correlation", {})
        corr_idx_exp = compute_correlation_index(scenario_assets, corr_cfg_exp)
        wt_exp = compute_weighted_time(scenario_assets, params)
        conc_exp = check_capital_concentration(scenario_tranches)
        env_result = check_envelope(
            scenario_summary, envelope, corr_idx_exp,
            weighted_time=wt_exp, concentration_issues=conc_exp["breaches"],
        )
        gate_rows = []
        fails = []
        for name, check in env_result["checks"].items():
            passed = check["pass"]
            if not passed:
                fails.append(name)
            gate_rows.append({
                "Gate": name,
                "Status": "✅ PASS" if passed else "❌ FAIL",
                "Actual": (f"{check['actual']:.4f}"
                           if isinstance(check["actual"], float)
                           else str(check["actual"])),
                "Threshold": str(check["threshold"]),
            })
        st.dataframe(pd.DataFrame(gate_rows),
                     use_container_width=True, hide_index=True)

        # ── Governance commentary ──
        if not fails:
            st.success("All envelope gates passing — "
                       "scenario approved for board review.")
        elif len(fails) == 1:
            st.error(f"**Governance flag:** {fails[0]} fails threshold. "
                     f"Scenario would require board review before capital "
                     f"commitment.")
        else:
            st.error(f"**Multiple gate failures** ({len(fails)}): "
                     f"{', '.join(fails)}. Scenario would require "
                     f"significant restructure to pass admission.")
    except Exception as e:
        st.warning(f"Envelope gates unavailable: {e}")

    # CHARTS_MARKER_V1 — scenario MC distributions, matching Portfolio tab styling
    st.markdown("### Scenario Distributions")
    st.caption(f"N={len(scenario_results_df):,} simulations")

    ex_c1, ex_c2, ex_c3 = st.columns(3)
    with ex_c1:
        st.markdown("**Exit Count Distribution**")
        _exit_counts = scenario_results_df["Num_Exits"].value_counts().sort_index()
        _fig_ex_exit = go.Figure(go.Bar(
            x=_exit_counts.index, y=_exit_counts.values,
            marker_color="#3182CE",
            hovertemplate="Exits: %{x}<br>Simulations: %{y:,}<extra></extra>",
        ))
        _fig_ex_exit.update_layout(
            xaxis_title="Exit Count", yaxis_title="Simulations",
            height=320, margin=dict(l=40, r=20, t=20, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(_fig_ex_exit, use_container_width=True, key="ex_exit")

    with ex_c2:
        st.markdown("**MOIC Distribution**")
        _valid_moic = scenario_results_df["MOIC"].dropna()
        if len(_valid_moic):
            _fig_ex_moic = go.Figure(go.Histogram(
                x=_valid_moic, nbinsx=25,
                marker_color="#3182CE", opacity=0.85,
                hovertemplate="MOIC: %{x:.1f}x<br>Count: %{y}<extra></extra>",
            ))
            try:
                _add_median_line(_fig_ex_moic, _valid_moic.median(),
                                 f"Median: {_valid_moic.median():.2f}x")
            except Exception:
                _fig_ex_moic.add_vline(x=_valid_moic.median(), line_dash="dash",
                                       line_color="#E53E3E")
            _fig_ex_moic.update_layout(
                xaxis_title="MOIC (x)", yaxis_title="Simulations",
                height=320, margin=dict(l=40, r=20, t=20, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(_fig_ex_moic, use_container_width=True, key="ex_moic")

    with ex_c3:
        st.markdown("**First Distribution Month**")
        _valid_dist = scenario_results_df["First_Dist_Month"].dropna()
        if len(_valid_dist):
            _dist_counts = _valid_dist.astype(int).value_counts().sort_index()
            _med_dist = float(_valid_dist.median())
            _fig_ex_dist = go.Figure(go.Bar(
                x=_dist_counts.index, y=_dist_counts.values,
                marker_color="#805AD5",
                hovertemplate="Month %{x}<br>Simulations: %{y:,}<extra></extra>",
            ))
            _fig_ex_dist.add_shape(
                type="line", x0=_med_dist, x1=_med_dist,
                y0=0, y1=_dist_counts.values.max(),
                line=dict(color="#E53E3E", dash="dash", width=2),
            )
            _fig_ex_dist.add_annotation(
                x=_med_dist, y=_dist_counts.values.max(),
                text=f"Median: Mo {_med_dist:.0f}", showarrow=False,
                yshift=15, font=dict(color="#E53E3E", size=11),
            )
            _fig_ex_dist.update_layout(
                xaxis_title="Month", yaxis_title="Simulations",
                height=320, margin=dict(l=40, r=20, t=30, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(_fig_ex_dist, use_container_width=True, key="ex_dist")

    st.divider()

    # ── Asset state reference (collapsed) ──
    with st.expander("Asset state detail", expanded=False):
        disp = asset_state.copy()
        disp["Active"] = disp["Asset_ID"].apply(
            lambda a: "✓" if st.session_state.explorer_toggles.get(a, True)
            else "—"
        )
        cols_show = ["Active", "Asset_ID"] + [
            c for c in disp.columns if c not in ("Active", "Asset_ID")
        ]
        st.dataframe(disp[cols_show], use_container_width=True, hide_index=True)

