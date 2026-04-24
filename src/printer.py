"""
PDF Print Module — Export any engine output section as a standalone PDF.

Each print function produces a single-section PDF with:
  - Discovery Biotech logo header
  - Section title + timestamp
  - Formatted data table or metrics
  - Footer with vehicle ID + confidentiality notice

Usage:
    from src.printer import print_portfolio_overview, print_mc_results, ...
    print_portfolio_overview(roster, asset_state, tranches, params, output_path="overview.pdf")

All functions accept an optional logo_path argument.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, HRFlowable,
)


# ── Brand colors ──────────────────────────────────────────────────────────

BRAND_NAVY = colors.HexColor("#4A5568")
BRAND_PINK = colors.HexColor("#C53A86")
HEADER_BG = colors.HexColor("#F7FAFC")
PASS_GREEN = colors.HexColor("#C6F6D5")
FAIL_RED = colors.HexColor("#FED7D7")
WARN_YELLOW = colors.HexColor("#FEFCBF")
ROW_ALT = colors.HexColor("#F0F4F8")


# ── Styles ────────────────────────────────────────────────────────────────

def _get_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "SectionTitle", parent=styles["Heading1"],
        fontSize=16, textColor=BRAND_NAVY, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "SubTitle", parent=styles["Heading2"],
        fontSize=12, textColor=BRAND_NAVY, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        "CellText", parent=styles["Normal"],
        fontSize=8, leading=10,
    ))
    styles.add(ParagraphStyle(
        "Footer", parent=styles["Normal"],
        fontSize=7, textColor=colors.grey, alignment=1,
    ))
    styles.add(ParagraphStyle(
        "Metric", parent=styles["Normal"],
        fontSize=11, leading=14,
    ))
    styles.add(ParagraphStyle(
        "MetricLabel", parent=styles["Normal"],
        fontSize=9, textColor=colors.grey, leading=11,
    ))
    return styles


# ── Shared helpers ────────────────────────────────────────────────────────

def _header_block(title: str, styles, logo_path: str | None = None,
                  vehicle_id: str = "V-001") -> list:
    """Build logo + title + timestamp header."""
    elements = []

    if logo_path and os.path.exists(logo_path):
        try:
            img = Image(logo_path, width=0.6 * inch, height=0.6 * inch)
            header_data = [[img, Paragraph(f"<b>Discovery Biotech</b>", styles["Normal"])]]
            ht = Table(header_data, colWidths=[0.8 * inch, 4 * inch])
            ht.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
            elements.append(ht)
            elements.append(Spacer(1, 4))
        except Exception:
            pass

    elements.append(Paragraph(title, styles["SectionTitle"]))
    ts = datetime.now().strftime("%B %d, %Y at %H:%M")
    elements.append(Paragraph(
        f"Vehicle: {vehicle_id}  |  Generated: {ts}", styles["MetricLabel"]
    ))
    elements.append(Spacer(1, 4))
    elements.append(HRFlowable(width="100%", thickness=1, color=BRAND_NAVY))
    elements.append(Spacer(1, 12))
    return elements


def _footer_block(styles, confidential: bool = True) -> list:
    elements = []
    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    elements.append(Spacer(1, 4))
    notice = "CONFIDENTIAL — For authorized recipients only." if confidential else ""
    elements.append(Paragraph(
        f"Discovery Biotech Portfolio Engine  |  {notice}", styles["Footer"]
    ))
    return elements


def _df_to_table(df: pd.DataFrame, styles, max_col_width: float = 1.8 * inch) -> Table:
    """Convert a DataFrame to a reportlab Table with alternating row colors."""
    headers = list(df.columns)
    data = [headers]
    for _, row in df.iterrows():
        data.append([str(v) if pd.notna(v) else "" for v in row])

    n_cols = len(headers)
    col_w = min(max_col_width, (7.0 * inch) / max(n_cols, 1))
    t = Table(data, colWidths=[col_w] * n_cols, repeatRows=1)

    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("FONTSIZE", (0, 1), (-1, -1), 7),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    # Alternating row colors
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), ROW_ALT))

    t.setStyle(TableStyle(style_cmds))
    return t


def _metrics_table(metrics: list[tuple[str, str]], styles) -> Table:
    """
    Render a list of (label, value) pairs as a 2-column metrics grid.
    metrics: list of (label_str, value_str)
    """
    # Arrange in 2-up grid
    rows = []
    for i in range(0, len(metrics), 2):
        row = []
        for j in range(2):
            idx = i + j
            if idx < len(metrics):
                label, val = metrics[idx]
                row.extend([
                    Paragraph(f"<font color='grey'>{label}</font>", styles["MetricLabel"]),
                    Paragraph(f"<b>{val}</b>", styles["Metric"]),
                ])
            else:
                row.extend(["", ""])
        rows.append(row)

    t = Table(rows, colWidths=[1.5 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _build_doc(output_path: str, orient: str = "portrait") -> SimpleDocTemplate:
    ps = letter if orient == "portrait" else landscape(letter)
    return SimpleDocTemplate(
        output_path, pagesize=ps,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
    )


# ══════════════════════════════════════════════════════════════════════════
# INDIVIDUAL PRINT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def print_portfolio_overview(
    roster: pd.DataFrame,
    asset_state: pd.DataFrame,
    tranches: pd.DataFrame,
    params: dict,
    output_path: str = "portfolio_overview.pdf",
    logo_path: str | None = None,
    vehicle_id: str = "V-001",
) -> str:
    """Print portfolio overview: roster, asset state, tranches, key metrics."""
    from src.optimization.correlation import compute_correlation_index
    from src.optimization.envelope import compute_weighted_time

    styles = _get_styles()
    doc = _build_doc(output_path)
    elements = _header_block("Portfolio Overview", styles, logo_path, vehicle_id)

    # Key metrics
    corr_index = compute_correlation_index(asset_state, params.get("correlation", {}))
    weighted_time = compute_weighted_time(asset_state, params)
    total_capital = tranches["Budget"].sum()

    elements.append(Paragraph("Key Metrics", styles["SubTitle"]))
    elements.append(_metrics_table([
        ("Assets", str(len(roster))),
        ("Planned Capital", f"${total_capital:,.0f}"),
        ("Tranches", str(len(tranches))),
        ("Correlation Index", f"{corr_index:.3f}"),
        ("Weighted Time", f"{weighted_time:.1f} months"),
    ], styles))
    elements.append(Spacer(1, 12))

    # Asset roster
    elements.append(Paragraph("Asset Roster", styles["SubTitle"]))
    display_cols = [c for c in roster.columns if c not in ("Notes",)]
    elements.append(_df_to_table(roster[display_cols], styles))
    elements.append(Spacer(1, 12))

    # Capital tranches
    elements.append(Paragraph("Capital Tranches", styles["SubTitle"]))
    tr_cols = ["Asset_ID", "Tranche_ID", "Purpose", "Budget", "Start_Month", "Stop_Month", "Status"]
    available_t = [c for c in tr_cols if c in tranches.columns]
    tr_display = tranches[available_t].copy()
    if "Budget" in tr_display.columns:
        tr_display["Budget"] = tr_display["Budget"].apply(lambda x: f"${x:,.0f}")
    elements.append(_df_to_table(tr_display, styles))

    elements.extend(_footer_block(styles))
    doc.build(elements)
    return output_path


def print_mc_results(
    summary: dict,
    envelope_result: dict,
    output_path: str = "mc_results.pdf",
    logo_path: str | None = None,
    vehicle_id: str = "V-001",
    n_sims: int = 0,
) -> str:
    """Print Monte Carlo simulation results: metrics + envelope gates."""
    styles = _get_styles()
    doc = _build_doc(output_path)
    elements = _header_block("Monte Carlo Simulation Results", styles, logo_path, vehicle_id)

    elements.append(Paragraph(f"Simulations: {n_sims:,}", styles["MetricLabel"]))
    elements.append(Spacer(1, 8))

    # Metrics
    def _f(v, fmt=".2f", suffix=""):
        return f"{v:{fmt}}{suffix}" if pd.notna(v) else "N/A"

    elements.append(Paragraph("Return Metrics", styles["SubTitle"]))
    elements.append(_metrics_table([
        ("Median MOIC", _f(summary["Median_MOIC"], ".2f", "x")),
        ("MOIC P10", _f(summary["MOIC_P10"], ".2f", "x")),
        ("Median IRR", _f(summary.get("Median_Annual_IRR"), ".1%")),
        ("IRR P10", _f(summary.get("IRR_P10"), ".1%")),
    ], styles))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("Exit Probabilities", styles["SubTitle"]))
    elements.append(_metrics_table([
        ("P(0 exits)", f"{summary['P_Zero_Exits']:.1%}"),
        ("P(1 exit)", f"{summary['P_One_Exit']:.1%}"),
        ("P(2 exits)", f"{summary['P_Two_Exits']:.1%}"),
        ("P(3+ exits)", f"{summary['P_ThreePlus_Exits']:.1%}"),
    ], styles))
    elements.append(Spacer(1, 12))

    # Envelope gates
    elements.append(Paragraph("Governance Envelope", styles["SubTitle"]))
    gate_data = [["Gate", "Status", "Actual", "Threshold"]]
    for name, check in envelope_result.get("checks", {}).items():
        status = "PASS" if check["pass"] else "FAIL"
        actual = f"{check['actual']:.4f}" if isinstance(check["actual"], float) else str(check["actual"])
        gate_data.append([name, status, actual, str(check["threshold"])])

    gt = Table(gate_data, colWidths=[2 * inch, 0.8 * inch, 1.5 * inch, 1.5 * inch], repeatRows=1)
    gate_style = [
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ]
    for i in range(1, len(gate_data)):
        if gate_data[i][1] == "PASS":
            gate_style.append(("BACKGROUND", (1, i), (1, i), PASS_GREEN))
        else:
            gate_style.append(("BACKGROUND", (1, i), (1, i), FAIL_RED))
    gt.setStyle(TableStyle(gate_style))
    elements.append(gt)

    elements.extend(_footer_block(styles))
    doc.build(elements)
    return output_path


def print_stress_results(
    stress_results: dict,
    output_path: str = "stress_results.pdf",
    logo_path: str | None = None,
    vehicle_id: str = "V-001",
) -> str:
    """Print stress test results: all 4 scenarios side by side."""
    styles = _get_styles()
    doc = _build_doc(output_path, orient="landscape")
    elements = _header_block("Stress Test Results", styles, logo_path, vehicle_id)

    scenarios = list(stress_results.keys())
    header = ["Metric"] + scenarios
    metrics_display = [
        ("Median MOIC", "Median_MOIC", ".2f", "x"),
        ("MOIC P10", "MOIC_P10", ".2f", "x"),
        ("Median IRR", "Median_Annual_IRR", ".1%", ""),
        ("IRR P10", "IRR_P10", ".1%", ""),
        ("P(3+ exits)", "P_ThreePlus_Exits", ".1%", ""),
        ("P(<=1 exit)", "P_Exits_LE1", ".1%", ""),
        ("P(0 exits)", "P_Zero_Exits", ".1%", ""),
    ]

    data = [header]
    for display_name, key, fmt, suffix in metrics_display:
        row = [display_name]
        for s in scenarios:
            val = stress_results[s]["summary"].get(key, float("nan"))
            row.append(f"{val:{fmt}}{suffix}" if pd.notna(val) else "N/A")
        data.append(row)

    # Envelope row
    env_row = ["Envelope"]
    for s in scenarios:
        env_row.append("PASS" if stress_results[s]["all_pass"] else "FAIL")
    data.append(env_row)

    n_cols = len(header)
    col_w = (9.0 * inch) / n_cols
    t = Table(data, colWidths=[col_w] * n_cols, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    # Color envelope row
    env_idx = len(data) - 1
    for j in range(1, n_cols):
        bg = PASS_GREEN if data[env_idx][j] == "PASS" else FAIL_RED
        style_cmds.append(("BACKGROUND", (j, env_idx), (j, env_idx), bg))
    t.setStyle(TableStyle(style_cmds))
    elements.append(t)

    elements.extend(_footer_block(styles))
    doc.build(elements)
    return output_path


def print_channel_summary(
    channel_lookup: dict,
    output_path: str = "channel_summary.pdf",
    logo_path: str | None = None,
    vehicle_id: str = "V-001",
) -> str:
    """Print CRO/Pharma channel effects summary."""
    from src.channel import summarize_channel
    styles = _get_styles()
    doc = _build_doc(output_path, orient="landscape")
    elements = _header_block("CRO / Pharma Channel Summary", styles, logo_path, vehicle_id)

    ch_df = summarize_channel(channel_lookup)
    # Format cost column
    if "Est_IND_Cost_Mode" in ch_df.columns:
        ch_df["Est_IND_Cost_Mode"] = ch_df["Est_IND_Cost_Mode"].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
    if "Est_IND_Time_Mode" in ch_df.columns:
        ch_df["Est_IND_Time_Mode"] = ch_df["Est_IND_Time_Mode"].apply(
            lambda x: f"{x:.0f}mo" if pd.notna(x) else "—")
    if "Engagement_Arbitrage_Mo" in ch_df.columns:
        ch_df["Engagement_Arbitrage_Mo"] = ch_df["Engagement_Arbitrage_Mo"].apply(
            lambda x: f"{x:+.1f}mo" if pd.notna(x) else "—")

    elements.append(_df_to_table(ch_df, styles, max_col_width=1.3 * inch))

    elements.extend(_footer_block(styles))
    doc.build(elements)
    return output_path


def print_sandbox_comparison(
    deltas: dict,
    label: str,
    output_path: str = "sandbox_comparison.pdf",
    logo_path: str | None = None,
    vehicle_id: str = "V-001",
) -> str:
    """Print sandbox scenario comparison."""
    styles = _get_styles()
    doc = _build_doc(output_path)
    elements = _header_block(f"Sandbox: {label}", styles, logo_path, vehicle_id)

    elements.append(Paragraph(
        "EXPLORATORY ONLY — Not governance-grade",
        ParagraphStyle("Warning", parent=styles["Normal"],
                       fontSize=10, textColor=colors.HexColor("#B7791F"),
                       backColor=WARN_YELLOW, borderPadding=6),
    ))
    elements.append(Spacer(1, 12))

    data = [["Metric", "Base", "Sandbox", "Delta", ""]]
    for key, d in deltas.items():
        if key.startswith("_"):
            continue
        unit = d["unit"]
        if unit == "x":
            b_s = f"{d['base']:.2f}x" if pd.notna(d["base"]) else "N/A"
            s_s = f"{d['sandbox']:.2f}x" if pd.notna(d["sandbox"]) else "N/A"
            d_s = f"{d['delta']:+.2f}x" if pd.notna(d["delta"]) else "—"
        else:
            b_s = f"{d['base']:.1%}" if pd.notna(d["base"]) else "N/A"
            s_s = f"{d['sandbox']:.1%}" if pd.notna(d["sandbox"]) else "N/A"
            d_s = f"{d['delta']:+.1%}" if pd.notna(d["delta"]) else "—"
        icon = "Better" if d["direction"] == "better" else ("Worse" if d["direction"] == "worse" else "Flat")
        data.append([key, b_s, s_s, d_s, icon])

    t = Table(data, colWidths=[2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 0.8 * inch], repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for i in range(1, len(data)):
        if data[i][4] == "Better":
            style_cmds.append(("BACKGROUND", (4, i), (4, i), PASS_GREEN))
        elif data[i][4] == "Worse":
            style_cmds.append(("BACKGROUND", (4, i), (4, i), FAIL_RED))
    t.setStyle(TableStyle(style_cmds))
    elements.append(t)

    elements.extend(_footer_block(styles))
    doc.build(elements)
    return output_path


def print_envelope_gates(
    envelope_result: dict,
    output_path: str = "envelope_gates.pdf",
    logo_path: str | None = None,
    vehicle_id: str = "V-001",
) -> str:
    """Print governance envelope gate results only."""
    styles = _get_styles()
    doc = _build_doc(output_path)
    elements = _header_block("Governance Envelope Gates", styles, logo_path, vehicle_id)

    all_pass = envelope_result.get("all_pass", False)
    status_text = "ALL GATES PASS" if all_pass else f"FAILED: {', '.join(envelope_result.get('failed_gates', []))}"
    status_color = PASS_GREEN if all_pass else FAIL_RED
    elements.append(Paragraph(
        f"<b>{status_text}</b>",
        ParagraphStyle("EnvStatus", parent=styles["Normal"],
                       fontSize=12, backColor=status_color, borderPadding=8),
    ))
    elements.append(Spacer(1, 12))

    data = [["Gate", "Status", "Actual", "Threshold"]]
    for name, check in envelope_result.get("checks", {}).items():
        status = "PASS" if check["pass"] else "FAIL"
        actual = f"{check['actual']:.4f}" if isinstance(check["actual"], float) else str(check["actual"])
        data.append([name, status, actual, str(check["threshold"])])

    t = Table(data, colWidths=[2.2 * inch, 0.8 * inch, 1.8 * inch, 1.8 * inch], repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for i in range(1, len(data)):
        bg = PASS_GREEN if data[i][1] == "PASS" else FAIL_RED
        style_cmds.append(("BACKGROUND", (1, i), (1, i), bg))
    t.setStyle(TableStyle(style_cmds))
    elements.append(t)

    elements.extend(_footer_block(styles))
    doc.build(elements)
    return output_path


def print_preflight(
    param_issues: list,
    activation: dict,
    concentration: dict,
    combined_prob: dict,
    output_path: str = "preflight.pdf",
    logo_path: str | None = None,
    vehicle_id: str = "V-001",
) -> str:
    """Print pre-flight check results."""
    styles = _get_styles()
    doc = _build_doc(output_path)
    elements = _header_block("Pre-Flight Checks", styles, logo_path, vehicle_id)

    checks = [
        ("Parameters", len(param_issues) == 0, f"{len(param_issues)} issue(s)" if param_issues else "OK"),
        ("Activation", activation["activated"], f"{len(activation['issues'])} issue(s)" if not activation["activated"] else "OK"),
        ("Concentration", len(concentration.get("breaches", [])) == 0,
         f"{len(concentration.get('breaches', []))} breach(es)" if concentration.get("breaches") else "OK"),
        ("Combined Prob", combined_prob["all_pass"],
         f"{len(combined_prob.get('failures', []))} below 45%" if not combined_prob["all_pass"] else "OK"),
    ]

    data = [["Check", "Status", "Detail"]]
    for name, passed, detail in checks:
        data.append([name, "PASS" if passed else "FAIL", detail])

    t = Table(data, colWidths=[2 * inch, 0.8 * inch, 3 * inch], repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for i in range(1, len(data)):
        bg = PASS_GREEN if data[i][1] == "PASS" else FAIL_RED
        style_cmds.append(("BACKGROUND", (1, i), (1, i), bg))
    t.setStyle(TableStyle(style_cmds))
    elements.append(t)

    # Detail sections
    if param_issues:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Parameter Issues", styles["SubTitle"]))
        for issue in param_issues:
            elements.append(Paragraph(f"- {issue}", styles["CellText"]))

    if not activation["activated"]:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Activation Issues", styles["SubTitle"]))
        for issue in activation["issues"]:
            elements.append(Paragraph(f"- {issue}", styles["CellText"]))

    elements.extend(_footer_block(styles))
    doc.build(elements)
    return output_path
