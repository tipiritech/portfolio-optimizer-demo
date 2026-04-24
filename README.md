# Portfolio Optimizer — Public Demo

Monte Carlo governance engine for portfolio-level drug development modeling and optimization.

**This is a demonstration build.** All data is synthetic. Authentication is disabled. Channel integration (CRO/Pharma vendor relationships) is stubbed out.

---

## What this demo shows

- **Interactive Portfolio Construction** — build a portfolio by adding assets from a pool of 10 synthetic candidates
- **Live Monte Carlo simulation** — every toggle triggers a quick recalculation (N=200 sims)
- **7-gate governance framework** — each scenario checked against portfolio admission criteria (MOIC, exit probabilities, correlation, concentration, weighted time)
- **Auto-Optimize** — greedy search suggesting adds/removes to improve the portfolio against governance constraints
- **Governance doctrine enforcement** — locked indications (IND-OV + IND-HEM) required in every portfolio
- **4 tabs** — Portfolio Overview, Asset Profiles, Interactive Portfolio, Sandbox, Settings

---

## How to use the demo

1. Open **🎛️ Interactive Portfolio** tab
2. Start state: 2 locked assets (A-0001 · IND-OV, A-0002 · IND-HEM)
3. Click **+ Add** on cards in the Available Assets pool to build up the portfolio
4. Watch scenario metrics, envelope gates, and distribution charts recalculate
5. Click **🔎 Optimize Portfolio** to get suggestions (gold = add, orange = remove)
6. Click **↺ Reset to start** to return to the 2-asset minimum

---

## Known limitations

This demo has known issues documented honestly:

- **IRR metric hidden.** The underlying IRR calculation has a 1000% cap that saturates, producing degenerate values. Removed from this demo pending a proper fix to `src/metrics.py annualize_monthly_irr`.
- **Monte Carlo noise at N=200.** Small Δ chips (~5%) may appear even when scenario equals baseline. This is sampling variance, not a bug.
- **Auto-Optimize scoring.** Current scoring function can declare "already optimal" on portfolios with multiple gate failures. Known issue — scoring upgrade planned.
- **Not production data.** Synthetic 10-asset portfolio. Governance thresholds and coding conventions match the production framework, but the portfolio composition is illustrative.

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Dashboard opens at `http://localhost:8501`.

---

## Doctrine reference

- Minimum MOIC: 3.0x median
- P(≥3 exits) ≥ 60%
- P(≤1 exit) ≤ 15%
- Capital concentration ≤ 20% per asset
- Weighted time-to-monetization ≤ 24 months
- Max duration: 36 months (hard stop, non-overridable)
- Correlation index: base floor 0.25, stress add-on +0.10
- Upfront threshold: $5M for qualifying exit
- Locked indications: IND-OV (Ovarian), IND-HEM (Hematologic) — required in every admitted portfolio

---

## Coding scheme

- **Tier-1** · Lead candidate
- **Tier-2** · Follow-on / backup candidate
- **DS-1** · Pre-discovery (excluded by doctrine)
- **DS-2** · Discovery (excluded by doctrine)
- **DS-3** · Lead optimization (activation threshold)
- **DS-4** · IND-enabling
- **DS-5** · Clinical-ready
- **IND-OV** · Ovarian (locked)
- **IND-HEM** · Hematologic (locked)
- **IND-NSCLC** · Non-Small Cell Lung
- **IND-CRC** · Colorectal
- **IND-MEL** · Melanoma
- **IND-BREAST** · Breast
- **IND-PROST** · Prostate
