# Architecture

The Discovery Portfolio Optimization Engine is organized into five core modules, each mapped to a top-level capability of the product. Modules are designed to be composable: the simulation engine consumes data assumptions, the optimizer consumes simulation output, and governance rules are enforced across the pipeline as hard constraints.

## 1. Simulation (`src/simulation/`)

Monte Carlo engine that models the full distribution of portfolio outcomes across thousands of scenarios. Handles draws for exit timing, exit value, attrition, and correlated macro factors. Provides the probabilistic backbone for all downstream analysis, including stress testing under tight-market and elevated-correlation regimes.

## 2. Governance (`src/governance/`)

Capital-discipline rules encoded as hard constraints rather than guidelines. Enforces no-recycling of returned capital, prevents parameter drift across runs, and blocks narrative-based admission of assets that do not meet quantitative thresholds. Every constraint is assertable and logged so that compliance can be audited after the fact.

## 3. Optimization (`src/optimization/`)

Portfolio construction and asset scoring. Each candidate asset is scored across five dimensions — exit density, IRR, downside protection, liquidity acceleration, and correlation diversification — and the optimizer searches for allocations that maximize risk-adjusted return subject to governance constraints.

## 4. Data (`src/data/`)

Oncology transaction data and the probability, cost, and timing assumptions derived from it. Every assumption used by the simulation and optimization modules is sourced here, versioned, and traceable back to the underlying transaction evidence. This is what makes assumptions auditable.

## 5. Tests (`tests/`)

Unit tests for individual components and integration tests that exercise the full pipeline: data assumptions → simulation → governance checks → optimization output. Governance rules in particular are covered by tests that assert they cannot be bypassed.

## Data flow

```
data/  ──►  simulation/  ──►  optimization/
              ▲                    ▲
              └── governance/ ─────┘
```

Governance rules apply at both the simulation input boundary (admission criteria) and the optimization output boundary (allocation constraints).
