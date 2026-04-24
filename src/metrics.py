"""
Financial metrics: IRR, MOIC, annualization.
"""

import numpy as np


def npv_from_rate(cashflows, rate):
    """Compute NPV for a monthly discount rate."""
    cashflows = np.array(cashflows, dtype=float)
    periods = np.arange(len(cashflows))

    if rate <= -0.999999:
        return np.nan

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        discount = (1.0 + rate) ** periods
        values = cashflows / discount

    if not np.all(np.isfinite(values)):
        return np.nan

    return np.sum(values)


def monthly_irr(cashflows, low=-0.95, high=1.0, max_iter=200, tol=1e-7):
    """Robust monthly IRR using bounded bisection."""
    cashflows = np.array(cashflows, dtype=float)

    if not (np.any(cashflows < 0) and np.any(cashflows > 0)):
        return np.nan

    npv_low = npv_from_rate(cashflows, low)
    npv_high = npv_from_rate(cashflows, high)

    if np.isnan(npv_low) or np.isnan(npv_high):
        return np.nan
    if npv_low * npv_high > 0:
        return np.nan

    mid = np.nan
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        npv_mid = npv_from_rate(cashflows, mid)

        if np.isnan(npv_mid):
            return np.nan
        if abs(npv_mid) < tol:
            return mid

        if npv_low * npv_mid < 0:
            high = mid
        else:
            low = mid
            npv_low = npv_mid

    return mid


def annualize_monthly_irr(monthly_rate):
    """Convert monthly IRR to annual via compounding."""
    if np.isnan(monthly_rate):
        return np.nan
    return (1.0 + monthly_rate) ** 12 - 1.0


def compute_moic(total_inflows, total_outflows):
    """Multiple on invested capital."""
    if total_outflows <= 0:
        return np.nan
    return total_inflows / total_outflows
