"""
Kelly criterion, leveraged-portfolio simulation and volatility-drag decomposition.

Conventions
-----------
* Returns are daily simple returns, aligned on trading days.
* ``r``   : simple total return of the risky asset on day t (dividends included through
            adjusted prices; fund expenses are already in the price and are not charged again).
* ``r_l`` : daily return on cash (lending rate) for day t.
* ``r_b`` : daily borrowing rate for day t.
* Timing  : the exposure L_t used on day t is fixed at the close of day t-1, using only
            information available then. The portfolio is rebalanced back to its target at
            each close. Financing accrues on the cash or debt position held over day t.
* Costs   : ``cost`` is charged per unit of exposure traded at the close of day t, in
            proportion of wealth at the close of t-1. The initial purchase is not charged,
            so that L = 1 reproduces buy-and-hold exactly.
* Wealth  : W_t = W_{t-1} * F_t with F_t = 1 + R_{p,t} - c_t. If F_t <= 0 the strategy is
            insolvent: wealth is set to zero and stays at zero (no artificial recovery).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

DAYS = 252


# --------------------------------------------------------------------------- simulation
@dataclass
class SimResult:
    wealth: pd.Series          # wealth path, W_0 = 1 at the close before the first return
    returns: pd.Series         # portfolio simple returns (NaN after insolvency)
    insolvent_on: object       # date of insolvency, or None
    margin_breaches: int       # days on which equity/assets fell below the maintenance ratio
    turnover: float            # average daily exposure traded, fraction of wealth


def portfolio_return(r, r_l, r_b, L):
    """Piecewise financing: cash at the lending rate if L <= 1, debt at the borrowing rate if L > 1."""
    r, r_l, r_b, L = (np.asarray(x, float) for x in (r, r_l, r_b, L))
    lev = L > 1
    return np.where(lev, L * r - (L - 1) * r_b, L * r + (1 - L) * r_l)


def simulate(r, r_l, r_b, L, cost=0.0, maintenance=0.25):
    """Daily-rebalanced constant (or time-varying) exposure L.

    r, r_l, r_b : pd.Series on the same index (daily).
    L           : scalar or pd.Series of target exposures for each day (decided the day before).
    """
    idx = r.index
    Lt = np.full(len(r), float(L)) if np.isscalar(L) else np.asarray(pd.Series(L).reindex(idx), float)
    rv, lv, bv = np.asarray(r, float), np.asarray(r_l, float), np.asarray(r_b, float)
    Rp = portfolio_return(rv, lv, bv, Lt)
    # exposure after the day's move, per unit of W_{t-1}: L_t (1 + r_t); target for tomorrow: L_{t+1} (1 + R_p,t)
    L_next = np.r_[Lt[1:], Lt[-1]]
    traded = np.abs(L_next * (1 + Rp) - Lt * (1 + rv))
    F = 1 + Rp - cost * traded
    # maintenance check before rebalancing (leveraged days only): equity / assets
    with np.errstate(divide="ignore", invalid="ignore"):
        eq_ratio = np.where(Lt > 1, (1 + Rp) / (Lt * (1 + rv)), 1.0)
    breaches = int(np.sum((Lt > 1) & (eq_ratio < maintenance)))
    dead = np.flatnonzero(F <= 0)
    insolvent_on = None
    if dead.size:
        k = dead[0]
        insolvent_on = idx[k]
        F = F.copy()
        F[k:] = 0.0
    W = np.cumprod(F)
    ret = pd.Series(F - 1, index=idx)
    if insolvent_on is not None:
        ret.iloc[dead[0] + 1:] = np.nan
    return SimResult(pd.Series(W, index=idx), ret, insolvent_on, breaches, float(np.mean(traded)))


def realised_log_growth(r, r_l, r_b, L, cost=0.0, years=None):
    """Annualised realised log growth log(W_T)/years of the exact daily recursion (-inf if insolvent)."""
    res = simulate(r, r_l, r_b, L, cost)
    if res.insolvent_on is not None:
        return -np.inf
    yrs = years if years is not None else len(r) / DAYS
    return float(np.log(res.wealth.iloc[-1]) / yrs)


# --------------------------------------------------------------------------- theory
def g_theory(L, mu, sigma2, r_l, r_b=None):
    """Expected log growth under the diffusion approximation, annualised.

    g(L) = r_l + L (mu - r_l) - L^2 sigma^2 / 2            for 0 <= L <= 1
    g(L) = r_b + L (mu - r_b) - L^2 sigma^2 / 2            for L > 1
    (mu: arithmetic expected return; sigma^2: variance; all annualised).
    """
    r_b = r_l if r_b is None else r_b
    L = np.asarray(L, float)
    return np.where(L <= 1, r_l + L * (mu - r_l), r_b + L * (mu - r_b)) - 0.5 * L ** 2 * sigma2


def kelly_leverage(mu, sigma2, r_l, r_b=None):
    """Maximiser of g_theory: interior solution on either side of the kink at L = 1."""
    r_b = r_l if r_b is None else r_b
    lo = (mu - r_l) / sigma2
    hi = (mu - r_b) / sigma2
    if lo <= 1:
        return max(lo, 0.0)
    if hi >= 1:
        return hi
    return 1.0


# --------------------------------------------------------------------------- compounding decomposition
def compounding_gap(r, b, L):
    """Exact cumulative log gap between a frictionless daily-reset leveraged position financed at b
    and the geometrically scaled reference, plus its small-return approximation.

    D_T(L) = sum_t [ log(1 + L r_t - (L-1) b_t) - L log(1 + r_t) + (L-1) log(1 + b_t) ]
    D_T(L) ~ -1/2 L (L-1) sum_t (r_t - b_t)^2
    Requires 1 + L r_t - (L-1) b_t > 0, 1 + r_t > 0 and 1 + b_t > 0 for every t.
    """
    r, b = np.asarray(r, float), np.asarray(b, float)
    lev = 1 + L * r - (L - 1) * b
    if np.any(lev <= 0):
        raise ValueError("leveraged wealth factor not positive: logarithm undefined")
    exact = np.log(lev) - L * np.log1p(r) + (L - 1) * np.log1p(b)
    approx = -0.5 * L * (L - 1) * (r - b) ** 2
    return np.cumsum(exact), np.cumsum(approx)


def geometric_reference(r, b, L):
    """Log of the geometrically scaled benchmark: L * sum log(1+r) - (L-1) * sum log(1+b).
    A reference for decomposition only; it is not an investable strategy."""
    r, b = np.asarray(r, float), np.asarray(b, float)
    return np.cumsum(L * np.log1p(r) - (L - 1) * np.log1p(b))


# --------------------------------------------------------------------------- uncertainty
def circular_block_bootstrap_kelly(ex, block, reps, rng):
    """Bootstrap distribution of the frictionless Kelly ratio mean(ex)/var(ex) of daily excess returns.
    block = 1 gives the i.i.d. bootstrap."""
    x = np.asarray(ex, float)
    n = len(x)
    nb = int(np.ceil(n / block))
    out = np.empty(reps)
    for k in range(reps):
        starts = rng.integers(0, n, size=nb)
        ii = ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
        s = x[ii]
        out[k] = s.mean() / s.var(ddof=1)
    return out


# --------------------------------------------------------------------------- validation
def validate(r, r_l, r_b, tol=1e-10):
    """Numerical checks on real data. Raises AssertionError on failure; returns a dict of results."""
    out = {}
    tr_path = np.cumprod(1 + np.asarray(r, float))
    s1 = simulate(r, r_l, r_b, 1.0, cost=0.0002)
    out["L1_max_rel_error"] = float(np.max(np.abs(s1.wealth.values / tr_path - 1)))
    assert out["L1_max_rel_error"] < tol, "L = 1 does not reproduce buy-and-hold"
    s0 = simulate(r, r_l, r_b, 0.0, cost=0.0002)
    cash = np.cumprod(1 + np.asarray(r_l, float))
    out["L0_max_rel_error"] = float(np.max(np.abs(s0.wealth.values / cash - 1)))
    assert out["L0_max_rel_error"] < tol, "L = 0 does not follow cash"
    mu, s2, rl, rb = 0.09, 0.04, 0.02, 0.035
    grid = np.linspace(0, 5, 500001)
    num = grid[np.argmax(g_theory(grid, mu, s2, rl, rb))]
    out["kelly_vs_grid"] = float(abs(num - kelly_leverage(mu, s2, rl, rb)))
    assert out["kelly_vs_grid"] < 1e-4, "closed-form Kelly does not match the numerical maximum"
    for L in (0.5, 2.0, 3.0):
        d_exact, _ = compounding_gap(r, r_b, L)
        sim = simulate(r, r_b, r_b, L, cost=0.0)   # frictionless, financing at b on both sides
        gap = np.log(sim.wealth.values) - geometric_reference(r, r_b, L)
        err = float(np.max(np.abs(gap - d_exact)))
        out[f"decomposition_error_L{L}"] = err
        assert err < 1e-8, "decomposition does not reconcile with simulated wealth"
    crash = pd.Series([0.01, -0.6, 0.5, 0.3], index=pd.RangeIndex(4))
    z = pd.Series(0.0, index=crash.index)
    sc = simulate(crash, z, z, 2.0)
    out["insolvency_no_recovery"] = bool(sc.insolvent_on == 1 and np.all(sc.wealth.values[1:] == 0))
    assert out["insolvency_no_recovery"], "wealth recovers after insolvency"
    return out
