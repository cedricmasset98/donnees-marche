"""
Full analysis of the Vanguard Total World Stock ETF (VT) and LaTeX report.

Reads only the CSV files in data/ (no request to Yahoo):
  data/VT.csv    VT daily prices (Close, Adj Close)
  data/IRX.csv   13-week T-bill rate, annualised % (risk-free rate)
  data/CHFX.csv  USD/CHF (CHF per 1 USD)

Outputs:
  report/figures/*.pdf   vector charts
  report/VT_analysis.tex LaTeX source, rendered from report/template.tex
  report/VT_analysis.pdf compiled report (latexmk + pdflatex)
"""
import os
import shutil
import subprocess
from datetime import date

import jinja2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from scipy import stats, optimize

DAYS_PER_YEAR = 252
TRANSACTION_COST = 0.001  # 0.10% per position switch (200-day MA strategy)
BLUE, ORANGE, GREEN, GREY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#898781", "#e34948"

# --------------------------------------------------------------------------
# Number formatting
# --------------------------------------------------------------------------

def num(x, d=2):
    return f"{x:,.{d}f}"


def pct(x, d=1, sign=False):
    return (f"{x * 100:+.{d}f}" if sign else f"{x * 100:.{d}f}") + "%"


def dt(d):
    return pd.Timestamp(d).strftime("%d %b %Y")


def mon(d):
    return pd.Timestamp(d).strftime("%b %Y")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load(name):
    return pd.read_csv(f"data/{name}.csv", parse_dates=["Date"], index_col="Date")


vt = load("VT")
px = vt["Close"]                         # price
tr = vt["Adj Close"]                     # total return (dividends reinvested)
idx = vt.index

irx = load("IRX")["Close"].reindex(idx, method="ffill") / 100
rf_daily = irx / DAYS_PER_YEAR
chf = load("CHFX")["Close"].reindex(idx, method="ffill")

r_tr = tr.pct_change().dropna()
lr_tr = np.log(tr).diff().dropna()
rf = rf_daily.reindex(r_tr.index)
years = (idx[-1] - idx[0]).days / 365.25

tr_m = tr.resample("ME").last()
r_m = tr_m.pct_change().dropna()
rf_m = (irx.resample("ME").mean() / 12).reindex(r_m.index)

# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def cagr(series):
    a = (series.index[-1] - series.index[0]).days / 365.25
    return (series.iloc[-1] / series.iloc[0]) ** (1 / a) - 1


def drawdown(series):
    return series / series.cummax() - 1


def to_monthly(r):
    """Compound daily simple returns into calendar-month returns."""
    return (1 + r).groupby(r.index.to_period("M")).prod() - 1


def sharpe_ci(ex_m, level=0.95):
    """Sharpe ratio of monthly excess returns with the standard error of
    Mertens (2002) / Opdyke (2007), which allows for skewness and kurtosis.
    Returns (annualised SR, annualised SE, lower, upper, two-sided p-value of SR = 0)."""
    T = len(ex_m)
    sr = ex_m.mean() / ex_m.std()
    g3 = stats.skew(ex_m)
    g4 = stats.kurtosis(ex_m, fisher=False)
    se = np.sqrt((1 + 0.5 * sr ** 2 - g3 * sr + (g4 - 3) / 4 * sr ** 2) / T)
    z = stats.norm.ppf(0.5 + level / 2)
    k = np.sqrt(12)
    return sr * k, se * k, (sr - z * se) * k, (sr + z * se) * k, 2 * stats.norm.sf(abs(sr / se))


def metrics(r, rfq):
    """r: daily simple returns; rfq: aligned daily risk-free rate.

    - annualised return: calendar time, from the close preceding the first return
      to the last close (same method as cagr());
    - volatility, Sharpe and Sortino: computed on calendar-month returns and
      annualised with sqrt(12). Daily returns are negatively autocorrelated, so
      scaling daily volatility by sqrt(252) would overstate risk (see the variance
      ratio test); the daily-scaled figure is kept as "vol_d" for comparison;
    - maximum drawdown: on the daily value path.
    """
    m, rf_mo = to_monthly(r), to_monthly(rfq)
    ex_m = m - rf_mo
    vol = m.std() * np.sqrt(12)
    down = np.sqrt((ex_m.clip(upper=0) ** 2).mean()) * np.sqrt(12)
    value = (1 + r).cumprod()
    start = idx[idx.get_loc(r.index[0]) - 1]
    ca = value.iloc[-1] ** (365.25 / (r.index[-1] - start).days) - 1
    mdd = drawdown(pd.concat([pd.Series([1.0]), value.reset_index(drop=True)])).min()
    sr, se, lo, hi, p = sharpe_ci(ex_m)
    return {"cagr": ca, "vol": vol, "vol_d": r.std() * np.sqrt(DAYS_PER_YEAR),
            "sharpe": sr, "sharpe_se": se, "sharpe_lo": lo, "sharpe_hi": hi, "sharpe_p": p,
            "sortino": ex_m.mean() * 12 / down, "mdd": mdd, "calmar": ca / abs(mdd)}


full = metrics(r_tr, rf)
rf_mean = irx.mean()

# Calendar-year returns
y_tr = tr.resample("YE").last().pct_change()
y_tr.iloc[0] = tr.resample("YE").last().iloc[0] / tr.iloc[0] - 1
y_px = px.resample("YE").last().pct_change()
y_px.iloc[0] = px.resample("YE").last().iloc[0] / px.iloc[0] - 1
y_tr.index = y_tr.index.year
y_px.index = y_px.index.year
partial_first = y_tr.index[0]
partial_last = y_tr.index[-1] if idx[-1].month < 12 else None
partials = [partial_first] + ([partial_last] if partial_last else [])
y_full = y_tr.drop(partials)

# Rolling volatility
roll_vol = r_tr.rolling(DAYS_PER_YEAR).std() * np.sqrt(DAYS_PER_YEAR)
# EWMA volatility (RiskMetrics, lambda = 0.94, zero mean): reacts like GARCH, without the
# "ghost" steps of an equally weighted window when an extreme day enters and leaves it
EWMA_LAMBDA = 0.94


def ewma_vol(r, lam=EWMA_LAMBDA, seed=25):
    x = np.asarray(r, float)
    var = np.empty_like(x)
    var[0] = np.mean(x[:seed] ** 2)
    for t in range(1, len(x)):
        var[t] = lam * var[t - 1] + (1 - lam) * x[t - 1] ** 2
    return pd.Series(np.sqrt(var * DAYS_PER_YEAR), index=r.index)


ew_vol = ewma_vol(r_tr).iloc[63:]          # drop the burn-in quarter

# Drawdowns (total return)
dd = drawdown(tr)


def drawdown_episodes(series, n=5):
    d = drawdown(series)
    res, falling, start = [], False, None
    for t, v in d.items():
        if v < 0 and not falling:
            falling, start = True, series.loc[:t].index[-2]
        elif v == 0 and falling:
            seg = d.loc[start:t]
            res.append((start, seg.idxmin(), t, seg.min()))
            falling = False
    if falling:
        seg = d.loc[start:]
        res.append((start, seg.idxmin(), None, seg.min()))
    res.sort(key=lambda x: x[3])
    return res[:n]


top_dd = drawdown_episodes(tr)

# Sub-periods
cuts = [("2008-06-01", "2012-12-31"), ("2013-01-01", "2017-12-31"),
        ("2018-01-01", "2022-12-31"), ("2023-01-01", str(idx[-1].date()))]
sub = []
for a, b in cuts:
    r_s = r_tr.loc[a:b]
    sub.append((r_s.index[0], r_s.index[-1], metrics(r_s, rf.loc[r_s.index])))

# Distribution
mu, sd = r_tr.mean(), r_tr.std()
skew = stats.skew(r_tr)
kurt = stats.kurtosis(r_tr, fisher=False)
jb_stat, jb_p = stats.jarque_bera(r_tr)
worst_day, best_day = r_tr.idxmin(), r_tr.idxmax()
beyond4 = (np.abs(r_tr - mu) > 4 * sd).sum()
expected4 = len(r_tr) * 2 * stats.norm.sf(4)

# VaR / ES
var_rows = []
for lvl in (0.95, 0.99):
    q = r_tr.quantile(1 - lvl)
    es = r_tr[r_tr <= q].mean()
    q_n = mu + sd * stats.norm.ppf(1 - lvl)
    es_n = mu - sd * stats.norm.pdf(stats.norm.ppf(1 - lvl)) / (1 - lvl)
    var_rows.append((lvl, q, es, q_n, es_n, (r_tr < q_n).mean()))


# Autocorrelation
def acf(x, h):
    x = np.asarray(x) - np.mean(x)
    v = np.sum(x ** 2)
    return np.array([np.sum(x[k:] * x[:-k]) / v for k in range(1, h + 1)])


def ljung_box(x, h):
    n = len(x)
    rho = acf(x, h)
    q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, h + 1)))
    return q, stats.chi2.sf(q, h)


H = 20
acf_r = acf(lr_tr, H)
acf_r2 = acf(lr_tr ** 2, H)
lb_r = ljung_box(lr_tr.values, 10)
lb_r2 = ljung_box((lr_tr ** 2).values, 10)

ac_freq = []
for name, s in [("Daily", lr_tr),
                ("Weekly", np.log(tr.resample("W-FRI").last()).diff().dropna()),
                ("Monthly", np.log(tr_m).diff().dropna())]:
    r1 = acf(s, 1)[0]
    t = r1 * np.sqrt(len(s))
    ac_freq.append((name, len(s), r1, t, 2 * stats.norm.sf(abs(t))))

non_crisis = ~((lr_tr.index < "2010-01-01") |
               ((lr_tr.index >= "2020-02-01") & (lr_tr.index < "2020-07-01")))
ac1_non_crisis = acf(lr_tr[non_crisis], 1)[0]


# Lo-MacKinlay (1988) variance ratio test, heteroskedasticity-robust z*
def variance_ratio(lp, q):
    r = np.diff(lp)
    n = len(r)
    m = np.mean(r)
    e = r - m
    var1 = np.sum(e ** 2) / (n - 1)
    rq = lp[q:] - lp[:-q] - q * m
    varq = np.sum(rq ** 2) / (q * (n - q + 1) * (1 - q / n))
    vr = varq / var1
    den = np.sum(e ** 2) ** 2
    theta = sum((2 * (q - j) / q) ** 2 * n * np.sum(e[j:] ** 2 * e[:-j] ** 2) / den
                for j in range(1, q))
    z = np.sqrt(n) * (vr - 1) / np.sqrt(theta)
    return vr, z, 2 * stats.norm.sf(abs(z))


vr_rows = [(q, *variance_ratio(np.log(tr.values), q)) for q in (2, 5, 10, 20)]

# Calendar effects
by_day = [r_tr[r_tr.index.dayofweek == k] for k in range(5)]
f_day, p_day = stats.f_oneway(*by_day)

by_month = r_m.groupby(r_m.index.month).mean()
jan = r_m[r_m.index.month == 1]
other = r_m[r_m.index.month != 1]
t_jan, p_jan = stats.ttest_ind(jan, other, equal_var=False)

# "Sell in May": May-October vs November-April (complete half-years)
summer, winter = [], []
for a in range(idx[0].year + 1, idx[-1].year + 1):
    s = tr_m.loc[f"{a}-04-30":f"{a}-10-31"]
    if len(s) == 7:
        summer.append(s.iloc[-1] / s.iloc[0] - 1)
    w = tr_m.loc[f"{a - 1}-10-31":f"{a}-04-30"]
    if len(w) == 7:
        winter.append(w.iloc[-1] / w.iloc[0] - 1)
summer, winter = np.array(summer), np.array(winter)
t_sim, p_sim = stats.ttest_ind(winter, summer, equal_var=False)

# 200-day moving-average strategy
ma200 = px.rolling(200).mean()
signal = (px > ma200).astype(float).shift(1)       # decision taken the day before
signal[ma200.shift(1).isna()] = np.nan
sig = signal.reindex(r_tr.index).dropna()
r_bh = r_tr.loc[sig.index]
r_cash = rf.loc[sig.index]
switches = sig.diff().abs().fillna(0)
r_ma = sig * r_bh + (1 - sig) * r_cash - switches * TRANSACTION_COST
m_bh = metrics(r_bh, r_cash)
m_ma = metrics(r_ma, r_cash)
n_switch = int(switches.sum())
share_invested = sig.mean()
val_bh = (1 + r_bh).cumprod()
val_ma = (1 + r_ma).cumprod()

# Rolling returns
roll_rows, roll_series = [], {}
for h, name in [(12, "1 year"), (60, "5 years"), (120, "10 years")]:
    g = ((tr_m / tr_m.shift(h)) ** (12 / h) - 1).dropna()
    roll_series[name] = g
    roll_rows.append((name, len(g), (g < 0).mean(), g.min(), g.idxmin(), g.median(), g.max()))

# Missing the best days
cap = 10_000
best_rows = []
ranked = r_tr.sort_values(ascending=False)
for n in (0, 10, 20, 30):
    v = cap * (1 + (r_tr.drop(ranked.index[:n]) if n else r_tr)).prod()
    best_rows.append((n, v, (v / cap) ** (1 / years) - 1))
worst_rows = []
ranked_w = r_tr.sort_values()
for n in (10, 20, 30):
    v = cap * (1 + r_tr.drop(ranked_w.index[:n])).prod()
    worst_rows.append((n, v, (v / cap) ** (1 / years) - 1))
best_near_worst = sum(1 for d in ranked.index[:10]
                      if any(abs((d - p).days) <= 14 for p in ranked_w.index[:10]))

# Monthly savings plan, lump sum vs 12-month averaging
contribution = 500
plan_prices = tr_m.iloc[:-1]  # month-end contributions, valued at the last price
units = (contribution / plan_prices).sum()
plan_value = units * tr.iloc[-1]
plan_invested = contribution * len(plan_prices)
flows = [(-contribution, d) for d in plan_prices.index] + [(plan_value, idx[-1])]


def irr(flows):
    t0 = flows[0][1]
    return optimize.brentq(
        lambda r: sum(c / (1 + r) ** ((d - t0).days / 365.25) for c, d in flows), -0.9, 1.0)


plan_irr = irr(flows)

cash_m = (1 + rf_m.reindex(tr_m.index).fillna(0)).cumprod()
P, C = tr_m.values, cash_m.values
ls_dca = []
for i in range(len(P) - 12):
    ls = P[i + 12] / P[i]
    avg = sum((1 / 12) * (C[i + k] / C[i]) * P[i + 12] / P[i + k] for k in range(12))
    ls_dca.append((tr_m.index[i], ls, avg))
ld = pd.DataFrame(ls_dca, columns=["date", "ls", "dca"]).set_index("date")
ls_win_share = (ld["ls"] > ld["dca"]).mean()
ls_mean_gap = (ld["ls"] - ld["dca"]).mean()

# Swiss franc perspective
tr_chf = tr * chf
cagr_usd, cagr_chf, cagr_fx = cagr(tr), cagr(tr_chf), cagr(chf)
vol_chf = tr_chf.resample("ME").last().pct_change().std() * np.sqrt(12)   # monthly, as for USD
corr_fx = r_tr.corr(chf.pct_change().reindex(r_tr.index))

# --------------------------------------------------------------------------
# Additional rigour: sampling frequency, GARCH, VaR backtest, robustness
# --------------------------------------------------------------------------
RNG_SEED = 20080626          # fixed seed: bootstrap results are reproducible
B = 2000                     # bootstrap replications

# Volatility by sampling frequency (non-overlapping returns)
vol_freq = []
for name, rule, k in [("Daily", None, DAYS_PER_YEAR), ("Weekly", "W-FRI", 52),
                      ("Monthly", "ME", 12), ("Quarterly", "QE", 4)]:
    s = r_tr if rule is None else tr.resample(rule).last().pct_change().dropna()
    vol_freq.append((name, len(s), s.std() * np.sqrt(k)))


# Robust (heteroskedasticity-consistent) test of first-order autocorrelation
def robust_ac1(x):
    e = np.asarray(x) - np.mean(x)
    rho = np.sum(e[1:] * e[:-1]) / np.sum(e ** 2)
    se = np.sqrt(np.sum(e[1:] ** 2 * e[:-1] ** 2)) / np.sum(e ** 2)
    z = rho / se
    return rho, z, 2 * stats.norm.sf(abs(z))


series_freq = {"Daily": lr_tr,
               "Weekly": np.log(tr.resample("W-FRI").last()).diff().dropna(),
               "Monthly": np.log(tr_m).diff().dropna()}
ac_rob = {k: robust_ac1(v) for k, v in series_freq.items()}


# Holm (1979) step-down adjustment over the family of efficiency tests
def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


family = ([("ac", k, ac_rob[k][2]) for k in ("Daily", "Weekly", "Monthly")]
          + [("vr", q, p) for q, vr, z, p in vr_rows]
          + [("cal", "day", p_day), ("cal", "jan", p_jan), ("cal", "sim", p_sim)])
holm_adj = dict(zip([(a, b) for a, b, _ in family], holm([p for _, _, p in family])))
n_family = len(family)
n_raw_sig = sum(p < 0.05 for _, _, p in family)
n_holm_sig = sum(v < 0.05 for v in holm_adj.values())

# GARCH(1,1) with Student-t innovations, estimated by maximum likelihood
from scipy.signal import lfilter
from scipy.special import gammaln

x_pct = lr_tr.values * 100   # daily log returns in percent


def garch_var(params, x):
    mu, omega, alpha, beta = params[:4]
    e = x - mu
    s0 = np.var(x)
    y = lfilter([0.0, alpha], [1.0, -beta], e ** 2)
    t = np.arange(len(x))
    s2 = y + omega * (1 - beta ** t) / (1 - beta) + beta ** t * s0
    return e, s2


def garch_nll(params, x):
    mu, omega, alpha, beta, nu = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999 or nu <= 2.05:
        return 1e10
    e, s2 = garch_var(params, x)
    ll = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
          - 0.5 * np.log(s2) - (nu + 1) / 2 * np.log1p(e ** 2 / ((nu - 2) * s2)))
    return -np.sum(ll)


def garch_fit(x, start=None):
    x0 = start if start is not None else [np.mean(x), 0.05 * np.var(x), 0.08, 0.90, 6.0]
    bounds = [(-1, 1), (1e-6, 5), (1e-4, 0.5), (0.5, 0.9995), (2.1, 100)]
    res = optimize.minimize(garch_nll, x0, args=(x,), method="L-BFGS-B", bounds=bounds)
    return res.x, res


def num_hessian(f, p, h=1e-4):
    p = np.asarray(p, float)
    n = len(p)
    Hm = np.zeros((n, n))
    step = h * np.maximum(np.abs(p), 1e-2)
    for i in range(n):
        for j in range(i, n):
            pp = p.copy(); pp[i] += step[i]; pp[j] += step[j]; fpp = f(pp)
            pm = p.copy(); pm[i] += step[i]; pm[j] -= step[j]; fpm = f(pm)
            mp = p.copy(); mp[i] -= step[i]; mp[j] += step[j]; fmp = f(mp)
            mm = p.copy(); mm[i] -= step[i]; mm[j] -= step[j]; fmm = f(mm)
            Hm[i, j] = Hm[j, i] = (fpp - fpm - fmp + fmm) / (4 * step[i] * step[j])
    return Hm


g_par, g_res = garch_fit(x_pct)
g_se = np.sqrt(np.diag(np.linalg.inv(num_hessian(lambda p: garch_nll(p, x_pct), g_par))))
g_mu, g_omega, g_alpha, g_beta, g_nu = g_par
g_persist = g_alpha + g_beta
g_half_life = np.log(0.5) / np.log(g_persist)
g_uncond_vol = np.sqrt(g_omega / (1 - g_persist) * DAYS_PER_YEAR) / 100
_, g_s2 = garch_var(g_par, x_pct)
garch_vol = pd.Series(np.sqrt(g_s2 * DAYS_PER_YEAR) / 100, index=lr_tr.index)

# Out-of-sample one-day VaR backtest (log returns)
WIN = 250
first_bt = lr_tr.index[2 * WIN]             # about two years of history before the first forecast
bt_idx = lr_tr.index[lr_tr.index >= first_bt]
lr = lr_tr.values
n_all = len(lr)
pos_bt = np.arange(n_all - len(bt_idx), n_all)

# GARCH: parameters re-estimated each calendar year on an expanding window,
# conditional variance updated every day with information up to t-1
g_fore = np.empty(len(pos_bt))
g_nu_t = np.empty(len(pos_bt))
g_mu_t = np.empty(len(pos_bt))
years_bt = bt_idx.year
prev = None
for yr in np.unique(years_bt):
    mask = years_bt == yr
    first_pos = pos_bt[mask][0]
    par, _ = garch_fit(x_pct[:first_pos], start=prev)
    prev = par
    # variance for day t uses returns up to t-1: run the filter on x[:t] and step once
    # s2[t] depends only on returns up to t-1: it is the one-step-ahead forecast for day t
    _, s2 = garch_var(par, x_pct)
    g_fore[mask] = np.sqrt(s2[pos_bt[mask]])
    g_nu_t[mask] = par[4]
    g_mu_t[mask] = par[0]

var_methods = {}
for lvl in (0.05, 0.01):
    hist = np.array([np.quantile(lr[p - WIN:p], lvl) for p in pos_bt])
    norm = np.array([lr[p - WIN:p].mean() + lr[p - WIN:p].std(ddof=1) * stats.norm.ppf(lvl) for p in pos_bt])
    q_t = stats.t.ppf(lvl, g_nu_t) * np.sqrt((g_nu_t - 2) / g_nu_t)
    # Zero conditional mean, standard at a one-day horizon. The ML location estimate of a
    # symmetric Student-t lies well above the sample mean when returns are negatively
    # skewed, which would push the VaR up; that variant is reported as a sensitivity check.
    garch_q = g_fore * q_t / 100
    garch_q_mle = (g_mu_t + g_fore * q_t) / 100
    var_methods[lvl] = {"Historical (250 days)": hist, "Normal (250 days)": norm, "GARCH(1,1)-t": garch_q}
    var_methods.setdefault("mle_mean", {})[lvl] = garch_q_mle


def kupiec(hits, p):
    n, x = len(hits), hits.sum()
    ph = x / n
    if x == 0:
        lr_uc = -2 * n * np.log(1 - p)
    else:
        lr_uc = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - ph) - x * np.log(ph))
    return lr_uc, stats.chi2.sf(lr_uc, 1)


def christoffersen(hits):
    h = hits.astype(int)
    a, b = h[:-1], h[1:]
    n00, n01 = np.sum((a == 0) & (b == 0)), np.sum((a == 0) & (b == 1))
    n10, n11 = np.sum((a == 1) & (b == 0)), np.sum((a == 1) & (b == 1))
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p1 = (n01 + n11) / (n00 + n01 + n10 + n11)

    def ll(p, k0, k1):
        return (k0 * np.log(1 - p) if k0 else 0) + (k1 * np.log(p) if k1 and p > 0 else 0)

    lr_ind = -2 * (ll(p1, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11))
    return lr_ind, stats.chi2.sf(lr_ind, 1)


bt_rows = []
realised = lr[pos_bt]
mle_breach = {lvl: (realised < var_methods["mle_mean"][lvl]).mean() for lvl in (0.05, 0.01)}
for lvl in (0.05, 0.01):
    for name, q in var_methods[lvl].items():
        hits = realised < q
        lr_uc, p_uc = kupiec(hits, lvl)
        lr_ind, p_ind = christoffersen(hits)
        p_cc = stats.chi2.sf(lr_uc + lr_ind, 2)
        bt_rows.append((lvl, name, hits.mean(), int(hits.sum()), p_uc, p_ind, p_cc))
n_bt = len(pos_bt)

# Sensitivity of the moving-average rule and paired block bootstrap
ma_windows = (50, 100, 150, 200, 250)
ma_start = px.index[249 + 1]                # first day on which every window is available
r_common = r_tr.loc[ma_start:]
rf_common = rf.loc[r_common.index]


def ma_returns(w):
    ma = px.rolling(w).mean()
    s = (px > ma).astype(float).shift(1).reindex(r_common.index)
    sw = s.diff().abs().fillna(0)
    return s * r_common + (1 - s) * rf_common - sw * TRANSACTION_COST, int(sw.sum()), s.mean()


ma_sens = []
for w in ma_windows:
    rr, nsw, inv = ma_returns(w)
    ma_sens.append((w, metrics(rr, rf_common), nsw, inv))
bh_common = metrics(r_common, rf_common)


def block_bootstrap_idx(n, block, reps, rng):
    """Circular block bootstrap indices (reps x n)."""
    starts = rng.integers(0, n, size=(reps, int(np.ceil(n / block))))
    idxs = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idxs.reshape(reps, -1)[:, :n]


rng = np.random.default_rng(RNG_SEED)
rr200, _, _ = ma_returns(200)
a_bh, a_ma = r_common.values, rr200.values
ii = block_bootstrap_idx(len(a_bh), 63, B, rng)
yrs_common = len(a_bh) / DAYS_PER_YEAR
g_bh = np.exp(np.log1p(a_bh[ii]).sum(axis=1) / yrs_common) - 1
g_ma = np.exp(np.log1p(a_ma[ii]).sum(axis=1) / yrs_common) - 1
diff_obs = (np.exp(np.log1p(a_bh).sum() / yrs_common) - 1) - (np.exp(np.log1p(a_ma).sum() / yrs_common) - 1)
diff_bs = g_bh - g_ma
diff_lo, diff_hi = np.percentile(diff_bs, [2.5, 97.5])
diff_p = (np.abs(diff_bs - diff_obs) >= abs(diff_obs)).mean()   # centred bootstrap, H0: no difference

# Probability of loss: stationary bootstrap of monthly log returns
lm = np.log(tr_m).diff().dropna().values


def stationary_bootstrap_sums(x, horizon, reps, mean_block, rng):
    n = len(x)
    p = 1 / mean_block
    pos = rng.integers(0, n, size=reps)
    total = x[pos].copy()
    for _ in range(1, horizon):
        jump = rng.random(reps) < p
        pos = np.where(jump, rng.integers(0, n, size=reps), (pos + 1) % n)
        total += x[pos]
    return total


rng = np.random.default_rng(RNG_SEED)
boot_ploss = {}
for h, name in [(12, "1 year"), (60, "5 years"), (120, "10 years")]:
    sims = stationary_bootstrap_sums(lm, h, 20000, 12, rng)
    boot_ploss[name] = (sims < 0).mean()

# Sensitivity to the starting date (to the last close)
start_sens = pd.Series({d: (tr.iloc[-1] / v) ** (365.25 / (idx[-1] - d).days) - 1
                        for d, v in tr_m.items() if (idx[-1] - d).days >= 5 * 365})
start_sens_annual = {y: start_sens[start_sens.index.year == y].iloc[-1]
                     for y in range(idx[0].year, idx[-1].year - 4)}

# Weekly correlation with USD/CHF (avoids the time-of-day mismatch of daily closes)
w_vt = tr.resample("W-FRI").last().pct_change()
w_fx = chf.resample("W-FRI").last().pct_change()
corr_fx_w = w_vt.corr(w_fx)

# Validation against Vanguard's published market-price returns
ref = pd.read_csv("data/reference/vanguard_vt_trailing_2026-09-30.csv")
REF_DATE = pd.Timestamp("2026-09-30")
INCEPTION = pd.Timestamp("2008-06-24")


def tr_at(d):
    return tr.loc[:d].iloc[-1]


val_rows = []
if idx[-1] >= REF_DATE:
    end_v = tr_at(REF_DATE)
    calc = {"quarter": end_v / tr_at("2026-06-30") - 1, "ytd": end_v / tr_at("2025-12-31") - 1}
    for y in (1, 3, 5, 10):
        calc[f"{y}y"] = (end_v / tr_at(REF_DATE - pd.DateOffset(years=y))) ** (1 / y) - 1
    calc["since_inception"] = (end_v / tr.iloc[0]) ** (365.25 / (REF_DATE - idx[0]).days) - 1
    labels = {"quarter": "Third quarter 2026", "ytd": "Year to date", "1y": "1 year", "3y": "3 years",
              "5y": "5 years", "10y": "10 years", "since_inception": "Since inception"}
    for _, rw in ref.iterrows():
        val_rows.append((labels[rw["period"]], rw["market_price_return"] / 100, calc[rw["period"]],
                         rw["nav_return"] / 100, rw["annualised"] == "yes"))
val_max_gap = max(abs(a - b) for _, a, b, _, _ in val_rows[:-1]) if val_rows else None

# The headline moving-average comparison uses the same common period as the
# sensitivity analysis and the bootstrap, so that all MA figures are comparable
sig = (px > px.rolling(200).mean()).astype(float).shift(1).reindex(r_common.index)
r_bh, r_ma = r_common, rr200
m_bh, m_ma = bh_common, metrics(rr200, rf_common)
n_switch = int(sig.diff().abs().fillna(0).sum())
share_invested = sig.mean()
val_bh = (1 + r_bh).cumprod()
val_ma = (1 + r_ma).cumprod()

# Reproducibility: data fingerprint and library versions
import hashlib
import platform
import scipy as _scipy
data_sha = hashlib.sha256(open("data/VT.csv", "rb").read()).hexdigest()[:16]
versions = (f"Python {platform.python_version()}, pandas {pd.__version__}, NumPy {np.__version__}, "
            f"SciPy {_scipy.__version__}, Matplotlib {matplotlib.__version__}")

# --------------------------------------------------------------------------
# Export for the website dashboard (heavy computations done once, here)
# --------------------------------------------------------------------------
import json


def _r(x, d=6):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


# Rolling efficiency statistics: trailing 3-year windows, evaluated at month ends
ROLL_WIN = 756
eff_roll = {"dates": [], "ac1": [], "ac1_se": [], "vr5": [], "z5": [], "vr20": [], "z20": []}
lr_vals = lr_tr.values
month_end_pos = [lr_tr.index.get_loc(d) for d in lr_tr.groupby(lr_tr.index.to_period("M")).tail(1).index]
for p_ in month_end_pos:
    if p_ + 1 < ROLL_WIN:
        continue
    w = lr_vals[p_ + 1 - ROLL_WIN:p_ + 1]
    e = w - w.mean()
    rho = np.sum(e[1:] * e[:-1]) / np.sum(e ** 2)
    se_ = np.sqrt(np.sum(e[1:] ** 2 * e[:-1] ** 2)) / np.sum(e ** 2)
    lp_w = np.r_[0.0, np.cumsum(w)]
    vr5, z5, _ = variance_ratio(lp_w, 5)
    vr20, z20, _ = variance_ratio(lp_w, 20)
    eff_roll["dates"].append(lr_tr.index[p_].strftime("%Y-%m-%d"))
    for k_, val in (("ac1", rho), ("ac1_se", se_), ("vr5", vr5), ("z5", z5), ("vr20", vr20), ("z20", z20)):
        eff_roll[k_].append(_r(val, 4))

export = {
    "generated": date.today().isoformat(),
    "last_date": idx[-1].strftime("%Y-%m-%d"),
    "report_metrics": {k: _r(val) for k, val in full.items()},
    "vol_by_frequency": [{"frequency": n_, "obs": int(o), "vol": _r(vv)} for n_, o, vv in vol_freq],
    "garch": {
        "params": {k: _r(val) for k, val in zip(["mu", "omega", "alpha", "beta", "nu"], g_par)},
        "se": {k: _r(val) for k, val in zip(["mu", "omega", "alpha", "beta", "nu"], g_se)},
        "persistence": _r(g_persist), "half_life_days": _r(g_half_life, 1), "long_run_vol": _r(g_uncond_vol),
        "dates": [d.strftime("%Y-%m-%d") for d in garch_vol.index],
        "vol": [_r(val, 4) for val in garch_vol.values],
    },
    "var_backtest": {
        "dates": [d.strftime("%Y-%m-%d") for d in bt_idx],
        "ret": [_r(val, 5) for val in realised],
        "var1": [_r(val, 5) for val in var_methods[0.01]["GARCH(1,1)-t"]],
        "var5": [_r(val, 5) for val in var_methods[0.05]["GARCH(1,1)-t"]],
        "table": [{"level": l_, "method": n_, "breach_rate": _r(rate), "breaches": k_, "expected": _r(l_ * n_bt, 1),
                   "p_kupiec": _r(puc), "p_christoffersen": _r(pind), "p_cc": _r(pcc)}
                  for l_, n_, rate, k_, puc, pind, pcc in bt_rows],
    },
    "efficiency": {
        "tests": ([{"test": f"Lag-1 autocorrelation, {k_.lower()}", "stat": _r(ac_rob[k_][0], 4),
                    "z": _r(ac_rob[k_][1], 3), "p": _r(ac_rob[k_][2], 4), "holm": _r(holm_adj[("ac", k_)], 4)}
                   for k_ in ("Daily", "Weekly", "Monthly")]
                  + [{"test": f"Variance ratio, q = {q_}", "stat": _r(vr_, 4), "z": _r(z_, 3), "p": _r(p_v, 4),
                      "holm": _r(holm_adj[("vr", q_)], 4)} for q_, vr_, z_, p_v in vr_rows]
                  + [{"test": "Weekday effect (ANOVA)", "stat": _r(f_day, 3), "z": None, "p": _r(p_day, 4),
                      "holm": _r(holm_adj[("cal", "day")], 4)},
                     {"test": "January effect (Welch)", "stat": _r(t_jan, 3), "z": None, "p": _r(p_jan, 4),
                      "holm": _r(holm_adj[("cal", "jan")], 4)},
                     {"test": "Sell in May (Welch)", "stat": _r(t_sim, 3), "z": None, "p": _r(p_sim, 4),
                      "holm": _r(holm_adj[("cal", "sim")], 4)}]),
        "rolling": eff_roll,
        "ma_sensitivity": [{"window": w_, "cagr": _r(m_["cagr"]), "vol": _r(m_["vol"]), "sharpe": _r(m_["sharpe"]),
                            "mdd": _r(m_["mdd"]), "switches": nsw_, "invested": _r(inv_)}
                           for w_, m_, nsw_, inv_ in ma_sens]
                          + [{"window": "Buy and hold", "cagr": _r(bh_common["cagr"]), "vol": _r(bh_common["vol"]),
                              "sharpe": _r(bh_common["sharpe"]), "mdd": _r(bh_common["mdd"]), "switches": 0,
                              "invested": 1.0}],
        "ma_start": r_common.index[0].strftime("%Y-%m-%d"),
        "ma_bootstrap": {"gap": _r(diff_obs), "lo": _r(diff_lo), "hi": _r(diff_hi), "p": _r(diff_p, 4)},
    },
    "probability_of_loss": [{"horizon": name, "windows": int(n_), "empirical": _r(pl), "bootstrap": _r(boot_ploss[name])}
                            for name, n_, pl, mn, dmn, med, mx in roll_rows],
    "validation": {"reference_date": "2026-09-30", "max_gap": _r(val_max_gap),
                   "rows": [{"period": lab, "vanguard_market": _r(a_), "computed": _r(b_), "vanguard_nav": _r(nav_)}
                            for lab, a_, b_, nav_, ann_ in val_rows]},
    "reproducibility": {"data_sha256_16": data_sha, "versions": versions, "seed": RNG_SEED},
}
with open("report/analysis_results.json", "w") as f:
    json.dump(export, f, separators=(",", ":"))

# --------------------------------------------------------------------------
# Charts (vector PDF, serif font matching the Times text of the report)
# --------------------------------------------------------------------------
FIG_DIR = "report/figures"
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.edgecolor": "#b9b8b0",
    "axes.labelcolor": "#3d3c39", "xtick.color": "#3d3c39", "ytick.color": "#3d3c39",
    "legend.frameon": False, "pdf.fonttype": 42,
})
W = 6.3  # figure width in inches (= text width)
fmt_pct = FuncFormatter(lambda v, _: f"{v * 100:.0f}%")


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


def year_axis(ax, step=2):
    ax.xaxis.set_major_locator(mdates.YearLocator(step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="x", visible=False)


def c_price():
    fig, ax = plt.subplots(figsize=(W, 2.15))
    ax.plot(px.index, px, color=BLUE, lw=1.0, label="Closing price")
    ax.plot(tr.index, tr, color=ORANGE, lw=1.0, ls="--", label="Total return (adjusted price)")
    ax.set_yscale("log")
    ax.set_yticks([20, 30, 40, 60, 80, 100, 150])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_ylabel("USD (log scale)")
    year_axis(ax)
    ax.legend(loc="upper left")
    save(fig, "price")


def c_annual():
    fig, ax = plt.subplots(figsize=(W, 2.6))
    x = np.arange(len(y_tr))
    ax.bar(x - 0.2, y_px.values, 0.4, color=BLUE, label="Price")
    ax.bar(x + 0.2, y_tr.values, 0.4, color=ORANGE, hatch="////", edgecolor="white", linewidth=0,
           label="Total return")
    ax.set_xticks(x, [f"{a}*" if a in partials else str(a) for a in y_tr.index], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(fmt_pct)
    ax.axhline(0, color="#b9b8b0", lw=0.8)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2)
    save(fig, "annual")


def c_vol():
    fig, ax = plt.subplots(figsize=(W, 2.5))
    ax.plot(garch_vol.index, garch_vol, color=ORANGE, lw=0.6, alpha=0.75,
            label="GARCH(1,1)-$t$, full-sample parameters")
    ax.plot(ew_vol.index, ew_vol, color=BLUE, lw=0.9, label=r"EWMA, $\lambda$ = 0.94")
    ref = full["vol_d"]
    ax.axhline(ref, color=GREY, lw=0.8, ls="--", label=f"Full period: {pct(ref)}")
    top = max(garch_vol.max(), ew_vol.max())
    ax.set_ylim(0, np.ceil(top * 1.08 * 10) / 10)
    ax.yaxis.set_major_formatter(fmt_pct)
    ax.set_ylabel(r"Annualised volatility (daily $\times\sqrt{252}$)")
    year_axis(ax)
    ax.legend(loc="upper right", fontsize=7.5)
    save(fig, "volatility")


def c_var_bt():
    fig, ax = plt.subplots(figsize=(W, 2.4))
    q = var_methods[0.01]["GARCH(1,1)-t"]
    ax.plot(bt_idx, realised * 100, color="#9a9992", lw=0.4, label="Daily log return")
    ax.plot(bt_idx, q * 100, color=BLUE, lw=0.9, label="1% VaR, GARCH(1,1)-t")
    hits = realised < q
    ax.scatter(bt_idx[hits], realised[hits] * 100, s=7, color=RED, zorder=3, label=f"Breaches ({hits.sum()})")
    ax.set_ylabel("Daily return (%)")
    year_axis(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3, fontsize=7.5)
    save(fig, "var_backtest")


def c_start():
    fig, ax = plt.subplots(figsize=(W, 2.2))
    ax.plot(start_sens.index, start_sens, color=BLUE, lw=1.1)
    ax.axhline(full["cagr"], color=GREY, lw=0.8, ls="--")
    ax.text(start_sens.index[-1], full["cagr"] - 0.012, f"since inception: {pct(full['cagr'])}",
            color=GREY, fontsize=8, ha="right")
    ax.yaxis.set_major_formatter(fmt_pct)
    ax.set_ylabel("Annualised return to date")
    ax.set_xlabel("Start of investment (month end)")
    year_axis(ax)
    save(fig, "start_date")


def c_dd():
    fig, ax = plt.subplots(figsize=(W, 2.3))
    ax.fill_between(dd.index, dd, 0, color=RED, alpha=0.18, lw=0)
    ax.plot(dd.index, dd, color=RED, lw=0.8)
    for start, trough, end, depth in top_dd[:3]:
        ax.annotate(f"{pct(depth, 0)}\n{dt(trough)}", (trough, depth), xytext=(6, 0),
                    textcoords="offset points", fontsize=7.5, va="center", color="#3d3c39")
    ax.yaxis.set_major_formatter(fmt_pct)
    ax.set_ylabel("Decline from peak")
    year_axis(ax)
    save(fig, "drawdown")


def c_distribution():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.4), gridspec_kw={"width_ratios": [1.5, 1]})
    a1.hist(r_tr, bins=160, density=True, color=BLUE, alpha=0.75, label="Observed")
    xs = np.linspace(-0.12, 0.12, 400)
    a1.plot(xs, stats.norm.pdf(xs, mu, sd), color=ORANGE, lw=1.3, ls="--", label=r"Normal (same $\mu$, $\sigma$)")
    a1.set_xlim(-0.06, 0.06)
    a1.xaxis.set_major_formatter(fmt_pct)
    a1.set_yticks([])
    a1.legend(loc="upper left", fontsize=7.5)
    (osm, osr), _ = stats.probplot(r_tr, dist="norm")
    a2.scatter(osm, osr, s=2, color=BLUE, alpha=0.6, label="Observations", rasterized=True)
    a2.plot(osm, mu + sd * osm, color=ORANGE, lw=1.1, ls="--", label="Normal")
    a2.yaxis.set_major_formatter(fmt_pct)
    a2.set_xlabel("Theoretical normal quantiles")
    a2.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    save(fig, "distribution")


def c_acf():
    fig, ax = plt.subplots(figsize=(W, 2.1))
    k = np.arange(1, H + 1)
    ax.bar(k - 0.2, acf_r, 0.4, color=BLUE, label="Returns")
    ax.bar(k + 0.2, acf_r2, 0.4, color=ORANGE, hatch="////", edgecolor="white", lw=0, label="Squared returns")
    band = 1.96 / np.sqrt(len(lr_tr))
    ax.axhspan(-band, band, color=GREY, alpha=0.18, lw=0, label="95% band")
    ax.axhline(0, color="#b9b8b0", lw=0.8)
    ax.set_xlabel("Lag (trading days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_xticks(k)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncols=3, fontsize=7.5)
    save(fig, "acf")


def c_calendar():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw={"width_ratios": [1, 1.6]})
    a1.bar(range(5), [s.mean() * 100 for s in by_day], color=BLUE,
           yerr=[1.96 * s.std() / np.sqrt(len(s)) * 100 for s in by_day],
           capsize=3, ecolor=GREY, error_kw={"lw": 0.8})
    a1.set_xticks(range(5), ["Mon", "Tue", "Wed", "Thu", "Fri"])
    a1.axhline(0, color="#b9b8b0", lw=0.8)
    a1.set_ylabel("Mean return (%)")
    a1.grid(axis="x", visible=False)
    sd_m = r_m.groupby(r_m.index.month).std()
    n_m = r_m.groupby(r_m.index.month).count()
    a2.bar(range(12), by_month.values * 100,
           color=[ORANGE if m in range(5, 11) else BLUE for m in range(1, 13)],
           yerr=1.96 * sd_m / np.sqrt(n_m) * 100, capsize=2, ecolor=GREY, error_kw={"lw": 0.8})
    a2.set_xticks(range(12), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    a2.axhline(0, color="#b9b8b0", lw=0.8)
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "calendar")


def c_ma200():
    fig, ax = plt.subplots(figsize=(W, 2.5))
    ax.plot(val_bh.index, val_bh * cap, color=BLUE, lw=1.0, label="Buy and hold")
    ax.plot(val_ma.index, val_ma * cap, color=ORANGE, lw=1.0, ls="--", label="200-day moving average")
    ax.fill_between(sig.index, 0, 1, where=sig == 0, transform=ax.get_xaxis_transform(),
                    color=GREY, alpha=0.18, lw=0, label="Out of the market")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_ylabel("USD")
    year_axis(ax)
    ax.legend(loc="upper left", fontsize=7.5)
    save(fig, "ma200")


def c_rolling():
    fig, ax = plt.subplots(figsize=(W, 2.4))
    styles = {"1 year": (BLUE, "-"), "5 years": (ORANGE, "--"), "10 years": (GREEN, ":")}
    for name, g in roll_series.items():
        c, l = styles[name]
        ax.plot(g.index, g, color=c, ls=l, lw=1.2, label=name)
    ax.axhline(0, color="#b9b8b0", lw=0.8)
    ax.yaxis.set_major_formatter(fmt_pct)
    ax.set_ylabel("Annualised return")
    year_axis(ax)
    ax.legend(loc="lower right", ncols=3, fontsize=7.5)
    save(fig, "rolling")


def c_best():
    fig, ax = plt.subplots(figsize=(W, 1.9))
    labels = ["All days"] + [f"Without the {n} best" for n, _, _ in best_rows[1:]]
    vals = [v for _, v, _ in best_rows]
    ax.barh(range(len(vals))[::-1], vals, color=[BLUE] + [ORANGE] * 3, height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + 400, len(vals) - 1 - i, f"USD {num(v, 0)}", va="center", fontsize=8)
    ax.set_yticks(range(len(vals))[::-1], labels)
    ax.set_xlim(0, max(vals) * 1.22)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.grid(axis="y", visible=False)
    save(fig, "best_days")


def c_plan():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.4))
    paid = pd.Series(contribution, index=plan_prices.index).cumsum()
    val = (contribution / plan_prices).cumsum() * plan_prices
    a1.plot(val.index, val / 1000, color=ORANGE, lw=1.1, label="Plan value")
    a1.plot(paid.index, paid / 1000, color=GREY, lw=1.0, ls="--", label="Amount paid in")
    a1.set_ylabel("USD thousands")
    a1.legend(loc="upper left", fontsize=7.5)
    year_axis(a1, 4)
    gap = ld["ls"] - ld["dca"]
    a2.bar(gap.index, gap.values * 100, width=25, color=np.where(gap.values > 0, BLUE, ORANGE))
    a2.axhline(0, color="#b9b8b0", lw=0.8)
    a2.set_ylabel("Gap (percentage points)")
    year_axis(a2, 4)
    fig.tight_layout()
    save(fig, "savings")


def c_chf():
    fig, ax = plt.subplots(figsize=(W, 2.4))
    ax.plot(tr.index, tr / tr.iloc[0] * 100, color=BLUE, lw=1.0, label="In USD")
    ax.plot(tr_chf.index, tr_chf / tr_chf.iloc[0] * 100, color=ORANGE, lw=1.0, ls="--", label="In CHF")
    ax.set_ylabel("Index (inception = 100)")
    year_axis(ax)
    ax.legend(loc="upper left")
    save(fig, "chf")


for chart in (c_price, c_annual, c_vol, c_dd, c_distribution, c_acf, c_calendar, c_ma200,
              c_rolling, c_best, c_plan, c_chf, c_var_bt, c_start):
    chart()

# --------------------------------------------------------------------------
# LaTeX report
# --------------------------------------------------------------------------

def lx(s):
    """Make a plain formatted value safe for LaTeX (percent sign, minus sign)."""
    s = str(s).replace("%", r"\%")
    return ("$-$" + s[1:]) if s.startswith("-") else s


def tp(x, d=1, sign=False):
    return lx(pct(x, d, sign))


def tn(x, d=2):
    return lx(num(x, d))


def tpv(p):
    return "$<$ 0.001" if p < 0.001 else num(p, 3)


def ci(m):
    return f"[{tn(m['sharpe_lo'])}, {tn(m['sharpe_hi'])}]"


AUTHOR = "Cédric Masset"
v = {"author": AUTHOR, "generated": date.today().strftime("%d %B %Y"),
     "first_date": dt(idx[0]), "last_date": dt(idx[-1]), "n_days": f"{len(vt):,}"}

# ---------------------------------------------------------------- Summary
v.update({
    "px0": num(px.iloc[0]), "px1": num(px.iloc[-1]),
    "cum_px": tp(px.iloc[-1] / px.iloc[0] - 1, 0), "cum_tr": tp(tr.iloc[-1] / tr.iloc[0] - 1, 0),
    "cagr_px": tp(cagr(px)), "cagr_tr": tp(full["cagr"]), "cagr_chf": tp(cagr_chf),
    "vol": tp(full["vol"]), "vol_d": tp(full["vol_d"]), "rf_mean": tp(rf_mean),
    "sharpe": tn(full["sharpe"]), "sharpe_ci": ci(full), "sortino": tn(full["sortino"]),
    "mdd": tp(full["mdd"]), "mdd_date": dt(top_dd[0][1]), "calmar": tn(full["calmar"]),
    "ploss_emp": " / ".join(tp(r[2], 0) for r in roll_rows),
    "ploss_boot": " / ".join(tp(boot_ploss[k], 0) for k in ("1 year", "5 years", "10 years")),
})
v["takeaways"] = [
    f"Over {num(years, 1)} years, VT returned {tp(full['cagr'])} a year with dividends reinvested, of which roughly "
    f"{tp(full['cagr'] - cagr(px))} a year came from dividends. "
    + (f"These figures match Vanguard's published market-price returns to within {num(val_max_gap * 100, 2)} "
       "percentage point." if val_rows else ""),
    f"Measured on monthly returns, annualised volatility is {tp(full['vol'])}; scaling daily volatility by "
    f"$\\sqrt{{252}}$ gives {tp(full['vol_d'])} because daily returns are negatively autocorrelated. The maximum "
    f"drawdown was {tp(full['mdd'], 0)} during the global financial crisis.",
    f"The Sharpe ratio of {tn(full['sharpe'])} has a 95\\% confidence interval of {ci(full)}: eighteen years of "
    "data are barely enough to distinguish it from zero.",
    f"Daily returns have fat tails (kurtosis {tn(kurt, 1)}) and strongly persistent volatility (GARCH persistence "
    f"{num(g_persist, 3)}, half-life of about {num(g_half_life, 0)} trading days). A GARCH(1,1)-$t$ model forecasts "
    "the 5\\% Value at Risk well out of sample; rolling historical and normal VaR do not, because their breaches "
    "cluster.",
    f"After correcting for multiple testing, the only departure from a random walk is a small negative "
    f"autocorrelation of daily returns ({tn(ac_freq[0][2], 3)}); it vanishes at weekly and monthly frequencies. "
    "No calendar effect is significant, and a 200-day moving-average rule underperformed buy and hold.",
    f"For a Swiss franc investor, the weaker dollar cut the annualised return by {tp(full['cagr'] - cagr_chf)} "
    f"({tp(cagr_chf)} in CHF), before taxes.",
]

# ---------------------------------------------------------------- 1. Return and risk
best_y, worst_y = y_full.idxmax(), y_full.idxmin()
v["annual_note"] = (
    f"* Partial years: {partial_first} (from inception)"
    + (f" and {partial_last} (up to {dt(idx[-1])})." if partial_last else ".")
    + f" Of the {len(y_full)} full years, the best was {best_y} ({tp(y_full[best_y], 1, True)}) and the worst "
      f"{worst_y} ({tp(y_full[worst_y], 1, True)}); {int((y_full < 0).sum())} years were negative.")
v["vol_freq_rows"] = [[name, f"{n:,}", tp(vol)] for name, n, vol in vol_freq]
v["vol_text"] = (
    f"Volatility varies sharply across regimes. The exponentially weighted estimate ranges from "
    f"{tp(ew_vol.min(), 0)} ({mon(ew_vol.idxmin())}) to {tp(ew_vol.max(), 0)} ({dt(ew_vol.idxmax())}), and the "
    f"GARCH conditional volatility peaks at {tp(garch_vol.max(), 0)} on {dt(garch_vol.idxmax())}. Both measures "
    "react within days to a shock and decay as markets calm down. An equally weighted one-year window would "
    "instead keep an extreme day at full weight for exactly 252 sessions and then drop it abruptly, creating "
    "plateaus that do not reflect the actual level of risk. Both curves use daily returns scaled by $\\sqrt{252}$, "
    f"so the reference line is the full-period figure on the same basis ({tp(full['vol_d'])}). The GARCH curve uses "
    "parameters estimated on the whole sample and is therefore descriptive; the out-of-sample forecasts are "
    "evaluated in Section~2.\n\n"
    "The annualised level also depends on the sampling frequency. Scaling a $k$-period standard deviation by "
    "$\\sqrt{k}$ is valid only if returns are serially uncorrelated. Daily returns are negatively autocorrelated "
    f"(Section~3), so daily data annualised with $\\sqrt{{252}}$ give {tp(vol_freq[0][2])}, whereas monthly data "
    f"give {tp(vol_freq[2][2])}. The ratio of the two variances, "
    f"{num((vol_freq[2][2] / vol_freq[0][2]) ** 2, 2)}, is of the same order as the variance ratio estimated "
    f"in Section~3 at a 20-day horizon ({num(vr_rows[-1][1], 2)}), given the sampling error of only "
    f"{vol_freq[2][1]} monthly observations. This report therefore uses monthly returns for annualised volatility and for the Sharpe and Sortino "
    "ratios.")
pos = {d: i for i, d in enumerate(idx)}
v["dd_rows"] = [[dt(s_), dt(t), dt(e) if e is not None else "--", tp(d),
                 str(pos[t] - pos[s_]), str(pos[e] - pos[t]) if e is not None else "not yet"]
                for s_, t, e, d in top_dd]
v["sub_rows"] = [[f"{d0.year}--{d1.year}" if d0 is not None else "Full period", tp(m["cagr"]), tp(m["vol"]),
                  tn(m["sharpe"]), ci(m), tn(m["sortino"]), tp(m["mdd"])]
                 for d0, d1, m in sub + [(None, None, full)]]
n_zero = sum(m["sharpe_lo"] < 0 < m["sharpe_hi"] for _, _, m in sub)
n_cover_full = sum(m["sharpe_lo"] < full["sharpe"] < m["sharpe_hi"] for _, _, m in sub)
words = {0: "none", 1: "one", 2: "two", 3: "three", 4: "all four"}
v["sub_text"] = (
    "The confidence intervals are wide: with about five years of monthly data per sub-period, "
    + ("all four sub-period intervals contain" if n_cover_full == len(sub) else
       f"{words[n_cover_full]} of the four sub-period intervals contain")
    + " the full-period Sharpe ratio, and "
    + f"{words[n_zero]} contain zero. Differences between sub-periods should be read as descriptive rather than as "
    + "evidence of a change in the fund's risk-return profile.")

# ---------------------------------------------------------------- 2. Distribution, GARCH, VaR
v.update({
    "mu": tp(mu, 3), "sd": tp(sd, 2), "skew": tn(skew), "kurt": tn(kurt),
    "jb": f"{num(jb_stat, 0)} ($p$ {tpv(jb_p)})", "beyond4": str(beyond4), "expected4": num(expected4, 2),
    "worst": f"{tp(r_tr.min())} ({dt(worst_day)})", "best": f"{tp(r_tr.max(), 1, True)} ({dt(best_day)})",
})
v["dist_text"] = (
    f"The tails are far fatter than those of a normal distribution: {beyond4} days moved by more than four "
    f"standard deviations, whereas a normal distribution would predict {num(expected4, 1)} over the period. The "
    f"{'negative' if skew < 0 else 'positive'} skewness means that large declines are "
    f"{'more' if skew < 0 else 'less'} pronounced than large gains. Part of this excess kurtosis reflects "
    "time-varying volatility: returns are closer to normal once scaled by their conditional volatility, as the "
    "GARCH estimates below show.")
v["acf_text"] = (
    f"Returns show little autocorrelation beyond the first lag (Ljung--Box with 10 lags: $Q = {num(lb_r[0], 1)}$). "
    f"Squared returns, by contrast, are strongly and persistently autocorrelated ($Q = {num(lb_r2[0], 0)}$, "
    "$p < 0.001$): a large move tends to be followed by further large moves of either sign.")
names = ["$\\mu$ (\\% per day)", "$\\omega$", "$\\alpha$", "$\\beta$", "$\\nu$ (degrees of freedom)"]
v["garch_rows"] = [[n_, num(p_, 4), num(se_, 4)] for n_, p_, se_ in zip(names, g_par, g_se)]
v["garch_text"] = (
    f"The model $\\sigma_t^2 = \\omega + \\alpha\\,\\varepsilon_{{t-1}}^2 + \\beta\\,\\sigma_{{t-1}}^2$ with "
    f"standardised Student-$t$ innovations is estimated by maximum likelihood on daily log returns in percent "
    f"\\citep{{bollerslev1986}}. Persistence $\\alpha + \\beta = {num(g_persist, 3)}$ implies that a volatility shock "
    f"loses half its effect after about {num(g_half_life, 0)} trading days. The estimated $\\nu = {num(g_nu, 1)}$ "
    "confirms that, even after accounting for volatility clustering, daily shocks have fatter tails than the normal "
    f"distribution. The implied long-run volatility, {tp(g_uncond_vol)}, uses daily scaling and is therefore "
    "comparable with the daily-scaled figure in Section~1.")
v["n_bt"] = f"{n_bt:,}"
v["n_family"] = n_family
v["bt_start"] = dt(bt_idx[0])
v["bt_rows"] = [[tp(l, 0), name, tp(rate, 2), f"{k} / {num(l * n_bt, 0)}", tpv(puc), tpv(pind), tpv(pcc)]
                for l, name, rate, k, puc, pind, pcc in bt_rows]
bt = {(l, n_): (rate, puc, pind, pcc) for l, n_, rate, k, puc, pind, pcc in bt_rows}
g5, g1 = bt[(0.05, "GARCH(1,1)-t")], bt[(0.01, "GARCH(1,1)-t")]
v["bt_text"] = (
    "Rolling historical and normal VaR react slowly to changes in volatility. Their breach rates at 5\\% are close "
    "to the target, but the breaches cluster in turbulent periods, so the independence test of "
    "\\citet{christoffersen1998} rejects them decisively. At 1\\%, both also fail the unconditional coverage test "
    f"of \\citet{{kupiec1995}}: {tp(bt[(0.01, 'Historical (250 days)')][0], 2)} and "
    f"{tp(bt[(0.01, 'Normal (250 days)')][0], 2)} of breaches instead of 1\\%. "
    f"The GARCH(1,1)-$t$ forecasts pass both tests at 5\\% (breach rate {tp(g5[0], 2)}; $p$-values "
    f"{tpv(g5[1])} and {tpv(g5[2])}). At 1\\%, their breach rate of {tp(g1[0], 2)} is acceptable "
    f"($p = {tpv(g1[1])}$), but extreme losses still arrive in clusters ($p = {tpv(g1[2])}$ for independence). A "
    "symmetric $t$ distribution cannot reproduce the negative skewness of large losses; a skewed distribution or an "
    "extreme-value model would be the natural next step.")
v["bt_note"] = (
    f"One-day VaR on daily log returns, forecasts from {dt(bt_idx[0])} to {dt(bt_idx[-1])} ({n_bt:,} days), each "
    "using only information available the day before. Historical and normal VaR use a 250-day rolling window. "
    "GARCH parameters are re-estimated at the start of each calendar year on all prior data, and the conditional "
    "variance is updated daily. The GARCH forecasts use a zero conditional mean, standard at a one-day horizon; with "
    f"the maximum-likelihood location estimate instead ({num(g_mu, 3)}\\% a day, well above the sample mean of "
    f"{num(lr_tr.mean() * 100, 3)}\\% because of negative skewness), the breach rates rise to "
    f"{tp(mle_breach[0.05], 2)} and {tp(mle_breach[0.01], 2)}. Kupiec: unconditional coverage; Christoffersen: "
    "independence; CC: conditional coverage (joint).")

# ---------------------------------------------------------------- 3. Efficiency
v["ac_rows"] = [[name, f"{n:,}", tn(r1, 3), tn(t, 2), tn(ac_rob[name][1], 2), tpv(ac_rob[name][2]),
                 tpv(holm_adj[("ac", name)])] for name, n, r1, t, p in ac_freq]
v["ac_text"] = (
    f"Daily autocorrelation is negative ({tn(ac_freq[0][2], 3)}): a gain tends to be slightly reversed the following "
    "day. The naive $t$-statistic overstates its significance; the heteroskedasticity-robust statistic is "
    f"{tn(ac_rob['Daily'][1], 2)}, still significant after the Holm correction. The effect is concentrated in crises: "
    f"excluding 2008--2009 and February to June 2020, the autocorrelation falls to {tn(ac1_non_crisis, 3)}. "
    "Plausible explanations, which these data cannot separate, are overreaction on turbulent days and temporary "
    "gaps between the ETF's closing price and the value of its foreign holdings, which trade at different hours. "
    "Testing the second would require daily net asset values. At weekly and monthly frequencies, no autocorrelation "
    "is detectable.")
v["vr_rows"] = [[str(q), num(vr, 3), tn(z, 2), tpv(p), tpv(holm_adj[("vr", q)]),
                 "not rejected" if holm_adj[("vr", q)] >= 0.05 else ("mean reversion" if vr < 1 else "persistence")]
                for q, vr, z, p in vr_rows]
v["mt_text"] = (
    f"Section~3 reports {n_family} hypothesis tests. At the 5\\% level, about one false rejection is expected by "
    "chance alone, so each $p$-value is also adjusted with the step-down procedure of \\citet{holm1979}, which "
    f"controls the probability of any false rejection. {n_raw_sig} tests are significant before adjustment and "
    f"{n_holm_sig} after. Both remaining rejections concern the same first-order daily autocorrelation, since "
    "$VR(2) \\approx 1 + \\hat\\rho_1$.")
v.update({
    "cal_day": f"$F = {num(f_day, 2)}$", "p_day": tpv(p_day), "h_day": tpv(holm_adj[("cal", "day")]),
    "cal_jan": f"{tp(jan.mean(), 2)} vs {tp(other.mean(), 2)} per month", "p_jan": tpv(p_jan),
    "h_jan": tpv(holm_adj[("cal", "jan")]),
    "n_halves": f"{len(winter)} and {len(summer)} half-years",
    "cal_sim": f"{tp(winter.mean(), 1)} vs {tp(summer.mean(), 1)}", "p_sim": tpv(p_sim),
    "h_sim": tpv(holm_adj[("cal", "sim")]),
    "cal_note": "With fewer than twenty observations per calendar month, these tests have little power: the "
                "absence of significance does not prove the absence of an effect, only that any effect is too "
                "small to detect in eighteen years of data.",
})
v["ma_start"] = dt(r_common.index[0])
v["ma"] = [[tp(m_bh["cagr"]), tp(m_ma["cagr"])], [tp(m_bh["vol"]), tp(m_ma["vol"])],
           [f"{tn(m_bh['sharpe'])} {ci(m_bh)}", f"{tn(m_ma['sharpe'])} {ci(m_ma)}"],
           [tp(m_bh["mdd"]), tp(m_ma["mdd"])],
           [f"USD {num(val_bh.iloc[-1] * cap, 0)}", f"USD {num(val_ma.iloc[-1] * cap, 0)}"]]
v["ma_sens_rows"] = ([[f"{w} days", tp(m["cagr"]), tp(m["vol"]), tn(m["sharpe"]), tp(m["mdd"]), str(nsw),
                       tp(inv, 0)] for w, m, nsw, inv in ma_sens]
                     + [["Buy and hold", tp(bh_common["cagr"]), tp(bh_common["vol"]), tn(bh_common["sharpe"]),
                         tp(bh_common["mdd"]), "0", "100\\%"]])
v["ma_note"] = (
    "Rule: invested in VT when the previous day's close is above its moving average, otherwise held in T-bills; "
    f"switches are executed at that same close and cost {tp(TRANSACTION_COST, 2)} each. All windows are compared "
    f"over the same period, starting {dt(r_common.index[0])}, the first day on which a 250-day average is "
    "available.")
all_under = all(m["cagr"] < bh_common["cagr"] for _, m, _, _ in ma_sens)
v["ma_text"] = (
    f"{'Every window underperformed' if all_under else 'Most windows underperformed'} buy and hold, by {tp(bh_common['cagr'] - max(m['cagr'] for _, m, _, _ in ma_sens))} "
    f"to {tp(bh_common['cagr'] - min(m['cagr'] for _, m, _, _ in ma_sens))} a year, while reducing the maximum "
    f"drawdown. For the 200-day rule, a circular block bootstrap ({B:,} replications, 63-day blocks) gives a 95\\% "
    f"confidence interval of [{tp(diff_lo)}, {tp(diff_hi)}] for the annual return gap ($p = {tpv(diff_p)}$): the "
    "underperformance is borderline significant. The sample starts in mid-2009, after the financial-crisis crash "
    "that such rules are designed to avoid, which works against them; a longer index history would be needed for "
    "a fair assessment. Late and repeated signals (``whipsaws'') are costly in a market that recovers quickly.")

# ---------------------------------------------------------------- 4. Investor
v["roll_rows"] = [[name, str(n), tp(pl, 0), tp(boot_ploss[name], 0), f"{tp(mn)} ({mon(dmn)})", tp(med), tp(mx)]
                  for name, n, pl, mn, dmn, med, mx in roll_rows]
v["roll_text"] = (
    "Overlapping windows understate the probability of loss at long horizons: the 10-year windows contain only "
    "about one and a half independent periods, none of which happened to lose money. A stationary bootstrap "
    f"\\citep{{politis1994}} that resamples blocks of monthly returns (mean length 12 months, 20,000 paths) "
    f"gives a more defensible estimate: {tp(boot_ploss['5 years'], 0)} over five years and "
    f"{tp(boot_ploss['10 years'], 0)} over ten. The bootstrap still assumes that the next decades resemble the "
    "past eighteen years.")
lo_y = min(start_sens_annual, key=start_sens_annual.get)
hi_y = max(start_sens_annual, key=start_sens_annual.get)
v["start_text"] = (
    f"Because VT was launched just before the 2008 crash, ``since inception'' figures are among the least "
    f"favourable available. Starting at any year end between {min(start_sens_annual)} and "
    f"{max(start_sens_annual)} would have delivered between {tp(start_sens_annual[lo_y])} ({lo_y}) and "
    f"{tp(start_sens_annual[hi_y])} ({hi_y}) a year to date, against {tp(full['cagr'])} since inception. "
    "Performance statistics should always be read together with their start and end dates.")
v["best_text"] = (
    f"Missing the 10 best days out of {len(r_tr):,} cuts the annualised return from {tp(best_rows[0][2])} to "
    f"{tp(best_rows[1][2])}. The reverse exercise is just as striking: avoiding the 10 worst days would have lifted "
    f"the final value to USD {num(worst_rows[0][1], 0)}. {best_near_worst} of the 10 best days fall within two "
    "weeks of one of the 10 worst. They occur in the same crisis periods: avoiding the worst days without missing "
    "the best would require perfect timing.")
v["contribution"] = str(contribution)
v["plan_text"] = (
    f"Investing USD {contribution} at the end of every month since inception ({len(plan_prices)} contributions, "
    f"USD {num(plan_invested, 0)} in total) would be worth USD {num(plan_value, 0)} today, an internal rate of "
    f"return of {tp(plan_irr)} a year, above the {tp(full['cagr'])} of a single investment at inception because "
    "most of the money was invested after the 2008--2009 crash.\n\n"
    f"Across the {len(ld)} possible monthly starting points, investing a sum all at once beat spreading it over "
    f"12 months (with the uninvested cash earning the T-bill rate) in {tp(ls_win_share, 0)} of cases, with an "
    f"average advantage of {tp(ls_mean_gap, 1)} after 12 months. Because the market rises most of the time, "
    "waiting costs money on average; averaging in mainly limits the regret of investing just before a decline. The "
    "starting points overlap, so this frequency is descriptive rather than a precise probability.")

# ---------------------------------------------------------------- 5. CHF
v.update({"cagr_usd": tp(cagr_usd), "vol_chf": tp(vol_chf),
          "fx_move": f"{num(chf.iloc[0], 4)} $\\rightarrow$ {num(chf.iloc[-1], 4)} ({tp(cagr_fx, 1, True)} a year)",
          "corr_w": tn(corr_fx_w, 2), "corr_d": tn(corr_fx, 2)})
v["chf_text"] = (
    f"The dollar depreciated against the franc, costing an unhedged Swiss investor about {tp(cagr_usd - cagr_chf)} "
    f"of return a year. The weekly correlation between VT and USD/CHF is {tn(corr_fx_w, 2)}; the daily correlation, "
    f"{tn(corr_fx, 2)}, is biased towards zero because Yahoo's daily currency and ETF closes are taken at different "
    "times of day. "
    + ("The dollar tends to strengthen against the franc when equities fall, which cushions part of the losses in "
       "CHF. " if corr_fx_w < 0 else "The franc tends to strengthen when equities fall, which deepens losses in CHF. ")
    + "Because VT is a global fund, its true exposure is to a basket of currencies: USD/CHF captures only part of the "
      "currency effect. All figures are before Swiss income tax and the 15\\% US withholding tax on dividends, which "
      "a Swiss resident can usually credit against Swiss tax.")

# ---------------------------------------------------------------- 6. Validation
v["val_rows"] = [[lab, tp(a, 2), tp(b, 2), lx(f"{(b - a) * 100:+.2f}"), tp(nav, 2)]
                 for lab, a, b, nav, ann in val_rows]
v["val_available"] = bool(val_rows)
v["val_text"] = (
    "Yahoo's adjusted prices are not an official source, so the total returns used in this report were checked "
    "against the market-price returns published by Vanguard. For all standard periods the two agree to within "
    f"{num(val_max_gap * 100, 2)} percentage point. The small gap since inception arises because Vanguard measures "
    "from the launch on 24 June 2008, two trading days before the first Yahoo closing price. The ETF's market "
    "price and its net asset value (NAV) also track each other closely." if val_rows else
    "Validation data are not available for the current sample.")

# ---------------------------------------------------------------- Reproducibility
v["method_text"] = (
    "Source: Yahoo Finance, downloaded with the \\texttt{yfinance} library and updated automatically (GitHub "
    "repository \\url{https://github.com/cedricmasset98/donnees-marche}). VT: daily closing prices and prices "
    "adjusted for dividends and splits (``Adj Close''). Risk-free rate: 13-week US T-bill (\\texttt{\\^{}IRX}). "
    "Currency: USD/CHF (\\texttt{CHF=X}). Annualised returns use calendar time; annualised volatility, Sharpe and "
    "Sortino ratios use monthly returns; Sharpe confidence intervals follow \\citet{mertens2002} and "
    f"\\citet{{opdyke2007}}, allowing for skewness and kurtosis. Bootstrap results use the fixed seed {RNG_SEED}. "
    f"Data file fingerprint (SHA-256 of \\texttt{{data/VT.csv}}): \\texttt{{{data_sha}}}. Software: {versions}. "
    "These analyses are descriptive and historical; they do not constitute investment advice.")

env = jinja2.Environment(
    block_start_string=r"\BLOCK{", block_end_string="}", variable_start_string=r"\VAR{",
    variable_end_string="}", comment_start_string=r"\#{", comment_end_string="}",
    trim_blocks=True, lstrip_blocks=True, autoescape=False, undefined=jinja2.StrictUndefined,
    loader=jinja2.FileSystemLoader("report"))
with open("report/VT_analysis.tex", "w", encoding="utf-8") as f:
    f.write(env.get_template("template.tex").render(**v))

subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-cd",
                "-outdir=build", "report/VT_analysis.tex"], check=True, stdout=subprocess.DEVNULL)
shutil.copy("report/build/VT_analysis.pdf", "report/VT_analysis.pdf")
print("report/VT_analysis.pdf created")
