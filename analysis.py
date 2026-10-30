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


def metrics(r, rfq):
    """r: daily returns; rfq: aligned daily risk-free rate.

    The annualised return uses calendar time (from the close preceding the first
    return to the last close), consistent with cagr(). The Sharpe ratio divides the
    mean excess return by the volatility of excess returns; Sortino uses the
    downside deviation of excess returns (target = risk-free rate).
    """
    ex = r - rfq
    vol = r.std() * np.sqrt(DAYS_PER_YEAR)
    down_vol = np.sqrt((ex.clip(upper=0) ** 2).mean()) * np.sqrt(DAYS_PER_YEAR)
    value = (1 + r).cumprod()
    start = idx[idx.get_loc(r.index[0]) - 1]
    ca = value.iloc[-1] ** (365.25 / (r.index[-1] - start).days) - 1
    mdd = drawdown(pd.concat([pd.Series([1.0]), value.reset_index(drop=True)])).min()
    return {"cagr": ca, "vol": vol,
            "sharpe": ex.mean() / ex.std() * np.sqrt(DAYS_PER_YEAR),
            "sortino": ex.mean() * DAYS_PER_YEAR / down_vol, "mdd": mdd, "calmar": ca / abs(mdd)}


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
vol_chf = tr_chf.pct_change().std() * np.sqrt(DAYS_PER_YEAR)
corr_fx = r_tr.corr(chf.pct_change().reindex(r_tr.index))

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
    fig, ax = plt.subplots(figsize=(W, 2.3))
    ax.plot(roll_vol.index, roll_vol, color=BLUE, lw=1.0)
    ax.axhline(full["vol"], color=GREY, lw=0.8, ls="--")
    ax.text(roll_vol.index[-1], full["vol"] + 0.012, f"full-period average: {pct(full['vol'])}",
            color=GREY, fontsize=8, ha="right")
    ax.yaxis.set_major_formatter(fmt_pct)
    ax.set_ylabel("Annualised volatility")
    year_axis(ax)
    save(fig, "volatility")


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
              c_rolling, c_best, c_plan, c_chf):
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


AUTHOR = "Cédric Masset"
v = {"author": AUTHOR, "generated": date.today().strftime("%d %B %Y"),
     "first_date": dt(idx[0]), "last_date": dt(idx[-1]), "n_days": f"{len(vt):,}"}

# Summary
v.update({
    "px0": num(px.iloc[0]), "px1": num(px.iloc[-1]),
    "cum_px": tp(px.iloc[-1] / px.iloc[0] - 1, 0), "cum_tr": tp(tr.iloc[-1] / tr.iloc[0] - 1, 0),
    "cagr_px": tp(cagr(px)), "cagr_tr": tp(full["cagr"]), "cagr_chf": tp(cagr_chf),
    "vol": tp(full["vol"]), "rf_mean": tp(rf_mean), "sharpe": tn(full["sharpe"]),
    "sortino": tn(full["sortino"]), "mdd": tp(full["mdd"]), "mdd_date": dt(top_dd[0][1]),
    "calmar": tn(full["calmar"]), "ploss": " / ".join(tp(r[2], 0) for r in roll_rows),
})
v["takeaways"] = [
    f"Over {num(years, 1)} years, VT returned {tp(full['cagr'])} a year with dividends reinvested, of which "
    f"roughly {tp(full['cagr'] - cagr(px))} a year came from dividends.",
    f"The price of that return was {tp(full['vol'], 0)} annual volatility and a maximum drawdown of "
    f"{tp(abs(full['mdd']), 0)} during the global financial crisis.",
    f"Daily returns are clearly non-normal (kurtosis of {tn(kurt, 1)}) and volatility clusters over time; their "
    "direction is only weakly predictable from one day to the next, and not at all over longer horizons.",
    f"For a Swiss franc investor, the weaker dollar cut the annualised return by {tp(full['cagr'] - cagr_chf)} a "
    f"year ({tp(cagr_chf)} in CHF).",
]

# 1. Return and risk
best_y, worst_y = y_full.idxmax(), y_full.idxmin()
v["annual_note"] = (
    f"* Partial years: {partial_first} (from inception)"
    + (f" and {partial_last} (up to {dt(idx[-1])})." if partial_last else ".")
    + f" Of the {len(y_full)} full years, the best was {best_y} ({tp(y_full[best_y], 1, True)}) and the worst "
      f"{worst_y} ({tp(y_full[worst_y], 1, True)}); {int((y_full < 0).sum())} years were negative.")
v["vol_text"] = (
    f"Volatility varies sharply across regimes: from {tp(roll_vol.min(), 0)} at its calmest (year ending "
    f"{dt(roll_vol.idxmin())}) to {tp(roll_vol.max(), 0)} at its most turbulent (year ending "
    f"{dt(roll_vol.idxmax())}). A single full-period average therefore hides very different environments.")
pos = {d: i for i, d in enumerate(idx)}
v["dd_rows"] = [[dt(s), dt(t), dt(e) if e is not None else "--", tp(d),
                 str(pos[t] - pos[s]), str(pos[e] - pos[t]) if e is not None else "not yet"]
                for s, t, e, d in top_dd]
v["sub_rows"] = [[f"{d0.year}--{d1.year}" if d0 is not None else "Full period", tp(m["cagr"]), tp(m["vol"]),
                  tn(m["sharpe"]), tn(m["sortino"]), tp(m["mdd"]), tn(m["calmar"])]
                 for d0, d1, m in sub + [(None, None, full)]]

# 2. Distribution
v.update({
    "mu": tp(mu, 3), "sd": tp(sd, 2), "skew": tn(skew), "kurt": tn(kurt),
    "jb": f"{num(jb_stat, 0)} ($p$ {tpv(jb_p)})", "beyond4": str(beyond4), "expected4": num(expected4, 2),
    "worst": f"{tp(r_tr.min())} ({dt(worst_day)})", "best": f"{tp(r_tr.max(), 1, True)} ({dt(best_day)})",
})
v["dist_text"] = (
    f"The tails are far fatter than those of a normal distribution: {beyond4} days moved by more than four "
    f"standard deviations, whereas a normal distribution would predict {num(expected4, 1)} over the period. The "
    f"{'negative' if skew < 0 else 'positive'} skewness means that large declines are "
    f"{'more' if skew < 0 else 'less'} pronounced than large gains.")
v["var_rows"] = [[tp(l, 0), tp(q, 2), tp(es, 2), tp(qn, 2), tp(esn, 2), f"{tp(br, 2)} (expected {tp(1 - l, 0)})"]
                 for l, q, es, qn, esn, br in var_rows]
v["var_text"] = (
    "At 99\\%, the parametric normal VaR understates risk: it is breached more often than expected, and the "
    "historical Expected Shortfall is markedly higher. This is the direct consequence of the fat tails documented "
    "above." if var_rows[1][5] > 0.01 else "The normal VaR remains close to the historical VaR over this period.")
v["acf_text"] = (
    f"Apart from a small effect at the first lag (see Section~3), returns show little autocorrelation (Ljung--Box "
    f"with 10 lags: $Q = {num(lb_r[0], 1)}$). Squared returns, by contrast, are strongly and persistently "
    f"autocorrelated ($Q = {num(lb_r2[0], 0)}$, $p < 0.001$). A large move tends to be followed by further large "
    "moves of either sign, the behaviour captured by ARCH and GARCH models.")

# 3. Efficiency
v["ac_rows"] = [[name, f"{n:,}", tn(r1, 3), tn(t, 2), tpv(p)] for name, n, r1, t, p in ac_freq]
v["ac_text"] = (
    f"Daily autocorrelation is negative ({tn(ac_freq[0][2], 3)}): a gain tends to be slightly reversed the "
    f"following day. The effect is concentrated in crises: excluding 2008--2009 and February to June 2020, it "
    f"falls to {tn(ac1_non_crisis, 3)}, still negative but less than half as large. Possible explanations include "
    "overreaction on turbulent days, or temporary gaps between the ETF price and the value of its foreign "
    "holdings, which trade at different hours. At weekly and monthly frequencies, no significant "
    "autocorrelation appears.")
v["vr_rows"] = [[str(q), num(vr, 3), tn(z, 2), tpv(p),
                 "not rejected" if p >= 0.05 else ("mean reversion" if vr < 1 else "persistence")]
                for q, vr, z, p in vr_rows]
rejected = [q for q, vr, z, p in vr_rows if p < 0.05]
v["vr_text"] = (("The random-walk hypothesis is not rejected at any horizon. " if not rejected else
                 f"The random-walk hypothesis is rejected at the 5\\% level for $q = {', '.join(map(str, rejected))}$. ")
                + "Even where a deviation is statistically significant, it is small in magnitude and would need to "
                  "survive transaction costs to be exploitable.")
v.update({
    "cal_day": f"$F = {num(f_day, 2)}$", "p_day": tpv(p_day),
    "cal_jan": f"{tp(jan.mean(), 2)} vs {tp(other.mean(), 2)} per month", "p_jan": tpv(p_jan),
    "n_halves": f"{len(winter)} and {len(summer)} half-years",
    "cal_sim": f"{tp(winter.mean(), 1)} vs {tp(summer.mean(), 1)} per half-year", "p_sim": tpv(p_sim),
    "cal_note": "With fewer than twenty observations per calendar month, differences between months are dominated "
                "by statistical noise."
                + (" None of the three effects is significant at the 5\\% level." if min(p_day, p_jan, p_sim) >= 0.05 else ""),
})
v["ma"] = [[tp(m_bh["cagr"]), tp(m_ma["cagr"])], [tp(m_bh["vol"]), tp(m_ma["vol"])],
           [tn(m_bh["sharpe"]), tn(m_ma["sharpe"])], [tp(m_bh["mdd"]), tp(m_ma["mdd"])],
           [f"USD {num(val_bh.iloc[-1] * cap, 0)}", f"USD {num(val_ma.iloc[-1] * cap, 0)}"]]
v["ma_note"] = (
    f"Rule: invested in VT when the previous day's close is above its 200-day moving average, otherwise held in "
    f"T-bills; switches are executed at that same close. Cost of {tp(TRANSACTION_COST, 2)} per switch, {n_switch} "
    f"switches in total; the strategy was invested {tp(share_invested, 0)} of the time. Common period starting "
    f"{dt(sig.index[0])}, since the moving average needs 200 days of history.")
v["ma_text"] = (
    f"The strategy {'reduces' if m_ma['mdd'] > m_bh['mdd'] else 'does not reduce'} the maximum drawdown "
    f"({tp(m_ma['mdd'], 0)} vs {tp(m_bh['mdd'], 0)}), "
    + (f"but at the cost of {tp(m_bh['cagr'] - m_ma['cagr'])} a year in return. " if m_ma["cagr"] < m_bh["cagr"]
       else f"and adds {tp(m_ma['cagr'] - m_bh['cagr'])} a year in return. ")
    + "It often exits after a fall and re-enters after the rebound: late and repeated signals (``whipsaws'') are "
      "costly in a market that recovers quickly.")

# 4. Investor
v["roll_rows"] = [[name, str(n), tp(pl, 0), f"{tp(mn)} ({mon(dmn)})", tp(med), tp(mx)]
                  for name, n, pl, mn, dmn, med, mx in roll_rows]
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
    f"return of {tp(plan_irr)} a year.\n\n"
    f"Across the {len(ld)} possible monthly starting points, investing a sum all at once beat spreading it over "
    f"12 months (with the uninvested cash earning the T-bill rate) in {tp(ls_win_share, 0)} of cases, with an "
    f"average advantage of {tp(ls_mean_gap, 1)} after 12 months. Because the market rises most of the time, "
    "waiting costs money on average; averaging in mainly limits the regret of investing just before a decline.")

# 5. CHF
v.update({"cagr_usd": tp(cagr_usd), "vol_chf": tp(vol_chf),
          "fx_move": f"{num(chf.iloc[0], 4)} $\\rightarrow$ {num(chf.iloc[-1], 4)} ({tp(cagr_fx, 1, True)} a year)"})
v["chf_text"] = (
    f"The dollar depreciated against the franc, costing an unhedged Swiss investor about {tp(cagr_usd - cagr_chf)} "
    f"of return a year. The daily correlation between VT and USD/CHF is {tn(corr_fx, 2)}: "
    + ("the franc tends to strengthen when equities fall, which deepens losses in CHF during crises. "
       if corr_fx > 0 else
       "the dollar tends to strengthen slightly when equities fall, which cushions losses in CHF a little, but the "
       "relationship is weak. ")
    + "Because VT is a global fund, its true exposure is to a basket of currencies: USD/CHF captures only part of "
      "the currency effect.")
v["method_text"] = (
    f"Source: Yahoo Finance, downloaded with the \\texttt{{yfinance}} library and updated automatically (GitHub "
    f"repository \\url{{https://github.com/cedricmasset98/donnees-marche}}). VT: daily closing prices and prices adjusted for "
    f"dividends and splits (``Adj Close''), from {dt(idx[0])} to {dt(idx[-1])}. Risk-free rate: 13-week US T-bill "
    f"(\\texttt{{\\^{{}}IRX}}), converted to a daily rate by dividing by 252. Currency: USD/CHF "
    f"(\\texttt{{CHF=X}}), last available rate for each VT trading day. Annualised volatilities use 252 trading "
    f"days; annualised returns use calendar time. These analyses are descriptive and historical; they do not "
    f"constitute investment advice.")

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
