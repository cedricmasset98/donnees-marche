"""
Full analysis of the Vanguard Total World Stock ETF (VT) and PDF report.

Reads only the CSV files in data/ (no request to Yahoo):
  data/VT.csv    VT daily prices (Close, Adj Close)
  data/IRX.csv   13-week T-bill rate, annualised % (risk-free rate)
  data/CHFX.csv  USD/CHF (CHF per 1 USD)

Output: report/VT_analysis.pdf
"""
import io
import os
from datetime import date

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from scipy import stats, optimize

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

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
    """r: daily returns; rfq: aligned daily risk-free rate."""
    ex = r - rfq
    vol = r.std() * np.sqrt(DAYS_PER_YEAR)
    neg = ex[ex < 0]
    down_vol = np.sqrt((neg ** 2).sum() / len(ex)) * np.sqrt(DAYS_PER_YEAR)
    value = (1 + r).cumprod()
    ca = value.iloc[-1] ** (DAYS_PER_YEAR / len(r)) - 1
    mdd = drawdown(pd.concat([pd.Series([1.0]), value.reset_index(drop=True)])).min()
    return {"cagr": ca, "vol": vol, "sharpe": ex.mean() * DAYS_PER_YEAR / vol,
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
# Charts
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "legend.frameon": False,
})
fmt_pct = FuncFormatter(lambda v, _: f"{v * 100:.0f}%")


def as_image(fig, width=17 * cm):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    w, h = fig.get_size_inches()
    return Image(buf, width=width, height=width * h / w)


def year_axis(ax, step=2):
    ax.xaxis.set_major_locator(mdates.YearLocator(step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="x", visible=False)


def c_price():
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(px.index, px, color=BLUE, lw=1.1, label="Closing price")
    ax.plot(tr.index, tr, color=ORANGE, lw=1.1, ls="--", label="Total return (adjusted price)")
    ax.set_yscale("log")
    ax.set_yticks([20, 30, 40, 60, 80, 100, 150])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_ylabel("USD (log scale)")
    year_axis(ax)
    ax.legend(loc="upper left")
    return as_image(fig)


def c_annual():
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(y_tr))
    ax.bar(x - 0.2, y_px.values, 0.4, color=BLUE, label="Price")
    ax.bar(x + 0.2, y_tr.values, 0.4, color=ORANGE, hatch="///", edgecolor="white", linewidth=0,
           label="Total return")
    ax.set_xticks(x, [f"{a}*" if a in partials else str(a) for a in y_tr.index], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(fmt_pct)
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2)
    return as_image(fig)


def c_vol():
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(roll_vol.index, roll_vol, color=BLUE, lw=1.1)
    ax.axhline(full["vol"], color=GREY, lw=0.8, ls="--")
    ax.text(roll_vol.index[-1], full["vol"] + 0.012, f"average: {pct(full['vol'])}",
            color=GREY, fontsize=8, ha="right")
    ax.yaxis.set_major_formatter(fmt_pct)
    ax.set_ylabel("Annualised volatility (1-year rolling)")
    year_axis(ax)
    return as_image(fig)


def c_dd():
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.fill_between(dd.index, dd, 0, color=RED, alpha=0.18, lw=0)
    ax.plot(dd.index, dd, color=RED, lw=0.9)
    for start, trough, end, depth in top_dd[:3]:
        ax.annotate(f"{pct(depth, 0)}\n{dt(trough)}", (trough, depth), xytext=(6, 0),
                    textcoords="offset points", fontsize=7.5, va="center", color="#52514e")
    ax.yaxis.set_major_formatter(fmt_pct)
    ax.set_ylabel("Decline from previous peak")
    year_axis(ax)
    return as_image(fig)


def c_distribution():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 2.8), gridspec_kw={"width_ratios": [1.5, 1]})
    a1.hist(r_tr, bins=160, density=True, color=BLUE, alpha=0.75, label="Observed returns")
    xs = np.linspace(-0.12, 0.12, 400)
    a1.plot(xs, stats.norm.pdf(xs, mu, sd), color=ORANGE, lw=1.4, ls="--", label="Normal (same μ, σ)")
    a1.set_xlim(-0.06, 0.06)
    a1.xaxis.set_major_formatter(fmt_pct)
    a1.set_yticks([])
    a1.legend(loc="upper left", fontsize=7.5)
    a1.set_title("Distribution of daily returns", fontsize=9)
    (osm, osr), _ = stats.probplot(r_tr, dist="norm")
    a2.scatter(osm, osr, s=3, color=BLUE, alpha=0.6, label="Observations")
    a2.plot(osm, mu + sd * osm, color=ORANGE, lw=1.2, ls="--", label="Normal")
    a2.yaxis.set_major_formatter(fmt_pct)
    a2.set_xlabel("Theoretical quantiles (normal)")
    a2.set_title("Q-Q plot", fontsize=9)
    a2.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    return as_image(fig)


def c_acf():
    fig, ax = plt.subplots(figsize=(8, 2.4))
    k = np.arange(1, H + 1)
    ax.bar(k - 0.2, acf_r, 0.4, color=BLUE, label="Returns")
    ax.bar(k + 0.2, acf_r2, 0.4, color=ORANGE, hatch="///", edgecolor="white", lw=0,
           label="Squared returns")
    band = 1.96 / np.sqrt(len(lr_tr))
    ax.axhspan(-band, band, color=GREY, alpha=0.15, lw=0, label="95% band")
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_xticks(k)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncols=3, fontsize=7.5)
    return as_image(fig)


def c_calendar():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 2.9), gridspec_kw={"width_ratios": [1, 1.6]})
    a1.bar(range(5), [s.mean() * 100 for s in by_day], color=BLUE,
           yerr=[1.96 * s.std() / np.sqrt(len(s)) * 100 for s in by_day],
           capsize=3, ecolor=GREY, error_kw={"lw": 0.8})
    a1.set_xticks(range(5), ["Mon", "Tue", "Wed", "Thu", "Fri"])
    a1.axhline(0, color="#c3c2b7", lw=0.8)
    a1.set_ylabel("Mean return (%)")
    a1.set_title("By weekday", fontsize=9)
    a1.grid(axis="x", visible=False)
    sd_m = r_m.groupby(r_m.index.month).std()
    n_m = r_m.groupby(r_m.index.month).count()
    a2.bar(range(12), by_month.values * 100,
           color=[ORANGE if m in range(5, 11) else BLUE for m in range(1, 13)],
           yerr=1.96 * sd_m / np.sqrt(n_m) * 100, capsize=2, ecolor=GREY, error_kw={"lw": 0.8})
    a2.set_xticks(range(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
                              "Oct", "Nov", "Dec"], fontsize=7.5)
    a2.axhline(0, color="#c3c2b7", lw=0.8)
    a2.set_title("By month (orange: May to October)", fontsize=9)
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    return as_image(fig)


def c_ma200():
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(val_bh.index, val_bh * cap, color=BLUE, lw=1.1, label="Buy and hold")
    ax.plot(val_ma.index, val_ma * cap, color=ORANGE, lw=1.1, ls="--", label="200-day moving average")
    ax.fill_between(sig.index, 0, 1, where=sig == 0, transform=ax.get_xaxis_transform(),
                    color=GREY, alpha=0.15, lw=0, label="Out of the market (T-bills)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_ylabel("Value of USD 10,000")
    year_axis(ax)
    ax.legend(loc="upper left", fontsize=7.5)
    return as_image(fig)


def c_rolling():
    fig, ax = plt.subplots(figsize=(8, 3))
    styles = {"1 year": (BLUE, "-"), "5 years": (ORANGE, "--"), "10 years": (GREEN, ":")}
    for name, g in roll_series.items():
        c, l = styles[name]
        ax.plot(g.index, g, color=c, ls=l, lw=1.3, label=f"{name} (annualised)")
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.yaxis.set_major_formatter(fmt_pct)
    ax.set_xlabel("End of holding period")
    year_axis(ax)
    ax.legend(loc="lower right", fontsize=7.5)
    return as_image(fig)


def c_best():
    fig, ax = plt.subplots(figsize=(8, 2.6))
    labels = ["All days"] + [f"Without the {n} best" for n, _, _ in best_rows[1:]]
    vals = [v for _, v, _ in best_rows]
    ax.barh(range(len(vals))[::-1], vals, color=[BLUE] + [ORANGE] * 3, height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + 300, len(vals) - 1 - i, f"USD {num(v, 0)}", va="center", fontsize=8)
    ax.set_yticks(range(len(vals))[::-1], labels)
    ax.set_xlim(0, max(vals) * 1.22)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Final value of USD 10,000 invested on {dt(idx[0])}")
    return as_image(fig)


def c_plan():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 2.9))
    paid = pd.Series(contribution, index=plan_prices.index).cumsum()
    val = (contribution / plan_prices).cumsum() * plan_prices
    a1.plot(val.index, val / 1000, color=ORANGE, lw=1.2, label="Plan value")
    a1.plot(paid.index, paid / 1000, color=GREY, lw=1.1, ls="--", label="Amount paid in")
    a1.set_ylabel("USD thousands")
    a1.set_title(f"USD {contribution} per month", fontsize=9)
    a1.legend(loc="upper left", fontsize=7.5)
    year_axis(a1, 4)
    gap = ld["ls"] - ld["dca"]
    a2.bar(gap.index, gap.values * 100, width=25, color=np.where(gap.values > 0, BLUE, ORANGE))
    a2.axhline(0, color="#c3c2b7", lw=0.8)
    a2.set_title("Lump sum minus 12-month averaging", fontsize=9)
    a2.set_ylabel("Gap after 12 months (% points)")
    year_axis(a2, 4)
    fig.tight_layout()
    return as_image(fig)


def c_chf():
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(tr.index, tr / tr.iloc[0] * 100, color=BLUE, lw=1.1, label="In USD")
    ax.plot(tr_chf.index, tr_chf / tr_chf.iloc[0] * 100, color=ORANGE, lw=1.1, ls="--", label="In CHF")
    ax.set_ylabel(f"Total return, {dt(idx[0])} = 100")
    year_axis(ax)
    ax.legend(loc="upper left")
    return as_image(fig)


# --------------------------------------------------------------------------
# PDF report
# --------------------------------------------------------------------------
fonts = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DV", os.path.join(fonts, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", os.path.join(fonts, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVI", os.path.join(fonts, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")

INK, SECOND, MUTED = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#898781")
s_title = ParagraphStyle("title", fontName="DVB", fontSize=20, leading=24, textColor=INK, spaceAfter=4)
s_sub = ParagraphStyle("sub", fontName="DV", fontSize=10.5, leading=14, textColor=SECOND, spaceAfter=14)
s_h1 = ParagraphStyle("h1", fontName="DVB", fontSize=13.5, leading=17, textColor=INK, spaceBefore=6, spaceAfter=6)
s_h2 = ParagraphStyle("h2", fontName="DVB", fontSize=10.5, leading=14, textColor=INK, spaceBefore=8, spaceAfter=3)
s_p = ParagraphStyle("p", fontName="DV", fontSize=9, leading=13, textColor=INK, alignment=TA_LEFT, spaceAfter=5)
s_note = ParagraphStyle("note", parent=s_p, fontSize=7.5, leading=10, textColor=MUTED)
s_cell = ParagraphStyle("cell", fontName="DV", fontSize=8, leading=10, textColor=INK)
s_cellb = ParagraphStyle("cellb", parent=s_cell, fontName="DVB")


def P(t, s=s_p):
    return Paragraph(t, s)


def table(rows, widths, header=True, right_from=1):
    data = [[Paragraph(str(c), s_cellb if (header and i == 0) else s_cell) for c in r]
            for i, r in enumerate(rows)]
    for row in data:
        for c in row[right_from:]:
            c.style = ParagraphStyle("r", parent=c.style, alignment=2)
    t = Table(data, colWidths=[w * cm for w in widths], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#c3c2b7")),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#e1e0d9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, "VT – Price analysis since inception")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def pval(p):
    return "< 0.001" if p < 0.001 else num(p, 3)


os.makedirs("report", exist_ok=True)
doc = SimpleDocTemplate("report/VT_analysis.pdf", pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=2 * cm,
                        title="VT – Price analysis since inception", author="Cédric Masset")
S = []

# --- Summary page
S += [P("Vanguard Total World Stock ETF (VT)", s_title),
      P(f"Price analysis since inception · {dt(idx[0])} – {dt(idx[-1])} · "
        f"{len(vt):,} trading days · Yahoo Finance data", s_sub)]

S.append(P("Summary", s_h1))
S.append(table([
    ["Metric", "Value"],
    ["Price at inception → latest price", f"USD {num(px.iloc[0])} → {num(px.iloc[-1])}"],
    ["Cumulative return (price / total return)",
     f"{pct(px.iloc[-1] / px.iloc[0] - 1, 0)} / {pct(tr.iloc[-1] / tr.iloc[0] - 1, 0)}"],
    ["Annualised return (price / total return)", f"{pct(cagr(px))} / {pct(full['cagr'])}"],
    ["Annualised total return in CHF", pct(cagr_chf)],
    ["Annualised volatility", pct(full["vol"])],
    [f"Sharpe ratio (average risk-free rate {pct(rf_mean)})", num(full["sharpe"])],
    ["Sortino ratio", num(full["sortino"])],
    ["Maximum drawdown (total return)", f"{pct(full['mdd'])} (trough on {dt(top_dd[0][1])})"],
    ["Calmar ratio", num(full["calmar"])],
    ["Probability of loss over 1 / 5 / 10 years", " / ".join(pct(r[2], 0) for r in roll_rows)],
], [10.5, 6.5]))
S.append(Spacer(1, 8))
S.append(c_price())
S.append(P("Log scale: equal slopes represent equal percentage returns. The widening gap between the two lines "
           "reflects reinvested dividends.", s_note))

S.append(P("Key takeaways", s_h2))
for t in [
    f"Over {num(years, 1)} years, VT returned {pct(full['cagr'])} a year with dividends reinvested, of which "
    f"roughly {pct(full['cagr'] - cagr(px))} a year came from dividends.",
    f"The price of that return was {pct(full['vol'], 0)} annual volatility and a maximum drawdown of "
    f"{pct(abs(full['mdd']), 0)} during the global financial crisis.",
    f"Daily returns are clearly non-normal (kurtosis of {num(kurt, 1)}) and volatility clusters over time; "
    "their direction is only weakly predictable from one day to the next, and not at all over longer horizons.",
    f"For a Swiss franc investor, the weaker dollar cut the annualised return by {pct(full['cagr'] - cagr_chf)} "
    f"a year ({pct(cagr_chf)} in CHF).",
]:
    S.append(P("•&nbsp;&nbsp;" + t))
S.append(PageBreak())

# --- 1. Return and risk
S.append(P("1. Return and risk", s_h1))
S.append(P("Calendar-year returns", s_h2))
S.append(c_annual())
best_y, worst_y = y_full.idxmax(), y_full.idxmin()
S.append(P(f"* Partial years: {partial_first} (from inception)"
           + (f" and {partial_last} (up to {dt(idx[-1])})." if partial_last else ".")
           + f" Of the {len(y_full)} full years, the best was {best_y} ({pct(y_full[best_y], 1, True)}) and the worst "
             f"{worst_y} ({pct(y_full[worst_y], 1, True)}); {int((y_full < 0).sum())} years were negative.", s_note))

S.append(P("One-year rolling volatility", s_h2))
S.append(c_vol())
S.append(P(f"Volatility varies sharply across regimes: from {pct(roll_vol.min(), 0)} at its calmest (year ending "
           f"{dt(roll_vol.idxmin())}) to {pct(roll_vol.max(), 0)} at its most turbulent (year ending "
           f"{dt(roll_vol.idxmax())}). A single full-period average therefore hides very different environments."))
S.append(PageBreak())

S.append(P("Drawdowns", s_h2))
S.append(c_dd())
rows = [["Peak", "Trough", "Recovery", "Drawdown", "Length of decline", "Time to recover"]]
for start, trough, end, depth in top_dd:
    rec = f"{np.busday_count(trough.date(), end.date())} days" if end is not None else "not yet"
    rows.append([dt(start), dt(trough), dt(end) if end is not None else "–", pct(depth),
                 f"{np.busday_count(start.date(), trough.date())} days", rec])
S.append(table(rows, [2.6, 2.6, 2.6, 2.2, 3.4, 3.6], right_from=3))
S.append(P("Based on total return. Lengths are in business days.", s_note))

S.append(P("Risk-adjusted ratios by sub-period", s_h2))
rows = [["Period", "Ann. return", "Volatility", "Sharpe", "Sortino", "Max. drawdown", "Calmar"]]
for d0, d1, m in sub + [(None, None, full)]:
    rows.append([f"{d0.year}–{d1.year}" if d0 is not None else "Full period", pct(m["cagr"]), pct(m["vol"]),
                 num(m["sharpe"]), num(m["sortino"]), pct(m["mdd"]), num(m["calmar"])])
S.append(table(rows, [3.4, 2.4, 2.1, 1.8, 1.8, 2.6, 1.8]))
S.append(P("Sharpe and Sortino ratios use returns in excess of the 13-week US T-bill (^IRX). Sortino uses the "
           "volatility of negative excess returns only. Calmar is the annualised return divided by the maximum "
           "drawdown of the sub-period.", s_note))
S.append(PageBreak())

# --- 2. Distribution
S.append(P("2. Distribution of returns", s_h1))
S.append(c_distribution())
S.append(table([
    ["Statistic (daily returns)", "Value", "Normal benchmark"],
    ["Mean / standard deviation", f"{pct(mu, 3)} / {pct(sd, 2)}", "–"],
    ["Skewness", num(skew), "0"],
    ["Kurtosis", num(kurt), "3"],
    ["Jarque-Bera test", f"{num(jb_stat, 0)} (p {pval(jb_p)})", "–"],
    ["Days beyond ±4 standard deviations", f"{beyond4}", f"{num(expected4, 2)} expected"],
    ["Worst / best day", f"{pct(r_tr.min())} ({dt(worst_day)}) / {pct(r_tr.max(), 1, True)} ({dt(best_day)})", "–"],
], [6.2, 6.6, 4.2]))
S.append(P(f"The tails are far fatter than those of a normal distribution: {beyond4} days moved by more than "
           f"4 standard deviations, whereas a normal distribution would predict {num(expected4, 1)} over the period. "
           f"The {'negative' if skew < 0 else 'positive'} skewness means that large declines are "
           f"{'more' if skew < 0 else 'less'} pronounced than large gains."))

S.append(P("Value at Risk and Expected Shortfall (1-day horizon)", s_h2))
rows = [["Level", "Historical VaR", "Historical ES", "Normal VaR", "Normal ES", "Breaches of normal VaR"]]
for lvl, q, es, qn, esn, br in var_rows:
    rows.append([pct(lvl, 0), pct(q, 2), pct(es, 2), pct(qn, 2), pct(esn, 2),
                 f"{pct(br, 2)} (expected {pct(1 - lvl, 0)})"])
S.append(table(rows, [1.6, 2.6, 2.5, 2.3, 2.3, 5.7]))
S.append(P("At 99%, the parametric normal VaR understates risk: it is breached more often than expected, and the "
           "historical Expected Shortfall is markedly higher. This is the direct consequence of the fat tails "
           "shown above." if var_rows[1][5] > 0.01 else
           "The normal VaR remains close to the historical VaR over this period."))

S.append(P("Volatility clustering", s_h2))
S.append(c_acf())
S.append(P(f"Apart from a small effect at the first lag (see section 3), returns show little autocorrelation "
           f"(Ljung-Box with 10 lags: Q = {num(lb_r[0], 1)}). Squared returns, by contrast, are strongly and "
           f"persistently autocorrelated (Q = {num(lb_r2[0], 0)}, p < 0.001). A large move tends to be followed by "
           "further large moves of either sign: this is the behaviour that ARCH and GARCH models capture."))
S.append(PageBreak())

# --- 3. Market efficiency
S.append(P("3. Market efficiency", s_h1))
S.append(P("First-order autocorrelation of log returns", s_h2))
rows = [["Frequency", "Observations", "Autocorrelation", "t-statistic", "p-value"]]
for name, n, r1, t, p in ac_freq:
    rows.append([name, f"{n:,}", num(r1, 3), num(t, 2), pval(p)])
S.append(table(rows, [3.6, 3, 3.4, 3.2, 2.8]))
S.append(P("t-statistic approximated by ρ·√n (standard error 1/√n under the random-walk hypothesis). It overstates "
           "significance when volatility clusters; the robust version of this test is the variance ratio VR(2) "
           "below.", s_note))
S.append(P(f"Daily autocorrelation is negative ({num(ac_freq[0][2], 3)}): a gain tends to be slightly reversed the "
           f"following day. The effect is concentrated in crises: excluding 2008–2009 and February to June 2020, it "
           f"falls to {num(ac1_non_crisis, 3)}. Possible explanations include overreaction on turbulent days, or "
           "temporary gaps between the ETF price and the value of its foreign holdings, which trade at different "
           "hours. At weekly and monthly frequencies, no significant autocorrelation appears."))

S.append(P("Variance ratio test (Lo and MacKinlay, 1988)", s_h2))
rows = [["Horizon q (days)", "Variance ratio VR(q)", "z* (robust)", "p-value", "Interpretation"]]
for q, vr, z, p in vr_rows:
    rows.append([str(q), num(vr, 3), num(z, 2), pval(p),
                 "not rejected" if p >= 0.05 else ("mean reversion" if vr < 1 else "persistence")])
S.append(table(rows, [3.2, 3.8, 2.8, 2.4, 4.8]))
S.append(P("Under the random-walk hypothesis, the variance of q-day returns equals q times the daily variance, so "
           "VR(q) = 1. A ratio below 1 signals mean reversion, a ratio above 1 persistence. The z* statistic is "
           "robust to heteroskedasticity, which is essential here given the volatility clustering.", s_note))
rejected = [q for q, vr, z, p in vr_rows if p < 0.05]
S.append(P(("The random-walk hypothesis is not rejected at any horizon. " if not rejected else
            f"The random-walk hypothesis is rejected at the 5% level for q = {', '.join(map(str, rejected))}. ")
           + "Even where a deviation is statistically significant, it is small in magnitude and would need to "
             "survive transaction costs to be exploitable."))

S.append(P("Calendar effects", s_h2))
S.append(c_calendar())
S.append(table([
    ["Test", "Result", "p-value"],
    ["Equal mean returns across weekdays (ANOVA)", f"F = {num(f_day, 2)}", pval(p_day)],
    ["January effect: January vs other months (Welch)",
     f"{pct(jan.mean(), 2)} vs {pct(other.mean(), 2)} per month", pval(p_jan)],
    [f"“Sell in May”: Nov–Apr vs May–Oct ({len(winter)} and {len(summer)} half-years)",
     f"{pct(winter.mean(), 1)} vs {pct(summer.mean(), 1)} per half-year", pval(p_sim)],
], [8.6, 5.6, 2.8]))
S.append(P("Error bars: 95% confidence interval of the mean. With fewer than twenty observations per calendar "
           "month, differences between months are dominated by statistical noise."
           + (" None of the three effects is significant at the 5% level." if min(p_day, p_jan, p_sim) >= 0.05 else ""),
           s_note))
S.append(PageBreak())

S.append(P("200-day moving-average strategy", s_h2))
S.append(c_ma200())
S.append(table([
    ["", "Buy and hold", "200-day moving average"],
    ["Annualised return", pct(m_bh["cagr"]), pct(m_ma["cagr"])],
    ["Volatility", pct(m_bh["vol"]), pct(m_ma["vol"])],
    ["Sharpe ratio", num(m_bh["sharpe"]), num(m_ma["sharpe"])],
    ["Maximum drawdown", pct(m_bh["mdd"]), pct(m_ma["mdd"])],
    ["Final value of USD 10,000", f"USD {num(val_bh.iloc[-1] * cap, 0)}", f"USD {num(val_ma.iloc[-1] * cap, 0)}"],
], [6, 5.5, 5.5]))
S.append(P(f"Rule: invested in VT if the previous day's close is above its 200-day moving average, otherwise held in "
           f"T-bills. Cost of {pct(TRANSACTION_COST, 2)} per switch, {n_switch} switches in total; the strategy was "
           f"invested {pct(share_invested, 0)} of the time. Common period starting {dt(sig.index[0])}, since the "
           "moving average needs 200 days of history.", s_note))
S.append(P(f"The strategy {'reduces' if m_ma['mdd'] > m_bh['mdd'] else 'does not reduce'} the maximum drawdown "
           f"({pct(m_ma['mdd'], 0)} vs {pct(m_bh['mdd'], 0)}), "
           + (f"but at the cost of {pct(m_bh['cagr'] - m_ma['cagr'])} a year in return. "
              if m_ma["cagr"] < m_bh["cagr"] else
              f"and adds {pct(m_ma['cagr'] - m_bh['cagr'])} a year in return. ")
           + "It often exits after a fall and re-enters after the rebound: late and repeated signals "
             "(“whipsaws”) are costly in a market that recovers quickly."))
S.append(PageBreak())

# --- 4. The investor's perspective
S.append(P("4. Holding period and investor behaviour", s_h1))
S.append(P("Rolling returns by holding period", s_h2))
S.append(c_rolling())
rows = [["Horizon", "Periods", "Probability of loss", "Worst (annualised)", "Median", "Best"]]
for name, n, pl, mn, dmn, med, mx in roll_rows:
    rows.append([name, str(n), pct(pl, 0), f"{pct(mn)} ({mon(dmn)})", pct(med), pct(mx)])
S.append(table(rows, [2.2, 2, 3.3, 4.2, 2.5, 2.8]))
S.append(P("Overlapping monthly windows, annualised total return. The longer the horizon, the narrower the "
           "dispersion of outcomes. Long windows overlap heavily and are therefore not independent: over 10 years, "
           "the history contains only about one and a half distinct periods.", s_note))

S.append(P("The cost of missing the best days", s_h2))
S.append(c_best())
S.append(P(f"Missing the 10 best days out of {len(r_tr):,} cuts the annualised return from {pct(best_rows[0][2])} "
           f"to {pct(best_rows[1][2])}. The reverse exercise is just as striking: avoiding the 10 worst days would "
           f"have lifted the final value to USD {num(worst_rows[0][1], 0)}. {best_near_worst} of the 10 best days "
           "fall within two weeks of one of the 10 worst. They occur in the same crisis periods: avoiding the worst "
           "days without missing the best would require perfect timing."))
S.append(PageBreak())

S.append(P("Monthly savings plan and lump-sum investing", s_h2))
S.append(c_plan())
S.append(P(f"Investing USD {contribution} at the end of every month since inception ({len(plan_prices)} "
           f"contributions, USD {num(plan_invested, 0)} in total) would be worth USD {num(plan_value, 0)} today, an "
           f"internal rate of return of {pct(plan_irr)} a year."))
S.append(P(f"Across the {len(ld)} possible monthly starting points, investing a sum all at once beat spreading it "
           f"over 12 months (with the uninvested cash earning the T-bill rate) in {pct(ls_win_share, 0)} of cases, "
           f"with an average advantage of {pct(ls_mean_gap, 1)} after 12 months. Because the market rises most of "
           "the time, waiting costs money on average; averaging in mainly limits the regret of investing just "
           "before a decline."))

# --- 5. CHF
S.append(P("5. A Swiss investor's perspective", s_h1))
S.append(c_chf())
S.append(table([
    ["", "In USD", "In CHF"],
    ["Annualised total return", pct(cagr_usd), pct(cagr_chf)],
    ["Annualised volatility", pct(full["vol"]), pct(vol_chf)],
    ["USD/CHF over the period", "–",
     f"{num(chf.iloc[0], 4)} → {num(chf.iloc[-1], 4)} ({pct(cagr_fx, 1, True)} a year)"],
], [7, 4, 6]))
S.append(P(f"The dollar depreciated against the franc, costing an unhedged Swiss investor about "
           f"{pct(cagr_usd - cagr_chf)} of return a year. The daily correlation between VT and USD/CHF is "
           f"{num(corr_fx, 2)}: "
           + ("the franc tends to strengthen when equities fall, which deepens losses in CHF during crises. "
              if corr_fx > 0 else
              "the dollar tends to strengthen slightly when equities fall, which cushions losses in CHF a little, "
              "but the relationship is weak. ")
           + "Because VT is a global fund, its true exposure is to a basket of currencies: USD/CHF captures only "
             "part of the currency effect."))

S.append(P("Data and methodology", s_h2))
S.append(P(f"Source: Yahoo Finance, downloaded with the yfinance library and updated automatically (GitHub "
           f"repository cedricmasset98/donnees-marche). VT: daily closing prices and prices adjusted for dividends "
           f"and splits (“Adj Close”), from {dt(idx[0])} to {dt(idx[-1])}. Risk-free rate: 13-week US T-bill (^IRX), "
           f"converted to a daily rate (÷ 252). Currency: USD/CHF (CHF=X), last available rate for each VT trading "
           f"day. Annualisation over 252 trading days. Report generated on {date.today():%d %b %Y}. These analyses "
           f"are descriptive and historical; they do not constitute investment advice.", s_note))

doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("report/VT_analysis.pdf created")
