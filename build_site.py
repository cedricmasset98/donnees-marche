"""
Builds the GitHub Pages dashboard in docs/ from files already in the repository.

- docs/dashboard.json : daily series (prices, total return, USD/CHF, risk-free rate)
                        + heavy results computed by analysis.py (GARCH, VaR backtest,
                        rolling efficiency tests, bootstrap, validation)
                        + data-quality checks
- docs/index.html     : rendered from site/index.template.html, with the headline
                        figures written into the HTML (link previews, no-JavaScript view)
- docs/data/*.csv, docs/VT_analysis.pdf : downloads

No request to Yahoo: everything comes from the CSV files and report/analysis_results.json.
"""
import html
import json
import os
import shutil

import numpy as np
import pandas as pd

os.makedirs("docs/data", exist_ok=True)
os.makedirs("docs/vendor", exist_ok=True)
for f in os.listdir("data"):
    if f.endswith(".csv"):
        shutil.copy(os.path.join("data", f), os.path.join("docs/data", f))
if os.path.exists("report/VT_analysis.pdf"):
    shutil.copy("report/VT_analysis.pdf", "docs/VT_analysis.pdf")
for f in os.listdir("site/vendor"):
    shutil.copy(os.path.join("site/vendor", f), os.path.join("docs/vendor", f))


def load(name):
    return pd.read_csv(f"data/{name}.csv", parse_dates=["Date"], index_col="Date")


raw = load("VT")
vt = raw.dropna(subset=["Close"])
idx = vt.index
fx = load("CHFX")["Close"].reindex(idx, method="ffill").bfill()
rf = (load("IRX")["Close"].reindex(idx, method="ffill").bfill() / 100)
res = json.load(open("report/analysis_results.json"))

# ---------------------------------------------------------------- data quality checks
r = vt["Adj Close"].pct_change().dropna()
gaps = pd.Series(idx[1:] - idx[:-1], index=idx[1:]).dt.days
quality = {
    "rows": int(len(raw)),
    "missing_close": int(raw["Close"].isna().sum()),
    "missing_adj": int(raw["Adj Close"].isna().sum()),
    "duplicate_dates": int(raw.index.duplicated().sum()),
    "max_gap_days": int(gaps.max()),
    "max_gap_end": gaps.idxmax().strftime("%Y-%m-%d"),
    "moves_above_10pct": int((r.abs() > 0.10).sum()),
    "adj_above_close": int((vt["Adj Close"] > vt["Close"] * 1.0001).sum()),
    "fx_last": load("CHFX").index[-1].strftime("%Y-%m-%d"),
    "rf_last": load("IRX").index[-1].strftime("%Y-%m-%d"),
}

# ---------------------------------------------------------------- dashboard.json
dash = {
    "dates": [d.strftime("%Y-%m-%d") for d in idx],
    "close": [round(float(x), 4) for x in vt["Close"]],
    "adj": [round(float(x), 4) for x in vt["Adj Close"]],
    "fx": [round(float(x), 5) for x in fx],
    "rf": [round(float(x), 5) for x in rf],
    "analysis": res,
    "quality": quality,
}
with open("docs/dashboard.json", "w") as f:
    json.dump(dash, f, separators=(",", ":"))

# ---------------------------------------------------------------- headline figures in the HTML
m = res["report_metrics"]
last = idx[-1]
price = float(vt["Close"].iloc[-1])
summary = {
    "LAST_DATE": last.strftime("%d %b %Y"),
    "PRICE": f"${price:.2f}",
    "CAGR": f"{m['cagr'] * 100:.1f}%",
    "VOL": f"{m['vol'] * 100:.1f}%",
    "SHARPE": f"{m['sharpe']:.2f}",
    "MDD": f"{m['mdd'] * 100:.1f}%",
    "INCEPTION": idx[0].strftime("%d %b %Y"),
}
summary["DESCRIPTION"] = html.escape(
    f"VT closed at {summary['PRICE']} on {summary['LAST_DATE']}. Since inception: {summary['CAGR']} a year with "
    f"dividends reinvested, {summary['VOL']} annualised volatility, maximum drawdown {summary['MDD']}. Interactive "
    "dashboard, full analysis report and data, updated every weekday.")
page = open("site/index.template.html", encoding="utf-8").read()
for k, val in summary.items():
    page = page.replace("{{" + k + "}}", val)
with open("docs/index.html", "w", encoding="utf-8") as f:
    f.write(page)

open("docs/.nojekyll", "w").close()
for old in ("docs/summary.json",):
    if os.path.exists(old):
        os.remove(old)
print("docs/ updated:", last.date(), f"({os.path.getsize('docs/dashboard.json') / 1024:.0f} KB)")
