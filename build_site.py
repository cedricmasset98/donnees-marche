"""
Builds the GitHub Pages site in docs/ from the CSV files and the PDF report.

- copies data/*.csv and report/VT_analysis.pdf into docs/
- writes docs/summary.json with the headline figures shown on the page

No request to Yahoo: everything comes from files already in the repository.
"""
import json
import os
import shutil

import numpy as np
import pandas as pd

os.makedirs("docs/data", exist_ok=True)
for f in os.listdir("data"):
    if f.endswith(".csv"):
        shutil.copy(os.path.join("data", f), os.path.join("docs/data", f))
if os.path.exists("report/VT_analysis.pdf"):
    shutil.copy("report/VT_analysis.pdf", "docs/VT_analysis.pdf")

vt = pd.read_csv("data/VT.csv", parse_dates=["Date"], index_col="Date")
px, tr = vt["Close"], vt["Adj Close"]
last = vt.index[-1]
r = tr.pct_change().dropna()
years = (last - vt.index[0]).days / 365.25


def ret_since(series, start):
    s = series.loc[:start]
    return float(series.iloc[-1] / s.iloc[-1] - 1) if len(s) else None


summary = {
    "updated": last.strftime("%Y-%m-%d"),
    "inception": vt.index[0].strftime("%Y-%m-%d"),
    "price": round(float(px.iloc[-1]), 2),
    "day_change": float(px.iloc[-1] / px.iloc[-2] - 1),
    "ytd": ret_since(tr, pd.Timestamp(last.year - 1, 12, 31)),
    "one_year": ret_since(tr, last - pd.DateOffset(years=1)),
    "cagr_tr": float((tr.iloc[-1] / tr.iloc[0]) ** (1 / years) - 1),
    "vol": float(r.std() * np.sqrt(252)),
    "max_dd": float((tr / tr.cummax() - 1).min()),
    "current_dd": float(tr.iloc[-1] / tr.max() - 1),
    "all_time_high": round(float(px.max()), 2),
    "all_time_high_date": px.idxmax().strftime("%Y-%m-%d"),
    "trading_days": int(len(vt)),
}
with open("docs/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Disable Jekyll processing so files are served as-is
open("docs/.nojekyll", "w").close()
print("docs/ updated:", summary["updated"])
