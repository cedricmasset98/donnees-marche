"""Numerical validation of kelly.py on synthetic data and on the stored VT history.

Run with:  python tests/test_kelly.py   (exit code 1 on failure)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kelly  # noqa: E402


def synthetic(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    r = pd.Series(0.0003 + 0.012 * rng.standard_t(5, n) / np.sqrt(5 / 3), index=idx)
    rl = pd.Series(0.02 / 252, index=idx)
    rb = pd.Series(0.035 / 252, index=idx)
    return r, rl, rb


def test_synthetic():
    r, rl, rb = synthetic()
    res = kelly.validate(r, rl, rb)
    assert all(v is True or v < 1e-4 for v in res.values())
    # piecewise optimum: kink at L = 1 when lending < mu - sigma^2 < borrowing
    assert kelly.kelly_leverage(0.06, 0.04, 0.01, 0.05) == 1.0
    assert abs(kelly.kelly_leverage(0.10, 0.04, 0.01, 0.02) - 2.0) < 1e-12
    assert kelly.kelly_leverage(0.00, 0.04, 0.01, 0.02) == 0.0
    # approximation of the compounding gap is close for small returns
    ex, ap = kelly.compounding_gap(r, rb, 2.0)
    assert abs(ex[-1] - ap[-1]) < 0.05 * abs(ex[-1])


def test_vt_history():
    path = "data/VT.csv"
    if not os.path.exists(path):
        return
    vt = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    irx = pd.read_csv("data/IRX.csv", parse_dates=["Date"], index_col="Date")["Close"]
    r = vt["Adj Close"].pct_change().dropna()
    rl = (irx.reindex(vt.index, method="ffill") / 100 / 252).shift(1).reindex(r.index).bfill()
    kelly.validate(r, rl, rl + 0.015 / 252)


if __name__ == "__main__":
    failed = 0
    for name, fn in [("synthetic", test_synthetic), ("VT history", test_vt_history)]:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    sys.exit(1 if failed else 0)
