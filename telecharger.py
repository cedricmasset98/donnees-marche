"""Télécharge l'historique quotidien complet (Yahoo Finance) des tickers listés."""
import os
import yfinance as yf

TICKERS = ["VT"]  # ajouter d'autres tickers ici, ex. "ACWI", "^SSMI"

os.makedirs("data", exist_ok=True)
for t in TICKERS:
    df = yf.download(t, period="max", auto_adjust=False, progress=False)
    df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"])
    nom = t.replace("^", "")
    df.to_csv(f"data/{nom}.csv")
    print(f"{t}: {len(df)} lignes, {df.index[0].date()} -> {df.index[-1].date()}")
