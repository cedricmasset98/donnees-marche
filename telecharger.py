"""
Met à jour l'historique quotidien (Yahoo Finance) des tickers listés, en limitant
les requêtes au strict nécessaire :

- un ticker sans fichier CSV : téléchargement complet, une seule fois ;
- un ticker déjà présent : seules les dernières séances sont demandées,
  puis ajoutées au fichier existant ;
- si le fichier est déjà à jour (aucune séance clôturée manquante),
  aucune requête n'est envoyée ;
- une pause sépare les tickers pour ne pas enchaîner les appels.

Le cours ajusté (« Adj Close ») de tout l'historique change après chaque
dividende. Plutôt que de tout retélécharger, l'ancien historique est remis
à l'échelle à partir d'une date commune aux anciennes et nouvelles données.
"""
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

TICKERS = ["VT"]  # ajouter d'autres tickers ici, ex. "ACWI", "^SSMI"
CHEVAUCHEMENT_JOURS = 10  # séances récentes redemandées (corrections, cours ajusté)
PAUSE_SECONDES = 5


def derniere_seance_cloturee():
    """Date de la dernière séance américaine terminée (clôture au plus tard 21h UTC)."""
    now = datetime.now(timezone.utc)
    d = now.date() if (now.hour, now.minute) >= (21, 30) else now.date() - timedelta(days=1)
    while d.weekday() >= 5:  # samedi/dimanche
        d -= timedelta(days=1)
    return pd.Timestamp(d)


def telecharger(ticker, **kwargs):
    df = yf.download(ticker, auto_adjust=False, progress=False, **kwargs)
    if df.empty:
        return df
    df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df.dropna(subset=["Close"])


def mettre_a_jour(ticker):
    chemin = f"data/{ticker.replace('^', '')}.csv"
    cible = derniere_seance_cloturee()

    if not os.path.exists(chemin):
        df = telecharger(ticker, period="max")
        df = df[df.index <= cible]  # n'enregistre que des séances clôturées
        df.to_csv(chemin)
        print(f"{ticker}: téléchargement complet, {len(df)} séances")
        return True

    ancien = pd.read_csv(chemin, parse_dates=["Date"], index_col="Date")
    if ancien.index[-1] >= cible:
        print(f"{ticker}: déjà à jour ({ancien.index[-1].date()}), aucune requête")
        return False

    debut = ancien.index[-1] - timedelta(days=CHEVAUCHEMENT_JOURS)
    nouveau = telecharger(ticker, start=debut.strftime("%Y-%m-%d"))
    nouveau = nouveau[nouveau.index <= cible]
    if nouveau.empty:
        print(f"{ticker}: aucune nouvelle donnée")
        return True

    # Remise à l'échelle du cours ajusté historique sur une date commune
    communes = ancien.index.intersection(nouveau.index)
    if len(communes) > 0:
        ref = communes[0]
        facteur = nouveau.loc[ref, "Adj Close"] / ancien.loc[ref, "Adj Close"]
        if abs(facteur - 1) > 1e-6:
            ancien["Adj Close"] *= facteur
            print(f"{ticker}: cours ajusté historique recalé (facteur {facteur:.6f})")

    df = pd.concat([ancien[ancien.index < nouveau.index[0]], nouveau])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "Date"
    df.to_csv(chemin)
    print(f"{ticker}: +{len(df) - len(ancien)} séance(s), jusqu'au {df.index[-1].date()}")
    return True


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    requetes = 0
    for t in TICKERS:
        if requetes:
            time.sleep(PAUSE_SECONDES)
        requetes += mettre_a_jour(t)
