# donnees-marche
## Contenu

- `data/` : historiques quotidiens Yahoo Finance (VT, USD/CHF, T-bill 13 semaines), mis à jour chaque soir de semaine par GitHub Actions, de façon incrémentale (`telecharger.py`).
- `rapport/VT_analyse.pdf` : analyse complète de VT (rendement et risque, distribution, efficience de marché, horizon de l'investisseur, perspective CHF), régénérée à partir des CSV par `analyse.py`, sans requête supplémentaire vers Yahoo.
