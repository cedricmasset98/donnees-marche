# donnees-marche
## Contenu

- `data/` : historiques quotidiens Yahoo Finance (VT, USD/CHF, T-bill 13 semaines), mis à jour chaque soir de semaine par GitHub Actions, de façon incrémentale (`telecharger.py`).
- `report/VT_analysis.pdf` : analyse complète de VT, en anglais (rendement et risque, distribution, efficience de marché, horizon de l'investisseur, perspective CHF), régénérée à partir des CSV par `analysis.py`, sans requête supplémentaire vers Yahoo.
