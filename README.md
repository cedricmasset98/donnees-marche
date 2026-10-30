# donnees-marche
## Contenu

- `data/` : historiques quotidiens Yahoo Finance (VT, USD/CHF, T-bill 13 semaines), mis à jour chaque soir de semaine par GitHub Actions, de façon incrémentale (`telecharger.py`).
- `docs/` : site GitHub Pages (https://cedricmasset98.github.io/donnees-marche/), régénéré par `build_site.py`.
- `report/VT_analysis.pdf` : analyse complète de VT, en anglais (rendement et risque, distribution, efficience de marché, horizon de l'investisseur, perspective CHF), régénérée à partir des CSV par `analysis.py` et mise en page en LaTeX (`report/template.tex`, source `report/VT_analysis.tex` utilisable dans Overleaf), sans requête supplémentaire vers Yahoo.
