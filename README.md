# donnees-marche
## Contenu

- `data/` : historiques quotidiens Yahoo Finance (VT, USD/CHF, T-bill 13 semaines), mis à jour chaque soir de semaine par GitHub Actions, de façon incrémentale (`telecharger.py`).
- `docs/` : tableau de bord GitHub Pages (https://cedricmasset98.github.io/donnees-marche/), généré par `build_site.py` à partir de `site/index.template.html`, des CSV et de `report/analysis_results.json` (calculs lourds faits par `analysis.py`). Graphiques : Apache ECharts, hébergé dans `site/vendor/` (licence Apache 2.0).
- `report/VT_analysis.pdf` : analyse complète de VT, en anglais (rendement et risque, distribution, efficience de marché, horizon de l'investisseur, perspective CHF), régénérée à partir des CSV par `analysis.py` et mise en page en LaTeX (`report/template.tex`, source `report/VT_analysis.tex` utilisable dans Overleaf), sans requête supplémentaire vers Yahoo.

- `kelly.py` : critère de Kelly, simulation de portefeuilles à levier (financement par morceaux, coûts, insolvabilité, marge) et décomposition du volatility drag, utilisés par la section 4.5 du rapport. Tests : `python tests/test_kelly.py`.
