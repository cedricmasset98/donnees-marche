"""
Analyse complète de l'ETF Vanguard Total World Stock (VT) et rapport PDF.

Lit uniquement les fichiers CSV du dossier data/ (aucune requête vers Yahoo) :
  data/VT.csv    cours quotidien de VT (Close, Adj Close)
  data/IRX.csv   taux T-bill 13 semaines, en % annualisé (taux sans risque)
  data/CHFX.csv  USD/CHF (CHF pour 1 USD)

Produit : rapport/VT_analyse.pdf
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
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy import stats, optimize

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

JOURS_AN = 252
COUT_TRANSACTION = 0.001  # 0,10 % par changement de position (stratégie MM200)
BLEU, ORANGE, VERT, GRIS, ROUGE = "#2a78d6", "#eb6834", "#1baf7a", "#898781", "#e34948"

# --------------------------------------------------------------------------
# Mise en forme des nombres (français)
# --------------------------------------------------------------------------

def nb(x, d=2):
    s = f"{x:,.{d}f}".replace(",", "'").replace(".", ",")
    return s


def pc(x, d=1, signe=False):
    s = f"{x * 100:+.{d}f}" if signe else f"{x * 100:.{d}f}"
    return s.replace(".", ",") + " %"


def dt(d):
    return pd.Timestamp(d).strftime("%d.%m.%Y")


# --------------------------------------------------------------------------
# Données
# --------------------------------------------------------------------------

def charger(nom):
    return pd.read_csv(f"data/{nom}.csv", parse_dates=["Date"], index_col="Date")


vt = charger("VT")
px = vt["Close"]                         # cours (prix)
tr = vt["Adj Close"]                     # rendement total (dividendes réinvestis)
idx = vt.index

irx = charger("IRX")["Close"].reindex(idx, method="ffill") / 100
rf_j = irx / JOURS_AN                    # taux sans risque quotidien
chf = charger("CHFX")["Close"].reindex(idx, method="ffill")

r_px = px.pct_change().dropna()
r_tr = tr.pct_change().dropna()
lr_tr = np.log(tr).diff().dropna()
rf = rf_j.reindex(r_tr.index)
annees = (idx[-1] - idx[0]).days / 365.25

tr_m = tr.resample("ME").last()
px_m = px.resample("ME").last()
r_m = tr_m.pct_change().dropna()
rf_m = (irx.resample("ME").mean() / 12).reindex(r_m.index)

# --------------------------------------------------------------------------
# Indicateurs
# --------------------------------------------------------------------------

def cagr(serie):
    a = (serie.index[-1] - serie.index[0]).days / 365.25
    return (serie.iloc[-1] / serie.iloc[0]) ** (1 / a) - 1


def drawdown(serie):
    return serie / serie.cummax() - 1


def indicateurs(r, rfq):
    """r : rendements quotidiens ; rfq : taux sans risque quotidien aligné."""
    ex = r - rfq
    vol = r.std() * np.sqrt(JOURS_AN)
    neg = ex[ex < 0]
    dd_vol = np.sqrt((neg ** 2).sum() / len(ex)) * np.sqrt(JOURS_AN)
    valeur = (1 + r).cumprod()
    a = len(r) / JOURS_AN
    ca = valeur.iloc[-1] ** (1 / a) - 1
    mdd = drawdown(pd.concat([pd.Series([1.0]), valeur.reset_index(drop=True)])).min()
    return {
        "cagr": ca,
        "vol": vol,
        "sharpe": ex.mean() * JOURS_AN / vol,
        "sortino": ex.mean() * JOURS_AN / dd_vol,
        "mdd": mdd,
        "calmar": ca / abs(mdd),
    }


glob = indicateurs(r_tr, rf)
rf_moy = irx.mean()

# Rendements annuels
an_tr = tr.resample("YE").last().pct_change()
an_tr.iloc[0] = tr.resample("YE").last().iloc[0] / tr.iloc[0] - 1
an_px = px.resample("YE").last().pct_change()
an_px.iloc[0] = px.resample("YE").last().iloc[0] / px.iloc[0] - 1
an_tr.index = an_tr.index.year
an_px.index = an_px.index.year
annee_partielle_debut = an_tr.index[0]
annee_partielle_fin = an_tr.index[-1] if idx[-1].month < 12 else None

# Volatilité glissante
vol_gl = r_tr.rolling(JOURS_AN).std() * np.sqrt(JOURS_AN)

# Drawdowns (rendement total)
dd = drawdown(tr)


def episodes_drawdown(serie, n=5):
    d = drawdown(serie)
    res = []
    en_baisse, debut = False, None
    for t, v in d.items():
        if v < 0 and not en_baisse:
            en_baisse, debut = True, serie.loc[:t].index[-2]
        elif v == 0 and en_baisse:
            seg = d.loc[debut:t]
            res.append((debut, seg.idxmin(), t, seg.min()))
            en_baisse = False
    if en_baisse:
        seg = d.loc[debut:]
        res.append((debut, seg.idxmin(), None, seg.min()))
    res.sort(key=lambda x: x[3])
    return res[:n]


top_dd = episodes_drawdown(tr)

# Sous-périodes
coupes = [("2008-06-01", "2012-12-31"), ("2013-01-01", "2017-12-31"),
          ("2018-01-01", "2022-12-31"), ("2023-01-01", str(idx[-1].date()))]
sous = []
for a, b in coupes:
    r_s = r_tr.loc[a:b]
    sous.append((r_s.index[0], r_s.index[-1], indicateurs(r_s, rf.loc[r_s.index])))

# Distribution
mu, sd = r_tr.mean(), r_tr.std()
asym = stats.skew(r_tr)
kurt = stats.kurtosis(r_tr, fisher=False)
jb_stat, jb_p = stats.jarque_bera(r_tr)
pire_j, meil_j = r_tr.idxmin(), r_tr.idxmax()
sig4 = (np.abs(r_tr - mu) > 4 * sd).sum()
attendu4 = len(r_tr) * 2 * stats.norm.sf(4)

# VaR / ES
var_rows = []
for niv in (0.95, 0.99):
    q = r_tr.quantile(1 - niv)
    es = r_tr[r_tr <= q].mean()
    q_n = mu + sd * stats.norm.ppf(1 - niv)
    es_n = mu - sd * stats.norm.pdf(stats.norm.ppf(1 - niv)) / (1 - niv)
    depass_n = (r_tr < q_n).mean()
    var_rows.append((niv, q, es, q_n, es_n, depass_n))

# Autocorrélations
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
for nom_f, s in [("Quotidien", lr_tr),
                 ("Hebdomadaire", np.log(tr.resample("W-FRI").last()).diff().dropna()),
                 ("Mensuel", np.log(tr_m).diff().dropna())]:
    r1 = acf(s, 1)[0]
    ac_freq.append((nom_f, len(s), r1, r1 * np.sqrt(len(s)), 2 * stats.norm.sf(abs(r1 * np.sqrt(len(s))))))


# Test du ratio de variance de Lo et MacKinlay (robuste à l'hétéroscédasticité)
def variance_ratio(lp, q):
    r = np.diff(lp)
    n = len(r)
    m = np.mean(r)
    e = r - m
    var1 = np.sum(e ** 2) / (n - 1)
    rq = lp[q:] - lp[:-q] - q * m
    mq = q * (n - q + 1) * (1 - q / n)
    varq = np.sum(rq ** 2) / mq
    vr = varq / var1
    den = np.sum(e ** 2) ** 2
    theta = 0.0
    for j in range(1, q):
        delta = n * np.sum(e[j:] ** 2 * e[:-j] ** 2) / den
        theta += (2 * (q - j) / q) ** 2 * delta
    z = np.sqrt(n) * (vr - 1) / np.sqrt(theta)
    return vr, z, 2 * stats.norm.sf(abs(z))


hors_crise = ~((lr_tr.index < "2010-01-01") | ((lr_tr.index >= "2020-02-01") & (lr_tr.index < "2020-07-01")))
ac1_hors_crise = acf(lr_tr[hors_crise], 1)[0]

lp = np.log(tr.values)
vr_rows = [(q, *variance_ratio(lp, q)) for q in (2, 5, 10, 20)]

# Effets calendaires
jours = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi"]
par_jour = [r_tr[r_tr.index.dayofweek == k] for k in range(5)]
f_jour, p_jour = stats.f_oneway(*par_jour)

mois_noms = ["Janv.", "Févr.", "Mars", "Avr.", "Mai", "Juin", "Juil.", "Août",
             "Sept.", "Oct.", "Nov.", "Déc."]
par_mois = r_m.groupby(r_m.index.month).mean()
janv = r_m[r_m.index.month == 1]
autres = r_m[r_m.index.month != 1]
t_janv, p_janv = stats.ttest_ind(janv, autres, equal_var=False)

# « Sell in May » : mai–octobre contre novembre–avril (années complètes)
ete, hiver = [], []
for a in range(idx[0].year + 1, idx[-1].year + 1):
    s = tr_m.loc[f"{a}-04-30":f"{a}-10-31"]
    if len(s) == 7:
        ete.append(s.iloc[-1] / s.iloc[0] - 1)
    w = tr_m.loc[f"{a - 1}-10-31":f"{a}-04-30"]
    if len(w) == 7:
        hiver.append(w.iloc[-1] / w.iloc[0] - 1)
ete, hiver = np.array(ete), np.array(hiver)
t_sim, p_sim = stats.ttest_ind(hiver, ete, equal_var=False)

# Stratégie moyenne mobile 200 jours
mm200 = px.rolling(200).mean()
signal = (px > mm200).astype(float).shift(1)       # décision prise la veille
signal[mm200.shift(1).isna()] = np.nan
sig = signal.reindex(r_tr.index).dropna()
r_bh = r_tr.loc[sig.index]
r_cash = rf.loc[sig.index]
changements = sig.diff().abs().fillna(0)
r_mm = sig * r_bh + (1 - sig) * r_cash - changements * COUT_TRANSACTION
ind_bh = indicateurs(r_bh, r_cash)
ind_mm = indicateurs(r_mm, r_cash)
n_switch = int(changements.sum())
part_investi = sig.mean()
val_bh = (1 + r_bh).cumprod()
val_mm = (1 + r_mm).cumprod()

# Rendements glissants
gl_rows, gl_series = [], {}
for h, nom_h in [(12, "1 an"), (60, "5 ans"), (120, "10 ans")]:
    g = (tr_m / tr_m.shift(h)) ** (12 / h) - 1
    g = g.dropna()
    gl_series[nom_h] = g
    gl_rows.append((nom_h, len(g), (g < 0).mean(), g.min(), g.idxmin(), g.median(), g.max()))

# Rater les meilleurs jours
cap = 10_000
best_rows = []
tri = r_tr.sort_values(ascending=False)
for n in (0, 10, 20, 30):
    r_s = r_tr.drop(tri.index[:n]) if n else r_tr
    v = cap * (1 + r_s).prod()
    best_rows.append((n, v, (v / cap) ** (1 / annees) - 1))
pire_rows = []
tri_p = r_tr.sort_values()
for n in (10, 20, 30):
    v = cap * (1 + r_tr.drop(tri_p.index[:n])).prod()
    pire_rows.append((n, v, (v / cap) ** (1 / annees) - 1))
meilleurs_proches = sum(
    1 for d in tri.index[:10]
    if any(abs((d - p).days) <= 14 for p in tri_p.index[:10]))

# Plan d'épargne mensuel et investissement unique contre étalement
versement = 500
prix_plan = tr_m.iloc[:-1]  # versements en fin de mois, valorisation au dernier cours
parts = (versement / prix_plan).sum()
valeur_plan = parts * tr.iloc[-1]
investi_plan = versement * len(prix_plan)
flux = [(-versement, d) for d in prix_plan.index] + [(valeur_plan, idx[-1])]


def tri_annuel(flux):
    t0 = flux[0][1]

    def vna(r):
        return sum(c / (1 + r) ** ((d - t0).days / 365.25) for c, d in flux)

    return optimize.brentq(vna, -0.9, 1.0)


tri_plan = tri_annuel(flux)

cash_m = (1 + rf_m.reindex(tr_m.index).fillna(0)).cumprod()
ls_vs_dca = []
P = tr_m.values
C = cash_m.values
for i in range(len(P) - 12):
    ls = P[i + 12] / P[i]
    dca = sum((1 / 12) * (C[i + k] / C[i]) * P[i + 12] / P[i + k] for k in range(12))
    ls_vs_dca.append((tr_m.index[i], ls, dca))
ld = pd.DataFrame(ls_vs_dca, columns=["date", "ls", "dca"]).set_index("date")
part_ls_gagne = (ld["ls"] > ld["dca"]).mean()
ecart_moyen = (ld["ls"] - ld["dca"]).mean()

# Perspective CHF
tr_chf = tr * chf
cagr_usd = cagr(tr)
cagr_chf = cagr(tr_chf)
cagr_fx = cagr(chf)
vol_chf = tr_chf.pct_change().std() * np.sqrt(JOURS_AN)
corr_fx = r_tr.corr(chf.pct_change().reindex(r_tr.index))

# --------------------------------------------------------------------------
# Graphiques
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "legend.frameon": False,
})


def fig_img(fig, largeur=17 * cm):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    w, h = fig.get_size_inches()
    return Image(buf, width=largeur, height=largeur * h / w)


def axe_annees(ax, pas=2):
    ax.xaxis.set_major_locator(mdates.YearLocator(pas))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="x", visible=False)


fmt_pc = FuncFormatter(lambda v, _: f"{v * 100:.0f} %")


def g_cours():
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(px.index, px, color=BLEU, lw=1.1, label="Cours de clôture")
    ax.plot(tr.index, tr, color=ORANGE, lw=1.1, ls="--", label="Rendement total (cours ajusté)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_yticks([20, 30, 40, 60, 80, 100, 150])
    ax.set_ylabel("USD (échelle logarithmique)")
    axe_annees(ax)
    ax.legend(loc="upper left")
    return fig_img(fig)


def g_annuels():
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(an_tr))
    ax.bar(x - 0.2, an_px.values, 0.4, color=BLEU, label="Prix")
    ax.bar(x + 0.2, an_tr.values, 0.4, color=ORANGE, label="Rendement total", hatch="///",
           edgecolor="white", linewidth=0)
    lab = [f"{a}*" if a in (annee_partielle_debut, annee_partielle_fin) else str(a) for a in an_tr.index]
    ax.set_xticks(x, lab, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(fmt_pc)
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2)
    return fig_img(fig)


def g_vol():
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(vol_gl.index, vol_gl, color=BLEU, lw=1.1)
    ax.axhline(glob["vol"], color=GRIS, lw=0.8, ls="--")
    ax.text(vol_gl.index[-1], glob["vol"] + 0.012, f"moyenne : {pc(glob['vol'])}",
            color=GRIS, fontsize=8, ha="right")
    ax.yaxis.set_major_formatter(fmt_pc)
    ax.set_ylabel("Volatilité annualisée (1 an glissant)")
    axe_annees(ax)
    return fig_img(fig)


def g_dd():
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.fill_between(dd.index, dd, 0, color=ROUGE, alpha=0.18, lw=0)
    ax.plot(dd.index, dd, color=ROUGE, lw=0.9)
    for k, (deb, creux, fin, prof) in enumerate(top_dd[:3]):
        ax.annotate(f"{pc(prof, 0)}\n{dt(creux)}", (creux, prof), xytext=(6, 0),
                    textcoords="offset points", fontsize=7.5, va="center", color="#52514e")
    ax.yaxis.set_major_formatter(fmt_pc)
    ax.set_ylabel("Baisse depuis le dernier sommet")
    axe_annees(ax)
    return fig_img(fig)


def g_distribution():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 2.8), gridspec_kw={"width_ratios": [1.5, 1]})
    a1.hist(r_tr, bins=160, density=True, color=BLEU, alpha=0.75, label="Rendements observés")
    xs = np.linspace(-0.12, 0.12, 400)
    a1.plot(xs, stats.norm.pdf(xs, mu, sd), color=ORANGE, lw=1.4, ls="--", label="Loi normale (même μ, σ)")
    a1.set_xlim(-0.06, 0.06)
    a1.xaxis.set_major_formatter(fmt_pc)
    a1.set_yticks([])
    a1.legend(loc="upper left", fontsize=7.5)
    a1.set_title("Distribution des rendements quotidiens", fontsize=9)
    (osm, osr), _ = stats.probplot(r_tr, dist="norm")
    a2.scatter(osm, osr, s=3, color=BLEU, alpha=0.6, label="Observations")
    a2.plot(osm, mu + sd * osm, color=ORANGE, lw=1.2, ls="--", label="Normale")
    a2.yaxis.set_major_formatter(fmt_pc)
    a2.set_xlabel("Quantiles théoriques (loi normale)")
    a2.set_title("Graphique quantile-quantile", fontsize=9)
    a2.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    return fig_img(fig)


def g_acf():
    fig, ax = plt.subplots(figsize=(8, 2.4))
    k = np.arange(1, H + 1)
    ax.bar(k - 0.2, acf_r, 0.4, color=BLEU, label="Rendements")
    ax.bar(k + 0.2, acf_r2, 0.4, color=ORANGE, hatch="///", edgecolor="white", lw=0,
           label="Rendements au carré")
    borne = 1.96 / np.sqrt(len(lr_tr))
    ax.axhspan(-borne, borne, color=GRIS, alpha=0.15, lw=0, label="Intervalle à 95 %")
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.set_xlabel("Décalage (jours)")
    ax.set_ylabel("Autocorrélation")
    ax.set_xticks(k)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncols=3, fontsize=7.5)
    return fig_img(fig)


def g_calendrier():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 2.9), gridspec_kw={"width_ratios": [1, 1.6]})
    m_j = [s.mean() * 100 for s in par_jour]
    e_j = [1.96 * s.std() / np.sqrt(len(s)) * 100 for s in par_jour]
    a1.bar(range(5), m_j, color=BLEU, yerr=e_j, capsize=3, ecolor=GRIS, error_kw={"lw": 0.8})
    a1.set_xticks(range(5), ["Lu", "Ma", "Me", "Je", "Ve"])
    a1.axhline(0, color="#c3c2b7", lw=0.8)
    a1.set_ylabel("Rendement moyen (%)")
    a1.set_title("Par jour de la semaine", fontsize=9)
    a1.grid(axis="x", visible=False)
    par_m_sd = r_m.groupby(r_m.index.month).std()
    par_m_n = r_m.groupby(r_m.index.month).count()
    a2.bar(range(12), par_mois.values * 100, color=[ORANGE if m in range(5, 11) else BLEU for m in range(1, 13)],
           yerr=1.96 * par_m_sd / np.sqrt(par_m_n) * 100, capsize=2, ecolor=GRIS, error_kw={"lw": 0.8})
    a2.set_xticks(range(12), [m[:3] for m in mois_noms], fontsize=7.5)
    a2.axhline(0, color="#c3c2b7", lw=0.8)
    a2.set_title("Par mois (orange : mai à octobre)", fontsize=9)
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig_img(fig)


def g_mm200():
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(val_bh.index, val_bh * cap, color=BLEU, lw=1.1, label="Achat-conservation")
    ax.plot(val_mm.index, val_mm * cap, color=ORANGE, lw=1.1, ls="--", label="Moyenne mobile 200 jours")
    hors = sig == 0
    ax.fill_between(sig.index, 0, 1, where=hors, transform=ax.get_xaxis_transform(),
                    color=GRIS, alpha=0.15, lw=0, label="Hors marché (T-bill)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_ylabel("Valeur de 10'000 USD")
    axe_annees(ax)
    ax.legend(loc="upper left", fontsize=7.5)
    return fig_img(fig)


def g_glissants():
    fig, ax = plt.subplots(figsize=(8, 3))
    styles = {"1 an": (BLEU, "-"), "5 ans": (ORANGE, "--"), "10 ans": (VERT, ":")}
    for nom_h, g in gl_series.items():
        c, l = styles[nom_h]
        ax.plot(g.index, g, color=c, ls=l, lw=1.3, label=f"Sur {nom_h} (annualisé)")
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.yaxis.set_major_formatter(fmt_pc)
    ax.set_xlabel("Fin de la période de détention")
    axe_annees(ax)
    ax.legend(loc="lower right", fontsize=7.5)
    return fig_img(fig)


def g_meilleurs():
    fig, ax = plt.subplots(figsize=(8, 2.6))
    lab = ["Tous les jours"] + [f"Sans les {n} meilleurs" for n, _, _ in best_rows[1:]]
    vals = [v for _, v, _ in best_rows]
    ax.barh(range(len(vals))[::-1], vals, color=[BLEU] + [ORANGE] * 3, height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + 300, len(vals) - 1 - i, f"{nb(v, 0)} USD", va="center", fontsize=8)
    ax.set_yticks(range(len(vals))[::-1], lab)
    ax.set_xlim(0, max(vals) * 1.22)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Valeur finale de 10'000 USD investis le {dt(idx[0])}")
    return fig_img(fig)


def g_plan():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 2.9))
    cum_inv = pd.Series(versement, index=prix_plan.index).cumsum()
    val = (versement / prix_plan).cumsum() * prix_plan
    a1.plot(val.index, val / 1000, color=ORANGE, lw=1.2, label="Valeur du plan")
    a1.plot(cum_inv.index, cum_inv / 1000, color=GRIS, lw=1.1, ls="--", label="Montant versé")
    a1.set_ylabel("milliers d'USD")
    a1.set_title(f"Plan de {versement} USD par mois", fontsize=9)
    a1.legend(loc="upper left", fontsize=7.5)
    axe_annees(a1, 4)
    diff = (ld["ls"] - ld["dca"])
    a2.bar(diff.index, diff.values * 100, width=25,
           color=np.where(diff.values > 0, BLEU, ORANGE))
    a2.axhline(0, color="#c3c2b7", lw=0.8)
    a2.set_title("Investissement unique moins étalement sur 12 mois", fontsize=9)
    a2.set_ylabel("Écart après 12 mois (points %)")
    axe_annees(a2, 4)
    fig.tight_layout()
    return fig_img(fig)


def g_chf():
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(tr.index, tr / tr.iloc[0] * 100, color=BLEU, lw=1.1, label="En USD")
    ax.plot(tr_chf.index, tr_chf / tr_chf.iloc[0] * 100, color=ORANGE, lw=1.1, ls="--", label="En CHF")
    ax.set_ylabel(f"Rendement total, base 100 au {dt(idx[0])}")
    axe_annees(ax)
    ax.legend(loc="upper left")
    return fig_img(fig)


# --------------------------------------------------------------------------
# Rapport PDF
# --------------------------------------------------------------------------
fonts = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DV", os.path.join(fonts, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", os.path.join(fonts, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVI", os.path.join(fonts, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")

ENCRE, SECOND, MUET = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#898781")
st_titre = ParagraphStyle("titre", fontName="DVB", fontSize=20, leading=24, textColor=ENCRE, spaceAfter=4)
st_sous = ParagraphStyle("sous", fontName="DV", fontSize=10.5, leading=14, textColor=SECOND, spaceAfter=14)
st_h1 = ParagraphStyle("h1", fontName="DVB", fontSize=13.5, leading=17, textColor=ENCRE, spaceBefore=6, spaceAfter=6)
st_h2 = ParagraphStyle("h2", fontName="DVB", fontSize=10.5, leading=14, textColor=ENCRE, spaceBefore=8, spaceAfter=3)
st_p = ParagraphStyle("p", fontName="DV", fontSize=9, leading=13, textColor=ENCRE, alignment=TA_LEFT, spaceAfter=5)
st_note = ParagraphStyle("note", parent=st_p, fontSize=7.5, leading=10, textColor=MUET)
st_cell = ParagraphStyle("cell", fontName="DV", fontSize=8, leading=10, textColor=ENCRE)
st_cellb = ParagraphStyle("cellb", parent=st_cell, fontName="DVB")


def P(t, s=st_p):
    return Paragraph(t, s)


def tableau(lignes, larg, entete=True, align_droite_depuis=1):
    data = [[Paragraph(str(c), st_cellb if (entete and i == 0) else st_cell) for c in l]
            for i, l in enumerate(lignes)]
    t = Table(data, colWidths=[w * cm for w in larg], hAlign="LEFT")
    style = [
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#c3c2b7")),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#e1e0d9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    t.setStyle(TableStyle(style))
    for row in data:
        for c in row[align_droite_depuis:]:
            c.style = ParagraphStyle("r", parent=c.style, alignment=2)
    return t


def pied(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(MUET)
    canvas.drawString(2 * cm, 1.2 * cm, "VT – Analyse du cours depuis le lancement")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


os.makedirs("rapport", exist_ok=True)
doc = SimpleDocTemplate("rapport/VT_analyse.pdf", pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=2 * cm,
                        title="VT – Analyse du cours depuis le lancement", author="Cédric Masset")
S = []

# --- Page de synthèse
S += [P("Vanguard Total World Stock ETF (VT)", st_titre),
      P(f"Analyse du cours depuis le lancement · {dt(idx[0])} – {dt(idx[-1])} · "
        f"{len(vt):,} séances · données Yahoo Finance".replace(",", "'"), st_sous)]

S.append(P("Synthèse", st_h1))
S.append(tableau([
    ["Indicateur", "Valeur"],
    ["Cours au lancement → dernier cours", f"{nb(px.iloc[0])} → {nb(px.iloc[-1])} USD"],
    ["Performance totale (prix / rendement total)", f"{pc(px.iloc[-1] / px.iloc[0] - 1, 0)} / {pc(tr.iloc[-1] / tr.iloc[0] - 1, 0)}"],
    ["Rendement annualisé (prix / rendement total)", f"{pc(cagr(px))} / {pc(glob['cagr'])}"],
    ["Rendement annualisé en CHF (rendement total)", pc(cagr_chf)],
    ["Volatilité annualisée", pc(glob["vol"])],
    [f"Ratio de Sharpe (taux sans risque moyen {pc(rf_moy)})", nb(glob["sharpe"])],
    ["Ratio de Sortino", nb(glob["sortino"])],
    ["Baisse maximale (rendement total)", f"{pc(glob['mdd'])} (creux le {dt(top_dd[0][1])})"],
    ["Ratio de Calmar", nb(glob["calmar"])],
    ["Probabilité de perte sur 1 an / 5 ans / 10 ans",
     " / ".join(pc(r[2], 0) for r in gl_rows)],
], [10.5, 6.5]))
S.append(Spacer(1, 8))
S.append(g_cours())
S.append(P("Échelle logarithmique : une même pente correspond à un même rendement en pourcentage. "
           "L'écart croissant entre les deux courbes correspond aux dividendes réinvestis.", st_note))

S.append(P("À retenir", st_h2))
pts = [
    f"Sur {nb(annees, 1)} ans, VT a rapporté {pc(glob['cagr'])} par an dividendes réinvestis, dont environ "
    f"{pc(glob['cagr'] - cagr(px))} par an provenant des dividendes.",
    f"Le prix de ce rendement est une volatilité de {pc(glob['vol'], 0)} par an et une baisse maximale de "
    f"{pc(abs(glob['mdd']), 0)} lors de la crise financière.",
    f"Les rendements quotidiens sont nettement non normaux (kurtosis de {nb(kurt, 1)}) et la volatilité se "
    f"regroupe dans le temps ; leur direction n'est que faiblement prévisible au jour le jour et pas du tout "
    f"à plus long terme.",
    f"Pour un investisseur en francs suisses, l'affaiblissement du dollar a réduit le rendement annualisé "
    f"de {pc(glob['cagr'] - cagr_chf)} par an ({pc(cagr_chf)} en CHF).",
]
for t in pts:
    S.append(P("•&nbsp;&nbsp;" + t))
S.append(PageBreak())

# --- 1. Rendement et risque
S.append(P("1. Rendement et risque", st_h1))
S.append(P("Rendements annuels", st_h2))
S.append(g_annuels())
meil_a, pire_a = an_tr.drop([annee_partielle_debut] + ([annee_partielle_fin] if annee_partielle_fin else [])).idxmax(), \
    an_tr.drop([annee_partielle_debut] + ([annee_partielle_fin] if annee_partielle_fin else [])).idxmin()
n_neg = int((an_tr.drop([annee_partielle_debut] + ([annee_partielle_fin] if annee_partielle_fin else [])) < 0).sum())
n_comp = len(an_tr) - 1 - (1 if annee_partielle_fin else 0)
S.append(P(f"* Années incomplètes : {annee_partielle_debut} (depuis le lancement)"
           + (f" et {annee_partielle_fin} (jusqu'au {dt(idx[-1])})." if annee_partielle_fin else ".")
           + f" Sur les {n_comp} années complètes, la meilleure est {meil_a} ({pc(an_tr[meil_a], 1, True)}), "
             f"la pire {pire_a} ({pc(an_tr[pire_a], 1, True)}) ; {n_neg} années ont été négatives.", st_note))

S.append(P("Volatilité glissante sur un an", st_h2))
S.append(g_vol())
S.append(P(f"La volatilité varie fortement selon les régimes : de {pc(vol_gl.min(), 0)} au plus calme "
           f"(année se terminant le {dt(vol_gl.idxmin())}) à {pc(vol_gl.max(), 0)} au plus agité "
           f"(année se terminant le {dt(vol_gl.idxmax())}). "
           "Une volatilité moyenne sur toute la période masque donc des périodes très différentes."))
S.append(PageBreak())

S.append(P("Baisses depuis le sommet (drawdowns)", st_h2))
S.append(g_dd())
lignes = [["Sommet", "Creux", "Retour au sommet", "Baisse", "Durée de la baisse", "Durée de récupération"]]
for deb, creux, fin, prof in top_dd:
    j_baisse = np.busday_count(deb.date(), creux.date())
    rec = f"{np.busday_count(creux.date(), fin.date())} séances" if fin is not None else "pas encore"
    lignes.append([dt(deb), dt(creux), dt(fin) if fin is not None else "–", pc(prof), f"{j_baisse} séances", rec])
S.append(tableau(lignes, [2.4, 2.4, 2.9, 2, 3.4, 3.9], align_droite_depuis=3))
S.append(P("Calculé sur le rendement total. Les durées sont exprimées en jours ouvrables.", st_note))

S.append(P("Ratios par sous-période", st_h2))
lignes = [["Période", "Rend. annualisé", "Volatilité", "Sharpe", "Sortino", "Baisse max.", "Calmar"]]
for d0, d1, ind in sous + [(r_tr.index[0], r_tr.index[-1], glob)]:
    nom = f"{d0.year}–{d1.year}" if (d0, d1) != (r_tr.index[0], r_tr.index[-1]) else "Toute la période"
    lignes.append([nom, pc(ind["cagr"]), pc(ind["vol"]), nb(ind["sharpe"]), nb(ind["sortino"]),
                   pc(ind["mdd"]), nb(ind["calmar"])])
S.append(tableau(lignes, [3.4, 2.4, 2.1, 1.8, 1.8, 2.3, 1.8]))
S.append(P("Sharpe et Sortino calculés sur les rendements excédentaires par rapport au T-bill 13 semaines "
           "(^IRX). Sortino : volatilité des seuls rendements excédentaires négatifs. "
           "Calmar : rendement annualisé divisé par la baisse maximale de la sous-période.", st_note))
S.append(PageBreak())

# --- 2. Distribution
S.append(P("2. Distribution des rendements", st_h1))
S.append(g_distribution())
S.append(tableau([
    ["Statistique (rendements quotidiens)", "Valeur", "Référence loi normale"],
    ["Moyenne / écart-type", f"{pc(mu, 3)} / {pc(sd, 2)}", "–"],
    ["Asymétrie (skewness)", nb(asym), "0"],
    ["Kurtosis", nb(kurt), "3"],
    ["Test de Jarque-Bera", f"{nb(jb_stat, 0)} (p < 0,001)" if jb_p < 0.001 else f"{nb(jb_stat, 0)} (p = {nb(jb_p, 3)})", "–"],
    ["Jours au-delà de ±4 écarts-types", f"{sig4}", f"{nb(attendu4, 2)} attendu(s)"],
    ["Pire / meilleure séance", f"{pc(r_tr.min())} ({dt(pire_j)}) / {pc(r_tr.max(), 1, True)} ({dt(meil_j)})", "–"],
], [6.2, 6.2, 4.6]))
S.append(P(f"Les queues sont beaucoup plus épaisses que celles d'une loi normale : {sig4} séances dépassent "
           f"4 écarts-types, alors que la loi normale en prévoirait {nb(attendu4, 1)} sur la période. "
           f"L'asymétrie {'négative' if asym < 0 else 'positive'} indique que les fortes baisses sont "
           f"{'plus' if asym < 0 else 'moins'} marquées que les fortes hausses."))

S.append(P("Value at Risk et Expected Shortfall (horizon 1 jour)", st_h2))
lignes = [["Niveau", "VaR historique", "ES historique", "VaR normale", "ES normale",
           "Dépassements de la VaR normale"]]
for niv, q, es, qn, esn, dep in var_rows:
    lignes.append([pc(niv, 0), pc(q, 2), pc(es, 2), pc(qn, 2), pc(esn, 2),
                   f"{pc(dep, 2)} (attendu {pc(1 - niv, 0)})"])
S.append(tableau(lignes, [1.6, 2.5, 2.4, 2.4, 2.3, 5.8]))
S.append(P("À 99 %, la VaR paramétrique normale sous-estime le risque : elle est dépassée plus souvent que "
           "prévu et l'Expected Shortfall historique est nettement plus élevée. C'est l'effet direct des queues "
           "épaisses observées ci-dessus." if var_rows[1][5] > 0.01 else
           "La VaR normale reste proche de la VaR historique sur cette période."))

S.append(P("Regroupement de la volatilité", st_h2))
S.append(g_acf())
S.append(P(f"Hormis un léger effet au premier retard (voir section 3), les rendements sont peu autocorrélés "
           f"(Ljung-Box à 10 retards : Q = {nb(lb_r[0], 1)}). Les rendements au carré le sont en revanche fortement et "
           f"durablement (Q = {nb(lb_r2[0], 0)}, p < 0,001). Une forte variation tend à être suivie "
           "d'autres fortes variations, quel que soit leur signe : c'est le phénomène que modélisent les modèles "
           "ARCH et GARCH."))
S.append(PageBreak())

# --- 3. Efficience
S.append(P("3. Efficience de marché", st_h1))
S.append(P("Autocorrélation d'ordre 1 des rendements logarithmiques", st_h2))
lignes = [["Fréquence", "Observations", "Autocorrélation", "Statistique t", "p-value"]]
for nom_f, n, r1, t, p in ac_freq:
    lignes.append([nom_f, f"{n:,}".replace(",", "'"), nb(r1, 3), nb(t, 2), nb(p, 3)])
S.append(tableau(lignes, [3.6, 3, 3.4, 3.2, 2.8]))
S.append(P("Statistique t approchée par ρ·√n (erreur-type 1/√n sous l'hypothèse de marche aléatoire). Elle surestime "
           "la significativité en présence de regroupement de la volatilité ; la version robuste de ce test est le "
           "ratio de variance VR(2) ci-dessous.", st_note))
S.append(P(f"L'autocorrélation quotidienne est négative ({nb(ac_freq[0][2], 3)}) : une hausse tend à être "
           f"légèrement corrigée le lendemain. L'effet est concentré dans les crises : hors 2008–2009 et hors "
           f"février à juin 2020, il tombe à {nb(ac1_hors_crise, 3)}. Plusieurs explications sont possibles : "
           "surréaction lors des séances agitées, ou écarts temporaires entre le cours de l'ETF et la valeur de "
           "ses actifs étrangers, cotés à d'autres heures. Aux fréquences hebdomadaire et mensuelle, aucune "
           "autocorrélation significative n'apparaît."))

S.append(P("Test du ratio de variance (Lo et MacKinlay, 1988)", st_h2))
lignes = [["Horizon q (jours)", "Ratio de variance VR(q)", "z* (robuste)", "p-value", "Lecture"]]
for q, vr, z, p in vr_rows:
    lec = ("non rejet" if p >= 0.05 else ("retour à la moyenne" if vr < 1 else "persistance"))
    lignes.append([str(q), nb(vr, 3), nb(z, 2), nb(p, 3), lec])
S.append(tableau(lignes, [3.2, 3.8, 2.8, 2.4, 4.8], align_droite_depuis=1))
S.append(P("Sous l'hypothèse de marche aléatoire, la variance des rendements sur q jours vaut q fois la "
           "variance quotidienne, donc VR(q) = 1. Un ratio inférieur à 1 signale un retour à la moyenne, un ratio "
           "supérieur à 1 une persistance. La statistique z* est robuste à l'hétéroscédasticité, indispensable "
           "ici compte tenu du regroupement de la volatilité.", st_note))

rejets = [q for q, vr, z, p in vr_rows if p < 0.05]
S.append(P(("L'hypothèse de marche aléatoire n'est rejetée pour aucun horizon. "
            if not rejets else
            f"L'hypothèse de marche aléatoire est rejetée à 5 % pour q = {', '.join(map(str, rejets))}. ")
           + "Même lorsqu'un écart est statistiquement significatif, il reste faible en valeur et il faudrait "
             "qu'il survive aux coûts de transaction pour être exploitable."))

S.append(P("Effets calendaires", st_h2))
S.append(g_calendrier())
S.append(tableau([
    ["Test", "Résultat", "p-value"],
    ["Égalité des rendements moyens par jour de la semaine (ANOVA)", f"F = {nb(f_jour, 2)}", nb(p_jour, 3)],
    ["Effet janvier : janvier contre les autres mois (Welch)",
     f"{pc(janv.mean(), 2)} contre {pc(autres.mean(), 2)} par mois", nb(p_janv, 3)],
    [f"« Sell in May » : nov.–avr. contre mai–oct. ({len(hiver)} et {len(ete)} semestres)",
     f"{pc(hiver.mean(), 1)} contre {pc(ete.mean(), 1)} par semestre", nb(p_sim, 3)],
], [8.6, 5.6, 2.8]))
S.append(P("Barres d'erreur : intervalle de confiance à 95 % de la moyenne. Avec moins de vingt observations par "
           "mois calendaire, les écarts entre mois sont dominés par le bruit statistique."
           + (" Aucun des trois effets n'est significatif à 5 %."
              if min(p_jour, p_janv, p_sim) >= 0.05 else ""), st_note))
S.append(PageBreak())

S.append(P("Stratégie de moyenne mobile à 200 jours", st_h2))
S.append(g_mm200())
S.append(tableau([
    ["", "Achat-conservation", "Moyenne mobile 200 j"],
    ["Rendement annualisé", pc(ind_bh["cagr"]), pc(ind_mm["cagr"])],
    ["Volatilité", pc(ind_bh["vol"]), pc(ind_mm["vol"])],
    ["Ratio de Sharpe", nb(ind_bh["sharpe"]), nb(ind_mm["sharpe"])],
    ["Baisse maximale", pc(ind_bh["mdd"]), pc(ind_mm["mdd"])],
    ["Valeur finale de 10'000 USD", f"{nb(val_bh.iloc[-1] * cap, 0)} USD", f"{nb(val_mm.iloc[-1] * cap, 0)} USD"],
], [6, 5.5, 5.5]))
S.append(P(f"Règle : investi dans VT si le cours de clôture de la veille dépasse sa moyenne mobile sur 200 séances, "
           f"sinon placé au taux T-bill. Coût de {pc(COUT_TRANSACTION, 2)} par changement de position, "
           f"{n_switch} changements au total ; la stratégie a été investie {pc(part_investi, 0)} du temps. "
           f"Période commune à partir du {dt(sig.index[0])}, la moyenne mobile nécessitant 200 séances d'historique.",
           st_note))
S.append(P(f"La stratégie {'réduit' if ind_mm['mdd'] > ind_bh['mdd'] else 'ne réduit pas'} la baisse maximale "
           f"({pc(ind_mm['mdd'], 0)} contre {pc(ind_bh['mdd'], 0)}), mais "
           + (f"au prix d'un rendement inférieur de {pc(ind_bh['cagr'] - ind_mm['cagr'])} par an. "
              if ind_mm["cagr"] < ind_bh["cagr"] else
              f"avec un rendement supérieur de {pc(ind_mm['cagr'] - ind_bh['cagr'])} par an. ")
           + "Elle sort souvent du marché après une chute et y revient après le rebond : les signaux tardifs "
             "et répétés (« whipsaws ») coûtent cher sur un marché qui récupère rapidement."))
S.append(PageBreak())

# --- 4. Investisseur
S.append(P("4. Horizon et comportement de l'investisseur", st_h1))
S.append(P("Rendements glissants selon l'horizon de détention", st_h2))
S.append(g_glissants())
lignes = [["Horizon", "Périodes", "Probabilité de perte", "Pire (annualisé)", "Médiane", "Meilleure"]]
for nom_h, n, pl, mn, dmn, med, mx in gl_rows:
    lignes.append([nom_h, str(n), pc(pl, 0), f"{pc(mn)} ({dmn.strftime('%m.%Y')})", pc(med), pc(mx)])
S.append(tableau(lignes, [2, 2, 3.3, 4.2, 2.6, 2.9]))
S.append(P("Fenêtres mensuelles chevauchantes, rendement total annualisé. Plus l'horizon s'allonge, plus "
           "la dispersion des résultats se resserre. Les fenêtres longues se chevauchent fortement et ne sont donc "
           "pas indépendantes : sur 10 ans, l'historique ne contient qu'environ une période et demie distincte.",
           st_note))

S.append(P("Le coût de rater les meilleurs jours", st_h2))
S.append(g_meilleurs())
S.append(P(f"Rater les 10 meilleures séances sur {len(r_tr):,} fait passer le rendement annualisé de "
           .replace(",", "'")
           + f"{pc(best_rows[0][2])} à {pc(best_rows[1][2])}. Mais l'exercice inverse est tout aussi spectaculaire : "
             f"éviter les 10 pires séances porterait la valeur finale à {nb(pire_rows[0][1], 0)} USD. "
             f"{meilleurs_proches} des 10 meilleures séances se situent à moins de deux semaines de l'une des 10 pires. "
             "Elles surviennent dans les mêmes périodes de crise : pour éviter les unes sans rater les autres, "
             "il faudrait un timing parfait."))
S.append(PageBreak())

S.append(P("Plan d'épargne mensuel et investissement unique", st_h2))
S.append(g_plan())
S.append(P(f"Un versement de {versement} USD à chaque fin de mois depuis le lancement ({len(prix_plan)} versements, "
           f"{nb(investi_plan, 0)} USD au total) vaut aujourd'hui {nb(valeur_plan, 0)} USD, soit un taux de "
           f"rendement interne de {pc(tri_plan)} par an."))
S.append(P(f"Sur les {len(ld)} points de départ mensuels possibles, investir un montant en une fois a fait mieux "
           f"que l'étaler sur 12 mois (la part non investie étant placée au T-bill) dans {pc(part_ls_gagne, 0)} des "
           f"cas, avec un avantage moyen de {pc(ecart_moyen, 1)} après 12 mois. Comme le marché monte la plupart du "
           "temps, attendre coûte en moyenne ; l'étalement limite surtout le regret d'avoir investi juste avant "
           "une baisse."))

# --- 5. CHF
S.append(P("5. Perspective d'un investisseur suisse", st_h1))
S.append(g_chf())
S.append(tableau([
    ["", "En USD", "En CHF"],
    ["Rendement annualisé (rendement total)", pc(cagr_usd), pc(cagr_chf)],
    ["Volatilité annualisée", pc(glob["vol"]), pc(vol_chf)],
    ["Évolution de l'USD/CHF sur la période", "–",
     f"{nb(chf.iloc[0], 4)} → {nb(chf.iloc[-1], 4)} ({pc(cagr_fx, 1, True)} par an)"],
], [7, 4, 6]))
S.append(P(f"Le dollar s'est déprécié face au franc, ce qui a coûté environ {pc(cagr_usd - cagr_chf)} de rendement "
           f"par an à un investisseur suisse non couvert. La corrélation quotidienne entre VT et l'USD/CHF est de "
           f"{nb(corr_fx, 2)} : "
           + ("le franc tend à s'apprécier quand les actions baissent, ce qui accentue les pertes en CHF lors des crises. "
              if corr_fx > 0 else "le dollar tend à s'apprécier légèrement quand les actions baissent, ce qui amortit "
              "un peu les pertes en CHF, mais la relation est faible. ")
           + "VT étant un fonds mondial, l'exposition réelle porte sur un panier de devises : l'USD/CHF ne capture "
             "qu'une partie de l'effet de change."))

# --- Méthodologie
S.append(P("Données et méthodologie", st_h2))
S.append(P(f"Source : Yahoo Finance, téléchargé avec la bibliothèque yfinance et mis à jour automatiquement "
           f"(dépôt GitHub cedricmasset98/donnees-marche). VT : cours quotidiens de clôture et cours ajustés "
           f"pour les dividendes et les fractionnements (« Adj Close »), du {dt(idx[0])} au {dt(idx[-1])}. "
           f"Taux sans risque : T-bill américain à 13 semaines (^IRX), converti en taux quotidien (÷ 252). "
           f"Change : USD/CHF (CHF=X), dernier cours connu pour chaque séance de VT. Annualisation sur 252 séances. "
           f"Rapport généré le {date.today():%d.%m.%Y}. Ces analyses sont descriptives et historiques ; elles ne "
           f"constituent pas un conseil en placement.", st_note))

doc.build(S, onFirstPage=pied, onLaterPages=pied)
print("rapport/VT_analyse.pdf créé")
