"""SYDE 572 - Assignment 3: PCA, marginal visualisation and ISOMAP on the Iris data.

Run every part of the assignment and write the figures, LaTeX tables and results.json
used by ../report.tex.        Usage:  python make_report.py

    Part 1  PCA by hand on the 5-point toy set (numerical check + figure)
    Part 2  marginal views of Iris: every 1-, 2- and 3-feature subset, plus a separability table
    Part 3  PCA with scikit-learn: projections on PC1, PC1-2, PC1-3 and the PC vectors
    Part 4  explained variance through the Frobenius norm of the centred data matrix
    Part 5  ISOMAP (k = 3 neighbours) vs PCA: embeddings, classification and compression metrics

Data: scikit-learn's bundled copy of the UCI Iris data (150 x 4, three classes of 50).
"""

import itertools
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, kneighbors_graph
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
TAB = os.path.join(HERE, "..", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

SEED = 0
K_ISO = 3          # number of neighbours the assignment asks for in Part 5
K_CLF = 5          # k of the k-NN classifier / decoder used as an evaluation tool

# ----------------------------------------------------------------------------- style
CLS_COL = ["#2a78d6", "#eb6834", "#1baf7a"]          # setosa, versicolor, virginica
CLS_MRK = ["o", "s", "^"]
INK, INK2, GRID, ACC = "#0b0b0b", "#52514e", "#e4e3de", "#1F4E79"
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 5, "savefig.bbox": "tight",
    "savefig.dpi": 200,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)


def write_table(name, rows):
    with open(os.path.join(TAB, name + ".tex"), "w") as f:
        f.write("\n".join(rows) + "\n")


def class_legend(names, ax=None, fig=None, **kw):
    handles = [Line2D([], [], ls="", marker=CLS_MRK[c], color=CLS_COL[c], markersize=6,
                      label=names[c]) for c in range(3)]
    return (fig or ax).legend(handles=handles, **kw)


def scatter_classes(ax, Z, y, s=18, three_d=False):
    for c in range(3):
        m = y == c
        if three_d:
            ax.scatter(Z[m, 0], Z[m, 1], Z[m, 2], s=s, c=CLS_COL[c], marker=CLS_MRK[c],
                       edgecolors="white", linewidths=0.3, depthshade=False)
        else:
            ax.scatter(Z[m, 0], Z[m, 1], s=s, c=CLS_COL[c], marker=CLS_MRK[c],
                       edgecolors="white", linewidths=0.4, alpha=0.9, zorder=3)


def style_3d(ax):
    ax.set_box_aspect(None, zoom=0.86)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("#fafaf8")
        axis.pane.set_edgecolor(GRID)
        axis._axinfo["grid"]["color"] = GRID
        axis._axinfo["grid"]["linewidth"] = 0.5
    ax.tick_params(labelsize=7, pad=0)


def hist_classes(ax, z, y, bins):
    for c in range(3):
        ax.hist(z[y == c], bins=bins, color=CLS_COL[c], alpha=0.45, edgecolor=CLS_COL[c],
                linewidth=1.0, histtype="stepfilled")
        ax.hist(z[y == c], bins=bins, color=CLS_COL[c], linewidth=1.2, histtype="step")
    ax.set_ylabel("count")


def strip_classes(ax, z, y, rng):
    for c in range(3):
        m = y == c
        ax.scatter(z[m], c + 0.25 * (rng.random(m.sum()) - 0.5), s=12, c=CLS_COL[c],
                   marker=CLS_MRK[c], edgecolors="white", linewidths=0.3, zorder=3)
    ax.set_yticks([0, 1, 2])
    ax.set_ylim(-0.5, 2.5)
    ax.invert_yaxis()


def knn_cv(model, X, y, n_rep=5):
    """Mean / std of repeated stratified 10-fold CV accuracy (the whole pipeline is
    refitted inside each fold, so the embedding never sees its own test points)."""
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=n_rep, random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = cross_val_score(model, X, y, cv=cv)
    return float(s.mean()), float(s.std())


def linearly_separable(Xs, lab):
    """True iff some hyperplane w.x + b separates the two groups exactly, i.e. the LP
    s_i (w.x_i + b) >= 1 (s_i = +-1) is feasible."""
    s_ = np.where(lab, 1.0, -1.0)
    A = -s_[:, None] * np.hstack([Xs, np.ones((len(Xs), 1))])
    res = linprog(np.zeros(Xs.shape[1] + 1), A_ub=A, b_ub=-np.ones(len(Xs)),
                  bounds=[(None, None)] * (Xs.shape[1] + 1), method="highs")
    return bool(res.status == 0)


def residual_variance(D_ref, Z):
    """Tenenbaum et al. (2000): 1 - R^2 between reference distances and embedding distances."""
    r = np.corrcoef(squareform(D_ref, checks=False), pdist(Z))[0, 1]
    return float(1 - r ** 2)


R = {}   # everything that goes into results.json

# =============================================================================
# Part 1 - PCA by hand
# =============================================================================
X1 = np.array([[-2, -1], [-1, 0], [0, 1], [1, 2], [2, 3]], dtype=float)
mu1 = X1.mean(0)
Xc1 = X1 - mu1
S1 = Xc1.T @ Xc1 / len(X1)                       # 1/N covariance
evals1, evecs1 = np.linalg.eigh(S1)
order = np.argsort(evals1)[::-1]
evals1, evecs1 = evals1[order], evecs1[:, order]
w1 = evecs1[:, 0] * np.sign(evecs1[0, 0])        # sign so that w = (+, +)


def recon1(x):
    a = w1 @ (x - mu1)
    xh = mu1 + a * w1
    return a, xh, float(np.linalg.norm(x - xh))


R["part1"] = {"mean": mu1.tolist(), "cov": S1.tolist(), "eigvals": evals1.tolist(),
              "w1": w1.tolist()}
for p in [(3, 3), (3, 4)]:
    a, xh, e = recon1(np.array(p, float))
    # what you would get if you forgot to centre the data (line through the origin)
    e_nc = float(np.linalg.norm(np.array(p) - (w1 @ np.array(p)) * w1))
    R["part1"][str(p)] = {"code": a, "xhat": xh.tolist(), "err": e, "err_uncentred": e_nc}

fig, ax = plt.subplots(figsize=(6.2, 4.6))
tt = np.linspace(-6.5, 6.5, 2)
ax.plot(mu1[0] + tt * w1[0], mu1[1] + tt * w1[1], color=ACC, lw=1.6,
        label=r"PC1: $\mathbf{x}=\bar{\mathbf{x}}+a\,\mathbf{w}_1$")
ax.plot(tt * w1[0], tt * w1[1], color=INK2, lw=1, ls=":",
        label="same direction through the origin (no centring)")
ax.scatter(X1[:, 0], X1[:, 1], s=46, color=CLS_COL[0], zorder=4, label="data")
ax.scatter(*mu1, s=80, marker="X", color=INK, zorder=5, label=r"mean $\bar{\mathbf{x}}=(0,1)$")
for p, col in [((3, 3), CLS_COL[1]), ((3, 4), CLS_COL[2])]:
    p = np.array(p, float)
    _, xh, e = recon1(p)
    ax.scatter(*p, s=70, marker="D", color=col, zorder=6, edgecolors="white")
    if e > 1e-9:
        ax.plot([p[0], xh[0]], [p[1], xh[1]], color=col, lw=1.6, ls="--", zorder=5)
        ax.scatter(*xh, s=40, marker="o", facecolors="white", edgecolors=col, zorder=6,
                   linewidths=1.5)
        ax.annotate(r"$\hat{\mathbf{x}}=(2.5,\,3.5)$", xh, xytext=(-66, 6),
                    textcoords="offset points", color=col, fontsize=9)
        ax.annotate(rf"$(3,3)$, error $={e:.3f}$", p, xytext=(8, -12),
                    textcoords="offset points", color=col, fontsize=9)
    else:
        ax.annotate("$(3,4)$ lies on PC1,\nerror $=0$", p, xytext=(10, -4),
                    textcoords="offset points", color=col, fontsize=9)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_xlim(-3, 6)
ax.set_ylim(-2, 5)
ax.set_aspect("equal")
ax.legend(loc="upper left", fontsize=8)
save(fig, "p1_pca_by_hand")

# =============================================================================
# Data
# =============================================================================
iris = load_iris()
X, y = iris.data, iris.target
FEAT = ["sepal length", "sepal width", "petal length", "petal width"]
FSHORT = ["SL", "SW", "PL", "PW"]
NAMES = [n.capitalize() for n in iris.target_names]
N, D = X.shape
rng = np.random.default_rng(SEED)
R["data"] = {"N": N, "D": D, "class_counts": np.bincount(y).tolist(),
             "mean": X.mean(0).tolist(), "std": X.std(0, ddof=1).tolist()}

# =============================================================================
# Part 2 - marginal (feature-subset) visualisations
# =============================================================================
# --- 1 feature: histogram + strip plot per feature
fig, axes = plt.subplots(2, 4, figsize=(11, 4.4), sharex="col",
                         gridspec_kw={"height_ratios": [2.2, 1]})
for j in range(4):
    bins = np.linspace(X[:, j].min(), X[:, j].max(), 22)
    hist_classes(axes[0, j], X[:, j], y, bins)
    axes[0, j].set_title(FEAT[j].capitalize() + " (cm)", fontsize=10)
    strip_classes(axes[1, j], X[:, j], y, rng)
    axes[1, j].set_yticklabels(NAMES if j == 0 else [], fontsize=8)
    axes[1, j].set_xlabel(FEAT[j] + " (cm)")
    if j:
        axes[0, j].set_ylabel("")
class_legend(NAMES, fig=fig, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
fig.tight_layout()
save(fig, "p2_1d")

# --- 2 features: full scatter-plot matrix
fig, axes = plt.subplots(4, 4, figsize=(9, 8.6))
for i in range(4):
    for j in range(4):
        ax = axes[i, j]
        if i == j:
            bins = np.linspace(X[:, j].min(), X[:, j].max(), 20)
            hist_classes(ax, X[:, j], y, bins)
            ax.set_ylabel("")
            ax.set_yticks([])
        else:
            scatter_classes(ax, X[:, [j, i]], y, s=10)
        if i == 3:
            ax.set_xlabel(FEAT[j] + " (cm)")
        else:
            ax.set_xticklabels([])
        if j == 0:
            ax.set_ylabel(FEAT[i] + " (cm)")
        elif i != j and not (i == 0 and j == 1):
            ax.set_yticklabels([])
class_legend(NAMES, fig=fig, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout()
save(fig, "p2_2d")

# --- 3 features: the four triplets
fig = plt.figure(figsize=(10, 9))
for k, trip in enumerate(itertools.combinations(range(4), 3)):
    ax = fig.add_subplot(2, 2, k + 1, projection="3d")
    scatter_classes(ax, X[:, trip], y, s=14, three_d=True)
    ax.set_xlabel(FEAT[trip[0]], fontsize=8, labelpad=-2)
    ax.set_ylabel(FEAT[trip[1]], fontsize=8, labelpad=-2)
    ax.set_zlabel(FEAT[trip[2]], fontsize=8, labelpad=-4)
    ax.set_title(" + ".join(FSHORT[t] for t in trip), fontsize=10)
    ax.view_init(elev=20, azim=-58)
    style_3d(ax)
class_legend(NAMES, fig=fig, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.99))
fig.subplots_adjust(wspace=0.02, hspace=0.08)
save(fig, "p2_3d")

# --- separability of every feature subset
rows, sep = [], []
for r in (1, 2, 3, 4):
    for sub in itertools.combinations(range(4), r):
        Xs = X[:, sub]
        acc, sd = knn_cv(KNeighborsClassifier(K_CLF), Xs, y)
        sil = float(silhouette_score(Xs, y))
        # Exact linear-separability tests (LP feasibility): setosa vs rest, versicolor vs virginica
        setosa_sep = linearly_separable(Xs, y == 0)
        m = y > 0
        vv_sep = linearly_separable(Xs[m], y[m] == 2)
        # graded measure for the hard pair: training errors of Fisher's LDA
        lda = LinearDiscriminantAnalysis().fit(Xs[m], y[m])
        vv_err = int(np.sum(lda.predict(Xs[m]) != y[m]))
        name = " + ".join(FSHORT[s] for s in sub)
        sep.append({"subset": name, "knn_acc": acc, "knn_std": sd, "silhouette": sil,
                    "setosa_separable": setosa_sep, "vv_separable": vv_sep,
                    "vv_lda_errors": vv_err})
for r in sep:
    rows.append(f"{r['subset']} & {100 * r['knn_acc']:.1f} $\\pm$ {100 * r['knn_std']:.1f} & "
                f"{r['silhouette']:.3f} & {'yes' if r['setosa_separable'] else 'no'} & "
                f"{'yes' if r['vv_separable'] else 'no'} & {r['vv_lda_errors']} \\\\")
    if r["subset"] in ("PW", "PL + PW", "SW + PL + PW"):
        rows.append("\\midrule")
write_table("p2_separability", rows)
R["part2"] = sep

# =============================================================================
# Part 3 - PCA with scikit-learn
# =============================================================================
pca = PCA(n_components=4).fit(X)
Z = pca.transform(X)
W = pca.components_                                   # rows are the PC directions
# cross-check against an eigen-decomposition of the (N-1) covariance matrix
C = np.cov(X.T)
ev, V = np.linalg.eigh(C)
ev, V = ev[::-1], V[:, ::-1]
assert np.allclose(ev, pca.explained_variance_)
assert np.allclose(np.abs(V.T), np.abs(W), atol=1e-10)

R["part3"] = {"mean": pca.mean_.tolist(), "components": W.tolist(),
              "explained_variance": pca.explained_variance_.tolist(),
              "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
              "singular_values": pca.singular_values_.tolist(), "cov": C.tolist()}

rows = []
for k in range(4):
    rows.append(f"PC{k + 1} & " + " & ".join(f"${v:+.4f}$" for v in W[k]) +
                f" & {pca.explained_variance_[k]:.4f} & {100 * pca.explained_variance_ratio_[k]:.2f}\\% \\\\")
write_table("p3_components", rows)

rows = [" & ".join(f"{v:.4f}" for v in C[i]) for i in range(4)]
rows = [FSHORT[i] + " & " + rows[i] + " \\\\" for i in range(4)]
write_table("p3_cov", rows)

# --- PC1 only
fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.9), sharex=True,
                         gridspec_kw={"height_ratios": [2.2, 1]})
bins = np.linspace(Z[:, 0].min(), Z[:, 0].max(), 30)
hist_classes(axes[0], Z[:, 0], y, bins)
strip_classes(axes[1], Z[:, 0], y, rng)
axes[1].set_yticklabels(NAMES, fontsize=8)
axes[1].set_xlabel(rf"$z_1$ = projection on PC1 (cm), {100 * pca.explained_variance_ratio_[0]:.2f}% of variance")
class_legend(NAMES, ax=axes[0], loc="upper center", ncol=3)
fig.tight_layout()
save(fig, "p3_pc1")

# --- PC1-PC2 with feature loadings (biplot)
fig, ax = plt.subplots(figsize=(6.8, 4.8))
scatter_classes(ax, Z[:, :2], y, s=22)
scale = 1.6
for j in range(4):
    ax.annotate("", xy=(scale * W[0, j], scale * W[1, j]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3), annotation_clip=False, zorder=5)
    off = {0: (4, 4), 1: (-30, 6), 2: (4, -12), 3: (-6, -14)}[j]
    ax.annotate(FEAT[j], (scale * W[0, j], scale * W[1, j]), xytext=off,
                textcoords="offset points", fontsize=8.5, color=INK)
ax.set_xlabel(rf"$z_1$ (PC1, {100 * pca.explained_variance_ratio_[0]:.2f}%)")
ax.set_ylabel(rf"$z_2$ (PC2, {100 * pca.explained_variance_ratio_[1]:.2f}%)")
ax.set_aspect("equal")
class_legend(NAMES, ax=ax, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.0))
save(fig, "p3_pc12")

# --- PC1-PC2-PC3
fig = plt.figure(figsize=(9.6, 4.6))
for k, (el, az) in enumerate([(22, -60), (18, 35)]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    scatter_classes(ax, Z[:, :3], y, s=16, three_d=True)
    ax.set_xlabel(r"$z_1$ (PC1)", fontsize=9, labelpad=-2)
    ax.set_ylabel(r"$z_2$ (PC2)", fontsize=9, labelpad=-2)
    ax.set_zlabel(r"$z_3$ (PC3)", fontsize=9, labelpad=-4)
    ax.view_init(elev=el, azim=az)
    ax.set_title(["oblique view", "rotated view"][k], fontsize=9)
    style_3d(ax)
class_legend(NAMES, fig=fig, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
fig.subplots_adjust(wspace=0.05)
save(fig, "p3_pc123")

# --- loadings bar chart
fig, axes = plt.subplots(1, 4, figsize=(10, 2.5), sharey=True)
for k in range(4):
    cols = [CLS_COL[0] if v >= 0 else CLS_COL[1] for v in W[k]]
    axes[k].bar(range(4), W[k], color=cols, width=0.65)
    axes[k].axhline(0, color=INK2, lw=0.8)
    axes[k].set_xticks(range(4))
    axes[k].set_xticklabels(FSHORT)
    axes[k].set_title(rf"$\mathbf{{w}}_{k + 1}$  ({100 * pca.explained_variance_ratio_[k]:.2f}%)", fontsize=10)
    axes[k].set_ylim(-1, 1)
axes[0].set_ylabel("loading")
fig.tight_layout()
save(fig, "p3_loadings")

# --- standardised PCA, for comparison (appendix)
Xs_std = StandardScaler().fit_transform(X)
pca_s = PCA(n_components=4).fit(Xs_std)
Zs = pca_s.transform(Xs_std)
R["part3"]["std"] = {"components": pca_s.components_.tolist(),
                     "explained_variance_ratio": pca_s.explained_variance_ratio_.tolist()}
rows = []
for k in range(4):
    rows.append(f"PC{k + 1} & " + " & ".join(f"${v:+.4f}$" for v in pca_s.components_[k]) +
                f" & {pca_s.explained_variance_[k]:.4f} & {100 * pca_s.explained_variance_ratio_[k]:.2f}\\% \\\\")
write_table("p3_components_std", rows)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
scatter_classes(axes[0], Z[:, :2], y)
axes[0].set_title("PCA on raw features (cm)", fontsize=10)
scatter_classes(axes[1], Zs[:, :2], y)
axes[1].set_title("PCA on standardised features ($z$-scores)", fontsize=10)
for ax in axes:
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
class_legend(NAMES, fig=fig, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout()
save(fig, "p3_std_compare")

# =============================================================================
# Part 4 - data compression: explained variance via the Frobenius norm
# =============================================================================
Xc = X - X.mean(0)
fro2 = float(np.linalg.norm(Xc, "fro") ** 2)
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
assert np.isclose(fro2, np.sum(s ** 2))
rows, p4 = [], []
for k in range(1, 5):
    Wk = Vt[:k].T
    Xhat = Xc @ Wk @ Wk.T
    res2 = float(np.linalg.norm(Xc - Xhat, "fro") ** 2)
    kept2 = float(np.linalg.norm(Xhat, "fro") ** 2)
    p4.append({"k": k, "sigma2": float(s[k - 1] ** 2), "kept": kept2, "residual": res2,
               "explained": kept2 / fro2, "rmse": float(np.sqrt(res2 / N))})
    rows.append(f"{k} & {s[k - 1]:.4f} & {s[k - 1] ** 2:.4f} & "
                f"{100 * s[k - 1] ** 2 / fro2:.2f}\\% & {kept2:.4f} & {res2:.4f} & "
                f"\\textbf{{{100 * kept2 / fro2:.2f}\\%}} \\\\")
write_table("p4_frobenius", rows)
cum_std = np.cumsum(pca_s.explained_variance_ratio_)
R["part4"] = {"fro2": fro2, "trace_cov_times_N_minus_1": float(np.trace(C) * (N - 1)),
              "rows": p4, "k50": int(np.argmax(np.cumsum(s ** 2) / fro2 >= 0.5) + 1),
              "k50_std": int(np.argmax(cum_std >= 0.5) + 1), "cum_std": cum_std.tolist(),
              "k95": int(np.argmax(np.cumsum(s ** 2) / fro2 >= 0.95) + 1),
              "k99": int(np.argmax(np.cumsum(s ** 2) / fro2 >= 0.99) + 1)}

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, ratio, title in [(axes[0], s ** 2 / fro2, "raw features (cm), as in Part 3"),
                         (axes[1], pca_s.explained_variance_ratio_, "standardised features")]:
    ks = np.arange(1, 5)
    cum = np.cumsum(ratio)
    ax.bar(ks, 100 * ratio, color=CLS_COL[0], alpha=0.8, width=0.6, label="individual PC")
    ax.plot(ks, 100 * cum, color=CLS_COL[1], marker="o", label="cumulative")
    for k, c in zip(ks, cum):
        ax.annotate(f"{100 * c:.2f}%", (k, 100 * c), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=8.5, color=CLS_COL[1])
    ax.axhline(50, color=INK2, ls="--", lw=1)
    ax.text(4.35, 51.5, "50%", color=INK2, fontsize=8.5, ha="right")
    ax.set_xticks(ks)
    ax.set_xlabel("number of principal components $k$")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 112)
axes[0].set_ylabel("variance explained (%)")
axes[0].legend(loc="upper right", bbox_to_anchor=(1.0, 0.8))
fig.tight_layout()
save(fig, "p4_scree")

# =============================================================================
# Part 5 - ISOMAP with k = 3 neighbours, compared with PCA
# =============================================================================
G = kneighbors_graph(X, K_ISO, mode="distance")
n_cc, cc_lab = connected_components(G, directed=False)
with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter("always")
    iso = Isomap(n_neighbors=K_ISO, n_components=3).fit(X)
iso_warn = [str(w.message) for w in wlist if "connected components" in str(w.message)]
Y = iso.embedding_
DG = iso.dist_matrix_                     # approximate geodesic distances (graph completed)
DE = squareform(pdist(X))

# Which edge did sklearn add to join the two components?  (the closest pair across them)
A, B = np.where(cc_lab == 0)[0], np.where(cc_lab == 1)[0]
dAB = DE[np.ix_(A, B)]
ia, ib = np.unravel_index(np.argmin(dAB), dAB.shape)
bridge = (int(A[ia]), int(B[ib]), float(dAB[ia, ib]))

# Same geodesic distances, embedded with metric MDS by SMACOF (iterative stress
# majorisation, the gradient-descent style MDS described in the assignment)
mds = MDS(n_components=2, metric="precomputed", metric_mds=True, init="random", n_init=8,
          max_iter=1000, eps=1e-9, random_state=SEED, normalized_stress=False)
Y_mds = mds.fit_transform(DG)


def procrustes_align(A_, B_):
    """Rotate/reflect B_ onto A_ (both centred) so that embeddings can be compared visually."""
    A0, B0 = A_ - A_.mean(0), B_ - B_.mean(0)
    u, _, vt = np.linalg.svd(B0.T @ A0)
    return B0 @ u @ vt


Y_mds = procrustes_align(Y[:, :2], Y_mds)
R["part5"] = {"n_components_graph": int(n_cc),
              "component_sizes": np.bincount(cc_lab).tolist(),
              "component_classes": [np.bincount(y[cc_lab == c], minlength=3).tolist()
                                    for c in range(n_cc)],
              "sklearn_warning": iso_warn[:1], "bridge": bridge,
              "bridge_classes": [int(y[bridge[0]]), int(y[bridge[1]])],
              "geo_max": float(DG.max()), "euc_max": float(DE.max()),
              "iso_eigvals": iso.kernel_pca_.eigenvalues_.tolist(),
              "iso_reconstruction_error_3d": float(iso.reconstruction_error()),
              "mds_stress": float(mds.stress_),
              "mds_vs_iso_rmsd": float(np.sqrt(np.mean(np.sum((Y_mds - (Y[:, :2] - Y[:, :2].mean(0))) ** 2, 1))))}
# fraction of the "geodesic" variance (classical-MDS eigenvalues) captured, like PCA's ratio
B_gram = -0.5 * (np.eye(N) - 1 / N) @ (DG ** 2) @ (np.eye(N) - 1 / N)
gev = np.sort(np.linalg.eigvalsh(B_gram))[::-1]
R["part5"]["geo_eig_ratio"] = (gev[:4] / gev[gev > 0].sum()).tolist()
R["part5"]["geo_neg_eig_frac"] = float(-gev[gev < 0].sum() / np.abs(gev).sum())

# --- main figure: PCA 2-D vs ISOMAP 2-D
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
scatter_classes(axes[0], Z[:, :2], y, s=20)
axes[0].set_title("PCA (linear), top 2 components", fontsize=10)
axes[0].set_xlabel("$z_1$")
axes[0].set_ylabel("$z_2$")
scatter_classes(axes[1], Y[:, :2], y, s=20)
axes[1].set_title(rf"ISOMAP, $k={K_ISO}$ nearest neighbours, 2 components", fontsize=10)
axes[1].set_xlabel("ISOMAP component 1")
axes[1].set_ylabel("ISOMAP component 2")
class_legend(NAMES, fig=fig, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
fig.tight_layout()
save(fig, "p5_isomap_vs_pca")

# --- ISOMAP alone, larger, with the 3-NN graph drawn in the embedding
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
Gs = G.maximum(G.T).tocoo()
for ax, E, ttl in [(axes[0], Z[:, :2], "3-NN graph drawn on the PCA projection"),
                   (axes[1], Y[:, :2], r"ISOMAP embedding ($k=3$) with the same graph")]:
    for i, j in zip(Gs.row, Gs.col):
        if i < j:
            ax.plot(E[[i, j], 0], E[[i, j], 1], color="#9b9a95", lw=0.5, zorder=1)
    i, j = bridge[:2]
    ax.plot(E[[i, j], 0], E[[i, j], 1], color="#d62728", lw=2, zorder=2,
            label=f"edge added by sklearn to join\nthe 2 components ({bridge[2]:.3f} cm)")
    scatter_classes(ax, E, y, s=16)
    ax.set_title(ttl, fontsize=10)
    ax.legend(loc="lower right" if ax is axes[0] else "upper left", fontsize=8)
axes[0].set_xlabel("$z_1$")
axes[0].set_ylabel("$z_2$")
axes[1].set_xlabel("ISOMAP component 1")
axes[1].set_ylabel("ISOMAP component 2")
class_legend(NAMES, fig=fig, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
fig.tight_layout()
save(fig, "p5_graph")

# --- ISOMAP (kernel-PCA / classical MDS) vs SMACOF metric MDS on the same geodesic distances
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True, sharey=True)
scatter_classes(axes[0], Y[:, :2] - Y[:, :2].mean(0), y)
axes[0].set_title("sklearn Isomap (classical MDS by eigen-decomposition)", fontsize=10)
scatter_classes(axes[1], Y_mds, y)
axes[1].set_title("metric MDS by SMACOF on the same geodesic distances", fontsize=10)
for ax in axes:
    ax.set_xlabel("component 1")
axes[0].set_ylabel("component 2")
class_legend(NAMES, fig=fig, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
fig.tight_layout()
save(fig, "p5_mds")

# --- ISOMAP 1-D vs PCA 1-D
fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.6))
strip_classes(axes[0], Z[:, 0], y, rng)
axes[0].set_yticklabels(NAMES, fontsize=8)
axes[0].set_title("PCA, PC1 only", fontsize=10)
strip_classes(axes[1], Y[:, 0], y, rng)
axes[1].set_yticklabels(NAMES, fontsize=8)
axes[1].set_title(r"ISOMAP ($k=3$), first component only", fontsize=10)
fig.tight_layout()
save(fig, "p5_1d")

# --- sweep over k
ks = [3, 5, 10, 20, 50]
fig, axes = plt.subplots(1, len(ks), figsize=(14, 3.0))
for ax, k in zip(axes, ks):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Yk = Isomap(n_neighbors=k, n_components=2).fit_transform(X)
    scatter_classes(ax, Yk, y, s=8)
    ax.set_title(f"$k={k}$", fontsize=10)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
class_legend(NAMES, fig=fig, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.1))
fig.tight_layout()
save(fig, "p5_ksweep")

# --- quantitative comparison for d = 1, 2, 3
cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)


def recon_rmse(method, d):
    """Out-of-fold reconstruction RMSE (cm per sample, over all 4 coordinates).
    PCA decodes linearly (inverse_transform). ISOMAP has no inverse map, so a k-NN
    regressor fitted on (embedding -> original features) of the training fold is used."""
    err = []
    for tr, te in cv10.split(X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method == "pca":
                m = PCA(d).fit(X[tr])
                Xh = m.inverse_transform(m.transform(X[te]))
            elif method == "pca_knn":
                m = PCA(d).fit(X[tr])
                dec = KNeighborsRegressor(K_CLF).fit(m.transform(X[tr]), X[tr])
                Xh = dec.predict(m.transform(X[te]))
            else:
                m = Isomap(n_neighbors=K_ISO, n_components=d).fit(X[tr])
                dec = KNeighborsRegressor(K_CLF).fit(m.embedding_, X[tr])
                Xh = dec.predict(m.transform(X[te]))
        err.append(np.sum((X[te] - Xh) ** 2, 1))
    return float(np.sqrt(np.mean(np.concatenate(err))))


cmp_rows, cmp = [], []
acc4 = knn_cv(KNeighborsClassifier(K_CLF), X, y)
for d in (1, 2, 3):
    Zd = Z[:, :d]
    Yd = Y
    if d < 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Yd = Isomap(n_neighbors=K_ISO, n_components=d).fit_transform(X)
    for meth, E, model in [
        ("PCA", Zd, make_pipeline(PCA(d), KNeighborsClassifier(K_CLF))),
        ("ISOMAP", Yd, make_pipeline(Isomap(n_neighbors=K_ISO, n_components=d),
                                     KNeighborsClassifier(K_CLF)))]:
        acc, sd = knn_cv(model, X, y)
        rec = {"d": d, "method": meth, "knn_acc": acc, "knn_std": sd,
               "trust": float(trustworthiness(X, E, n_neighbors=K_CLF)),
               "rv_euc": residual_variance(DE, E), "rv_geo": residual_variance(DG, E),
               "sil": float(silhouette_score(E, y)),
               "recon": recon_rmse("pca" if meth == "PCA" else "iso", d)}
        if meth == "PCA":
            rec["recon_knn"] = recon_rmse("pca_knn", d)
        cmp.append(rec)
best = {}
for key, better in [("knn_acc", max), ("trust", max), ("rv_euc", min), ("rv_geo", min),
                    ("sil", max), ("recon", min)]:
    for d in (1, 2, 3):
        vals = [r[key] for r in cmp if r["d"] == d]
        best[(key, d)] = better(vals)


def fmt(r, key, txt):
    return f"\\textbf{{{txt}}}" if np.isclose(r[key], best[(key, r['d'])]) else txt


for r in cmp:
    cmp_rows.append(
        f"{r['d'] if r['method'] == 'PCA' else ''} & {r['method']} & "
        + fmt(r, "knn_acc", f"{100 * r['knn_acc']:.1f}") + f" $\\pm$ {100 * r['knn_std']:.1f} & "
        + fmt(r, "sil", f"{r['sil']:.3f}") + " & "
        + fmt(r, "trust", f"{r['trust']:.4f}") + " & "
        + fmt(r, "rv_euc", f"{r['rv_euc']:.4f}") + " & "
        + fmt(r, "rv_geo", f"{r['rv_geo']:.4f}") + " & "
        + fmt(r, "recon", f"{r['recon']:.3f}") + " \\\\")
    if r["method"] == "ISOMAP" and r["d"] < 3:
        cmp_rows.append("\\midrule")
write_table("p5_compare", cmp_rows)
R["part5"]["compare"] = cmp
R["part5"]["knn_acc_4d"] = acc4

# k sweep metrics (2-D)
sweep = []
for k in [3, 5, 8, 10, 15, 20, 30, 50]:
    model = make_pipeline(Isomap(n_neighbors=k, n_components=2), KNeighborsClassifier(K_CLF))
    acc, sd = knn_cv(model, X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        isok = Isomap(n_neighbors=k, n_components=2).fit(X)
    ncc = connected_components(kneighbors_graph(X, k), directed=False)[0]
    sweep.append({"k": k, "acc": acc, "std": sd, "n_cc": int(ncc),
                  "trust": float(trustworthiness(X, isok.embedding_, n_neighbors=K_CLF)),
                  "rv_geo": residual_variance(isok.dist_matrix_, isok.embedding_)})
R["part5"]["sweep"] = sweep
write_table("p5_sweep", [f"{s_['k']} & {s_['n_cc']} & {100 * s_['acc']:.1f} $\\pm$ {100 * s_['std']:.1f}"
                         f" & {s_['trust']:.4f} & {s_['rv_geo']:.4f} \\\\" for s_ in sweep])

with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(R, f, indent=2)
print(json.dumps({k: R[k] for k in ("part1", "part4")}, indent=1))
print(json.dumps(R["part5"], indent=1)[:6000])
for r in R["part2"]:
    print(r)
print(np.array2string(W, precision=4), pca.explained_variance_ratio_)
