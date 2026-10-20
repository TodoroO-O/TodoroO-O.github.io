"""Run every part of Assignment 2 and write the figures, LaTeX tables and results.json
used by ../report.tex.     Usage:  python make_report.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from regression import (FEATURES, LAMBDAS, N_TRAIN, design_polynomial, gaussian_regression,
                        gaussian_regression_reg, load_data, polynomial_regression,
                        polynomial_regression_1d_vis, polynomial_regression_reg)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
TAB = os.path.join(HERE, "..", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# ----------------------------------------------------------------------------- style
TRAIN, TEST, THIRD = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"
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


def fmt(x, nd=3):
    """Number for a LaTeX table: plain for normal sizes, a x 10^k for huge ones."""
    if abs(x) >= 1e5:
        m, e = f"{x:.2e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{x:.{nd}f}"


def lam_tex(lam):
    return "0" if lam == 0 else f"{lam:g}"


X, t = load_data(os.path.join(HERE, "housing.data"))
R = {}

# ============================================================================= (a)
a = polynomial_regression(X, t)
R["a"] = {k: a[k] for k in ("degree", "train", "test", "rank", "n_params")}

# extra diagnostic: test RMS restricted to the test towns whose 13 features all lie inside the
# range spanned by the 100 training towns (no extrapolation needed)
Ztr, Zte = (X[:N_TRAIN] - a["mu"]) / a["sd"], (X[N_TRAIN:] - a["mu"]) / a["sd"]
inside = np.all((Zte >= Ztr.min(0) - 1e-9) & (Zte <= Ztr.max(0) + 1e-9), axis=1)
a["test_inside"] = [float(np.sqrt(np.mean((design_polynomial(Zte[inside], M) @ w_ - t[N_TRAIN:][inside]) ** 2)))
                    for M, w_ in zip(a["degree"], a["weights"])]
R["a"]["test_inside"], R["a"]["n_inside"] = a["test_inside"], int(inside.sum())

fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.semilogy(a["degree"], a["train"], "o-", color=TRAIN, label="Training RMS")
ax.semilogy(a["degree"], a["test"], "s-", color=TEST, label="Test RMS (all 406 test towns)")
ax.semilogy(a["degree"], a["test_inside"], "D--", color=THIRD, lw=1.6, ms=4.5,
            label=f"Test RMS ({inside.sum()} test towns inside the training range)")
ax.axhline(np.std(t[N_TRAIN:]), color=INK2, lw=1, ls=":",
           label=f"Predict-the-mean baseline on test ({np.std(t[N_TRAIN:]):.2f})")
for d, e in zip(a["degree"], a["test"]):
    ax.annotate(f"{e:.3g}" if e < 1e4 else f"{e:.1e}", (d, e), textcoords="offset points",
                xytext=(0, 7), ha="center", fontsize=7.5, color=INK2)
ax.set_xlabel("Polynomial degree $M$")
ax.set_ylabel("RMS error ($1000s, log scale)")
ax.set_xticks(a["degree"])
ax.legend(loc="upper left", fontsize=8.5)
save(fig, "a_poly_rms")

rows = [f"{d} & {n} & {rk} & {fmt(tr)} & {fmt(te)} & {fmt(ti)} \\\\"
        for d, n, rk, tr, te, ti in zip(a["degree"], a["n_params"], a["rank"], a["train"],
                                        a["test"], a["test_inside"])]
open(os.path.join(TAB, "a_poly.tex"), "w").write("\n".join(rows) + "\n")

# ============================================================================= (b)
w = a["weights"][0]
w = np.where(np.abs(w) < 1e-9, 0.0, w)  # CHAS is all-zero in training -> weight exactly 0
mu, sd = a["mu"], a["sd"]
w_raw = w[1:] / sd                       # effect of +1 raw unit of feature j
b_raw = w[0] - np.sum(w[1:] * mu / sd)  # intercept in raw units
R["b"] = {"w": w.tolist(), "w_raw": w_raw.tolist(), "b_raw": float(b_raw),
          "mu": mu.tolist(), "sd": sd.tolist()}
units = ["per capita crime rate", "\\% land zoned $>$25k ft$^2$", "\\% non-retail acres",
         "river dummy", "NOX (pp10m)", "rooms per dwelling", "\\% built pre-1940",
         "distance index", "highway index", "tax per \\$10k", "pupils per teacher",
         "$1000(B_k-0.63)^2$", "\\% lower status"]
rows = [f"$w_{{0}}$ & bias & -- & -- & {w[0]:.3f} & {b_raw:.2f} \\\\"]
for j, name in enumerate(FEATURES):
    hl = "\\rowcolor{boxbg} " if name in ("RM", "DIS") else ""
    sdj = "--" if name == "CHAS" else f"{sd[j]:.3g}"
    rows.append(f"{hl}$w_{{{j + 1}}}$ & {name} & {units[j]} & {sdj} & {w[j + 1]:.3f} & "
                f"{w_raw[j]:.3f} \\\\")
open(os.path.join(TAB, "b_weights.tex"), "w").write("\n".join(rows) + "\n")

fig, ax = plt.subplots(figsize=(6.2, 3.0))
order = np.arange(13)
cols = [TEST if FEATURES[j] in ("RM", "DIS") else TRAIN for j in order]
ax.bar(order, w[1:], color=cols, width=0.62, zorder=3)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(order)
ax.set_xticklabels(FEATURES, rotation=40, ha="right", fontsize=8.5)
ax.set_ylabel("Weight ($1000s per std. dev.)")
for j in (5, 7):
    ax.annotate(f"{w[j + 1]:.3f}", (j, w[j + 1]), textcoords="offset points",
                xytext=(0, 4 if w[j + 1] >= 0 else -11), ha="center", fontsize=8, color=INK)
ax.grid(axis="x", visible=False)
save(fig, "b_weights")

# ============================================================================= (c)
c = polynomial_regression_1d_vis(X, t, feature=1)
R["c"] = {"degree": list(c["fits"]), "train": [f["train"] for f in c["fits"].values()],
          "test": [f["test"] for f in c["fits"].values()], "mu": c["mu"], "sd": c["sd"],
          "n_unique_train": int(len(np.unique(c["xtr"])))}


def panel(ax, M, show_legend=False):
    f = c["fits"][M]
    ax.scatter(c["xte"], c["tte"], s=12, color=TEST, alpha=0.45, lw=0, label="Test points", zorder=2)
    ax.scatter(c["xtr"], c["ttr"], s=18, color=TRAIN, edgecolor="white", lw=0.5,
               label="Training points", zorder=3)
    ax.plot(c["grid"], f["curve"], color=INK, lw=1.8, label="Learned polynomial", zorder=4)
    ax.set_title(f"Degree {M}:  train {f['train']:.2f},  test {f['test']:.2f}", fontsize=9.5)
    ax.set_ylim(0, 55)
    if show_legend:
        ax.legend(loc="upper right", fontsize=8)


SHOW = (1, 4, 7)
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.7), sharey=True)
for ax, M in zip(axes, SHOW):
    panel(ax, M, show_legend=False)
    ax.set_xlabel("ZN (normalised)")
axes[0].set_ylabel("MEDV ($1000s)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03), fontsize=8.5)
fig.tight_layout(rect=(0, 0.05, 1, 1))
save(fig, "c_1d_selected")

fig, axes = plt.subplots(2, 4, figsize=(6.8, 3.9), sharey=True, sharex=True)
for k, ax in enumerate(axes.flat):
    if k < 7:
        panel(ax, k + 1)
        ax.title.set_fontsize(8)
    else:
        ax.axis("off")
for ax in axes[1]:
    ax.set_xlabel("ZN (normalised)", fontsize=8.5)
for ax in axes[:, 0]:
    ax.set_ylabel("MEDV ($1000s)", fontsize=8.5)
h, l = axes[0, 0].get_legend_handles_labels()
axes[1, 3].legend(h, l, loc="center", fontsize=8.5)
fig.tight_layout()
save(fig, "c_1d_all")

fig, ax = plt.subplots(figsize=(6.2, 2.8))
ax.plot(R["c"]["degree"], R["c"]["train"], "o-", color=TRAIN, label="Training RMS")
ax.plot(R["c"]["degree"], R["c"]["test"], "s-", color=TEST, label="Test RMS")
ax.set_xlabel("Polynomial degree $M$")
ax.set_ylabel("RMS error ($1000s)")
ax.legend(loc="center right")
save(fig, "c_1d_rms")


# ============================================================================= (d), (f)
def cv_plot(res, name, title_note):
    lams = np.array(res["lambdas"][1:], float)
    val = res["val"][1:]
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    for f in range(val.shape[1]):
        ax.scatter(lams * np.exp(np.random.default_rng(f).uniform(-0.12, 0.12)), val[:, f],
                   s=10, color=TRAIN, alpha=0.25, lw=0, zorder=2,
                   label="Individual folds" if f == 0 else None)
    ax.semilogx(lams, res["mean_train"][1:], "^--", color=THIRD, lw=1.5, ms=5,
                label="Mean training-fold RMS", zorder=3)
    ax.semilogx(lams, res["mean_val"][1:], "o-", color=TRAIN, label="Mean validation RMS", zorder=4)
    ax.axhline(res["mean_val"][0], color=TEST, lw=1.4, ls="--",
               label=f"$\\lambda=0$ baseline ({res['mean_val'][0]:.2f})")
    ax.axhline(res["const_val"], color=INK2, lw=1, ls=":",
               label=f"Constant (mean) model ({res['const_val']:.2f})")
    b = res["best_index"]
    ax.scatter([res["lambdas"][b]], [res["mean_val"][b]], s=110, facecolor="none",
               edgecolor=INK, lw=1.4, zorder=5)
    ax.annotate(f"best $\\lambda={res['lambdas'][b]:g}$\nRMS {res['mean_val'][b]:.2f}",
                (res["lambdas"][b], res["mean_val"][b]), textcoords="axes fraction",
                xytext=(0.7, 0.33) if title_note == "d" else (0.5, 0.1), ha="center", fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8, shrinkB=7))
    ax.set_yscale("log")
    ax.set_xlabel("Regularisation constant $\\lambda$ (log scale)")
    ax.set_ylabel("RMS error ($1000s, log scale)")
    ax.legend(loc="upper left" if title_note == "f" else "upper right", fontsize=8,
              ncol=1)
    return fig, ax


d = polynomial_regression_reg(X, t, feature=1, degree=8)
R["d"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
fig, ax = cv_plot(d, "d", "d")
ax.set_ylim(top=ax.get_ylim()[1] * 4)
save(fig, "d_poly_reg")

f = gaussian_regression_reg(X, t, K=90)
R["f"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in f.items()}
fig, ax = cv_plot(f, "f", "f")
ax.set_ylim(top=ax.get_ylim()[1] * 3)
save(fig, "f_gauss_reg")

for key, res in (("d", d), ("f", f)):
    rows = []
    for i, lam in enumerate(res["lambdas"]):
        hl = "\\rowcolor{boxbg} " if i == res["best_index"] else ""
        rows.append(f"{hl}{lam_tex(lam)} & {fmt(res['mean_train'][i])} & "
                    f"{fmt(res['mean_val'][i])} & {fmt(float(np.median(res['val'][i])))} & "
                    f"{fmt(res['test'][i])} \\\\")
    open(os.path.join(TAB, f"{key}_cv.tex"), "w").write("\n".join(rows) + "\n")

# ============================================================================= (e)
e = gaussian_regression(X, t, seed=0)
R["e"] = e
reps = [gaussian_regression(X, t, seed=s) for s in range(30)]
te_all = np.array([r["test"] for r in reps])
tr_all = np.array([r["train"] for r in reps])
R["e_repeats"] = {"test_median": np.median(te_all, 0).tolist(),
                  "train_median": np.median(tr_all, 0).tolist(),
                  "test_q25": np.percentile(te_all, 25, 0).tolist(),
                  "test_q75": np.percentile(te_all, 75, 0).tolist()}

fig, ax = plt.subplots(figsize=(6.2, 3.4))
K = np.array(e["K"])
ax.fill_between(K, np.percentile(te_all, 25, 0), np.percentile(te_all, 75, 0), color=TEST,
                alpha=0.13, lw=0, label="Test RMS, IQR over 30 permutations")
ax.fill_between(K, np.percentile(tr_all, 25, 0), np.percentile(tr_all, 75, 0), color=TRAIN,
                alpha=0.13, lw=0, label="Train RMS, IQR over 30 permutations")
ax.semilogy(K, e["train"], "o-", color=TRAIN, label="Training RMS (permutation used)")
ax.semilogy(K, e["test"], "s-", color=TEST, label="Test RMS (permutation used)")
ax.axhline(np.std(t[N_TRAIN:]), color=INK2, lw=1, ls=":",
           label=f"Predict-the-mean baseline on test ({np.std(t[N_TRAIN:]):.2f})")
ax.set_xticks(K)
ax.set_xlabel("Number of Gaussian basis functions $K$")
ax.set_ylabel("RMS error ($1000s, log scale)")
ax.set_ylim(top=3e3)
ax.legend(loc="upper left", fontsize=8)
save(fig, "e_gauss_rms")

rows = [f"{k} & {fmt(tr)} & {fmt(te)} & {fmt(mt)} \\\\" for k, tr, te, mt in
        zip(e["K"], e["train"], e["test"], R["e_repeats"]["test_median"])]
open(os.path.join(TAB, "e_gauss.tex"), "w").write("\n".join(rows) + "\n")

# width sensitivity (extra)
widths = [e["s"] / 2, e["s"], e["s"] * 2]
fig, ax = plt.subplots(figsize=(6.2, 3.0))
R["e_width"] = {}
for s_, col, mk in zip(widths, (THIRD, TEST, "#6250d6"), ("^", "s", "D")):
    te = np.median([gaussian_regression(X, t, s=s_, seed=k)["test"] for k in range(30)], 0)
    R["e_width"][f"{s_:.3f}"] = te.tolist()
    ax.semilogy(K, te, marker=mk, color=col, lw=1.6,
                label=f"$s={s_:.2f}$" + ("  (median heuristic)" if s_ == e["s"] else ""))
ax.axhline(np.std(t[N_TRAIN:]), color=INK2, lw=1, ls=":")
ax.set_xticks(K)
ax.set_xlabel("Number of Gaussian basis functions $K$")
ax.set_ylabel("Median test RMS (log scale)")
ax.legend(loc="upper left", fontsize=8)
save(fig, "e_width")

# ============================================================================= data facts
R["data"] = {
    "crim_train_max": float(X[:N_TRAIN, 0].max()), "crim_test_max": float(X[N_TRAIN:, 0].max()),
    "z_test_max": np.abs((X[N_TRAIN:] - a["mu"]) / a["sd"]).max(0).tolist(),
    "t_train_mean": float(t[:N_TRAIN].mean()), "t_train_std": float(t[:N_TRAIN].std()),
    "t_test_mean": float(t[N_TRAIN:].mean()), "t_test_std": float(t[N_TRAIN:].std()),
    "chas_train_sum": float(X[:N_TRAIN, 3].sum()), "chas_test_sum": float(X[N_TRAIN:, 3].sum()),
    "s_median": e["s"],
}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items() if k != "weights"}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


json.dump(clean(R), open(os.path.join(HERE, "results.json"), "w"), indent=1)
print(json.dumps({k: clean(R[k]) for k in ("a", "c", "data")}, indent=None)[:3000])
print("d", np.round(d["mean_val"], 3), d["best_lambda"], np.round(d["test"], 3), d["const_val"])
print("f", np.round(f["mean_val"], 3), f["best_lambda"], np.round(f["test"], 3), f["const_val"], f["s"])
print("e", np.round(e["train"], 3), np.round(e["test"], 3))
print("b", np.round(w, 3), np.round(w_raw, 3), round(b_raw, 3))
