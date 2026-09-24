"""
SYDE 572 - Assignment 1, Part 2
================================
Least-squares fitting of a line  y = m x + b  and a parabola  y = a x^2 + b x + c
to the points (0, 0.5), (2, 3.5), (1, 1.5), (3, 7.5), by minimising

    MSE(theta) = (1/n) * sum_i ( g(x_i; theta) - y_i )^2 .

Run this file to reproduce every Part 2 result:

    python fitting.py

It prints the parameter/MSE tables and writes the Part 2 figures to ../media/.

Contents
--------
Section 1   The data, the design matrix, the MSE and its derivatives
Section 2   Analytical solution: the normal equations
Section 3   Numerical solution: Newton-Raphson, one parameter per step
Section 4   Numerical solution: the full multivariate Newton step (comparison)
Section 5   Plots of the intermediate steps
Section 6   Tables and the demonstration run

Both models are linear in their parameters, so writing them as
y_hat = Phi @ theta lets one gradient and one Hessian serve both (and any
polynomial degree).  numpy is used only for array arithmetic; the linear solve
is a hand-written Gaussian elimination.
"""

import os

import matplotlib
matplotlib.use("Agg")                      # write PNG files, no interactive window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

MEDIA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "media")

# =============================================================================
# Section 1.  Data, model, and the MSE
# =============================================================================
# The assigned data set (order is irrelevant to the fit).
X = np.array([0.0, 2.0, 1.0, 3.0])
Y = np.array([0.5, 3.5, 1.5, 7.5])


def design(x, degree):
    """Design matrix Phi with columns x^degree, ..., x, 1.

    degree 1 -> [x, 1]       and theta = [m, b]      (the line)
    degree 2 -> [x^2, x, 1]  and theta = [a, b, c]   (the parabola)

    With this, the model is simply y_hat = Phi @ theta for both fits.
    """
    return np.vstack([x ** p for p in range(degree, -1, -1)]).T


def mse(theta, x=X, y=Y):
    """Mean squared error of the model with parameters theta."""
    residuals = design(x, len(theta) - 1) @ theta - y
    return float(np.mean(residuals ** 2))


def gradient(theta, x=X, y=Y):
    """d(MSE)/d(theta) = (2/n) Phi^T (Phi theta - y).

    For the line this is exactly the pair
        dMSE/dm = (14m + 6b - 31)/2,   dMSE/db = (6m + 4b - 13)/2
    obtained by hand from the sums of x_i, x_i^2, y_i, x_i y_i.
    """
    P = design(x, len(theta) - 1)
    return 2 / len(x) * P.T @ (P @ theta - y)


def hessian(theta, x=X):
    """d^2(MSE)/d(theta)^2 = (2/n) Phi^T Phi.

    It does not depend on theta, because the MSE is quadratic in the parameters.
    The diagonal entries are the second derivatives used by the one-parameter
    Newton steps; the off-diagonal entries are the cross terms that the
    one-parameter scheme ignores (and the full Newton step uses).
    """
    P = design(x, len(theta) - 1)
    return 2 / len(x) * P.T @ P


# =============================================================================
# Section 2.  Analytical solution: the normal equations
# =============================================================================
def gauss_solve(A, r):
    """Solve A t = r by Gaussian elimination with partial pivoting.

    Written out rather than calling numpy.linalg.solve, since the assignment
    asks for the analytical procedure to be implemented.
    """
    A = np.array(A, float)
    r = np.array(r, float)
    n = len(r)

    # Forward elimination: pivot on the largest entry for numerical stability.
    for i in range(n):
        p = i + np.argmax(abs(A[i:, i]))
        A[[i, p]], r[[i, p]] = A[[p, i]], r[[p, i]]
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            r[j] -= factor * r[i]

    # Back substitution.
    t = np.zeros(n)
    for i in range(n - 1, -1, -1):
        t[i] = (r[i] - A[i, i + 1:] @ t[i + 1:]) / A[i, i]
    return t


def normal_equations(degree, x=X, y=Y):
    """Analytical least-squares fit: solve (Phi^T Phi) theta = Phi^T y.

    Setting every partial derivative of the MSE to zero gives exactly this
    linear system - the 2x2 system for the line and the 3x3 system for the
    parabola that are solved by hand in the write-up.
    """
    P = design(x, degree)
    return gauss_solve(P.T @ P, P.T @ y)


# =============================================================================
# Section 3.  Numerical solution: Newton-Raphson, one parameter per step
# =============================================================================
def coordinate_newton(theta0, x=X, y=Y, tolerance=1e-10, max_iter=500):
    """Optimise one parameter at a time, as the assignment specifies.

    One outer iteration ("sweep") runs, for each parameter j in turn:

        theta_j <- theta_j - (dMSE/dtheta_j) / (d^2 MSE/dtheta_j^2)

    with every other parameter held at its latest value.  Because the MSE is
    quadratic in each parameter separately, this 1-D Newton step is exact: it
    lands on that parameter's minimiser given the others.  For the line the two
    updates reduce to m <- (31 - 6b)/14 and b <- (13 - 6m)/4.

    Convergence is only linear, because the coordinate directions ignore the
    cross term in the Hessian (see full_newton for the contrast).

    Returns (theta, history) where history records theta and the MSE after every
    single sub-step, which is what the intermediate-step plots use.
    """
    theta = np.array(theta0, float)
    history = [dict(iter=0, step="start", theta=theta.copy(), mse=mse(theta, x, y))]

    for it in range(1, max_iter + 1):
        previous = theta.copy()
        for j in range(len(theta)):
            g = gradient(theta, x, y)[j]        # first derivative w.r.t. theta_j
            h = hessian(theta, x)[j, j]         # second derivative w.r.t. theta_j
            theta[j] -= g / h
            history.append(dict(iter=it, step=j, theta=theta.copy(), mse=mse(theta, x, y)))
        if np.max(abs(theta - previous)) < tolerance:
            break

    return theta, history


# =============================================================================
# Section 4.  Numerical solution: the full multivariate Newton step
# =============================================================================
def full_newton(theta0, x=X, y=Y, tolerance=1e-12, max_iter=50):
    """Update all parameters together: theta <- theta - H^{-1} grad.

    For a quadratic MSE the Hessian is constant and this reaches the exact
    minimum in a single step from any starting point.  Included to show what the
    one-parameter-at-a-time scheme gives up.
    """
    theta = np.array(theta0, float)
    history = [dict(iter=0, theta=theta.copy(), mse=mse(theta, x, y))]

    for it in range(1, max_iter + 1):
        step = gauss_solve(hessian(theta, x), gradient(theta, x, y))
        theta = theta - step
        history.append(dict(iter=it, theta=theta.copy(), mse=mse(theta, x, y)))
        if np.max(abs(step)) < tolerance:
            break

    return theta, history


# =============================================================================
# Section 5.  Plots
# =============================================================================
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
SURFACE = "#fcfcfb"
ITER_CMAP = LinearSegmentedColormap.from_list("iter", ["#b9d4f3", "#2a78d6", "#0f3c75"])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 170,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
    "lines.linewidth": 2, "mathtext.fontset": "dejavusans",
})


def save(fig, name):
    os.makedirs(MEDIA, exist_ok=True)
    fig.savefig(os.path.join(MEDIA, name), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("  figure:", name)


def plot_fits(line, parab, name="p2_fits.png"):
    """The two fitted curves with their residuals, and a residual bar chart."""
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.5, 1]))
    xx = np.linspace(-0.3, 3.3, 300)

    ax = axs[0]
    ax.plot(xx, design(xx, 1) @ line, color=BLUE, lw=2.4,
            label=f"line  y = {line[0]:.2f}x {line[1]:+.2f}   (MSE = {mse(line):.4f})")
    ax.plot(xx, design(xx, 2) @ parab, color=ORANGE, lw=2.4,
            label=f"parabola  y = {parab[0]:.2f}x² {parab[1]:+.2f}x {parab[2]:+.2f}"
                  f"   (MSE = {mse(parab):.4f})")
    # residual whiskers, one per data point and model
    for xi, yi in zip(X, Y):
        ax.plot([xi, xi], [yi, design(np.array([xi]), 1)[0] @ line], color=BLUE, lw=1, alpha=0.5)
        ax.plot([xi + 0.03] * 2, [yi, design(np.array([xi]), 2)[0] @ parab],
                color=ORANGE, lw=1, alpha=0.6)
    ax.scatter(X, Y, s=85, color=INK, edgecolor=SURFACE, linewidth=2, zorder=6, label="data")
    for xi, yi in zip(X, Y):
        ax.annotate(f"({xi:g}, {yi:g})", (xi, yi), xytext=(-12, 10),
                    textcoords="offset points", fontsize=9.5, color=INK2, ha="right")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Least-squares fits (thin lines = residuals)")
    ax.legend(loc="upper left", fontsize=9.5)

    ax = axs[1]
    order = np.argsort(X)
    w = 0.36
    ax.bar(np.arange(4) - w / 2 - 0.01, (design(X, 1) @ line - Y)[order], w,
           color=BLUE, label="line")
    ax.bar(np.arange(4) + w / 2 + 0.01, (design(X, 2) @ parab - Y)[order], w,
           color=ORANGE, label="parabola")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(4), [f"x = {v:g}" for v in X[order]])
    ax.set_ylabel("residual  ŷ − y")
    ax.set_title("Residuals")
    ax.legend(fontsize=9.5)

    fig.tight_layout()
    save(fig, name)


def plot_line_path(line, coord_hist, full_hist, name="p2_line_path.png"):
    """Intermediate steps for the line fit.

    (a) contours of MSE(m, b) with the coordinate-wise path (axis-aligned
        zig-zag, magnified in an inset) and the single full Newton step
    (b) the fitted line after each sweep
    """
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # ---- (a) contour plot of the objective ---------------------------------
    ax = axs[0]
    mm, bb = np.meshgrid(np.linspace(-0.3, 3.3, 300), np.linspace(-2.2, 1.6, 300))
    Z = np.mean((mm[..., None] * X + bb[..., None] - Y) ** 2, axis=-1)
    cs = ax.contour(mm, bb, Z, levels=[0.6, 0.75, 1, 1.5, 2.5, 4, 6, 9, 13, 18, 25],
                    cmap=LinearSegmentedColormap.from_list("c", ["#9085e9", "#d6d2f5"]),
                    linewidths=1)
    ax.clabel(cs, fontsize=7.5, fmt="%g")

    P = np.array([e["theta"] for e in coord_hist[:1 + 2 * 15]])
    ax.plot(P[:, 0], P[:, 1], color=ORANGE, lw=1.6, marker="o", ms=4.5, mec=SURFACE,
            label="coordinate Newton (m-step, then b-step)")
    ax.annotate("start (0, 0)", P[0], xytext=(6, 6), textcoords="offset points",
                fontsize=9, color=INK)

    # inset: the steps are tiny next to the full contour range
    ins = ax.inset_axes([0.06, 0.40, 0.36, 0.36])
    ins.contour(mm, bb, Z, levels=[0.576, 0.58, 0.585, 0.6, 0.65, 0.75],
                colors="#b8b0f0", linewidths=0.8)
    ins.plot(P[:, 0], P[:, 1], color=ORANGE, lw=1.3, marker="o", ms=3.5, mec=SURFACE)
    for i, lab in ((1, "m₁"), (2, "b₁"), (3, "m₂"), (4, "b₂")):
        ins.annotate(lab, P[i], xytext=(4, 3), textcoords="offset points",
                     fontsize=8.5, color=INK)
    ins.scatter([line[0]], [line[1]], s=90, marker="*", color=AQUA, edgecolor=INK, zorder=7)
    ins.set_xlim(2.18, 2.33); ins.set_ylim(-0.24, 0.03)
    ins.set_title("zoom: axis-aligned m-step / b-step zig-zag", fontsize=8.5,
                  fontweight="normal")
    ins.tick_params(labelsize=7.5); ins.grid(False)
    ax.indicate_inset_zoom(ins, edgecolor=MUTED)

    F = np.array([e["theta"] for e in full_hist])
    ax.plot(F[:2, 0], F[:2, 1], color=BLUE, lw=1.6, ls="--", label="full Newton (one step)")
    ax.scatter([line[0]], [line[1]], s=110, marker="*", color=AQUA, edgecolor=INK, zorder=7,
               label=f"optimum m = {line[0]:.2f}, b = {line[1]:.2f}")
    ax.set_xlabel("slope m"); ax.set_ylabel("intercept b")
    ax.set_title("(a) MSE(m, b) contours and the optimisation path")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(False)

    # ---- (b) the fitted line after each sweep ------------------------------
    ax = axs[1]
    xx = np.linspace(-0.3, 3.3, 100)
    sweeps = [e for e in coord_hist if e["step"] in ("start", 1)][:7]
    for i, e in enumerate(sweeps):
        ax.plot(xx, design(xx, 1) @ e["theta"], color=ITER_CMAP(i / (len(sweeps) - 1)), lw=1.8,
                label=f"sweep {e['iter']}: y = {e['theta'][0]:.3f}x {e['theta'][1]:+.3f}"
                      f"  (MSE {e['mse']:.3f})")
    ax.scatter(X, Y, s=70, color=INK, edgecolor=SURFACE, linewidth=2, zorder=6)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("(b) Intermediate lines after each sweep")
    ax.legend(fontsize=8.3, loc="upper left")

    fig.tight_layout()
    save(fig, name)


def plot_parabola_and_convergence(line, parab, line_hist, parab_hist,
                                  line_full, parab_full, name="p2_convergence.png"):
    """(a) the parabola after selected sweeps; (b) excess MSE per sweep, log scale."""
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.0))
    xx = np.linspace(-0.3, 3.3, 100)

    ax = axs[0]
    show = [0, 1, 2, 3, 5, 10, 30]
    by_sweep = {e["iter"]: e for e in parab_hist if e["step"] in ("start", 2)}
    for i, s in enumerate(show):
        e = by_sweep[s]
        a_, b_, c_ = e["theta"]
        ax.plot(xx, design(xx, 2) @ e["theta"], color=ITER_CMAP(i / (len(show) - 1)), lw=1.8,
                label=f"sweep {s}: {a_:.3f}x² {b_:+.3f}x {c_:+.3f}  (MSE {e['mse']:.4f})")
    ax.plot(xx, design(xx, 2) @ parab, color=ORANGE, lw=2.4, ls="--", label="analytical optimum")
    ax.scatter(X, Y, s=70, color=INK, edgecolor=SURFACE, linewidth=2, zorder=6)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("(a) Parabola after each coordinate sweep")
    ax.legend(fontsize=8, loc="upper left")

    ax = axs[1]
    # Excess MSE above the optimum; floored so the log axis stays readable once
    # the iteration reaches floating-point noise.
    lm = [e["mse"] - mse(line) for e in line_hist if e["step"] in ("start", 1)]
    pm = [e["mse"] - mse(parab) for e in parab_hist if e["step"] in ("start", 2)]
    ax.semilogy(range(len(lm)), np.maximum(lm, 1e-15), color=BLUE, lw=2,
                label="line (2 parameters)")
    ax.semilogy(range(len(pm)), np.maximum(pm, 1e-15), color=ORANGE, lw=2,
                label="parabola (3 parameters)")
    ax.scatter([1, 1], [max(line_full[1]["mse"] - mse(line), 1e-15),
                        max(parab_full[1]["mse"] - mse(parab), 1e-15)],
               marker="*", s=120, color=AQUA, edgecolor=INK, zorder=6,
               label="full Newton after 1 step")
    ax.set_xlim(-2, 150)
    ax.set_xlabel("sweep (one Newton step per parameter)")
    ax.set_ylabel("MSE − MSE*  (floored at 1e-15)")
    ax.set_title("(b) Convergence of coordinate-wise Newton")
    ax.legend(fontsize=9)

    fig.tight_layout()
    save(fig, name)


# =============================================================================
# Section 6.  Tables and the demonstration run
# =============================================================================
def print_sweep_table(history, names, nsweeps):
    """Print theta and the MSE after every sub-step of the first few sweeps."""
    header = "  ".join(f"{n:>10}" for n in names)
    print(f"  {'sweep':>5} {'updated':>8}  {header}  {'MSE':>10}")
    for e in history[:1 + len(names) * nsweeps]:
        step = "start" if e["step"] == "start" else names[e["step"]]
        vals = "  ".join(f"{v:10.6f}" for v in e["theta"])
        print(f"  {e['iter']:5d} {step:>8}  {vals}  {e['mse']:10.6f}")


def main():
    print("\n" + "=" * 78)
    print("PART 2  ---  fitting a line and a parabola by minimising the MSE")
    print("=" * 78)
    print("  data:", ", ".join(f"({x:g}, {y:g})" for x, y in zip(X, Y)))

    results = {}
    for label, degree, names in (("line", 1, ["m", "b"]),
                                 ("parabola", 2, ["a", "b", "c"])):
        analytic = normal_equations(degree)
        coord, coord_hist = coordinate_newton(np.zeros(degree + 1))
        full, full_hist = full_newton(np.zeros(degree + 1))

        print(f"\n{label.upper()}  ({', '.join(names)})")
        print(f"  analytical (normal equations) : "
              f"{', '.join(f'{n}={v:.6f}' for n, v in zip(names, analytic))}   "
              f"MSE = {mse(analytic):.6f}")
        print(f"  Newton, one parameter at a time: "
              f"{', '.join(f'{n}={v:.6f}' for n, v in zip(names, coord))}   "
              f"MSE = {mse(coord):.6f}   ({coord_hist[-1]['iter']} sweeps)")
        print(f"  Newton, full multivariate step : "
              f"{', '.join(f'{n}={v:.6f}' for n, v in zip(names, full))}   "
              f"MSE = {mse(full):.6f}   (1 step)")

        residuals = design(X, degree) @ analytic - Y
        print("  residuals (y_hat - y):",
              ", ".join(f"x={x:g}: {r:+.4f}" for x, r in zip(X, residuals)))
        print(f"\n  first sweeps of the one-parameter-at-a-time iteration:")
        print_sweep_table(coord_hist, names, 6 if degree == 1 else 4)

        results[label] = dict(analytic=analytic, coord=coord, coord_hist=coord_hist,
                              full=full, full_hist=full_hist)

    line, parab = results["line"], results["parabola"]
    plot_fits(line["analytic"], parab["analytic"])
    plot_line_path(line["analytic"], line["coord_hist"], line["full_hist"])
    plot_parabola_and_convergence(line["analytic"], parab["analytic"],
                                  line["coord_hist"], parab["coord_hist"],
                                  line["full_hist"], parab["full_hist"])

    print(f"\n  ANSWERS:  line     y = {line['analytic'][0]:.2f}x {line['analytic'][1]:+.2f}"
          f"   MSE = {mse(line['analytic']):.4f}")
    print(f"            parabola y = {parab['analytic'][0]:.2f}x² "
          f"{parab['analytic'][1]:+.2f}x {parab['analytic'][2]:+.2f}"
          f"   MSE = {mse(parab['analytic']):.4f}")
    print("\nAll Part 2 figures written to ../media/\n")


if __name__ == "__main__":
    main()
