"""
SYDE 572 - Assignment 1, Part 1
================================
Shortest distance from a point (x0, y0) to a curve y = f(x).

Run this file to reproduce every Part 1 result:

    python distance.py

It prints the result tables and writes all Part 1 figures to ../media/.

Contents
--------
Section 1   The objective function D(x) and its derivatives
Section 2   Analytical solution for a parabola (Cardano's cubic formula)
Section 3   Numerical method A: Newton-Raphson on D'(x) = 0
Section 4   Numerical method B: Golden section search on D(x)
Section 5   Convenience wrappers (any point, any parabola, any function)
Section 6   Plotting helpers
Section 7   Demonstrations: the assigned points, other curves, non-polynomial
            functions, and a case with several stationary points

Only numpy and matplotlib are used, and no optimisation library: the three
solvers, the cubic solver and the finite-difference derivatives are all written
out here.
"""

import math
import os

import matplotlib
matplotlib.use("Agg")                      # write PNG files, no interactive window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# The golden ratio and the fraction of the bracket used by golden section search.
PHI = (1 + math.sqrt(5)) / 2               # 1.618034
RESPHI = 2 - PHI                           # 1/phi^2 = 0.381966

MEDIA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "media")


# =============================================================================
# Section 1.  The objective function
# =============================================================================
def dist_sq(x, x0, y0, f):
    """Squared distance D(x) from the point (x0, y0) to the curve point (x, f(x)).

    The true distance is sqrt(D(x)).  Because the square root is monotonically
    increasing, the x that minimises D also minimises the distance, so every
    solver below works with D and takes the square root only at the very end.
    """
    return (x - x0) ** 2 + (f(x) - y0) ** 2


def _finite(v):
    """True if v is a usable number (not NaN or +-inf)."""
    return isinstance(v, (int, float, np.floating)) and math.isfinite(float(v))


def numeric_d1(f, h=1e-5):
    """First derivative of f by central differences: used when f' is unknown.

    Returned as a function so it can be dropped in wherever an analytic
    derivative would go.  h = 1e-5 keeps the truncation error (order h^2) and
    the floating-point cancellation error (order eps/h) comparably small.

    Near the edge of the domain (e.g. ln x at x = 1e-6) one of the two sample
    points can fall outside it and return NaN, so the routine falls back to a
    one-sided difference there.
    """
    def d1(x):
        fp, fm_ = f(x + h), f(x - h)
        if _finite(fp) and _finite(fm_):
            return (fp - fm_) / (2 * h)
        if _finite(fp):                     # forward difference
            return (fp - f(x)) / h
        return (f(x) - fm_) / h             # backward difference
    return d1


def numeric_d2(f, h=1e-4):
    """Second derivative of f by central differences (larger h: divides by h^2).

    Same domain-edge fallback as numeric_d1: if a sample point is outside the
    domain, the stencil is shifted to one side.
    """
    def d2(x):
        f0, fp, fm_ = f(x), f(x + h), f(x - h)
        if _finite(fp) and _finite(fm_):
            return (fp - 2 * f0 + fm_) / (h * h)
        if _finite(fp):                     # forward stencil
            return (f(x + 2 * h) - 2 * fp + f0) / (h * h)
        return (f0 - 2 * fm_ + f(x - 2 * h)) / (h * h)
    return d2


# =============================================================================
# Section 2.  Analytical solution for a parabola
# =============================================================================
def _cbrt(v):
    """Real cube root, including for negative v (v ** (1/3) fails for v < 0)."""
    return math.copysign(abs(v) ** (1 / 3), v)


def solve_cubic(A, B, C, D):
    """All real roots of A x^3 + B x^2 + C x + D = 0 (A != 0).

    Substituting x = t - B/(3A) removes the quadratic term and leaves the
    depressed cubic t^3 + p t + q = 0, which Cardano's formula solves.  The sign
    of the discriminant decides how many real roots there are:

        disc > 0  ->  one real root   (two cube roots added)
        disc < 0  ->  three real roots (trigonometric form; cube roots of a
                                        complex number would be needed instead)
        disc = 0  ->  a repeated root
    """
    b, c, d = B / A, C / A, D / A
    p = c - b * b / 3
    q = 2 * b ** 3 / 27 - b * c / 3 + d
    shift = -b / 3                                  # undo the substitution later
    disc = (q / 2) ** 2 + (p / 3) ** 3

    if disc > 1e-14:                                # one real root
        s = math.sqrt(disc)
        t = [_cbrt(-q / 2 + s) + _cbrt(-q / 2 - s)]
    elif disc < -1e-14:                             # three distinct real roots
        r = 2 * math.sqrt(-p / 3)
        phi = math.acos(3 * q / (p * r))
        t = [r * math.cos((phi - 2 * math.pi * k) / 3) for k in range(3)]
    else:                                           # repeated roots
        u = _cbrt(-q / 2)
        t = [2 * u, -u]

    return sorted(ti + shift for ti in t)


def analytical_parabola(x0, y0, a, b, c):
    """Exact closest point on the parabola y = a x^2 + b x + c to (x0, y0).

    Setting D'(x) = 0 gives
        (x - x0) + (a x^2 + b x + c - y0)(2a x + b) = 0,
    which expands to the cubic
        2a^2 x^3 + 3ab x^2 + (b^2 + 2a(c - y0) + 1) x + (b(c - y0) - x0) = 0.

    Every real root is a stationary point of D, so the root with the smallest D
    is the answer (D -> infinity as |x| -> infinity, so a minimum always exists).

    If a = 0 the "parabola" is the straight line y = b x + c, for which the
    condition is linear and gives x* = (x0 + b(y0 - c)) / (1 + b^2) directly.

    Returns (distance, x*, all real roots).
    """
    f = lambda x: a * x * x + b * x + c
    k = c - y0

    if a == 0:                                      # degenerate case: a line
        xs = (x0 + b * (y0 - c)) / (1 + b * b)
        return math.sqrt(dist_sq(xs, x0, y0, f)), xs, [xs]

    roots = solve_cubic(2 * a * a, 3 * a * b, b * b + 2 * a * k + 1, b * k - x0)
    best = min(roots, key=lambda x: dist_sq(x, x0, y0, f))
    return math.sqrt(dist_sq(best, x0, y0, f)), best, roots


# =============================================================================
# Section 3.  Numerical method A: Newton-Raphson
# =============================================================================
def newton_distance(x0, y0, f, df=None, ddf=None, initial_guess=0.0,
                    tolerance=1e-7, max_iter=100, domain=None):
    """Newton-Raphson applied to the root-finding problem D'(x) = 0.

        D'(x)  = 2(x - x0) + 2(f(x) - y0) f'(x)
        D''(x) = 2 + 2 f'(x)^2 + 2(f(x) - y0) f''(x)
        x_{k+1} = x_k - D'(x_k) / D''(x_k)

    df, ddf   : derivatives of f.  If omitted, central differences are used, so
                the routine works for any f the caller can evaluate.
    domain    : (lo, hi) to clip each iterate into the domain of f - needed for
                ln x, sqrt(x), 1/x, where an unconstrained step can overshoot
                into x <= 0 and produce NaN.
    tolerance : stop once the step is smaller than this.

    Returns (distance, x*, history), where history holds x_k, D, D', D'' and the
    next iterate for every step, which is what the step plots are drawn from.
    """
    df = df or numeric_d1(f)
    ddf = ddf or numeric_d2(f)

    x = initial_guess
    history = []
    for k in range(max_iter):
        d1 = 2 * (x - x0) + 2 * (f(x) - y0) * df(x)
        d2 = 2 + 2 * df(x) ** 2 + 2 * (f(x) - y0) * ddf(x)

        next_x = x - d1 / d2                        # the Newton step itself
        if domain is not None:
            next_x = min(max(next_x, domain[0]), domain[1])

        history.append(dict(k=k, x=x, D=dist_sq(x, x0, y0, f),
                            d1=d1, d2=d2, next_x=next_x))
        if abs(next_x - x) < tolerance:
            x = next_x
            break
        x = next_x

    return math.sqrt(dist_sq(x, x0, y0, f)), x, history


def newton_multistart(x0, y0, f, df=None, ddf=None, starts=(-5, -2, 0, 2, 5), **kw):
    """Run Newton from several starting guesses and keep the closest result.

    Newton converges to a *stationary* point of D, which may be a local maximum
    or a non-global minimum (see multiple_minima_demo).  Trying several starts
    is the cheapest guard against that.
    """
    runs = [newton_distance(x0, y0, f, df, ddf, initial_guess=s, **kw) for s in starts]
    # A run can diverge to NaN (e.g. a starting guess right on the edge of the
    # domain of f); those are discarded rather than compared, since any
    # comparison against NaN is False and would corrupt the minimum.
    usable = [r for r in runs if math.isfinite(r[0])]
    if not usable:
        raise ValueError("every Newton start diverged; check the starting guesses or the domain")
    return min(usable, key=lambda r: r[0])


# =============================================================================
# Section 4.  Numerical method B: Golden section search
# =============================================================================
def golden_section(x0, y0, f, a, b, tolerance=1e-7, max_iter=500):
    """Golden section search for the minimum of D on the bracket [a, b].

    Two interior probes are placed symmetrically:
        x1 = a + r(b - a),  x2 = b - r(b - a),  r = 2 - phi = 0.381966.
    If D(x1) < D(x2) the minimum of a unimodal D cannot lie in (x2, b], so the
    bracket becomes [a, x2]; otherwise it becomes [x1, b].  The golden ratio is
    chosen so the surviving probe is already in the right place for the new
    bracket, so each iteration costs one new evaluation of D and shrinks the
    bracket by 1/phi = 0.618.

    No derivatives are used anywhere, which is the whole point of this method.

    Returns (distance, x*, history) with the bracket and both probes per step.
    """
    D = lambda x: dist_sq(x, x0, y0, f)

    x1 = a + RESPHI * (b - a)
    x2 = b - RESPHI * (b - a)
    f1, f2 = D(x1), D(x2)

    history = []
    k = 0
    while abs(b - a) > tolerance and k < max_iter:
        history.append(dict(k=k, a=a, b=b, x1=x1, x2=x2, D1=f1, D2=f2))
        if f1 < f2:                     # minimum is in [a, x2]; x1 becomes the new x2
            b, x2, f2 = x2, x1, f1
            x1 = a + RESPHI * (b - a)
            f1 = D(x1)
        else:                           # minimum is in [x1, b]; x2 becomes the new x1
            a, x1, f1 = x1, x2, f2
            x2 = b - RESPHI * (b - a)
            f2 = D(x2)
        k += 1

    history.append(dict(k=k, a=a, b=b, x1=x1, x2=x2, D1=f1, D2=f2))
    best_x = (a + b) / 2                # the midpoint of the final bracket
    return math.sqrt(D(best_x)), best_x, history


# =============================================================================
# Section 5.  Convenience wrappers
# =============================================================================
def parabola(a, b, c):
    """Return f, f', f'' for y = a x^2 + b x + c."""
    return (lambda x: a * x * x + b * x + c,
            lambda x: 2 * a * x + b,
            lambda x: 2 * a)


def distance_to_parabola(x0, y0, a, b, c, bracket=(-10, 10)):
    """Distance from any point to any parabola, by all three methods.

    >>> r = distance_to_parabola(-4, 0, a=1, b=0, c=5)
    >>> round(r['analytical'][0], 6), round(r['newton'][0], 6), round(r['golden'][0], 6)
    (6.289845, 6.289845, 6.289845)
    """
    f, df, ddf = parabola(a, b, c)
    da, xa, _ = analytical_parabola(x0, y0, a, b, c)
    dn, xn, hn = newton_distance(x0, y0, f, df, ddf, initial_guess=0.0)
    dg, xg, hg = golden_section(x0, y0, f, *bracket)
    return dict(analytical=(da, xa),
                newton=(dn, xn, len(hn)),
                golden=(dg, xg, len(hg) - 1))


def distance_to_curve(x0, y0, f, bracket, domain=None, starts=None):
    """Distance from any point to any curve y = f(x), without needing f' or f''.

    This is the "slight modification" asked for in the assignment: the
    derivatives are estimated numerically and the search is confined to the
    domain of f, so exponential, logarithmic, rational and radical functions all
    work with the same two solvers.
    """
    lo, hi = bracket
    if starts is None:
        starts = sorted({min(max(x0, lo + 0.1), hi - 0.1), (lo + hi) / 2,
                         lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)})
    dn, xn, hn = newton_multistart(x0, y0, f, starts=starts, domain=domain)
    dg, xg, hg = golden_section(x0, y0, f, lo, hi)

    # Brute-force reference: the minimum over a fine grid, used only as a check.
    grid = np.linspace(lo, hi, 400001)
    ref_x = float(grid[np.argmin(dist_sq(grid, x0, y0, f))])

    return dict(newton=(dn, xn, len(hn)), newton_history=hn,
                golden=(dg, xg, len(hg) - 1),
                grid=(math.sqrt(dist_sq(ref_x, x0, y0, f)), ref_x))


# =============================================================================
# Section 6.  Plotting helpers
# =============================================================================
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948"
SURFACE = "#fcfcfb"
# Light -> dark blue ramp: early iterates are pale, the converged one is dark.
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
    """Save a figure into ../media/ and report it."""
    os.makedirs(MEDIA, exist_ok=True)
    fig.savefig(os.path.join(MEDIA, name), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("  figure:", name)


def draw_shortest_segment(ax, x0, y0, xs, f):
    """Dashed segment from the query point to the closest curve point."""
    ax.plot([x0, xs], [y0, f(xs)], color=ORANGE, lw=2, ls=(0, (4, 2)), zorder=4)
    ax.scatter([xs], [f(xs)], s=70, color=AQUA, edgecolor=SURFACE, linewidth=2, zorder=6)


def draw_query_point(ax, x0, y0, text=None, dx=0.3, dy=-0.9):
    ax.scatter([x0], [y0], s=80, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=6)
    if text:
        ax.annotate(text, (x0, y0), xytext=(x0 + dx, y0 + dy), color=INK, fontsize=10)


def plot_overview(points, f, results, name="p1_overview.png"):
    """All five query points, their closest curve points and the distances."""
    fig, ax = plt.subplots(figsize=(9, 5.6))
    xs = np.linspace(-3.6, 3.6, 400)
    ax.plot(xs, f(xs), color=BLUE, lw=2.5, label=r"$y = x^2 + 5$", zorder=3)

    for (x0, y0), r in zip(points, results):
        xa = r["analytical"][1]
        ax.plot([x0, xa], [y0, f(xa)], color=ORANGE, lw=1.8, ls=(0, (4, 2)), zorder=4)
        ax.scatter([xa], [f(xa)], s=60, color=AQUA, edgecolor=SURFACE, linewidth=2, zorder=6)
        ax.scatter([x0], [y0], s=80, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=6)
        ax.annotate(f"({x0}, {y0})\nd = {r['analytical'][0]:.4f}", (x0, y0),
                    xytext=(0, -62 if x0 == 2 else -34), textcoords="offset points",
                    ha="center", fontsize=9.5, color=INK)

    # legend proxies
    ax.plot([], [], color=ORANGE, lw=1.8, ls=(0, (4, 2)), label="shortest segment")
    ax.scatter([], [], s=60, color=AQUA, label="closest point on curve")
    ax.scatter([], [], s=80, color=ORANGE, label="query point")

    ax.set_xlim(-9.5, 7.5); ax.set_ylim(-4.6, 13)
    ax.set_aspect("equal")                  # so the right angles look like right angles
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Shortest distance from five points to y = x² + 5")
    ax.legend(loc="upper left", fontsize=9.5)
    save(fig, name)


def plot_newton_steps(x0, y0, f, df, ddf, history, x_exact, x_final, d_final):
    """Three panels of Newton's intermediate steps for one query point.

    (a) the iterates projected onto the curve
    (b) the tangent construction on D'(x): each tangent is followed to its zero
    (c) |x_k - x*| on a log scale, showing the quadratic collapse
    """
    iterates = [h["x"] for h in history] + [x_final]
    n = len(iterates)
    cols = [ITER_CMAP(i / max(n - 1, 1)) for i in range(n)]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6),
                            gridspec_kw=dict(width_ratios=[1.15, 1.15, 0.9]))

    # ---- (a) geometry -----------------------------------------------------
    ax = axs[0]
    lo, hi = min(x0, -2.5) - 1, max(x0, 2.5) + 1
    xx = np.linspace(lo, hi, 400)
    ax.plot(xx, f(xx), color=BLUE, lw=2.5, label="y = x² + 5")
    for i, xi in enumerate(iterates):
        ax.plot([x0, xi], [y0, f(xi)], color=cols[i], lw=1.2, alpha=0.9)
        ax.scatter([xi], [f(xi)], s=36, color=cols[i], edgecolor=SURFACE, linewidth=1, zorder=5)
    draw_shortest_segment(ax, x0, y0, x_exact, f)
    draw_query_point(ax, x0, y0, f"({x0}, {y0})", dx=0.2, dy=-1.3)
    ax.set_xlim(lo, hi); ax.set_ylim(-1.5, 10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("(a) Newton iterates on the curve")
    ax.text(0.02, 0.97, f"d* = {d_final:.6f}\nx* = {x_final:+.6f}", transform=ax.transAxes,
            va="top", fontsize=9.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID))

    # ---- (b) the tangent steps on D'(x) -----------------------------------
    ax = axs[1]
    Dp = lambda x: 2 * (x - x0) + 2 * (f(x) - y0) * df(x)
    xl, xh = min(iterates), max(iterates)
    pad = max(xh - xl, 0.3) * 0.35
    xx = np.linspace(xl - pad, xh + pad, 400)
    ax.axhline(0, color=MUTED, lw=1)
    ax.plot(xx, Dp(xx), color=VIOLET, lw=2.3, label="D′(x) = 4x³ + 22x − 2x₀")
    for i, h in enumerate(history):
        ax.plot([h["x"], h["x"]], [0, h["d1"]], color=cols[i], lw=1, ls=":")
        ax.plot([h["x"], h["next_x"]], [h["d1"], 0], color=cols[i], lw=1.6)
        ax.scatter([h["x"]], [h["d1"]], s=34, color=cols[i], zorder=5, edgecolor=SURFACE)
        if i < 2:
            ax.annotate(f"x{i}", (h["x"], 0), xytext=(0, 7 if h["d1"] < 0 else -14),
                        textcoords="offset points", ha="center", fontsize=9, color=INK2)
    ax.scatter([x_exact], [0], s=70, color=AQUA, edgecolor=SURFACE, linewidth=2, zorder=6)
    ax.set_xlabel("x"); ax.set_ylabel("D′(x)")
    ax.set_title("(b) Tangent steps on D′(x) = 0")
    ax.legend(loc="upper left", fontsize=9)

    # ---- (c) convergence --------------------------------------------------
    ax = axs[2]
    err = [max(abs(xi - x_exact), 1e-17) for xi in iterates]    # clip for the log axis
    ax.semilogy(range(n), err, color=BLUE, marker="o", ms=6, mec=SURFACE)
    ax.set_xlabel("iteration k"); ax.set_ylabel("|x_k − x*|")
    ax.set_title("(c) Quadratic convergence")
    ax.set_xticks(range(n))

    fig.suptitle(f"Newton–Raphson  ·  point ({x0}, {y0})  ·  initial guess x = 0",
                 x=0.01, ha="left", fontsize=13.5, fontweight="bold", color=INK, y=1.03)
    fig.tight_layout()
    save(fig, f"p1_newton_{x0}_{y0}.png")


def plot_golden_steps(x0, y0, f, history, x_exact, bracket):
    """Three panels of the golden section intermediate steps for one point.

    (a) the probe pairs on D(x)
    (b) the bracket ladder: [a, b] for the first iterations
    (c) bracket width and midpoint error, both decaying by 0.618 per step
    """
    D = lambda x: dist_sq(x, x0, y0, f)
    nshow = 12
    cols = [ITER_CMAP(i / (nshow - 1)) for i in range(nshow)]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6),
                            gridspec_kw=dict(width_ratios=[1.2, 1.1, 0.9]))

    # ---- (a) D(x) with the first probe pairs -------------------------------
    ax = axs[0]
    xx = np.linspace(*bracket, 600)
    ax.plot(xx, D(xx), color=VIOLET, lw=2.3, label="D(x) = (x − x₀)² + (x² + 5 − y₀)²")
    ax.set_yscale("log")
    for i, h in enumerate(history[:6]):
        ax.scatter([h["x1"], h["x2"]], [h["D1"], h["D2"]], s=40, color=cols[i * 2],
                   zorder=5, edgecolor=SURFACE)
    ax.scatter([x_exact], [D(x_exact)], s=80, color=AQUA, edgecolor=SURFACE, linewidth=2,
               zorder=6, label=f"minimum  x* = {x_exact:+.4f}")
    ax.set_xlabel("x"); ax.set_ylabel("D(x)  (log scale)")
    ax.set_title("(a) Probe points x₁, x₂ on D(x)")
    ax.legend(loc="upper center", fontsize=8.8)

    # ---- (b) the shrinking bracket ----------------------------------------
    ax = axs[1]
    for i, h in enumerate(history[:nshow]):
        ax.plot([h["a"], h["b"]], [i, i], color=cols[i], lw=5, solid_capstyle="round")
        ax.scatter([h["x1"], h["x2"]], [i, i], s=18, color="white", edgecolor=INK2,
                   zorder=5, linewidth=1)
    ax.axvline(x_exact, color=AQUA, lw=1.6, ls="--")
    ax.invert_yaxis()
    ax.set_yticks(range(nshow))
    ax.set_xlabel("x"); ax.set_ylabel("iteration k")
    ax.set_title("(b) Bracket [a, b] shrinks by 1/φ")
    ax.grid(axis="y", visible=False)

    # ---- (c) width and error ----------------------------------------------
    ax = axs[2]
    k = [h["k"] for h in history]
    width = [h["b"] - h["a"] for h in history]
    err = [max(abs((h["a"] + h["b"]) / 2 - x_exact), 1e-17) for h in history]
    ax.semilogy(k, width, color=BLUE, lw=2, label="bracket width b − a")
    ax.semilogy(k, err, color=ORANGE, lw=2, label="|midpoint − x*|")
    ax.axhline(1e-7, color=MUTED, lw=1, ls=":")
    ax.text(k[-1] * 0.55, 1.6e-7, "tolerance 1e-7", ha="right", fontsize=8.5, color=INK2)
    ax.set_xlabel("iteration k")
    ax.set_title("(c) Linear convergence (ratio 0.618)")
    ax.legend(fontsize=8.8, loc="lower left")

    fig.suptitle(f"Golden Section Search  ·  point ({x0}, {y0})  ·  bracket [−10, 10]",
                 x=0.01, ha="left", fontsize=13.5, fontweight="bold", color=INK, y=1.03)
    fig.tight_layout()
    save(fig, f"p1_golden_{x0}_{y0}.png")


def plot_convergence_comparison(points, results, name="p1_convergence.png"):
    """Error per iteration for both methods and all five points, side by side."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for j, ((x0, y0), r) in enumerate(zip(points, results)):
        x_exact = r["analytical"][1]
        col = [BLUE, ORANGE, AQUA, VIOLET, RED][j]
        label = f"({x0}, {y0})"

        xs_newton = [h["x"] for h in r["newton_history"]] + [r["newton"][1]]
        axs[0].semilogy(range(len(xs_newton)),
                        [max(abs(v - x_exact), 1e-16) for v in xs_newton],
                        color=col, marker="o", ms=5, mec=SURFACE, label=label)

        # (0,0) is skipped on the right: its bracket is symmetric about x* = 0,
        # so the midpoint hits the exact answer every other step and the error
        # drops to 0, which cannot be drawn on a log axis.
        if (x0, y0) == (0, 0):
            continue
        hg = r["golden_history"]
        axs[1].semilogy([h["k"] for h in hg],
                        [max(abs((h["a"] + h["b"]) / 2 - x_exact), 1e-12) for h in hg],
                        color=col, lw=1.8, label=label)

    axs[0].set_title("Newton–Raphson: error per iteration")
    axs[1].set_title("Golden section: error per iteration")
    for ax in axs:
        ax.set_xlabel("iteration k")
    axs[0].set_ylabel("|x_k − x*|")
    axs[0].legend(title="point", fontsize=9, title_fontsize=9)
    axs[1].text(0.98, 0.97, "(0, 0) omitted: its midpoint equals x* = 0\n"
                            "exactly on every other step (error = 0)",
                transform=axs[1].transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
    fig.tight_layout()
    save(fig, name)


# =============================================================================
# Section 7.  Demonstrations
# =============================================================================
ASSIGNED_POINTS = [(0, 0), (-4, 0), (-8, 0), (2, 0), (6, 0)]
BRACKET = (-10.0, 10.0)          # starting bracket for golden section search


def assigned_points_demo():
    """Part 1 as assigned: the five points and the parabola y = x^2 + 5.

    Runs all three methods, prints the comparison table and draws the overview,
    the per-point intermediate-step figures and the convergence comparison.
    """
    a, b, c = 1.0, 0.0, 5.0
    f, df, ddf = parabola(a, b, c)

    print("\n" + "=" * 78)
    print("PART 1  ---  y = x^2 + 5")
    print("=" * 78)
    print(f"{'point':>9} | {'x* (exact)':>12} {'d (exact)':>11} | "
          f"{'d (Newton)':>11} {'it':>3} | {'d (golden)':>11} {'it':>3}")
    print("-" * 78)

    results = []
    for (x0, y0) in ASSIGNED_POINTS:
        da, xa, roots = analytical_parabola(x0, y0, a, b, c)
        dn, xn, hn = newton_distance(x0, y0, f, df, ddf, initial_guess=0.0)
        dg, xg, hg = golden_section(x0, y0, f, *BRACKET)
        results.append(dict(point=(x0, y0), analytical=(da, xa), roots=roots,
                            newton=(dn, xn), newton_history=hn,
                            golden=(dg, xg), golden_history=hg))
        print(f"{str((x0, y0)):>9} | {xa:12.6f} {da:11.6f} | "
              f"{dn:11.6f} {len(hn):3d} | {dg:11.6f} {len(hg) - 1:3d}")

    # Per-point iterate tables, i.e. the intermediate steps in numbers.
    print("\nNewton iterates (initial guess x = 0):")
    for r in results:
        xs = [h["x"] for h in r["newton_history"]] + [r["newton"][1]]
        print(f"  {str(r['point']):>9}: " + "  ".join(f"{v:+.7f}" for v in xs))

    print("\nGolden section, first four brackets:")
    for r in results:
        print(f"  {str(r['point']):>9}: " +
              "  ".join(f"[{h['a']:+.4f}, {h['b']:+.4f}]" for h in r["golden_history"][:4]))

    # Figures.
    plot_overview(ASSIGNED_POINTS, f, results)
    for r in results:
        x0, y0 = r["point"]
        plot_newton_steps(x0, y0, f, df, ddf, r["newton_history"],
                          r["analytical"][1], r["newton"][1], r["newton"][0])
        plot_golden_steps(x0, y0, f, r["golden_history"], r["analytical"][1], BRACKET)
    plot_convergence_comparison(ASSIGNED_POINTS, results)
    return results


# Other curves: two more parabolas plus the non-polynomial cases required by the
# assignment (exponential, logarithmic, variable in the denominator, radical).
OTHER_CURVES = [
    dict(name="y = 0.5x² − 2x + 1", kind="parabola",
         f=lambda x: 0.5 * x ** 2 - 2 * x + 1, abc=(0.5, -2, 1), pt=(5, -2),
         bracket=(-10, 10), domain=None, xr=(-1, 7)),
    dict(name="y = −x² + 4x − 1", kind="parabola",
         f=lambda x: -x ** 2 + 4 * x - 1, abc=(-1, 4, -1), pt=(-1, 4),
         bracket=(-10, 10), domain=None, xr=(-2.5, 4.5)),
    dict(name="y = eˣ", kind="exponential",
         f=np.exp, pt=(2, 1), bracket=(-5, 3), domain=None, xr=(-2.5, 3)),
    dict(name="y = ln x", kind="logarithmic",
         f=np.log, pt=(0, 1), bracket=(1e-3, 6), domain=(1e-6, 1e6), xr=(0.02, 5)),
    dict(name="y = √x", kind="radical",
         f=np.sqrt, pt=(4, 0), bracket=(0, 10), domain=(1e-9, 1e6), xr=(0, 6)),
    dict(name="y = 1/x", kind="rational (x in denominator)",
         f=lambda x: 1 / x, pt=(0, 0), bracket=(0.05, 8), domain=(1e-3, 1e6), xr=(0.15, 4)),
    dict(name="y = 4/(1 + x²)", kind="rational (x in denominator)",
         f=lambda x: 4 / (1 + x ** 2), pt=(3, 3), bracket=(-2, 8), domain=None, xr=(-3, 5)),
    dict(name="y = x^(1/3)", kind="cube root",
         f=np.cbrt, pt=(-1, 2), bracket=(-6, 6), domain=None, xr=(-4, 3)),
]


def other_curves_demo():
    """The same two solvers on other parabolas and on non-polynomial curves.

    Each answer is checked against a brute-force grid search, and the parabolas
    are additionally checked against the exact cubic.  Two 2x2 figures are drawn
    (upright, so the labels stay readable).
    """
    print("\n" + "=" * 78)
    print("PART 1  ---  other curves and other points")
    print("=" * 78)
    print(f"{'function':>18} {'point':>9} | {'x* (Newton)':>12} {'d (Newton)':>11} {'it':>3} | "
          f"{'d (golden)':>11} | {'d (grid)':>10} | {'d (exact)':>10}")
    print("-" * 100)

    figA, axsA = plt.subplots(2, 2, figsize=(10, 9))
    figB, axsB = plt.subplots(2, 2, figsize=(10, 9))
    axes = list(axsA.ravel()) + list(axsB.ravel())

    rows = []
    for g, ax in zip(OTHER_CURVES, axes):
        f = g["f"]
        x0, y0 = g["pt"]
        r = distance_to_curve(x0, y0, f, g["bracket"], domain=g["domain"])
        dn, xn, it = r["newton"]

        exact = "---"
        if g["kind"] == "parabola":
            da, _, _ = analytical_parabola(x0, y0, *g["abc"])
            exact = f"{da:10.6f}"
        print(f"{g['name']:>18} {str(g['pt']):>9} | {xn:12.6f} {dn:11.6f} {it:3d} | "
              f"{r['golden'][0]:11.6f} | {r['grid'][0]:10.6f} | {exact:>10}")
        rows.append(dict(name=g["name"], kind=g["kind"], point=[x0, y0], **r))

        # ---- one panel per curve: the curve, the Newton iterates, the segment
        xx = np.linspace(*g["xr"], 500)
        ax.plot(xx, f(xx), color=BLUE, lw=2.4)
        iterates = [h["x"] for h in r["newton_history"]] + [xn]
        for i, xi in enumerate(iterates):
            ax.scatter([xi], [f(xi)], s=26, color=ITER_CMAP(i / max(len(iterates) - 1, 1)),
                       zorder=5, edgecolor=SURFACE, linewidth=0.8)
        draw_shortest_segment(ax, x0, y0, xn, f)
        draw_query_point(ax, x0, y0)

        xa_, xb_ = min(g["xr"][0], x0 - 0.5), max(g["xr"][1], x0 + 0.5)
        yv = np.concatenate([f(xx), [y0]])
        ax.set_xlim(xa_, xb_); ax.set_ylim(yv.min() - 0.6, yv.max() + 0.6)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(g["name"], fontsize=12.5)
        right = g["name"] in ("y = ln x", "y = −x² + 4x − 1")
        ax.text(0.97 if right else 0.03, 0.04,
                f"point ({x0}, {y0})\nd = {dn:.5f}\nx* = {xn:.5f}",
                ha="right" if right else "left", transform=ax.transAxes, fontsize=9.3,
                color=INK, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID))

    for fg, nm, sub in ((figA, "p1_gallery_a.png", "other parabolas, exponential, logarithm"),
                        (figB, "p1_gallery_b.png", "radical, rational and cube-root functions")):
        fg.suptitle(f"Same code, {sub} — Newton iterates (light → dark) and the shortest segment",
                    x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
        fg.tight_layout()
        save(fg, nm)
    return rows


def multiple_minima_demo():
    """A case where D has several stationary points: the point (0, 8).

    D'(x) = 2x(2x^2 - 5) has roots 0 and +-sqrt(2.5).  x = 0 is a local *maximum*
    of D, so Newton started at 0 stops on the wrong answer, while the analytical
    solver enumerates all three roots and keeps the smallest D.  This is why
    newton_multistart exists.
    """
    x0, y0 = 0, 8
    a, b, c = 1.0, 0.0, 5.0
    f, df, ddf = parabola(a, b, c)
    D = lambda x: dist_sq(x, x0, y0, f)

    da, xa, roots = analytical_parabola(x0, y0, a, b, c)
    starts = [-2.5, -0.4, 0.0, 0.4, 2.5]
    runs = [newton_distance(x0, y0, f, df, ddf, initial_guess=s) for s in starts]
    dg, xg, _ = golden_section(x0, y0, f, *BRACKET)

    print("\n" + "=" * 78)
    print("PART 1  ---  several stationary points: the point (0, 8)")
    print("=" * 78)
    print("  stationary points:", ", ".join(f"{r:+.6f}" for r in roots))
    print(f"  analytical: d = {da:.6f} at x* = {xa:+.6f}")
    for s, (d, x, _) in zip(starts, runs):
        flag = "  <-- local maximum, wrong answer" if abs(x) < 1e-9 else ""
        print(f"  Newton from x = {s:+.1f}: d = {d:.6f} at x* = {x:+.6f}{flag}")
    print(f"  golden section on [-10, 10]: d = {dg:.6f} at x* = {xg:+.6f}")

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axs[0]
    xx = np.linspace(-3.2, 3.2, 400)
    ax.plot(xx, f(xx), color=BLUE, lw=2.5, label="y = x² + 5")
    for rt in roots:
        is_min = abs(D(rt) - da ** 2) < 1e-9
        ax.plot([x0, rt], [y0, f(rt)], color=ORANGE if is_min else MUTED,
                lw=1.8, ls=(0, (4, 2)))
        ax.scatter([rt], [f(rt)], s=60, color=AQUA if is_min else MUTED,
                   edgecolor=SURFACE, linewidth=2, zorder=6)
    draw_query_point(ax, x0, y0, "(0, 8)", dx=0.2, dy=0.5)
    ax.annotate("x = 0 is a critical point too\n(local MAX of D, d = 3)", (0, 5),
                xytext=(0.5, 2.3), fontsize=9, color=INK2,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_aspect("equal"); ax.set_ylim(1.5, 11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("(a) Two equally close points")

    ax = axs[1]
    xx = np.linspace(-2.6, 2.6, 400)
    ax.plot(xx, D(xx), color=VIOLET, lw=2.3, label="D(x) = x² + (x² − 3)²")
    for s, (_, x, _) in zip(starts, runs):
        ax.annotate("", xy=(x, D(x)), xytext=(s, D(s)),
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4,
                                    connectionstyle="arc3,rad=-0.3"))
        ax.scatter([s], [D(s)], s=30, color=ORANGE, zorder=5)
    for rt in roots:
        ax.scatter([rt], [D(rt)], s=70, color=AQUA if D(rt) < 8 else RED,
                   edgecolor=SURFACE, linewidth=2, zorder=6)
    ax.set_xlabel("x"); ax.set_ylabel("D(x)")
    ax.set_title("(b) Newton from 5 starting guesses")
    ax.text(0, 10.2, "x₀ = 0 → stuck at\nthe local max", ha="center", fontsize=9, color=INK2)
    ax.legend(loc="upper center", fontsize=9, bbox_to_anchor=(0.5, 0.82))
    fig.tight_layout()
    save(fig, "p1_multimin.png")

    return dict(point=[x0, y0], roots=roots, d=da, x=xa,
                newton=[dict(start=s, d=r[0], x=r[1]) for s, r in zip(starts, runs)],
                golden=dict(d=dg, x=xg))


def main():
    assigned_points_demo()
    other_curves_demo()
    multiple_minima_demo()
    print("\nAll Part 1 figures written to ../media/\n")


if __name__ == "__main__":
    main()
