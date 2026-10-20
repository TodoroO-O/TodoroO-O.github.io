"""SYDE 572 - Assignment 2: linear basis function regression on the Boston Housing data.

Implements the five functions asked for in the assignment:

    polynomial_regression          (a, b)  polynomial basis, degrees 1..7, all 13 features
    polynomial_regression_1d_vis   (c)     polynomial basis on the 2nd feature only, with plots
    polynomial_regression_reg      (d)     L2-regularised degree-8 fit, 10-fold CV over lambda
    gaussian_regression            (e)     Gaussian basis, K = 5, 15, ..., 95 centres
    gaussian_regression_reg        (f)     L2-regularised 90-centre Gaussian fit, 10-fold CV

Conventions used everywhere:
  * Inputs are standardised with the mean / std of the 100 training points (z-scores).
    A feature that is constant on the training set (CHAS) keeps std = 1, so it becomes 0.
  * The target (MEDV, in $1000s) is not normalised, so every RMS error is in $1000s.
  * The design matrix always has a bias column phi_0 = 1, and the bias is not penalised.
  * lambda = 0 uses the pseudo-inverse (minimum-norm least squares), so rank-deficient
    design matrices are handled.
"""

import numpy as np

FEATURES = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
            "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
N_TRAIN = 100
LAMBDAS = [0, 0.01, 0.1, 1, 10, 100, 1000]


# ----------------------------------------------------------------------------- data
def load_data(path="housing.data"):
    """Return X (506 x 13 raw features) and t (506 targets, MEDV in $1000s)."""
    data = np.loadtxt(path)
    return data[:, :13], data[:, 13]


def normalize(X_ref, *others):
    """Standardise X_ref with its own mean/std and apply the same map to the other arrays."""
    mu = X_ref.mean(axis=0)
    sd = X_ref.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    out = [(X_ref - mu) / sd] + [(X - mu) / sd for X in others]
    return out, mu, sd


def split(X, t, n_train=N_TRAIN):
    return X[:n_train], t[:n_train], X[n_train:], t[n_train:]


# ----------------------------------------------------------------------------- bases
def design_polynomial(X, degree):
    """[1, x_1..x_D, x_1^2..x_D^2, ..., x_1^M..x_D^M]  (monomials, no cross terms)."""
    X = np.atleast_2d(X.T).T
    cols = [np.ones((X.shape[0], 1))] + [X ** p for p in range(1, degree + 1)]
    return np.hstack(cols)


def design_gaussian(X, centres, s):
    """[1, exp(-||x - mu_j||^2 / (2 s^2)) for j = 1..K]."""
    sq = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    return np.hstack([np.ones((X.shape[0], 1)), np.exp(-sq / (2.0 * s ** 2))])


# ----------------------------------------------------------------------------- fitting
def fit(Phi, t, lam=0.0):
    """Least squares  w = argmin ||t - Phi w||^2 + lam ||w_{1:}||^2  (bias not penalised)."""
    if lam == 0:
        return np.linalg.pinv(Phi) @ t
    I = np.eye(Phi.shape[1])
    I[0, 0] = 0.0
    return np.linalg.solve(Phi.T @ Phi + lam * I, Phi.T @ t)


def median_width(Xn):
    """Kernel-width heuristic: median Euclidean distance between pairs of (normalised) points."""
    d = np.sqrt(((Xn[:, None, :] - Xn[None, :, :]) ** 2).sum(axis=2))
    return float(np.median(d[np.triu_indices(len(Xn), 1)]))


def rms(Phi, w, t):
    return float(np.sqrt(np.mean((Phi @ w - t) ** 2)))


def kfold_indices(n, k=10, seed=0):
    """Shuffle 0..n-1 once and cut it into k folds of (almost) equal size."""
    perm = np.random.default_rng(seed).permutation(n)
    return np.array_split(perm, k)


# ----------------------------------------------------------------------------- (a), (b)
def polynomial_regression(X, t, degrees=range(1, 8), n_train=N_TRAIN):
    """Unregularised polynomial regression on all features.

    Returns a dict with the degrees, the train / test RMS for each degree and the
    weight vectors (index 0 is the bias, index j is feature j for degree 1).
    """
    Xtr, ttr, Xte, tte = split(X, t, n_train)
    (Xtr, Xte), mu, sd = normalize(Xtr, Xte)
    out = {"degree": [], "train": [], "test": [], "weights": [], "rank": [], "n_params": []}
    for M in degrees:
        Ptr, Pte = design_polynomial(Xtr, M), design_polynomial(Xte, M)
        w = fit(Ptr, ttr)
        out["degree"].append(M)
        out["train"].append(rms(Ptr, w, ttr))
        out["test"].append(rms(Pte, w, tte))
        out["weights"].append(w)
        out["rank"].append(int(np.linalg.matrix_rank(Ptr)))
        out["n_params"].append(Ptr.shape[1])
    out["mu"], out["sd"] = mu, sd
    return out


# ----------------------------------------------------------------------------- (c)
def polynomial_regression_1d_vis(X, t, feature=1, degrees=range(1, 8), n_train=N_TRAIN):
    """Polynomial regression on a single feature (default: the 2nd feature, ZN).

    Returns the normalised train/test inputs, a dense grid, and the fitted curve for every
    degree so the caller can plot training points, test points and the learned polynomial.
    """
    Xtr, ttr, Xte, tte = split(X[:, [feature]], t, n_train)
    (xtr, xte), mu, sd = normalize(Xtr, Xte)
    xtr, xte = xtr[:, 0], xte[:, 0]
    lo, hi = min(xtr.min(), xte.min()), max(xtr.max(), xte.max())
    grid = np.linspace(lo, hi, 600)
    fits = {}
    for M in degrees:
        w = fit(design_polynomial(xtr, M), ttr)
        fits[M] = {"w": w,
                   "curve": design_polynomial(grid, M) @ w,
                   "train": rms(design_polynomial(xtr, M), w, ttr),
                   "test": rms(design_polynomial(xte, M), w, tte)}
    return {"xtr": xtr, "ttr": ttr, "xte": xte, "tte": tte, "grid": grid,
            "fits": fits, "mu": float(mu[0]), "sd": float(sd[0]), "feature": feature}


# ----------------------------------------------------------------------------- (d)
def polynomial_regression_reg(X, t, feature=1, degree=8, lambdas=LAMBDAS,
                              k=10, n_train=N_TRAIN, seed=0):
    """L2-regularised degree-`degree` regression on one feature, 10-fold CV over lambda.

    The first n_train points are the only data used for selection.  Inputs are normalised
    with the statistics of all n_train points (a fixed, label-free preprocessing step).
    Returns mean / per-fold validation RMS for each lambda, the chosen lambda, and the
    test RMS of the model refit on all n_train points with that lambda.
    """
    Xtr, ttr, Xte, tte = split(X[:, [feature]], t, n_train)
    (xtr, xte), _, _ = normalize(Xtr, Xte)
    Ptr, Pte = design_polynomial(xtr[:, 0], degree), design_polynomial(xte[:, 0], degree)
    folds = kfold_indices(len(ttr), k, seed)
    val = np.zeros((len(lambdas), k))
    trn = np.zeros((len(lambdas), k))
    for i, lam in enumerate(lambdas):
        for f, vidx in enumerate(folds):
            tidx = np.setdiff1d(np.arange(len(ttr)), vidx)
            w = fit(Ptr[tidx], ttr[tidx], lam)
            val[i, f] = rms(Ptr[vidx], w, ttr[vidx])
            trn[i, f] = rms(Ptr[tidx], w, ttr[tidx])
    const = np.mean([np.sqrt(np.mean((ttr[v] - np.delete(ttr, v).mean()) ** 2)) for v in folds])
    mean_val = val.mean(axis=1)
    best = int(np.argmin(mean_val))
    test = [rms(Pte, fit(Ptr, ttr, lam), tte) for lam in lambdas]
    return {"lambdas": list(lambdas), "val": val, "mean_val": mean_val,
            "mean_train": trn.mean(axis=1), "best_index": best,
            "best_lambda": lambdas[best], "test": test, "const_val": float(const)}


# ----------------------------------------------------------------------------- (e)
def gaussian_regression(X, t, Ks=range(5, 100, 10), s=None, n_train=N_TRAIN, seed=0):
    """Unregularised Gaussian basis function regression on all features.

    One random permutation of the training points is drawn; the first K points of it are
    the centres for K basis functions (so the centre sets are nested as K grows).
    s = None uses the median pairwise distance of the normalised training inputs.
    """
    Xtr, ttr, Xte, tte = split(X, t, n_train)
    (Xtr, Xte), _, _ = normalize(Xtr, Xte)
    s = median_width(Xtr) if s is None else s
    perm = np.random.default_rng(seed).permutation(len(ttr))
    out = {"K": [], "train": [], "test": []}
    for K in Ks:
        C = Xtr[perm[:K]]
        Ptr, Pte = design_gaussian(Xtr, C, s), design_gaussian(Xte, C, s)
        w = fit(Ptr, ttr)
        out["K"].append(K)
        out["train"].append(rms(Ptr, w, ttr))
        out["test"].append(rms(Pte, w, tte))
    out["s"] = s
    return out


# ----------------------------------------------------------------------------- (f)
def gaussian_regression_reg(X, t, K=90, s=None, lambdas=LAMBDAS, k=10,
                            n_train=N_TRAIN, seed=0):
    """L2-regularised Gaussian regression (all features, K centres), 10-fold CV over lambda.

    Within each fold the K centres are drawn at random from that fold's training points only,
    so the held-out fold never contributes a centre.  With 100 points and 10 folds each
    training fold has exactly 90 points, so for K = 90 every training point is a centre.
    The same fold split and centres are used for every lambda.
    """
    Xtr, ttr, Xte, tte = split(X, t, n_train)
    (Xtr, Xte), _, _ = normalize(Xtr, Xte)
    s = median_width(Xtr) if s is None else s
    rng = np.random.default_rng(seed)
    folds = kfold_indices(len(ttr), k, seed)
    val = np.zeros((len(lambdas), k))
    trn = np.zeros((len(lambdas), k))
    for f, vidx in enumerate(folds):
        tidx = np.setdiff1d(np.arange(len(ttr)), vidx)
        C = Xtr[rng.permutation(tidx)[:K]]
        Pt, Pv = design_gaussian(Xtr[tidx], C, s), design_gaussian(Xtr[vidx], C, s)
        for i, lam in enumerate(lambdas):
            w = fit(Pt, ttr[tidx], lam)
            val[i, f] = rms(Pv, w, ttr[vidx])
            trn[i, f] = rms(Pt, w, ttr[tidx])
    const = np.mean([np.sqrt(np.mean((ttr[v] - np.delete(ttr, v).mean()) ** 2)) for v in folds])
    mean_val = val.mean(axis=1)
    best = int(np.argmin(mean_val))
    # refit on all 100 training points with K centres drawn from them, report test RMS
    C = Xtr[rng.permutation(len(ttr))[:K]]
    Ptr, Pte = design_gaussian(Xtr, C, s), design_gaussian(Xte, C, s)
    test = [rms(Pte, fit(Ptr, ttr, lam), tte) for lam in lambdas]
    return {"lambdas": list(lambdas), "val": val, "mean_val": mean_val,
            "mean_train": trn.mean(axis=1), "best_index": best,
            "best_lambda": lambdas[best], "test": test, "const_val": float(const), "s": s}
