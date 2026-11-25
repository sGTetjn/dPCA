"""Functional dPCA API: a 1:1 port of the reference MATLAB implementation (matlab/*.m).

Every function corresponds to one MATLAB file and reproduces its results exactly (signs,
component order, tie-breaking, cross-validation and shuffling procedures). To make porting of
MATLAB analysis scripts straightforward, the MATLAB conventions are kept:

* ``Xfull``        array (N, d1, ..., dD): trial-averaged data, neurons first. The **last
                   parameter is time**.
* ``Xtrial``       array (N, d1, ..., dD, max_trials): single trials, **trials last**, NaN-padded
                   at the end (trials 1..numOfTrials are the valid ones).
* ``num_trials``   integer array (N, d1, ..., d_{D-1}): number of trials per neuron and condition.
* Parameters are numbered **1..D** (as in MATLAB), e.g.
  ``combined_params=[[1, [1, 3]], [2, [2, 3]], [3], [[1, 2], [1, 2, 3]]]``.
* ``which_marg`` and decoding class ids are **1-based**.
* Flattening ``X(:,:)`` uses column-major (Fortran) order, like MATLAB.

Random numbers: MATLAB's ``rand`` and ``randperm`` are replaced by :func:`rand` and
:func:`randperm` drawing from a ``numpy.random.Generator`` (pass ``rng=`` an int seed or a
Generator for reproducibility). Plotting functions are not ported.
"""
import warnings

import numpy as np

__all__ = [
    "dpca", "dpca_marginalize", "dpca_explained_variance", "dpca_get_noise_covariance",
    "dpca_get_test_trials", "dpca_optimize_lambda", "dpca_classification_accuracy",
    "dpca_classification_shuffled", "dpca_shuffle_trials", "dpca_signif_components",
]

_EPS = np.finfo(float).eps


# ============================================================================ helpers
def rand(rng, n):
    """Uniform draws on [0, 1), the analogue of MATLAB ``rand([n 1])``."""
    return rng.random(n)


def randperm(rng, n, k=None):
    """1-based random permutation, the analogue of MATLAB ``randperm(n)`` / ``randperm(n, k)``."""
    p = rng.permutation(int(n)) + 1
    return p if k is None else p[:k]


def _rng(rng):
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _flat(A):
    """MATLAB ``A(:,:)``."""
    return np.reshape(A, (A.shape[0], -1), order="F")


def _unflat(A, shape):
    return np.reshape(A, shape, order="F")


def _center(X):
    """Subtract each neuron's mean over all conditions."""
    return X - np.mean(_flat(X), 1).reshape((-1,) + (1,) * (X.ndim - 1))


def _ints(v):
    return sorted(int(x) for x in np.atleast_1d(v))


def _subsets(items):
    """All non-empty subsets in MATLAB ``dpca_marginalize>subsets`` order: by size, then by the
    binary number whose bit j (LSB first) marks items[j]."""
    items = list(items)
    d = len(items)
    masks = sorted(range(1, 2 ** d), key=lambda m: bin(m).count("1"))       # stable sort by size
    return [[items[j] for j in range(d) if m >> j & 1] for m in masks]


def _quiet_nanmean(X, axis, keepdims=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(X, axis=axis, keepdims=keepdims)


def _per_condition(a, ndim):
    """Broadcast an (N, d1..d_{D-1}) array over the trailing time axis."""
    return np.reshape(a, a.shape + (1,) * (ndim - a.ndim))


# ============================================================================ dpca_marginalize.m
def dpca_marginalize(X, combined_params=None, if_full=True, if_flat=False, time_splits=None,
                     time_parameter=None, not_to_split=None):
    """Marginalize data over all parameter subsets (MATLAB ``dpca_marginalize``).

    Returns ``(margs, marg_nums)``: a list of marginalized arrays and the (1-based) index of the
    parameter combination each one belongs to (relevant with ``time_splits``).
    """
    X = _center(X)
    params = list(range(2, X.ndim + 1))                 # MATLAB dimension numbers of the parameters

    YY, done = [], {}
    for rest in _subsets(params):
        marg_dims = sorted(set(params) - set(rest))     # averaged-out dimensions
        Y = _quiet_nanmean(X, tuple(d - 1 for d in marg_dims), keepdims=True) if marg_dims else X
        done[tuple(marg_dims)] = len(YY)
        for s in _subsets(rest)[:-1]:                   # subtract all lower-order terms
            Y = Y - YY[done[tuple(sorted(s + marg_dims))]]
        YY.append(Y)

    param_subsets = _subsets([p - 1 for p in params])
    marg_nums = list(range(1, len(YY) + 1))

    if combined_params:
        lookup = {tuple(s): k for k, s in enumerate(param_subsets)}
        combined = []
        for group in combined_params:
            idx = [lookup[tuple(_ints(s))] for s in group]
            Y = YY[idx[0]]
            for j in idx[1:]:
                Y = Y + YY[j]
            combined.append(Y)
        YY = combined
        param_subsets = list(combined_params)
        marg_nums = list(range(1, len(combined_params) + 1))

    if if_full:
        YY = [np.broadcast_to(Y, X.shape) * 1.0 for Y in YY]

    if time_splits is not None and len(time_splits) > 0:
        groups = param_subsets if combined_params else [[s] for s in param_subsets]
        begins = [1] + list(time_splits)
        ends = [t - 1 for t in time_splits] + [X.shape[-1]]
        ZZ, marg_nums = [], []
        for i, Y in enumerate(YY):
            if not_to_split and _group_listed(groups[i], not_to_split):
                ZZ.append(Y)
                marg_nums.append(i + 1)
                continue
            for b, e in zip(begins, ends):
                ind = [slice(None)] * X.ndim
                ind[time_parameter] = slice(b - 1, e)
                Z = np.zeros(Y.shape)
                Z[tuple(ind)] = Y[tuple(ind)]
                ZZ.append(Z)
                marg_nums.append(i + 1)
        YY = ZZ

    if if_flat:
        YY = [_flat(Y) for Y in YY]
    return YY, np.array(marg_nums)


def _group_listed(group, groups):
    """MATLAB notToSplit matching: same number of subsets and every subset of ``group`` occurs."""
    for other in groups:
        if len(group) != len(other):
            continue
        others = [_ints(o) for o in other]
        if all(_ints(s) in others for s in group):
            return True
    return False


# ============================================================================ dpca.m
def _ridge_operator(Xm, X, Cnoise, total_var, lam):
    """C = Xm X' (X X' + Cnoise + (total_var*lam)^2 I)^-1, with MATLAB's fallback to
    lambda = 1e-10 when the matrix is singular to working precision (rcond < eps)."""
    N = X.shape[0]

    def gram(l):
        return X @ X.T + Cnoise + (total_var * l) ** 2 * np.eye(N)

    def near_singular(A):
        # exact 1-norm reciprocal condition number (MATLAB uses LAPACK's estimate of the same)
        try:
            rc = 1.0 / np.linalg.cond(A, 1)
        except np.linalg.LinAlgError:
            return True
        return not np.isfinite(rc) or rc < _EPS

    A = gram(lam)
    if near_singular(A):
        print("Matrix close to singular, using tiny regularization, lambda = 1e-10")
        A = gram(1e-10)
        if near_singular(A):
            # in MATLAB the retry still runs with near-singular warnings turned into errors
            raise np.linalg.LinAlgError(
                "X*X' is singular to working precision even with lambda = 1e-10 (MATLAB raises "
                "MATLAB:nearlySingularMatrix here). Use a positive regularizer (lambda) or Cnoise.")
    return np.linalg.solve(A.T, (Xm @ X.T).T).T


def dpca(Xfull, num_comps, combined_params=None, lambda_=0, order=True, time_splits=None,
         time_parameter=None, not_to_split=None, scale=False, Cnoise=None):
    """dPCA decoder/encoder (MATLAB ``dpca``).

    Parameters
    ----------
    Xfull : array (N, d1, ..., dD)
    num_comps : int or sequence
        A single number extracts that many components per marginalization and keeps the
        ``num_comps`` components with the highest explained variance **in total**. A sequence
        gives the number of components per marginalization.
    lambda_ : float or sequence
        Ridge parameter, multiplied by the total variance of the data (per marginalization if a
        sequence).
    order : bool
        Sort components by explained variance (always done when ``num_comps`` is scalar).
    scale : bool
        Rescale each decoder to minimize reconstruction error.
    Cnoise : array (N, N) or None
        Noise covariance (see :func:`dpca_get_noise_covariance`) added to X X'.

    Returns
    -------
    W : (N, S) decoder, V : (N, S) encoder, which_marg : (S,) 1-based marginalization index
    """
    X = _flat(Xfull)
    X = X - X.mean(1, keepdims=True)
    total_var = np.sum(X ** 2)
    margs, marg_nums = dpca_marginalize(_unflat(X, Xfull.shape), combined_params=combined_params,
                                        time_splits=time_splits, time_parameter=time_parameter,
                                        not_to_split=not_to_split, if_flat=True)
    nc_all = np.atleast_1d(num_comps).astype(int)
    lam_all = np.atleast_1d(lambda_).astype(float)
    if Cnoise is None or (np.ndim(Cnoise) > 0 and np.size(Cnoise) == 0):
        Cnoise = 0.0

    decoders, encoders, which = [], [], []
    for i, Xm in enumerate(margs):
        nc = nc_all[0] if nc_all.size == 1 else nc_all[marg_nums[i] - 1]
        lam = lam_all[0] if lam_all.size == 1 else lam_all[marg_nums[i] - 1]
        if nc == 0:
            continue
        if nc > X.shape[0]:
            raise ValueError("Cannot extract %d components from %d neurons in marginalization %d."
                             % (nc, X.shape[0], i + 1))
        C = _ridge_operator(Xm, X, Cnoise, total_var, lam)
        M = C @ X
        w, U = np.linalg.eigh(M @ M.T)
        U = U[:, np.argsort(-np.abs(w), kind="stable")[:nc]]      # eigs(M*M', nc)
        P, D = U, U.T @ C
        if scale:
            for u in range(D.shape[0]):
                B = np.outer(P[:, u], D[u]) @ X
                D[u] *= np.sum(Xm * B) / np.sum(B * B)
        decoders.append(D)
        encoders.append(P)
        which += [i + 1] * nc

    V = np.hstack(encoders)
    W = np.vstack(decoders).T
    which = np.array(which, dtype=int)

    flip = np.sum(np.sign(V), 0) < 0                 # encoders with mostly positive entries
    W[:, flip] *= -1
    V[:, flip] *= -1

    expl_var = np.zeros(0)
    if time_splits is not None and len(time_splits) > 0:
        keep = []
        for i in range(1, int(marg_nums.max()) + 1):
            comps = np.nonzero(np.isin(which, np.nonzero(marg_nums == i)[0] + 1))[0]
            expl_var = np.sum((W[:, comps].T @ X) ** 2, 1)
            nc = nc_all[0] if nc_all.size == 1 else nc_all[i - 1]
            keep += list(comps[np.argsort(-expl_var, kind="stable")[:nc]])
        W, V = W[:, keep], V[:, keep]
        which = marg_nums[which[keep] - 1]

    if nc_all.size == 1 or order:
        # explained variance of each component. (Like MATLAB, entries left over from the
        # time-split step are overwritten in place, not cleared.)
        ev = np.array(expl_var, dtype=float)
        if ev.size < W.shape[1]:
            ev = np.concatenate([ev, np.zeros(W.shape[1] - ev.size)])
        for i in range(W.shape[1]):
            ev[i] = 1 - np.sum((X - np.outer(V[:, i], W[:, i] @ X)) ** 2) / total_var
        o = np.argsort(-ev, kind="stable")[: int(nc_all[0]) if nc_all.size == 1 else int(nc_all.sum())]
        W, V, which = W[:, o], V[:, o], which[o]
    return W, V, which


# ============================================================================ dpca_explainedVariance.m
def dpca_explained_variance(Xfull, W, V, combined_params=None, X_trial=None, num_trials=None, Cnoise=None,
                            rng=None):
    """Explained-variance measures (MATLAB ``dpca_explainedVariance``); returns a dict with the
    MATLAB field names (``totalVar``, ``totalMarginalizedVar``, ``componentVar``, ``margVar``,
    ``cumulativePCA``, ``cumulativeDPCA`` and, with trial data or ``Cnoise``, the signal-variance
    fields). Percentages, like MATLAB."""
    X = _flat(Xfull)
    mu = X.mean(1, keepdims=True)
    Xfull = Xfull - mu.reshape((-1,) + (1,) * (Xfull.ndim - 1))
    X = X - mu
    margs, _ = dpca_marginalize(Xfull, combined_params=combined_params, if_flat=True)

    ev = {"totalVar": np.sum(X ** 2)}
    ev["totalMarginalizedVar"] = np.array([np.sum(m ** 2) for m in margs])
    tv = ev["totalVar"]
    k = W.shape[1]

    S = np.linalg.svd(X.T, compute_uv=False)
    if k > S.size:
        raise ValueError("More components (%d) than PCA components (%d); MATLAB fails here as well "
                         "(index exceeds matrix dimensions)." % (k, S.size))
    S = S[:k]
    ev["cumulativePCA"] = np.cumsum(S ** 2 / tv * 100)

    Z = W.T @ X
    ev["cumulativeDPCA"] = np.array([100 - np.sum((X - V[:, :i + 1] @ Z[:i + 1]) ** 2) / tv * 100 for i in range(k)])
    ev["componentVar"] = np.array([100 - np.sum((X - np.outer(V[:, i], Z[i])) ** 2) / tv * 100 for i in range(k)])
    ev["margVar"] = np.array([[(ev["totalMarginalizedVar"][j] - np.sum((m - np.outer(V[:, i], W[:, i] @ m)) ** 2))
                               / tv * 100 for i in range(k)] for j, m in enumerate(margs)])
    if np.max(np.abs(ev["cumulativePCA"] - ev["cumulativeDPCA"])) < 1e-10:
        ev["cumulativeDPCA"] = np.zeros(0)     # PCA axes were supplied

    if X_trial is not None and num_trials is not None:
        # old approach: noise sample = difference of two random trials per neuron & condition
        rng = _rng(rng)
        nt = np.asarray(num_trials)
        draws = [randperm(rng, n, 2) for n in nt.ravel(order="F")]
        i1 = _unflat(np.array([d[0] for d in draws]), nt.shape) - 1
        i2 = _unflat(np.array([d[1] for d in draws]), nt.shape) - 1
        tshape = X_trial.shape[:-1]
        i1 = np.broadcast_to(_per_condition(i1, len(tshape)), tshape)[..., None]
        i2 = np.broadcast_to(_per_condition(i2, len(tshape)), tshape)[..., None]
        dif = (np.take_along_axis(X_trial, i1, -1) - np.take_along_axis(X_trial, i2, -1))[..., 0]
        noise_full = dif / np.sqrt(2 * _per_condition(nt, len(tshape)))
        noise_full = _center(noise_full)
        margs_noise, _ = dpca_marginalize(noise_full, combined_params=combined_params)
        ev["totalVar_noise"] = np.sum(noise_full ** 2)
        ev["totalVar_signal"] = tv - ev["totalVar_noise"]
        ev["totalMarginalizedVar_signal"] = ev["totalMarginalizedVar"] - np.array([np.sum(m ** 2) for m in margs_noise])

    if Cnoise is not None and num_trials is not None:
        # new approach (hard-coded in MATLAB for 3 parameters with 4 combined marginalizations)
        nt = np.asarray(num_trials)
        Ktilde = _flat(nt).mean(1)
        ev["totalVar_noise"] = np.sum(np.diag(Cnoise) / Ktilde)
        ev["totalVar_signal"] = tv - ev["totalVar_noise"]
        sz = list(Xfull.shape) + [1] * 4
        S_, Q_, T_ = sz[1], sz[2], sz[3]
        frac = np.array([S_ * T_ - T_, Q_ * T_ - T_, T_ - 1, S_ * Q_ * T_ - S_ * T_ - Q_ * T_ + T_]) / (S_ * Q_ * T_ - 1)
        if frac.size != ev["totalMarginalizedVar"].size and ev["totalMarginalizedVar"].size != 1:
            raise ValueError("The Cnoise-based signal variance (as in MATLAB) requires data with 3 parameters "
                             "and 4 combined marginalizations (stimulus, decision, time, interaction).")
        ev["totalMarginalizedVar_signal"] = ev["totalMarginalizedVar"] - frac * ev["totalVar_noise"]
    return ev


# ============================================================================ dpca_getNoiseCovariance.m
def dpca_get_noise_covariance(Xfull, Xtrial, num_trials, simultaneous=False, type="pooled"):
    """Noise covariance matrix (MATLAB ``dpca_getNoiseCovariance``). ``type`` is 'pooled' or
    'averaged' ('averaged' is not available for simultaneous recordings, as in MATLAB)."""
    nt = np.asarray(num_trials)
    if not simultaneous:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ss = np.nansum(Xtrial ** 2, -1) - Xfull ** 2 * _per_condition(nt, Xfull.ndim)
            ss_t = _flat(ss.sum(-1))
            avg = _flat(nt).astype(float)
            avg[avg == 0] = np.nan
            avg = np.nanmean(avg, 1)
            if type == "pooled":
                return np.diag(np.nansum(ss_t, 1) / avg)
            if type == "averaged":
                return np.diag(np.nansum(ss_t / _flat(nt), 1))
        raise ValueError("type must be 'pooled' or 'averaged'")
    if type == "pooled":
        noise = _flat(Xtrial - Xfull[..., None])
        noise = noise[:, ~np.isnan(noise[0])]
        return noise @ noise.T / noise.shape[1] * np.prod(Xfull.shape[1:])
    raise NotImplementedError("Averaged noise covariance computation is not implemented for "
                              "simultaneously recorded data (as in the MATLAB code).")


# ============================================================================ dpca_getTestTrials.m
def dpca_get_test_trials(Xtrial, num_trials, rng=None, simultaneous=False):
    """Pick one random test trial per neuron and condition (the same trial for all time bins;
    for simultaneous recordings also the same for all neurons). Returns ``(Xtest, Xtrain_full)``
    where ``Xtrain_full`` is ``Xtrial`` with the test trials set to NaN."""
    rng = _rng(rng)
    nt = np.asarray(num_trials)
    if not simultaneous:
        test = np.ceil(rand(rng, nt.size) * nt.ravel(order="F"))
    else:
        per_cond = nt.ravel(order="F")[:: nt.shape[0]]
        test = np.repeat(np.ceil(rand(rng, per_cond.size) * per_cond), nt.shape[0])
    test = _unflat(test.astype(int) - 1, nt.shape)
    idx = np.broadcast_to(_per_condition(test, Xtrial.ndim - 1), Xtrial.shape[:-1])[..., None]
    Xtest = np.take_along_axis(Xtrial, idx, -1)[..., 0]
    Xtrain = Xtrial.copy()
    np.put_along_axis(Xtrain, idx, np.nan, -1)
    return Xtest, Xtrain


def _split(Xfull, Xtrial, nt, rng, simultaneous):
    Xtest, Xtrain_full = dpca_get_test_trials(Xtrial, nt, rng=rng, simultaneous=simultaneous)
    ntc = _per_condition(nt, Xfull.ndim)
    Xtrain = (Xfull * ntc - Xtest) / (ntc - 1)
    return Xtest, Xtrain_full, Xtrain


# ============================================================================ dpca_optimizeLambda.m
def dpca_optimize_lambda(Xfull, Xtrial, num_trials, num_comps=25, lambdas=None, num_rep=10, combined_params=None,
                         method="training", simultaneous=False, noise_cov_type="pooled", rng=None, verbose=False):
    """Cross-validated choice of the regularization parameter (MATLAB ``dpca_optimizeLambda``).

    ``method``: 'training' (default), 'neuronwise' or 'naive' (deprecated).
    ``noise_cov_type``: 'pooled' (default), 'averaged' or 'none'.

    Returns ``(optimal_lambda, optimal_lambdas, errors, errors_marg)`` where ``optimal_lambdas``
    holds the optimum per marginalization, ``errors`` is (n_lambdas, num_rep) and
    ``errors_marg`` is (n_marg, n_lambdas, num_rep).
    """
    rng = _rng(rng)
    lambdas = 1e-7 * 1.5 ** np.arange(26) if lambdas is None else np.asarray(lambdas, float)
    nt = np.asarray(num_trials)
    if nt.min() <= 0:
        raise ValueError("Some neurons seem to have no trials in some condition(s). "
                         "Please ensure that min(num_trials) > 0.")
    if nt.min() == 1:
        raise ValueError("Cannot perform cross-validation when there are neurons having only one trial "
                         "per some condition(s). Please ensure that min(num_trials) > 1.")

    errors = np.zeros((len(lambdas), num_rep))
    errors_marg = None
    for rep in range(num_rep):
        if verbose:
            print("Iteration #%d out of %d" % (rep + 1, num_rep))
        Xtest, Xtrain_full, Xtrain = _split(Xfull, Xtrial, nt, rng, simultaneous)
        Cn = 0.0 if noise_cov_type == "none" else dpca_get_noise_covariance(
            Xtrain, Xtrain_full, nt - 1, simultaneous=simultaneous, type=noise_cov_type)

        Xtest_c = _center(Xtest)
        test_margs, _ = dpca_marginalize(Xtest_c, combined_params=combined_params, if_flat=True)
        train_margs, _ = dpca_marginalize(_center(Xtrain), combined_params=combined_params, if_flat=True)
        targets = train_margs if method == "training" else test_margs
        norm = np.array([np.sum(m ** 2) for m in targets])
        if errors_marg is None:
            errors_marg = np.zeros((len(test_margs), len(lambdas), num_rep))
        Xt = _flat(Xtest_c)

        for l, lam in enumerate(lambdas):
            W, V, which = dpca(Xtrain, num_comps, combined_params=combined_params, lambda_=lam, Cnoise=Cn)
            cum = 0.0
            for i in range(len(test_margs)):
                sel = which == i + 1
                recon = V[:, sel] @ (W[:, sel].T @ Xt)
                if method == "naive":
                    err = np.sum((test_margs[i] - recon) ** 2)
                elif method == "training":
                    err = np.sum((train_margs[i] - recon) ** 2)
                elif method == "neuronwise":
                    # predict each neuron without using that neuron's own test activity
                    err = np.sum((test_margs[i] - recon + np.sum(V[:, sel] * W[:, sel], 1)[:, None] * Xt) ** 2)
                else:
                    raise ValueError("method must be 'training', 'neuronwise' or 'naive'")
                errors_marg[i, l, rep] = err / norm[i]
                cum += err
            errors[l, rep] = cum / norm.sum()

    optimal_lambda = lambdas[int(np.argmin(errors.mean(1)))]
    optimal_lambdas = lambdas[np.argmin(errors_marg.mean(2), 1)]
    return optimal_lambda, optimal_lambdas, errors, errors_marg


# ============================================================================ dpca_classificationAccuracy.m
def _time_marginalization(Xfull, combined_params):
    """Index (1-based) of the pure time marginalization, as determined in MATLAB."""
    D = Xfull.ndim - 1
    if not combined_params:
        return [D]
    for k, group in enumerate(combined_params):
        first = np.atleast_1d(group[0])
        if first.size > 0 and np.all(first == D):
            return [k + 1]
    return []


def dpca_classification_accuracy(Xfull, Xtrial, num_trials, num_comps=3, lambda_=0, num_rep=100,
                                 combined_params=None, decoding_classes=None, time_splits=None, time_parameter=None,
                                 not_to_split=None, simultaneous=False, noise_cov_type="pooled", rng=None,
                                 verbose=False):
    """Cross-validated classification accuracy of dPCA components at each time point (MATLAB
    ``dpca_classificationAccuracy``).

    ``decoding_classes`` is a list with one entry per marginalization: an array shaped like the
    condition dimensions (d1, ..., d_{D-1}) holding 1-based class ids (None for time). The
    default is one class per condition.

    Returns ``(accuracy, brier)``, each (n_marg, num_comps, T); the time marginalization is NaN.
    """
    rng = _rng(rng)
    nt = np.asarray(num_trials)
    dim = Xfull.shape
    Cnoise = dpca_get_noise_covariance(Xfull, Xtrial, nt, type=noise_cov_type, simultaneous=simultaneous)
    time_comp = _time_marginalization(Xfull, combined_params)

    n_marg = len(combined_params) if combined_params else 2 ** (Xfull.ndim - 1) - 1
    comps_to_use = np.tile(np.atleast_1d(num_comps).astype(int), n_marg)     # repmat(numComps, [1 nMarg])
    for tc in time_comp:
        comps_to_use[tc - 1] = 0

    if decoding_classes is None:
        # one class per condition. (MATLAB's reshape(1:prod(dims), dims) fails for a single
        # condition dimension; the evident intent is used.)
        cdims = dim[1:-1]
        classes = _unflat(np.arange(1, int(np.prod(cdims)) + 1), cdims)
        decoding_classes = [classes] * n_marg

    fit_kw = dict(combined_params=combined_params, lambda_=lambda_, time_splits=time_splits,
                  time_parameter=time_parameter, not_to_split=not_to_split, order=False)
    _, Vfull, _ = dpca(Xfull, comps_to_use, Cnoise=Cnoise, **fit_kw)

    correct = brier = None
    for rep in range(num_rep):
        if verbose:
            print("Repetition # %d out of %d" % (rep + 1, num_rep))
        Xtest, Xtrain_full, Xtrain = _split(Xfull, Xtrial, nt, rng, simultaneous)
        Cn_train = dpca_get_noise_covariance(Xtrain, Xtrain_full, nt - 1, simultaneous=simultaneous,
                                             type=noise_cov_type)
        W, V, which = dpca(Xtrain, comps_to_use, Cnoise=Cn_train, **fit_kw)
        margs = sorted(set(which.tolist()) - set(time_comp))

        if correct is None:
            n1 = max(set(which.tolist()) | set(time_comp))
            n2 = max(int(np.sum(which == m)) for m in set(which.tolist()))
            correct = np.zeros((n1, n2, dim[-1], num_rep))
            brier = np.zeros((n1, n2, dim[-1], num_rep))

        # re-order each split's components to match the full-data components. MATLAB computes a
        # greedy assignment and then overwrites it with [~, order] = max(crc); the latter is used.
        k = V.shape[1]
        cc = np.abs(np.corrcoef(np.hstack([V, Vfull]), rowvar=False)[k:, :k])
        for m in margs:
            d = np.nonzero(which == m)[0]
            order = np.argmax(cc[np.ix_(d, d)], axis=0)
            W[:, d] = W[:, d[order]]
            V[:, d] = V[:, d[order]]

        Xtr = _flat(Xtrain)
        centering = Xtr.mean(1, keepdims=True)       # test data are centered with the training mean
        Xtr = Xtr - centering
        Xte = _flat(Xtest) - centering

        for m in margs:
            cls = np.asarray(decoding_classes[m - 1]).ravel(order="F").astype(int)
            n_unique = len(np.unique(cls))
            counts = np.bincount(cls, minlength=cls.max() + 1)[1:]
            onehot = np.zeros((cls.size, cls.max()))
            onehot[np.arange(cls.size), cls - 1] = 1
            for j, comp in enumerate(np.nonzero(which == m)[0]):
                ztr = _unflat(W[:, comp] @ Xtr, dim[1:]).reshape(-1, dim[-1], order="F")   # (conditions, T)
                zte = _unflat(W[:, comp] @ Xte, dim[1:]).reshape(-1, dim[-1], order="F")
                sums = np.zeros((cls.max(), dim[-1]))
                np.add.at(sums, cls - 1, ztr)
                with np.errstate(invalid="ignore", divide="ignore"):
                    means = np.where(counts[:, None] > 0, sums / np.maximum(counts, 1)[:, None], 0.0)  # accumarray(@mean)
                for t in range(dim[-1]):
                    dist = zte[:, t][:, None] - means[:, t][None, :]
                    correct[m - 1, j, t, rep] = np.sum(np.argmin(np.abs(dist), 1) + 1 == cls)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        prob = 1.0 / np.abs(dist)
                        prob = prob / prob.sum(1, keepdims=True)
                    brier[m - 1, j, t, rep] = np.sum((prob - onehot) ** 2) / n_unique

    n_cond = int(np.prod(dim[1:-1]))
    accuracy = correct.sum(3) / (num_rep * n_cond)
    brier = brier.sum(3) / (num_rep * n_cond)
    for tc in time_comp:
        accuracy[tc - 1] = np.nan
        brier[tc - 1] = np.nan
    return accuracy, brier


# ============================================================================ dpca_classificationShuffled.m
def dpca_shuffle_trials(Xtrial, num_trials, rng=None, simultaneous=False):
    """Shuffle whole trials across conditions **within each neuron** (or jointly for all neurons
    when ``simultaneous``), preserving each condition's number of trials. As in MATLAB, unused
    trial slots of the returned array are 0 (not NaN)."""
    rng = _rng(rng)
    nt = np.asarray(num_trials)
    N, T, max_tr = nt.shape[0], Xtrial.shape[-2], Xtrial.shape[-1]
    nt_cond = _flat(nt)                                             # (N, conditions)
    order = list(range(Xtrial.ndim))
    order[-2], order[-1] = order[-1], order[-2]
    Xc = np.reshape(np.transpose(Xtrial, order), (N, -1, T), order="F")   # (N, conditions*trials, T)
    # slot k <-> (condition k % n_cond, trial k // n_cond)
    present = np.arange(max_tr)[None, None, :] < nt_cond[:, :, None]        # (N, cond, trials)
    present = np.reshape(present, (N, -1), order="F")
    out = np.zeros(Xc.shape)
    if not simultaneous:
        for n in range(N):
            slots = np.nonzero(present[n])[0]
            out[n, slots] = Xc[n, slots[randperm(rng, slots.size) - 1]]
    else:
        slots = np.nonzero(present[0])[0]
        out[:, slots] = Xc[:, slots[randperm(rng, slots.size) - 1]]
    shape_t = [Xtrial.shape[o] for o in order]
    return np.transpose(np.reshape(out, shape_t, order="F"), order)


def dpca_classification_shuffled(Xtrial, num_trials, num_shuffles=100, lambda_=0, num_rep=100, combined_params=None,
                                 decoding_classes=None, time_splits=None, time_parameter=None, not_to_split=None,
                                 simultaneous=False, noise_cov_type="pooled", rng=None, verbose=False):
    """Null distribution of classification accuracy under shuffled condition labels (MATLAB
    ``dpca_classificationShuffled``). Returns ``(accuracy_shuffle, brier_shuffle)``, each
    (n_marg, T, num_shuffles), for the first component of each marginalization."""
    rng = _rng(rng)
    nt = np.asarray(num_trials)
    acc_all, brier_all = [], []
    for s in range(num_shuffles):
        if verbose:
            print("Shuffle #%d out of %d" % (s + 1, num_shuffles))
        Xs = dpca_shuffle_trials(Xtrial, nt, rng=rng, simultaneous=simultaneous)
        Xavg = Xs.sum(-1) / _per_condition(nt, Xs.ndim - 1)
        acc, br = dpca_classification_accuracy(
            Xavg, Xs, nt, num_comps=1, lambda_=lambda_, num_rep=num_rep, combined_params=combined_params,
            decoding_classes=decoding_classes, time_splits=time_splits, time_parameter=time_parameter,
            not_to_split=not_to_split, simultaneous=simultaneous, noise_cov_type=noise_cov_type, rng=rng)
        acc_all.append(acc[:, 0, :])
        brier_all.append(br[:, 0, :])
    return np.stack(acc_all, -1), np.stack(brier_all, -1)


# ============================================================================ dpca_signifComponents.m
def dpca_signif_components(accuracy, accuracy_shuffle, which_marg, min_chunk=10):
    """Time bins where a component's accuracy exceeds the maximum over all shuffles (MATLAB
    ``dpca_signifComponents``). Only the first 3 components of each marginalization are tested.

    Short runs are then removed exactly as in MATLAB: a run of L consecutive significant bins
    is discarded when ``L + 1 < min_chunk`` (i.e. runs of length <= min_chunk - 2).
    Returns an array (n_components, T) of 0/1.
    """
    which_marg = np.asarray(which_marg)
    sig = np.zeros((len(which_marg), accuracy.shape[2]))
    for m in range(1, accuracy.shape[0] + 1):
        if np.isnan(accuracy[m - 1, 0, 0]):
            continue                                       # time marginalization
        thresh = np.max(accuracy_shuffle[m - 1], axis=1)
        for num, comp in enumerate(np.nonzero(which_marg == m)[0][:3]):
            sig[comp] = accuracy[m - 1, num] > thresh

    for row in sig:
        zeros = np.r_[0, np.nonzero(row == 0)[0] + 1, row.size + 1]        # 1-based positions
        for a, b in zip(zeros[:-1], zeros[1:]):
            if 1 < b - a < min_chunk:
                row[a:b - 1] = 0
    return sig
