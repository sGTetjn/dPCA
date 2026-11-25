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
import scipy.linalg

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
    if not np.isnan(X).any():
        # same summation and division as nanmean, without its NaN-replaced copy
        return np.mean(X, axis=axis, keepdims=keepdims)
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
def _near_singular(A):
    # exact 1-norm reciprocal condition number (MATLAB uses LAPACK's estimate of the same)
    try:
        rc = 1.0 / np.linalg.cond(A, 1)
    except np.linalg.LinAlgError:
        return True
    return not np.isfinite(rc) or rc < _EPS


class _Ridge:
    """Ridge operators C_i = Xm_i X' (X X' + Cnoise + (total_var*lam)^2 I)^-1 of all
    marginalizations, with MATLAB's fallback to lambda = 1e-10 when the matrix is singular to
    working precision (rcond < eps).

    ``operator(i, lam)`` returns C_i and M_i M_i' = C_i (X X') C_i' (M_i = C_i X), whose leading
    eigenvectors are the encoders. Everything that does not depend on lambda (X X', Xm_i X') is
    computed once, and the singularity check and LU factorization are done once per lambda rather
    than once per marginalization.

    With ``spectral=True`` (used by the lambda search) X X' + Cnoise = Q diag(s) Q' is also
    eigendecomposed once, so that each further lambda costs only matrix products:
    C_i = (Xm_i X' Q) diag(1 / (s + c)) Q'. The spectral route is used only where the eigenvalues
    certify that the matrix is far from singular (see ``_certified``); otherwise the LU route,
    identical to the plain fit, is taken.
    """

    def __init__(self, X, margs, Cnoise, total_var, spectral=False):
        self.X, self.margs, self.total_var = X, margs, total_var
        self.N = X.shape[0]
        self.XXt = X @ X.T
        self.G = self.XXt + Cnoise
        self._B, self._BQ, self._lu, self._used = {}, {}, {}, {}
        self.eig = None
        if spectral and np.array_equal(self.G, self.G.T):
            s, Q = np.linalg.eigh(self.G)
            self.eig = (s, Q, Q.T @ self.XXt @ Q)

    def _shift(self, lam):
        return (self.total_var * lam) ** 2

    def _gram(self, lam):
        return self.G + self._shift(lam) * np.eye(self.N)

    def _certified(self, lam):
        """True if the eigenvalues prove rcond_1(A) >= eps. For symmetric positive definite A,
        rcond_1 >= 1 / (N * cond_2), and computed eigenvalues are accurate to O(N eps max|s|), so
        min/max >= 100 N eps is a safe sufficient condition."""
        if self.eig is None:
            return False
        s = self.eig[0]
        c = self._shift(lam)
        lo, hi = s[0] + c, s[-1] + c
        return lo > 0 and lo >= 100 * self.N * _EPS * hi

    def _resolve(self, lam):
        """The lambda actually used: ``lam``, MATLAB's fallback 1e-10, or None if both fail."""
        if lam not in self._used:
            if self._certified(lam) or not _near_singular(self._gram(lam)):
                self._used[lam] = lam
            elif self._certified(1e-10) or not _near_singular(self._gram(1e-10)):
                self._used[lam] = 1e-10
            else:
                self._used[lam] = None
        used = self._used[lam]
        if used != lam:
            print("Matrix close to singular, using tiny regularization, lambda = 1e-10")
        if used is None:
            # in MATLAB the retry still runs with near-singular warnings turned into errors
            raise np.linalg.LinAlgError(
                "X*X' is singular to working precision even with lambda = 1e-10 (MATLAB raises "
                "MATLAB:nearlySingularMatrix here). Use a positive regularizer (lambda) or Cnoise.")
        return used

    def _xm_xt(self, i):
        if i not in self._B:
            self._B[i] = self.margs[i] @ self.X.T
        return self._B[i]

    def operator(self, i, lam):
        lam = self._resolve(lam)
        if self._certified(lam):
            s, Q, H = self.eig
            if i not in self._BQ:
                self._BQ[i] = self._xm_xt(i) @ Q
            K = self._BQ[i] / (s + self._shift(lam))
            return K @ Q.T, K @ H @ K.T
        if lam not in self._lu:
            # same LAPACK calls as np.linalg.solve(A.T, B.T) (getrf + getrs), factorized once
            self._lu[lam] = scipy.linalg.lu_factor(self._gram(lam).T, check_finite=False)
        C = scipy.linalg.lu_solve(self._lu[lam], self._xm_xt(i).T, check_finite=False).T
        return C, C @ self.XXt @ C.T


def _dpca_prepare(Xfull, combined_params=None, time_splits=None, time_parameter=None, not_to_split=None):
    """Centred flattened data, its total variance and the flattened marginalizations."""
    X = _flat(Xfull)
    X = X - X.mean(1, keepdims=True)
    total_var = np.sum(X ** 2)
    margs, marg_nums = dpca_marginalize(_unflat(X, Xfull.shape), combined_params=combined_params,
                                        time_splits=time_splits, time_parameter=time_parameter,
                                        not_to_split=not_to_split, if_flat=True)
    return X, total_var, margs, marg_nums


def _component_var(X, W, V):
    """||X||^2 - ||X - v_i w_i' X||^2 for every component i, without forming the residuals:
    with z = X' w and y = X' v, ||X - v z'||^2 = ||X||^2 - 2 y.z + ||v||^2 ||z||^2."""
    Z = W.T @ X
    Y = V.T @ X
    return 2 * np.sum(Y * Z, 1) - np.sum(V ** 2, 0) * np.sum(Z ** 2, 1)


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
    X, total_var, margs, marg_nums = _dpca_prepare(Xfull, combined_params, time_splits, time_parameter,
                                                   not_to_split)
    if Cnoise is None or (np.ndim(Cnoise) > 0 and np.size(Cnoise) == 0):
        Cnoise = 0.0
    ridge = _Ridge(X, margs, Cnoise, total_var)
    return _dpca_fit(X, total_var, margs, marg_nums, ridge, num_comps, lambda_, order, time_splits, scale)


def _dpca_fit(X, total_var, margs, marg_nums, ridge, num_comps, lambda_, order=True, time_splits=None,
              scale=False):
    """Body of :func:`dpca` on prepared data (see ``_dpca_prepare``) and a ``_Ridge``."""
    nc_all = np.atleast_1d(num_comps).astype(int)
    lam_all = np.atleast_1d(lambda_).astype(float)

    decoders, encoders, which = [], [], []
    for i, Xm in enumerate(margs):
        nc = nc_all[0] if nc_all.size == 1 else nc_all[marg_nums[i] - 1]
        lam = lam_all[0] if lam_all.size == 1 else lam_all[marg_nums[i] - 1]
        if nc == 0:
            continue
        if nc > X.shape[0]:
            raise ValueError("Cannot extract %d components from %d neurons in marginalization %d."
                             % (nc, X.shape[0], i + 1))
        C, MMt = ridge.operator(i, lam)
        w, U = np.linalg.eigh(MMt)
        U = U[:, np.argsort(-np.abs(w), kind="stable")[:nc]]      # eigs(M*M', nc)
        P, D = U, U.T @ C
        if scale:
            for u in range(D.shape[0]):
                # B = P_u D_u X = p z'  =>  sum(Xm .* B) = p' Xm z,  sum(B .* B) = |p|^2 |z|^2
                p, z = P[:, u], D[u] @ X
                D[u] *= ((p @ Xm) @ z) / ((p @ p) * (z @ z))
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
        ev[:W.shape[1]] = _component_var(X, W, V) / total_var
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

    # Residual norms are expanded instead of forming the (N x conditions*T) residuals:
    # ||X - V_k Z_k||^2 = ||X||^2 - 2 sum_{j<=k} y_j.z_j + sum_{j,l<=k} (V'V)_jl (Z Z')_jl.
    Z = W.T @ X
    Y = V.T @ X
    cross = np.cumsum(np.sum(Y * Z, 1))
    gram = np.cumsum(np.cumsum((V.T @ V) * (Z @ Z.T), 0), 1)
    ev["cumulativeDPCA"] = (2 * cross - np.diag(gram)) / tv * 100
    ev["componentVar"] = _component_var(X, W, V) / tv * 100
    ev["margVar"] = np.array([_component_var(m, W, V) / tv * 100 for m in margs]).reshape(len(margs), k)
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
        return _noise_cov_from_sumsq(_trial_sumsq(Xtrial), Xfull, nt, type)
    if type == "pooled":
        noise = _flat(Xtrial - Xfull[..., None])
        noise = noise[:, ~np.isnan(noise[0])]
        return noise @ noise.T / noise.shape[1] * np.prod(Xfull.shape[1:])
    raise NotImplementedError("Averaged noise covariance computation is not implemented for "
                              "simultaneously recorded data (as in the MATLAB code).")


def _trial_sumsq(Xtrial):
    """Sum of squares over the trials (missing trials are NaN)."""
    return np.nansum(Xtrial ** 2, -1)


def _noise_cov_from_sumsq(sumsq, Xfull, nt, type):
    """Noise covariance of non-simultaneous recordings from the per-condition sum of squares
    over trials (see :func:`dpca_get_noise_covariance`)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ss = sumsq - Xfull ** 2 * _per_condition(nt, Xfull.ndim)
        ss_t = _flat(ss.sum(-1))
        avg = _flat(nt).astype(float)
        avg[avg == 0] = np.nan
        avg = np.nanmean(avg, 1)
        if type == "pooled":
            return np.diag(np.nansum(ss_t, 1) / avg)
        if type == "averaged":
            return np.diag(np.nansum(ss_t / _flat(nt), 1))
    raise ValueError("type must be 'pooled' or 'averaged'")


# ============================================================================ dpca_getTestTrials.m
def dpca_get_test_trials(Xtrial, num_trials, rng=None, simultaneous=False):
    """Pick one random test trial per neuron and condition (the same trial for all time bins;
    for simultaneous recordings also the same for all neurons). Returns ``(Xtest, Xtrain_full)``
    where ``Xtrain_full`` is ``Xtrial`` with the test trials set to NaN."""
    test = _draw_test_trials(num_trials, _rng(rng), simultaneous)
    return _apply_test_trials(Xtrial, test)


def _draw_test_trials(num_trials, rng, simultaneous):
    """The random part of :func:`dpca_get_test_trials`: 0-based test trial index per neuron and
    condition. Drawing is kept separate from the arrays so that all random numbers of a
    cross-validation can be drawn up front, in the serial order, and the repetitions then run in
    parallel with results identical to a serial run."""
    nt = np.asarray(num_trials)
    if not simultaneous:
        test = np.ceil(rand(rng, nt.size) * nt.ravel(order="F"))
    else:
        per_cond = nt.ravel(order="F")[:: nt.shape[0]]
        test = np.repeat(np.ceil(rand(rng, per_cond.size) * per_cond), nt.shape[0])
    return _unflat(test.astype(int) - 1, nt.shape)


def _test_index(Xtrial, test):
    return np.broadcast_to(_per_condition(test, Xtrial.ndim - 1), Xtrial.shape[:-1])[..., None]


def _apply_test_trials(Xtrial, test):
    idx = _test_index(Xtrial, test)
    Xtest = np.take_along_axis(Xtrial, idx, -1)[..., 0]
    Xtrain = Xtrial.copy()
    np.put_along_axis(Xtrain, idx, np.nan, -1)
    return Xtest, Xtrain


def _split_with_noise(Xfull, Xtrial, nt, test, sumsq, simultaneous, noise_cov_type):
    """Train/test split of one cross-validation repetition and the noise covariance of the
    training trials (0 for noise_cov_type 'none'): ``(Xtest, Xtrain, Cnoise_train)``.

    For non-simultaneous recordings the training trials' sum of squares is the full one
    (``sumsq``, computed once for all repetitions) minus the test trial's, so the trial array
    need not be copied and summed again for every repetition."""
    ntc = _per_condition(nt, Xfull.ndim)
    if sumsq is not None and noise_cov_type != "none":
        Xtest = np.take_along_axis(Xtrial, _test_index(Xtrial, test), -1)[..., 0]
        if not np.isnan(Xtest).any():               # NaNs inside valid trials: general route
            Xtrain = (Xfull * ntc - Xtest) / (ntc - 1)
            return Xtest, Xtrain, _noise_cov_from_sumsq(sumsq - Xtest ** 2, Xtrain, nt - 1, noise_cov_type)
    Xtest, Xtrain_full = _apply_test_trials(Xtrial, test)
    Xtrain = (Xfull * ntc - Xtest) / (ntc - 1)
    Cn = 0.0 if noise_cov_type == "none" else dpca_get_noise_covariance(
        Xtrain, Xtrain_full, nt - 1, simultaneous=simultaneous, type=noise_cov_type)
    return Xtest, Xtrain, Cn


def _map(func, arg_list, n_jobs):
    """[func(*args) for args in arg_list], in parallel processes when n_jobs != 1 (joblib; -1 =
    all cores). Large arrays are shared with the workers via memory mapping, and each worker's
    BLAS is limited to its share of the cores (otherwise n_jobs x n_cores threads would compete)."""
    if n_jobs is None or n_jobs == 1 or len(arg_list) <= 1:
        return [func(*args) for args in arg_list]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs)(delayed(func)(*args) for args in arg_list)


# ============================================================================ dpca_optimizeLambda.m
def _sumsq_for_splits(Xtrial, simultaneous, noise_cov_type):
    """Trial sum of squares shared by all splits (see ``_split_with_noise``), or None."""
    return None if simultaneous or noise_cov_type == "none" else _trial_sumsq(Xtrial)


def _optimize_lambda_rep(rep, num_rep, Xfull, Xtrial, nt, test, sumsq, lambdas, num_comps, combined_params, method,
                         simultaneous, noise_cov_type, verbose):
    """One cross-validation repetition of :func:`dpca_optimize_lambda`: errors (n_lambdas,) and
    errors per marginalization (n_marg, n_lambdas)."""
    if verbose:
        print("Iteration #%d out of %d" % (rep + 1, num_rep))
    Xtest, Xtrain, Cn = _split_with_noise(Xfull, Xtrial, nt, test, sumsq, simultaneous, noise_cov_type)

    Xtest_c = _center(Xtest)
    test_margs, _ = dpca_marginalize(Xtest_c, combined_params=combined_params, if_flat=True)
    # the training marginalizations are exactly those dpca(Xtrain) works with
    X, total_var, train_margs, marg_nums = _dpca_prepare(Xtrain, combined_params)
    targets = train_margs if method == "training" else test_margs
    norm = np.array([np.sum(m ** 2) for m in targets])
    Xt = _flat(Xtest_c)
    ridge = _Ridge(X, train_margs, Cn, total_var, spectral=True)     # shared by all lambdas

    errors = np.zeros(len(lambdas))
    errors_marg = np.zeros((len(test_margs), len(lambdas)))
    for l, lam in enumerate(lambdas):
        W, V, which = _dpca_fit(X, total_var, train_margs, marg_nums, ridge, num_comps, lam)
        cum = 0.0
        for i in range(len(test_margs)):
            sel = which == i + 1
            recon = V[:, sel] @ (W[:, sel].T @ Xt)
            if method == "naive":
                err = np.sum((test_margs[i] - recon) ** 2)
            elif method == "training":
                err = np.sum((train_margs[i] - recon) ** 2)
            else:  # neuronwise: predict each neuron without using that neuron's own test activity
                err = np.sum((test_margs[i] - recon + np.sum(V[:, sel] * W[:, sel], 1)[:, None] * Xt) ** 2)
            errors_marg[i, l] = err / norm[i]
            cum += err
        errors[l] = cum / norm.sum()
    return errors, errors_marg


def dpca_optimize_lambda(Xfull, Xtrial, num_trials, num_comps=25, lambdas=None, num_rep=10, combined_params=None,
                         method="training", simultaneous=False, noise_cov_type="pooled", rng=None, verbose=False,
                         n_jobs=1):
    """Cross-validated choice of the regularization parameter (MATLAB ``dpca_optimizeLambda``).

    ``method``: 'training' (default), 'neuronwise' or 'naive' (deprecated).
    ``noise_cov_type``: 'pooled' (default), 'averaged' or 'none'.
    ``n_jobs``: number of parallel processes for the repetitions (-1 = all cores). The results
    do not depend on it.

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
    if method not in ("training", "neuronwise", "naive"):
        raise ValueError("method must be 'training', 'neuronwise' or 'naive'")

    tests = [_draw_test_trials(nt, rng, simultaneous) for _ in range(num_rep)]
    sumsq = _sumsq_for_splits(Xtrial, simultaneous, noise_cov_type)
    out = _map(_optimize_lambda_rep, [(rep, num_rep, Xfull, Xtrial, nt, test, sumsq, lambdas, num_comps,
                                       combined_params, method, simultaneous, noise_cov_type, verbose)
                                      for rep, test in enumerate(tests)], n_jobs)
    errors = np.stack([e for e, _ in out], -1)
    errors_marg = np.stack([em for _, em in out], -1)

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


def _classification_rep(rep, num_rep, Xfull, Xtrial, nt, test, sumsq, Vfull, comps_to_use, fit_kw,
                        decoding_classes, time_comp, simultaneous, noise_cov_type, verbose):
    """One repetition of :func:`dpca_classification_accuracy`: (correct, brier), each
    (n1, n2, T), as numbers of conditions (summed, not yet averaged)."""
    if verbose:
        print("Repetition # %d out of %d" % (rep + 1, num_rep))
    dim = Xfull.shape
    Xtest, Xtrain, Cn_train = _split_with_noise(Xfull, Xtrial, nt, test, sumsq, simultaneous, noise_cov_type)
    W, V, which = dpca(Xtrain, comps_to_use, Cnoise=Cn_train, **fit_kw)
    margs = sorted(set(which.tolist()) - set(time_comp))

    n1 = max(set(which.tolist()) | set(time_comp))
    n2 = max(int(np.sum(which == m)) for m in set(which.tolist()))
    correct = np.zeros((n1, n2, dim[-1]))
    brier = np.zeros((n1, n2, dim[-1]))

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
            # all time points at once: dist is (conditions, classes, T)
            dist = zte[:, None, :] - means[None, :, :]
            correct[m - 1, j] = np.sum(np.argmin(np.abs(dist), 1) + 1 == cls[:, None], 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                prob = 1.0 / np.abs(dist)
                prob = prob / prob.sum(1, keepdims=True)
            brier[m - 1, j] = np.sum((prob - onehot[:, :, None]) ** 2, axis=(0, 1)) / n_unique
    return correct, brier


def _classification(Xfull, Xtrial, nt, tests, num_comps, lambda_, combined_params, decoding_classes, time_splits,
                    time_parameter, not_to_split, simultaneous, noise_cov_type, verbose, n_jobs):
    """:func:`dpca_classification_accuracy` with the test trials of all repetitions given."""
    dim = Xfull.shape
    sumsq = _sumsq_for_splits(Xtrial, simultaneous, noise_cov_type)
    if sumsq is not None:
        Cnoise = _noise_cov_from_sumsq(sumsq, Xfull, nt, noise_cov_type)
    else:
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

    num_rep = len(tests)
    out = _map(_classification_rep, [(rep, num_rep, Xfull, Xtrial, nt, test, sumsq, Vfull, comps_to_use, fit_kw,
                                      decoding_classes, time_comp, simultaneous, noise_cov_type, verbose)
                                     for rep, test in enumerate(tests)], n_jobs)
    correct = np.stack([c for c, _ in out], -1)
    brier = np.stack([b for _, b in out], -1)

    n_cond = int(np.prod(dim[1:-1]))
    accuracy = correct.sum(3) / (num_rep * n_cond)
    brier = brier.sum(3) / (num_rep * n_cond)
    for tc in time_comp:
        accuracy[tc - 1] = np.nan
        brier[tc - 1] = np.nan
    return accuracy, brier


def dpca_classification_accuracy(Xfull, Xtrial, num_trials, num_comps=3, lambda_=0, num_rep=100,
                                 combined_params=None, decoding_classes=None, time_splits=None, time_parameter=None,
                                 not_to_split=None, simultaneous=False, noise_cov_type="pooled", rng=None,
                                 verbose=False, n_jobs=1):
    """Cross-validated classification accuracy of dPCA components at each time point (MATLAB
    ``dpca_classificationAccuracy``).

    ``decoding_classes`` is a list with one entry per marginalization: an array shaped like the
    condition dimensions (d1, ..., d_{D-1}) holding 1-based class ids (None for time). The
    default is one class per condition. ``n_jobs``: number of parallel processes for the
    repetitions (-1 = all cores); the results do not depend on it.

    Returns ``(accuracy, brier)``, each (n_marg, num_comps, T); the time marginalization is NaN.
    """
    rng = _rng(rng)
    nt = np.asarray(num_trials)
    tests = [_draw_test_trials(nt, rng, simultaneous) for _ in range(num_rep)]
    return _classification(Xfull, Xtrial, nt, tests, num_comps, lambda_, combined_params, decoding_classes,
                           time_splits, time_parameter, not_to_split, simultaneous, noise_cov_type, verbose, n_jobs)


# ============================================================================ dpca_classificationShuffled.m
def dpca_shuffle_trials(Xtrial, num_trials, rng=None, simultaneous=False):
    """Shuffle whole trials across conditions **within each neuron** (or jointly for all neurons
    when ``simultaneous``), preserving each condition's number of trials. As in MATLAB, unused
    trial slots of the returned array are 0 (not NaN)."""
    perms = _draw_shuffle(num_trials, Xtrial.shape[-1], _rng(rng), simultaneous)
    return _apply_shuffle(Xtrial, num_trials, perms, simultaneous)


def _trial_slots(nt, max_tr):
    """(N, conditions*trials) mask of the slots holding a trial; slot k <-> (condition
    k % n_cond, trial k // n_cond)."""
    nt_cond = _flat(nt)                                             # (N, conditions)
    present = np.arange(max_tr)[None, None, :] < nt_cond[:, :, None]        # (N, cond, trials)
    return np.reshape(present, (nt.shape[0], -1), order="F")


def _draw_shuffle(num_trials, max_tr, rng, simultaneous):
    """The random part of :func:`dpca_shuffle_trials`: one 0-based permutation of the trial
    slots per neuron (a single one if ``simultaneous``)."""
    present = _trial_slots(np.asarray(num_trials), max_tr)
    rows = range(present.shape[0]) if not simultaneous else [0]
    return [randperm(rng, int(present[n].sum())) - 1 for n in rows]


def _apply_shuffle(Xtrial, num_trials, perms, simultaneous):
    nt = np.asarray(num_trials)
    N, T = nt.shape[0], Xtrial.shape[-2]
    order = list(range(Xtrial.ndim))
    order[-2], order[-1] = order[-1], order[-2]
    Xc = np.reshape(np.transpose(Xtrial, order), (N, -1, T), order="F")   # (N, conditions*trials, T)
    present = _trial_slots(nt, Xtrial.shape[-1])
    out = np.zeros(Xc.shape)
    if not simultaneous:
        for n in range(N):
            slots = np.nonzero(present[n])[0]
            out[n, slots] = Xc[n, slots[perms[n]]]
    else:
        slots = np.nonzero(present[0])[0]
        out[:, slots] = Xc[:, slots[perms[0]]]
    shape_t = [Xtrial.shape[o] for o in order]
    return np.transpose(np.reshape(out, shape_t, order="F"), order)


def _shuffled_one(s, num_shuffles, Xtrial, nt, perms, tests, opts, verbose):
    """One shuffle of :func:`dpca_classification_shuffled`."""
    if verbose:
        print("Shuffle #%d out of %d" % (s + 1, num_shuffles))
    Xs = _apply_shuffle(Xtrial, nt, perms, opts["simultaneous"])
    Xavg = Xs.sum(-1) / _per_condition(nt, Xs.ndim - 1)
    acc, br = _classification(Xavg, Xs, nt, tests, num_comps=1, verbose=False, n_jobs=1, **opts)
    return acc[:, 0, :], br[:, 0, :]


def dpca_classification_shuffled(Xtrial, num_trials, num_shuffles=100, lambda_=0, num_rep=100, combined_params=None,
                                 decoding_classes=None, time_splits=None, time_parameter=None, not_to_split=None,
                                 simultaneous=False, noise_cov_type="pooled", rng=None, verbose=False, n_jobs=1):
    """Null distribution of classification accuracy under shuffled condition labels (MATLAB
    ``dpca_classificationShuffled``). Returns ``(accuracy_shuffle, brier_shuffle)``, each
    (n_marg, T, num_shuffles), for the first component of each marginalization. ``n_jobs``:
    number of parallel processes for the shuffles (-1 = all cores); the results do not depend
    on it."""
    rng = _rng(rng)
    nt = np.asarray(num_trials)
    # all random numbers in the serial order: shuffle s, then the num_rep splits of shuffle s
    draws = []
    for _ in range(num_shuffles):
        perms = _draw_shuffle(nt, Xtrial.shape[-1], rng, simultaneous)
        draws.append((perms, [_draw_test_trials(nt, rng, simultaneous) for _ in range(num_rep)]))
    opts = dict(lambda_=lambda_, combined_params=combined_params, decoding_classes=decoding_classes,
                time_splits=time_splits, time_parameter=time_parameter, not_to_split=not_to_split,
                simultaneous=simultaneous, noise_cov_type=noise_cov_type)
    out = _map(_shuffled_one, [(s, num_shuffles, Xtrial, nt, perms, tests, opts, verbose)
                               for s, (perms, tests) in enumerate(draws)], n_jobs)
    return np.stack([a for a, _ in out], -1), np.stack([b for _, b in out], -1)


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
