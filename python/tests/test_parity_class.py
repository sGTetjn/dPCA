"""Parity of the sklearn-style class API (dPCA.dPCA.dPCA) with the MATLAB reference.

Mapping used throughout (independent of the package internals):
  * labels 'st' / 'sdt' -> MATLAB parameters 1..D (last label = time)
  * an unjoined key such as 'sd' is the MATLAB marginalization over parameter subset {1,2}
  * n_components=k (int) corresponds to MATLAB numComps = [k k ... k] (k per marginalization),
    with components ordered by explained variance inside each marginalization ('order','yes')
"""
import contextlib
import io

import numpy as np
import pytest

import matlab_oracle as M
from conftest import make_data
from dPCA.dPCA import dPCA

RTOL, ATOL = 1e-7, 1e-9


def close(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, (a.shape, b.shape)
    np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL * max(1.0, np.nanmax(np.abs(b))), equal_nan=True)


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def key_of(subset, labels):
    return "".join(labels[p - 1] for p in sorted(subset))


def oracle_groups(labels, join):
    """MATLAB combinedParams + key per marginalization for a labels/join specification."""
    D = len(labels)
    if join is None:
        subs = M.subsets(range(1, D + 1))
        return None, [key_of(s, labels) for s in subs]
    groups, keys = [], []
    joined = {c for combs in join.values() for c in combs}
    for s in M.subsets(range(1, D + 1)):
        k = key_of(s, labels)
        if k not in joined:
            groups.append([s]); keys.append(k)
    for k, combs in join.items():
        groups.append([[labels.index(ch) + 1 for ch in c] for c in combs]); keys.append(k)
    return groups, keys


def split_by_key(W, V, wm, keys):
    return ({k: V[:, wm == i + 1] for i, k in enumerate(keys)},
            {k: W[:, wm == i + 1] for i, k in enumerate(keys)})


CASES = [
    ("st", None, dict(N=31, conds=(4,), T=20, seed=1)),
    ("sdt", None, dict(N=31, conds=(3, 3), T=15, seed=2)),
    ("sdt", {"s": ["s", "st"], "d": ["d", "dt"], "sd": ["sd", "sdt"]}, dict(N=31, conds=(3, 3), T=15, seed=2)),
]


@pytest.mark.parametrize("labels,join,dkw", CASES)
@pytest.mark.parametrize("reg", [None, 1e-4, 1e-2])
def test_fit_matches_matlab(labels, join, dkw, reg):
    d = make_data(**dkw)
    nc = 2
    groups, keys = oracle_groups(labels, join)
    m = quiet(dPCA, labels=labels, join=join, n_components=nc, regularizer=reg)
    quiet(m.fit, d["Xfull"])
    W, V, wm = M.dpca(d["Xfull"], [nc] * len(keys), combinedParams=groups, lambda_=0 if reg is None else reg)
    P, D = split_by_key(W, V, wm, keys)
    assert set(m.P) == set(keys)
    for k in keys:
        close(m.P[k], P[k])
        close(m.D[k], D[k])


def test_fit_with_cnoise_matches_matlab():
    d = make_data(N=31, conds=(4,), T=20, seed=4)
    C = M.dpca_getNoiseCovariance(d["Xfull"], d["Xtrial"], d["numOfTrials"])
    m = quiet(dPCA, labels="st", n_components=3, regularizer=1e-4)
    quiet(m.fit, d["Xfull"], Cnoise=C)
    W, V, wm = M.dpca(d["Xfull"], [3, 3, 3], lambda_=1e-4, Cnoise=C)
    P, D = split_by_key(W, V, wm, ["s", "t", "st"])
    for k in P:
        close(m.P[k], P[k])
        close(m.D[k], D[k])


def test_transform_and_explained_variance_match_matlab():
    d = make_data(N=31, conds=(4,), T=20, seed=5)
    m = quiet(dPCA, labels="st", n_components=3, regularizer=1e-4)
    Z = quiet(m.fit_transform, d["Xfull"])
    W, V, wm = M.dpca(d["Xfull"], [3, 3, 3], lambda_=1e-4)
    ev = M.dpca_explainedVariance(d["Xfull"], W, V)
    X = d["Xfull"] - d["Xfull"].reshape(31, -1).mean(1)[:, None, None]
    for i, k in enumerate(["s", "t", "st"]):
        sel = wm == i + 1
        close(Z[k], np.einsum("nc,nst->cst", W[:, sel], X))
        close(np.asarray(m.explained_variance_ratio_[k]) * 100, ev["margVar"][i, sel])


@pytest.mark.parametrize("cv_kw", [dict(), dict(cv_method="neuronwise", noise_cov_type="none")])
def test_auto_regularizer_matches_dpca_optimizeLambda(cv_kw):
    d = make_data(N=31, conds=(4,), T=20, seed=6)
    m = quiet(dPCA, labels="st", n_components=3, regularizer="auto", random_state=42, n_trials=3, **cv_kw)
    quiet(m.fit, d["Xfull"], d["trialX"])
    lam, _, _, _ = M.dpca_optimizeLambda(d["Xfull"], d["Xtrial"], d["numOfTrials"], np.random.default_rng(42),
                                         numComps=[3, 3, 3], numRep=3,
                                         method=cv_kw.get("cv_method", "training"),
                                         noiseCovType=cv_kw.get("noise_cov_type", "pooled"))
    assert m.regularizer == pytest.approx(lam, rel=1e-12)


@pytest.mark.parametrize("labels,join,dkw", CASES[:2])
def test_significance_analysis_matches_matlab_pipeline(labels, join, dkw):
    """masks == dpca_classificationAccuracy -> dpca_classificationShuffled -> dpca_signifComponents."""
    d = make_data(**dkw)
    groups, keys = oracle_groups(labels, join)
    lam, nc, nsh, nsp, chunk = 1e-4, 2, 3, 2, 3   # nc <= rank of every marginalization
    m = quiet(dPCA, labels=labels, join=join, n_components=nc, regularizer=lam, random_state=7)
    quiet(m.fit, d["Xfull"])
    masks = quiet(m.significance_analysis, d["Xfull"], d["trialX"], n_shuffles=nsh, n_splits=nsp,
                  n_consecutive=chunk, n_components=nc)

    rng = np.random.default_rng(7)
    acc, _ = M.dpca_classificationAccuracy(d["Xfull"], d["Xtrial"], d["numOfTrials"], rng, numComps=nc,
                                           lambda_=lam, numRep=nsp, combinedParams=groups)
    shuf, _ = M.dpca_classificationShuffled(d["Xtrial"], d["numOfTrials"], rng, numShuffles=nsh, numRep=nsp,
                                            lambda_=lam, combinedParams=groups)
    W, V, wm = M.dpca(d["Xfull"], [nc] * len(keys), combinedParams=groups, lambda_=lam)
    sig = M.dpca_signifComponents(acc, shuf, wm, minChunk=chunk)
    time_key = labels[-1]
    assert set(masks) == set(keys) - {time_key}
    for i, k in enumerate(keys):
        if k == time_key:
            continue
        np.testing.assert_array_equal(np.asarray(masks[k], bool), sig[wm == i + 1].astype(bool))


def test_auto_regularizer_custom_grid_and_errors():
    d = make_data(N=31, conds=(4,), T=20, seed=6)
    lams = 1e-6 * 2.0 ** np.arange(8)
    m = quiet(dPCA, labels="st", n_components=3, regularizer="auto", random_state=3, n_trials=2, lambdas=lams)
    quiet(m.fit, d["Xfull"], d["trialX"])
    lam, _, errors, errors_marg = M.dpca_optimizeLambda(d["Xfull"], d["Xtrial"], d["numOfTrials"],
                                                        np.random.default_rng(3), numComps=[3, 3, 3], numRep=2,
                                                        lambdas=lams)
    assert m.regularizer == pytest.approx(lam, rel=1e-12)
    close(m.cv_errors_, errors)
    close(m.cv_errors_marg_, errors_marg)


def test_significance_analysis_with_join_matches_matlab_pipeline():
    d = make_data(N=31, conds=(3, 3), T=15, seed=2)
    join = {"s": ["s", "st"], "d": ["d", "dt"], "sd": ["sd", "sdt"]}
    groups, keys = oracle_groups("sdt", join)
    classes = {"s": np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]), "d": np.array([[1, 2, 3]] * 3),
               "sd": np.arange(1, 10).reshape(3, 3, order="F")}
    lam, nc, nsh, nsp, chunk = 1e-4, 2, 3, 2, 3
    m = quiet(dPCA, labels="sdt", join=join, n_components=nc, regularizer=lam, random_state=9)
    quiet(m.fit, d["Xfull"])
    masks, acc_c, shuf_c = quiet(m.significance_analysis, d["Xfull"], d["trialX"], n_shuffles=nsh, n_splits=nsp,
                                 n_consecutive=chunk, n_components=nc, decoding_classes=classes, full=True)
    cls = [classes.get(k) for k in keys]
    rng = np.random.default_rng(9)
    acc, _ = M.dpca_classificationAccuracy(d["Xfull"], d["Xtrial"], d["numOfTrials"], rng, numComps=nc,
                                           lambda_=lam, numRep=nsp, combinedParams=groups, decodingClasses=cls)
    shuf, _ = M.dpca_classificationShuffled(d["Xtrial"], d["numOfTrials"], rng, numShuffles=nsh, numRep=nsp,
                                            lambda_=lam, combinedParams=groups, decodingClasses=cls)
    W, V, wm = M.dpca(d["Xfull"], [nc] * len(keys), combinedParams=groups, lambda_=lam)
    sig = M.dpca_signifComponents(acc, shuf, wm, minChunk=chunk)
    assert set(masks) == {"s", "d", "sd"}
    for i, k in enumerate(keys):
        if k in masks:
            np.testing.assert_array_equal(masks[k], sig[wm == i + 1].astype(bool))
            close(acc_c[k], acc[i])
            close(shuf_c[k], shuf[i])
