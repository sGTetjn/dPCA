"""1:1 parity of dPCA.core (functional MATLAB port) with the literal MATLAB transliteration.

Every stochastic routine is driven by identically seeded generators on both sides, so results
must agree to numerical precision, including signs, component order and tie-breaking.
"""
import numpy as np
import pytest

import matlab_oracle as M
from conftest import COMBINED3, make_data

core = pytest.importorskip("dPCA.core")

RTOL, ATOL = 1e-7, 1e-9


def close(a, b, rtol=RTOL, atol=ATOL):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, (a.shape, b.shape)
    scale = max(1.0, np.nanmax(np.abs(b))) if b.size else 1.0
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol * scale, equal_nan=True)


def rngs(seed=123):
    return np.random.default_rng(seed), np.random.default_rng(seed)


# ------------------------------------------------------------------------------ marginalization
@pytest.mark.parametrize("conds", [(4,), (3, 2), (3, 2, 2)])
@pytest.mark.parametrize("flat", [False, True])
def test_marginalize_all_subsets(conds, flat):
    d = make_data(conds=conds, T=6, N=11, seed=5)
    got, gn = core.dpca_marginalize(d["Xfull"], if_flat=flat)
    ref, rn = M.dpca_marginalize(d["Xfull"], ifFlat="yes" if flat else "no")
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        close(g, r)
    np.testing.assert_array_equal(gn, rn)


def test_marginalize_combined(data3):
    got, gn = core.dpca_marginalize(data3["Xfull"], combined_params=COMBINED3)
    ref, rn = M.dpca_marginalize(data3["Xfull"], combinedParams=COMBINED3)
    for g, r in zip(got, ref):
        close(g, r)
    np.testing.assert_array_equal(gn, rn)


def test_marginalize_not_full(data3):
    got, _ = core.dpca_marginalize(data3["Xfull"], if_full=False)
    ref, _ = M.dpca_marginalize(data3["Xfull"], ifFull="no")
    for g, r in zip(got, ref):
        close(g, r)


@pytest.mark.parametrize("combined", [None, COMBINED3])
@pytest.mark.parametrize("not_to_split", [None, [[[3]]]])
def test_marginalize_time_splits(data3, combined, not_to_split):
    kw = dict(timeSplits=[5, 11], timeParameter=3, notToSplit=not_to_split)
    got, gn = core.dpca_marginalize(data3["Xfull"], combined_params=combined, time_splits=[5, 11],
                                    time_parameter=3, not_to_split=not_to_split, if_flat=True)
    ref, rn = M.dpca_marginalize(data3["Xfull"], combinedParams=combined, ifFlat="yes", **kw)
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        close(g, r)
    np.testing.assert_array_equal(gn, rn)


# ------------------------------------------------------------------------------ dpca
def check_dpca(X, num_comps, **kw):
    mkw = {{"combined_params": "combinedParams", "time_splits": "timeSplits",
            "time_parameter": "timeParameter", "not_to_split": "notToSplit"}.get(k, k): v for k, v in kw.items()}
    if "order" in mkw:
        mkw["order"] = "yes" if mkw["order"] else "no"
    if "scale" in mkw:
        mkw["scale"] = "yes" if mkw["scale"] else "no"
    W, V, wm = core.dpca(X, num_comps, **kw)
    Wr, Vr, wmr = M.dpca(X, num_comps, **mkw)
    np.testing.assert_array_equal(wm, wmr)
    close(W, Wr)
    close(V, Vr)


@pytest.mark.parametrize("num_comps", [3, 7, [3, 2, 3]])
@pytest.mark.parametrize("lam", [0, 1e-5, 1e-3, 1e-2])
def test_dpca_2param(data2, num_comps, lam):
    check_dpca(data2["Xfull"], num_comps, lambda_=lam)


@pytest.mark.parametrize("num_comps", [5, [2, 1, 3, 3]])
@pytest.mark.parametrize("lam", [0, 1e-4, [1e-5, 1e-4, 1e-3, 1e-2]])
def test_dpca_3param_combined(data3, num_comps, lam):
    check_dpca(data3["Xfull"], num_comps, combined_params=COMBINED3, lambda_=lam)


def test_dpca_cnoise(data3):
    C = M.dpca_getNoiseCovariance(data3["Xfull"], data3["Xtrial"], data3["numOfTrials"])
    check_dpca(data3["Xfull"], [2, 1, 3, 3], combined_params=COMBINED3, lambda_=1e-4, Cnoise=C)
    check_dpca(data3["Xfull"], 6, combined_params=COMBINED3, Cnoise=C)


def test_dpca_order_no_and_scale(data3):
    check_dpca(data3["Xfull"], [2, 1, 3, 3], combined_params=COMBINED3, order=False)
    check_dpca(data3["Xfull"], [2, 1, 3, 3], combined_params=COMBINED3, lambda_=1e-4, scale=True)


def test_dpca_singular_fallback_succeeds():
    """N > conditions*time: X*X' is singular; MATLAB retries with lambda = 1e-10, which is enough
    when the total variance is large (the ridge (totalVar*1e-10)^2 then dominates eps*||XX'||)."""
    d = make_data(N=101, conds=(3,), T=12, seed=9)
    check_dpca(d["Xfull"] * 1e5, [2, 3, 3])   # retried system well conditioned (rcond ~1e-6)


def test_dpca_singular_fallback_fails_like_matlab():
    """If the matrix is still near-singular after the retry, MATLAB raises (warnings are errors)."""
    d = make_data(N=101, conds=(3,), T=12, seed=9)
    with pytest.raises(np.linalg.LinAlgError):
        M.dpca(d["Xfull"], [2, 3, 3])
    with pytest.raises(np.linalg.LinAlgError):
        core.dpca(d["Xfull"], [2, 3, 3])


def test_dpca_time_splits(data3):
    check_dpca(data3["Xfull"], [2, 1, 2, 2], combined_params=COMBINED3, time_splits=[8], time_parameter=3,
               not_to_split=[[[3]]])
    check_dpca(data3["Xfull"], 6, combined_params=COMBINED3, time_splits=[8], time_parameter=3)


# ------------------------------------------------------------------------------ explained variance
def test_explained_variance_basic(data3):
    W, V, _ = M.dpca(data3["Xfull"], 8, combinedParams=COMBINED3)
    got = core.dpca_explained_variance(data3["Xfull"], W, V, combined_params=COMBINED3)
    ref = M.dpca_explainedVariance(data3["Xfull"], W, V, combinedParams=COMBINED3)
    assert set(got) == set(ref)
    for k in ref:
        close(got[k], ref[k])


def test_explained_variance_pca_axes_drop_cumulative_dpca(data2):
    X = data2["Xfull"].reshape(31, -1, order="F")
    X = X - X.mean(1, keepdims=True)
    U = np.linalg.svd(X, full_matrices=False)[0][:, :4]
    got = core.dpca_explained_variance(data2["Xfull"], U, U)
    assert np.size(got["cumulativeDPCA"]) == 0


def test_explained_variance_signal_old_and_new(data3):
    W, V, _ = M.dpca(data3["Xfull"], 8, combinedParams=COMBINED3)
    C = M.dpca_getNoiseCovariance(data3["Xfull"], data3["Xtrial"], data3["numOfTrials"])
    r1, r2 = rngs(7)
    got = core.dpca_explained_variance(data3["Xfull"], W, V, combined_params=COMBINED3, X_trial=data3["Xtrial"],
                                       num_trials=data3["numOfTrials"], rng=r1)
    ref = M.dpca_explainedVariance(data3["Xfull"], W, V, combinedParams=COMBINED3, X_trial=data3["Xtrial"],
                                   numOfTrials=data3["numOfTrials"], rng=r2)
    for k in ref:
        close(got[k], ref[k])
    got = core.dpca_explained_variance(data3["Xfull"], W, V, combined_params=COMBINED3, Cnoise=C,
                                       num_trials=data3["numOfTrials"])
    ref = M.dpca_explainedVariance(data3["Xfull"], W, V, combinedParams=COMBINED3, Cnoise=C,
                                   numOfTrials=data3["numOfTrials"])
    for k in ref:
        close(got[k], ref[k])


# ------------------------------------------------------------------------------ noise covariance
@pytest.mark.parametrize("kind", ["pooled", "averaged"])
def test_noise_covariance(data3, kind):
    got = core.dpca_get_noise_covariance(data3["Xfull"], data3["Xtrial"], data3["numOfTrials"], type=kind)
    ref = M.dpca_getNoiseCovariance(data3["Xfull"], data3["Xtrial"], data3["numOfTrials"], type=kind)
    close(got, ref)


def test_noise_covariance_simultaneous(data3_sim):
    d = data3_sim
    got = core.dpca_get_noise_covariance(d["Xfull"], d["Xtrial"], d["numOfTrials"], simultaneous=True)
    ref = M.dpca_getNoiseCovariance(d["Xfull"], d["Xtrial"], d["numOfTrials"], simultaneous=True)
    close(got, ref)
    with pytest.raises(Exception):
        core.dpca_get_noise_covariance(d["Xfull"], d["Xtrial"], d["numOfTrials"], simultaneous=True,
                                       type="averaged")


# ------------------------------------------------------------------------------ test/train split
@pytest.mark.parametrize("sim", [False, True])
def test_get_test_trials(data3, data3_sim, sim):
    d = data3_sim if sim else data3
    r1, r2 = rngs(11)
    for _ in range(3):
        a, b = core.dpca_get_test_trials(d["Xtrial"], d["numOfTrials"], rng=r1, simultaneous=sim)
        c, e = M.dpca_getTestTrials(d["Xtrial"], d["numOfTrials"], r2, simultaneous=sim)
        close(a, c)
        close(b, e)


# ------------------------------------------------------------------------------ lambda optimisation
@pytest.mark.parametrize("method", ["naive", "training", "neuronwise"])
@pytest.mark.parametrize("noise", ["pooled", "averaged", "none"])
def test_optimize_lambda(data2, method, noise):
    lams = 1e-7 * 1.5 ** np.arange(0, 26, 3)
    r1, r2 = rngs(21)
    got = core.dpca_optimize_lambda(data2["Xfull"], data2["Xtrial"], data2["numOfTrials"], num_comps=6,
                                    lambdas=lams, num_rep=3, method=method, noise_cov_type=noise, rng=r1)
    ref = M.dpca_optimizeLambda(data2["Xfull"], data2["Xtrial"], data2["numOfTrials"], r2, numComps=6,
                                lambdas=lams, numRep=3, method=method, noiseCovType=noise)
    for g, r in zip(got, ref):
        close(g, r)


def test_optimize_lambda_combined_simultaneous(data3_sim):
    d = data3_sim
    lams = 1e-7 * 1.5 ** np.arange(0, 26, 5)
    r1, r2 = rngs(22)
    got = core.dpca_optimize_lambda(d["Xfull"], d["Xtrial"], d["numOfTrials"], num_comps=[2, 1, 3, 3],
                                    lambdas=lams, num_rep=2, combined_params=COMBINED3, simultaneous=True, rng=r1)
    ref = M.dpca_optimizeLambda(d["Xfull"], d["Xtrial"], d["numOfTrials"], r2, numComps=[2, 1, 3, 3],
                                lambdas=lams, numRep=2, combinedParams=COMBINED3, simultaneous=True)
    for g, r in zip(got, ref):
        close(g, r)


def test_optimize_lambda_default_grid(data2):
    r1, r2 = rngs(23)
    got = core.dpca_optimize_lambda(data2["Xfull"], data2["Xtrial"], data2["numOfTrials"], num_comps=6,
                                    num_rep=1, rng=r1)
    ref = M.dpca_optimizeLambda(data2["Xfull"], data2["Xtrial"], data2["numOfTrials"], r2, numComps=6, numRep=1)
    for g, r in zip(got, ref):
        close(g, r)


def test_optimize_lambda_too_few_trials(data2):
    ntr = data2["numOfTrials"].copy()
    ntr[0, 0] = 1
    with pytest.raises(ValueError):
        core.dpca_optimize_lambda(data2["Xfull"], data2["Xtrial"], ntr, num_rep=1)


# ------------------------------------------------------------------------------ classification
def test_classification_accuracy_default_classes(data2):
    """2-parameter data with default classes: MATLAB's reshape(1:S, S) errors here, so this tests
    the documented extension (one class per condition), not MATLAB itself. The MATLAB-defined
    default (>= 2 condition dims) is covered by test_classification_3param_default_classes."""
    r1, r2 = rngs(31)
    got = core.dpca_classification_accuracy(data2["Xfull"], data2["Xtrial"], data2["numOfTrials"], num_rep=4,
                                            lambda_=1e-4, rng=r1)
    ref = M.dpca_classificationAccuracy(data2["Xfull"], data2["Xtrial"], data2["numOfTrials"], r2, numRep=4,
                                        lambda_=1e-4)
    for g, r in zip(got, ref):
        close(g, r)


def demo_classes(S):
    """decodingClasses from dpca_demo.m for (stimulus, decision, time)."""
    return [np.c_[np.arange(1, S + 1), np.arange(1, S + 1)], np.tile([1, 2], (S, 1)), None,
            np.c_[np.arange(1, S + 1), S + np.arange(1, S + 1)]]


@pytest.mark.parametrize("sim", [False, True])
def test_classification_accuracy_demo_classes(data3, data3_sim, sim):
    d = data3_sim if sim else data3
    r1, r2 = rngs(32)
    got = core.dpca_classification_accuracy(d["Xfull"], d["Xtrial"], d["numOfTrials"], num_rep=3, lambda_=1e-4,
                                            combined_params=COMBINED3, decoding_classes=demo_classes(3),
                                            simultaneous=sim, rng=r1)
    ref = M.dpca_classificationAccuracy(d["Xfull"], d["Xtrial"], d["numOfTrials"], r2, numRep=3, lambda_=1e-4,
                                        combinedParams=COMBINED3, decodingClasses=demo_classes(3),
                                        simultaneous=sim)
    for g, r in zip(got, ref):
        close(g, r)


@pytest.mark.parametrize("sim", [False, True])
def test_classification_shuffled(data3, data3_sim, sim):
    d = data3_sim if sim else data3
    r1, r2 = rngs(33)
    got = core.dpca_classification_shuffled(d["Xtrial"], d["numOfTrials"], num_shuffles=3, num_rep=2,
                                            lambda_=1e-4, combined_params=COMBINED3,
                                            decoding_classes=demo_classes(3), simultaneous=sim, rng=r1)
    ref = M.dpca_classificationShuffled(d["Xtrial"], d["numOfTrials"], r2, numShuffles=3, numRep=2, lambda_=1e-4,
                                        combinedParams=COMBINED3, decodingClasses=demo_classes(3),
                                        simultaneous=sim)
    for g, r in zip(got, ref):
        close(g, r)


@pytest.mark.parametrize("min_chunk", [1, 2, 3, 5, 10])
def test_signif_components(min_chunk):
    rng = np.random.default_rng(min_chunk)
    acc = rng.random((4, 3, 40))
    acc[2] = np.nan                                         # time marginalization
    shuf = rng.random((4, 40, 6)) * 0.9
    wm = np.array([1, 4, 2, 1, 4, 1, 2, 3, 3, 1, 2, 4, 4, 1])
    got = core.dpca_signif_components(acc, shuf, wm, min_chunk=min_chunk)
    ref = M.dpca_signifComponents(acc, shuf, wm, minChunk=min_chunk)
    close(got, ref)


# ------------------------------------------------------------------------------ additional coverage
def test_explained_variance_near_pca_axes_keeps_cumulative_dpca(data2):
    """cumulativeDPCA is dropped only when it equals PCA to 1e-10, not for nearby axes."""
    X = data2["Xfull"].reshape(31, -1, order="F")
    X = X - X.mean(1, keepdims=True)
    U = np.linalg.svd(X, full_matrices=False)[0][:, :4]
    U = U + 1e-6 * np.random.default_rng(0).normal(size=U.shape)
    got = core.dpca_explained_variance(data2["Xfull"], U, U)
    ref = M.dpca_explainedVariance(data2["Xfull"], U, U)
    assert np.size(ref["cumulativeDPCA"]) == 4
    for k in ref:
        close(got[k], ref[k])


def test_noise_covariance_with_empty_condition(data3):
    d = data3
    nt = d["numOfTrials"].copy(); nt[0, 0, 0] = 0
    Xt = d["Xtrial"].copy(); Xt[0, 0, 0] = np.nan
    Xf = np.nanmean(Xt, -1)
    for kind in ("pooled", "averaged"):
        close(core.dpca_get_noise_covariance(Xf, Xt, nt, type=kind), M.dpca_getNoiseCovariance(Xf, Xt, nt, type=kind))


def test_classification_component_reordering_is_exercised():
    """Noisy data where the per-split components are re-ordered to match the full-data ones
    (the order differs from the identity and argmax over rows differs from over columns)."""
    d = make_data(N=15, conds=(4,), T=8, noise=4.0, n_latent=2, max_trials=5, seed=1)
    r1, r2 = rngs(1)
    got = core.dpca_classification_accuracy(d["Xfull"], d["Xtrial"], d["numOfTrials"], num_comps=3, num_rep=4,
                                            lambda_=1e-4, rng=r1)
    ref = M.dpca_classificationAccuracy(d["Xfull"], d["Xtrial"], d["numOfTrials"], r2, numComps=3, numRep=4,
                                        lambda_=1e-4)
    for g, r in zip(got, ref):
        close(g, r)


def test_classification_noncontiguous_classes(data3):
    """Class ids {1,3,5}: accumarray fills absent classes with 0; Brier divides by #unique classes."""
    cls = [np.array([[1, 1], [3, 3], [5, 5]])] * 4
    r1, r2 = rngs(4)
    got = core.dpca_classification_accuracy(data3["Xfull"], data3["Xtrial"], data3["numOfTrials"], num_comps=2,
                                            num_rep=3, lambda_=1e-4, combined_params=COMBINED3, decoding_classes=cls,
                                            rng=r1)
    ref = M.dpca_classificationAccuracy(data3["Xfull"], data3["Xtrial"], data3["numOfTrials"], r2, numComps=2,
                                        numRep=3, lambda_=1e-4, combinedParams=COMBINED3, decodingClasses=cls)
    for g, r in zip(got, ref):
        close(g, r)


@pytest.mark.parametrize("num_comps", [2, [2, 1, 3, 2]])
def test_classification_3param_default_classes(data3, num_comps):
    """Default decoding classes for >= 2 condition dimensions (well defined in MATLAB);
    a vector numComps is tiled with repmat as in MATLAB."""
    r1, r2 = rngs(5)
    got = core.dpca_classification_accuracy(data3["Xfull"], data3["Xtrial"], data3["numOfTrials"],
                                            num_comps=num_comps, num_rep=3, lambda_=1e-4,
                                            combined_params=COMBINED3, rng=r1)
    ref = M.dpca_classificationAccuracy(data3["Xfull"], data3["Xtrial"], data3["numOfTrials"], r2,
                                        numComps=num_comps, numRep=3, lambda_=1e-4, combinedParams=COMBINED3)
    for g, r in zip(got, ref):
        close(g, r)


def test_classification_time_splits(data3):
    kw = dict(time_splits=[8], time_parameter=3, not_to_split=[[[3]]])
    r1, r2 = rngs(6)
    got = core.dpca_classification_accuracy(data3["Xfull"], data3["Xtrial"], data3["numOfTrials"], num_comps=2,
                                            num_rep=3, lambda_=1e-4, combined_params=COMBINED3,
                                            decoding_classes=demo_classes(3), rng=r1, **kw)
    ref = M.dpca_classificationAccuracy(data3["Xfull"], data3["Xtrial"], data3["numOfTrials"], r2, numComps=2,
                                        numRep=3, lambda_=1e-4, combinedParams=COMBINED3,
                                        decodingClasses=demo_classes(3), timeSplits=[8], timeParameter=3,
                                        notToSplit=[[[3]]])
    for g, r in zip(got, ref):
        close(g, r)


def test_classification_4params():
    d = make_data(N=41, conds=(3, 2, 2), T=8, seed=12)
    r1, r2 = rngs(7)
    got = core.dpca_classification_accuracy(d["Xfull"], d["Xtrial"], d["numOfTrials"], num_comps=1, num_rep=2,
                                            lambda_=1e-4, rng=r1)
    ref = M.dpca_classificationAccuracy(d["Xfull"], d["Xtrial"], d["numOfTrials"], r2, numComps=1, numRep=2,
                                        lambda_=1e-4)
    for g, r in zip(got, ref):
        close(g, r)


def test_dpca_4params():
    d = make_data(N=41, conds=(3, 2, 2), T=8, seed=12)
    check_dpca(d["Xfull"], 1, lambda_=1e-4)
    check_dpca(d["Xfull"], 6, lambda_=1e-4)
