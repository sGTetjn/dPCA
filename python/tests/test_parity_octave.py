"""1:1 parity of the Python package with the REAL MATLAB reference code (../../matlab/*.m), run
under GNU Octave via octave_bridge. Skipped when Octave is not installed.

Stochastic functions: the Python call is run first while recording its random draws; the same
draws are replayed to the MATLAB code (see octave_bridge), so results must agree exactly.
"""
import contextlib
import io

import numpy as np
import pytest

import octave_bridge as O
from octave_bridge import Cell, cell_groups, same
from conftest import COMBINED3, make_data

pytestmark = pytest.mark.skipif(O.OCTAVE is None, reason="GNU Octave not installed")

from dPCA import core  # noqa: E402
from dPCA.dPCA import dPCA  # noqa: E402


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def run_py(fn):
    """Run a Python call, recording random draws; returns (output, draws)."""
    return O.record_draws(lambda: quiet(fn))


def opts(**kw):
    """Python-friendly name/value pairs -> MATLAB argument list."""
    out = []
    for k, v in kw.items():
        if v is None:
            continue
        if k in ("combinedParams", "notToSplit"):
            v = cell_groups(v)
        elif k == "decodingClasses":
            v = Cell(None if c is None else np.asarray(c, float) for c in v)
        out += [k, v]
    return out


def demo_classes(S):
    return [np.c_[np.arange(1, S + 1), np.arange(1, S + 1)], np.tile([1, 2], (S, 1)), None,
            np.c_[np.arange(1, S + 1), S + np.arange(1, S + 1)]]


# ============================================================================ marginalization
@pytest.mark.parametrize("conds", [(4,), (3, 2), (3, 2, 2)])
def test_marginalize(conds):
    d = make_data(conds=conds, T=6, N=11, seed=5)
    got, gn = core.dpca_marginalize(d["Xfull"], if_flat=True)
    ref, rn = O.call("dpca_marginalize", d["Xfull"], "ifFlat", "yes", nargout=2)
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        same(g, r)
    same(gn, rn)


def test_marginalize_combined_not_full(data3):
    got, _ = core.dpca_marginalize(data3["Xfull"], combined_params=COMBINED3)
    (ref,) = O.call("dpca_marginalize", data3["Xfull"], *opts(combinedParams=COMBINED3))
    for g, r in zip(got, ref):
        same(g, r)
    got, _ = core.dpca_marginalize(data3["Xfull"], if_full=False)
    (ref,) = O.call("dpca_marginalize", data3["Xfull"], "ifFull", "no")
    for g, r in zip(got, ref):
        same(g, r)


@pytest.mark.parametrize("combined", [None, COMBINED3])
@pytest.mark.parametrize("not_to_split", [None, [[[3]]]])
def test_marginalize_time_splits(data3, combined, not_to_split):
    got, gn = core.dpca_marginalize(data3["Xfull"], combined_params=combined, time_splits=[5, 11],
                                    time_parameter=3, not_to_split=not_to_split, if_flat=True)
    ref, rn = O.call("dpca_marginalize", data3["Xfull"], "ifFlat", "yes", "timeSplits", [5, 11],
                     "timeParameter", 3, *opts(combinedParams=combined, notToSplit=not_to_split), nargout=2)
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        same(g, r)
    same(gn, rn)


# ============================================================================ dpca
def check_dpca(X, num_comps, lambda_=0, combined=None, Cnoise=None, order=True, scale=False, time_splits=None,
               time_parameter=None, not_to_split=None):
    W, V, wm = quiet(core.dpca, X, num_comps, combined_params=combined, lambda_=lambda_, Cnoise=Cnoise, order=order,
                     scale=scale, time_splits=time_splits, time_parameter=time_parameter, not_to_split=not_to_split)
    args = ["lambda", np.atleast_1d(np.asarray(lambda_, float)), "order", "yes" if order else "no",
            "scale", "yes" if scale else "no"]
    if Cnoise is not None:
        args += ["Cnoise", Cnoise]
    if time_splits is not None:
        args += ["timeSplits", time_splits, "timeParameter", time_parameter]
    args += opts(combinedParams=combined, notToSplit=not_to_split)
    Wr, Vr, wmr = O.call("dpca", X, np.asarray(num_comps, float), *args, nargout=3)
    same(wm, wmr)
    same(W, Wr)
    same(V, Vr)


@pytest.mark.parametrize("num_comps", [3, 7, [3, 2, 3]])
@pytest.mark.parametrize("lam", [0, 1e-5, 1e-3, 1e-2])
def test_dpca_2param(data2, num_comps, lam):
    check_dpca(data2["Xfull"], num_comps, lambda_=lam)


@pytest.mark.parametrize("num_comps", [5, [2, 1, 3, 3]])
@pytest.mark.parametrize("lam", [0, 1e-4, [1e-5, 1e-4, 1e-3, 1e-2]])
def test_dpca_3param_combined(data3, num_comps, lam):
    check_dpca(data3["Xfull"], num_comps, lambda_=lam, combined=COMBINED3)


def test_dpca_cnoise_order_scale(data3):
    C = core.dpca_get_noise_covariance(data3["Xfull"], data3["Xtrial"], data3["numOfTrials"])
    check_dpca(data3["Xfull"], [2, 1, 3, 3], lambda_=1e-4, combined=COMBINED3, Cnoise=C)
    check_dpca(data3["Xfull"], 6, combined=COMBINED3, Cnoise=C)
    check_dpca(data3["Xfull"], [2, 1, 3, 3], combined=COMBINED3, order=False)
    check_dpca(data3["Xfull"], [2, 1, 3, 3], lambda_=1e-4, combined=COMBINED3, scale=True)


def test_dpca_time_splits(data3):
    check_dpca(data3["Xfull"], [2, 1, 2, 2], combined=COMBINED3, time_splits=[8], time_parameter=3,
               not_to_split=[[[3]]])
    check_dpca(data3["Xfull"], 6, combined=COMBINED3, time_splits=[8], time_parameter=3)


def test_dpca_4params():
    d = make_data(N=41, conds=(3, 2, 2), T=8, seed=12)
    check_dpca(d["Xfull"], 1, lambda_=1e-4)
    check_dpca(d["Xfull"], 6, lambda_=1e-4)


def test_dpca_singular_fallback_succeeds():
    d = make_data(N=101, conds=(3,), T=12, seed=9)
    check_dpca(d["Xfull"] * 1e5, [2, 3, 3])   # retried system well conditioned (rcond ~1e-6)


def test_dpca_singular_fallback_fails_like_matlab():
    d = make_data(N=101, conds=(3,), T=12, seed=9)
    with pytest.raises(np.linalg.LinAlgError):
        quiet(core.dpca, d["Xfull"], [2, 3, 3])
    with pytest.raises(O.OctaveError, match="singular"):
        O.call("dpca", d["Xfull"], [2, 3, 3], nargout=3)


# ============================================================================ explained variance
def check_ev(got, ref):
    for k, v in ref.items():
        if np.size(v) == 0:
            assert np.size(got[k]) == 0, k
        else:
            same(got[k], v)
    assert set(got) == set(ref)


def test_explained_variance(data3):
    W, V, _ = core.dpca(data3["Xfull"], 8, combined_params=COMBINED3)
    got = core.dpca_explained_variance(data3["Xfull"], W, V, combined_params=COMBINED3)
    (ref,) = O.call("dpca_explainedVariance", data3["Xfull"], W, V, *opts(combinedParams=COMBINED3))
    check_ev(got, ref)


def test_explained_variance_pca_axes(data2):
    X = data2["Xfull"].reshape(31, -1, order="F")
    X = X - X.mean(1, keepdims=True)
    U = np.linalg.svd(X, full_matrices=False)[0][:, :4]
    for UU in (U, U + 1e-6 * np.random.default_rng(0).normal(size=U.shape)):
        got = core.dpca_explained_variance(data2["Xfull"], UU, UU)
        (ref,) = O.call("dpca_explainedVariance", data2["Xfull"], UU, UU)
        check_ev(got, ref)


def test_explained_variance_signal(data3):
    W, V, _ = core.dpca(data3["Xfull"], 8, combined_params=COMBINED3)
    got, draws = run_py(lambda: core.dpca_explained_variance(
        data3["Xfull"], W, V, combined_params=COMBINED3, X_trial=data3["Xtrial"], num_trials=data3["numOfTrials"],
        rng=np.random.default_rng(7)))
    (ref,) = O.call("dpca_explainedVariance", data3["Xfull"], W, V, "X_trial", data3["Xtrial"], "numOfTrials",
                    data3["numOfTrials"], *opts(combinedParams=COMBINED3), draws=draws)
    check_ev(got, ref)
    C = core.dpca_get_noise_covariance(data3["Xfull"], data3["Xtrial"], data3["numOfTrials"])
    got = core.dpca_explained_variance(data3["Xfull"], W, V, combined_params=COMBINED3, Cnoise=C,
                                       num_trials=data3["numOfTrials"])
    (ref,) = O.call("dpca_explainedVariance", data3["Xfull"], W, V, "Cnoise", C, "numOfTrials",
                    data3["numOfTrials"], *opts(combinedParams=COMBINED3))
    check_ev(got, ref)


# ============================================================================ noise covariance & splits
@pytest.mark.parametrize("kind", ["pooled", "averaged"])
def test_noise_covariance(data3, kind):
    got = core.dpca_get_noise_covariance(data3["Xfull"], data3["Xtrial"], data3["numOfTrials"], type=kind)
    (ref,) = O.call("dpca_getNoiseCovariance", data3["Xfull"], data3["Xtrial"], data3["numOfTrials"], "type", kind)
    same(got, ref)


def test_noise_covariance_empty_condition_and_simultaneous(data3, data3_sim):
    nt = data3["numOfTrials"].copy(); nt[0, 0, 0] = 0
    Xt = data3["Xtrial"].copy(); Xt[0, 0, 0] = np.nan
    with np.errstate(all="ignore"):
        Xf = np.nanmean(Xt, -1)
    for kind in ("pooled", "averaged"):
        got = core.dpca_get_noise_covariance(Xf, Xt, nt, type=kind)
        (ref,) = O.call("dpca_getNoiseCovariance", Xf, Xt, nt, "type", kind)
        same(got, ref)
    d = data3_sim
    got = core.dpca_get_noise_covariance(d["Xfull"], d["Xtrial"], d["numOfTrials"], simultaneous=True)
    (ref,) = O.call("dpca_getNoiseCovariance", d["Xfull"], d["Xtrial"], d["numOfTrials"], "simultaneous", True)
    same(got, ref)


@pytest.mark.parametrize("sim", [False, True])
def test_get_test_trials(data3, data3_sim, sim):
    d = data3_sim if sim else data3
    (a, b), draws = run_py(lambda: core.dpca_get_test_trials(d["Xtrial"], d["numOfTrials"],
                                                             rng=np.random.default_rng(11), simultaneous=sim))
    c, e = O.call("dpca_getTestTrials", d["Xtrial"], d["numOfTrials"], "simultaneous", sim, nargout=2, draws=draws)
    same(a, c)
    same(b, e)


# ============================================================================ lambda optimisation
@pytest.mark.parametrize("method", ["naive", "training", "neuronwise"])
@pytest.mark.parametrize("noise", ["pooled", "averaged", "none"])
def test_optimize_lambda(data2, method, noise):
    lams = 1e-7 * 1.5 ** np.arange(0, 26, 3)
    (lam, lams_m, _, _), draws = run_py(lambda: core.dpca_optimize_lambda(
        data2["Xfull"], data2["Xtrial"], data2["numOfTrials"], num_comps=6, lambdas=lams, num_rep=3, method=method,
        noise_cov_type=noise, rng=np.random.default_rng(21)))
    ref_lam, ref_lams = O.call("dpca_optimizeLambda", data2["Xfull"], data2["Xtrial"], data2["numOfTrials"],
                               "numComps", 6, "lambdas", lams, "numRep", 3, "method", method, "noiseCovType", noise,
                               "display", "no", nargout=2, draws=draws)
    same(lam, ref_lam)
    same(lams_m, ref_lams)


def test_optimize_lambda_errors_combined_simultaneous(data3_sim, tmp_path):
    """Also compares the full error arrays (MATLAB saves them via the 'filename' option)."""
    d = data3_sim
    lams = 1e-7 * 1.5 ** np.arange(0, 26, 5)
    fname = str(tmp_path / "lam.mat")
    (lam, lams_m, errors, errors_marg), draws = run_py(lambda: core.dpca_optimize_lambda(
        d["Xfull"], d["Xtrial"], d["numOfTrials"], num_comps=[2, 1, 3, 3], lambdas=lams, num_rep=2,
        combined_params=COMBINED3, simultaneous=True, rng=np.random.default_rng(22)))
    ref_lam, ref_lams = O.call("dpca_optimizeLambda", d["Xfull"], d["Xtrial"], d["numOfTrials"],
                               "numComps", [2, 1, 3, 3], "lambdas", lams, "numRep", 2, "simultaneous", True,
                               "display", "no", "filename", fname, *opts(combinedParams=COMBINED3), nargout=2,
                               draws=draws)
    import scipy.io
    saved = scipy.io.loadmat(fname)
    same(lam, ref_lam)
    same(lams_m, ref_lams)
    same(errors, saved["errors"])
    same(errors_marg, saved["errorsMarg"])


def test_optimize_lambda_default_grid(data2):
    (lam, lams_m, _, _), draws = run_py(lambda: core.dpca_optimize_lambda(
        data2["Xfull"], data2["Xtrial"], data2["numOfTrials"], num_comps=6, num_rep=1, rng=np.random.default_rng(23)))
    ref_lam, ref_lams = O.call("dpca_optimizeLambda", data2["Xfull"], data2["Xtrial"], data2["numOfTrials"],
                               "numComps", 6, "numRep", 1, "display", "no", nargout=2, draws=draws)
    same(lam, ref_lam)
    same(lams_m, ref_lams)


# ============================================================================ classification
def check_accuracy(d, py_kw, m_args, seed):
    (acc, brier), draws = run_py(lambda: core.dpca_classification_accuracy(
        d["Xfull"], d["Xtrial"], d["numOfTrials"], rng=np.random.default_rng(seed), **py_kw))
    ref_acc, ref_brier = O.call("dpca_classificationAccuracy", d["Xfull"], d["Xtrial"], d["numOfTrials"], *m_args,
                                "verbose", "no", nargout=2, draws=draws)
    same(acc, ref_acc)
    same(brier, ref_brier)


@pytest.mark.parametrize("sim", [False, True])
def test_classification_accuracy_demo_classes(data3, data3_sim, sim):
    d = data3_sim if sim else data3
    check_accuracy(d, dict(num_rep=3, lambda_=1e-4, combined_params=COMBINED3, decoding_classes=demo_classes(3),
                           simultaneous=sim),
                   ["numRep", 3, "lambda", 1e-4, "simultaneous", sim,
                    *opts(combinedParams=COMBINED3, decodingClasses=demo_classes(3))], 32)


@pytest.mark.parametrize("num_comps", [2, [2, 1, 3, 2]])
def test_classification_default_classes_3param(data3, num_comps):
    check_accuracy(data3, dict(num_comps=num_comps, num_rep=3, lambda_=1e-4, combined_params=COMBINED3),
                   ["numComps", np.asarray(num_comps, float), "numRep", 3, "lambda", 1e-4,
                    *opts(combinedParams=COMBINED3)], 5)


def test_classification_default_classes_2param_is_an_extension(data2):
    """MATLAB itself errors for 2-parameter data with default decodingClasses (reshape with a
    scalar size); Python uses one class per condition. Verify the MATLAB error and that passing
    the equivalent classes explicitly gives parity."""
    with pytest.raises(O.OctaveError):
        O.call("dpca_classificationAccuracy", data2["Xfull"], data2["Xtrial"], data2["numOfTrials"], "numRep", 1,
               "verbose", "no", nargout=2)
    S = data2["Xfull"].shape[1]
    cls = [np.arange(1, S + 1)[:, None]] * 3
    (acc, brier), draws = run_py(lambda: core.dpca_classification_accuracy(
        data2["Xfull"], data2["Xtrial"], data2["numOfTrials"], num_rep=3, lambda_=1e-4,
        rng=np.random.default_rng(31)))
    ref_acc, ref_brier = O.call("dpca_classificationAccuracy", data2["Xfull"], data2["Xtrial"], data2["numOfTrials"],
                                "numRep", 3, "lambda", 1e-4, *opts(decodingClasses=cls), "verbose", "no",
                                nargout=2, draws=draws)
    same(acc, ref_acc)
    same(brier, ref_brier)


def test_classification_reordering_noisy():
    d = make_data(N=15, conds=(4,), T=8, noise=4.0, n_latent=2, max_trials=5, seed=1)
    cls = [np.arange(1, 5)[:, None]] * 3
    check_accuracy(d, dict(num_comps=3, num_rep=4, lambda_=1e-4, decoding_classes=cls),
                   ["numComps", 3, "numRep", 4, "lambda", 1e-4, *opts(decodingClasses=cls)], 1)


def test_classification_noncontiguous_classes(data3):
    cls = [np.array([[1, 1], [3, 3], [5, 5]])] * 4
    check_accuracy(data3, dict(num_comps=2, num_rep=3, lambda_=1e-4, combined_params=COMBINED3,
                               decoding_classes=cls),
                   ["numComps", 2, "numRep", 3, "lambda", 1e-4, *opts(combinedParams=COMBINED3, decodingClasses=cls)],
                   4)


def test_classification_time_splits(data3):
    check_accuracy(data3, dict(num_comps=2, num_rep=3, lambda_=1e-4, combined_params=COMBINED3,
                               decoding_classes=demo_classes(3), time_splits=[8], time_parameter=3,
                               not_to_split=[[[3]]]),
                   ["numComps", 2, "numRep", 3, "lambda", 1e-4, "timeSplits", [8], "timeParameter", 3,
                    *opts(combinedParams=COMBINED3, decodingClasses=demo_classes(3), notToSplit=[[[3]]])], 6)


def test_classification_4params():
    d = make_data(N=41, conds=(3, 2, 2), T=8, seed=12)
    check_accuracy(d, dict(num_comps=1, num_rep=2, lambda_=1e-4),
                   ["numComps", 1, "numRep", 2, "lambda", 1e-4], 7)


@pytest.mark.parametrize("sim", [False, True])
def test_classification_shuffled(data3, data3_sim, sim):
    d = data3_sim if sim else data3
    (acc, brier), draws = run_py(lambda: core.dpca_classification_shuffled(
        d["Xtrial"], d["numOfTrials"], num_shuffles=3, num_rep=2, lambda_=1e-4, combined_params=COMBINED3,
        decoding_classes=demo_classes(3), simultaneous=sim, rng=np.random.default_rng(33)))
    ref_acc, ref_brier = O.call("dpca_classificationShuffled", d["Xtrial"], d["numOfTrials"], "numShuffles", 3,
                                "numRep", 2, "lambda", 1e-4, "simultaneous", sim, "verbose", "no",
                                *opts(combinedParams=COMBINED3, decodingClasses=demo_classes(3)),
                                nargout=2, draws=draws)
    same(acc, ref_acc)
    same(brier, ref_brier)


@pytest.mark.parametrize("min_chunk", [1, 2, 3, 5, 10])
def test_signif_components(min_chunk):
    rng = np.random.default_rng(min_chunk)
    acc = rng.random((4, 3, 40))
    acc[2] = np.nan
    shuf = rng.random((4, 40, 6)) * 0.9
    wm = np.array([1, 4, 2, 1, 4, 1, 2, 3, 3, 1, 2, 4, 4, 1])
    got = core.dpca_signif_components(acc, shuf, wm, min_chunk=min_chunk)
    (ref,) = O.call("dpca_signifComponents", acc, shuf, wm, "minChunk", min_chunk)
    same(got, ref)


# ============================================================================ class API vs MATLAB
def key_of(subset, labels):
    return "".join(labels[p - 1] for p in sorted(subset))


def matlab_subsets(D):
    import matlab_oracle
    return matlab_oracle.subsets(range(1, D + 1))


def groups_for(labels, join):
    D = len(labels)
    if join is None:
        return None, [key_of(s, labels) for s in matlab_subsets(D)]
    joined = {c for combs in join.values() for c in combs}
    groups, keys = [], []
    for s in matlab_subsets(D):
        if key_of(s, labels) not in joined:
            groups.append([s]); keys.append(key_of(s, labels))
    for k, combs in join.items():
        groups.append([[labels.index(ch) + 1 for ch in c] for c in combs]); keys.append(k)
    return groups, keys


CASES = [("st", None, dict(N=31, conds=(4,), T=20, seed=1)),
         ("sdt", None, dict(N=31, conds=(3, 3), T=15, seed=2)),
         ("sdt", {"s": ["s", "st"], "d": ["d", "dt"], "sd": ["sd", "sdt"]}, dict(N=31, conds=(3, 3), T=15, seed=2))]


@pytest.mark.parametrize("labels,join,dkw", CASES)
@pytest.mark.parametrize("reg", [None, 1e-4])
def test_class_fit_transform_explained_variance(labels, join, dkw, reg):
    d = make_data(**dkw)
    groups, keys = groups_for(labels, join)
    m = quiet(dPCA, labels=labels, join=join, n_components=2, regularizer=reg)
    Z = quiet(m.fit_transform, d["Xfull"])
    W, V, wm = O.call("dpca", d["Xfull"], [2.0] * len(keys), "lambda", 0.0 if reg is None else reg,
                      *opts(combinedParams=groups), nargout=3)
    (ev,) = O.call("dpca_explainedVariance", d["Xfull"], W, V, *opts(combinedParams=groups))
    wm = np.asarray(wm).ravel()
    Xc = d["Xfull"] - d["Xfull"].reshape(31, -1).mean(1).reshape((-1,) + (1,) * (d["Xfull"].ndim - 1))
    for i, k in enumerate(keys):
        sel = wm == i + 1
        same(m.P[k], V[:, sel])
        same(m.D[k], W[:, sel])
        same(Z[k], np.tensordot(W[:, sel].T, Xc, axes=1))
        same(np.asarray(m.explained_variance_ratio_[k]) * 100, np.asarray(ev["margVar"])[i, sel])


def test_class_cnoise_and_auto_regularizer(tmp_path):
    d = make_data(N=31, conds=(4,), T=20, seed=6)
    lams = 1e-6 * 2.0 ** np.arange(8)
    m = quiet(dPCA, labels="st", n_components=3, regularizer="auto", random_state=3, n_trials=2, lambdas=lams)
    _, draws = run_py(lambda: m.fit(d["Xfull"], d["trialX"]))
    fname = str(tmp_path / "lam.mat")
    ref_lam, _ = O.call("dpca_optimizeLambda", d["Xfull"], d["Xtrial"], d["numOfTrials"], "numComps", [3.0, 3, 3],
                        "lambdas", lams, "numRep", 2, "display", "no", "filename", fname, nargout=2, draws=draws)
    import scipy.io
    saved = scipy.io.loadmat(fname)
    same(m.regularizer, ref_lam)
    same(m.cv_errors_, saved["errors"])
    same(m.cv_errors_marg_, saved["errorsMarg"])
    # final fit as in dpca_demo.m: lambda from CV plus the noise covariance
    C = m.get_noise_covariance(d["Xfull"], d["trialX"])
    (Cm,) = O.call("dpca_getNoiseCovariance", d["Xfull"], d["Xtrial"], d["numOfTrials"])
    same(C, Cm)
    quiet(m.fit, d["Xfull"], Cnoise=C)
    W, V, wm = O.call("dpca", d["Xfull"], [3.0, 3, 3], "lambda", m.regularizer, "Cnoise", Cm, nargout=3)
    wm = np.asarray(wm).ravel()
    for i, k in enumerate(["s", "t", "st"]):
        same(m.P[k], V[:, wm == i + 1])
        same(m.D[k], W[:, wm == i + 1])


@pytest.mark.parametrize("labels,join,dkw", CASES[1:])
def test_class_significance_analysis(labels, join, dkw):
    """masks == dpca_classificationAccuracy -> dpca_classificationShuffled -> dpca_signifComponents."""
    d = make_data(**dkw)
    groups, keys = groups_for(labels, join)
    lam, nc, nsh, nsp, chunk = 1e-4, 2, 3, 2, 3
    m = quiet(dPCA, labels=labels, join=join, n_components=nc, regularizer=lam, random_state=7)
    quiet(m.fit, d["Xfull"])
    (masks, acc_c, shuf_c), draws = run_py(lambda: m.significance_analysis(
        d["Xfull"], d["trialX"], n_shuffles=nsh, n_splits=nsp, n_consecutive=chunk, n_components=nc, full=True))
    n_first = nsp                                                # one rand() per repetition
    acc, _ = O.call("dpca_classificationAccuracy", d["Xfull"], d["Xtrial"], d["numOfTrials"], "numComps", nc,
                    "numRep", nsp, "lambda", lam, "verbose", "no", *opts(combinedParams=groups), nargout=2,
                    draws=draws[:n_first])
    shuf, _ = O.call("dpca_classificationShuffled", d["Xtrial"], d["numOfTrials"], "numShuffles", nsh, "numRep", nsp,
                     "lambda", lam, "verbose", "no", *opts(combinedParams=groups), nargout=2, draws=draws[n_first:])
    W, V, wm = O.call("dpca", d["Xfull"], [float(nc)] * len(keys), "lambda", lam, *opts(combinedParams=groups),
                      nargout=3)
    (sig,) = O.call("dpca_signifComponents", acc, shuf, wm, "minChunk", chunk)
    wm = np.asarray(wm).ravel()
    sig = np.asarray(sig)
    assert set(masks) == set(keys) - {labels[-1]}
    for i, k in enumerate(keys):
        if k in masks:
            np.testing.assert_array_equal(masks[k], sig[wm == i + 1].astype(bool))
            same(acc_c[k], np.asarray(acc)[i])
            same(shuf_c[k], np.asarray(shuf)[i])
