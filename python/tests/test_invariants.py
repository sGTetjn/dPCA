"""Oracle-independent checks: mathematical invariants and regressions for the bugs in REPORT.md."""
import contextlib
import io
import itertools

import numpy as np
import pytest

from conftest import make_data
from dPCA.dPCA import dPCA


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


core = pytest.importorskip("dPCA.core")


# ---------------------------------------------------------------- marginalization
def test_marginalizations_sum_to_centered_data_and_are_orthogonal(data3):
    X = data3["Xfull"]
    margs, _ = core.dpca_marginalize(X)
    Xc = X - X.reshape(X.shape[0], -1).mean(1).reshape(-1, 1, 1, 1)
    np.testing.assert_allclose(sum(margs), Xc, atol=1e-10)
    for a, b in itertools.combinations(margs, 2):     # orthogonal decomposition (per neuron)
        np.testing.assert_allclose(np.sum(a * b, axis=(1, 2, 3)), 0, atol=1e-8)


# ---------------------------------------------------------------- fit
def test_lambda0_decoder_is_exact_reduced_rank_regression(data2):
    """P must be the exact top eigenvectors of (C X)(C X)', with C = X_phi X^+ (no randomized SVD)."""
    X = data2["Xfull"]
    m = quiet(dPCA, labels="st", n_components=5).fit(X)
    Xc = X.reshape(X.shape[0], -1); Xc = Xc - Xc.mean(1, keepdims=True)
    margs, _ = core.dpca_marginalize(X)
    for k, Xm in zip(["s", "t", "st"], margs):
        Xm = Xm.reshape(X.shape[0], -1)
        CX = Xm @ np.linalg.pinv(Xc) @ Xc
        s = np.linalg.svd(CX, compute_uv=False)
        n = min(5, int(np.sum(s > 1e-8 * s[0])))
        got = np.linalg.norm(m.P[k][:, :n].T @ CX, axis=1)
        np.testing.assert_allclose(np.sort(got)[::-1], s[:n], rtol=1e-8)


def test_more_components_than_neurons_raises():
    X = np.random.default_rng(0).normal(size=(8, 4, 30))
    with pytest.raises(ValueError):
        quiet(dPCA, labels="st", n_components=10).fit(X)


def test_fit_is_deterministic(data2):
    a = quiet(dPCA, labels="st", n_components=4, regularizer=1e-4).fit(data2["Xfull"])
    b = quiet(dPCA, labels="st", n_components=4, regularizer=1e-4).fit(data2["Xfull"])
    for k in a.P:
        np.testing.assert_array_equal(a.P[k], b.P[k])


def test_reconstruct_is_encoder_times_latent(data2):
    m = quiet(dPCA, labels="st", n_components=3).fit(data2["Xfull"])
    Z = m.transform(data2["Xfull"], "st")
    Z2 = Z + 5.0                                     # a latent with nonzero mean must not be re-centered
    R = m.inverse_transform(Z2, "st")
    np.testing.assert_allclose(R, np.einsum("nc,c...->n...", m.P["st"], Z2))


def test_explained_variance_not_overwritten_by_transform(data2):
    m = quiet(dPCA, labels="st", n_components=3).fit(data2["Xfull"])
    m.transform(data2["Xfull"])
    ev = {k: np.array(v) for k, v in m.explained_variance_ratio_.items()}
    m.transform(data2["Xfull"][:, :2])
    for k in ev:
        np.testing.assert_array_equal(ev[k], m.explained_variance_ratio_[k])


# ---------------------------------------------------------------- CV split
def test_train_split_is_mean_of_remaining_trials_unequal_counts():
    d = make_data(seed=4)
    rng = np.random.default_rng(0)
    Xtest, XtrainFull = core.dpca_get_test_trials(d["Xtrial"], d["numOfTrials"], rng=rng)
    # the held-out trial is the same for all time bins of a neuron x condition
    held = np.isnan(XtrainFull) & ~np.isnan(d["Xtrial"])
    assert np.all(held.sum(-1) == 1)
    assert np.all(held.all(axis=-2) == held.any(axis=-2))
    Xtrain = (d["Xfull"] * d["numOfTrials"][..., None] - Xtest) / (d["numOfTrials"][..., None] - 1)
    np.testing.assert_allclose(Xtrain, np.nanmean(XtrainFull, -1), atol=1e-12)


def test_simultaneous_split_uses_same_trial_for_all_neurons(data3_sim):
    d = data3_sim
    _, XtrainFull = core.dpca_get_test_trials(d["Xtrial"], d["numOfTrials"], rng=np.random.default_rng(1),
                                              simultaneous=True)
    held = np.isnan(XtrainFull[..., 0, :]) & ~np.isnan(d["Xtrial"][..., 0, :])   # (N, S, D, trials)
    assert np.all(held == held[:1])


# ---------------------------------------------------------------- shuffling
def test_shuffle_is_within_neuron_and_preserves_trials(data3):
    d = data3
    sh = core.dpca_shuffle_trials(d["Xtrial"], d["numOfTrials"], rng=np.random.default_rng(0))
    N, T = d["Xtrial"].shape[0], d["Xtrial"].shape[-2]
    for n in range(N):
        a = np.moveaxis(d["Xtrial"][n], -2, -1).reshape(-1, T)
        b = np.moveaxis(sh[n], -2, -1).reshape(-1, T)
        a = a[~np.isnan(a[:, 0])]
        b = b[np.any(b != 0, axis=1) & ~np.isnan(b[:, 0])]
        # same multiset of whole trials (time courses kept intact) within each neuron
        np.testing.assert_allclose(a[np.lexsort(a.T)], b[np.lexsort(b.T)])
    # trial counts per condition are preserved: trial slots beyond numOfTrials stay empty (zero)
    for idx in np.ndindex(*d["numOfTrials"].shape):
        assert np.all(sh[idx][..., d["numOfTrials"][idx]:] == 0)


def test_permutation_is_uniform():
    counts = {}
    rng = np.random.default_rng(0)
    for _ in range(30000):
        p = tuple(core.randperm(rng, 3))
        counts[p] = counts.get(p, 0) + 1
    assert len(counts) == 6
    assert all(abs(c / 30000 - 1 / 6) < 0.015 for c in counts.values())


# ---------------------------------------------------------------- significance
@pytest.mark.parametrize("s,min_chunk,expected", [
    # MATLAB rule (dpca_signifComponents.m): a run of ones of length L is removed iff 1 <= L <= minChunk-2
    ([1, 1, 0, 1, 0, 0, 0], 3, [1, 1, 0, 0, 0, 0, 0]),
    ([1, 1, 1, 0, 0, 1, 0, 0], 3, [1, 1, 1, 0, 0, 0, 0, 0]),
    ([0, 0, 1, 1, 1, 1, 0, 1, 1], 4, [0, 0, 1, 1, 1, 1, 0, 0, 0]),
    ([1, 0, 1, 1, 0, 1, 1, 1], 1, [1, 0, 1, 1, 0, 1, 1, 1]),
])
def test_chunk_removal(s, min_chunk, expected):
    acc = np.array(s, float)[None, None, :]
    shuf = np.full((1, len(s), 2), 0.5)
    got = core.dpca_signif_components(acc, shuf, np.array([1]), min_chunk=min_chunk)
    np.testing.assert_array_equal(got[0], expected)


@pytest.mark.parametrize("reg", [None, 1e-4])
def test_significance_analysis_runs_and_keeps_fitted_model(reg):
    d = make_data(N=25, conds=(3,), T=12, seed=8)
    m = quiet(dPCA, labels="st", n_components=2, regularizer=reg, random_state=0).fit(d["Xfull"])
    P = {k: v.copy() for k, v in m.P.items()}
    masks, acc, shuf = quiet(m.significance_analysis, d["Xfull"], d["trialX"], n_shuffles=2, n_splits=2,
                             full=True)
    assert set(masks) == {"s", "st"}
    assert masks["s"].shape == (2, 12)
    for k in P:
        np.testing.assert_array_equal(P[k], m.P[k])


def test_decoding_sanity():
    """Strong stimulus tuning -> near-perfect decoding; label-shuffled data -> near chance."""
    rng = np.random.default_rng(10)
    N, S, T, K = 21, 4, 10, 6
    t = np.linspace(0, 1, T)
    Xtrue = (rng.normal(size=N)[:, None, None] * np.arange(S)[None, :, None] * 2      # stimulus, constant in time
             + rng.normal(size=N)[:, None, None] * np.sin(2 * np.pi * t)[None, None, :])
    Xtrial = Xtrue[..., None] + 0.3 * rng.normal(size=(N, S, T, K))
    ntr = np.full((N, S), K)
    Xfull = Xtrial.mean(-1)
    acc, _ = core.dpca_classification_accuracy(Xfull, Xtrial, ntr, num_rep=5, lambda_=1e-4,
                                               rng=np.random.default_rng(0))
    assert np.all(acc[0, 0] > 0.95)            # first stimulus component, every time bin
    assert np.all(np.isnan(acc[1]))            # time marginalization is not decoded
    shuf, _ = core.dpca_classification_shuffled(Xtrial, ntr, num_shuffles=10, num_rep=5, lambda_=1e-4,
                                                rng=np.random.default_rng(1))
    assert np.nanmean(shuf[0]) < 0.45          # chance = 0.25


# ---------------------------------------------------------------- parallel execution
@pytest.mark.parametrize("sim", [False, True])
def test_n_jobs_gives_identical_results(data3, data3_sim, sim):
    """All random numbers are drawn up front in the serial order, so parallel runs must give
    exactly the same results as serial ones."""
    d = data3_sim if sim else data3
    combined = [[1, [1, 3]], [2, [2, 3]], [3], [[1, 2], [1, 2, 3]]]
    args = (d["Xfull"], d["Xtrial"], d["numOfTrials"])
    kw = dict(combined_params=combined, simultaneous=sim)
    for n_jobs in (1, 2):
        lam = quiet(core.dpca_optimize_lambda, *args, num_rep=3, rng=1, n_jobs=n_jobs, **kw)
        acc = core.dpca_classification_accuracy(*args, num_rep=3, lambda_=1e-5, rng=2, n_jobs=n_jobs, **kw)
        shuf = core.dpca_classification_shuffled(d["Xtrial"], d["numOfTrials"], num_shuffles=3, num_rep=2,
                                                 lambda_=1e-5, rng=3, n_jobs=n_jobs, **kw)
        if n_jobs == 1:
            ref = (lam, acc, shuf)
        else:
            for a, b in zip(lam + acc + shuf, ref[0] + ref[1] + ref[2]):
                np.testing.assert_array_equal(a, b)
