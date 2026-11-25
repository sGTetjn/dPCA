import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore", category=RuntimeWarning)


def make_data(N=31, conds=(4,), T=20, max_trials=8, min_trials=3, n_latent=8, simultaneous=False,
              equal_trials=False, noise=1.0, seed=0):
    """Generic synthetic dataset (all marginalizations have full rank, distinct spectra).

    Returns a dict with
      Xfull       (N, *conds, T)              trial average (nanmean over trials)
      Xtrial      (N, *conds, T, max_trials)  MATLAB layout, NaN-padded at the end
      numOfTrials (N, *conds)
      trialX      (max_trials, N, *conds, T)  layout of the Python class API
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, T)
    shape = (N,) + tuple(conds) + (T,)
    Xtrue = np.zeros(shape)
    for _ in range(n_latent):
        enc = rng.normal(size=N)
        cond_amp = rng.normal(size=tuple(conds) + (1,))
        freq, phase = rng.uniform(0.5, 3), rng.uniform(0, 2 * np.pi)
        f = cond_amp * np.sin(2 * np.pi * freq * t + phase) + rng.normal(size=tuple(conds) + (1,)) * t
        Xtrue += enc.reshape((N,) + (1,) * (len(shape) - 1)) * f[None]
    Xtrue += rng.normal(size=(N,) + (1,) * (len(shape) - 1)) * 5          # baselines
    if equal_trials:
        ntr = np.full((N,) + tuple(conds), max_trials)
    elif simultaneous:
        ntr = np.broadcast_to(rng.integers(min_trials, max_trials + 1, size=(1,) + tuple(conds)),
                              (N,) + tuple(conds)).copy()
    else:
        ntr = rng.integers(min_trials, max_trials + 1, size=(N,) + tuple(conds))
    Xtrial = Xtrue[..., None] + rng.normal(size=shape + (max_trials,)) * noise
    for idx in np.ndindex(*ntr.shape):
        Xtrial[idx + (slice(None), slice(ntr[idx], None))] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xfull = np.nanmean(Xtrial, -1)
    return dict(Xfull=Xfull, Xtrial=Xtrial, numOfTrials=ntr, trialX=np.moveaxis(Xtrial, -1, 0))


@pytest.fixture
def data2():
    """2 parameters (stimulus x time)."""
    return make_data(seed=1)


@pytest.fixture
def data3():
    """3 parameters (stimulus x decision x time)."""
    return make_data(conds=(3, 2), T=15, seed=2)


@pytest.fixture
def data3_sim():
    """3 parameters, simultaneously recorded (same trial count for all neurons)."""
    return make_data(conds=(3, 2), T=15, simultaneous=True, seed=3)


# demo-style combinedParams for (stimulus, decision, time)
COMBINED3 = [[1, [1, 3]], [2, [2, 3]], [3], [[1, 2], [1, 2, 3]]]
