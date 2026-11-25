""" demixed Principal Component Analysis

sklearn-style interface. All numerics are delegated to :mod:`dPCA.core`, a 1:1 port of the
reference MATLAB implementation, so results match MATLAB exactly.
"""

# Author: Wieland Brendel <wieland.brendel@neuro.fchampalimaud.org>
#
# License: BSD 3 clause

import warnings
from collections import OrderedDict
from itertools import combinations, chain

import numpy as np
from sklearn.base import BaseEstimator

from . import core


class dPCA(BaseEstimator):
    """ demixed Principal component analysis (dPCA)

    dPCA is a linear dimensionality reduction technique that automatically discovers
    and highlights the essential features of complex population activities. The
    population activity is decomposed into a few demixed components that capture most
    of the variance in the data and that highlight the dynamic tuning of the population
    to various task parameters, such as stimuli, decisions, rewards, etc.

    Parameters
    ----------
    labels : int or string
        Labels of the parameter axes (one character per axis), e.g. 'st' for stimulus and
        time. If int, the corresponding number of labels is taken from 'abcde...'.
        As in the MATLAB implementation, the **last axis is assumed to be time**.

    join : None or dict
        Marginalizations to join (MATLAB 'combinedParams'), e.g. {'s': ['s', 'st']}. The
        letters of every key and combination must follow the order of ``labels``.

    n_components : int or dict
        Number of components per marginalization (an int applies to every marginalization,
        a dict sets them individually, e.g. {'t': 10, 'st': 5}). Within each marginalization
        components are ordered by explained variance and signed so that encoders have mostly
        positive entries (as in MATLAB). Defaults to 10.

    regularizer : None, float or 'auto'
        Ridge parameter lambda (MATLAB 'lambda'); the penalty is (lambda * total variance)^2.
        If 'auto', lambda is chosen by cross-validation during ``fit`` (MATLAB
        ``dpca_optimizeLambda``), which requires trial-by-trial data. As in MATLAB, the noise
        covariance is used inside the cross-validation, but the final fit only uses it if
        passed explicitly: to reproduce ``dpca_demo.m`` call
        ``fit(X, trialX, Cnoise=model.get_noise_covariance(X, trialX))``.
        Note: MATLAB's scalar ``numComps`` mode (keep the top k components overall) is
        available through :func:`dPCA.core.dpca`, not through this class.

    random_state : None, int or numpy.random.Generator
        Seed for cross-validation splits and label shuffles.

    n_trials : int (default 10)
        Number of cross-validation repetitions used when regularizer='auto' (MATLAB 'numRep').

    lambdas : array or None
        Grid of lambdas searched when regularizer='auto' (default 1e-7 * 1.5**[0..25]).

    cv_method : 'training' (default), 'neuronwise' or 'naive'
        Cross-validation objective for regularizer='auto' (MATLAB 'method').

    noise_cov_type : 'pooled' (default), 'averaged' or 'none'
        Noise covariance used inside cross-validation and significance analysis.

    simultaneous : bool (default False)
        Whether the neurons were recorded simultaneously (same trials for all neurons).

    copy, n_iter :
        Unused; kept for backwards compatibility.

    Attributes
    ----------
    P, D : dict
        Encoder and decoder matrices (n_features, n_components) per marginalization.
    explained_variance_ratio_ : dict
        Fraction of total variance explained by each component within its own
        marginalization (MATLAB ``explVar.margVar``), computed on the training data.
    explained_var_ : dict
        All explained-variance measures of MATLAB ``dpca_explainedVariance`` (percentages)
        for the components concatenated in ``component_order_``.

    Notes
    -----
    Implements the dPCA model from:
    D Kobak*, W Brendel*, C Constantinidis, C Feierstein, A Kepecs, Z Mainen,
    R Romo, X-L Qi, N Uchida, C Machens. Demixed principal component analysis of
    neural population data. eLife 2016;5:e10989.

    Examples
    --------
    >>> import numpy as np
    >>> from dPCA import dPCA
    >>> X = np.random.randn(20, 4, 50)        # neurons x stimuli x time
    >>> model = dPCA.dPCA(labels='st', n_components=3)
    >>> Z = model.fit_transform(X)
    >>> Z['st'].shape
    (3, 4, 50)
    """

    def __init__(self, labels=None, join=None, n_components=10, regularizer=None, copy=True, n_iter=None,
                 random_state=None, n_trials=10, lambdas=None, cv_method="training", noise_cov_type="pooled",
                 simultaneous=False):
        if isinstance(labels, str):
            self.labels = labels
        elif isinstance(labels, int):
            self.labels = "abcdefghijklmnopqrstuvwxyz"[:labels]
        else:
            raise TypeError('Wrong type for labels. Please either set labels to the number of variables or provide '
                            'the axis labels as a single string of characters (like "ts" for time and stimulus)')
        if n_iter is not None:
            warnings.warn("n_iter is ignored: decompositions are computed exactly (as in MATLAB).",
                          DeprecationWarning, stacklevel=2)

        self._join = join
        self.join = join
        self.regularizer = 0 if regularizer is None else regularizer
        self.opt_regularizer_flag = isinstance(regularizer, str) and regularizer == "auto"
        self.n_components = n_components
        self.copy = copy
        self.n_iter = n_iter
        self.random_state = random_state
        self.n_trials = n_trials
        self.lambdas = lambdas
        self.cv_method = cv_method
        self.noise_cov_type = noise_cov_type
        self.simultaneous = simultaneous
        self.marginalizations = self._get_parameter_combinations()
        self.debug = 2

    # ------------------------------------------------------------------ bookkeeping
    def _get_parameter_combinations(self, join=True):
        ''' Returns all parameter combinations, e.g. for labels = 'xyz'

            {'x' : {0}, 'y' : {1}, 'z' : {2}, 'xy' : {0,1}, 'xz' : {0,2}, 'yz' : {1,2}, 'xyz' : {0,1,2}}

            If join == True, parameter combinations are condensed according to self._join (the value
            of a joined key is the list of the joined sets).
        '''
        D = len(self.labels)
        subsets = list(chain.from_iterable(combinations(range(D), r) for r in range(1, D + 1)))
        pcombs = OrderedDict((''.join(self.labels[i] for i in s), set(s)) for s in subsets)
        if isinstance(self._join, dict) and join:
            for key, combs in self._join.items():
                tmp = [pcombs[comb] for comb in combs]
                for comb in combs:
                    del pcombs[comb]
                pcombs[key] = tmp
        return pcombs

    def _combined_params(self):
        """MATLAB combinedParams (1-based parameters) following the order of self.marginalizations."""
        groups = []
        for val in self.marginalizations.values():
            sets = val if isinstance(val, list) else [val]
            groups.append([sorted(i + 1 for i in s) for s in sets])
        return groups

    def _num_comps(self, n_components=None):
        n_components = self.n_components if n_components is None else n_components
        if isinstance(n_components, dict):
            return [int(n_components[k]) for k in self.marginalizations]
        return [int(n_components)] * len(self.marginalizations)

    def _rng(self):
        return self.random_state if isinstance(self.random_state, np.random.Generator) \
            else np.random.default_rng(self.random_state)

    def _matlab_trials(self, X, trialX):
        """(n_trials, N, ...) -> MATLAB layout (N, ..., n_trials) plus numOfTrials (N, conditions)."""
        trialX = np.asarray(trialX, dtype=float)
        if trialX.shape[1:] != X.shape:
            raise ValueError("trialX must have shape (n_trials,) + X.shape")
        Xtrial = np.moveaxis(trialX, 0, -1)
        valid = ~np.isnan(Xtrial[..., 0, :])                   # (N, conditions, trials) at first time bin
        num_trials = valid.sum(-1)
        if np.any(valid != (np.arange(valid.shape[-1]) < num_trials[..., None])):
            raise ValueError("Missing trials must be NaN and placed at the end of the trial axis.")
        return Xtrial, num_trials

    # ------------------------------------------------------------------ fitting
    def fit(self, X, trialX=None, Cnoise=None):
        """Fit the model with X.

        Parameters
        ----------
        X : array-like, shape (n_features, n_1, n_2, ...)
            Trial-averaged data (neurons first; the last parameter axis is time).
        trialX : array-like, shape (n_trials, n_features, n_1, n_2, ...), optional
            Single-trial data, NaN-padded at the end of the trial axis. Required for
            regularizer='auto'.
        Cnoise : array (n_features, n_features), optional
            Noise covariance (see ``get_noise_covariance``), MATLAB dpca(..., 'Cnoise', Cnoise).

        Returns
        -------
        self
        """
        self._fit(X, trialX=trialX, Cnoise=Cnoise)
        return self

    def fit_transform(self, X, trialX=None, Cnoise=None):
        """Fit the model with X and return ``transform(X)``."""
        self._fit(X, trialX=trialX, Cnoise=Cnoise)
        return self.transform(X)

    def _fit(self, X, trialX=None, Cnoise=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != len(self.labels) + 1:
            raise ValueError("X must have one axis per label plus the leading neuron axis")
        if self.opt_regularizer_flag:
            if trialX is None:
                raise ValueError('To optimize the regularization parameter, the trial-by-trial data trialX needs '
                                 'to be provided.')
            self._optimize_regularization(X, trialX)

        nc = self._num_comps()
        W, V, which = core.dpca(X, nc, combined_params=self._combined_params(), lambda_=self.regularizer,
                                order=True, Cnoise=Cnoise)
        keys = list(self.marginalizations)
        self.P = {k: V[:, which == i + 1] for i, k in enumerate(keys)}
        self.D = {k: W[:, which == i + 1] for i, k in enumerate(keys)}

        # explained variance on the training data, components grouped per marginalization
        Wc = np.hstack([self.D[k] for k in keys])
        Vc = np.hstack([self.P[k] for k in keys])
        self.component_order_ = [(k, j) for k in keys for j in range(self.D[k].shape[1])]
        try:
            self.explained_var_ = core.dpca_explained_variance(X, Wc, Vc, combined_params=self._combined_params())
        except ValueError:                  # more components than PCA components (MATLAB errors as well)
            self.explained_var_ = None
        # MATLAB explVar.margVar of each component in its own marginalization
        Xc = self._zero_mean(X).reshape(X.shape[0], -1)
        total = np.sum(Xc ** 2)
        margs = self._marginalize(X)
        self.explained_variance_ratio_ = {}
        for k in keys:
            Xm, P, D = margs[k], self.P[k], self.D[k]
            self.explained_variance_ratio_[k] = [
                (np.sum(Xm ** 2) - np.sum((Xm - np.outer(P[:, j], D[:, j] @ Xm)) ** 2)) / total
                for j in range(D.shape[1])]

    def _optimize_regularization(self, X, trialX):
        Xtrial, num_trials = self._matlab_trials(X, trialX)
        lam, lams, errors, errors_marg = core.dpca_optimize_lambda(
            X, Xtrial, num_trials, num_comps=self._num_comps(), lambdas=self.lambdas, num_rep=self.n_trials,
            combined_params=self._combined_params(), method=self.cv_method, simultaneous=self.simultaneous,
            noise_cov_type=self.noise_cov_type, rng=self._rng(), verbose=self.debug > 1)
        lambdas = 1e-7 * 1.5 ** np.arange(26) if self.lambdas is None else np.asarray(self.lambdas)
        if self.debug > 0 and lam in (lambdas[0], lambdas[-1]):
            print("Warning: Optimal regularization parameter lies at the boundary of the search interval. "
                  "Please provide a different search list (lambdas).")
        self.regularizer = lam
        self.cv_errors_ = errors
        self.cv_errors_marg_ = errors_marg
        self.opt_regularizer_flag = False
        if self.debug > 1:
            print('Optimized regularization, optimal lambda = ', lam)

    def get_noise_covariance(self, X, trialX, type="pooled"):
        """Noise covariance matrix (MATLAB dpca_getNoiseCovariance) for ``fit(..., Cnoise=...)``."""
        Xtrial, num_trials = self._matlab_trials(np.asarray(X, float), trialX)
        return core.dpca_get_noise_covariance(np.asarray(X, float), Xtrial, num_trials,
                                              simultaneous=self.simultaneous, type=type)

    # ------------------------------------------------------------------ projections
    def _zero_mean(self, X):
        """ Subtracts the mean from each observable """
        return X - np.mean(X.reshape((X.shape[0], -1)), 1).reshape((X.shape[0],) + (len(X.shape) - 1) * (1,))

    def _marginalize(self, X):
        """Marginalized data as a dict of arrays flattened to (n_features, -1)."""
        margs, _ = core.dpca_marginalize(np.asarray(X, float), combined_params=self._combined_params())
        return OrderedDict((k, m.reshape(m.shape[0], -1)) for k, m in zip(self.marginalizations, margs))

    def transform(self, X, marginalization=None):
        """Project the (centered) data onto the decoders.

        Returns a dict {marginalization: array (n_components, n_1, n_2, ...)} or, if
        ``marginalization`` is given, only that array.
        """
        X = self._zero_mean(np.asarray(X, float))
        Xr = X.reshape((X.shape[0], -1))

        def proj(key):
            D = self.D[key]
            return np.dot(D.T, Xr).reshape((D.shape[1],) + X.shape[1:])

        if marginalization is not None:
            return proj(marginalization)
        return {key: proj(key) for key in self.marginalizations}

    def inverse_transform(self, X, marginalization):
        """Map latent components (n_components, n_1, ...) back to data space: P @ X."""
        X = np.asarray(X, float)
        P = self.P[marginalization]
        return np.dot(P, X.reshape((X.shape[0], -1))).reshape((P.shape[0],) + X.shape[1:])

    def reconstruct(self, X, marginalization):
        """Equivalent to inverse_transform(transform(X, marginalization), marginalization)."""
        return self.inverse_transform(self.transform(X, marginalization), marginalization)

    # ------------------------------------------------------------------ cross-validation helpers
    def train_test_split(self, X, trialX):
        """One random test trial per neuron and condition (same trial at all time points), as in
        MATLAB. Returns (trainX, testX): the mean of the remaining trials and the test trial."""
        X = np.asarray(X, float)
        Xtrial, num_trials = self._matlab_trials(X, trialX)
        Xtest, _ = core.dpca_get_test_trials(Xtrial, num_trials, rng=self._rng(), simultaneous=self.simultaneous)
        ntc = num_trials.reshape(num_trials.shape + (1,))
        return (X * ntc - Xtest) / (ntc - 1), Xtest

    def shuffle_labels(self, trialX):
        """Return a copy of trialX with whole trials shuffled across conditions within each
        neuron (preserving trial counts per condition); missing trials stay NaN."""
        trialX = np.asarray(trialX, float)
        Xtrial, num_trials = self._matlab_trials(np.nanmean(trialX, 0), trialX)
        sh = core.dpca_shuffle_trials(Xtrial, num_trials, rng=self._rng(), simultaneous=self.simultaneous)
        sh[np.isnan(Xtrial)] = np.nan
        return np.moveaxis(sh, -1, 0)

    # ------------------------------------------------------------------ significance
    def significance_analysis(self, X, trialX, n_shuffles=100, n_splits=100, n_consecutive=1, axis=None,
                              full=False, n_components=3, decoding_classes=None):
        '''Cross-validated significance of the demixed components over time (MATLAB
        ``dpca_classificationAccuracy`` + ``dpca_classificationShuffled`` + ``dpca_signifComponents``).

        For every non-time marginalization, dPCA is refitted on ``n_splits`` random train/test
        splits and each test trial is classified by the nearest training class mean at each time
        point. A time point of a component is significant if its accuracy exceeds the maximum
        accuracy over ``n_shuffles`` label shuffles. The fitted model (P, D) is not modified.

        Parameters
        ----------
        X : array (n_features, n_1, ..., n_T)
        trialX : array (n_trials, n_features, n_1, ..., n_T), NaN-padded at the end
        n_shuffles : int
            Number of label shuffles (MATLAB 'numShuffles').
        n_splits : int
            Number of train/test splits (MATLAB 'numRep').
        n_consecutive : int
            MATLAB 'minChunk': runs of L significant bins with L + 1 < n_consecutive are removed.
            The default 1 disables this filter (MATLAB's default is minChunk = 10).
        axis : ignored (significance is always computed per time point, as in MATLAB).
        full : bool
            Also return the accuracies.
        n_components : int
            Number of components tested per marginalization (MATLAB 'numComps', default 3;
            at most 3 are used for the masks, as in MATLAB).
        decoding_classes : dict or None
            {marginalization: array of 1-based class ids shaped like the condition axes};
            default is one class per condition.

        Returns
        -------
        masks : dict {marginalization: bool array (n_components of the fitted model, T)}
        accuracy, accuracy_shuffle : dicts (only if full=True), arrays (n_components, T) and
            (T, n_shuffles).
        '''
        if axis is not None:
            warnings.warn("axis is ignored; significance is computed for every time point.", stacklevel=2)
        X = np.asarray(X, float)
        if self.opt_regularizer_flag:
            self._optimize_regularization(X, trialX)
        Xtrial, num_trials = self._matlab_trials(X, trialX)
        keys = list(self.marginalizations)
        groups = self._combined_params()
        classes = None
        if decoding_classes is not None:
            classes = [decoding_classes.get(k) for k in keys]

        rng = self._rng()
        common = dict(lambda_=self.regularizer, num_rep=n_splits, combined_params=groups, decoding_classes=classes,
                      simultaneous=self.simultaneous, noise_cov_type=self.noise_cov_type, rng=rng,
                      verbose=self.debug > 1)
        acc, _ = core.dpca_classification_accuracy(X, Xtrial, num_trials, num_comps=n_components, **common)
        shuf, _ = core.dpca_classification_shuffled(Xtrial, num_trials, num_shuffles=n_shuffles, **common)

        which = np.concatenate([np.full(self.D[k].shape[1], i + 1) for i, k in enumerate(keys)])
        sig = core.dpca_signif_components(acc, shuf, which, min_chunk=n_consecutive).astype(bool)

        masks, accuracy, accuracy_shuffle = {}, {}, {}
        for i, k in enumerate(keys):
            if np.isnan(acc[i, 0, 0]):
                continue                     # time marginalization
            masks[k] = sig[which == i + 1]
            accuracy[k] = acc[i]
            accuracy_shuffle[k] = shuf[i]
        if full:
            return masks, accuracy, accuracy_shuffle
        return masks
