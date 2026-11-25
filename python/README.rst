demixed Principal Component Analysis (dPCA)
===========================================

dPCA is a linear dimensionality reduction technique that automatically discovers and highlights the essential features of complex population activities. The population activity is decomposed into a few demixed components that capture most of the variance in the data and that highlight the dynamic tuning of the population to various task parameters, such as stimuli, decisions, rewards, etc.

.. code-block::

    @article{kobak2016dpca,
       title={Demixed principal component analysis of neural population data},
       volume={5},
       ISSN={2050-084X},
       url={http://dx.doi.org/10.7554/eLife.10989},
       DOI={10.7554/elife.10989},
       journal={eLife},
       publisher={eLife Sciences Publications, Ltd},
       author={Kobak, Dmitry and Brendel, Wieland and Constantinidis, Christos and Feierstein, Claudia E and Kepecs, Adam and Mainen, Zachary F and Qi, Xue-Lian and Romo, Ranulfo and Uchida, Naoshige and Machens, Christian K},
       year={2016},
       month={Apr}
    }

🚀 Quickstart
-------------

.. code-block:: bash

   pip install dpca


🎉 Use dPCA
-----------
Simple example code for surrogate data can be found in [**dpca_demo.ipynb**](http://nbviewer.ipython.org/github/wielandbrendel/dPCA/blob/master/python/dPCA_demo.ipynb) and **dpca_demo.m**.

API of dPCA is similar to sklearn. To use dPCA, you should first import dPCA and initialize it before callling the fitting function,

.. code-block:: python

   from dPCA import dPCA
   dpca = dPCA.dPCA(labels, n_components, regularizer)
   Z = dpca.fit_transform(X)
   
The required initialization parameters are:

* **X:** A multidimensional array containing the trial-averaged data. E.g. X[n,t,s,d] could correspond to the mean response of the *n*-th neuron at time *t* in trials with stimulus *s* and decision *d*. The observable (e.g. neuron index) needs to come first.
* **labels:** Optional; list of characters with which to describe the parameter axes, e.g. 'tsd' to denote time, stimulus and decision axis. All marginalizations (e.g. time-stimulus) are refered to by subsets of those characters (e.g. 'ts').
* **n_components:** Dictionary or integer; if integer use the same number of components in each marginalization, otherwise every (key,value) pair refers to the number of components (value) in a marginalization (key).

More detailed documentation, and additional options, can be found in **dpca.py**.

Parity with the MATLAB implementation
-------------------------------------

All numerics are delegated to ``dPCA.core``, a 1:1 port of the MATLAB code (``dpca``,
``dpca_marginalize``, ``dpca_explainedVariance``, ``dpca_getNoiseCovariance``,
``dpca_optimizeLambda``, ``dpca_classificationAccuracy``, ``dpca_classificationShuffled``,
``dpca_signifComponents``). ``dPCA.core`` keeps the MATLAB conventions (single trials as the
*last* axis, 1-based parameter numbers and class ids) so MATLAB scripts can be ported directly.
The test suite in ``tests/`` checks parity against a literal transliteration of the MATLAB files.

As in MATLAB, the last parameter axis is assumed to be time. The workflow of ``dpca_demo.m``
with the class API:

.. code-block:: python

   # X: (neurons, stimuli, time); trialX: (trials, neurons, stimuli, time), NaN-padded at the end
   model = dPCA.dPCA(labels='st', n_components=10, regularizer='auto', random_state=0)
   model.fit(X, trialX)                              # lambda chosen by cross-validation
   Cnoise = model.get_noise_covariance(X, trialX)
   Z = model.fit_transform(X, Cnoise=Cnoise)         # final fit with noise covariance
   masks = model.significance_analysis(X, trialX, n_shuffles=100, n_splits=100, n_consecutive=10)

``n_components`` is the number of components *per marginalization*; components are ordered by
explained variance and signed as in MATLAB. ``explained_variance_ratio_`` is MATLAB's
``margVar`` and ``explained_var_`` holds all measures of ``dpca_explainedVariance``.
