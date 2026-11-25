"""Literal NumPy transliteration of the MATLAB reference implementation (../../matlab/*.m).

THIS FILE IS A TEST ORACLE ONLY. It is written to mirror the MATLAB source statement by
statement (column-major flattening, 1-based parameter/class numbers, MATLAB tie-breaking),
not to be fast or pretty. Each function names the .m file it transliterates. Plotting,
'display' and 'filename' options are not ported.

Conventions (identical to MATLAB):
  Xfull        (N, d1, ..., dD)                 trial-averaged data, last parameter = time
  Xtrial       (N, d1, ..., dD, maxTrials)      single trials, NaN-padded at the END
  numOfTrials  (N, d1, ..., d_{D-1})            number of trials per neuron & condition
  combinedParams  list of groups, each a list of parameter subsets (1-based), e.g.
                  [[1, [1, 3]], [2, [2, 3]], [3], [[1, 2], [1, 2, 3]]]
  decodingClasses list (one per marginalization) of arrays of 1-based class ids, or None

Randomness: MATLAB's rand / randperm are replaced by the two primitives below. dPCA.core uses
the same primitives in the same order, so with identically seeded generators both
implementations see identical draws.
"""
import numpy as np

EPS = np.finfo(float).eps


# --------------------------------------------------------------------------- MATLAB helpers
def rand(rng, n):
    """rand([n 1])"""
    return rng.random(n)


def randperm(rng, n, k=None):
    """randperm(n) / randperm(n, k), 1-based"""
    p = rng.permutation(int(n)) + 1
    return p if k is None else p[:k]


def colon(A):
    """A(:)"""
    return np.asarray(A).ravel(order="F")


def flat2(A):
    """A(:,:)"""
    return np.reshape(A, (A.shape[0], -1), order="F")


def reshapeF(A, shape):
    return np.reshape(A, tuple(int(s) for s in shape), order="F")


def msize(A, d):
    """size(A, d) with 1-based d (1 beyond ndims)."""
    return A.shape[d - 1] if d <= A.ndim else 1


def as_list(v):
    return [int(x) for x in np.atleast_1d(v)]


def subsets(X):
    """dpca_marginalize.m :: subsets  -- dec2bin / sort / fliplr ordering."""
    X = list(X)
    d = len(X)
    pc = np.array([[int(b) for b in format(k, "0%db" % d)] for k in range(1, 2 ** d)])
    ind = np.argsort(pc.sum(1), kind="stable")
    pc = np.fliplr(pc[ind, :])
    return [[X[j] for j in np.nonzero(row)[0]] for row in pc]


def nanmmean(X, dimlist):
    """dpca_marginalize.m :: nanmmean  (dimlist 1-based)"""
    if len(dimlist) == 0:
        return X
    dims = X.shape
    dimrest = [d for d in range(1, len(dims) + 1) if d not in dimlist]
    perm = [d - 1 for d in dimrest + list(dimlist)]
    Y = np.transpose(X, perm)
    Y = reshapeF(Y, [dims[d - 1] for d in dimrest] + [int(np.prod([dims[d - 1] for d in dimlist]))])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Y = np.nanmean(Y, axis=Y.ndim - 1, keepdims=True)
    # ipermute: Y now has shape [dims(dimrest) 1]; append singleton dims for the rest
    Y = Y.reshape(Y.shape + (1,) * (len(dimlist) - 1))
    return np.transpose(Y, np.argsort(perm))


# --------------------------------------------------------------------------- dpca_marginalize.m
def dpca_marginalize(X, combinedParams=None, ifFull="yes", ifFlat="no",
                     timeSplits=None, timeParameter=None, notToSplit=None):
    X = X - np.mean(flat2(X), 1).reshape((-1,) + (1,) * (X.ndim - 1))   # bsxfun(@minus, X, mean(X(:,:),2))

    params = list(range(2, X.ndim + 1))
    paramsubsets = subsets(params)

    alreadyProcessed = {}
    YY = []
    for subs in range(len(paramsubsets)):
        indRest = paramsubsets[subs]
        indMarg = sorted(set(params) - set(indRest))
        YY.append(nanmmean(X, indMarg))
        alreadyProcessed[str(sorted(indMarg))] = subs

        paramsubsubsets = subsets(indRest)
        for s in range(len(paramsubsubsets) - 1):
            indM = paramsubsubsets[s] + indMarg
            YY[subs] = YY[subs] + -YY[alreadyProcessed[str(sorted(indM))]]

    paramsubsets = subsets([p - 1 for p in params])
    margNums = list(range(1, len(YY) + 1))

    if combinedParams:
        YYY = []
        for i in range(len(combinedParams)):
            margsToAdd = []
            for j in range(len(combinedParams[i])):
                cj = sorted(as_list(combinedParams[i][j]))
                for k in range(len(paramsubsets)):
                    if len(paramsubsets[k]) == len(cj) and sorted(paramsubsets[k]) == cj:
                        margsToAdd.append(k)
            Y = YY[margsToAdd[0]]
            for j in range(1, len(margsToAdd)):
                Y = Y + YY[margsToAdd[j]]
            YYY.append(Y)
        YY = YYY
        paramsubsets = combinedParams
        margNums = list(range(1, len(combinedParams) + 1))

    if ifFull == "yes":
        YY = [Y * np.ones(X.shape) for Y in YY]

    if timeSplits is not None and len(timeSplits) > 0:
        ZZ, margNums = [], []
        timeSplitsBeg = [1] + list(timeSplits)
        timeSplitsEnd = [t - 1 for t in timeSplits] + [msize(X, X.ndim)]
        # same format as combinedParams: "if ~iscell(paramsubsets{k}), paramsubsets{k} = {paramsubsets{k}}"
        # (without combinedParams the entries are numeric vectors, with it they are cell arrays)
        if not combinedParams:
            paramsubsets = [[ps] for ps in paramsubsets]
        for i in range(len(YY)):
            if notToSplit:
                toSkip = False
                for k in range(len(notToSplit)):
                    if len(paramsubsets[i]) == len(notToSplit[k]):
                        identical = True
                        for zz in range(len(paramsubsets[i])):
                            found = False
                            for zzz in range(len(notToSplit[k])):
                                a = sorted(as_list(paramsubsets[i][zz]))
                                b = sorted(as_list(notToSplit[k][zzz]))
                                if len(a) == len(b) and a == b:
                                    found = True
                                    break
                            if not found:
                                identical = False
                                break
                        if identical:
                            toSkip = True
                            break
                if toSkip:
                    ZZ.append(YY[i])
                    margNums.append(i + 1)
                    continue
            for split in range(len(timeSplits) + 1):
                ind = [slice(None)] * X.ndim
                ind[timeParameter] = slice(timeSplitsBeg[split] - 1, timeSplitsEnd[split])
                Z = np.zeros(YY[i].shape)
                Z[tuple(ind)] = YY[i][tuple(ind)]
                ZZ.append(Z)
                margNums.append(i + 1)
        YY = ZZ

    if ifFlat == "yes":
        YY = [flat2(Y) for Y in YY]
    return YY, np.array(margNums)


# --------------------------------------------------------------------------- dpca.m
def _mrdivide_with_fallback(B, Xc, Cnoise, totalVar, thisLambda):
    """C = B/(X*X' + Cnoise + (totalVar*lambda)^2*eye) with MATLAB's singular-matrix catch."""
    N = Xc.shape[0]

    def build(lam):
        return Xc @ Xc.T + Cnoise + (totalVar * lam) ** 2 * np.eye(N)

    def near_singular(A):
        try:
            rc = 1.0 / np.linalg.cond(A, 1)
        except np.linalg.LinAlgError:
            return True
        return not np.isfinite(rc) or rc < EPS

    A = build(thisLambda)
    if near_singular(A):                          # try ... catch: retry with lambda = 1e-10
        thisLambda = 1e-10
        A = build(thisLambda)
        if near_singular(A):                      # retry is still under warning('error', ...) -> uncaught
            raise np.linalg.LinAlgError("MATLAB:nearlySingularMatrix")
    return np.linalg.solve(A.T, B.T).T


def dpca(Xfull, numComps, combinedParams=None, lambda_=0, order="yes", timeSplits=None,
         timeParameter=None, notToSplit=None, scale="no", Cnoise=None):
    X = flat2(Xfull)
    X = X - np.mean(X, 1, keepdims=True)
    XfullCen = reshapeF(X, Xfull.shape)
    totalVar = np.sum(colon(X) ** 2)

    Xmargs, margNums = dpca_marginalize(XfullCen, combinedParams=combinedParams, timeSplits=timeSplits,
                                        timeParameter=timeParameter, notToSplit=notToSplit, ifFlat="yes")
    numComps = np.atleast_1d(numComps)
    lambdas = np.atleast_1d(lambda_)

    decoder, encoder, whichMarg = [], [], []
    if Cnoise is None or (np.ndim(Cnoise) > 0 and np.size(Cnoise) == 0):
        Cnoise = np.zeros((X.shape[0], X.shape[0]))

    for i in range(len(Xmargs)):
        nc = int(numComps[0]) if len(numComps) == 1 else int(numComps[margNums[i] - 1])
        thisLambda = lambdas[0] if len(lambdas) == 1 else lambdas[margNums[i] - 1]
        if nc == 0:
            continue
        C = _mrdivide_with_fallback(Xmargs[i] @ X.T, X, Cnoise, totalVar, thisLambda)
        M = C @ X
        w, U = np.linalg.eigh(M @ M.T)
        idx = np.argsort(-np.abs(w), kind="stable")[:nc]          # eigs(M*M', nc): largest magnitude
        U = U[:, idx]
        P = U
        D = U.T @ C
        if scale == "yes":
            for uu in range(D.shape[0]):
                A = Xmargs[i]
                B = np.outer(P[:, uu], D[uu, :]) @ X
                scalingFactor = (colon(A) @ colon(B)) / (colon(B) @ colon(B))
                D[uu, :] = scalingFactor * D[uu, :]
        decoder.append(D)
        encoder.append(P)
        whichMarg += [i + 1] * nc

    V = np.hstack(encoder)
    W = np.vstack(decoder).T
    whichMarg = np.array(whichMarg)

    toFlip = np.nonzero(np.sum(np.sign(V), 0) < 0)[0]
    W[:, toFlip] = -W[:, toFlip]
    V[:, toFlip] = -V[:, toFlip]

    explVar = np.zeros(0)
    if timeSplits is not None and len(timeSplits) > 0:
        toKeep = []
        for i in range(1, int(margNums.max()) + 1):
            components = np.nonzero(np.isin(whichMarg, np.nonzero(margNums == i)[0] + 1))[0]
            Z = W[:, components].T @ X
            explVar = np.sum(Z ** 2, 1)
            o = np.argsort(-explVar, kind="stable")
            nc = int(numComps[0]) if len(numComps) == 1 else int(numComps[i - 1])
            toKeep += list(components[o[:nc]])
        W, V = W[:, toKeep], V[:, toKeep]
        whichMarg = whichMarg[toKeep]
        whichMarg = margNums[whichMarg - 1]

    if len(numComps) == 1 or order == "yes":
        explVar = np.array(explVar, dtype=float)
        for i in range(W.shape[1]):
            Z = X - np.outer(V[:, i], W[:, i] @ X)
            val = 1 - np.sum(colon(Z) ** 2) / totalVar
            if i < len(explVar):
                explVar[i] = val
            else:
                explVar = np.append(explVar, val)
        o = np.argsort(-explVar, kind="stable")
        L = int(numComps[0]) if len(numComps) == 1 else int(np.sum(numComps))
        W = W[:, o[:L]]
        V = V[:, o[:L]]
        whichMarg = whichMarg[o[:L]]
    return W, V, whichMarg


# --------------------------------------------------------------------------- dpca_explainedVariance.m
def dpca_explainedVariance(Xfull, W, V, combinedParams=None, X_trial=None, numOfTrials=None,
                           Cnoise=None, rng=None):
    X = flat2(Xfull)
    Xfull = Xfull - np.mean(X, 1).reshape((-1,) + (1,) * (Xfull.ndim - 1))
    X = X - np.mean(X, 1, keepdims=True)

    Xmargs, _ = dpca_marginalize(Xfull, combinedParams=combinedParams, ifFlat="yes")
    ev = {}
    ev["totalVar"] = np.sum(X ** 2)
    ev["totalMarginalizedVar"] = np.array([np.sum(colon(m) ** 2) for m in Xmargs])

    S = np.linalg.svd(X.T, compute_uv=False)
    S = S[: W.shape[1]]
    ev["cumulativePCA"] = np.cumsum(S ** 2 / ev["totalVar"] * 100)

    Z = W.T @ X
    k = W.shape[1]
    ev["cumulativeDPCA"] = np.zeros(k)
    ev["componentVar"] = np.zeros(k)
    ev["margVar"] = np.zeros((len(Xmargs), k))
    for i in range(k):
        ev["cumulativeDPCA"][i] = 100 - np.sum((X - V[:, : i + 1] @ Z[: i + 1, :]) ** 2) / ev["totalVar"] * 100
        ev["componentVar"][i] = 100 - np.sum((X - np.outer(V[:, i], Z[i, :])) ** 2) / ev["totalVar"] * 100
        for j in range(len(Xmargs)):
            ZZ = Xmargs[j] - np.outer(V[:, i], W[:, i] @ Xmargs[j])
            ev["margVar"][j, i] = (ev["totalMarginalizedVar"][j] - np.sum(colon(ZZ) ** 2)) / ev["totalVar"] * 100

    if np.max(np.abs(ev["cumulativePCA"] - ev["cumulativeDPCA"])) < 1e-10:
        ev["cumulativeDPCA"] = np.zeros(0)

    if X_trial is not None and numOfTrials is not None:
        dims = X_trial.shape
        neuronsConditions = colon(numOfTrials)
        ind1 = np.zeros(len(neuronsConditions), dtype=int)
        ind2 = np.zeros(len(neuronsConditions), dtype=int)
        for i in range(len(neuronsConditions)):
            tmp = randperm(rng, neuronsConditions[i], 2)
            ind1[i] = tmp[0]
            ind2[i] = tmp[1]
        ind1 = reshapeF(ind1, numOfTrials.shape)
        ind2 = reshapeF(ind2, numOfTrials.shape)
        ind1 = colon(np.ones(dims[:-1]) * ind1[..., None]).astype(int)
        ind2 = colon(np.ones(dims[:-1]) * ind2[..., None]).astype(int)
        P = int(np.prod(dims[:-1]))
        Xt2 = reshapeF(X_trial, (P, dims[-1]))
        dif = Xt2[np.arange(P), ind1 - 1] - Xt2[np.arange(P), ind2 - 1]
        XfullNN = np.ones(dims[:-1]) * numOfTrials[..., None]
        dif = dif / np.sqrt(2 * colon(XfullNN))
        XnoiseFull = reshapeF(dif, dims[:-1])

        Xnoise = flat2(XnoiseFull)
        XnoiseFull = XnoiseFull - np.mean(Xnoise, 1).reshape((-1,) + (1,) * (XnoiseFull.ndim - 1))
        Xnoise = Xnoise - np.mean(Xnoise, 1, keepdims=True)
        XmargsNoise, _ = dpca_marginalize(XnoiseFull, combinedParams=combinedParams)

        ev["totalVar_noise"] = np.sum(Xnoise ** 2)
        ev["totalVar_signal"] = ev["totalVar"] - ev["totalVar_noise"]
        marginalizedVarNoise = np.array([np.sum(colon(m) ** 2) for m in XmargsNoise])
        ev["totalMarginalizedVar_signal"] = ev["totalMarginalizedVar"] - marginalizedVarNoise

    if Cnoise is not None and numOfTrials is not None:
        Ktilde = np.mean(reshapeF(numOfTrials, (numOfTrials.shape[0], -1)), 1)
        ev["totalVar_noise"] = np.sum(np.diag(Cnoise) / Ktilde)
        ev["totalVar_signal"] = ev["totalVar"] - ev["totalVar_noise"]
        S_, Q_, T_ = msize(Xfull, 2), msize(Xfull, 3), msize(Xfull, 4)
        vec = np.array([S_ * T_ - T_, Q_ * T_ - T_, T_ - 1, S_ * Q_ * T_ - S_ * T_ - Q_ * T_ + T_]) / (S_ * Q_ * T_ - 1)
        if len(vec) != len(ev["totalMarginalizedVar"]) and len(ev["totalMarginalizedVar"]) != 1:
            raise ValueError("Matrix dimensions must agree.")
        ev["totalMarginalizedVar_signal"] = ev["totalMarginalizedVar"] - vec * ev["totalVar_noise"]
    return ev


# --------------------------------------------------------------------------- dpca_getNoiseCovariance.m
def dpca_getNoiseCovariance(Xfull, Xtrial, numOfTrials, simultaneous=False, type="pooled"):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if not simultaneous:
            SSnoise = np.nansum(Xtrial ** 2, axis=Xtrial.ndim - 1) - Xfull ** 2 * numOfTrials[..., None]
            SSnoiseSumOverT = np.sum(SSnoise, axis=SSnoise.ndim - 1)
            numOfTrialsAverage = flat2(numOfTrials).astype(float)
            numOfTrialsAverage[numOfTrialsAverage == 0] = np.nan
            numOfTrialsAverage = np.nanmean(numOfTrialsAverage, 1)
            if type == "pooled":
                return np.diag(np.nansum(flat2(SSnoiseSumOverT), 1) * (1.0 / numOfTrialsAverage))
            elif type == "averaged":
                return np.diag(np.nansum(flat2(SSnoiseSumOverT) * (1.0 / flat2(numOfTrials)), 1))
            raise ValueError("Output argument Cnoise not assigned")
        else:
            if type == "pooled":
                Xnoise = Xtrial - Xfull[..., None]
                Xnoise = flat2(Xnoise)
                Xnoise = Xnoise[:, ~np.isnan(Xnoise[0, :])]
                SSnoise = Xnoise @ Xnoise.T
                Cnoise = SSnoise / Xnoise.shape[1]
                dimProd = np.prod(Xfull.shape[1:])
                return Cnoise * dimProd
            raise ValueError("Averaged noise covariance not implemented for simultaneous data "
                             "(MATLAB: output argument Cnoise not assigned)")


# --------------------------------------------------------------------------- dpca_getTestTrials.m
def dpca_getTestTrials(firingRatesPerTrial, numOfTrials, rng, simultaneous=False):
    dim = firingRatesPerTrial.shape
    if not simultaneous:
        neuronsConditions = colon(numOfTrials)
        testTrials = np.ceil(rand(rng, len(neuronsConditions)) * neuronsConditions)
    else:
        neuronsConditions = colon(numOfTrials)
        neuronsConditions = neuronsConditions[:: numOfTrials.shape[0]]
        testTrials = np.ceil(rand(rng, len(neuronsConditions)) * neuronsConditions)
        testTrials = np.ones((numOfTrials.shape[0], 1)) * testTrials[None, :]
        testTrials = colon(testTrials)

    ind = reshapeF(testTrials, numOfTrials.shape)
    ind = np.ones(dim[:-1]) * ind[..., None]
    ind = colon(ind).astype(int)
    P = int(np.prod(dim[:-1]))
    flat = reshapeF(firingRatesPerTrial, (P, dim[-1]))
    Xtest = flat[np.arange(P), ind - 1]
    Xtest = reshapeF(Xtest, dim[:-1])

    flat = flat.copy()
    flat[np.arange(P), ind - 1] = np.nan
    Xtrain = reshapeF(flat, dim)
    return Xtest, Xtrain


# --------------------------------------------------------------------------- dpca_optimizeLambda.m
def dpca_optimizeLambda(Xfull, Xtrial, numOfTrials, rng, numComps=25, lambdas=None, numRep=10,
                        combinedParams=None, method="training", simultaneous=False, noiseCovType="pooled"):
    if lambdas is None:
        lambdas = 1e-07 * 1.5 ** np.arange(0, 26)
    if np.min(numOfTrials) <= 0:
        raise ValueError("dPCA:tooFewTrials0")
    elif np.min(numOfTrials) == 1:
        raise ValueError("dPCA:tooFewTrials1")

    Xsum = Xfull * numOfTrials[..., None]
    nM = None
    errors = np.zeros((len(lambdas), numRep))
    errorsMarg = None
    for rep in range(numRep):
        Xtest, XtrainFull = dpca_getTestTrials(Xtrial, numOfTrials, rng, simultaneous=simultaneous)
        Xtrain = (Xsum - Xtest) * (1.0 / (numOfTrials[..., None] - 1))

        if noiseCovType != "none":
            CnoiseTrain = dpca_getNoiseCovariance(Xtrain, XtrainFull, numOfTrials - 1,
                                                  simultaneous=simultaneous, type=noiseCovType)
        else:
            CnoiseTrain = 0

        XtestCen = Xtest - np.mean(flat2(Xtest), 1).reshape((-1,) + (1,) * (Xtest.ndim - 1))
        XtestMargs, _ = dpca_marginalize(XtestCen, combinedParams=combinedParams, ifFlat="yes")
        margTestVar = np.array([np.sum(colon(m) ** 2) for m in XtestMargs])

        XtrainCen = Xtrain - np.mean(flat2(Xtrain), 1).reshape((-1,) + (1,) * (Xtrain.ndim - 1))
        XtrainMargs, _ = dpca_marginalize(XtrainCen, combinedParams=combinedParams, ifFlat="yes")
        margTrainVar = np.array([np.sum(colon(m) ** 2) for m in XtrainMargs])

        margVar_toNormalize = margTestVar if method in ("naive", "neuronwise") else margTrainVar
        if errorsMarg is None:
            nM = len(XtestMargs)
            errorsMarg = np.zeros((nM, len(lambdas), numRep))

        Xt2 = flat2(XtestCen)
        for l in range(len(lambdas)):
            W, V, whichMarg = dpca(Xtrain, numComps, combinedParams=combinedParams,
                                   lambda_=lambdas[l], Cnoise=CnoiseTrain)
            cumError = 0
            for i in range(len(XtestMargs)):
                sel = whichMarg == i + 1
                if method == "naive":
                    recError = np.sum((XtestMargs[i] - V[:, sel] @ W[:, sel].T @ Xt2) ** 2)
                elif method == "training":
                    recError = np.sum((XtrainMargs[i] - V[:, sel] @ W[:, sel].T @ Xt2) ** 2)
                elif method == "neuronwise":
                    diagVW = np.diag(np.sum(V[:, sel] * W[:, sel], 1))
                    recError = np.sum((XtestMargs[i] - V[:, sel] @ W[:, sel].T @ Xt2 + diagVW @ Xt2) ** 2)
                else:
                    recError = 0
                errorsMarg[i, l, rep] = recError / margVar_toNormalize[i]
                cumError = cumError + recError
            errors[l, rep] = cumError / np.sum(margVar_toNormalize)

    meanError = np.mean(errors, 1)
    ind = int(np.argmin(meanError))
    optimalLambda = lambdas[ind]
    meanErrorMarg = np.mean(errorsMarg, 2)
    indm = np.argmin(meanErrorMarg, 1)
    optimalLambdas = np.asarray(lambdas)[indm]
    return optimalLambda, optimalLambdas, errors, errorsMarg


# --------------------------------------------------------------------------- dpca_classificationAccuracy.m
def _matlab_if_true(v):
    v = np.atleast_1d(v)
    return v.size > 0 and bool(np.all(v != 0))


def dpca_classificationAccuracy(Xfull, Xtrial, numOfTrials, rng, numComps=3, lambda_=0, numRep=100,
                                combinedParams=None, decodingClasses=None, timeSplits=None,
                                timeParameter=None, notToSplit=None, simultaneous=False,
                                noiseCovType="pooled"):
    Xsum = Xfull * numOfTrials[..., None]
    Cnoise = dpca_getNoiseCovariance(Xfull, Xtrial, numOfTrials, type=noiseCovType, simultaneous=simultaneous)

    timeComp = []
    if combinedParams:
        for k in range(len(combinedParams)):
            if _matlab_if_true(np.atleast_1d(combinedParams[k][0]) == Xfull.ndim - 1):
                timeComp = [k + 1]
                break
    else:
        timeComp = [Xfull.ndim - 1]

    nMarg = len(combinedParams) if combinedParams else 2 ** (Xfull.ndim - 1) - 1
    numCompsToUse = np.tile(np.atleast_1d(numComps), nMarg)        # repmat(numComps, [1 nMarg])
    for tc in timeComp:
        numCompsToUse[tc - 1] = 0

    if decodingClasses is None:
        dims = Xfull.shape[1:-1]
        # NOTE: MATLAB's reshape(1:prod(dims), dims) errors when dims is a scalar (2-parameter data);
        # the evident intent (one class per condition) is used here.
        classes = reshapeF(np.arange(1, int(np.prod(dims)) + 1), dims)
        decodingClasses = [classes for _ in range(nMarg)]

    dim = Xfull.shape
    _, Vfull, _ = dpca(Xfull, numCompsToUse, combinedParams=combinedParams, lambda_=lambda_,
                       timeSplits=timeSplits, timeParameter=timeParameter, notToSplit=notToSplit,
                       order="no", Cnoise=Cnoise)

    correct, brier = None, None
    for rep in range(numRep):
        Xtest, XtrainFull = dpca_getTestTrials(Xtrial, numOfTrials, rng, simultaneous=simultaneous)
        Xtrain = (Xsum - Xtest) * (1.0 / (numOfTrials[..., None] - 1))
        CnoiseTrain = dpca_getNoiseCovariance(Xtrain, XtrainFull, numOfTrials - 1,
                                              simultaneous=simultaneous, type=noiseCovType)
        W, V, whichMarg = dpca(Xtrain, numCompsToUse, combinedParams=combinedParams, lambda_=lambda_,
                               timeSplits=timeSplits, timeParameter=timeParameter, notToSplit=notToSplit,
                               order="no", Cnoise=CnoiseTrain)

        if correct is None:
            margs_all = sorted(set(whichMarg.tolist()) | set(timeComp))
            n1 = max(margs_all)
            n2 = max(np.sum(whichMarg == m) for m in set(whichMarg.tolist()))
            correct = np.zeros((n1, n2, dim[-1], numRep))
            brier = np.zeros((n1, n2, dim[-1], numRep))

        cc = np.corrcoef(np.hstack([V, Vfull]), rowvar=False)
        crosscorr = np.abs(cc[V.shape[1]:, : V.shape[1]])
        for marg in sorted(set(np.unique(whichMarg).tolist()) - set(timeComp)):
            d = np.nonzero(whichMarg == marg)[0]
            crc = crosscorr[np.ix_(d, d)]
            order = np.argmax(crc, axis=0)       # [~, order] = max(crc)  (greedy result is discarded)
            W[:, d] = W[:, d[order]]
            V[:, d] = V[:, d[order]]

        Xtrain2D = flat2(Xtrain)
        centeringTrain = np.mean(Xtrain2D, 1, keepdims=True)
        Xtrain2D = Xtrain2D - centeringTrain
        Xtest2D = flat2(Xtest) - centeringTrain

        for marg in sorted(set(np.unique(whichMarg).tolist()) - set(timeComp)):
            dd = np.nonzero(whichMarg == marg)[0]
            cls = colon(decodingClasses[marg - 1]).astype(int)
            nUnique = len(np.unique(cls))
            for d in range(len(dd)):
                XtrainComp = reshapeF(W[:, dd[d]] @ Xtrain2D, dim[1:])
                XtestComp = reshapeF(W[:, dd[d]] @ Xtest2D, dim[1:])
                for t in range(dim[-1]):
                    trainClasses = colon(XtrainComp[..., t])
                    trainClassMeans = np.zeros(cls.max())
                    for c in range(1, cls.max() + 1):
                        if np.any(cls == c):
                            trainClassMeans[c - 1] = np.mean(trainClasses[cls == c])
                    testValues = colon(XtestComp[..., t])
                    dist = testValues[:, None] - trainClassMeans[None, :]
                    classification = np.argmin(np.abs(dist), axis=1) + 1
                    correct[marg - 1, d, t, rep] = np.sum(classification == cls)

                    with np.errstate(divide="ignore", invalid="ignore"):
                        prob = 1.0 / np.abs(dist)
                        prob = prob * (1.0 / np.sum(prob, 1, keepdims=True))
                    actual = np.zeros(prob.shape)
                    actual[np.arange(actual.shape[0]), cls - 1] = 1
                    brier[marg - 1, d, t, rep] = np.sum((prob - actual) ** 2) / nUnique

    numConditions = int(np.prod(dim[1:-1]))
    accuracy = np.sum(correct, 3) * (1.0 / (numRep * numConditions))
    brier = np.sum(brier, 3) * (1.0 / (numRep * numConditions))
    for tc in timeComp:
        accuracy[tc - 1, :, :] = np.nan
        brier[tc - 1, :, :] = np.nan
    return accuracy, brier


# --------------------------------------------------------------------------- dpca_classificationShuffled.m
def dpca_classificationShuffled(Xtrial, numOfTrials, rng, numShuffles=100, lambda_=0, numRep=100,
                                combinedParams=None, decodingClasses=None, timeSplits=None,
                                timeParameter=None, notToSplit=None, simultaneous=False,
                                noiseCovType="pooled"):
    D = numOfTrials.shape[0]
    dim = Xtrial.shape
    T = dim[-2]
    maxTrialN = dim[-1]
    numCond = int(np.prod(dim[1:-2]))

    numOfTrialsCond = reshapeF(numOfTrials, (D, -1))
    trialsMissing = np.zeros((D, maxTrialN * numCond))
    for n in range(D):
        this = np.zeros((numCond, maxTrialN))
        for c in range(numCond):
            this[c, :] = np.r_[np.zeros(numOfTrialsCond[n, c]), np.ones(maxTrialN - numOfTrialsCond[n, c])]
        trialsMissing[n, :] = colon(this)

    orderDim = list(range(Xtrial.ndim))
    orderDim[-2], orderDim[-1] = orderDim[-1], orderDim[-2]
    XtrialCond = np.transpose(Xtrial, orderDim)
    XtrialCond = reshapeF(XtrialCond, (D, -1, T))

    accuracyShuffle, brierShuffle = [], []
    for shuffle in range(numShuffles):
        XtrialCondShuffle = np.zeros(XtrialCond.shape)
        if not simultaneous:
            for n in range(D):
                presentTrials = np.nonzero(trialsMissing[n, :] == 0)[0]
                shuffledOrder = presentTrials[randperm(rng, len(presentTrials)) - 1]
                XtrialCondShuffle[n, presentTrials, :] = XtrialCond[n, shuffledOrder, :]
        else:
            presentTrials = np.nonzero(trialsMissing[0, :] == 0)[0]
            shuffledOrder = presentTrials[randperm(rng, len(presentTrials)) - 1]
            XtrialCondShuffle[:, presentTrials, :] = XtrialCond[:, shuffledOrder, :]
        XtrialShuffle = np.transpose(reshapeF(XtrialCondShuffle, [dim[o] for o in orderDim]), orderDim)

        firingRatesAverageShuffle = np.sum(XtrialShuffle, XtrialShuffle.ndim - 1)
        firingRatesAverageShuffle = firingRatesAverageShuffle * (1.0 / numOfTrials[..., None])

        accuracy, brier = dpca_classificationAccuracy(
            firingRatesAverageShuffle, XtrialShuffle, numOfTrials, rng, numComps=1,
            combinedParams=combinedParams, lambda_=lambda_, numRep=numRep, decodingClasses=decodingClasses,
            timeSplits=timeSplits, timeParameter=timeParameter, notToSplit=notToSplit,
            simultaneous=simultaneous, noiseCovType=noiseCovType)
        accuracyShuffle.append(accuracy[:, 0, :])           # squeeze(accuracy(:,1,:)) for nMarg, T > 1
        brierShuffle.append(brier[:, 0, :])
    return np.stack(accuracyShuffle, -1), np.stack(brierShuffle, -1)


# --------------------------------------------------------------------------- dpca_signifComponents.m
def dpca_signifComponents(accuracy, accuracyShuffle, whichMarg, minChunk=10):
    whichMarg = np.asarray(whichMarg)
    componentsSignif = np.zeros((len(whichMarg), accuracy.shape[2]))
    for marg in range(1, accuracy.shape[0] + 1):
        if np.isnan(accuracy[marg - 1, 0, 0]):
            continue
        comps = np.nonzero(whichMarg == marg)[0][:3]
        for num in range(len(comps)):
            componentsSignif[comps[num], :] = accuracy[marg - 1, num, :] > np.max(accuracyShuffle[marg - 1, :, :], 1)

    for i in range(componentsSignif.shape[0]):
        s = componentsSignif[i, :].copy()
        ns = np.r_[0, np.nonzero(s == 0)[0] + 1, len(s) + 1]          # 1-based positions
        dns = np.diff(ns)
        tooShort = np.nonzero((dns > 1) & (dns < minChunk))[0]
        for t in tooShort:
            s[ns[t]: ns[t + 1] - 1] = 0                                   # s(ns(t)+1 : ns(t+1)-1) = 0
        componentsSignif[i, :] = s
    return componentsSignif
