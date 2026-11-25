"""Run the real MATLAB reference code (../../matlab/*.m) under GNU Octave from Python.

* Inputs are passed through a .mat file; outputs come back through another .mat file.
* Random numbers: the Python call under test is run while recording every draw it makes via
  dPCA.core.rand / dPCA.core.randperm. The draws are handed to Octave, where shims for
  ``rand`` (inside dpca_getTestTrials) and ``randperm`` (inside dpca_classificationShuffled and
  dpca_explainedVariance) replay them in order. A different number, order or size of draws in
  the MATLAB code raises an error, so the draw protocol itself is verified.
* ``nanmean`` / ``nansum`` (Statistics Toolbox in MATLAB, absent from Octave core) are provided
  by minimal shims in octave_shims/.
* dpca.m turns 'MATLAB:singularMatrix' / 'MATLAB:nearlySingularMatrix' warnings into errors; the
  Octave equivalents ('Octave:singular-matrix', 'Octave:nearly-singular-matrix') are turned
  into errors here so the same try/catch fallback is taken.
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np
import scipy.io

HERE = os.path.dirname(os.path.abspath(__file__))
MATLAB_DIR = os.environ.get("DPCA_MATLAB_DIR") or os.path.normpath(os.path.join(HERE, "..", "..", "matlab"))
SHIMS = os.path.join(HERE, "octave_shims")
OCTAVE = shutil.which("octave-cli") or shutil.which("octave")


class OctaveError(RuntimeError):
    def __init__(self, identifier, message):
        super().__init__("%s: %s" % (identifier, message))
        self.identifier = identifier


class Cell(list):
    """A MATLAB cell array literal."""


class Raw(str):
    """Verbatim MATLAB expression."""


def cell_groups(groups):
    """combinedParams / notToSplit (list of groups of parameter subsets) -> nested Cell."""
    if groups is None:
        return None
    return Cell(Cell(np.atleast_1d(np.asarray(s, float)) for s in g) for g in groups)


def cell_of(items):
    return Cell(items)


def record_draws(fn):
    """Run fn() while recording all random draws made through dPCA.core."""
    from dPCA import core
    log = []
    orig_rand, orig_perm = core.rand, core.randperm

    def rand(rng, n):
        v = orig_rand(rng, n)
        log.append((1, np.asarray(v, float)))
        return v

    def randperm(rng, n, k=None):
        v = orig_perm(rng, n, k)
        log.append((2, np.asarray(v, float)))
        return v

    core.rand, core.randperm = rand, randperm
    try:
        out = fn()
    finally:
        core.rand, core.randperm = orig_rand, orig_perm
    return out, log


def call(func, *args, nargout=1, draws=None, timeout=1800):
    """Call MATLAB function ``func`` in Octave. ``args`` are positional values; name/value pairs
    are given as consecutive positional values, exactly as in MATLAB. Returns a list of outputs
    (loaded with scipy.io.loadmat(simplify_cells=True)). Raises OctaveError on MATLAB errors or if
    the recorded random draws were not all consumed."""
    if OCTAVE is None:
        raise RuntimeError("Octave not found")
    tmp = tempfile.mkdtemp(prefix="dpca_oct_")
    try:
        inputs = {}

        def lit(v):
            if isinstance(v, Raw):
                return str(v)
            if v is None:
                return "[]"
            if isinstance(v, str):
                return "'%s'" % v
            if isinstance(v, (bool, np.bool_)):
                return "true" if v else "false"
            if isinstance(v, Cell):
                return "{" + ", ".join(lit(x) for x in v) + "}"
            if isinstance(v, (int, float, np.integer, np.floating)):
                name = "in%d" % len(inputs)
                inputs[name] = np.array(float(v))
                return name
            arr = np.asarray(v, dtype=float)
            name = "in%d" % len(inputs)
            inputs[name] = arr if arr.ndim >= 2 else arr.reshape(1, -1)
            return name

        arg_src = ", ".join(lit(a) for a in args)
        draws = draws or []
        if draws:
            q = np.empty(len(draws), dtype=object)
            for i, (_, v) in enumerate(draws):
                q[i] = np.asarray(v, float).reshape(1, -1)
            inputs["DPCA_Q_in"] = q
            inputs["DPCA_K_in"] = np.array([k for k, _ in draws], float).reshape(1, -1)
        in_file = os.path.join(tmp, "in.mat")
        out_file = os.path.join(tmp, "out.mat")
        scipy.io.savemat(in_file, inputs, oned_as="row")

        outs = ", ".join("o%d" % i for i in range(nargout))
        script = f"""
addpath('{MATLAB_DIR}'); addpath('{SHIMS}');
warning('off', 'Octave:shadowed-function'); save_default_options('-mat7-binary');
warning('error', 'Octave:singular-matrix'); warning('error', 'Octave:nearly-singular-matrix');
global DPCA_Q DPCA_K DPCA_I
load('{in_file}');
if exist('DPCA_Q_in', 'var'), DPCA_Q = DPCA_Q_in; DPCA_K = DPCA_K_in; else, DPCA_Q = {{}}; DPCA_K = []; end
DPCA_I = 0;
try
  [{outs}] = {func}({arg_src});
  draws_used = DPCA_I; draws_total = numel(DPCA_Q);
  save('-mat7-binary', '{out_file}', {", ".join("'o%d'" % i for i in range(nargout))}, 'draws_used', 'draws_total');
catch err
  err_id = err.identifier; err_msg = err.message;
  save('-mat7-binary', '{out_file}', 'err_id', 'err_msg');
end
"""
        script_file = os.path.join(tmp, "run.m")
        with open(script_file, "w") as f:
            f.write(script)
        r = subprocess.run([OCTAVE, "--no-gui", "--quiet", "--norc", script_file], cwd=tmp,
                           capture_output=True, text=True, timeout=timeout)
        if not os.path.exists(out_file):
            raise RuntimeError("Octave failed:\n" + r.stdout[-3000:] + r.stderr[-3000:])
        res = scipy.io.loadmat(out_file, simplify_cells=True)
        if "err_id" in res:
            raise OctaveError(str(res["err_id"]), str(res["err_msg"]))
        if int(res["draws_used"]) != int(res["draws_total"]):
            raise OctaveError("dpcatest:queue", "MATLAB used %d of %d recorded random draws"
                              % (res["draws_used"], res["draws_total"]))
        return [res["o%d" % i] for i in range(nargout)]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def same(py, m, rtol=1e-7, atol=1e-9):
    """Compare a Python result with a MATLAB result, ignoring singleton-dimension differences
    (MATLAB row vectors, dropped trailing singletons). Column-major element order on both sides."""
    a = np.asarray(py, dtype=float)
    b = np.asarray(m, dtype=float)
    sa = tuple(s for s in a.shape if s != 1)
    sb = tuple(s for s in b.shape if s != 1)
    assert sa == sb, "shape %s (python) vs %s (MATLAB)" % (a.shape, b.shape)
    scale = max(1.0, float(np.nanmax(np.abs(b)))) if b.size and not np.all(np.isnan(b)) else 1.0
    np.testing.assert_allclose(a.ravel(order="F"), b.ravel(order="F"), rtol=rtol, atol=atol * scale,
                               equal_nan=True)
