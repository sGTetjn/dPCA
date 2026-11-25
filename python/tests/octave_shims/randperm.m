function p = randperm(n, varargin)
% Test shim: inside the dPCA shuffling / explained-variance code, return the next permutation
% recorded from the Python run; elsewhere use the builtin.
  global DPCA_Q DPCA_K DPCA_I
  st = dbstack;
  if numel(st) >= 2 && any(strcmp(st(2).name, {'dpca_classificationShuffled', 'dpca_explainedVariance'}))
    DPCA_I = DPCA_I + 1;
    if DPCA_I > numel(DPCA_Q) || DPCA_K(DPCA_I) ~= 2
      error('dpcatest:queue', 'randperm call #%d does not match the Python draw sequence', DPCA_I);
    end
    v = DPCA_Q{DPCA_I};
    if isempty(varargin), k = n; else, k = varargin{1}; end
    if numel(v) ~= k || any(v > n) || numel(unique(v)) ~= numel(v)
      error('dpcatest:queue', 'randperm call #%d: (n=%d, k=%d) inconsistent with the Python draw', DPCA_I, n, k);
    end
    p = reshape(v, 1, []);
  else
    p = builtin('randperm', n, varargin{:});
  end
end
