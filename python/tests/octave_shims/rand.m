function r = rand(varargin)
% Test shim: inside dpca_getTestTrials, return the next uniform draw vector recorded from the
% Python run (so both implementations see identical random numbers); elsewhere use the builtin.
  global DPCA_Q DPCA_K DPCA_I
  st = dbstack;
  if numel(st) >= 2 && strcmp(st(2).name, 'dpca_getTestTrials')
    DPCA_I = DPCA_I + 1;
    if DPCA_I > numel(DPCA_Q) || DPCA_K(DPCA_I) ~= 1
      error('dpcatest:queue', 'rand call #%d does not match the Python draw sequence', DPCA_I);
    end
    v = DPCA_Q{DPCA_I};
    sz = double(varargin{1});
    if numel(varargin) ~= 1 || ~isequal(sz(:)', [numel(v) 1])
      error('dpcatest:queue', 'rand call #%d: size mismatch with the Python draw', DPCA_I);
    end
    r = reshape(v, sz);
  else
    r = builtin('rand', varargin{:});
  end
end
