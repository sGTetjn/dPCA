function y = nanmean(x, dim)
% Minimal stand-in for the Statistics Toolbox nanmean (not shipped with Octave core):
% mean over dimension DIM ignoring NaNs; all-NaN slices give NaN.
  if nargin < 2
    dim = find(size(x) ~= 1, 1);
    if isempty(dim), dim = 1; end
  end
  m = ~isnan(x);
  x(~m) = 0;
  y = sum(x, dim) ./ sum(m, dim);
end
