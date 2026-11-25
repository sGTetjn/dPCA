function y = nansum(x, dim)
% Minimal stand-in for the Statistics Toolbox nansum: sum over DIM ignoring NaNs.
  if nargin < 2
    dim = find(size(x) ~= 1, 1);
    if isempty(dim), dim = 1; end
  end
  x(isnan(x)) = 0;
  y = sum(x, dim);
end
