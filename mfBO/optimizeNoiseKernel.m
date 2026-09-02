function [bwOpt, scaleOpt] = optimizeNoiseKernel(X, Y, meanFuncs, targetBw, targetScale, ...
						 bwRange, scaleRange, diRectOpts)
  numFidel = numel(X);
  bwOpt = [];
  scaleOpt = [];
  
  for i = 1 : numFidel - 1
    nlmlF = @(t) - normMargLikelihoodFixTarget(targetBw, targetScale, X{i}, Y{i}, ...
					       meanFuncs{i}, exp(t(1)), exp(t(2)));
    [~, optParams] = maximizeScore(nlmlF, [log(bwRange(1, :)); log(scaleRange(1, :))]);
    %[~, optParams] = diRectWrap(nlmlF, [log(bwRange(1, :)); log(scaleRange(1, :))], ...
%				diRectOpts);

    bwOpt = [bwOpt; exp(optParams(1))];
    scaleOpt = [scaleOpt; exp(optParams(2))];
  end
end


function nlml = normMargLikelihoodFixTarget(targetBw, targetScale, X, Y, meanFunc, bw, scale)
  % Computes the log normalised Marginal Likelihood for a single GP
  numData = size(X, 1);
  Ky = sqExpKernel(targetBw, targetScale, X, X) + sqExpKernel(bw, scale, X, X);
  Y_ = Y - meanFunc(X);
  L = stableCholesky(Ky);
  alpha = L' \ (L \ Y_);
  nlml = -0.5 * Y_' * alpha - sum( log(diag(L + 0.001)) ) -  log(2*pi)*numData/2;
end
