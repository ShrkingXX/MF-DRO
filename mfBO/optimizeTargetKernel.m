function [bwOpt, scaleOpt] = optimizeTargetKernel(X, Y, meanFuncs, noiseBw, noiseScale, ...
						  bwRange, scaleRange, noiseVar)
  bwOpt = [];
  scaleOpt = [];
  allNlmlF = 0;
  allNlmlF = @(t) - allNormMargLikelihoodFixNoise(exp(t(1)), exp(t(2)), X, Y, ...
						  meanFuncs, noiseBw, noiseScale, noiseVar);
  [~, optParams] = maximizeScore(allNlmlF, [log(bwRange(end, :)); log(scaleRange(end, :))]);
%  [~, optParams] = diRectWrap(allNlmlF, [log(bwRange(end, :)); ...
%					 log(scaleRange(end, :))], diRectOpts);
					 
  bwOpt = exp(optParams(1));
  scaleOpt = exp(optParams(2));
end


function allNlml = allNormMargLikelihoodFixNoise(targetBw, targetScale, X, Y, ...
						 meanFunc, bw, scale, noiseVar)
  % Computes the sum of log normalised Marginal Likelihood for GPs
  numFidel = numel(X);
  allNlml = 0;
  for i = 1 : numFidel - 1
    numData = size(X{i}, 1);
    if numData > 2
      nlml = normMargLikelihoodFixNoise(targetBw, targetScale, X{i}, Y{i}, meanFunc{i}, ...
					bw(i), scale(i));
      allNlml = allNlml + nlml;
    end
  end

  numData = size(X{numFidel}, 1);
  if numData > 2
    nlml = normMargLikelihood(targetBw, targetScale, X{numFidel}, Y{numFidel}, ...
			      meanFunc{numFidel}, noiseVar);
    allNlml = allNlml + 10 * nlml; %% new changes
  end  
end


function nlml = normMargLikelihoodFixNoise(targetBw, targetScale, X, Y, meanFunc, bw, scale)
  numData = size(X, 1);
  Ky = sqExpKernel(targetBw, targetScale, X, X) + ...
       sqExpKernel(bw, scale, X, X);
  Y_ = Y - meanFunc(X);
  L = stableCholesky(Ky);
  alpha = L' \ (L \ Y_);
  nlml = -0.5 * Y_' * alpha - sum( log(diag(L + 0.001)) ) -  log(2*pi)*numData/2;
end
