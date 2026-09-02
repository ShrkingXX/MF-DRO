function [teMean, teK, teStd, funcH] = ...
  GPRegression(Xte, Xtr, Ytr, bw, scale, meanFunc, noiseFunc)
% Outputs the posterior mean (nTe x 1), standard deviations (nTe x 1) and covariance
% matrix (nTe x nTe) of the test data Xte. In addition, returns a function Handle
% for the GPs.

  numTrData = size(Xtr, 1);

  if numTrData == 0
    Ytr = zeros(0, 1);
  end

  if nargin(noiseFunc) == 0
    noiseVar = diag(noiseFunc() * ones(numTrData, 1));
  elseif nargin(noiseFunc) == 2
    noiseVar = noiseFunc(Xtr, Xtr);
  end
  Ktrtr = sqExpKernel(bw, scale, Xtr, Xtr) + noiseVar;
  Y_ = Ytr - meanFunc(Xtr);
  L = stableCholesky(Ktrtr);
  alpha = L' \ (L \ Y_);

  % obtain the function handle
  funcH = @(X) GPComputeOutputs(X, Xtr, L, alpha, bw, scale, meanFunc, noiseFunc);
  % Compute outputs for the test data
  if ~isempty(Xte)
    [teMean, teK, teStd] = funcH(Xte);
  else
    teMean = []; teK = []; teStd = [];
  end

end


function [yMu, yK, yStd] = GPComputeOutputs(Xte, Xtr, L, alpha, bw, scale, meanFunc, noiseFunc)

  meanXte = meanFunc(Xte);
  Ktetr = sqExpKernel(bw, scale, Xte, Xtr);
  Ktete = sqExpKernel(bw, scale, Xte, Xte);

  if nargin(noiseFunc) == 2
    Ktetr = Ktetr + noiseFunc(Xte, Xtr);
    Ktete = Ktete + noiseFunc(Xte, Xte);
  end

  % Predictive Mean
  yMu = meanXte + Ktetr * alpha;
  % Predictive Variance
  V = L \ (Ktetr)';
  yK = Ktete - V'*V;
  yStd = sqrt(real(diag(yK)));

end

