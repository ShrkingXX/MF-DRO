function [teMean, teK, teStd, funcH] = ...
	 MFGPRegression(Xte, Xtr, Ytr, f_bw, noise_bw, ...
			f_scale, noise_scale, f_meanFunc, noise_meanFunc)
% Outputs the posterior mean (nTe x 1), standard deviations (nTe x 1) and covariance
% matrix (nTe x nTe) of the test data Xte. In addition, returns a function Handle
% for the GPs.

  numTrData = size(Xtr, 1);

  if numTrData == 0
    Ytr = zeros(0, 1);
  end

  noiseVar = sqExpKernel(noise_bw, noise_scale, Xtr, Xtr);
  Ktrtr = sqExpKernel(f_bw, f_scale, Xtr, Xtr) + noiseVar;
  Y_ = Ytr - f_meanFunc(Xtr) - noise_meanFunc(Xtr);
  L = stableCholesky(Ktrtr);
  alpha = L' \ (L \ Y_);

  % obtain the function handle
  funcH = @(X) MFGPComputeOutputs(X, Xtr, L, alpha, bw, scale, meanFunc);
  % Compute outputs for the test data
  if ~isempty(Xte)
    [teMean, teK, teStd] = funcH(Xte);
  else
    teMean = []; teK = []; teStd = [];
  end

end


function [yMu, yK, yStd] = MFGPComputeOutputs(Xte, Xtr, L, alpha, bw, scale, meanFunc)

  meanXte = meanFunc(Xte);
  Ktetr = sqExpKernel(bw, scale, Xte, Xtr);
  Ktete = sqExpKernel(bw, scale, Xte, Xte);

  % Predictive Mean
  yMu = meanXte + Ktetr * alpha;
  % Predictive Variance
  V = L \ (Ktetr)';
  yK = Ktete - V'*V;
  yStd = sqrt(real(diag(yK)));

end

