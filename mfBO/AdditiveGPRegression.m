function [newFuncHs] = ...
  AdditiveGPRegression(Xtr, Ytr, bws, scales, meanFuncs, noiseFuncs)
% meanFuncs are means of f_m + mu(e_i)
% noise funcs are zero mean GPs

  numTrData = size(Xtr, 1);

  if numTrData == 0
      Ytr = zeros(0, 1);
  end
  
  numFidels = length(noiseFuncs);
  
  % compute the joint kernel matrix
  Ktrtr = sqExpKernelAdditive(bws, scales, noiseFuncs, Xtr, Xtr); % Yuxin: note here we are ignoring the high fidelity

  Ytr_joint = [];
  meanFuncs_joint = [];
  for i = 1:numFidels
        Ytr_joint = [Ytr_joint; Ytr{i}];
        meanFuncs_joint = [meanFuncs_joint; meanFuncs{i}(Xtr{i})];
  end
  Y_ = Ytr_joint - meanFuncs_joint;
  L = stableCholesky(Ktrtr);
  alpha = L' \ (L \ Y_);

  % obtain the function handle
  newFuncHs = cell(numFidels, 1);
  for i = 1:numFidels
      newFuncHs{i} = @(X) MFGPComputeOutputs(X, Xtr, i, L, alpha, bws, scales, noiseFuncs, meanFuncs);
  end
  
%   newNoiseFuncs = cell(numFidels, 1);
%   newNoiseFuncs{numFidels} = noiseFuncs{numFidels};
%   for i = 1:numFidels - 1
%       % newNoiseFuncs{i} = @(Xtr, Xte) (newFuncHs{i}(Xtr) - newFuncHs{numFidels}(Xte));
%       newNoiseFuncs{i} = @(X1, X2) sqrt((MFGPComputeNoise(X1, Xtr, i, L, bws, scales) - MFGPComputeNoise(X2, Xtr, numFidels, L, bws, scales)));
%   end  
end

% 
% function [yVar] = MFGPComputeNoise(X, Xtr, i, L, bws, scales)
% 
%   Xte = cell(length(bws),1);
%   Xte{i} = X;
% 
%   % compute the joint kernel matrix
%   Ktetr = sqExpKernelAdditive(bws, scales, Xte, Xtr);
%   Ktete = sqExpKernelAdditive(bws, scales, Xte, Xte);
% 
%   % Predictive Variance
%   V = L \ (Ktetr)';
%   yK = Ktete - V'*V;
%   yVar = real(diag(yK));
% 
% end


function [yMu, yK, yStd] = MFGPComputeOutputs(X, Xtr, i, L, alpha, bws, scales, noiseFuncs, meanFuncs)

  Xte = cell(length(bws),1);
  Xte{i} = X;

  meanXte = meanFuncs{i}(Xte{i});
  % compute the joint kernel matrix
  Ktetr = sqExpKernelAdditive(bws, scales, noiseFuncs, Xte, Xtr);
  Ktete = sqExpKernelAdditive(bws, scales, noiseFuncs, Xte, Xte);

  % Predictive Mean
  yMu = meanXte + Ktetr * alpha;
  % Predictive Variance
  V = L \ (Ktetr)';
  yK = Ktete - V'*V;
  yStd = sqrt(real(diag(yK)));

end

