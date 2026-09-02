function [acq, uncerts] = acqMFGPUCB(x, funcHs, t, zetas, bounds)

  numDims = size(x, 2);
  numFidels = numel(funcHs);
  if ~isa(funcHs, 'cell') 
    funcHs = {funcHs};
  end

  %beta_t = numDims * log(2*numDims*numFidels*t^2);
  %beta_t = 0.2 * numDims * log(2*numDims*t^2);

  xmax = bounds(:, 2);
  xmin = bounds(:, 1);
  beta_t = (2*log(t^2*2*pi^2/(3*0.01)) + ...
	    2*numDims*log(t^2*numDims*max(xmax-xmin)*(log(4*numDims/0.01))^0.5))^0.5;
  
  uncerts = zeros(numFidels, 1);
  augZetas = [zetas; 0];

  indUCBs = zeros(numFidels, 1);
  for i = 1:numFidels
    [mu, ~, sigma] = funcHs{i}(x);
    if ~isreal(sigma)
      sigma
    end
    uncerts(i) = sqrt(beta_t) * real(sigma);
    indUCBs(i) = mu + uncerts(i) + augZetas(i);
  end
  acq = min(indUCBs);

  acq = -acq;
end

