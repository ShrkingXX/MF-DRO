function [nextPt, nextFidel, nextPtAcq, episodeBestBCR] = ...
	 strategyMFMIGreedy(t, funcHs, noiseFuncHs, bounds, params)
    
  numFidels = numel(funcHs);
  numDims = size(bounds, 1);

  start = 1;

  % First fix fidelity level and maximize
  % mutual information cost ratio
  for i = start : numFidels
      acquisition = @(arg) -acqMFMIGreedy(arg, i, funcHs, noiseFuncHs, params.costs);
      % nextPtAcq represents the max benefit cost ratio
      % nextPt represents the next point to query
      [nextPtAcq, nextPt] = maximizeScore(acquisition, bounds);
      
%       i
%       nextPtAcq
%       nextPt
      
      if i == start
          best_fidel = i;
          best_acq = nextPtAcq;
          best_nextPt = nextPt;
      elseif nextPtAcq > best_acq
          best_fidel = i;
          best_acq = nextPtAcq;
          best_nextPt = nextPt;
    end
  end
  nextPt = best_nextPt;
  nextFidel = best_fidel;
  nextPtAcq = best_acq;
  
  % keep track of the best benefit cost ratio (BCR) of the current episode
  if params.currEpisodeBestBCR == -1
      episodeBestBCR = best_acq;
  else
      episodeBestBCR = params.currEpisodeBestBCR;
  end
  
  
  %  nextFidel
  %  params.meanAcq
  %  nextPtAcq
  %  params.lambda / sqrt(params.remainBudget)
  %  pause
  
  if params.isFirstEpisode
      if nextFidel == numFidels
          nextFidel = 1;
          nextPtAcq = best_acq;
      end
      if (params.meanAcq * params.costLowFidel + nextPtAcq * params.costs(nextFidel)) / (params.costLowFidel + params.costs(nextFidel)) < ...
          params.lambda * episodeBestBCR  * sqrt(params.totalBudget / params.remainBudget)
      % if params.numLowFidel > 500 
      % signal terminating current episode
          nextFidel = numFidels;
          newAcq = @(arg) acqMFGPUCB(arg, funcHs{numFidels}, t, zeros(0, 1), bounds);
          [nextPtAcq, nextPt] = maximizeScore(newAcq, bounds);
          
          % reset the best benefit cost ratio
          episodeBestBCR = -1;
      end
      
  elseif (nextFidel < numFidels & ...
          (params.meanAcq * params.costLowFidel + nextPtAcq * params.costs(nextFidel)) / (params.costLowFidel + params.costs(nextFidel)) < ...
          params.lambda * episodeBestBCR  * sqrt(params.totalBudget / params.remainBudget) | ...
          (nextFidel == numFidels) | params.numLowFidel > 20)
      % signal terminating current episode
      nextFidel = numFidels;
      newAcq = @(arg) acqMFGPUCB(arg, funcHs{numFidels}, t, zeros(0, 1), bounds);
      [nextPtAcq, nextPt] = maximizeScore(newAcq, bounds);
      
      % reset the best benefit cost ratio
      episodeBestBCR = -1;
      
  end
end
  
