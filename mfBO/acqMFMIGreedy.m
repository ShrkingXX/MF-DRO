function [acq] = acqMFMIGreedy(x, i, funcs, noiseFuncHs, costs)
    [~, ~, nextStd] = funcs{i}(x); 
    [nextNoiseStd] = noiseFuncHs{i}(x, x);

    nextStd = real(nextStd);
    acq = 0.5 * log(1 + nextStd^2 / nextNoiseStd^2) / costs(i);
end

% function [acq] = acqMFMIGreedy(x, i, funcs, noiseFuncHs, costs)
%     numFidels = numel(funcs);
%     [~, ~, nextStd] = funcs{i}(x);
%     nextStd = real(nextStd);
%     nextNoiseVar = noiseFuncHs{numFidels}(x, x);
%     acq = 0.5 * log(1 + nextStd^2 / nextNoiseVar) / costs(i);
% end

% function [acq] = acqMFMIGreedy(x, i, funcs, noiseFuncHs, costs)
%     numFidels = numel(funcs);
%     [~, ~, nextStd] = funcs{i}(x);
%     [~, ~, nextStdHigh] = funcs{numFidels}(x);
% 
%     nextStd = real(nextStd);
%     
%     % if the lower fidelity variance is larger, view it as the sum of the target fidelity variance and the noise variance
%     if i == numFidels
%         nextNoiseVar = noiseFuncHs{numFidels}(x, x);
%         acq = 0.5 * log(1 + nextStdHigh^2 / nextNoiseVar) / costs(i);    
%     elseif nextStdHigh < nextStd
%         nextNoiseVar = nextStd^2 - nextStdHigh^2;
%         acq = 0.5 * log(1 + nextStdHigh^2 / nextNoiseVar) / costs(i);
%     else
%         % if the lower fidelity variance is smaller, then it provides zero
%         % information about the target fidelity
%         acq = 0;
%     end
% end

% function [acq] = acqMFMIGreedy(x, func, noiseFunc, cost)
%   [~, ~, nextStd] = func(x);
%   [nextNoiseStd] = noiseFunc(x, x);
%   nextStd = real(nextStd);
%   nextNoiseStd = real(nextNoiseStd);
%   acq = 0.5 * log(nextStd^2 / nextNoiseStd^2) / cost;
% end
