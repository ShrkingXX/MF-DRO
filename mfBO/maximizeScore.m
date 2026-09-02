function [score, pt] = maximizeScore(fnOptimize, bounds)

  gridSize = 200;
  d = size(bounds, 1);
  %Xgrid = repmat(bounds(:, 1)', gridSize, 1) + ...
%	  repmat((bounds(:, 2) - bounds(:, 1))', gridSize, 1) .* ...
%	  repmat(linspace(0.0, 1.0, gridSize)', 1, d);

  while 1
    Xgrid = repmat(bounds(:, 1)', gridSize, 1) + ...
	    repmat((bounds(:, 2) - bounds(:, 1))', gridSize, 1) .* ...
	    rand(gridSize, d);;  

    for i = 1 : gridSize
      score = fnOptimize(Xgrid(i, :));
      if i == 1
	best_score = score;
	best_init = Xgrid(i, :);
      elseif score < best_score
	best_score = score;
	best_init = Xgrid(i, :);
      end
    end
  %best_score
  %fnOptimize(best_init)
  %best_init
    options = optimoptions('fmincon', 'Display','off');
				%fnOptimize(best_init)
    try
      [pt, score] = fmincon(fnOptimize, best_init, [], [], [], [], ...
			    bounds(:, 1), bounds(:, 2), [], options);
      break;
    catch e
      fprintf(1,'The identifier was:\n%s',e.identifier);
      fprintf(1,'There was an error! The message was:\n%s',e.message);
      [acq, uncerts] = fnOptimize(best_init);
      acq
      uncerts
    end
  end
    
  score = - score;
end
