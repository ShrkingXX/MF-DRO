function K = sqExpKernelAdditive(bws, scales, noiseFuncs, X, Y)
% returns a block diagonal matrix 
    if ~exist('Y', 'var')
        Y = X;
    end
    
    numFidels = length(bws);
    diagonal_k = cell(numFidels);

    kernel_offset = [];
    % offset of diagonal entries
    for i = 1:numFidels % test
        if ~isempty(X{i}) && ~isempty(Y{i})
            D = distSquaredGP(X{i}, Y{i});
            if i == numFidels            
                diagonal_k{i} = noiseFuncs{i}(X{i}, Y{i});
            else
                diagonal_k{i} = scales(i) * exp( -D / (2*bws(i)^2) );
            end
            kernel_offset = blkdiag(kernel_offset, diagonal_k{i});
        end
    end
    X_joint=[];
    Y_joint=[];
    for i = 1:numFidels
        if ~isempty(X{i})
            X_joint = [X_joint; X{i}]; % Yuxin: check if direction is correct
        end
        if ~isempty(Y{i})
            Y_joint = [Y_joint; Y{i}];
        end
    end
    D_joint = distSquaredGP(X_joint, Y_joint);
    kernel_base = scales(numFidels) * exp( -D_joint / (2*bws(numFidels)^2));
    new_kernel_offset = zeros(size(kernel_base));
    new_kernel_offset(1:size(kernel_offset,1),1:size(kernel_offset,2)) = kernel_offset;
    K = kernel_base + new_kernel_offset;
end
