function [X] = dataVectorize(Xori, T)
% Transform variant-length series to fixed-length vector series
%   Replicate the last frame of the series or truncate the tail
%   Vectorize the same-length data
% Input:    Xori: Cells, size N. Each cell is matrix, size T_i*D.
%           T: Temporal length of output
% Output:   X: Matrix, size N*(T*D).

n = length(Xori);           % # of data
[~, D] = size(Xori{1});     % # of frame dimensions
td = T*D;                   % # of output vector dimensions
X = zeros(n, td);
for i=1:n
    Ti = size(Xori{i}, 1);  % temporal length of i-th series
	if Ti < T
		% extend by replicating the last frame
		Xt = [Xori{i}; repmat(Xori{i}(Ti,:), T-Ti, 1) ];
    elseif Ti > T
        % truncate longer data
        Xt = Xori{i}(1:T, :);
    else
        Xt = Xori{i};
	end
	X(i,:) = reshape(Xt, 1, td);
end
%fprintf('    N: %d, D: %d\n', size(X, 1), size(X, 2));

end

