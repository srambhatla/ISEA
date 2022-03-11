function [ X ] = dataNormalizeRange(Xorigin, T, Ranges)
% Normalize the data with the given empirical ranges
% Input:    Xorigin: Cell, size N. Each cell is matrix, size T_i*D. 
%           T: Vector, size N. Length of Xtrain{i}.
%           Ranges: Matrix, size D*3. Each row is [1st , 50th, 99th] percentile of dimension d
% Output:   X: Cell, size N. Each cell is matrix, size T_i*D. Normalized from Xtrain.

X = cell(size(Xorigin));
for i=1:length(Xorigin)
	% x_new = (x-mean)/(max-min)
    X{i} = (Xorigin{i}-repmat(Ranges(:,2)', T(i), 1)) ./ repmat((Ranges(:,3)-Ranges(:,1))', T(i), 1);
end

end

