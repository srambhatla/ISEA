function [ X ] = dataNormalizeSelf(Xorigin)
% Self z-normalize each MVT series
% Input:    Xorigin: Cell, size N. Each cell is matrix, size T_i*D. 
% Output:   X: Cell, size N. Each cell is matrix, size T_i*D. Z-Normalized Xtrain.

X = cell(size(Xorigin));
stdeps = 1e-6;

for i=1:length(Xorigin)
    selfMean = mean(Xorigin{i});
    % std could be zero for some dimensions, then we'll set std to 1
    selfStd = std(Xorigin{i});
	selfStd(selfStd<=stdeps) = 1;
    T = size(Xorigin{i},1);
    X{i} = (Xorigin{i}-repmat(selfMean, T, 1)) ./ repmat(selfStd, T, 1);
end

end

