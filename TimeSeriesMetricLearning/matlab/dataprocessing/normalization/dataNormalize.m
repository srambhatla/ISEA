function [ X, X2 ] = dataNormalize(Xtrain, T, Xtest)
% Z-normalize the training/testing data with mean/std of training data
% Input:    Xtrain: Cell, size Ntrain. Each cell is matrix, size T_i*D. 
%           T: Vector, size Ntrain. Length of Xtrain{i}.
%           Xtest: Cell, size Ntest. Each cell is matrix, size T_i*D.
% Output:   X: Cell, size Ntrain. Each cell is matrix, size T_i*D. Normalized from Xtrain.
%           X2: Cell, size Ntest. Each cell is matrix, size T_i*D. Normalized from Xtest.

Xvert = vertcat(Xtrain{:});
trainMean = mean(Xvert);

trainStd = std(Xvert);
% std could be zero for some dimensions, then we'll set std to 1
stdeps = 1e-6;
trainStd(trainStd<=stdeps) = 1;

% Z-normalize the training data for each dimension
X = cell(size(Xtrain));
X2 = [];

for i=1:length(Xtrain)
    X{i} = (Xtrain{i}-repmat(trainMean, T(i), 1)) ./ repmat(trainStd, T(i), 1);
end
% trainMean = mean(vertcat(X{:}));
% trainStd = std(vertcat(X{:}));
% fprintf('    For training data, mean = %f, std = %f\n', [trainMean; trainStd] );

if nargin < 3
    return;
end

% Normalized the testing data with mean/std of training data
X2 = cell(size(Xtest));
for i=1:length(Xtest)
    X2{i} = (Xtest{i}-repmat(trainMean, size(Xtest{i},1), 1)) ./ repmat(trainStd, size(Xtest{i},1), 1);
end
% testMean = mean(vertcat(X2{:}));
% testStd = std(vertcat(X2{:}));
% fprintf('    For test data, mean = %f, std = %f\n', [testMean; testStd] );

end