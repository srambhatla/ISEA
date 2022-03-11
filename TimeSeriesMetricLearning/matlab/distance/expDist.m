function [ dist, trainTime, computeTime ] = expDist(xcelltrain, xcelltest)
% Compute the Expected Distance between elements of two sets
% Input:    xtrain: Matrix, size Ntrain*(T*D)
%           xtest: Matrix, size Ntest*(T*D)
% Output:   distLp: Matrix, size N_test*N_train
%           trainTime
%           computeTime

timer1 = cputime;
ntrain = length(xcelltrain);
timer2 = cputime;
ntest = length(xcelltest);
dist = zeros(ntest, ntrain);
for i = 1 : ntest
  if mod(i,100) ==0
    fprintf('\tExpD-dist i: %d\n', i);
  end
  for j = 1 : ntrain
	% Speed up by no sqrt
    pmat = prob_matrix_expd(length(xcelltest{i}, length(xcelltrain{j})));
    dist(i,j) = sum(sum(((xcelltest{i} - xcelltrain{j}).^2) .* pmat));
  end
end
fprintf('\tED-dist i: %d\tfinished\n', ntest);
timer3 = cputime;

trainTime = timer2 - timer1;
computeTime = timer3 - timer2;

end

