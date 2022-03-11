function [ dist, trainTime, computeTime ] = ED(xcelltrain, xcelltest)
% Compute the Squared Euclidean distance between elements of two sets
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
  if mod(i,1000) ==0
    fprintf('\tED-dist i: %d\n', i);
  end
  for j = 1 : ntrain
	% Speed up by no sqrt
    dist(i,j) = sum(sum((xcelltest{i} - xcelltrain{j}).^2));
  end
end
fprintf('\tED-dist i: %d\tfinished\n', ntest);
timer3 = cputime;

trainTime = timer2 - timer1;
computeTime = timer3 - timer2;

end
