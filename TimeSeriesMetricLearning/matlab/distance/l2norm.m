function [ distLp, trainTime, computeTime ] = l2norm(xtrain, xtest)
% Compute the Squared Euclidean distance between elements of two sets
% Input:    xtrain: Matrix, size Ntrain*(T*D)
%           xtest: Matrix, size Ntest*(T*D)
% Output:   distLp: Matrix, size N_test*N_train
%           trainTime
%           computeTime

timer1 = cputime;
ntrain = size(xtrain, 1);
timer2 = cputime;
ntest = size(xtest, 1);
distLp = zeros(ntest, ntrain);
for i = 1 : ntest
  if mod(i,1000) ==0
    fprintf('\tL2-dist i: %d\n', i);
  end
  for j = 1 : ntrain
	% Speed up by no sqrt
    distLp(i,j) = sum((xtest(i,:) - xtrain(j,:)).^2);
  end
end
fprintf('\tL2-dist i: %d\tfinished\n', ntest);
timer3 = cputime;

trainTime = timer2 - timer1;
computeTime = timer3 - timer2;

end
