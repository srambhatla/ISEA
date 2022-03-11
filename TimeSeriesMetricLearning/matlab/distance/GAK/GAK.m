function [distGAK, trainTime, computeTime, logK_GAtrain, logK_GAtest] = GAK(xcelltrain, xcelltest, sigma, T)
% Compute the Squared multivariate GAK between elements of two sets
% mex ..\external\GA\logGAK.c;
% logGAK returns the logarithm of K_ga. 
% Local (TGA) kernel is log of Ae^d/(2-Ae^d)
% Input:    xcelltrain: Cell, size Ntrain. Each cell is matrix, size T_i*D.
%           xcelltest: Cell, size Ntest. Each cell is matrix, size T_i*D.
%           sigma: GA hyperparameter
%           T: GA hyperparameter
% Output:   distGAK: Matrix, size Ntest*Ntrain. Distances (-log K_ga)
%           trainTime:
%           computeTime:
%           logK_GAtrain: Vector, size Ntrain. Self log K_ga of xcelltrain.
%           logK_GAtest: Vector, size Ntest. Self log K_ga of xcelltest.

if nargin <= 2
   sigma = 5;
   T = 0;
end

timer1 = cputime;
ntrain = length(xcelltrain);
logK_GAtrain = zeros(1, ntrain);
for i=1:ntrain
    if mod(i,100) ==0
		fprintf('\tGAtrain-Kernel i: %d\n', i);
    end
   logK_GAtrain(i) = logGAK(xcelltrain{i}, xcelltrain{i}, sigma, T);
end
fprintf('\tGAtrain-Kernel i: %d\n', ntrain);

timer2 = cputime;
ntest = length(xcelltest);
logK_GAtest = zeros(ntest, 1);
for i=1:ntest
    if mod(i,100)==0
		fprintf('\tGAtest-Kernel i: %d\n', i);
    end
   logK_GAtest(i) = logGAK(xcelltest{i}, xcelltest{i}, sigma, T);
end
fprintf('\tGAtest-Kernel i: %d\n', ntest);

logK = zeros(ntest, ntrain);
for i = 1 : ntest
  if mod(i,100) ==0
    fprintf('\tGA-dist i: %d\n', i);
  end
  for j = 1 : ntrain
    logK(i,j) = logGAK(xcelltest{i}, xcelltrain{j}, sigma, T);
    % Not useful! the nn is always the longest one
    %distGAK(i,j) =  2 * exp(logGAK(xcelltest{i}, xcelltrain{j}, sigma, T )) - GAKtest(i) - GAKtrain(j);
    % Such normalization doesn't work well either
    %distGAK(i,j) = distGAK(i,j) / sqrt(GAKtest(i)*GAKtrain(j));
    % Normalize the kernel rather than distance
    %logK_ga(i,j) = logK_ga(i,j) - 0.5 * (logK_GAtest(i) + logK_GAtrain(j));
  end
end
fprintf('\tGA-dist i: %d\n', ntest);
logK = logK - 0.5 * (repmat(logK_GAtrain, ntest, 1) + repmat(logK_GAtest, 1, ntrain));
distGAK = -logK;

timer3 = cputime;
trainTime = timer2 - timer1;
computeTime = timer3 - timer2;

end

