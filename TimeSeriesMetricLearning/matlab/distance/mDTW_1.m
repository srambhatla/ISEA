function [distmDTW, trainTime, computeTime] = mDTW_1(xcelltrain, xcelltest)
% Compute the Squared multivariate DTW between elements of two sets
% Required: mex ..\external\mDTW\mdtw_c.c;
% In mdtw_c, the distance is squared.
% Input:    xcelltrain: Cell, size Ntrain. Each cell is matrix, size T_i*D.
%           xcelltest: Cell, size Ntest. Each cell is matrix, size T_i*D.
% Output:   distmDTW: Matrix, size Ntest*Ntrain
%           trainTime
%           computeTime

timer1 = cputime;
ntrain = length(xcelltrain);
% lentrain = zeros(ntrain);
% for i = 1: ntrain
%    lentrain(i) = size(xcelltrain{i},1);
% end

timer2 = cputime;
w = 10;
ntest = length(xcelltest);
distmDTW = zeros(ntest, ntrain);
p = size(xcelltrain{1},2);
for i = 1 : ntest
  if mod(i,100) ==0
    fprintf('\tMDTW-dist i: %d\n', i);
  end
%   lentest = size(xcelltest{i},1);
  for j = 1 : ntrain
      tmpDist = zeros(1,p);
      for k = 1:p
          [tmpDist(k),~] = mdtw_c_path(xcelltest{i}(:,k), xcelltrain{j}(:,k), w);
      end
      distmDTW(i,j) = sum(tmpDist);
  end
end
fprintf('\tMDTW-dist i: %d\tfinished\n', ntest);

timer3 = cputime;
trainTime = timer2 - timer1;
computeTime = timer3 - timer2;

end

