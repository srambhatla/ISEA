function [sigma, T, trainTime] = GAparameter(celldata)
% Estimate the hyperparameters for GAK.
% Formulas from 'Fast Global Alignment Kernels', Cuturi, 2011
% sigma = 2 * med(x-y) * \sqrt{med(l_x)}
% T = 0.5 * med(l_x)
% Input:    celldata: Cell, size N. Each cell is matrix, size T_i*D.
% Output:   sigma: GA hyperparameter
%           T: GA hyperparameter
%           trainTime:

timer1 = cputime;
totL = 0;
totDistance = 0;

% Sampling from nsample(100) series. Mean instead of median.
ndata = length(celldata);
nsample = min([ndata, 100]);
d = size(celldata{1},2);
R = randperm(ndata, nsample);
ind = zeros(nsample);
% Compute the median of series lengths
for i=1: nsample
   ind(i) = randi(size(celldata{R(i)},1), 1);
   totL = totL + size(celldata{R(i)}, 1);
end
medL = totL / nsample;

% Compute the median of frame distances
for i=1:nsample
   for j=i+1:nsample
      % L-2 distance between celldata{R(i)}(ind(i),:) celldata{R(j)}(ind(j),:)
      totDistance = totDistance + norm(celldata{R(i)}(ind(i),:) -  celldata{R(j)}(ind(j),:), 2);
   end
end
% @# Replace the for-loops by matrix operations
% framesample = zeros(nsample, d);
% for i=1:nsample
%     framesample(i,:) = celldata{R(i)}(ind(i),:);
% end
% sample2 = sum(framesample, 2); % nsample*1
% distancematrix = repmat(sample2, 1, nsample) + repmat(sample2.', nsample, 1) - 2*framesample*framesample.';
% distancematrix = sqrt(distancematrix);
% medDistance = mean(mean(distancematrix)) * nsample / (nsample-1);

medDistance = 2 * totDistance / (nsample * (nsample-1));

sigma = 2 * medDistance * sqrt(medL);
T = 0.5 * medL;
timer2 = cputime;
trainTime = timer2 - timer1;
fprintf('\tParameters of GAK: sigma = %f, T = %f\n', sigma, T);
end

