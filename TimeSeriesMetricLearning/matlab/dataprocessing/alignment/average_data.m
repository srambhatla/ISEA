function [ X ] = average_data(  Xori )
%AVERAGE_DATA Summary of this function goes here
%   Detailed explanation goes here
n = length(Xori);           % # of data
[T, D] = size(Xori{1});     % # of frame dimensions
X = zeros(T,D);

for j=1:T
    for i=1:n
        X(j,:) = Xori{i}(j,:) + X(j,:);
    end
end
X = X./n;

end

