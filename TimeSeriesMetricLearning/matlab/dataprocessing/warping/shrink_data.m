function [X] = shrink_data( Xori, T)
%SHRINK_DATA Summary of this function goes here
%   Detailed explanation goes here

n = length(Xori);           % # of data
[~, D] = size(Xori{1});     % # of frame dimensions
X = cell(n,1);

for i=1:n
    Ti = size(Xori{i}, 1);  % temporal length of i-th series
    for j=1:T
        tt = (j-1)*(Ti-1)/(T-1)+1;
        if (tt <=1 )
            X{i}(j,:)=Xori{i}(1,:);
        elseif (tt >=Ti)
            X{i}(j,:)=Xori{i}(Ti,:);
        else
            baset = round(tt-0.5);
           X{i}(j,:) = double(baset+1-tt)*Xori{i}(baset,:) + double(tt-baset)*Xori{i}(baset+1,:);
        end
    end
end

end

