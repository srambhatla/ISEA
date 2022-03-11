function [x] = warp_series(x1, x2)
%SHRINK_DATA Summary of this function goes here
%   Detailed explanation goes here
% warp x1 to length(x2)

[T, D] = size(x2);
x = zeros(T,D);
[~, path] =  mdtw_c_path(x1, x2);
cnt = zeros(T,1);
for k=1:size(path,1)
    p1 = path(k,1); p2 = path(k,2);
    x(p2,:) = x1(p1,:) + x(p2,:);
    cnt(p2) = cnt(p2) + 1;
end
% normalize
x=x./repmat(cnt, [1,D]);
end

