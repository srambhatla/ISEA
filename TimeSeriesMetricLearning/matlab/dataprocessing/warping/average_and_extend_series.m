function [xave] = average_and_extend_series( x1, x2)
%SHRINK_DATA Summary of this function goes here
%   Detailed explanation goes here
% average and warp x1, x2 to length(alignment(x1,x2))
% path: (ti_1, ti_2)
window_size = 10;

[~,D] = size(x1);
[~, path] =  mdtw_c_path(x1, x2, window_size);
T = size(path,1);
x = zeros(T,D);
for k=1:size(path,1)
    p1 = path(k,1); p2 = path(k,2);
    x(k,:) = x1(p1,:) + x2(p2,:);
end
% normalize
xave = x/2;
end

