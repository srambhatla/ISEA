function [xave, path] = average_and_warp_series( x1, x2, x1c, x2c)
%SHRINK_DATA Summary of this function goes here
%   Detailed explanation goes here
% average and warp x1, x2 to max(length(x1, x2))
% path: (ti_1, ti_2)
window_size = 10;

[T1,D] = size(x1);
[T2,~] = size(x2);
long_id=1;
if (T1>T2)
    T=T1;
    xlong=x1; xshort = x2;
    xlongc=x1c; xshortc = x2c;
else
    long_id=2;
    T=T2;
    xlong=x2; xshort = x1;
    xlongc=x2c; xshortc = x1c;
end
x = zeros(T,D);
[~, path] =  mdtw_c_path(xshort, xlong, window_size);
cnt = zeros(T,1);
for k=1:size(path,1)
    p1 = path(k,1); p2 = path(k,2);
    x(p2,:) = xshort(p1,:) + x(p2,:);
    cnt(p2) = cnt(p2) + 1;
end
% normalize
x=x./repmat(cnt, [1,D]);
xave = (x*xshortc + xlong*xlongc)/(xshortc+xlongc);
if (long_id==1)
    path = path(:,[2,1]);
end
%xave = (x + xlong)/2;
end

