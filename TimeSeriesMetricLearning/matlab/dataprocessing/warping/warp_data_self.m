function [X] = warp_data_self( Xori, Xref)
%SHRINK_DATA Summary of this function goes here
%   Detailed explanation goes here
window_size = 10;
n = length(Xori);           % # of data
X = cell(n,1);
for i=1:n 
    [T,D] = size(Xref{i});
    x = zeros(T,D);
    [~, path] =  mdtw_c_path(Xori{i}, Xref{i}, window_size);
    cnt = zeros(T,1);
    for k=1:size(path,1)
        p1 = path(k,1); p2 = path(k,2);
        x(p2,:) = Xori{i}(p1,:) + x(p2,:);
        cnt(p2) = cnt(p2) + 1;
    end
    % normalize
    X{i}=x./repmat(cnt, [1,D]);
end

end

