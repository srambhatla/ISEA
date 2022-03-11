function [X, Xpath] = warp_data( Xori, x_ref)
%SHRINK_DATA Summary of this function goes here
%   Detailed explanation goes here
window_size = 10;
n = length(Xori);           % # of data
X = cell(n,1);
Xpath = cell(n,1);
[T,D] = size(x_ref);
for i=1:n 
    x = zeros(T,D);
    xpath = zeros(size(Xori{i},1), 2);
    xpath(1,1)=1;
    xpath(size(Xori{i},1), 2)=T;
    xpathloc=1;
    [~, path] =  mdtw_c_path(Xori{i}, x_ref, window_size);
    cnt = zeros(T,1);
    for k=1:size(path,1)
        p1 = path(k,1); p2 = path(k,2);
        x(p2,:) = Xori{i}(p1,:) + x(p2,:);
        cnt(p2) = cnt(p2) + 1;
        if (k>1&&p1~=xpathloc)
            xpath(xpathloc,2)=path(k-1, 2);
            xpathloc = p1;
            xpath(p1,1)=p2;
        end
    end
    
%     if (max(xpath(:)) > T)
%        xpath 
%        assert(1==0);
%     end
    % normalize
    X{i}=x./repmat(cnt, [1,D]);
    Xpath{i} = xpath;
end
end

