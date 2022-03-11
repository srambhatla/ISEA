function [X, keep] = warp_data_part( Xori, x_ref)
%SHRINK_DATA Summary of this function goes here
%   Detailed explanation goes here
window_size = 10;
n = length(Xori);           % # of data
X = cell(n,1);
keep = cell(n,1);
[T,D] = size(x_ref);
for i=1:n 
    x = zeros(T,D);
    kp = zeros(T,1);
    [~, path] =  mdtw_c_path(Xori{i}, x_ref, window_size);
    for k=1:size(path,1)
        p1 = path(k,1); p2 = path(k,2);
        x(p2,:) = Xori{i}(p1,:);
        if k==1
            kp(k)=1;
        elseif path(k,1) ~= path(k-1,1)
            kp(k)=1;
        else
            kp(k)=0;
        end
    end
    % copy
    X{i}=x;
    keep{i} = kp;

end
end

