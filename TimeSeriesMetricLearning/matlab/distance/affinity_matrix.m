function [ C ] = affinity_matrix( X1, X2, T1, T2, n, W)
%AFFINITY_MATRIX Summary of this function goes here
%   Detailed explanation goes here

% compute affine matrix C
% C_jk = -(X1_j-X2_k)'W(X1_j-X2_k)
% which is inversed distance

C = cell(n,1);
for i=1:n
    C{i} = zeros(T1(i), T2(i));
    for j1=1:T1(i)
       for j2=1:T2(i)
           xdiff = X1{i}(j1,:)-X2{i}(j2,:);  %1*p
           C{i}(j1, j2) = -xdiff*W*xdiff';
          %C{i}(j1, j2)= -sum((X1{i}(j1,:)-X2{i}(j2,:)).^2);
       end
    end
end

end

