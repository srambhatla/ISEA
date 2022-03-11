function [ W ] = metric_matrix(X1, X2, T1, T2, Y, Z, n, p, lambda)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

W = zeros(p,p);
for i=1:n
    % W = \sum \phi(Z); compute phi
    diffYZ= Z{i}-Y{i};
    phi=zeros(p,p);
    for j1=1:T1(i)
       for j2=1:T2(i)
           xdiff = X1{i}(j1,:)-X2{i}(j2,:);
           phi = phi + (xdiff'*xdiff)*diffYZ(j1,j2);
       end
    end
    % phi could be projected
    W = W + phi;
end

W = 1/n * 1/lambda * W;

end

