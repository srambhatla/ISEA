function [ loss_hinge, loss_origin ] = metric_loss_area(X1, X2, T1, T2, n, W, Y, Z, lambda)
%METRIC_LOSS Summary of this function goes here
%   Detailed explanation goes here
C = affinity_matrix(X1, X2, T1, T2, n, W);
window_size = 10;
loss_hinge = 0;
loss_origin = 0;
for i=1:n
    % notice that phi is negative!
    diffY = Y{i} - Z{i};
    l_area = 0.5*( sum(sum((tril(ones(T1(i)))*diffY).^2)) + sum(sum((diffY*tril(ones(T2(i)))).^2)));
    l_phi = 0;
    for j1=1:T1(i)
       for j2=1:T2(i)
           diffX = X1{i}(j1,:)-X2{i}(j2,:);
           l_phi = l_phi + diffY(j1,j2)*diffX*W*diffX';
       end
    end
    loss_hinge = loss_hinge + l_phi + l_area;
    % original loss
    loss_origin = loss_origin + l_area;
%     [~,ZZ]=mdtw_c_align_bymatrix(-C{i}, window_size);
%     diffY = ZZ-Y{i};
%     loss_origin = loss_origin +0.5*( sum(sum((tril(ones(T1(i)))*diffY).^2)) + sum(sum((diffY*tril(ones(T2(i)))).^2)));
end
f_norm = norm(W, 'fro');
loss_hinge = 1/n*loss_hinge + lambda/2*f_norm*f_norm;
loss_origin =  1/n*loss_origin + lambda/2*f_norm*f_norm;
end

