function [ loss_hinge, loss_origin ] = metric_loss(X1, X2, T1, T2, n, W, Y, lambda)
%METRIC_LOSS Summary of this function goes here
%   Detailed explanation goes here
window_size = 10;
Z = cell(numel(Y),1);
C = affinity_matrix(X1, X2, T1, T2, n, W);
loss_hinge = 0;
loss_origin = 0;
for i=1:n
    deltaF = -C{i} - ones(T1(i),T2(i)) + 2*Y{i};
    [~,Z{i}]=mdtw_c_align_bymatrix(deltaF, window_size);
    % notice that phi is negative!
    diffY = Y{i} - Z{i};
    l_ham = sum(sum(abs(diffY)));
    l_phi = 0;
    for j1=1:T1(i)
       for j2=1:T2(i)
           diffX = X1{i}(j1,:)-X2{i}(j2,:);
           l_phi = l_phi + diffY(j1,j2)*diffX*W*diffX';
       end
    end
    loss_hinge = loss_hinge + l_phi + l_ham;
    % original loss
    [~,Z{i}]=mdtw_c_align_bymatrix(-C{i}, window_size);
    diffY = Z{i}-Y{i};
    loss_origin = loss_origin + sum(sum(abs(diffY)));
end
f_norm = norm(W, 'fro');
loss_hinge = 1/n*loss_hinge + lambda/2*f_norm*f_norm;
loss_origin =  1/n*loss_origin + lambda/2*f_norm*f_norm;
end

