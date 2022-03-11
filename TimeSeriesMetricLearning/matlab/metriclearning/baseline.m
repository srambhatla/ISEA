function [acc_val, acc_tst] = baseline(X_tr, y_tr, X_va, y_va, X_tst, y_tst, P)
acc_val = compute_accuracy(X_tr, y_tr, X_va, y_va, eye(P), 0);
acc_tst = compute_accuracy([X_tr;X_va], [y_tr;y_va], X_tst, y_tst, eye(P), 0);


% X_tr = [X_tr;X_va];
% y_tr = [y_tr;y_va];

% W = eye(P);
% correct = 0;
% total = length(y_tst);
% 
% for i=1: length(y_tst)
%     % for each validation data, compute the best label.
%     minDist = 999999;
%     minLabel = 1;
%     query = X_tst{i};
%     
%     for j=1:length(y_tr)
%         data = X_tr{j};
%         
%         
%         [dist,~] = mdtw_c_path(query*W, data*W,10);
%         if dist <= minDist
%             minDist = dist;
%             minLabel = y_tr(j);
%         end
%     end
%     if minLabel == y_tst(i)
%         correct = correct + 1;
%     end
%     
% end
% 
% acc_tst = correct/total;