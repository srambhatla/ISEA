function [] = run_experiment(X,y,p,algo, deltas)


name = 'physionet';
folder = strcat('result_',name);
load physionet;

% T = zeros(length(X),1);
% for i=1:length(T)
%     T(i) = length(X{i});
% end
% [X, ~] =  dataNormalize(X, T);

for i = 1:length(deltas)
    delta = deltas(i);
    
    for itr = 1:5
        split_tst = itr;
        split_va = split_tst + 1;
        if split_va > 5
            split_va = split_va - 5;
        end
        
        X_tst = X(find(folds == split_tst));
        y_tst = y(find(folds == split_tst));
        
        X_va1 = X(find(folds == split_va));
        y_va = y(find(folds == split_va));
        T_va1 = T(find(folds == split_va));
        
        X_tr1 = X(find(folds ~= split_va & folds ~= split_tst));
        y_tr = y(find(folds ~= split_va & folds ~= split_tst));
        T_tr1 = T(find(folds ~= split_va & folds ~= split_tst));
        
        n  = length(T_tr1);
        [X_new, X_tst] = dataNormalize([X_tr1;X_va1], [T_tr1;T_va1], X_tst);
        X_tr = X_new(1:n);
        X_va = X_new(n+1:end);
        
        clear X_val1;
        clear X_tr1;
        
        if algo==1
            % baseline
            disp('baseline accuracy is:   ');
            [base_acc_val, base_acc_tst] = baseline(X_tr, y_tr, X_va, y_va, X_tst, y_tst, p)
            save(strcat(folder,'/','base_acc', num2str(split_tst), name), 'base_acc_val','base_acc_tst');
        elseif algo==2
            % 1. linear metric
            disp('running linear metric learning......');
            [linear_acc_val, linear_acc_tst, linear_obj, linear_dist_pair, linear_best_W] = linear_metric_learning(X_tr, y_tr, X_va, y_va, X_tst, y_tst, p, delta);
            save(strcat(folder,'/','linear', num2str(split_tst), name, '_',num2str(delta)), 'linear_acc_val','linear_acc_tst','linear_obj','linear_dist_pair', 'linear_best_W');
            
        else
            % 2. expected linear metric
            disp('running linear metric learning with expected distance......');
            [linear_expected_acc_val, linear_expected_acc_tst, linear_expected_obj, linear_expected_dist_pair, linear_expected_best_W] = linear_metric_learning_expected2(X_tr, y_tr, X_va, y_va, X_tst, y_tst, p, delta);
            save(strcat(folder,'/','linear_expected', num2str(split_tst), name,'_',num2str(delta)), 'linear_expected_acc_val','linear_expected_acc_tst','linear_expected_obj','linear_expected_dist_pair','linear_expected_best_W');
        end
    end
end






  
    
    %% 1. DTW with linear metric (expected distance)
%     disp('running linear metric learning with expected distance......');
%     [linear_expected_acc_val, linear_expected_acc_tst, linear_expected_obj, linear_expected_dist_pair, linear_expected_best_W] = linear_metric_learning_expected(X_tr, y_tr, X_va, y_va, X_tst, y_tst, p);
%     save(strcat('linear_expected', num2str(split), name), 'linear_expected_acc_val','non_linear_expected_acc_tst','non_linear_expected_obj','non_linear_expected_dist_pair','non_linear_expected_best_W');

%     save(strcat('non_linear_acc_val', num2str(split), name), 'non_linear_acc_val');
%     save(strcat('non_linear_obj', num2str(split), name), 'non_linear_obj');
%     save(strcat('non_linear_dist_pair', num2str(split), name), 'non_linear_dist_pair');
%     save(strcat('non_linear_best_W', num2str(split), name), 'non_linear_best_W');


    % 4. DTW with non-linear metric (expected distance)
%     disp('running non-linear metric learning with expected distance.....');
%     [non_linear_expected_acc_val, non_linear_expected_acc_tst, non_linear_expected_obj, non_linear_expected_dist_pair, non_linear_expected_best_W] = non_linear_metric_learning_expected(X_tr, y_tr, X_va, y_va, X_tst, y_tst, p);
%     save(strcat('non_linear_expected', num2str(split_tst)), 'non_linear_expected_acc_val', 'non_linear_expected_acc_tst','non_linear_expected_obj','non_linear_expected_dist_pair','non_linear_expected_best_W');
%   
  
     %% 3. non-linear metric
%       disp('running non-linear metric learning.....');
%     [non_linear_acc_val, non_linear_acc_tst, non_linear_obj, non_linear_dist_pair, non_linear_best_W] = non_linear_metric_learning(X_tr, y_tr, X_va, y_va, X_tst, y_tst, p);
%     save(strcat('non_linear', num2str(split_tst), name), 'non_linear_acc_tst','non_linear_acc_val','non_linear_obj','non_linear_dist_pair','non_linear_best_W');
%     
%     

% for i=1: length(y_tst)
%     % for each validation data, compute the best label.
%     minDist = 9999999999;
%     minLabel = 1;
%     query = X_tst{i};
%     
%     for j=1:length(y_tr)
%         data = X_tr{j};
%         
%         
%         [dist,~] = mdtw_c_path(query, data,10);
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
% acc_tst = correct/total

