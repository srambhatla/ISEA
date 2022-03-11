function [acc_val, acc_tst, obj, dist_pair, best_W] = linear_metric_learning_expected2(X_tr, y_tr, X_val, y_val, X_tst, y_tst, P, delta)
%   X_tr:  training time series.  X_tr{i} is the ith time series in the 
%          training data, which is n by p matrix(n is the length of time
%          seires, and p is the dimension of time series). 
%   y_tr:  labels for training time series
%   X_val, y_val, X_tst, y_tst:  for validation and testing data. 
%   p:       dimensionality of the time series
% 
%   objective function:
%       We use large margin approach, which means that the distance
%       within the same class should be small, and large otherwise. 
%       the objective we used here is :
%       f(W) = \sum_{i,j} d(x_i, x_j) + lamda* \sum_{i,j,k} (delta+d(x_i,x_j)-d(x_i, x_k))_{+}   
%       lamda should be chosen based on the number of {i,j} and {i,j,k}
%       j is the target neighbors of i,  and k is the imposters of i. 
%       delta is the allowed margin violation, which is
%       hyperparameter. 
%       * However, the difference between this method and 
%         linear_metric_learning.m is that:  we replace d(x_i, x_j) with the 
%         expected distance, which means we need to consider all possible
%         paths. In this case, we don't need to find the optimal warping *
%  
%
%   hyperparameters:
%      lamda: trade off between the first and second terms in the objective 
%             function. By default, we use this based on the ratio between 
%             number of target neighbors and imposters. 
%      H:     number of hidden units. If H is less than P, it means that
%             the input will be transformed into the lower dimensional
%             space via linear transform. By default, we set this equal to 
%             the original dimension. 
%      k:     number of target neighbors. By default, we use 3
%      im:    number of imposters. This depends on the number of total data
%             points. We could incrase this as we get more data. Same rule
%             applies to k as well. 
% step_size:  step size for batch gradient descent 
%  MAX_ITER:  max number of outer iterations. The outer iteraion does
%             re-alignment (warping)
%   delta:    the margin violation we allowe. This may play important role
%             for different data sets. Carefully turning this parameter
%             will boost the performance a lot. One criteria might be used
%             is:  try to record the number of vilated constraints every
%             iteration and see if that decreases, OR plot the distance
%             between pairs of instance (plot(sort(dist)), and select the 
%             reasonable value.
%
%    return:
%      acc_val:  the change of validation accuracy
%      acc_tst:  the change of testing accuracy. 
%    dist_pair:  final distance between pairs of time series
%       best_W:  the final W has the best accuracy on the validation set. 


%% hyperparameters. 
MAX_ITER = 300;             % max number of iterations. (we don't need re-align steps) 
H = P;                      % number of hidden units.
k = 3;                      % number of target neighbors
im = 10;                    % number of imposters. 
%delta = 200;                 % margin allows to be violated. 
lamda = 1;                  % default value for lamda, this should be adaptive 
step_size = 0.00001;      % learning rate of the batch gradient descent. 
window = 10;                % the dtw window size. 

%% Initialization for parameters. 
N_t = length(X_tr);         % number of training time series. 
path_pair = cell(N_t, N_t); % store pair-wise paths computed based on DTW.
W = eye(P);                 % transformation matrix, main parameters for the model 
targets = zeros(N_t, k);    % store target neighbors for each time series
imposters = cell(N_t,1);    % store imposters for each time series. We could define 
                            % different number of imposters for each time
                            % series (this is where cell comes in).
                            % However, by default, we use the same number
                            % of imposters. 

%% Save the temporary/final results. 
acc_val = {};
obj = [];
best_acc = 0;
best_W = W;



%% Pre-processing steps...
% 1. Compute the pair-wise DTW within training data.
disp('computing pair-wise DTW....')
dist_pair = zeros(N_t, N_t);
for i=1:N_t
    for j=1:N_t
        [dist_pair(i,j),path_pair{i,j}] = mdtw_c_path(X_tr{i}, X_tr{j},window);
    end
end

% 2. Find target neighbors..
disp('finding target neighbors.....');
for i = 1: N_t
    label = y_tr(i);
    idxs = find(y_tr == label);
    dists = dist_pair(i,idxs);
    
    % find the top K time series and make them as target neighbors.
    [sortedValues,sortIndex] = sort(dists);
    targets(i,:) = idxs(sortIndex(2:k+1));    % the first one is self-> remove
    
    % pre-compute the probability matrix for all possible warpings...
    for j=1:length(targets(i,:))
        M{i,targets(i,j)} = create_pmatrix(length(X_tr{i}),length(X_tr{targets(i,j)}));
    end
end


% 3. Find imposters.
disp('finding imposters...');
for i=1:N_t
    label = y_tr(i);
    idxs = find(y_tr ~= label);
    dist_diff = dist_pair(i, idxs);
    [sortedValues, sortedIndex] = sort(dist_diff);
    
    imposter_candidates = [];
    if length(sortIndex) < im
        imposter_candidates = idxs(sortedIndex(1:end));
    else
        imposter_candidates = idxs(sortedIndex(1:im)); % select the most distance data points, as the candiates. 
    end
    imposters{i} = imposter_candidates;
    
    % pre-compute the probability matrix for all possible warpings....
    for j=1:length(imposter_candidates)
        M{i,imposter_candidates(j)} = create_pmatrix(length(X_tr{i}),length(X_tr{imposter_candidates(j)}));
    end
end

% 4. set the lamda based on the ratio of target neighbors and imposters..
num_imposters = 0;
for i=1:length(imposters)
    num_imposters = num_imposters + length(imposters{i});
end
lamda = N_t * k / num_imposters;



% 5. pre-compute the expected distance
grad_sum = zeros(P, P);
for i1 = 1: N_t
    % for target neighbors
    target = targets(i1,:);
    for i2=1:length(target)
        
        % get two time series....
        t_i = X_tr{i1};
        t_j = X_tr{target(i2)};
        [val, dist] = compute_M(t_i, t_j, P, M{i1, target(i2)});
        %tmp_M{i1, target(i2)} = val;
        tmp_M_dist{i1, target(i2)} = dist;
        grad_sum  = grad_sum + dist;
        imposter = imposters{i1};
        for i3=1:length(imposter)
            t_im = X_tr{imposter(i3)};
     
            [val, dist] = compute_M(t_i, t_im, P, M{i1, imposter(i3)});
            %tmp_M{i1, imposter(i3)} = val;
            tmp_M_dist{i1, imposter(i3)} = dist;
        end
    end
end




%% main learning part.
for itr=1:MAX_ITER
    % first compute the objective function...
    cons(itr) = 0;
    obj_tmp = 0;
    for i1 = 1: N_t
        target = targets(i1,:);
        for i2=1:length(target)
            dist_ij = expected_distance(X_tr{i1}, X_tr{target(i2)}, length(X_tr{i1}), length(X_tr{target(i2)}), M{i1,target(i2)}, W, tmp_M_dist{i1, target(i2)},0);
            obj_tmp = obj_tmp + dist_ij;
            
            imposter = imposters{i1};
            for i3=1:length(imposter)
                dist_ik = expected_distance(X_tr{i1}, X_tr{imposter(i3)}, length(X_tr{i1}), length(X_tr{imposter(i3)}), M{i1, imposter(i3)}, W, tmp_M_dist{i1, imposter(i3)},0);
                tmp = delta + dist_ij - dist_ik;
                if tmp > 0
                    obj_tmp = obj_tmp + lamda * tmp;
                end
            end
        end
    end
   
    obj(itr) = obj_tmp;

    if mod(itr, 1) == 0
        %disp('compute the accuracy for validation data');
        acc_val{itr} = compute_accuracy(X_tr, y_tr, X_val, y_val, W, 0);
        if acc_val{itr}(1) >= best_acc
            best_acc = acc_val{itr}(1);
            best_W = W;
        end
    end
    
    grad = zeros(P,H);         % gradient for current pair
    grad = grad + 2 * grad_sum * W;
    for i1 = 1: N_t
        % for target neighbors
        target = targets(i1,:);
        for i2=1:length(target)            
            dist_ij = expected_distance(X_tr{i1}, X_tr{target(i2)}, length(X_tr{i1}), length(X_tr{target(i2)}), M{i1,target(i2)}, W, tmp_M_dist{i1, target(i2)}, 0);
            % for imposters....
            imposter = imposters{i1};
            for i3=1:length(imposter)
                dist_ik = expected_distance(X_tr{i1}, X_tr{imposter(i3)}, length(X_tr{i1}), length(X_tr{imposter(i3)}), M{i1,imposter(i3)}, W, tmp_M_dist{i1, imposter(i3)},0);
                if delta + dist_ij - dist_ik > 0
                    grad = grad - lamda * 2 * tmp_M_dist{i1, imposter(i3)}*W;               
                    grad = grad + lamda * 2 * tmp_M_dist{i1, target(i2)}*W;
                    cons(itr) = cons(itr)  +1;
                end
            end   
        end       
    end
    W = W - step_size * grad;
    
    if itr > 1
        if acc_val{itr}(1) < acc_val{itr-1}(1)
            step_size = step_size * 1.05;
        elseif acc_val{itr}(1) > acc_val{itr-1}(1)
            step_size = step_size * 0.5;
        end
        if step_size < 1e-10
            step_size = 1e-10;
        end
    end
    
    disp(strcat('iter: ', num2str(itr),'  val acc: ', num2str(acc_val{itr}(1)), '  obj: ', num2str(obj(itr)), '  step size: ', num2str(step_size), '  cons: ', num2str(cons(itr))));
end


%% finally, recompute the pair-wise distance with the best weight matrix, this is used
%  for clustering. 
disp('compute pair-wise DTW using the best W')
dist_pair = zeros(N_t, N_t);
for i=1:N_t
    for j=1:N_t
        [dist_pair(i,j),path_pair{i,j}] = mdtw_c_path(X_tr{i}*best_W, X_tr{j}*best_W,window);
    end
end

acc_tst = compute_accuracy([X_tr;X_val], [y_tr;y_val], X_tst, y_tst, best_W, 0);





