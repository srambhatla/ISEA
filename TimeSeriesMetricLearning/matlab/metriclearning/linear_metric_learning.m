function [acc_val, acc_tst, obj, dist_pair, best_W] = linear_metric_learning(X_tr, y_tr, X_val, y_val, X_tst, y_tst, P, delta)
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


%% Hyper-parameters....
MAX_ITER = 2;               % max number outer iterations (re-alignment)
MAX_INNER = 50;            % max number of inner iterations. 
H = P;                      % number of hidden units.
k = 3;                      % number of target neighbors
im = 10;                    % number of imposters. 
%delta = 200;                % margin allows to be violated. 
lamda = 1;                  % default value for lamda, this should be adaptive 
step_size = 0.000005;      % learning rate of the batch gradient descent. 
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
disp('computing pair-wise DTW....');
dist_pair = zeros(N_t, N_t);
for i=1:N_t
    for j=1:N_t
        [dist_pair(i,j),path_pair{i,j}] = mdtw_c_path(X_tr{i}, X_tr{j},window);
    end
end

% 2. Find target neighbors..
disp('find targeting neighbors...');
for i = 1:N_t
    label = y_tr(i);
    idxs = find(y_tr == label);
    dists = dist_pair(i,idxs); 
    
    % find the top K time series and make them as target neighbors.
    [sortedValues,sortIndex] = sort(dists);
    targets(i,:) = idxs(sortIndex(2:k+1));    % the first one is self-> remove
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
        imposter_candidates = idxs(sortedIndex(1:im));
    end
    imposters{i} = imposter_candidates;
end

% 4. set the lamda based on the ratio of target neighbors and imposters..
num_imposters = 0;
for i=1:length(imposters)
    num_imposters = num_imposters + length(imposters{i});
end
num_targets = N_t*k;
lamda = num_targets/num_imposters;



dist_res = [];
dist_res(1) = 0;
for i1 = 2: N_t
    [dist_res(i1),~] = mdtw_c_path(X_tr{1}*W, X_tr{i1}*W,window);
end
  

% 5. pre-compute some gradients, this will accelerate the learning
%    procedure.
disp('pre-computing the gradients...');
grad_tmp = cell(N_t,N_t);
grad_target = zeros(P,P);

for i1 = 1: N_t
    % for target neighbors
    target = targets(i1,:);
    for i2=1:length(target)
        path = path_pair{i1,target(i2)};   % get the current best path
        % get two time series....
        t_i = X_tr{i1};
        t_j = X_tr{target(i2)};
        
        grad_tmp{i1, target(i2)} = zeros(P,P);
        for k =1: length(path)
            u = path(k,1);
            v = path(k,2);
            grad_tmp{i1, target(i2)}= grad_tmp{i1, target(i2)} + (t_i(u,:)'-t_j(v,:)')*(t_i(u,:)-t_j(v,:));
        end
        
        grad_target = grad_target + grad_tmp{i1, target(i2)};
        
        imposter = imposters{i1};
        for i3 = 1:length(imposter)
            grad_tmp{i1, imposter(i3)} = zeros(P,P);
            t_im = X_tr{imposter(i3)};
            path = path_pair{i1,imposter(i3)};   % get the current best path
            for k =1: length(path)
                u = path(k,1);
                v = path(k,2);
                grad_tmp{i1, imposter(i3)} = grad_tmp{i1, imposter(i3)} + (t_i(u,:)'-t_im(v,:)')*(t_i(u,:)-t_im(v,:));
            end
        end
    end
end


%% main learning part.
disp('start learning algorithm........');
                
for itr=1:MAX_ITER
    disp(strcat('outer iteration: ', num2str(itr)));
    
    for itr_inner = 1: MAX_INNER
        %disp(strcat('outer itr: ', num2str(itr),'  inner itr: ', num2str(itr_inner)));
        
        idx = (itr-1) *MAX_INNER + itr_inner;
        cons(idx) = 0;    % check the change of violated constraints. 
        % first, compute the objective function...
        %disp('compute the objective function...');
        obj_tmp = 0;
        for i1 = 1: N_t
            target = targets(i1,:);
            for i2=1:length(target)
                [dist,~] = mdtw_c_path(X_tr{i1}*W, X_tr{target(i2)}*W,window);
                obj_tmp = obj_tmp + dist;
                
                % compute the (1+ d(i,j)-d(i,k))_+  for each target
                % neighbor
                imposter = imposters{i1};
                for i3 = 1:length(imposter)
                    dist_ij = dist;
                    [dist_ik,~] = mdtw_c_path(X_tr{i1}*W, X_tr{imposter(i3)}*W,window);
                    tmp = delta + dist_ij - dist_ik;
                    if tmp > 0
                        obj_tmp = obj_tmp + lamda * tmp;
                    end
                end
            end      
        end
        
        obj(idx) = obj_tmp;
        
        
        if mod(itr_inner, 1) == 0
            %disp('computing the accuracy for validation data: ');
            acc_val{idx} = compute_accuracy(X_tr, y_tr, X_val, y_val, W, 0);
            
            
            %  check the validation accuracy for iteration, and store the
            %  best W that gives best validation accuracy. 
            if acc_val{idx}(1) >= best_acc 
                best_acc = acc_val{idx}(1);
                best_W = W;
            end
        end
        
        
        %disp('computing the gradient....');
        grad = zeros(P,H);         % gradient for current pair
        
        % compute gradient for the 1st term.
        grad = grad + 2 * grad_target * W;
        
        % compute the gradient for the 2nd term. 
        for i1 = 1: N_t
            % for target neighbors
            target = targets(i1,:);
            for i2=1:length(target)                
                [dist_ij,~] = mdtw_c_path(X_tr{i1}*W, X_tr{target(i2)}*W,window);
                imposter = imposters{i1};
                for i3 = 1:length(imposter)
                    [dist_ik,~] = mdtw_c_path(X_tr{i1}*W, X_tr{imposter(i3)}*W,window);
                    if delta + dist_ij - dist_ik > 0
                        cons(idx) = cons(idx) + 1;
                        grad = grad - lamda * 2 * grad_tmp{i1, imposter(i3)} * W;
                        grad = grad + lamda * 2 * grad_tmp{i1, target(i2)} * W;
                    end
                end 
            end
        end
       
        W = W - step_size * grad;
        
        if idx > 1
            if acc_val{idx}(1) < acc_val{idx-1}(1)
                step_size = step_size * 1.05;
            elseif acc_val{idx}(1) > acc_val{idx-1}(1)
                step_size = step_size * 0.5;
            end
            if step_size < 1e-10
                step_size = 1e-10;
            end
        end
        
        disp(strcat('outer: ', num2str(itr), '  inner: ', num2str(itr_inner),'  val acc: ', num2str(acc_val{idx}(1)), '  obj: ', num2str(obj(idx)), '  step size: ', num2str(step_size), '  cons: ', num2str(cons(idx))));
    end
    
    % re-compute the optimal warping for target neighbors and imposters. 
    disp('re-computing the optimal warping...');
    for i=1:N_t
        target = targets(i,:);
        for j=1:length(target)
            [~, path_pair{i,target(j)}] = mdtw_c_path(X_tr{i}*W, X_tr{target(j)}*W,window);
        end
        
        imposter = imposters{i};
        for j=1:length(imposter)
            [~, path_pair{i,imposter(j)}] = mdtw_c_path(X_tr{i}*W,X_tr{imposter(j)}*W,window);
        end
    end
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

save('cons','cons');



