function [acc_val, acc_tst, obj, dist_pair, best_W] = non_linear_metric_learning(X, y, X_val, y_val, X_tst, y_tst, P)
%   IN-PROGRESS....
%   X{i}:    ith time series in the training set
%   p:       dimensionality of the time series
%   method:  'linear':  learn the linear transformation matrix
%            'neural':  learn the neural network
%            'neural-expect':  the objective function is based on GAK

%% Hyper-parameters....
N_t = length(X);
path_pair = cell(N_t, N_t);   % store pair-wise paths computed based on DTW.
MAX_ITER = 2;
MAX_INNER = 30;

H = P;                      % number of hidden units.
%W = rand(P,H);             % weight matrix for neural network W = P * H
W = eye(P);
lamda = 1;                  % default value for lamda, this should be adaptive 
step_size = 0.00000005;       % learning rate of the SGD, depends on the number of samples. 
                            % we use batch gradient descent here. 

k = 3;                      % number of target neighbors
im = 10;                    % number of imposters....
targets = zeros(N_t, k);
imposters = cell(N_t,1);    % imposters are defined for each time series

acc = {};
acc_idx = 1;
obj = [];
obj_idx = 1;

best_acc = 0;
best_W = W;
dist_permit =100;


% res = zeros(N_t,1);
% dist_pair = zeros(N_t, N_t);
% for i=1:N_t
%     for j=1:N_t
%         [dist_pair(i,j),path_pair{i,j}] = mdtw_c_path(sigmoid_m(X{i}), sigmoid_m(X{j}),10);
%         res(j) = dist_pair(i,j);
%     end
% end



    
%% Pre-processing steps......
% 1. Compute all pair-wise DTW distances. 
disp('compute pair-wise DTW....')
dist_pair = zeros(N_t, N_t);
for i=1:N_t
    for j=1:N_t
        [dist_pair(i,j),path_pair{i,j}] = mdtw_c_path(X{i}, X{j},10);
    end
end


% 2. Find target neighbors.
disp('find target neighbors...');
for i = 1: N_t
    curr_data = X{i};
    label = y(i);
    idxs = find(y == label);
    dists = dist_pair(i,idxs);   % row vector
    
    % find the top K time series and make them as target neighbors.
    [sortedValues,sortIndex] = sort(dists);
    targets(i,:) = idxs(sortIndex(2:k+1));
end

% 3. Find imposters. These are defined as the time series whose distance is
%    larger than target neighbors. 
disp('find imposters...');
for i=1:N_t
    label = y(i);
    idxs = find(y ~= label);
    dist_diff = dist_pair(i, idxs);
    [sortedValues, sortedIndex] = sort(dist_diff);
    
    imposter_candidates = [];
    if length(sortIndex) < im
        imposter_candidates = idxs(sortedIndex(1:end));
    else
        imposter_candidates = idxs(sortedIndex(1:im)); % select the most distance data points, as the candiates. 
    end
    imposters{i} = imposter_candidates;

end

% 4. set the lamda based on the ratio of target neighbors and imposters..
num_imposters = 0;
for i=1:length(imposters)
    num_imposters = num_imposters + length(imposters{i});
end
lamda = N_t * k / num_imposters;





%% main algorithm part.....
disp('start learning algorithm........');
% \sum_{i,i->j} d(i,i->j) +  lamda * \sum_{i,i->j, k} (1 + d(i,j) - d(i,k)) 
for itr=1:MAX_ITER
       
    % re-compute the gradien
    for itr_inner = 1: MAX_INNER
        cons((itr-1) *MAX_INNER + itr_inner) = 0;
        % first compute the objective function...
        obj_tmp = 0;
        for i1 = 1: N_t
            target = targets(i1,:);
            for i2=1:length(target)
                [dist,~] = mdtw_c_path(sigmoid_m(X{i1}*W), sigmoid_m(X{target(i2)}*W),10);
                obj_tmp = obj_tmp + dist;
                
                % imposters...
                imposter = imposters{i1};
                for i3=1:length(imposter)
                    dist_ij = dist;
                    [dist_ik,~] = mdtw_c_path(sigmoid_m(X{i1}*W), sigmoid_m(X{imposter(i3)}*W),10);
                    tmp = dist_permit + dist_ij - dist_ik;
                    if tmp > 0
                        obj_tmp = obj_tmp + lamda * tmp;
                    end
                    
                end
                
            end
            
        end
        
        obj(obj_idx) = obj_tmp;
        obj_idx = obj_idx + 1;
        
        if mod(itr_inner, 1) == 0
            disp('compute the accuracy for validation data');
            acc_val{acc_idx} = compute_accuracy(X, y, X_val, y_val, W, 1);
            acc_val{acc_idx}
            if acc_val{acc_idx}(1) >= best_acc
                best_acc = acc_val{acc_idx}(1);
                best_W = W;
            end
            acc_idx = acc_idx + 1;
        end
        disp(obj_tmp);
        
        
        
        grad = zeros(P,H);                 % gradient for current pair
        for i1 = 1: N_t
            % for target neighbors
            target = targets(i1,:);
            for i2=1:length(target)
                path = path_pair{i1,target(i2)};   % get the current best path
                t_i = X{i1};
                t_j = X{target(i2)};
                
                [dist_ij,~] = mdtw_c_path(sigmoid_m(X{i1}*W), sigmoid_m(X{target(i2)}*W),10);
    
                for k =1: length(path)
                    u = path(k,1);
                    v = path(k,2);
                    for h = 1:H
                        
                        tmp1 = sigmoid(W(:,h)'*t_i(u,:)');
                        tmp2 = sigmoid(W(:,h)'*t_j(v,:)');
                        grad(:,h) = grad(:,h) + 2 * (tmp1 - tmp2) * (tmp1 *(1-tmp1)*t_i(u,:)' - tmp2 * (1-tmp2) * t_j(v,:)');
                    end
                end
                
                imposter = imposters{i1};
                for i3=1:length(imposter)
                    t_im = X{imposter(i3)};
                    [dist_ik,~] = mdtw_c_path(sigmoid_m(X{i1}*W), sigmoid_m(X{imposter(i3)}*W),10);
                    if dist_permit + dist_ij - dist_ik > 0
                        cons((itr-1) *MAX_INNER + itr_inner) = cons((itr-1) *MAX_INNER + itr_inner) + 1;
                        % for target neighbor
                        path = path_pair{i1,target(i2)};   % get the current best path
                        for k =1: length(path)
                            u = path(k,1);
                            v = path(k,2);
                            for h = 1:H
                                tmp1 = sigmoid(W(:,h)'*t_i(u,:)');
                                tmp2 = sigmoid(W(:,h)'*t_j(v,:)');
                                grad(:,h) = grad(:,h) + lamda * 2 * (tmp1 - tmp2) * (tmp1 *(1-tmp1)*t_i(u,:)' - tmp2 * (1-tmp2) * t_j(v,:)');
                                
                            end
                        end
                        
                        % for imposter
                        path = path_pair{i1,imposter(i3)};   % get the current best path
                        for k =1: length(path)
                            u = path(k,1);
                            v = path(k,2);
                            for h = 1:H
                                tmp1 = sigmoid(W(:,h)'*t_i(u,:)');
                                tmp2 = sigmoid(W(:,h)'*t_im(v,:)');
                                grad(:,h) = grad(:,h) - lamda * 2 * (tmp1 - tmp2) * (tmp1 *(1-tmp1)*t_i(u,:)' - tmp2 * (1-tmp2) * t_im(v,:)');
                            end
                        end
                        
                    end      
                end
            end
            
        end
        W = W - step_size * grad;
        cons((itr-1) *MAX_INNER + itr_inner)
    end
    
    % re-align using DTW. 
    disp('re-compute the optimal warping...');
    for i=1:N_t
        target = targets(i,:);
        for j=1:length(target)
            [dist, path_pair{i,target(j)}] = mdtw_c_path(sigmoid_m(X{i}*W), sigmoid_m(X{target(j)}*W),10);
        end
        
        imposter = imposters{i};
        for j=1:length(imposter)
            [dist, path_pair{i,imposter(j)}] = mdtw_c_path(sigmoid_m(X{i}*W), sigmoid_m(X{imposter(j)}*W),10);
        end
    end
end

save('cons_non_linear','cons');

%% finally, recompute the pair-wise distance with the best weight matrix, this is used
%  for clustering. 
disp('compute pair-wise DTW using the best W')
dist_pair = zeros(N_t, N_t);
for i=1:N_t
    for j=1:N_t
        [dist_pair(i,j),path_pair{i,j}] = mdtw_c_path(sigmoid_m(X{i}*best_W), sigmoid_m(X{j}*best_W), 10);
    end
end

acc_tst = compute_accuracy([X;X_val], [y;   y_val], X_tst, y_tst, best_W, 1);





