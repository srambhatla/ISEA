function [acc_val, acc_tst, obj, dist_pair, best_W] = non_linear_metric_learning_expected(X, y, X_val, y_val, X_tst, y_tst, P)
%   X{i}:    ith time series in the training set
%   p:       dimensionality of the time series
%   method:  'linear':  learn the linear transformation matrix
%            'neural':  learn the neural network
%            'neural-expect':  the objective function is based on GAK


%% pre-process.
%% Hyper-parameters....
N_t = length(X);
path_pair = cell(N_t, N_t);   % store pair-wise paths computed based on DTW.

MAX_ITER = 20;
M = cell(N_t, N_t);
H = P;                      % number of hidden units.
%W = rand(P,H);             % weight matrix for neural network W = P * H
W = eye(P);
lamda = 1;                % default value for lamda, this should be adaptive
step_size = 0.00001;      % learning rate of the SGD, depends on the number of samples.
% we use batch gradien t descent here.

k = 3;                      % number of target neighbors
im = 5;                    % number of imposters....
targets = zeros(N_t, k);
imposters = cell(N_t,1);      % imposters are defined for each time series


acc = {};
acc_idx = 1;
obj = [];
obj_idx = 1;

best_acc = 0;
best_W = W;


%% Pre-processing steps......
% 1. Compute all pair-wise DTW distances.
disp('compute the pair-wise DTW....');
dist_pair = zeros(N_t, N_t);
for i=1:N_t
    for j=1:N_t
        [dist_pair(i,j),path_pair{i,j}] = mdtw_c_path(X{i}, X{j},10);
    end
end

% 2. Find target neighbors.
disp('find the target neighbors....');
for i = 1: N_t
    curr_data = X{i};
    label = y(i);
    idxs = find(y == label);
    dists = dist_pair(i,idxs);   % row vector
    
    % find the top K time series and make them as target neighbors.
    [sortedValues,sortIndex] = sort(dists);
    targets(i,:) = idxs(sortIndex(2:k+1));
    for j=1:length(targets(i,:))
        M{i,targets(i,j)} = create_pmatrix(length(X{i}),length(X{targets(i,j)}));
    end
end

% 3. Find imposters. These are defined as the time series whose distance is
%    larger than target neighbors.
disp('find the imposters....');
for i=1:N_t
    dist_to_targets = dist_pair(i, targets(i,:));
    max_dist = max(dist_to_targets);
    curr_data = X{i};
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
    for j=1:length(imposter_candidates)
        M{i,imposter_candidates(j)} = create_pmatrix(length(X{i}),length(X{imposter_candidates(j)}));
    end
end

% 4. set the lamda based on the ratio of target neighbors and imposters..
num_imposters = 0;
for i=1:length(imposters)
    num_imposters = num_imposters + length(imposters{i});
end
lamda = N_t * k / num_imposters;




%% main gradient computations.......

for itr=1:MAX_ITER
    itr
    % first compute the objective function...
    disp('compute the objective function..');
    obj_tmp = 0;
    for i1 = 1: N_t
        
        target = targets(i1,:);
        for i2=1:length(target)
            dist_ij = expected_distance(X{i1}, X{target(i2)}, length(X{i1}), length(X{target(i2)}), M{i1,target(i2)}, W, 1);
            obj_tmp = obj_tmp + dist_ij;
            
            imposter = imposters{i1};
            for i3=1:length(imposter)
                dist_ik = expected_distance(X{i1}, X{imposter(i3)}, length(X{i1}), length(X{imposter(i3)}), M{i1, imposter(i3)}, W, 1);
                tmp = 1 + dist_ij - dist_ik;
                if tmp > 0
                    obj_tmp = obj_tmp + lamda * tmp;
                end
            end
        end
        
    end
    disp(obj_tmp);
    
    obj(obj_idx) = obj_tmp;
    obj_idx = obj_idx + 1;
    % might change to expected distance.
    if mod(itr, 1) == 0
        disp('compute the accuracy for validation data');
        acc_val{acc_idx} = compute_accuracy(X, y, X_val, y_val, W, 1);
        acc_val{acc_idx}
        if acc_val{acc_idx}(1) >= best_acc
            best_acc = acc_val{acc_idx}(1);
            best_W = W;
        end
        acc_idx = acc_idx + 1;
    end
    
  
    grad = zeros(P,H);         % gradient
    for i1 = 1: N_t
        % for target neighbors
        target = targets(i1,:);
        for i2=1:length(target)
            
            % get two time series....
            t_i = X{i1};
            t_j = X{target(i2)};
            
            [xpaths, ypaths] = sample_path(length(t_i), length(t_j), M{i1, target(i2)}, 10);
            for s=1: length(xpaths)
                xpath = xpaths{s};
                ypath = ypaths{s};
                for k =1: length(xpath)
                    u = xpath(k);
                    v = ypath(k);
                    for h = 1:H
                        tmp1 = sigmoid(W(:,h)'*t_i(u,:)');
                        tmp2 = sigmoid(W(:,h)'*t_j(v,:)');
                        grad(:,h) = grad(:,h) + 2/10 * (tmp1 - tmp2) * (tmp1 *(1-tmp1)*t_i(u,:)' - tmp2 * (1-tmp2) * t_j(v,:)');
                    end
                end
            end
            
            dist_ij = expected_distance(X{i1}, X{target(i2)}, length(X{i1}), length(X{target(i2)}), M{i1,target(i2)}, W, 1);
            % for imposters....
            imposter = imposters{i1};
            for i3=1:length(imposter)
                dist_ik = expected_distance(X{i1}, X{imposter(i3)}, length(X{i1}), length(X{imposter(i3)}), M{i1, imposter(i3)}, W, 1);
                if 1+dist_ij - dist_ik > 0
                    t_im = X{imposter(i3)};
                    [xpaths, ypaths] = sample_path(length(t_i), length(t_im), M{i1, imposter(i3)}, 10);
                    for s=1: length(xpaths)
                        xpath = xpaths{s};
                        ypath = ypaths{s};
                        for k =1: length(xpath)
                            u = xpath(k);
                            v = ypath(k);
                            for h = 1:H
                                tmp1 = sigmoid(W(:,h)'*t_i(u,:)');
                                tmp2 = sigmoid(W(:,h)'*t_im(v,:)');
                                grad(:,h) = grad(:,h) - lamda * 2/10 * (tmp1 - tmp2) * (tmp1 *(1-tmp1)*t_i(u,:)' - tmp2 * (1-tmp2) * t_im(v,:)');
                            end
                        end
                    end
         
                    [xpaths, ypaths] = sample_path(length(t_i), length(t_j), M{i1, target(i2)}, 10);
                    for s=1: length(xpaths)
                        xpath = xpaths{s};
                        ypath = ypaths{s};
                        for k =1: length(xpath)
                            u = xpath(k);
                            v = ypath(k);
                            for h = 1:H
                                tmp1 = sigmoid(W(:,h)'*t_i(u,:)');
                                tmp2 = sigmoid(W(:,h)'*t_j(v,:)');
                                grad(:,h) = grad(:,h) + lamda * 2/10 * (tmp1 - tmp2) * (tmp1 *(1-tmp1)*t_i(u,:)' - tmp2 * (1-tmp2) * t_j(v,:)');
                            end
                        end
                    end  
                end
                
            end
            
        end
        
    end
    W = W - step_size * grad;
end
%% finally, recompute the pair-wise distance with the best weight matrix, this is used
%  for clustering. 
disp('finally compute pair-wise DTW using the best W')
dist_pair = zeros(N_t, N_t);
for i=1:N_t
    for j=1:N_t
        [dist_pair(i,j),path_pair{i,j}] = mdtw_c_path(sigmoid_m(X{i}*best_W), sigmoid_m(X{j}*best_W), 10);
    end
end

acc_tst = compute_accuracy([X;X_val], [y;y_val], X_tst, y_tst, best_W, 1);





