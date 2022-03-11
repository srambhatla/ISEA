function [W_ret, Z, Y] = metric_learning_tsa_fw(n, p, X1, X2, method, Ystr)%, Y)
%METRIC_LEARNING_TSA Summary of this function goes here
%   Detailed explanation goes here


% number of pairs n; dimention p
% cells of time series (X1, X2)
% cell of ground-truth alignment Y

% hyperparameters
lambda = 1; % L2 regularizer
window_size = 10;
fprintf('Metric MDTW (FW-Dual solver) for %d pairs of time series (T by %d), Loss: %s, Y:%s ...\n', n, p, method, Ystr);

T1 = zeros(n,1);
T2 = zeros(n,1);
for i=1:n
    T1(i) = size(X1{i},1);
    T2(i) = size(X2{i},1);
end

W=diag(ones(p,1));
C = affinity_matrix(X1, X2, T1, T2, n, W);
dist = zeros(n,1);

% if Y does not exist, compute it
Y=cell(n,1);
if strcmp(Ystr, 'dtwY')
    % Initial with DTW alignment;
    for i=1:n
       [dist(i), Y{i}]=mdtw_c_align_bymatrix(-C{i}, window_size); 
    end
elseif strcmp(Ystr, 'diagY')
    % Initial with diagonal alignment:
    for i=1:n 
        Y{i} = zeros(T1(i), T2(i));
        if (T1(i) >= T2(i))
           for j1=1:T1(i)
              Y{i}(j1, round((T2(i)-1)*(j1-1)/(T1(i)-1))+1) = 1;
           end
        else
           for j2=1:T2(i)
              Y{i}(round((T1(i)-1)*(j2-1)/(T2(i)-1))+1, j2) = 1;
           end
        end
    end
end

numK = 60;
W_ret = cell(numK,1);

Z = Y;
W = metric_matrix(X1, X2, T1, T2, Y, Z, n, p, lambda);

% Frank¨CWolfe algorithm iterations
for k = 1:numK
    % for each aligned pairs
    %fprintf('Iteration of FW alg: %d\n', k);
    gamma = 2/(k+1);
    
    % compute W
    % compute affine matrix C
    C = affinity_matrix(X1, X2, T1, T2, n, W);
    
    for i=1:n
       % compute dF(z_i)
       
       if strcmp(method, 'hammL')
        % Hamming loss:
           deltaF = -ones(T1(i),T2(i)) +2*Y{i}; 
       elseif strcmp(method, 'areaL')
            % Symmetric area loss: 
            %MINUS: 2(L'La-Da)Z + DaU- 2L'LaY + 2Z(LL'b-Db) + UDb - 2YLL'b
           La = tril(ones(T1(i))); LLa = La' * La;  Da = max(eig(LLa));
           Lb = tril(ones(T2(i))); LLb = Lb * (Lb'); Db = max(eig(LLb));
           deltaF1 = 2*(LLa-Da*ones(T1(i)))*Z{i} + Da*ones(T1(i))*ones(T1(i),T2(i)) - 2*LLa*Y{i};
           deltaF2 = 2*Z{i}*(LLb-Db*ones(T2(i))) + Db*ones(T1(i),T2(i))*ones(T2(i)) - 2*Y{i}*LLb;
           deltaF = -deltaF1-deltaF2;
       end
       %new Affine Matrix for Hamming Distance
       deltaF = deltaF - lambda*C{i};
       % find new alignment to minimize z'*deltaF
       [~,Z_k]=mdtw_c_align_bymatrix(deltaF, window_size);

       % update Z
       Z{i} = (1-gamma)*Z{i}+gamma*Z_k;
       
       % test
       %[dist(i), ~]=mdtw_c_align_bymatrix(-C{i});
    end
    %disp('Iteration finished');
    W = metric_matrix(X1, X2, T1, T2, Y, Z, n, p, lambda);
    
    [V,D] = eig(W);
    D(D<1e-9) = 1e-9;
    W = V*D*V';
    W_ret{k}=W;
    %disp('Iteration finished');
    
    % evaluate W
    if (mod(k,10)==0)
        fprintf('\tMetric MDTW i: %d\n', k);
%         for i=1:n
%             [~,Y{i}]=mdtw_c_align_bymatrix(-C{i});
%         end
        fprintf('\t\tNorms of W: Fro: %f\tL2: %f\n', norm(W,'fro'), norm(W,2));
        % compute the loss function:
%         if strcmp(method, 'hammL')
%             [loss_hinge, loss_origin] = metric_loss_hamming(X1, X2, T1, T2, n, W, Y, lambda);
%         elseif strcmp(method, 'areaL')
%             [loss_hinge, loss_origin] = metric_loss_area(X1, X2, T1, T2, n, W, Y, Z, lambda);
%         end
%         fprintf('\t\tHinge Loss: %f, Original Loss: %f\n', loss_hinge, loss_origin);
         fprintf('\t\tTop Eigen of W: ');
        eigW = sort(real(eig(W)'), 'descend');
        disp(eigW(1:min(10,numel(eigW))));
    end
    
end

%W_ret = W_ret(end);

C = affinity_matrix(X1, X2, T1, T2, n, W);
for i=1:n
   [dist(i), Y{i}]=mdtw_c_align_bymatrix(-C{i}, window_size); 
end

disp('Metric MDTW FW finished');
end

