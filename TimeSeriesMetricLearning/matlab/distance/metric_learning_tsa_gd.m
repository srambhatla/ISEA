function [W_ret, Z, Y] = metric_learning_tsa_gd(n, p, X1, X2, Ystr, Wstr)
%METRIC_LEARNING_TSA Summary of this function goes here
%   Detailed explanation goes here


% number of pairs n; dimention p
% cells of time series (X1, X2)
% cell of ground-truth alignment Y

% hyperparameters
lambda = 1; % L2 regularizer
update_align = false;
window_size = 10;

fprintf('Metric MDTW for %d pairs of time series (T by %d) Y:%s, W:%s ...\n', n, p, Ystr, Wstr);
T1 = zeros(n,1);
T2 = zeros(n,1);
for i=1:n
    T1(i) = size(X1{i},1);
    T2(i) = size(X2{i},1);
end

W=diag(ones(p,1));
C = affinity_matrix(X1, X2, T1, T2, n, W);
dist = zeros(n,1);
Z = cell(n,1);

% if Y does not exist, compute it
Y=cell(n,1);
if strcmp(Ystr, 'dtwY')
    % % Initial with DTW alignment;
    for i=1:n
       [dist(i), Y{i}]=mdtw_c_align_bymatrix(-C{i}, window_size); 
    end
elseif strcmp(Ystr, 'diagY')
    % % Initial with diagonal alignment:
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
if strcmp(Wstr, 'diagW')
    W= diag(ones(p,1));
elseif strcmp(Wstr, 'onesW')
    W=ones(p);
end

% Subgradient Descent algorithm iterations
numK = 60;
if update_align
    numK = 100;    
end
W_ret = cell(numK,1);
for k = 1:numK
    % for each aligned pairs
    %fprintf('Iteration of GD alg: %d\n', k);
    gamma = 1/(k+5);
    % compute affine matrix C
    C = affinity_matrix(X1, X2, T1, T2, n, W);

    dW = zeros(p,p);
    for i=1:n
       % compute dF(z_i)
       deltaF = -C{i} - ones(T1(i),T2(i)) + 2*Y{i}; %new Distance Matrix for Hamming Distance
       % find new alignment to minimize z'*deltaF
       [~,Z{i}]=mdtw_c_align_bymatrix(deltaF, window_size);
       % test   % notice that phi is negative!
       diffYZ= Y{i}-Z{i};
        phi=zeros(p,p);
        for j1=1:T1(i)
           for j2=1:T2(i)
               xdiff = X1{i}(j1,:)-X2{i}(j2,:);
               phi = phi + (xdiff'*xdiff)*diffYZ(j1,j2);
           end
        end
        dW = dW + phi;
    end
    dW = lambda*W + 1/n*dW;
    %dW = 1/n*dW;
    %W_f_prev = norm(W,'fro');
    W = W-gamma*dW;
    
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
%         [loss_hinge, loss_origin] = metric_loss_hamming(X1, X2, T1, T2, n, W, Y, lambda);
%         fprintf('\t\tHinge Loss: %f, Original Loss: %f\n', loss_hinge, loss_origin);
        fprintf('\t\tTop Eigen of W: ');
        eigW = sort(real(eig(W)'), 'descend');
        disp(eigW(1:min(10,numel(eigW))));
    end
    
    if (update_align && mod(k,20) == 0)
        for i=1:n
            [dist(i), Y{i}]=mdtw_c_align_bymatrix(-C{i}, window_size); 
        end
    end   
end

%W_ret = W_ret(end);

C = affinity_matrix(X1, X2, T1, T2, n, W);
for i=1:n
   [dist(i), Y{i}]=mdtw_c_align_bymatrix(-C{i}, window_size); 
end

disp('Metric MDTW GD finished');
end

