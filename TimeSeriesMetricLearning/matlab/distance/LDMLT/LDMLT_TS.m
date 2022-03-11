function M = LDMLT_TS(X,Y, params)
% ---------------------------------------------------------------------------------------
% SIGNATURE
% ---------------------------------------------------------------------------------------
% Author: Jiangyuan Mei
% E-Mail: meijiangyuan@gmail.com
% Date  : Sep 23 2014
% ---------------------------------------------------------------------------------------

%generate Mahalanobis Distance  M which based Dynamic Time Warping Measure
%using LogDet Divergence based Metric Learning with Triplet Constraints
%algorithms
%Input:     X             --> X{i} (i=1,2,...,n) is an D x t matrix
%           Y             --> label
%           params        --> structure containing the parameters
%           params.tripletsfactor;  --> (quantity of triplets in each cycle) = params.tripletsfactor x (quantity of training instances)
%           params.cycle;           --> the maximum cycle of metric learning process
%           params.alphafactor;     --> alpha = params.alphafactor/(quantity of triplets in each cycle)
%Output:    M             --> learned PSD matrix

% Jiangyuan Mei, Meizhu Liu, Hamid Reza Karimi, and Huijun Gao, 
%"LogDet Divergence based Metric Learning with Triplet Constraints 
% and Its Applications", IEEE Transactions on image processing, Accepted.

% Jiangyuan Mei, Meizhu Liu, Yuan-Fang Wang, and Huijun Gao, 
%"Learning a Mahalanobis Distance based Dynamic
%Time Warping Measure for Multivariate Time Series
%Classification". 

X = reshape(X, [1, numel(X)]);


if (~exist('params')),
    params = struct();
    params = SetDefaultParams(params);
end

numberCandidate =size(X,2);
numberFeature=size(X{1},2);

%The Mahalanobis matrix M starts from identity matrix
M=eye(numberFeature,numberFeature);

% Get all the lables of the data
Y_kind=unique(Y);
[X,Y]=data_rank(X,Y,Y_kind); % rank the original data according to their label

%S record whether dissimilar or not
S = zeros(numberCandidate,numberCandidate); 
for i=1:numberCandidate
    for j=1:numberCandidate
        if Y(i)==Y(j)
            S(i,j) = 1;
        end
    end
end



[Triplet,rho,Error_old]=Select_Triplets(X,params.tripletsfactor,M,Y,S); % dynamic triplets building strategy
iter=size(Triplet,1);    
total_iter=iter;

for i=1:params.cycle
    alpha=params.alphafactor/iter;
    rho=0;
    M=update_M(M,X,Triplet,alpha,rho);  % update the Mahalanobis matrix M
    [Triplet,rho,Error_new]=Select_Triplets(X,params.tripletsfactor,M,Y,S); % dynamic triplets building strategy
    iter=size(Triplet,1);
    total_iter=total_iter+iter; % record the toatl ietrations in metric learning process.
    params.tripletsfactor=Error_new/Error_old*params.tripletsfactor; % the quantity of triplets reduces with the shrink of error
    cov=(Error_old-Error_new)/Error_old;
    if abs(cov)<10e-5
        break;
    end
    disp(sprintf('Cycle: %d, Error: %d, tol: %f,iteration: %d', i, Error_new, cov,iter));
    Error_old=Error_new;
end
disp(sprintf('LDMLT converged to error: %d, total cycle: %d, total iteration: %d ', Error_new,i,total_iter));


% The proposed LDMLT algorithm
function M=update_M(M,X,triplet,gamma,rho)
M=M/trace(M);
i=1;
options=zeros(1,5);
options(5) = 1;
while (i<size(triplet,1))
    i1 = triplet(i,1);
    i2 = triplet(i,2);
    i3 = triplet(i,3);
    
    [Dist1,swi1,swi2]=dtw_metric(X{i1},X{i2},M);  
    P=swi1-swi2;
    [Dist2,swi1,swi3]=dtw_metric(X{i1},X{i3},M);
    Q=swi1-swi3;
    IP=eye(size(P,2));
    IQ=eye(size(Q,2));
    if Dist2-Dist1<rho 
%         setlmis([]);
%         alpha = lmivar(1,[1 1]);
%         lmiterm([1 1 1 alpha],1,Q*Q'-P*P');
%         lmiterm([-1 1 1 0],M^-1);
%         lmiterm([-2 1 1 alpha],1,1);
%         lmiterm([2 1 1 0],0);
%         lmis = getlmis;
%         
%         [tmin,xfeas] = feasp(lmis,options);
%         if size(xfeas,1)==0
%             alpha=0;
%         else
%             alpha=gamma*xfeas;
%         end
        alpha=gamma/trace((eye(size(M,1))-M)^(-1)*M*Q*Q');
        M_temp=M - alpha*M*P*(IP + alpha*P'*M*P)^(-1)*P'*M;
        M = M_temp + alpha*M_temp*Q*(IQ - alpha*Q'*M_temp*Q)^(-1)*Q'*M_temp;
        [L S R]=svd(M);
        M=M/sum(trace(S));
        M=M/trace(M);
    end
    
    
    i = i + 1;
end
M=M*size(M,1);

function [X,Y]=data_rank(X,Y,Y_kind)

X_data=[];
Y_data=[];
for l=1:length(Y_kind)
    index=find(Y==Y_kind(l));
    X_data=[X_data X(index)];
    %syn-data
    Y_data=[Y_data Y(index)'];
    %non syn
    %Y_data=[Y_data Y(index)];
end
X=X_data;
Y=Y_data;
