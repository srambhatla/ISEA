function [X, y, T, Basis, A] = synthGenSqr(MaxShift, N, Tleng, p, sig)
% Maxshift: maximum shift in the deformation operator
% T: Length of time series
% sig: noise standard deviation
B = 3;		% B: number of bases/clusters
% N: number of data points.  All clusters get equal number of data points


b = [Tleng/4, Tleng/2, Tleng*3/4];
% assert(numel(b) = B)

BT = cell(p,1);
for j = 1:p
    BT{j} = 0.5*j*exp( - ((0.5+rand(1))*ones(B, 1)*(1:Tleng) - (0.5+rand(1))*b'*ones(1, Tleng)).^2/20  );
end

Basis = cell(1,B);
for i=1:B
    Basis{i} = zeros(Tleng,p);
    for j=1:p
       Basis{i}(:,j) = BT{j}(i,:)';
    end
end
A = eye(p);

%plot(Basis')
X = cell(N,1);
y = zeros(N,1);
T = Tleng*ones(N,1);
% generate a p by p psd matrix as Sigma
Sigma = rand(p,p); Sigma = Sigma + Sigma' + p*eye(p);
[~,D] = eig(Sigma); ev = sort(diag(D),'ascend');
Sigma = sig / ev(1) * Sigma;


for k = 1:B
    %Bases = [Basis(1, :); Basis(2, :)];
    for i = 1:N/B
        ni = i + (k-1)*round(N/B);
        X{ni} = zeros(Tleng, p);
        y(ni) = k;
        for j = 1:B
            if j==k 
                continue;
            end
            shift = randi(MaxShift*2+1) - MaxShift - 1;
            weight = 0.5+abs(randn(1));
            ind = (1:Tleng) + shift;
            ind2 = find( (ind > 0) & (ind < Tleng+1) );
            tMat = 10*Basis{j}(ind2,:)*A;
            X{ni}(ind(ind2), :) = X{ni}(ind(ind2), :) + weight * tMat;
        end
        % T*p
        X{ni}(:,1) = X{ni}(:,1).^2;
        X{ni}(:,2) = log(abs(X{ni}(:,2)));
        X{ni}(:,3) = abs(X{ni}(:,3)) - log(X{ni}(:,3));
        X{ni}(:,4) = X{ni}(:,4).^3 - 2*X{ni}(:,4).^2;
        X{ni}(:,5) = X{ni}(:,5)*3;
        for t = 1:Tleng
            X{ni}(t,:) = mvnrnd(X{ni}(t,:), Sigma);
        end
    end
end

save('syn-data.mat', 'X', 'y', 'T');