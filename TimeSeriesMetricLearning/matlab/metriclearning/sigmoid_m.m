function X = sigmoid_m(X)
X = 1./(1+exp(-X));
% for i=1:size(X,1)
%     for j=1:size(X,2)
%         X(i,j)= sigmoid(X(i,j));
%     end
% end