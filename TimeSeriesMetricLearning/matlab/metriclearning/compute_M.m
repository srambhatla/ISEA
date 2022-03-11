function [val,dist] = compute_M(X_1, X_2, P, pMatrix)

val = zeros(P,P);
dist = zeros(P,P);
for i=1:length(X_1)
    for j=1:length(X_2)
        tmp = (X_1(i,:)-X_2(j,:))'*(X_1(i,:)-X_2(j,:));
        val = val + tmp;
        dist = dist + tmp * pMatrix(i,j);
    end
end