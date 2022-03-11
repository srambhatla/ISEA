function exp_dist = expected_distance(X_1, X_2, m, n, pMatrix, W, tmp_dist, method)
%   compute the expected distance between X_1 and X_2. 
%   X_1, X_2 are both colume vectors.  X_1 is 
exp_dist = 0;


if method == 0
    exp_dist = trace(tmp_dist * W * W');
else
    for i=1:m
        for j=1:n
            if method == 0
                exp_dist = exp_dist + ((X_1(i,:)-X_2(j,:))*W)*((X_1(i,:)-X_2(j,:))*W)' * pMatrix(i,j);
            else
                exp_dist = exp_dist + (sigmoid_m((X_1(i,:)-X_2(j,:))*W))*(sigmoid_m((X_1(i,:)-X_2(j,:))*W))' * pMatrix(i,j);
            end
            
        end
    end
end


