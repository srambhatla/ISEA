function M = create_pmatrix(m, n)
    M = zeros(m,n);
    tmp = zeros(m,n);
    tmp(:,1)=1;
    tmp(1,:)=1;
    for i=2:n
        for j=2:m
            tmp(j,i) = tmp(j,i-1) + tmp(j-1,i-1) + tmp(j-1,i);
        end
    end
    
    
    for i=1:m
        for j=1:n
            M(i,j) = tmp(i,j) * tmp(m-i+1,n-j+1);
        end
    end
    
    s = sum(sum(M));
    for i=1:m
        for j=1:n
            M(i,j) = M(i,j)/s;
        end
    end
end