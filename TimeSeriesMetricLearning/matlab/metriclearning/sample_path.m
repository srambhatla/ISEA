function [xpaths, ypaths] = sample_path(m, n, pMatrix, k)
%   m : the length of x-axis
%   n : the length of y-axis
%   k : the number of sampled paths. 

xpaths = cell(1,k);
ypaths = cell(1,k);

bounds = zeros(m-1,n-1,3);
bounds(:,:,1) = pMatrix(1:m-1,2:n);
bounds(:,:,2) = bounds(:,:,1) + pMatrix(2:m,2:n);
bounds(:,:,3) = bounds(:,:,2) + pMatrix(2:m,1:n-1);
xpath = zeros(1,m+n);
ypath = zeros(1,m+n);

for i=1:k
    % sample ith path.... each time we sample uniformly based
    % on the probability matrix $pMatrix$, there are three directions
    % we can choose each time.  i.e at (i,j), we will choose (i+1,j), 
    % (i+1,j+1), (i,j+1) based on the probability pMatrix(i+1,j),
    % pMatrix(i+1, j+1), pMatrix(i,j+1). sampling process will terminate
    % whenever we reach the boundary. 

    % first position
    xpath(1)=1;
    ypath(1)=1;
    
    % index of movements for x-axis and y-axis
    u=1;
    v=1;
    idx = 2;   
    
    % sample uniformly based on pMatrix(u,v+1), pMatrix(u+1,v+1), pMatrix(u+1,v)
    while not (u==m || v==n)
        % select the next move based on probability 
        r = rand * bounds(u,v,3);
        sample_idx = find(bounds(u,v,:) >= r,1);
        
        if sample_idx == 1
            v = v + 1;  
        elseif sample_idx == 2
            u = u + 1;
            v = v + 1;
        else
            u = u + 1;
        end
        
        xpath(idx) = u;
        ypath(idx) = v;
        idx = idx + 1;    
    end
    if (v<n)
        % go upwards until reach the top
        l = n-v;
        ypath(idx:idx+l-1) = v+1:n;
        xpath(idx:idx+l-1) = m;
        idx = idx + l;
    end
    if (u<m)
       % go right until reach the boundary
       l = m-u;
       xpath(idx:idx+l-1) = u+1:m;
       ypath(idx:idx+l-1) = n;        
       idx = idx + l;
    end
       
    xpaths{i} = xpath(1:idx-1);
    ypaths{i} = ypath(1:idx-1);
   
end







