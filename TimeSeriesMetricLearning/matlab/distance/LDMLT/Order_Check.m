function [Distance,Disorder]=Order_Check(X,M,Y)
% ---------------------------------------------------------------------------------------
% SIGNATURE
% ---------------------------------------------------------------------------------------
% Author: Jiangyuan Mei
% E-Mail: meijiangyuan@gmail.com
% Date  : Sep 23 2014
% ---------------------------------------------------------------------------------------

%compute the Mahalanobis distance of all the sample pairs and the disorder using the the
%current Mahalanobis matrix M

%Input:     X             --> X{i} (i=1,2,...,n) is an D x t matrix
%           Y             --> label
%           M             --> the current Mahalanobis matrix
%Output:    Distance      --> Mahalanobis distance of all the sample pairs {(i, j)} in X
%           disorder      --> n-by-1 vector, which record the disorder of each row

% Jiangyuan Mei, Meizhu Liu, Hamid Reza Karimi, and Huijun Gao, 
%"LogDet Divergence based Metric Learning with Triplet Constraints 
% and Its Applications", IEEE Transactions on image processing, Accepted.

% Jiangyuan Mei, Meizhu Liu, Yuan-Fang Wang, and Huijun Gao, 
%"Learning a Mahalanobis Distance based Dynamic
%Time Warping Measure for Multivariate Time Series
%Classification". 



numberCandidate=size(X,2);


compactfactor=2;
Y_kind=sort(unique(Y),'ascend');
index=1;
j=0;
for i=1:numberCandidate   
    if Y(i)==Y(index) & j<compactfactor
        map_vector(i)=index;
        j=j+1;
    else
        index=index+j;
        map_vector(i)=index;
        j=1;
    end   
end

map_vector_kind=unique(map_vector);
S = zeros(length(map_vector_kind),length(map_vector_kind)); 
for i=1:length(map_vector_kind)
    for j=1:length(map_vector_kind)
        if Y(map_vector_kind(i))==Y(map_vector_kind(j))
            S(i,j) = 1;
        end
    end
end



Distance=zeros(length(map_vector_kind),length(map_vector_kind));


for i=1:length(map_vector_kind)
    for j=i:length(map_vector_kind)
        [Dist,sw1,sw2]=dtw_metric(X{map_vector_kind(i)},X{map_vector_kind(j)},M);
        Distance(i,j)=Dist;
    end
end

for i=1:length(map_vector_kind)
    for j=1:i-1
        Distance(i,j)=Distance(j,i);
    end
end

for i=1:length(map_vector_kind)
    Distance_i=Distance(i,:);
    S_i=S(i,:);
    [~,index_ascend]=sort(Distance_i,'ascend');
    S_new=S_i(index_ascend);
    sum_in=sum(S_new==1);
    rs1=sum_in;
    rs2=0;   
    for j=1:length(map_vector_kind)
        if S_new(j)==0;
            rs2=rs2+rs1;
        else
            rs1=rs1-1;            
        end
    end 
    index=find(map_vector==map_vector_kind(i));
    Disorder(index)=rs2;
end

Distance_Low=Distance;
for i=1:length(map_vector_kind)
    index_i=find(map_vector==map_vector_kind(i));
    for j=1:length(map_vector_kind)        
        index_j=find(map_vector==map_vector_kind(j));
        Distance(index_i,index_j)=Distance_Low(i,j);
    end
end