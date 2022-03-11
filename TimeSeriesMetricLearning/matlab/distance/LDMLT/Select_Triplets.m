function [triplet,rho,error]=Select_Triplets(X,factor,Mt,Y,S)
% ---------------------------------------------------------------------------------------
% SIGNATURE
% ---------------------------------------------------------------------------------------
% Author: Jiangyuan Mei
% E-Mail: meijiangyuan@gmail.com
% Date  : Sep 23 2014
% ---------------------------------------------------------------------------------------

%compute the Mahalanobis distance of all the sample pairs and the disorder using the the
%current Mahalanobis matrix M

%Input:     X             --> X{i} (i=1,2,...,n) is an D x t matrix (! t by d matrix)
%           factor        --> (quantity of triplets) = (quantity of instances) x factor
%           Y             --> label
%           Mt             --> the current Mahalanobis matrix
%           S             --> similar matrix S, recording whether dissimilar or not
% Output:   triplet       --> the built triplets
%           rho           --> the desired margin between different categories.
%           error         --> the total disorder using the current Mahalanobis matrix, 
%                             which can be ultilized to judge if the algorithm is converged.

% Jiangyuan Mei, Meizhu Liu, Hamid Reza Karimi, and Huijun Gao, 
%"LogDet Divergence based Metric Learning with Triplet Constraints 
% and Its Applications", IEEE Transactions on image processing, Accepted.

% Jiangyuan Mei, Meizhu Liu, Yuan-Fang Wang, and Huijun Gao, 
%"Learning a Mahalanobis Distance based Dynamic
%Time Warping Measure for Multivariate Time Series
%Classification". 

bias=3;
numberCandidate=size(X,2);
triplet=[];
[Distance,Disorder]=Order_Check(X,Mt,Y);

% computer the parameter rho
[f, c] = hist(Distance, 100);
l = c(floor(20));
u = c(floor(80));
rho=u-l;
%figure(1),imshow(log(Distance+1),[])
pause(0.01);

error=sum(Disorder(:));
Disorder=Disorder/(sum(Disorder(:))+eps);
Triplet_N=factor*numberCandidate;

for l=1:numberCandidate
    Sample_Length=round(sqrt(Disorder(l)*Triplet_N));
    if Sample_Length<1
        continue;
    end
    S_l=S(l,:);
    Distance_l=Distance(l,:);
    index_in=find(S_l==1);
    index_out=find(S_l==0);
    [~,index_descend]=sort(Distance_l(index_in),'descend');
    [~,index_ascend]=sort(Distance_l(index_out),'ascend');
    triplet_itemi=l;
    triplet_itemj=index_in(index_descend(bias+1:min(bias+Sample_Length,length(index_in))));
    triplet_itemk=index_out(index_ascend(bias+1:min(bias+Sample_Length,length(index_out))));
    [itemi,itemj,itemk] = meshgrid(triplet_itemi,triplet_itemj,triplet_itemk);
    new_triplet=[itemi(:),itemj(:),itemk(:)];
    triplet = [triplet; new_triplet];
end