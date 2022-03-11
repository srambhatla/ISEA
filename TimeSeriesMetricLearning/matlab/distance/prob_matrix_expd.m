function [ pmat ] = prob_matrix_expd(t1, t2)
%PROB_MATRIX_EXPD Summary of this function goes here
%   Detailed explanation goes here
pmat = ones(t1, t2);
for i=1:t1
   for j=1:t2
       pmat(i,j)=pmat(i-1,j)+pmat(i,j-1)+pmat(i-1,j-1);
   end
end
pmat = pmat .* pmat(end:-1:1,end:-1:1);
pmat = pmat / sum(pmat(:));
end

