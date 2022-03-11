function [ path1, path2 ] = split_path( path, len1, len2)
%SPLIT_PATH Summary of this function goes here
%   Detailed explanation goes here
%	if len1 > len2, keep x1; else keep x2
% path: (ti_1, ti_2)
% path1, path2: (t_begin, t_end)

T = size(path,1);
path1 = zeros(len1, 2);
path2 = zeros(len2, 2);
if (len1 > len2)
    % keep x1, find path2
   for i=1:len1
      path1(i,:) = [i,i]; 
   end
   path2(1,1)=1;
   pt2 = 1;
   for i=2:T
       if (path(i,2) ~= pt2)
          path2(pt2,2) = path(i-1,1);
          pt2 = pt2 + 1;
          path2(pt2,1) = path(i,1);
       end
   end
   path2(len2, 2) = len1;
else 
   for i=1:len2
      path2(i,:) = [i,i];
   end
   path1(1,1)=1;
   pt1 = 1;
   for i=2:T
       if (path(i,1) ~= pt1)
          path1(pt1,2) = path(i-1,2);
          pt1 = pt1 + 1;
          path1(pt1,1) = path(i,2);
       end
   end
   path1(len1, 2) = len2;
end
end

