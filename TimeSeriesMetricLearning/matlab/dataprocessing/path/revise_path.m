function [ new_path ] = revise_path( base_path, sub_path )
%REVISE_PATH Summary of this function goes here
%   Detailed explanation goes here
% new_path, base_path, sub_path: (t_begin, t_end)

new_path = zeros(size(sub_path));
new_path(:,1) = base_path(sub_path(:,1),1);
new_path(:,2) = base_path(sub_path(:,2),2);
for i=1:size(sub_path,1)-1
   j=i;
   while (j<size(sub_path,1) && new_path(i,2) > new_path(j+1,1))
      j=j+1; 
   end
   if (j>i)
      % Average path between [i,j] from new_path(i,1) to new_path(j,2)
      initp = new_path(j,1);
      diffp = new_path(j,2)-new_path(j,1);
      diffi = j-i;
      if (diffp >= diffi)
          % longer length; no-overlapping
         for k=1:diffi
            new_path(k+i-1,2) = initp + int32((k)/(diffi+1)*diffp-0.5);
            new_path(k+i,1) = initp + int32(k/(diffi+1)*diffp+0.5);
         end
      else
         % all length 1; overlapping
         if (initp > new_path(i,1) && 1/diffi*diffp<0.5)
            new_path(i,2) = initp-1; 
         else 
            new_path(i,2) = initp;
         end
         for k=1:diffi
            new_path(k+i,1)=initp + int32(k/diffi*diffp);
            new_path(k+i,2)=initp + int32(k/diffi*diffp);
         end
      end
   end
end
for i=1:size(sub_path,1)-1
    if (new_path(i,1)< new_path(i,2) && new_path(i,2) == new_path(i+1,1))
       new_path(i,2) =  new_path(i,2)-1;
    end
    if (new_path(i+1,1)< new_path(i+1,2) && new_path(i,2) == new_path(i+1,1))
       new_path(i+1,1) =  new_path(i+1,1)+1;
    end
end

end

