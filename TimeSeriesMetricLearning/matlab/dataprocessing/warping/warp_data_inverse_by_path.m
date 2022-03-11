function [ Xret ] = warp_data_inverse_by_path( Xcell, Xpath )
%WARP_DATA_BY_PATH Summary of this function goes here
%   Detailed explanation goes here
n = length(Xcell);
Xret = cell(n,1);
[~,D] = size(Xcell{1});
for i=1:n
   T = size(Xpath{i},1);
   Xret{i} = zeros(T,D);
   cnt = zeros(T,1);
   for j=1:T
      for k=Xpath{i}(j,1):Xpath{i}(j,2)
         if (k > size(Xcell{i},1))
             assert(1==0);
         end
         Xret{i}(j,:) = Xret{i}(j,:) + Xcell{i}(k,:); 
         cnt(j) = cnt(j) +1;
      end
   end
   %Xret{i}=Xret{i}./repmat(Xpath{i}(:,2)-Xpath{i}(:,1)+1, [1,D]);
   Xret{i}=Xret{i}./repmat(cnt, [1,D]);
end

end

