function [ Xret ] = warp_data_by_path( Xcell, Xpath )
%WARP_DATA_BY_PATH Summary of this function goes here
%   Detailed explanation goes here
n = length(Xcell);
Xret = cell(n,1);
[~,D] = size(Xcell{1});
for i=1:n
   T = Xpath{i}(size(Xpath{i},1),2);
   Xret{i} = zeros(T,D);
   cnt = zeros(T,1);
   for j=1:size(Xpath{i},1)
      for k=Xpath{i}(j,1):Xpath{i}(j,2)
         Xret{i}(k,:) = Xret{i}(k,:) + Xcell{i}(j,:); 
         cnt(k) = cnt(k)+1;
      end
   end
   Xret{i}=Xret{i}./repmat(cnt, [1,D]);
end

end

