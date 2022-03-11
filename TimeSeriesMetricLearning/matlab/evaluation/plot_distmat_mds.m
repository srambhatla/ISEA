% take the 4-th fold: the order is 1,2,3,5 

id_dist = 1; % think carefully
dist_used = [];
dist_used = reshape(dist_tvtvs(id_dist,:,:), size(dist_tvtvs, 2), size(dist_tvtvs, 3));
%dist_used = distanceTrTr{4};
y_tr = y(folds<4);
y_va = y(folds==5);
y_used = [y_tr;y_va];

[y_sorted, ind] = sort(y_used);
dist_sorted = dist_used(ind, ind);

clf;
imagesc(dist_sorted);            %# Create a colored plot of the matrix values
colormap(flipud(gray));
saveas(gcf, 'distmat.fig');

clf;
cl = {'r', 'g', 'b'};
dist_2d = mdscale(dist_sorted, 2);
for i=1:3
    idx = find(y_sorted==i);
    scatter(dist_2d(idx,1), dist_2d(idx,2), cl{i});
    hold on;
end
saveas(gcf, 'mds-scatter.fig');

clf;
cl = {'r', 'g', 'b'};
dist_3d = mdscale(dist_sorted, 3);
for i=1:3
    idx = find(y_sorted==i);
    scatter3(dist_3d(idx,1), dist_3d(idx,2), dist_3d(idx,3), cl{i});
    hold on;
end
saveas(gcf, 'mds-scatter3d.fig');

