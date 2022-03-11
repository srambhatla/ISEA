% allprecisionmean = allprecisionmean([1,3,4,6,8])
% for i=1:5
% precision{i}=allprecisionmean{i}(1:2:7)
% precision{i+5} = precision{i}
% end

clf;
plotColor = {[0 0 0.7], [1 0 0], [0 0.7 0], [0.4 0.1 0], [1 0 1], [0.8 0.5 0], [0.7 0 0.2]};
plotMarker = '^v<>o+*xsdph';
R = 1:2:20;
hold off;

hold all;
set(gca,'FontSize',30,'LineWidth',2)
for i=1:4
    %errorbar(R, precisionmean, precisionstd, strcat('-', plotColor(mod(i,5)+1), plotMarker(mod(i,4)+1)));
    p = plot(R, acc_mean{i}, 'Color', plotColor{i}, 'Marker', plotMarker(i), 'LineStyle', '-.', 'LineWidth',3,  'MarkerSize',5);
end
for i=9:2:11
    %errorbar(R, precisionmean, precisionstd, strcat('-', plotColor(mod(i,5)+1), plotMarker(mod(i,4)+1)));
    p = plot(R, acc_mean{i}, 'Color', plotColor{i-4}, 'Marker', plotMarker(i), 'LineStyle', '-', 'LineWidth',4, 'MarkerSize',15);
end
xlim([1,19])
set(gca,'XTick',[1:6:19])
y_min = min(vertcat(acc_mean{:}));
y_max = max(vertcat(acc_mean{:}));

ylim([y_min-(y_max-y_min)*0.05 y_max+(y_max-y_min)*0.05])
%title('data-set name')

%legend(methods, 'Location','northoutside','Orientation','horizontal');
saveas(gcf, 'knn-plot.fig');
saveas(gcf, 'knn-plot.pdf')
save('knn.mat', 'acc_mean', 'acc_std', 'methods');


% get the legend

clf;
plotColor = {[0 0 0.7], [1 0 0], [0 0.7 0], [0.4 0.1 0], [1 0 1], [0.8 0.5 0], [0.7 0 0.2]};
plotMarker = '^v<>o+*xsdph';
hold off;
hold all;
set(gca,'FontSize',15,'LineWidth',2)
for i=1:4
    %errorbar(R, precisionmean, precisionstd, strcat('-', plotColor(mod(i,5)+1), plotMarker(mod(i,4)+1)));
    p = plot([0], [0], 'Color', plotColor{i}, 'Marker', plotMarker(i), 'LineStyle', '-.', 'LineWidth',1.5,  'MarkerSize',2.5);
    set(p, 'visible', 'off');
end
methods{1} = 'MDTW-D';
legend(methods(1:4), 'Location','northoutside','Orientation','horizontal');
legend boxoff;
set(gca, 'visible', 'off');
saveas(gcf, 'knn-legend1.fig');
saveas(gcf, 'knn-legend1.pdf')

clf;
hold off;
hold all;
set(gca,'FontSize',15,'LineWidth',2)
for i=9:2:11
    %errorbar(R, precisionmean, precisionstd, strcat('-', plotColor(mod(i,5)+1), plotMarker(mod(i,4)+1)));
    p = plot([0], [0], 'Color', plotColor{i-4}, 'Marker', plotMarker(i), 'LineStyle', '-', 'LineWidth',2, 'MarkerSize',7.5);
    set(p, 'visible', 'off');
end
legend(methods(9:2:11), 'Location','northoutside','Orientation','horizontal');
legend boxoff;
set(gca, 'visible', 'off');
saveas(gcf, 'knn-legend2.fig');
saveas(gcf, 'knn-legend2.pdf')