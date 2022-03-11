% % modified from draw_knn
% acc = lm_lin;
% clf;
% plotColor = {[0 0 0.7], [1 0 0], [0 0.7 0], [0.4 0.1 0], 'm', [0.8 0.5 0]};
% plotMarker = '^v<>o+*xsd';
% hold off;
% R = [1:2:20]
% %R = [1 dim]
% R = [4 8 16 32 64]
% %[2 3 5 7 10]
% R = dim
% hold all;
% set(gca,'FontSize',30,'LineWidth',2)
% for i=1:4
%     %errorbar(R, precisionmean, precisionstd, strcat('-', plotColor(mod(i,5)+1), plotMarker(mod(i,4)+1)));
%     p = plot(R, acc(2:end,i), 'Color', plotColor{i}, 'Marker', plotMarker(i), 'LineStyle', '-.', 'LineWidth',3,  'MarkerSize',5);
% end
% for i=5:10
%     %errorbar(R, precisionmean, precisionstd, strcat('-', plotColor(mod(i,5)+1), plotMarker(mod(i,4)+1)));
%     p = plot(R, acc(2:end,i), 'Color', plotColor{i-4}, 'Marker', plotMarker(i), 'LineStyle', '-', 'LineWidth',4, 'MarkerSize',15);
% end
% % i=1
% % p = plot(R, acc(2,:), 'Color', plotColor{i}, 'Marker', plotMarker(i), 'LineStyle', '-.', 'LineWidth',3,  'MarkerSize',5);
% % i=2
% % p = plot(R, acc(3,:), 'Color', plotColor{i}, 'Marker', plotMarker(i), 'LineStyle', '-.', 'LineWidth',3,  'MarkerSize',5);
% % i=3
% % p = plot(R, acc(5,:), 'Color', plotColor{i}, 'Marker', plotMarker(i), 'LineStyle', '-.', 'LineWidth',3,  'MarkerSize',5);
% % i=4
% % p = plot(R, acc(7,:), 'Color', plotColor{i}, 'Marker', plotMarker(i), 'LineStyle', '-.', 'LineWidth',3,  'MarkerSize',5);
% % i=5
% % p = plot(R, acc(10,:), 'Color', plotColor{i}, 'Marker', plotMarker(i), 'LineStyle', '-.', 'LineWidth',3,  'MarkerSize',5);
% 
% %xlim([1,19])
% %set(gca,'XTick',[1:6:20])
% y_min = min(vertcat(acc(:)));
% y_max = max(vertcat(acc(:)));
% 
% xlim([0,64])
% %set(gca,'xscale','log');
% 
% 
% ylim([y_min-(y_max-y_min)*0.05 y_max+(y_max-y_min)*0.05])
% %title('data-set name')
% 
% method{1} = 'MDTW-D';
% for i=2:10
%     method{i}=int2str(dim(i-1));
% end
% 
% %legend(method, 'Location','northoutside','Orientation','horizontal');
% saveas(gcf, 'dim-plot.fig');
% saveas(gcf, 'dim-plot.pdf')
% 
% % 
% % % get the legend
% % 
% % clf;
% % plotColor = {[0 0 0.7], [1 0 0], [0 0.7 0], [0.4 0.1 0], 'm', [0.8 0.5 0]};
% % plotMarker = '^v<>o+*xsd';
% % hold off;
% % hold all;
% % set(gca,'FontSize',15,'LineWidth',2)
% % for i=1:4
% %     %errorbar(R, precisionmean, precisionstd, strcat('-', plotColor(mod(i,5)+1), plotMarker(mod(i,4)+1)));
% %     p = plot([0], [0], 'Color', plotColor{i}, 'Marker', plotMarker(i), 'LineStyle', '-.', 'LineWidth',1.5,  'MarkerSize',2.5);
% %     set(p, 'visible', 'off');
% % end
% % methods{1} = 'MDTW-D';
% % legend(methods(1:4), 'Location','northoutside','Orientation','horizontal');
% % legend boxoff;
% % set(gca, 'visible', 'off');
% % saveas(gcf, 'knn-legend1.fig');
% % saveas(gcf, 'knn-legend1.pdf')
% % 
% % clf;
% % hold off;
% % hold all;
% % set(gca,'FontSize',15,'LineWidth',2)
% % for i=5:10
% %     %errorbar(R, precisionmean, precisionstd, strcat('-', plotColor(mod(i,5)+1), plotMarker(mod(i,4)+1)));
% %     p = plot([0], [0], 'Color', plotColor{i-4}, 'Marker', plotMarker(i), 'LineStyle', '-', 'LineWidth',2, 'MarkerSize',7.5);
% %     set(p, 'visible', 'off');
% % end
% % legend(methods(5:10), 'Location','northoutside','Orientation','horizontal');
% % legend boxoff;
% % set(gca, 'visible', 'off');
% % saveas(gcf, 'knn-legend2.fig');
% % saveas(gcf, 'knn-legend2.pdf')




method = {'MDTW-D-1NN', 'MDTW-D-5NN', 'LM-e-1NN','LM-e-5NN', 'LM-s-1NN', 'LM-s-5NN'};
R = [4 8 16 32 48 64];
acc = cell(6,1);
% results for eeg-bin
% acc{1} = 0.6577*ones(1,6);
% acc{2} = 0.6557*ones(1,6);
% acc{3} = [0.6990    0.7070    0.7109    0.6954    0.6906    0.6999];
% acc{4} = [0.6952    0.7066    0.7021    0.7159    0.7063    0.7137];
% acc{5} = [0.66966633 0.65743175 0.69074823 0.68114257 0.69929221 0.69479272];
% acc{6} = [0.69089990 0.67618807 0.70409505 0.70844287 0.71076845 0.69974722];

% results for eeg
acc{1} = 0.3026*ones(1,6);
acc{2} = 0.3652*ones(1,6);
acc{3} = [0.35553922 0.34666667 0.35164216 0.33617647 0.35056373 0.34485294]
acc{4} = [0.37281863 0.37879902 0.37227941 0.37436275 0.39654412 0.36757353]
acc{5} = [0.32892157 0.35735294 0.37227941 0.36811275 0.37987745 0.38875   ]
acc{6} = [0.38821078 0.3845098  0.39580882 0.39056373  0.39291667 0.38794118]

plotColor = {[0 0 0.7], [1 0 0], [0 0.7 0], [0.4 0.1 0], 'm', [0.8 0.5 0], [0 0 0]};
plotMarker = '^v<>o+*xsd';

hold off;
hold all;
set(gca,'FontSize',30,'LineWidth',2)
for i=1:0
    p = plot(R, acc{i}, 'Color', plotColor{i}, 'Marker', plotMarker(i), 'LineStyle', '-.', 'LineWidth',3,  'MarkerSize',5);
    set(p, 'visible', 'off');
end
for i=3:4
    p = plot(R, acc{i-2}, 'Color', plotColor{i-2}, 'Marker', plotMarker(i), 'LineStyle', '-', 'LineWidth',4, 'MarkerSize',15);
    set(p, 'visible', 'off');
end
y_min = min(vertcat(acc{:}));
y_max = max(vertcat(acc{:}));
xlim([1,64]);
set(gca,'XTick',[4 8 16 32 48 64])
set(gca, 'visible', 'off');
%legend(method);

% for h-legend
legend( method(1:2), 'Location','northoutside')
%,  'Orientation', 'horizontal'
legend boxoff;
set(gca, 'visible', 'off');
saveas(gcf, 'h-legend.fig');
saveas(gcf, 'h-legend.pdf')