tt = cell(6,1);
clf;
plotColor = {[0 0 0.7], [1 0 0], [0 0.7 0], [0.4 0.1 0], 'm', [0.8 0.5 0]};
plotMarker = '^v<>o+*xsd';
R = 1:2:20;
hold off;
hold all;
set(gca,'FontSize',30,'LineWidth',2)
for i=8:10
    %errorbar(R, precisionmean, precisionstd, strcat('-', plotColor(mod(i,5)+1), plotMarker(mod(i,4)+1)));
    tt{i-4} = plot(R, acc_mean{i-4}, 'Color', plotColor{i-4}, 'Marker', plotMarker(i), 'LineStyle', '-', 'LineWidth',4, 'MarkerSize',15);
    set(tt{i-4},'visible', 'off');
end
set(tt{1}, 'LineStyle', '-.', 'LineWidth',3, 'MarkerSize',5);
xlim([1,19])
set(gca,'XTick',[1:6:20])
y_min = min(vertcat(acc_mean{:}));
y_max = max(vertcat(acc_mean{:}));
ylim([y_min-(y_max-y_min)*0.05 y_max+(y_max-y_min)*0.05])
set(gca,'YTick',[round(y_min*100)/100:0.02:y_max])
set(gca, 'visible', 'off');


%legend('MDTW-D','(1,3)', '(3,10)', '(5,30)',  '(10,60)', '(20,120)',  'Location','eastoutside', 'Orientation', 'horizontal')

% for h-legend:


legend( '(5,30)',  '(10,60)', '(20,120)', 'Location','northoutside',  'Orientation', 'horizontal')
legend boxoff
% %========
% [legh,objh,outh,outm]=legend('MDTW-D','(1,3)', '(3,10)', 'Location','eastoutside','Orientation','horizontal');
% legh2=copyobj(legh,gcf);
% [legh2,objh2]=legend([tt{4},tt{5},tt{6}],  '(5,30)', '(10,60)', '(20,120)',2, 'Location','northoutside','Orientation','horizontal');