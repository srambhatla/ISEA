function [precisionfigurefile] = show_precision(datafile, nfolds, methodtypes)
% Plot the prediction precisions for different methods.
% Input:    datafile: File path of the raw data.
%           nfolds: Vector, size N, number of folds.
%           distancetypes: Cell, size ntypes. 
% Output:   precisionfigurefile: File path of the prediction precision figure.

[datapath,filename,fileext] = fileparts(datafile);
ntypes = length(methodtypes);

%R = [1:5,6:3:25];
R=[1:2:7 10:5:20];
R=[1:2:20]
clf;
plotColor = 'brgmc';
plotMarker = 'o+*xsdph';
%ptt=[];
allprecisionmean = cell(ntypes, 1);
allprecisionstd = cell(ntypes, 1);
for i=1:ntypes
    predictionfile = make_prediction(datafile, nfolds, methodtypes{i}, R);
    load(predictionfile, 'precisionmean', 'precisionstd');
    %predictionfile = make_prediction_auc(datafile, nfolds, methodtypes{i}, R);
	%load(predictionfile, 'precisionmean', 'precisionstd', 'aucmean', 'aucstd');
	hold all;
    
    %%% accuracy
	%errorbar(R, precisionmean, precisionstd, strcat('-', plotColor(mod(i,5)+1), plotMarker(mod(i,4)+1)));
  
    plot(R, precisionmean, strcat('-', plotColor(mod(i,5)+1), plotMarker(mod(i,6)+1)));
    allprecisionmean{i} = precisionmean;
    allprecisionstd{i} = precisionstd;

    %    ptt = [ptt,precisionmean];

    %%% AUC
    %plot(R, aucmean, strcat('-', plotColor(mod(i,5)+1), plotMarker(mod(i,6)+1)));
    %allaucmean{i} = aucmean;
    %allaucstd{i} = aucstd;
    

end
%for i=1:size(ptt,2)
%    plot([1:size(ptt,2)], ptt(i,:)', strcat('-', plotColor(mod(i,5)+1), plotMarker(mod(i,4)+1)));
%end

legend(methodtypes);

precisionfigurefile = fullfile(datapath, [filename '-results-precision' '.fig']);
saveas(gcf, precisionfigurefile);
precisiondatafile = fullfile(datapath, [filename '-results-precision' '.mat']);
save(precisiondatafile, 'methodtypes', 'allprecisionmean', 'allprecisionstd');
disp(['Precision results: ', precisionfigurefile]);

end