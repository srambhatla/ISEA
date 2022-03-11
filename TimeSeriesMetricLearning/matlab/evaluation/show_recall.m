function [recallfigurefile] = show_recall(datafile, nfolds, distancetypes, refdistancetypes)
% Plot the recall for different methods
% Input:    datafile: File path of the raw data.
%           nfolds: Vector, size N, number of folds.
%           distancetypes: Cell, size ntypes. 
%               Each cell is string, name of distance/kernel types. ('ED'; 'MDTW'; 'GAK'; 'VAR')
% Output:   recallfigurefile: File path of the recall figure.

[datapath,filename,fileext] = fileparts(datafile);
ntypes = length(distancetypes);
assert(ntypes == length(refdistancetypes));

%Q = [5,10:10:30];
Q=[5,10];
R = [5,10:5:30];
plotColor = 'brgmc';
plotMarker = 'o+s*';
legendary = [];
clf;
for i=1:length(distancetypes)
    recallresultfile = compute_recall(datafile, nfolds, distancetypes{i}, refdistancetypes{i}, Q, R);
	load(recallresultfile, 'recallmean', 'recallstd');
	%errorbar(R, recallmean, recallstd, strcat('-', plotColor(mod(i,5)+1), plotMarker(mod(i,4)+1)));
	for j=1:length(Q)
		legendary = [legendary; {[num2str(Q(j)) '-NN ' distancetypes{i} '/' refdistancetypes{i}]}];
		plot(R, recallmean(j,:), strcat('-', plotColor(mod(i,5)+1), plotMarker(mod(j,4)+1)));
		hold all;
    end
end
legend(legendary);

recallfigurefile = fullfile(datapath, [filename '-results-recall' '.fig']);
saveas(gcf, recallfigurefile);
disp(['Recall plot: ', recallfigurefile]);

end