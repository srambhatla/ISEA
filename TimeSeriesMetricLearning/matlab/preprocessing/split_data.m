function foldfile = split_data(datafile, nfolds)
% Split data to nfolds equally. 
% Input:    datafile: File path of the raw data.
%               X: Cell, size N. Each cell is matrix, size T_i*D
%               T: Vector, size N, lengths of series;
%               Y: Vector, size N, labels.
%           nfolds: Vector, size N, number of folds.
% Output:   foldfile: File path of the folds.
%               folds: Vector, size N, between 1 and nfolds.
%           statsfile: File path of the folding statistics.
%               stats: Matrix, size nfolds * N_labels
            
[datapath,filename,fileext] = fileparts(datafile);
foldfile = fullfile(datapath, [filename '-' int2str(nfolds) 'folds' fileext]);
statsfile = fullfile(datapath, [filename '-' int2str(nfolds) 'folds' '-stats' fileext]);
if exist(foldfile, 'file')==2
   return; 
end

disp(['Split Data: Loading data: ', datafile]);
load(datafile);
label = y;

disp(['Splitting data: ', foldfile]);
folds = data_kfold(label, nfolds, true);
save(foldfile, 'folds');

disp(['Statistics of folded data: ', statsfile]);
u = sort(unique(label), 'ascend');
stats = zeros(nfolds, length(u));
for i=1:nfolds
  I = (folds == i);
  for j=1:length(u)
	stats(i,j) = length(find(label(I) == u(j)));
  end
end
save(statsfile, 'stats', '-v7.3');
disp(['load_data' ' finished.']);
end