function recallresultfile = compute_recall(datafile, nfolds, distancetype, refdistancetype, Q, R)
% Compute the recall for different methods
% Input:    datafile: File path of the raw data.
%           nfolds: Vector, size N, number of folds.
%           distancetype: string, name of distance/kernel types. ('ED'; 'MDTW'; 'GAK'; 'VAR')
%           Q: Vector, # of ground-truth nearest neighbors
%           R: Vector, # of retrieved nearest neighbors
% Output:   recallresultfile: File path of the recall results.
if nargin <=3
    Q = [5,10:10:30];
    R = [5,10:10:200];
end

[datapath,filename,fileext] = fileparts(datafile);
foldfile = fullfile(datapath, [filename '-' int2str(nfolds) 'folds' fileext]);
refneighborfile = fullfile(datapath, [filename '-' refdistancetype '-neighbor' fileext]);
neighborfile = fullfile(datapath, [filename '-' distancetype '-neighbor' fileext]);
recallresultfile = fullfile(datapath, [filename '-' distancetype '_' refdistancetype '-results-recall' fileext]);

disp(['Computing Recall... Method: ', distancetype, '/', refdistancetype ' Loading data: ', datafile]);
load(foldfile, 'folds');
refneighbors = load(refneighborfile);
neighbors = load(neighborfile);

recall = zeros(nfolds, length(Q), length(R));
R0 = [0,R];
for i=1:nfolds
    fprintf('Take %d-th fold as validation data, others as training data...\n', i);
	refneighborTrTe = refneighbors.neighborTrTe{i}; neighborTrTe = neighbors.neighborTrTe{i};
	ntest = size(refneighborTrTe,1);
	rec = zeros(ntest, length(Q), length(R));
    % @# avoid this for-loops
	for l=1:ntest
        for j=1:length(Q)
            thissetj = refneighborTrTe(l, 1:Q(j));
            for k=1:length(R)
                rec(l, j, k) = length(intersect(thissetj, neighborTrTe(l, R0(k)+1:R0(k+1))));
            end
		end
    end
    rec = squeeze(sum(rec));
    rec = cumsum(rec, 2);
    recall(i,:,:) = rec ./ (ntest * Q' * ones(1, length(R)));
	fprintf('Iteration %d finished.\n', i);
end

recallmean = squeeze(mean(recall));
recallstd = squeeze(std(recall));
save(recallresultfile, 'recall', 'recallmean', 'recallstd');
disp(['Recall results: ', recallresultfile] );
end