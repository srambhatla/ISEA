function [predictionfile] = make_prediction_auc(datafile, nfolds, distancetype, R)
% Compute the prediction for different methods
% Input:    datafile: File path of the raw data.
%           nfolds: Vector, size N, number of folds.
%           distancetype: string, name of distance/kernel types. ('ED'; 'MDTW'; 'GAK'; 'VAR')
%           R: Vector, # of retrieved nearest neighbors
% Output:   predictionfile: File path of the prediction precision results.
%%
if nargin <=3
    R=[1:2:7 10:5:20];
end
%%

%datafile = '../new_data/ras-data/ras-data.mat';
%nfolds = 5
%distancetype = 'MDTW'
%R=[1:20]


[datapath,filename,fileext] = fileparts(datafile);
foldfile = fullfile(datapath, [filename '-' int2str(nfolds) 'folds' fileext]);
neighborfile = fullfile(datapath, [filename '-' distancetype '-neighbor' fileext]);
predictionfile = fullfile(datapath, [filename '-' distancetype '-prediction' fileext]);
%%
disp(['Making Prediction... Method: ', distancetype, ' Loading data: ', datafile]);
load(datafile, 'y');
load(foldfile, 'folds');
load(neighborfile, 'neighborTrTe');

I = folds;

prediction = cell(nfolds, 1);
pred = cell(nfolds, 1);
pred_pr = cell(nfolds, 1);
precision = zeros(nfolds, length(R));
AUC = zeros(nfolds, length(R));
ymin=min(y); ymax=max(y);
%%
for i=1:nfolds    
    fprintf('Take %d-th fold as validation data, others as training data...\n', i);
    trI = find(I~=i); testI = find(I==i);
    ytrain = y(trI); ytest = y(testI);
    n_test = length(ytest);
    prediction{i} = ytrain(neighborTrTe{i});
    corr = zeros(n_test, length(R));
    % Finer knn
    count = zeros(ymax-ymin+1, 1);
    pred{i} = zeros(n_test, length(R));
    pred_p{i} = zeros(n_test, length(R));
    for ll = 1:n_test
       for k=1:length(R)
          % most frequent prediction{i}(ll,1-R(k)) ==> prediction{i}(ll, k)
          for yy =ymin:ymax
             count(yy+1-ymin) = numel(find(prediction{i}(ll,1:R(k))==yy));
           
          end
          [freq, ylabel] = sort(count, 'descend');
          pred{i}(ll,k) = ylabel(1)+ymin-1;
          pred_pr{i}(ll,k) = mean((prediction{i}(ll,1:R(k))==ymax));
          
          id_nearest = length(ytrain);
          for yy=2:ymax+1-ymin
             if freq(yy) < freq(1)
                 break;
             end
             id = find(prediction{i}(ll,:)==ylabel(yy), 1);
             if (id < id_nearest)
                  pred{i}(ll,k) = ylabel(1)+ymin-1;
             end
          end
       end
    end  
    
    for j=1:n_test
        corr(j,:) = (pred{i}(j,:) == ytest(j));
        
    end
    for k = 1:length(R)
    
    [~,~,~,auc] = perfcurve(ytest',pred_pr{i}(:,k),ymax);
    AUC(i, k) = auc;
    end
    
    for k=1:length(R)
       precision(i, k) = sum(corr(:,k));
    end
    precision(i,:) = precision(i,:) ./ (n_test);
    fprintf('Iteration %d finished.\n', i);
end
% confusion matrix
% cm = zeros(9,9);
% for i=1:151
% for j =1:5
% cm(ytrue(i),pred(i,j)) = cm(ytrue(i),pred(i,j)) +0.2;
% end
% end
% disp(cm);

precisionmean = zeros(length(R),1);
precisionstd = zeros(length(R),1);

aucmean = mean(AUC,1);
aucstd = std(AUC,1);


for k=1:length(R)
    precisionmean(k) = mean(precision(:,k));
    precisionstd(k) = std(precision(:,k));
    
end

save(predictionfile, 'prediction', 'pred', 'precisionmean', 'precisionstd', 'aucmean', 'aucstd');
disp(['Prediction results: ', predictionfile] );
end