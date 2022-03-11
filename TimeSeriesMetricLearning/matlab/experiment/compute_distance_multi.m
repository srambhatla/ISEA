function [distancefile, neighborfile] = compute_distance(datafile, distancetype, nfolds)
% Compute the distances between series
% Input:    datafile: File path of the raw data.
%           distancetype: String, name of distance/kernel types.      
%           nfolds: Number of folds.
% Output:   distancefile: File path of the computed distances.
%               distanceTrTr: Cell, size nfolds. Each cell is matrix, size Ntrain*Nbase.
%               distanceTrTe: Cell, size nfolds. Each cell is matrix, size Ntest*NTrain.
%               baseI: Vector, size Nbase, between 1 and Ntrain, index of the kernel base elements
%           neighborfile: File path of the nearest neighbors.
%               neighborTrTe: Cell, size nfolds. Each cell is matrix, size Ntest*NTrain.

[datapath,filename,fileext] = fileparts(datafile);
foldfile = fullfile(datapath, [filename '-' int2str(nfolds) 'folds' fileext]);
distancefile = fullfile(datapath, [filename '-' distancetype '-distance' fileext]);
neighborfile = fullfile(datapath, [filename '-' distancetype '-neighbor' fileext]);
newdatafile =  fullfile(datapath, [filename '-new-' distancetype fileext]);
newdata = struct;
save_newdata = false;
network_type = 'sda';
reftype = strcat('treeMsa-', network_type);
if  strcmp(distancetype, 'mapBackSplit') || strcmp(distancetype, 'warpSelfSplit') ...
        || strcmp(distancetype, 'treeMDTWSplit') || strcmp(distancetype, 'treeEDSplit')
   reftype = strcat('treeMsaSplit-', network_type);
end
pathtype = reftype;
if  strcmp(distancetype, 'mapBack') 
    pathtype = 'treeMsa';
elseif  strcmp(distancetype, 'mapBackSplit') 
    pathtype = 'treeMsaSplit';
end

refdatafile =  fullfile(datapath, [filename '-ref-' reftype fileext]);
refdistancefile = fullfile(datapath, [filename '-MDTW-distance' fileext]);
refpathfile = fullfile(datapath, [filename '-new-' pathtype fileext]);
ref = [];
refdistance = [];
refpath = [];

disp(['Computing baseline distance... Method: ',distancetype, '. Loading data: ', datafile, '; ', foldfile]);
original = load(datafile, 'X', 'T', 'y');
folds = load(foldfile, 'folds');
I = folds.folds;
n_labels = max(original.y(:))-min(original.y(:))+1;

distanceTrTr = cell(nfolds, 1);
distanceTrTe = cell(nfolds, 1);
neighborTrTe = cell(nfolds, 1);

if  strcmp(distancetype, 'mapBack') || strcmp(distancetype, 'mapBackSplit')
   refpath = load(refpathfile, 'X_tr_path', 'X_te_path'); 
end
if strcmp(distancetype, 'netHash')
    refdatafile =  fullfile(datapath, [filename '-hid-' reftype fileext]);
    ref = load(refdatafile, 'X_tr', 'X_te');
end
if strcmp(distancetype, 'warpSelf') || strcmp(distancetype, 'warpSelfSplit') ...
            || strcmp(distancetype, 'treeMDTW') || strcmp(distancetype, 'treeED') ...
            || strcmp(distancetype, 'treeMDTWSplit') || strcmp(distancetype, 'treeEDSplit') ...
            || strcmp(distancetype, 'mapBack') || strcmp(distancetype, 'mapBackSplit')
    ref = load(refdatafile, 'X_tr', 'X_te');
end
if strcmp(distancetype, 'treeMsa') || strcmp(distancetype, 'treeMsaSplit') ...
            || strcmp(distancetype, 'treeMsaNew') || strcmp(distancetype, 'treeExMsa') || strcmp(distancetype, 'treeMsaPart')
    refdistance = load(refdistancefile, 'distanceTrTr');
end
if strcmp(distancetype, 'warpSelf') || strcmp(distancetype, 'warpSelfSplit') ...
            || strcmp(distancetype, 'treeMsa') || strcmp(distancetype, 'treeMsaSplit') ...
            || strcmp(distancetype, 'treeMsaNew') || strcmp(distancetype, 'simpleAve')...
            || strcmp(distancetype, 'warpAve') || strcmp(distancetype, 'warpAveSplit') ...
            || strcmp(distancetype, 'treeExMsa') || strcmp(distancetype, 'treeMsaPart')...
            || strcmp(distancetype, 'mapBack') || strcmp(distancetype, 'mapBackSplit')...
            || strcmp(distancetype, 'MDTW') ||  strcmp(distancetype, 'metricMDTW')
    save_newdata = true;
    newdata.X_tr = cell(nfolds, 1);
    newdata.X_te = cell(nfolds, 1);
end
if  strcmp(distancetype, 'metricMDTW')
   newdata.W = cell(nfolds, 1); 
end
if strcmp(distancetype, 'treeMsa') || strcmp(distancetype, 'treeMsaSplit') || strcmp(distancetype, 'treeMsaNew')...
        || strcmp(distancetype, 'warpAve') || strcmp(distancetype, 'warpAveSplit') ...
    newdata.x_ref = cell(nfolds, 1);
end
if strcmp(distancetype, 'treeMsa') || strcmp(distancetype, 'treeMsaSplit') || strcmp(distancetype, 'treeMsaNew')
    save_newdata = true;
    newdata.X_tr_path = cell(nfolds, 1);
    newdata.X_te_path = cell(nfolds, 1);
end
if strcmp(distancetype, 'treeMsaPart')
    save_newdata = true;
    newdata.Keep_tr = cell(nfolds,1);
    newdata.Keep_te = cell(nfolds,1);
end

%%% Main part

for i=1:nfolds
    fprintf('Take %d-th fold as validation data, others as training data...\n', i);
    trI = find(I~=i); 
    testI = find(I==i);
    xcelltrain = original.X(trI); 
    xcelltest = original.X(testI);
    y = original.y(trI);
    T = original.T(trI);
    newT = max(T);
	[xcelltrain, xcelltest] = dataNormalize(xcelltrain, T, xcelltest);
	
    %%% Compute the distances
    if strcmp(distancetype, 'MDTW')
        [distanceTrTr{i}, ~, ~] = mDTW(xcelltrain, xcelltrain);
		[distanceTrTe{i}, ~, ~] = mDTW(xcelltrain, xcelltest);
        newdata.X_tr{i} = xcelltrain;
        newdata.X_te{i} = xcelltest;       
    elseif strcmp(distancetype, 'metricMDTW')
        % Get pairs;
        n_ml = 0;
        X1=cell( numel(xcelltrain),1);
        X2=cell( numel(xcelltrain),1);
        for j=min(y):max(y)
            yid = find(y==j);
            n_id =numel(yid);
            yid = yid(randperm(n_id));
            for k=1:2:1*(n_id-1)
               n_ml = n_ml + 1;
               X1{n_ml} = xcelltrain{yid(k)};
               X2{n_ml} = xcelltrain{yid(k+1)};
            end
        end
        
%         X1 = xcelltrain{i};
%         n_ml = numel(X1); X2=cell(n_ml,1);
%         for j=1:length(X1)
%             id = randi(n_ml);
%             while (id == j || y(id)~=y(j))
%                 id = randi(n_ml);
%             end
%             X2{j} = X1{id};
%         end
        % metric learning on pairs
        p = size(X1{1},2);
        [W_ret, ~, ~] = metric_learning_tsa_gd(n_ml, p, X1, X2);
        xcelltrain_ret = cell(numel(W_ret),1);
        xcelltest_ret = cell(numel(W_ret),1);
        for iw=1:numel(W_ret)
            W=W_ret{iw};
            %W = (W+W')/2;
            % Compute new samples;
            [V,D]=eig(W);
            D(D<1e-6)=1e-6;
            WhT=V*sqrt(D);
            W=WhT*WhT';
            % dX'WdX = dX'(Wh'Wh)Dx=(WhDx)'(WhDx)
            for j=1:numel(trI)
               xcelltrain{j} =  xcelltrain{j}*WhT;
            end
            for j=1:numel(testI)
               xcelltest{j} = xcelltest{j}*WhT;
            end
           % [distanceTrTr{iw}{i}, ~, ~] = mDTW(xcelltrain, xcelltrain);
            [distanceTrTe{iw}{i}, ~, ~] = mDTW(xcelltrain, xcelltest);
            W_ret{iw} = W;
            xcelltrain_ret{iw} = xcelltrain;
            xcelltest_ret{iw} = xcelltest;
            neighborTrTe{iw}{i} = zeros(numel(testI), numel(trI));
            for j = 1 : numel(testI)
              [~, neighborTrTe{iw}{i}(j,:)] = sort(distanceTrTe{iw}{i}(j,:),'ascend');
            end   
        end
        xcelltrain = xcelltrain_ret;
        xcelltest = xcelltest_ret;
        W=W_ret;
        
        
        newdata.X_tr{i} = xcelltrain;
        newdata.X_te{i} = xcelltest;
        newdata.W{i} = W;
    elseif strcmp(distancetype, 'ED')
		xtrain = dataVectorize(xcelltrain, max(T));
		xtest = dataVectorize(xcelltest, max(T));
		[distanceTrTe{i},~,~] = l2norm(xtrain, xtest);
    elseif strcmp(distancetype, 'warpSelf')
        X_tr_ref = ref.X_tr{i};
        X_te_ref = ref.X_te{i};
        xcelltrain=warp_data_self(xcelltrain, X_tr_ref);
        xcelltest=warp_data_self(xcelltest, X_te_ref);
        [distanceTrTe{i}, ~, ~] = ED(xcelltrain, xcelltest);
        newdata.X_tr{i} = xcelltrain;
        newdata.X_te{i} = xcelltest;
    elseif strcmp(distancetype, 'warpSelfSplit')
        X_tr_ref = ref.X_tr{i};
        X_te_ref = ref.X_te{i};
        distanceTrTe{i}=zeros(numel(testI), numel(trI));
        xcelltrain_split=cell(n_labels, 1);
        xcelltest_split=cell(n_labels, 1);
       for j=min(y):max(y)
            yid = find(y==j);
            xcelltrain_split{j}=warp_data_self(xcelltrain(yid), X_tr_ref{j});
            xcelltest_split{j}=warp_data_self(xcelltest, X_te_ref{j});
            % better to do calibration
            % only for one class
            [distanceTrTe{i}(:,yid), ~, ~] = ED( xcelltrain_split{j}, xcelltest_split{j});
        end
        newdata.X_tr{i} = xcelltrain_split;
        newdata.X_te{i} = xcelltest_split;
    elseif strcmp(distancetype, 'treeMsa')
        distanceTrTr{i}=refdistance.distanceTrTr{i};
        [x_ref, ~] = tree_msa(xcelltrain, distanceTrTr{i}, 1);
        [xcelltrain, xcelltrainpath] = warp_data(xcelltrain, x_ref);
        [xcelltest, xcelltestpath] = warp_data(xcelltest, x_ref);
        newdata.X_tr{i} = xcelltrain;
        newdata.X_te{i} = xcelltest;
        newdata.x_ref{i} = x_ref;
        newdata.X_tr_path{i} = xcelltrainpath;
        newdata.X_te_path{i} = xcelltestpath;
        [distanceTrTe{i}, ~, ~] = ED(xcelltrain, xcelltest);
    elseif strcmp(distancetype, 'treeMsaSplit')
        distanceTrTr{i}=refdistance.distanceTrTr{i};
        xcelltrain_split=cell(n_labels, 1);
        xcelltest_split=cell(n_labels, 1);
        x_ref_split = cell(n_labels, 1);
        xcelltrain_path_split = cell(n_labels, 1);
        xcelltest_path_split =cell(n_labels, 1);
        distanceTrTe{i}=zeros(numel(testI), numel(trI));
        for j=min(y):max(y) % 1 to N
            yid = find(y==j);
            [x_ref, ~] = tree_msa(xcelltrain(yid), distanceTrTr{i}(yid,yid), 1);
            [xcelltrain_split{j}, xcelltrain_path_split{j}] = warp_data(xcelltrain(yid), x_ref);
            [xcelltest_split{j}, xcelltest_path_split{j}] = warp_data(xcelltest, x_ref);
            [distanceTrTe{i}(:,yid), ~, ~] = ED(xcelltrain_split{j}, xcelltest_split{j});
            x_ref_split{j} = x_ref;
        end
        newdata.X_tr{i} = xcelltrain_split;
        newdata.X_te{i} = xcelltest_split;
        newdata.x_ref{i} = x_ref_split;
        newdata.X_tr_path{i} = xcelltrain_path_split;
        newdata.X_te_path{i} = xcelltest_path_split;
    elseif strcmp(distancetype, 'mapBack')
        X_tr_ref = ref.X_tr{i};
        X_te_ref = ref.X_te{i};
        X_tr_path = refpath.X_tr_path{i};
        X_te_path = refpath.X_te_path{i};
        xcelltrain = warp_data_inverse_by_path( X_tr_ref, X_tr_path);
        xcelltest = warp_data_inverse_by_path( X_te_ref, X_te_path);
        [distanceTrTr{i}, ~, ~] = mDTW(xcelltrain, xcelltrain);
		[distanceTrTe{i}, ~, ~] = mDTW(xcelltrain, xcelltest);
        newdata.X_tr{i} = xcelltrain;
        newdata.X_te{i} = xcelltest;
    elseif strcmp(distancetype, 'mapBackSplit')
        X_tr_ref = ref.X_tr{i};
        X_te_ref = ref.X_te{i};
        X_tr_path = refpath.X_tr_path{i};
        X_te_path = refpath.X_te_path{i};
        distanceTrTe{i}=zeros(numel(testI), numel(trI));
        xcelltrain_split=cell(n_labels, 1);
        xcelltest_split=cell(n_labels, 1);
       for j=min(y):max(y)
            yid = find(y==j);
            xcelltrain_split{j}=warp_data_inverse_by_path( X_tr_ref{j}, X_tr_path{j});
            xcelltrain(yid)=xcelltrain_split{j};
            xcelltest_split{j}=warp_data_inverse_by_path(X_te_ref{j}, X_te_path{j});
            [distanceTrTe{i}(:,yid), ~, ~] = mDTW( xcelltrain_split{j}, xcelltest_split{j});
       end
        [distanceTrTr{i}, ~, ~] =  mDTW( xcelltrain, xcelltrain);
        newdata.X_tr{i} = xcelltrain_split;
        newdata.X_te{i} = xcelltest_split;
elseif strcmp(distancetype, 'treeMDTW')
        xcelltrain = ref.X_tr{i};
        xcelltest = ref.X_te{i};
        [distanceTrTr{i}, ~, ~] = mDTW(xcelltrain, xcelltrain);
        [distanceTrTe{i}, ~, ~] = mDTW(xcelltrain, xcelltest); 
	elseif strcmp(distancetype, 'treeMDTWSplit')
        xcelltrain = ref.X_tr{i};
        xcelltest = ref.X_te{i};
        distanceTrTe{i}=zeros(numel(testI), numel(trI));
        for j=min(y):max(y)
            yid = find(y==j);
            [distanceTrTe{i}(:,yid), ~, ~] = mDTW(xcelltrain{j}, xcelltest{j});
        end
    elseif strcmp(distancetype, 'treeED')
        xcelltrain = ref.X_tr{i};
        xcelltest = ref.X_te{i};
        [distanceTrTr{i}, ~, ~] = ED(xcelltrain, xcelltrain);
        [distanceTrTe{i}, ~, ~] = ED(xcelltrain, xcelltest);         
    elseif strcmp(distancetype, 'treeEDSplit')
        xcelltrain = ref.X_tr{i};
        xcelltest = ref.X_te{i};
        distanceTrTe{i}=zeros(numel(testI), numel(trI));
        for j=min(y):max(y)
            yid = find(y==j);
            [distanceTrTe{i}(:,yid), ~, ~] = ED(xcelltrain{j}, xcelltest{j});
        end        
    elseif strcmp(distancetype, 'netHash')
        X_tr_ref = ref.X_tr{i};
        X_te_ref = ref.X_te{i};
        xcelltrain=X_tr_ref;
        xcelltest=X_te_ref;
        [distanceTrTe{i}, ~, ~] = ED(xcelltrain, xcelltest);
        newdata.X_tr{i} = xcelltrain;
        newdata.X_te{i} = xcelltest;
        
%%% comparisons        
    elseif strcmp(distancetype, 'warpAve')
        x_celltrain_ave=shrink_data(xcelltrain, newT);
        x_ref = average_data(x_celltrain_ave);
        xcelltrain=warp_data(xcelltrain, x_ref);
        xcelltest=warp_data(xcelltest, x_ref);
        newdata.X_tr{i} = xcelltrain;
        newdata.X_te{i} = xcelltest;
        newdata.x_ref = x_ref;
        [distanceTrTe{i}, ~, ~] = ED(xcelltrain, xcelltest);
    elseif strcmp(distancetype, 'warpAveSplit')
        xcelltrain_split=cell(n_labels, 1);
        xcelltest_split=cell(n_labels, 1);
        x_ref_split = cell(n_labels, 1);
        for j=min(y):max(y)
            yid = find(y==j);
            x_celltrain_ave=shrink_data(xcelltrain(yid), newT);
            x_ref = average_data(x_celltrain_ave);
            xcelltrain_split{j}=warp_data(xcelltrain(yid), x_ref);
            xcelltest_split{j}=warp_data(xcelltest, x_ref);
            [distanceTrTe{i}(:,yid), ~, ~] = ED( xcelltrain_split{j}, xcelltest_split{j});
            x_ref_split{j} = x_ref;
        end
        newdata.X_tr{i} = xcelltrain_split;
        newdata.X_te{i} = xcelltest_split;
        newdata.x_ref{i} = x_ref_split;
    elseif strcmp(distancetype, 'simpleAve')
        xcelltrain=shrink_data(xcelltrain, newT);
        xcelltest=shrink_data(xcelltest, newT);
        [distanceTrTe{i}, ~, ~] = ED(xcelltrain, xcelltest);
        newdata.X_tr{i} = xcelltrain;
        newdata.X_te{i} = xcelltest;        
    elseif strcmp(distancetype, 'treeMsaExtend')
        % Extend the series everytime adding a sample. Not effective...
        distanceTrTr{i}=refdistance.distanceTrTr{i};
        x_ref = tree_msa(xcelltrain, distanceTrTr{i}, 2);
        disp(size(x_ref));
        xcelltrain=warp_data(xcelltrain, x_ref);
        xcelltest=warp_data(xcelltest, x_ref);
        newdata.X_tr{i} = xcelltrain;
        newdata.X_te{i} = xcelltest;
        [distanceTrTe{i}, ~, ~] = ED(xcelltrain, xcelltest); 
    elseif strcmp(distancetype, 'treeMsaWPath')
        distanceTrTr{i}=refdistance.distanceTrTr{i};
        [x_ref, xcelltrainpath] = tree_msa(xcelltrain, distanceTrTr{i}, 1);
        xcelltrain = warp_data_by_path(xcelltrain, xcelltrainpath);
        [xcelltest, xcelltestpath] = warp_data(xcelltest, x_ref);
        newdata.X_tr{i} = xcelltrain;
        newdata.X_te{i} = xcelltest;
        newdata.x_ref{i} = x_ref;
        newdata.X_tr_path{i} = xcelltrainpath;
        newdata.X_te_path{i} = xcelltestpath;
        [distanceTrTe{i}, ~, ~] = ED(xcelltrain, xcelltest);        
    elseif strcmp(distancetype, 'treeMsaPart')
        distanceTrTr{i}=refdistance.distanceTrTr{i};
        [x_ref,~] = tree_msa(xcelltrain, distanceTrTr{i}, 1);
        [xcelltrain, xcelltrainkeep] = warp_data_part(xcelltrain, x_ref);
        [xcelltest, xcelltestkeep] = warp_data_part(xcelltest, x_ref);
        newdata.X_tr{i} = xcelltrain;
        newdata.X_te{i} = xcelltest;
        newdata.Keep_tr{i} = xcelltrainkeep;
        newdata.Keep_te{i} = xcelltestkeep;
        [distanceTrTe{i}, ~, ~] = ED(xcelltrain, xcelltest);
    end

%%% all end
    fprintf('Iteration %d finished.\n', i);    

end

% save(distancefile, 'distanceTrTr', 'distanceTrTe', '-v7.3');

neighborTrTeAll = neighborTrTe;
for i = 1:numel(newdata.W{1})
    neighborTrTe = neighborTrTeAll{i};
    neighborfile = fullfile(datapath, [filename '-' distancetype  int2str(i) '-neighbor' fileext]);
    save(neighborfile, 'neighborTrTe');
end

% if save_newdata
%     disp(['saving newdata: ', newdatafile,]);
%     save(newdatafile, '-struct', 'newdata');    
% end
disp(['Baseline distance: Finished saving data: ', distancefile, '; ', neighborfile]);
end