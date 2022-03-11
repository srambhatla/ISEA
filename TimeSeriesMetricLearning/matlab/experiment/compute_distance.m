function [distancefile, neighborfile] = compute_distance(datafile, typeStr, nfolds)
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

typeStrs = strsplit(typeStr, '-');
distancetype = typeStrs{1};

[datapath,filename,fileext] = fileparts(datafile);
foldfile = fullfile(datapath, [filename '-' int2str(nfolds) 'folds' fileext]);
distancefile = fullfile(datapath, [filename '-' typeStr '-distance' fileext]);
neighborfile = fullfile(datapath, [filename '-' typeStr '-neighbor' fileext]);

newdatafile =  fullfile(datapath, [filename '-new-' typeStr fileext]);
newdata = struct;
save_newdata = false;

reftype = '';
if  strcmp(distancetype, 'mapBack') || strcmp(distancetype, 'warpSelf') ...
        || strcmp(distancetype, 'treeMDTW') || strcmp(distancetype, 'treeED')
   reftype = strcat('treeMsa-', typeStrs{2});
end
if  strcmp(distancetype, 'mapBackSplit') || strcmp(distancetype, 'warpSelfSplit') ...
        || strcmp(distancetype, 'treeMDTWSplit') || strcmp(distancetype, 'treeEDSplit')
   reftype = strcat('treeMsaSplit-', typeStrs{2});
end

pathtype = '';
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
   
    y = original.y(trI);
    T = original.T(trI);
    newT = max(T);
    xcelltrain = original.X(trI); 
    xcelltest = original.X(testI);
    [xcelltrain, xcelltest] = dataNormalize(xcelltrain, T, xcelltest);
	
%     [xcellall,~] = dataNormalize(original.X, original.T); 
%     xcelltrain = xcellall(trI);
%     xcelltest = xcellall(testI);
    
    %%% Compute the distances
    if strcmp(distancetype, 'MDTW')
        [distanceTrTr{i}, ~, ~] = mDTW(xcelltrain, xcelltrain);
		[distanceTrTe{i}, ~, ~] = mDTW(xcelltrain, xcelltest);
        newdata.X_tr{i} = xcelltrain;
        newdata.X_te{i} = xcelltest;  
    elseif strcmp(distancetype, 'MDTW_1')
        [distanceTrTr{i}, ~, ~] = mDTW_1(xcelltrain, xcelltrain);
		[distanceTrTe{i}, ~, ~] = mDTW_1(xcelltrain, xcelltest);
        newdata.X_tr{i} = xcelltrain;
        newdata.X_te{i} = xcelltest;  
    elseif strcmp(distancetype, 'LDMLT_TS')
        params_LDMLT.tripletsfactor= 20;    % (quantity of triplets in each cycle) = params.tripletsfactor x (quantity of training instances)
        params_LDMLT.cycle= 15;         % the maximum cycle 
        params_LDMLT.alphafactor= 5;    % alpha = params.alphafactor/(quantity of triplets in each cycle)
        M_LDMLT= LDMLT_TS(xcelltrain, y, params_LDMLT);
        distanceTrTe{i}=zeros(numel(trI), numel(trI));
        distanceTrTe{i}=zeros(numel(testI), numel(trI));
        for i1=1:numel(trI)
            for i2=i1:numel(trI)
                distanceTrTr{i}(i1, i2) = dtw_metric(xcelltrain{i1}, xcelltrain{i2}, M_LDMLT);
                distanceTrTr{i}(i2, i1) = distanceTrTr{i}(i1, i2);
            end
            for i2=1:numel(testI)
                distanceTrTe{i}(i2, i1) = dtw_metric(xcelltrain{i1}, xcelltest{i2}, M_LDMLT);
            end
        end
    elseif strcmp(distancetype, 'GAK')
        [sigma_ga, T_ga, ~] = GAparameter(xcelltrain);
		[distanceTrTr{i}, ~, ~] = GAK(xcelltrain, xcelltrain, sigma_ga, T_ga);
		[distanceTrTe{i}, ~, ~] = GAK(xcelltrain, xcelltest, sigma_ga, T_ga);
        newdata.X_tr{i} = xcelltrain;
        newdata.X_te{i} = xcelltest;         
    elseif strcmp(distancetype, 'expDist')
        [distanceTrTr{i}, ~, ~] = expDist(xcelltrain, xcelltrain);
		[distanceTrTe{i}, ~, ~] = expDist(xcelltrain, xcelltest);
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
            for k=1:2:(n_id-1)
               n_ml = n_ml + 1;
               X1{n_ml} = xcelltrain{yid(k)};
               X2{n_ml} = xcelltrain{yid(k+1)};
            end
        end

        % metric learning on pairs
        p = size(X1{1},2);
        lossStr = typeStrs{2};
        if strcmp(lossStr, 'hammL')
            Ystr = typeStrs{3}; Wstr = typeStrs{4};
            [W_ret, ~, ~] = metric_learning_tsa_gd(n_ml, p, X1, X2, Ystr, Wstr);
        elseif strcmp(lossStr, 'areaL')
            Ystr = typeStrs{3};
            [W_ret, ~, ~] = metric_learning_tsa_fw(n_ml, p, X1, X2, lossStr, Ystr);
        end
        xcelltrain_ret = cell(numel(W_ret),1);
        xcelltest_ret = cell(numel(W_ret),1);
        xcelltrain_iw = cell(numel(xcelltrain), 1);
        xcelltest_iw = cell(numel(xcelltest), 1);
        for iw=1:numel(W_ret)
            W_iw=W_ret{iw};
            % Compute new samples;
            [V,D]=eig(W_iw);
            D(D<1e-9)=1e-9;
            WhT=V*sqrt(D);
            W_iw=WhT*WhT';
            % dX'WdX = dX'(Wh'Wh)Dx=(WhDx)'(WhDx)
            for j=1:numel(trI)
               xcelltrain_iw{j} =  xcelltrain{j}*WhT;
            end
            for j=1:numel(testI)
               xcelltest_iw{j} = xcelltest{j}*WhT;
            end
            W_ret{iw} = W_iw;
            xcelltrain_ret{iw} = xcelltrain_iw;
            xcelltest_ret{iw} = xcelltest_iw;
        end
        xcelltrain = xcelltrain_ret;
        xcelltest = xcelltest_ret;
        W=W_ret{numel(W_ret)};
        
        [distanceTrTr{i}, ~, ~] = mDTW(xcelltrain{end}, xcelltrain{end});
        [distanceTrTe{i}, ~, ~] = mDTW(xcelltrain{end}, xcelltest{end});
        
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
    
    % Get the nearest neighbors
    neighborTrTe{i} = zeros(numel(testI), numel(trI));
    for j = 1 : numel(testI)
      [~, neighborTrTe{i}(j,:)] = sort(distanceTrTe{i}(j,:),'ascend');
    end   
    fprintf('Iteration %d finished.\n', i);    

end

save(distancefile, 'distanceTrTr', 'distanceTrTe', '-v7.3');
save(neighborfile, 'neighborTrTe');
if save_newdata
    disp(['saving newdata: ', newdatafile,]);
    save(newdatafile, '-struct', 'newdata');    
end
disp(['Baseline distance: Finished saving data: ', distancefile, '; ', neighborfile]);
end