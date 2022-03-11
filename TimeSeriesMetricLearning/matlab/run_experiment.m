% Multivariate Time Series Metric and Hashing Learning

addpath(genpath(pwd));
% Indicate the path/name to the input

%datafile = '../new_data/ucieeg-new/ucieeg-new.mat';
%datafile = '../new_data/ras-data/ras-data.mat';
nfolds = 5;
%datafile = '../new_data/physio-data/physio-data.mat';
%foldfile = '../new_data/physio-data/physio-data-5folds.mat';

%ucieeg
%datafile = '../new_data/ucieeg-new/ucieeg-new.mat';
%foldfile = '../new_data/ucieeg-new/ucieeg-new-5folds.mat';

%AspReg
%datafile = '../new_data/ucr-data-aspreg/ucr-data-aspreg.mat';
%foldfile = '../new_data/ucr-data-aspreg/ucr-data-aspreg-new-5folds.mat';

%syn data 100 - 10
%datafile = '../new_data/syn-data/syn-data.mat';
%foldfile = '../new_data/syn-data/syn-data-5folds.mat';

%syn data 100 - 20
datafile = '../new_data/syn-data-100-20/syn-data-100-20.mat';
foldfile = '../new_data/syn-data-100-20/syn-data-100-20-5folds.mat';

%'ED', 'MDTW',
%distancetypes = {'metricMDTW-hammL-diagY-onesW', 'GAK'};
distancetypes = {'MDTW', 'GAK', 'treeMsa', 'metricMDTW-hammL-diagY-onesW', 'LDMLT_TS'}%, 'treeMDTW-sda'}
distancetypes2 = {};
%distancetypes2 = {'MDTW', 'ED', 'treeMsa'}; %, 'treeMsaSplit'; ,'LDMLT_TS','GAK'
%distancetypes2 = [distancetypes2, 'warpAve', 'warpAveSplit', 'simpleAve'];
%distancetypes = [distancetypes, 'metricMDTW-hammL-diagY-onesW']; %, 'metricMDTW-hammL-dtwY-diagW'
%distancetypes = [distancetypes, 'MDTW_1', 'expDist'];
% 'MDTW', 'GAK', 'treeMsa', 'metricMDTW-hammL-diagY-onesW', 
%distancetypes = {'LDMLT_TS', };
%distancetypes = {'MDTW', 'LDMLT_TS'};
%distancetypes2 = [distancetypes2, 'metricMDTW-areaL-diagY', 'metricMDTW-areaL-dtwY'];
% methods after network training...
%distancetypes = {'treeMDTW-dae', 'mapBack-dae', }; % 'warpSelf-dae', 'treeED-dae',
%distancetypes = [distancetypes, 'warpSelfSplit-dae',  'treeMDTWSplit-dae', 'treeEDSplit-dae','mapBackSplit-dae'];
%distancetypes = [distancetypes,  'treeMDTW-sda', 'mapBack-sda']; %'warpSelf-sda', 'treeED-sda', 
%distancetypes = [distancetypes, 'warpSelf-rtrbm', 'treeMDTW-rtrbm', 'treeED-rtrbm', 'mapBack-rtrbm'];
%distancetypes = [distancetypes2, distancetypes];
%distancetypes = distancetypes2;
%distancetypes2 = {};
%distancetypes2 = [distancetypes2, distancetypes];
%distancetypes = {'MDTW_1',};

% Split data to nfolds equally 
%foldfile = split_data(datafile, nfolds);

% Get basic distance and corresponding kernelized hashing
ntypes = numel(distancetypes);
for index=1:ntypes
	% Compute the distance
	%[distancefile{index}, neighborfileneig = loadmat(neighbor_file){index}] = ...
		compute_distance(datafile, distancetypes{index}, nfolds);
end

%% Plot the classification result
if (1==1)
%     distancetypes = {};
%     for i=1:10
%         distancetypes = [distancetypes, strcat('metricMDTW', int2str(i))];
%     end
    methodtypes = [distancetypes2, distancetypes];%methodtypes=distancetypes;
    show_precision(datafile, nfolds, methodtypes);
end

if (12==1)
    methodtypes = [distancetypes2, distancetypes];
    refmethodtypes = {};
    for i=1:length(methodtypes)
        refmethodtypes = [refmethodtypes, 'MDTW'];
    end
    show_recall(datafile, nfolds, methodtypes, refmethodtypes);
end