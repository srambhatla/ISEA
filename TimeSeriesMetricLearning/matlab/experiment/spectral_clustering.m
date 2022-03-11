function [ features ] = spectral_clustering( dist_matrix, y_label, k)
%SPECTRAL_CLUSTERING Summary of this function goes here
%   Detailed explanation goes here

% Create the affinity matrix
%dist_matrix = dist_matrix(y_label<3, y_label<3);
%y_label = y_label(y_label<3);

sim = -dist_matrix / max(dist_matrix(:));
sim = exp(sim);
A = abs(sim) + abs(sim');

% Normalized Spectral Clustering according to Ng & Jordan & Weiss
eps = 2.2204e-16;
DN = diag( 1./sqrt(sum(A)+eps) );
LapN = DN*A*DN; 
LapN = (LapN+LapN')/2;   % Avoiding numerical errors
[features, lams] = eigs(LapN, k, 'lm');    % nF is the dimensionality of learned features.

disp(lams)

hold off;
ymin = min(y_label);
ymax = max(y_label);
color='rgbymckrgbymck';
hold on;
for yy=ymin:ymax
%for yy=1:2
    scatter(features(y_label==yy,1), features(y_label==yy,2), 20, color(yy));
end

saveas(gcf, 'sc_2d.fig');

hold off;
scatter3(0, 0, 0, 20, 'w')
hold on;
for yy=ymin:ymax
%for yy=1:2
    scatter3(features(y_label==yy,1), features(y_label==yy,2),  features(y_label==yy,3), 20, color(yy));
end
saveas(gcf, 'sc_3d.fig');
