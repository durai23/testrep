function kmeans_demo(k)
%%%what is going to be the value of k - how is it determined?

% function to demonstrate k-means clustering on some toy MRI brain data
% Usage: kmeans_demo(k) where k = number of clusters
% 
% The data here is diffusion MRI probabilistic tractography.
% Probabilisitic tractography attempts to model structural brain
% connections (also known as white matter fibres / tracts / bundles of
% neurons)
% 
% In short this is done by creating a model of water diffusion in every 
% volume element (voxel) of the MRI. The assumption is that fibre bundles
% will hinder the diffusion of water orthogonal to their direction.
% 
% By reconstructing the principle fibre direction trajectory one can
% perform tractography.
% 
% In this experiment 5000 streamlines were propagated from a single seed 
% point (the semi hidden red dot). Each streamline will draw it's next step
% from the orienation distribution function (modeled in each voxel) and we
% end up with 5000 different tractography streamlines.
% 
% We note that they terminate in different regions of the brain and while
% there seems to be a cloud of connections, the terminations are not quite
% random on the brain. Clustering can help us make sense of the
% terminations.
% 
% Note that we only show one hemisphere (left) but there are some
% streamlines that terminate in the opposite (right) hemisphere
% 
% 
% Claude J Bajada c.bajada@fz-juelich.de

%% import data and organise
%%%durai comments with 3 %
%%%read in tck files. these will be read in as structs
tract=read_mrtrix_tracks('left_5_seed_endpoints.tck'); % endpoints data
streaml = read_mrtrix_tracks('left_5_seed.tck'); % full tract - these two must be the derived from same data
%%%the data attrib is a cell data type
%%%both streaml.data and tract.data are of same dimensions
streaml = streaml.data;
%%%only for the tract - convert to mat WHY - maybe to extract seed and endpoint
%%%not only that cell2mat FAILS on streaml.data but works on tract.data - so tehy are different
tract=cell2mat(tract.data);
%%%now streaml and tract are of different dimensions due to cell2mat
%%%size(streaml,2) refers to 2nd dimension of the size of streaml
%%%take 2X15000 tract and put each row of tract in a different variable each of size 3X5000
%%%what are these 2 rows and why are they being reshaped into 3X5000
seed = reshape(tract(1,:) , [3 size(streaml,2)])';
tract = reshape(tract(2,:) , [3 size(streaml,2)])';

%% load brain surface
brain_surface = gifti('100206.L.white_MSMAll.32k_fs_LR.surf.gii');
sz = 15; % size of marker
col = 'g'; % color of marker

%% compute kmeans with k clusters
% parameters remained at default euclidean distance - which makes sense in
% this case
% 100 replications
IDX = kmeans(tract, k, 'Replicates',100);

%% plot brains with endpoints
plot_brain(col)
plot_brain(IDX)

%% plot brain with streamlines
figure;
P = parula(k);
str=plot(brain_surface); hold on
set(str , 'FaceColor' , [.85 .85 .85])
set(str , 'Facealpha' , 0.3);
hold on; st = streamline(streaml);
for i = 1:size(streaml,2)
    st(i).Color= P(IDX(i),:);
end

disp('Finished')

%% internal function to plot the brains
function plot_brain(label)
    
    figure;
    brain=plot(brain_surface); hold on
    set(brain , 'FaceColor' , [.85 .85 .85])
    
    % plot seed point - this will be colored red (will be hard to see in
    % this example
    scatter3(seed(:,1) , seed(:,2) , seed(:,3) , 30 , 'r', 'filled');
    % plot destinations
    scatter3(tract(:,1) , tract(:,2) , tract(:,3) , sz , label, 'filled'); 
    axis equal
    
end
end
