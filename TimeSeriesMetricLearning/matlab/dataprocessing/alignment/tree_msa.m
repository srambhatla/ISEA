function [xref, xpath] = tree_msa(xcell, xdist, method)
%TREE_MSA Summary of this function goes here
%   Detailed explanation goes here
%   xdist: n*n pairwise distance
%   xpath: n*1 cell, each is t*2. xpath{i}(1,1)=1; xpath{i}(t,2)=T
%           (ti_begin, ti_end)
nx = length(xcell);
xname = cell(nx,1);
xpath = cell(nx,1);
for i=1:nx
   xname{i}=int2str(i);
end
tree = seqlinkage(xdist, 'average', xname);
treeparas = get(tree);
xtreenodes=cell(treeparas.NumNodes,1);
xcounts = ones(treeparas.NumNodes,1);
xtreepaths = cell(treeparas.NumNodes,1);
for i=1:treeparas.NumLeaves
    xtreenodes{i}=xcell{str2num(treeparas.LeafNames{i})};
end
for i=1:treeparas.NumBranches
    if mod(i,100) ==0
        fprintf('\tTree Msa i: %d\n', i);
    end
    %get two xtreenodes, align and average 
    id1 = treeparas.Pointers(i,1);
    id2 = treeparas.Pointers(i,2);
    x1 = xtreenodes{id1};
    x2 = xtreenodes{id2};
    x1count = xcounts(id1);
    x2count = xcounts(id2);
    if method == 1
        % if T1>T2, warp along x1; otherwise warp along x2
        [xtreenodes{i+treeparas.NumLeaves}, pair_path]=average_and_warp_series(x1, x2, x1count, x2count);
    elseif method == 2
        xtreenodes{i+treeparas.NumLeaves}=average_and_extend_series(x1, x2);
    end
    xcounts(i+treeparas.NumLeaves) = x1count + x2count;
    % from pair_path to two xtreepaths
    [ xtreepaths{id1}, xtreepaths{id2} ]= split_path( pair_path, size(x1,1), size(x2,1));
    % for debugging
    %assert(min(xtreepaths{id1}(2:size(xtreepaths{id1},1),1) - xtreepaths{id1}(1:size(xtreepaths{id1},1)-1,2)) >= 0 )
    %assert(min(xtreepaths{id2}(2:size(xtreepaths{id2},1),1) - xtreepaths{id2}(1:size(xtreepaths{id2},1)-1,2)) >= 0 )
end
%assert(xcounts(treeparas.NumNodes) == treeparas.NumLeaves)
% first make up the last path (identical alignment)
xtreepaths{treeparas.NumNodes} = zeros(size(xtreenodes{treeparas.NumNodes},1),2);
for i=1:length(xtreenodes{treeparas.NumNodes})
    xtreepaths{treeparas.NumNodes}(i,:) = [i,i];
end
% recover the path backward!
for i=treeparas.NumBranches:-1:1
    id1 = treeparas.Pointers(i,1);
    id2 = treeparas.Pointers(i,2);
    %sub_path = base_path(sub_path);
    tmpx = xtreepaths{id1};
    xtreepaths{id1}= revise_path(xtreepaths{i+treeparas.NumLeaves}, xtreepaths{id1});
    xtreepaths{id2} = revise_path(xtreepaths{i+treeparas.NumLeaves}, xtreepaths{id2});
    % need to check: if p(i,2)==p(i+1,1), then p(i,1)=p(i+1,2)
    % need to do this iff len(path)>1 !
    if size(xtreepaths{id1},1) > 1
        assert(min(xtreepaths{id1}(2:size(xtreepaths{id1},1),1) - xtreepaths{id1}(1:size(xtreepaths{id1},1)-1,2)) >= 0 )
    end
    if size(xtreepaths{id2},1) > 1
        assert(min(xtreepaths{id2}(2:size(xtreepaths{id2},1),1) - xtreepaths{id2}(1:size(xtreepaths{id2},1)-1,2)) >= 0 )
    end
%     for j=1:size(xtreepaths{id1,1},1)-1
%        if (xtreepaths{id1}(j,2) ==xtreepaths{id1}(j+1,1) ) 
%           if (xtreepaths{id1}(j,1) < xtreepaths{id1}(j+1,2) )
%               tmpx'
%               xtreepaths{id1}'
%               xtreepaths{i+treeparas.NumLeaves}'
%               j
%               assert(1==0);
%           end
%        end
%     end
%     for j=1:size(xtreepaths{id2,1},1)-1
%        if (xtreepaths{id2}(j,2) ==xtreepaths{id2}(j+1,1) ) 
%           if (xtreepaths{id2}(j,1)  < xtreepaths{id2}(j+1,2) )
%               xtreepaths{id2}
%               j
%               assert(1==0);
%           end
%        end
%     end
    
end
for i=1:treeparas.NumLeaves
     xpath{str2num(treeparas.LeafNames{i})} = xtreepaths{i};
end
xref = xtreenodes{treeparas.NumNodes};
end

