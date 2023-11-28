%
% Compute the local outlier factor scoare for each of the test (query)
% points. The train and test consists of MxN matrix of data (i.e., no 
% labels are included).
%
% returns local outlier factor scores
%
% Implemented based on the following paper:
%   Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, 
%   and Jorg Sander. 2000. LOF: identifying density-based local outliers. 
%   Proceedings of the 2000 ACM SIGMOD International Conference on 
%   Management of Data (SIGMOD 00), 93-104.
%   http://dx.doi.org/10.1145/342009.335388
%

function [lof_scores] = lof(train, test, k)

% this includes duplicate points if they appears within 'k' points
% therefore, the number of points found can be greater than 'k'
[train_idx, train_dist] = knnsearch(train, train, 'K', k, 'IncludeTies', true);
[test_idx, test_dist] = knnsearch(train, test, 'K', k, 'IncludeTies', true);

% *** Can we avoid doing this search without using a loop?
% this set does not includes duplicate points
% it is used to return the kth distance
[~, train2_dist] = knnsearch(train, train, 'K', k);
kth_dist = train2_dist(:,k);

[rowTrain, ~] = size(train);
[rowTest ~] = size(test);

% table to speed up the lrd (local rearchable distance calculation)
% it does a table lookup instead of recomputing the lrd for the same point
alrd = zeros(rowTrain,1);

% pre-allocate the memory which is another speedup technique
lof_scores = zeros(rowTest,1);

for i = 1:rowTest
    idxTestNN = test_idx{i};

    lrdQuery = compute_lrd(idxTestNN, test_dist{i}, kth_dist);

    [~,nQueryNN] = size(idxTestNN); % this can be higher than 'k'

    score = 0;

% *** Can we change this loop to a matrix operation?
    for j = 1: nQueryNN
        idx = idxTestNN(j);

        if alrd(idx) > 0
        else
            alrd(idx) = compute_lrd(train_idx{idx}, train_dist{idx}, kth_dist);
        end

        score = score + (alrd(idx) / lrdQuery);
    end

    lof_scores(i) = score / nQueryNN;
end

end

function [value] = compute_lrd(idx, query_dist, kth_dist)
    [~,count] = size(idx);

    value = count / sum(max(kth_dist(idx), query_dist'));
end
