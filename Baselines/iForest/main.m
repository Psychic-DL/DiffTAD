clear
load('Isolet.mat')
% data are attribute values
% class is label where 1 represents anomaly

data = normalize(data);

rounds = 10; % rounds of repeat
%% iNNE
psi = 4; % subsample size psi can be [2 4 8 16 32 64 128 256]
t = 100; % ensemble size

auc = zeros(rounds, 1);
for r = 1:rounds
    %   pause(r/100)
    %  disp(['rounds ', num2str(r), ':']);
    %  tic
    Score= iNNE(data,data,t,psi );
    %  toc
    auc(r) = Measure_AUC(Score, class);
    % [~,~,~,auc(r)] = perfcurve(logical(class),Score,'true');
    %   disp(['auc = ', num2str(auc(r)), '.']);
end

%auc
iNNE_results = [mean(auc), std(auc)]


%% iForest
NumTree = 100; % number of isolation trees
NumSub = 2^8; % subsample size NumSub can be [2 4 8 16 32 64 128 256]

auc = zeros(rounds, 1);
for r = 1:rounds
    % pause(r/100)
    rseed(r) = sum(100 * clock);
    Forest = IsolationForest(data, NumTree,NumSub,rseed(r));
    [Mass, ~] = IsolationEstimation(data, Forest);
    Score = - mean(Mass, 2);
    iFauc(r) = Measure_AUC(Score, class);
end

%auc
iForest_results = [mean(iFauc), std(iFauc)]

%% LOF
k=0.1*ceil(size(data,1));
scores = lof(data,data,k);
LOF_result = Measure_AUC(Score, class)


%% SP

NumSub =10;
for r = 1:rounds
    %pause(r/100)
    rng('shuffle','multFibonacci')    
    CurtData=data(randperm(size(data,1),NumSub),:);
    SimMatrix=pdist2(data,CurtData,'minkowski',2);
    Score=min(SimMatrix')';    
    auc(r)=Measure_AUC(Score, class);
    % disp(['auc = ', num2str(auc(r)), '.']);
end

%auc
SP_results = [mean(auc), std(auc)]

