function [ index ] = Cigrid( data,psi )
% index structure:
% center id
% center's 1nn id
% center's 1nn dist
% center's 1nn's 1nn dist
% isolation score
Ndata=[];
%% sampling
while size(Ndata,1)<2 % prevent only one subsample
    CurtIndex = datasample(1:size(data,1),psi,'Replace',false);
    Ndata = data(CurtIndex,:);
    
    %% filter out repeat
    
    [~,IA,~] = unique(Ndata,'rows'); %  C = A(IA) and A = C(IC) (or A(:) = C(IC), if A is a matrix or array).
    NCurtIndex=CurtIndex(IA);
    Ndata = data(NCurtIndex,:);
    
end
%% calculate isolation score

[D,I] = pdist2(Ndata,Ndata,'minkowski',2,'Smallest',2);
index=[NCurtIndex(I);D(2,:);D(2,I(2,:));1-D(2,I(2,:))./D(2,:)];

end

