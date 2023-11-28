function [ Iscore ] = iNNE( traindata,testdata,t,psi )

rng('shuffle','multFibonacci')
index=cell(t,1);
for i=1:t
    index{i} = Cigrid( traindata,psi );
end

n=size(testdata,1);
Iso=zeros(n,t)+1;


for i=1:t
    pindex=index{i}(1,:);
    distIndex=index{i}(3,:);
    ratioindex=index{i}(end,:);    
    pdata=traindata(pindex,:);
    

%   [D,I] = pdist2(pdata,testdata,'minkowski',2,'Smallest',1);  %  find nearest ball       
%   Iso(:,i)=ratioindex(I);  
%   Iso(distIndex(I)<=D,i)=1;
    
    
    [D] = pdist2(pdata,testdata,'minkowski',2);
    radiusMatrix=repmat(distIndex',1,n);
    I=D<radiusMatrix; % find balls covering x
    Cradius=radiusMatrix.*I;
    Cradius(Cradius==0)=1;
    [~,II]=min(Cradius,[],1); % find cnn(x) 
    Cratio=ratioindex(II);
    Cratio(sum(I,1)==0)=1;
    Iso(:,i)=Cratio;
end

Iscore=mean(Iso,2);
    