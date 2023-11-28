function [ ndata ] = normalize( data )
%NORMALIZE Summary of this function goes here
%   Detailed explanation goes here
ndata=[];
for i=1:size(data,2)
    d=data(:,i);
    if (max(d)-min(d))==0
 %       d=zeros(size(d,1),1);
        ndata=[ndata d];
    else
    d=(d-min(d))./(max(d)-min(d));
    ndata=[ndata d];
    end
end

