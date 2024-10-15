function [afea, t, segment] = DownSampleMatByAvg(sig, numSegment)

DSFactor = fix(size(sig,1) / numSegment);
afea = [];
segment = [];
for j=1:size(sig, 2)    
    csig = filter(ones(1,DSFactor)/DSFactor,1, sig(:,j));
    %afea =[afea csig(1:DSFactor:end)];
    afea =[afea csig(DSFactor:DSFactor:end)]; %fixed bug WFS
end;
t = 1:DSFactor:size(sig, 1);